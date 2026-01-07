#!/usr/bin/env python3
"""
Process raw continuous data into organized market folders.

This script takes the continuous raw data from data_v2/raw/ and splits it into 
15-minute market folders in data_v2/markets_6levels/ for use by the build script.
(6-level order book depth is the standard for all new collections)

Workflow:
1. Collect data: python scripts/cli_v2.py collect --assets BTC --duration 60
2. Process into markets: python scripts/process_raw_data.py
3. Build dataset: python scripts/cli_v2.py build
4. Validate: python scripts/cli_v2.py validate
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import STORAGE, SUPPORTED_ASSETS


def load_continuous_data(asset: str, raw_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load continuous data for an asset."""
    cl_path = raw_dir / "chainlink" / asset / f"chainlink_{asset}_continuous.csv"
    pm_path = raw_dir / "polymarket" / asset / f"polymarket_{asset}_continuous.csv"
    
    df_cl = pd.DataFrame()
    df_pm = pd.DataFrame()
    
    # Chainlink expected columns
    cl_columns = ["timestamp", "data_timestamp_raw", "asset", "mid", "bid", "ask", 
                  "source", "is_observed", "collected_at"]
    
    if cl_path.exists():
        try:
            # Try reading with headers first
            df_cl = pd.read_csv(cl_path, on_bad_lines='skip')
            
            # Check if headers are missing (first row looks like a timestamp)
            if len(df_cl.columns) > 0:
                first_col = str(df_cl.columns[0])
                # If first column looks like a timestamp (has T and + or Z), headers are missing
                if 'T' in first_col and ('+' in first_col or 'Z' in first_col):
                    # Headers are missing, read again with proper column names
                    df_cl = pd.read_csv(cl_path, names=cl_columns, on_bad_lines='skip')
                    print(f"  Loaded {len(df_cl)} Chainlink records for {asset} (headers were missing, auto-assigned)")
                else:
                    print(f"  Loaded {len(df_cl)} Chainlink records for {asset}")
            else:
                print(f"  Warning: Chainlink CSV for {asset} appears empty")
            
            # Handle both old and new column names (timestamp vs source_timestamp)
            # New collectors use source_timestamp, old use timestamp
            if 'source_timestamp' in df_cl.columns:
                # New format - use source_timestamp as the data timestamp
                df_cl['timestamp'] = df_cl['source_timestamp']  # Create alias for compatibility
            elif 'timestamp' not in df_cl.columns:
                # Fallback to received_timestamp if available
                if 'received_timestamp' in df_cl.columns:
                    df_cl['timestamp'] = df_cl['received_timestamp']
                elif 'collected_at' in df_cl.columns:
                    df_cl['timestamp'] = df_cl['collected_at']
            
            # Convert timestamp columns
            if 'timestamp' in df_cl.columns:
                df_cl['timestamp'] = pd.to_datetime(df_cl['timestamp'], format='mixed', utc=True, errors='coerce')
            if 'collected_at' in df_cl.columns:
                df_cl['collected_at'] = pd.to_datetime(df_cl['collected_at'], format='mixed', utc=True, errors='coerce')
        except Exception as e:
            print(f"  Error loading Chainlink data for {asset}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  No Chainlink data found for {asset}")
    
    if pm_path.exists():
        try:
            # Read file line by line to handle mixed formats (old 12-col vs new 24-col)
            import csv as csv_module
            rows_old = []
            rows_new = []
            with open(pm_path, 'r', encoding='utf-8') as f:
                reader = csv_module.reader(f)
                header = next(reader)  # Read header
                
                # Old format columns
                pm_columns_old = ["timestamp", "timestamp_ms", "asset", "market_id",
                                  "up_mid", "up_bid", "up_ask",
                                  "down_mid", "down_bid", "down_ask",
                                  "is_observed", "collected_at"]
                
                # New format columns (from orchestrator _init_output_files) - 6 levels
                pm_columns_new = [
                    "source_timestamp", "received_timestamp", "timestamp_ms", "asset", "market_id",
                    "up_mid", "up_bid", "up_bid_size", "up_ask", "up_ask_size",
                    "up_bid_2", "up_bid_2_size", "up_ask_2", "up_ask_2_size",
                    "up_bid_3", "up_bid_3_size", "up_ask_3", "up_ask_3_size",
                    "up_bid_4", "up_bid_4_size", "up_ask_4", "up_ask_4_size",
                    "up_bid_5", "up_bid_5_size", "up_ask_5", "up_ask_5_size",
                    "up_bid_6", "up_bid_6_size", "up_ask_6", "up_ask_6_size",
                    "down_mid", "down_bid", "down_bid_size", "down_ask", "down_ask_size",
                    "down_bid_2", "down_bid_2_size", "down_ask_2", "down_ask_2_size",
                    "down_bid_3", "down_bid_3_size", "down_ask_3", "down_ask_3_size",
                    "down_bid_4", "down_bid_4_size", "down_ask_4", "down_ask_4_size",
                    "down_bid_5", "down_bid_5_size", "down_ask_5", "down_ask_5_size",
                    "down_bid_6", "down_bid_6_size", "down_ask_6", "down_ask_6_size",
                    "is_observed"
                ]
                
                # Read all rows, categorizing by column count
                for row in reader:
                    if len(row) == 12:
                        # Old format - pad to match expected columns
                        rows_old.append(row)
                    elif len(row) >= 20:
                        # New format - truncate to expected length
                        rows_new.append(row[:len(pm_columns_new)])
            
            # Combine both formats
            df_pm_old = None
            df_pm_new = None
            
            if rows_old:
                df_pm_old = pd.DataFrame(rows_old, columns=pm_columns_old)
                # Map old format to new format structure
                if 'timestamp' in df_pm_old.columns:
                    df_pm_old['source_timestamp'] = df_pm_old['timestamp']
                if 'collected_at' in df_pm_old.columns:
                    df_pm_old['received_timestamp'] = df_pm_old['collected_at']
            
            if rows_new:
                df_pm_new = pd.DataFrame(rows_new, columns=pm_columns_new)
            
            # Combine DataFrames
            if df_pm_old is not None and df_pm_new is not None:
                # Align columns and combine
                common_cols = ['source_timestamp', 'received_timestamp', 'timestamp_ms', 'asset', 'market_id', 
                              'up_mid', 'down_mid', 'is_observed']
                # For old format, add missing columns as None
                for col in pm_columns_new:
                    if col not in df_pm_old.columns:
                        df_pm_old[col] = None
                # Reorder old format to match new
                df_pm_old = df_pm_old[[c for c in pm_columns_new if c in df_pm_old.columns]]
                df_pm = pd.concat([df_pm_old, df_pm_new], ignore_index=True)
                print(f"  Loaded {len(df_pm)} Polymarket records for {asset} (mixed format: {len(rows_old)} old + {len(rows_new)} new)")
            elif df_pm_old is not None:
                df_pm = df_pm_old
                print(f"  Loaded {len(df_pm)} Polymarket records for {asset} (old format only)")
            elif df_pm_new is not None:
                df_pm = df_pm_new
                print(f"  Loaded {len(df_pm)} Polymarket records for {asset} (new format only)")
            else:
                # Fall back to pandas
                df_pm = pd.read_csv(pm_path, on_bad_lines='skip', low_memory=False)
                print(f"  Loaded {len(df_pm)} Polymarket records for {asset} (pandas fallback)")
            
            # Handle both old and new column names (timestamp vs source_timestamp)
            # New collectors use source_timestamp, old use timestamp
            if 'source_timestamp' in df_pm.columns:
                # New format - use source_timestamp as the data timestamp
                df_pm['timestamp'] = df_pm['source_timestamp']  # Create alias for compatibility
            elif 'timestamp' not in df_pm.columns:
                # Fallback to received_timestamp if available
                if 'received_timestamp' in df_pm.columns:
                    df_pm['timestamp'] = df_pm['received_timestamp']
                elif 'collected_at' in df_pm.columns:
                    df_pm['timestamp'] = df_pm['collected_at']
            
            # For old format, ensure we have collected_at if missing
            if 'collected_at' not in df_pm.columns and 'received_timestamp' in df_pm.columns:
                df_pm['collected_at'] = df_pm['received_timestamp']
            
            if 'timestamp' in df_pm.columns:
                df_pm['timestamp'] = pd.to_datetime(df_pm['timestamp'], format='mixed', utc=True, errors='coerce')
            if 'collected_at' in df_pm.columns:
                df_pm['collected_at'] = pd.to_datetime(df_pm['collected_at'], format='mixed', utc=True, errors='coerce')
        except Exception as e:
            print(f"  Error loading Polymarket data for {asset}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  No Polymarket data found for {asset}")
    
    return df_cl, df_pm


def identify_market_periods(df_pm: pd.DataFrame, df_cl: pd.DataFrame = None) -> List[Dict]:
    """Identify 15-minute market periods from the data."""
    if df_pm.empty:
        return []
    
    # Use timestamp (actual data time) for market identification
    # This ensures we match CL price at time T with PM price at time T (same market moment)
    # CL timestamps are ~65s behind PM, so we need longer collection to get overlap
    ts_col = 'timestamp' if 'timestamp' in df_pm.columns else 'collected_at'
    if ts_col not in df_pm.columns:
        raise ValueError(f"Neither 'timestamp' nor 'collected_at' found in Polymarket data. Columns: {list(df_pm.columns)}")
    
    # Floor timestamps to 15-minute boundaries
    df_pm = df_pm.copy()
    df_pm['market_start'] = df_pm[ts_col].dt.floor('15min')
    
    cl_ts_col = None
    if df_cl is not None and not df_cl.empty:
        df_cl = df_cl.copy()
        cl_ts_col = 'timestamp' if 'timestamp' in df_cl.columns else 'collected_at'
        if cl_ts_col not in df_cl.columns:
            print(f"  Warning: Neither 'timestamp' nor 'collected_at' found in Chainlink data. Columns: {list(df_cl.columns)}")
            df_cl = pd.DataFrame()  # Clear df_cl if we can't process it
        else:
            df_cl['market_start'] = df_cl[cl_ts_col].dt.floor('15min')
    
    markets = []
    # #region agent log
    import json
    log_path = Path(__file__).parent.parent / ".cursor" / "debug.log"
    pm_groups = list(df_pm.groupby('market_start'))
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "market-identification",
                "hypothesisId": "B",
                "location": "process_raw_data.py:identify_market_periods",
                "message": "PM groups found",
                "data": {
                    "pm_group_count": len(pm_groups),
                    "pm_total_rows": len(df_pm),
                    "pm_time_col": ts_col,
                    "pm_has_timestamp": 'timestamp' in df_pm.columns,
                    "cl_has_data": df_cl is not None and not df_cl.empty,
                    "cl_time_col": cl_ts_col
                },
                "timestamp": pd.Timestamp.now().timestamp() * 1000
            }) + "\n")
    except:
        pass
    # #endregion
    
    for market_start, group in df_pm.groupby('market_start'):
        market_end = market_start + timedelta(minutes=15)
        
        # Get strike price (first Chainlink price in this market period)
        strike_price = None
        if df_cl is not None and not df_cl.empty and cl_ts_col is not None:
            cl_market = df_cl[df_cl['market_start'] == market_start]
            if not cl_market.empty and 'mid' in cl_market.columns:
                cl_sorted = cl_market.sort_values(cl_ts_col)
                first_price = cl_sorted.iloc[0]['mid']
                if pd.notna(first_price):
                    try:
                        # Convert to float if it's a string
                        price_val = float(first_price) if isinstance(first_price, str) else first_price
                        strike_price = int(round(price_val))
                    except (ValueError, TypeError):
                        pass  # Skip if conversion fails
        
        # Create market name
        base_name = market_start.strftime('%Y%m%d_%H%M')
        if strike_price is not None:
            market_name = f"{base_name}_{strike_price}"
        else:
            market_name = base_name
        
        markets.append({
            'start': market_start,
            'end': market_end,
            'name': market_name,
            'strike_price': strike_price,
            'pm_count': len(group)
        })
    
    # #region agent log
    try:
        last_market = markets[-1] if markets else None
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "market-identification",
                "hypothesisId": "B",
                "location": "process_raw_data.py:identify_market_periods",
                "message": "Markets created",
                "data": {
                    "total_markets": len(markets),
                    "last_market_name": last_market['name'] if last_market else None,
                    "last_market_start": str(last_market['start']) if last_market else None,
                    "last_market_pm_count": last_market['pm_count'] if last_market else None
                },
                "timestamp": pd.Timestamp.now().timestamp() * 1000
            }) + "\n")
    except:
        pass
    # #endregion
    
    return sorted(markets, key=lambda x: x['start'])


def create_market_folder(
    asset: str,
    market: Dict,
    df_cl: pd.DataFrame,
    df_pm: pd.DataFrame,
    output_dir: Path
) -> Optional[Path]:
    """Create a market folder with data files."""
    market_dir = output_dir / asset / market['name']
    market_dir.mkdir(parents=True, exist_ok=True)
    
    start = market['start']
    end = market['end']
    
    # Use timestamp (actual data time) for filtering
    # This ensures CL price at time T matches PM price at time T (same market moment)
    ts_col = 'timestamp' if 'timestamp' in df_pm.columns else 'collected_at'
    cl_ts_col = 'timestamp' if 'timestamp' in df_cl.columns else 'collected_at'
    
    # Filter data for this market period
    pm_mask = (df_pm[ts_col] >= start) & (df_pm[ts_col] < end)
    pm_market = df_pm[pm_mask].copy()
    
    cl_market = pd.DataFrame()
    if not df_cl.empty:
        cl_mask = (df_cl[cl_ts_col] >= start) & (df_cl[cl_ts_col] < end)
        cl_market = df_cl[cl_mask].copy()
    
    if pm_market.empty and cl_market.empty:
        print(f"  No data for {market['name']}")
        return None
    
    # Calculate seconds from market start
    if not pm_market.empty:
        pm_market['seconds'] = (pm_market[ts_col] - start).dt.total_seconds()
    if not cl_market.empty:
        cl_market['seconds'] = (cl_market[cl_ts_col] - start).dt.total_seconds()
    
    # Save data files
    pm_path = market_dir / "polymarket.csv"
    cl_path = market_dir / "chainlink.csv"
    
    # Polymarket columns - save all available columns including size data
    # Raw data has: up_mid, up_bid, up_ask, down_mid, down_bid, down_ask
    # New format also has: up_bid_size, up_ask_size, down_bid_size, down_ask_size
    # Plus level 2: up_bid_2, up_bid_2_size, up_ask_2, up_ask_2_size, etc.
    # Build script expects: up_best_bid, up_best_ask, etc. (we'll map these)
    pm_columns = ['timestamp', 'collected_at', 'seconds', 
                  'up_mid', 'up_bid', 'up_ask',
                  'down_mid', 'down_bid', 'down_ask',
                  # Size columns (new format)
                  'up_bid_size', 'up_ask_size', 'down_bid_size', 'down_ask_size',
                  # Levels 2-6 depth (new format)
                  'up_bid_2', 'up_bid_2_size', 'up_ask_2', 'up_ask_2_size',
                  'up_bid_3', 'up_bid_3_size', 'up_ask_3', 'up_ask_3_size',
                  'up_bid_4', 'up_bid_4_size', 'up_ask_4', 'up_ask_4_size',
                  'up_bid_5', 'up_bid_5_size', 'up_ask_5', 'up_ask_5_size',
                  'up_bid_6', 'up_bid_6_size', 'up_ask_6', 'up_ask_6_size',
                  'down_bid_2', 'down_bid_2_size', 'down_ask_2', 'down_ask_2_size',
                  'down_bid_3', 'down_bid_3_size', 'down_ask_3', 'down_ask_3_size',
                  'down_bid_4', 'down_bid_4_size', 'down_ask_4', 'down_ask_4_size',
                  'down_bid_5', 'down_bid_5_size', 'down_ask_5', 'down_ask_5_size',
                  'down_bid_6', 'down_bid_6_size', 'down_ask_6', 'down_ask_6_size']
    pm_export_cols = [c for c in pm_columns if c in pm_market.columns]
    if pm_export_cols and not pm_market.empty:
        # Rename to match what build script expects
        rename_map = {
            'up_bid': 'up_best_bid',
            'up_ask': 'up_best_ask',
            'down_bid': 'down_best_bid',
            'down_ask': 'down_best_ask',
            # Size columns
            'up_bid_size': 'up_best_bid_size',
            'up_ask_size': 'up_best_ask_size',
            'down_bid_size': 'down_best_bid_size',
            'down_ask_size': 'down_best_ask_size',
        }
        pm_export = pm_market[pm_export_cols].copy()
        pm_export.rename(columns=rename_map, inplace=True)
        # Only add size columns as None if they don't already exist with real data
        for col in ['up_best_bid_size', 'up_best_ask_size', 'down_best_bid_size', 'down_best_ask_size']:
            if col not in pm_export.columns:
                pm_export[col] = None
        pm_export.to_csv(pm_path, index=False)
    
    # Chainlink columns
    cl_columns = ['timestamp', 'collected_at', 'seconds', 'mid', 'bid', 'ask', 'asset']
    cl_export_cols = [c for c in cl_columns if c in cl_market.columns]
    if cl_export_cols and not cl_market.empty:
        # VALIDATION: Check that numeric columns are actually numeric
        # This catches column misalignment bugs early
        for numeric_col in ['mid', 'bid', 'ask']:
            if numeric_col in cl_market.columns:
                # Check if any values are strings that aren't NaN
                sample_val = cl_market[numeric_col].dropna().iloc[0] if len(cl_market[numeric_col].dropna()) > 0 else None
                if sample_val is not None and isinstance(sample_val, str):
                    print(f"  WARNING: Chainlink column '{numeric_col}' contains string values ('{sample_val}')!")
                    print(f"  This indicates a column misalignment bug in the raw data.")
                    print(f"  Skipping market {market['name']} to prevent data corruption.")
                    return None
        cl_market[cl_export_cols].to_csv(cl_path, index=False)
    
    # Calculate unique seconds coverage (not row count)
    # A market has 900 seconds (15 minutes)
    cl_unique_seconds = set()
    pm_unique_seconds = set()
    
    if not cl_market.empty and 'seconds' in cl_market.columns:
        cl_unique_seconds = set(cl_market['seconds'].astype(int).unique())
        cl_unique_seconds = {s for s in cl_unique_seconds if 0 <= s < 900}
    
    if not pm_market.empty and 'seconds' in pm_market.columns:
        pm_unique_seconds = set(pm_market['seconds'].astype(int).unique())
        pm_unique_seconds = {s for s in pm_unique_seconds if 0 <= s < 900}
    
    both_unique_seconds = cl_unique_seconds & pm_unique_seconds  # Intersection
    
    # Create summary with CORRECT coverage metrics
    summary = {
        'asset': asset,
        'market_id': market['name'],
        'market_start': start.isoformat(),
        'market_end': end.isoformat(),
        'strike_price': market.get('strike_price'),
        'cl_records': len(cl_market),
        'pm_records': len(pm_market),
        'cl_unique_seconds': len(cl_unique_seconds),
        'pm_unique_seconds': len(pm_unique_seconds),
        'both_unique_seconds': len(both_unique_seconds),
        'cl_coverage': len(cl_unique_seconds) / 900 * 100,
        'pm_coverage': len(pm_unique_seconds) / 900 * 100,
        'both_coverage': len(both_unique_seconds) / 900 * 100
    }
    
    summary_path = market_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return market_dir


def process_asset(asset: str, raw_dir: Path, markets_dir: Path, min_coverage: float = 50.0) -> Dict:
    """Process raw data for a single asset into market folders."""
    print(f"\nProcessing {asset}...")
    
    # #region agent log
    import json
    log_path = Path(__file__).parent.parent / ".cursor" / "debug.log"
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "market-identification",
                "hypothesisId": "A",
                "location": "process_raw_data.py:process_asset",
                "message": "Starting asset processing",
                "data": {"asset": asset},
                "timestamp": pd.Timestamp.now().timestamp() * 1000
            }) + "\n")
    except:
        pass
    # #endregion
    
    # Load continuous data
    df_cl, df_pm = load_continuous_data(asset, raw_dir)
    
    # #region agent log
    try:
        cl_rows = len(df_cl) if df_cl is not None and not df_cl.empty else 0
        pm_rows = len(df_pm) if df_pm is not None and not df_pm.empty else 0
        cl_time_range = None
        pm_time_range = None
        if cl_rows > 0 and 'timestamp' in df_cl.columns:
            cl_times = pd.to_datetime(df_cl['timestamp'], errors='coerce').dropna()
            if len(cl_times) > 0:
                cl_time_range = {"min": str(cl_times.min()), "max": str(cl_times.max())}
        if pm_rows > 0 and 'timestamp' in df_pm.columns:
            pm_times = pd.to_datetime(df_pm['timestamp'], errors='coerce').dropna()
            if len(pm_times) > 0:
                pm_time_range = {"min": str(pm_times.min()), "max": str(pm_times.max())}
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "market-identification",
                "hypothesisId": "A",
                "location": "process_raw_data.py:process_asset",
                "message": "Data loaded",
                "data": {
                    "cl_rows": cl_rows,
                    "pm_rows": pm_rows,
                    "cl_time_range": cl_time_range,
                    "pm_time_range": pm_time_range
                },
                "timestamp": pd.Timestamp.now().timestamp() * 1000
            }) + "\n")
    except:
        pass
    # #endregion
    
    if df_pm.empty:
        print(f"  No Polymarket data for {asset}, skipping")
        return {'asset': asset, 'markets': 0, 'skipped': 0}
    
    # Identify market periods
    markets = identify_market_periods(df_pm, df_cl)
    
    # #region agent log
    try:
        market_names = [m['name'] for m in markets]
        market_starts = [str(m['start']) for m in markets]
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "market-identification",
                "hypothesisId": "A",
                "location": "process_raw_data.py:process_asset",
                "message": "Markets identified",
                "data": {
                    "market_count": len(markets),
                    "market_names": market_names[:10],  # First 10
                    "market_starts": market_starts[:10],
                    "last_market": market_names[-1] if market_names else None
                },
                "timestamp": pd.Timestamp.now().timestamp() * 1000
            }) + "\n")
    except:
        pass
    # #endregion
    
    print(f"  Found {len(markets)} potential market periods")
    
    # Process each market
    processed = 0
    skipped = 0
    already_processed = 0
    
    for market in markets:
        market_dir = markets_dir / asset / market['name']
        summary_path = market_dir / "summary.json"
        
        # Skip if market already exists and has valid summary
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    existing_summary = json.load(f)
                # Check if it's a valid processed market (has coverage data)
                if 'both_coverage' in existing_summary:
                    print(f"    {market['name']}: [SKIP] Already processed (Both={existing_summary.get('both_coverage', 0):.1f}%)")
                    already_processed += 1
                    continue
            except (json.JSONDecodeError, KeyError):
                # Invalid summary, reprocess
                pass
        
        # Create/update market folder
        market_path = create_market_folder(asset, market, df_cl, df_pm, markets_dir)
        
        if market_path:
            # Check coverage
            summary_path = market_path / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                
                cl_cov = summary.get('cl_coverage', 0)
                pm_cov = summary.get('pm_coverage', 0)
                both_cov = summary.get('both_coverage', 0)
                
                # Use INTERSECTION coverage (both_coverage) as the threshold
                # This ensures we have overlapping data for both sources
                if both_cov >= min_coverage:
                    print(f"    {market['name']}: CL={cl_cov:.1f}%, PM={pm_cov:.1f}%, Both={both_cov:.1f}% [OK]")
                    processed += 1
                else:
                    print(f"    {market['name']}: CL={cl_cov:.1f}%, PM={pm_cov:.1f}%, Both={both_cov:.1f}% [LOW COVERAGE - KEEPING]")
                    # Keep all markets regardless of coverage (user requested no auto-delete)
                    processed += 1
                    skipped += 1  # Still count as skipped for reporting purposes
    
    return {'asset': asset, 'markets': processed, 'skipped': skipped, 'already_processed': already_processed}


def main():
    parser = argparse.ArgumentParser(
        description="Process raw continuous data into market folders"
    )
    parser.add_argument(
        '--raw-dir',
        type=str,
        default=STORAGE.raw_dir,
        help=f'Raw data directory (default: {STORAGE.raw_dir})'
    )
    parser.add_argument(
        '--markets-dir',
        type=str,
        default=STORAGE.markets_dir,
        help=f'Output markets directory (default: {STORAGE.markets_dir})'
    )
    parser.add_argument(
        '--assets',
        type=str,
        default=None,
        help='Comma-separated list of assets (default: all found in raw data)'
    )
    parser.add_argument(
        '--min-coverage',
        type=float,
        default=70.0,
        help='Minimum BOTH coverage percentage (CL+PM intersection). Default: 70.0. '
             'Markets below this are marked as low coverage but kept (no auto-delete).'
    )
    
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    markets_dir = Path(args.markets_dir)
    
    print("=" * 60)
    print("Processing Raw Data into Market Folders")
    print("=" * 60)
    print(f"  Raw directory: {raw_dir}")
    print(f"  Markets directory: {markets_dir}")
    
    # Find assets to process
    if args.assets:
        assets = [a.strip().upper() for a in args.assets.split(',')]
    else:
        # Auto-detect from raw data
        assets = []
        pm_dir = raw_dir / "polymarket"
        if pm_dir.exists():
            for d in pm_dir.iterdir():
                if d.is_dir() and d.name.upper() in SUPPORTED_ASSETS:
                    assets.append(d.name.upper())
    
    if not assets:
        print("\nNo assets found to process!")
        return 1
    
    print(f"  Assets: {', '.join(assets)}")
    
    # Process each asset
    results = []
    for asset in assets:
        result = process_asset(asset, raw_dir, markets_dir, args.min_coverage)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)
    
    total_markets = sum(r['markets'] for r in results)
    total_skipped = sum(r['skipped'] for r in results)
    total_already_processed = sum(r.get('already_processed', 0) for r in results)
    
    for r in results:
        already = r.get('already_processed', 0)
        if already > 0:
            print(f"  {r['asset']}: {r['markets']} markets processed, {r['skipped']} low coverage, {already} already processed (skipped)")
        else:
            print(f"  {r['asset']}: {r['markets']} markets processed, {r['skipped']} low coverage")
    
    if total_already_processed > 0:
        print(f"\n  Total: {total_markets} markets processed, {total_skipped} low coverage, {total_already_processed} already processed (skipped)")
    else:
        print(f"\n  Total: {total_markets} markets processed, {total_skipped} low coverage")
    print(f"\n  Output: {markets_dir}")
    
    if total_markets == 0 and total_already_processed == 0:
        print("\n[WARN] No markets found. Collect more data!")
        return 1
    elif total_markets == 0 and total_already_processed > 0:
        print(f"\n[INFO] All markets already processed ({total_already_processed} skipped). No new markets to process.")
        return 0
    
    print("\n[OK] Processing complete!")
    print("\nNext steps:")
    print(f"  1. Build dataset: python scripts/cli_v2.py build")
    print(f"  2. Validate: python scripts/cli_v2.py validate")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

