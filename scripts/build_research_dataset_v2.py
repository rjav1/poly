"""
Build canonical research dataset from market data - Version 2.

Enhanced version with:
1. Explicit forward-fill flags (cl_ffill, pm_ffill) 
2. Fixed coverage math (consistent denominators)
3. Ground truth integration for K and outcomes
4. Multi-asset support
5. Better missingness tracking

Output: A single canonical panel dataset where each row is:
(market_id, asset, t, τ, Δ, polymarket prices, chainlink prices, spreads, ffill flags, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ground_truth import GroundTruthBuilder, GroundTruthRepository, MarketGroundTruth
from config.settings import STORAGE, SUPPORTED_ASSETS, get_asset_config

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. QC plots will be skipped.")


# =============================================================================
# CONFIGURATION
# =============================================================================

MARKET_DURATION_SECONDS = 900  # 15 minutes
OUTPUT_DIR = Path(STORAGE.research_dir)
MARKETS_DIR = Path(STORAGE.markets_dir)


# =============================================================================
# DATA LOADING
# =============================================================================

def find_market_folders(asset: Optional[str] = None) -> List[Path]:
    """
    Find all valid market folders with both CL and PM data.
    
    New structure: data_v2/markets/{asset}/{market_id}/
    Legacy structure: data_v2/markets/{market_id}/ (for backwards compatibility)
    
    Args:
        asset: Optional asset filter (BTC, ETH, SOL, XRP)
    
    Returns:
        List of valid market folder paths
    """
    if not MARKETS_DIR.exists():
        print(f"Warning: Markets directory not found: {MARKETS_DIR}")
        return []
    
    valid_folders = []
    
    # Check new organized structure: data_v2/markets/{asset}/{market_id}/
    for asset_dir in sorted(MARKETS_DIR.iterdir()):
        if not asset_dir.is_dir():
            continue
        
        # If asset filter specified, skip other assets
        if asset and asset_dir.name.upper() != asset.upper():
            continue
        
        # Look for market folders within asset directory
        for market_folder in sorted(asset_dir.iterdir()):
            if not market_folder.is_dir():
                continue
            
            # Look for CSV files (new naming: chainlink.csv, polymarket.csv)
            cl_files = list(market_folder.glob("chainlink*.csv")) + list(market_folder.glob("chainlink*.parquet"))
            pm_files = list(market_folder.glob("polymarket*.csv")) + list(market_folder.glob("polymarket*.parquet"))
            
            if cl_files and pm_files:
                valid_folders.append(market_folder)
            else:
                print(f"  Skipping {asset_dir.name}/{market_folder.name}: missing CL or PM data")
    
    # Also check legacy flat structure for backwards compatibility
    if not valid_folders:
        for folder in sorted(MARKETS_DIR.iterdir()):
            if not folder.is_dir():
                continue
            
            # Skip if this looks like an asset directory (all caps, short name)
            if folder.name.upper() in ["BTC", "ETH", "SOL", "XRP"]:
                continue
            
            cl_files = list(folder.glob("chainlink_*.csv"))
            pm_files = list(folder.glob("polymarket_*.csv"))
            
            if cl_files and pm_files:
                valid_folders.append(folder)
                print(f"  Found legacy market: {folder.name}")
    
    return valid_folders


def parse_market_info_from_folder(folder: Path) -> Dict:
    """Parse market info from folder name and parent directory."""
    name = folder.name
    parts = name.split('_')
    
    # Extract asset from parent directory (new structure) or default to BTC
    asset = "BTC"
    if folder.parent.name.upper() in ["BTC", "ETH", "SOL", "XRP"]:
        asset = folder.parent.name.upper()
    
    # Format: YYYYMMDD_HHMM or YYYYMMDD_HHMM_PRICE
    date_str = parts[0]
    time_str = parts[1]
    price_to_beat = int(parts[2]) if len(parts) > 2 else None
    
    dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
    market_start = dt.replace(tzinfo=timezone.utc)
    market_end = market_start + timedelta(seconds=MARKET_DURATION_SECONDS)
    
    return {
        'market_id': name,
        'market_start': market_start,
        'market_end': market_end,
        'price_to_beat_from_folder': price_to_beat,
        'asset': asset
    }


def load_market_data(folder: Path) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load Chainlink and Polymarket data for a market."""
    # Try new naming first: chainlink.csv, polymarket.csv
    cl_files = list(folder.glob("chainlink.csv")) + list(folder.glob("chainlink_*.csv"))
    pm_files = list(folder.glob("polymarket.csv")) + list(folder.glob("polymarket_*.csv"))
    summary_files = list(folder.glob("summary.json")) + list(folder.glob("summary_*.json"))
    
    if not cl_files or not pm_files:
        raise FileNotFoundError(f"Missing data files in {folder}")
    
    df_cl = pd.read_csv(cl_files[0], on_bad_lines='skip', low_memory=False)
    df_pm = pd.read_csv(pm_files[0], on_bad_lines='skip', low_memory=False)
    
    # Ensure numeric columns are actually numeric (may have strings from misaligned rows)
    numeric_cols_cl = ['mid', 'bid', 'ask']
    for col in numeric_cols_cl:
        if col in df_cl.columns:
            df_cl[col] = pd.to_numeric(df_cl[col], errors='coerce')
    
    # Parse timestamps using ISO8601 format for mixed precision
    # IMPORTANT: Use 'timestamp' (actual data time) for coverage calculation, NOT 'collected_at'
    # CL data has ~2 min delay between timestamp (data time) and collected_at (fetch time)
    if 'timestamp' in df_cl.columns:
        df_cl['timestamp'] = pd.to_datetime(df_cl['timestamp'], format='ISO8601', errors='coerce')
    if 'collected_at' in df_cl.columns:
        df_cl['collected_at'] = pd.to_datetime(df_cl['collected_at'], format='ISO8601', errors='coerce')
    
    if 'timestamp' in df_pm.columns:
        df_pm['timestamp'] = pd.to_datetime(df_pm['timestamp'], format='ISO8601', errors='coerce')
    if 'collected_at' in df_pm.columns:
        df_pm['collected_at'] = pd.to_datetime(df_pm['collected_at'], format='ISO8601', errors='coerce')
    
    # Load summary if available
    summary = {}
    if summary_files:
        with open(summary_files[0], 'r') as f:
            summary = json.load(f)
    
    return df_cl, df_pm, summary


# =============================================================================
# CANONICALIZATION (1-second grid) WITH FFILL FLAGS
# =============================================================================

def create_second_grid(market_start: datetime, duration_seconds: int = MARKET_DURATION_SECONDS) -> pd.DataFrame:
    """Create a 1-second grid from market start to end."""
    grid = pd.DataFrame({
        't': range(duration_seconds),
        'tau': [duration_seconds - i for i in range(duration_seconds)],
        'timestamp': [market_start + timedelta(seconds=i) for i in range(duration_seconds)]
    })
    return grid


def collapse_to_second_with_observed(
    df: pd.DataFrame, 
    market_start: datetime, 
    value_cols: List[str]
) -> Tuple[pd.DataFrame, set]:
    """
    Collapse data to 1-second grid using last observation per second.
    
    Returns:
        (collapsed_df, set of seconds with actual observations)
    """
    if df.empty:
        return pd.DataFrame(columns=['t'] + value_cols), set()
    
    df = df.copy()
    
    # Calculate seconds from market start
    # Use timestamp (actual data time) for coverage calculation
    # This ensures CL price at time T matches PM price at time T (same market moment)
    # CL timestamps are ~65s behind PM, so overlap requires longer collection
    if 'seconds' in df.columns:
        # Use pre-calculated seconds column (from process_raw_data.py)
        df['t'] = df['seconds'].astype(int)
    elif 'timestamp' in df.columns:
        # Use timestamp (actual data time) - this is the correct way
        df['t'] = (df['timestamp'] - market_start).dt.total_seconds().astype(int)
    else:
        # Fallback to collected_at (legacy)
        df['t'] = (df['collected_at'] - market_start).dt.total_seconds().astype(int)
    
    # Filter to valid range [0, 899]
    df = df[(df['t'] >= 0) & (df['t'] < MARKET_DURATION_SECONDS)]
    
    if df.empty:
        return pd.DataFrame(columns=['t'] + value_cols), set()
    
    # Track which seconds have observations
    observed_seconds = set(df['t'].unique())
    
    # Keep only needed columns
    keep_cols = ['t'] + [c for c in value_cols if c in df.columns]
    df = df[keep_cols]
    
    # Group by second, take last observation
    df = df.groupby('t').last().reset_index()
    
    return df, observed_seconds


def forward_fill_to_grid_with_flags(
    grid: pd.DataFrame, 
    data: pd.DataFrame, 
    value_cols: List[str], 
    observed_seconds: set,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Merge data onto grid, forward-fill, and add ffill flags.
    
    Adds {prefix}ffill column: 1 if value was forward-filled, 0 if observed.
    """
    result = grid.copy()
    
    # Add ffill flag column
    ffill_col = f'{prefix}ffill' if prefix else 'ffill'
    
    if data.empty:
        # Add empty columns
        for col in value_cols:
            result[f'{prefix}{col}'] = np.nan
        result[ffill_col] = 1  # All forward-filled (no data)
        return result
    
    # Merge on t
    merge_cols = ['t'] + [c for c in value_cols if c in data.columns]
    result = result.merge(data[merge_cols], on='t', how='left')
    
    # Rename columns with prefix
    if prefix:
        for col in value_cols:
            if col in result.columns:
                result.rename(columns={col: f'{prefix}{col}'}, inplace=True)
    
    # Set ffill flag BEFORE forward-filling
    result[ffill_col] = result['t'].apply(lambda x: 0 if x in observed_seconds else 1)
    
    # Forward fill
    for col in value_cols:
        col_name = f'{prefix}{col}' if prefix else col
        if col_name in result.columns:
            result[col_name] = result[col_name].ffill()
    
    return result


# =============================================================================
# STRIKE, SETTLEMENT, AND OUTCOME (with Ground Truth integration)
# =============================================================================

def compute_strike_and_settlement(
    df_cl: pd.DataFrame, 
    market_start: datetime, 
    market_end: datetime,
    ground_truth: Optional[Dict] = None
) -> Tuple[float, float, int, Dict]:
    """
    Compute official strike (K), settlement price, and outcome (Y).
    
    Uses ground truth if available, otherwise computes from data.
    
    Returns:
        (K, settlement_price, Y, metadata_dict)
    """
    metadata = {
        'source': 'computed',
        'k_offset_seconds': 0,
        'settlement_offset_seconds': 0,
        'has_exact_k': False,
        'has_exact_settlement': False
    }
    
    # Try to use ground truth first
    if ground_truth and ground_truth.get('strike_K') is not None:
        K = ground_truth['strike_K']
        settlement = ground_truth.get('settlement_price')
        if ground_truth.get('computed_outcome'):
            Y = 1 if ground_truth['computed_outcome'].lower() == 'up' else 0
        elif K is not None and settlement is not None:
            Y = 1 if settlement >= K else 0
        else:
            Y = np.nan
        metadata['source'] = 'ground_truth'
        return K, settlement, Y, metadata
    
    # Compute from data
    if df_cl.empty or 'mid' not in df_cl.columns:
        return np.nan, np.nan, np.nan, metadata
    
    df = df_cl.copy()
    # Use timestamp (actual data time) for strike/settlement, not collected_at
    ts_col = 'timestamp' if 'timestamp' in df.columns else 'collected_at'
    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col], format='ISO8601')
    
    # Strike: first observation at/after market start
    df_start = df[df[ts_col] >= market_start].sort_values(ts_col)
    if df_start.empty:
        # Fallback: last observation before start
        df_before = df[df[ts_col] < market_start].sort_values(ts_col, ascending=False)
        if df_before.empty:
            K = np.nan
        else:
            K = df_before.iloc[0]['mid']
            metadata['k_offset_seconds'] = (market_start - df_before.iloc[0][ts_col]).total_seconds()
    else:
        K = df_start.iloc[0]['mid']
        offset = (df_start.iloc[0][ts_col] - market_start).total_seconds()
        metadata['k_offset_seconds'] = offset
        metadata['has_exact_k'] = offset <= 5
    
    # Settlement: first observation at/after market end (or last before if none after)
    df_end = df[df[ts_col] >= market_end].sort_values(ts_col)
    if df_end.empty:
        df_before_end = df[df[ts_col] < market_end].sort_values(ts_col, ascending=False)
        if df_before_end.empty:
            settlement = np.nan
        else:
            settlement = df_before_end.iloc[0]['mid']
            metadata['settlement_offset_seconds'] = (market_end - df_before_end.iloc[0][ts_col]).total_seconds()
    else:
        settlement = df_end.iloc[0]['mid']
        offset = (df_end.iloc[0][ts_col] - market_end).total_seconds()
        metadata['settlement_offset_seconds'] = offset
        metadata['has_exact_settlement'] = offset <= 5
    
    # Outcome
    if pd.isna(K) or pd.isna(settlement):
        Y = np.nan
    else:
        Y = 1 if settlement >= K else 0
    
    return K, settlement, Y, metadata


# =============================================================================
# DERIVED FEATURES
# =============================================================================

def add_derived_features(df: pd.DataFrame, K: float) -> pd.DataFrame:
    """Add derived features for strategy research."""
    df = df.copy()
    
    # Polymarket midpoint prices (use pm_up_mid directly if available)
    if 'pm_up_mid' not in df.columns:
        if 'pm_up_best_bid' in df.columns and 'pm_up_best_ask' in df.columns:
            df['pm_up_mid'] = (df['pm_up_best_bid'] + df['pm_up_best_ask']) / 2
    
    if 'pm_down_mid' not in df.columns:
        if 'pm_down_best_bid' in df.columns and 'pm_down_best_ask' in df.columns:
            df['pm_down_mid'] = (df['pm_down_best_bid'] + df['pm_down_best_ask']) / 2
    
    # Spreads (in probability points)
    if 'pm_up_best_bid' in df.columns and 'pm_up_best_ask' in df.columns:
        df['pm_up_spread'] = df['pm_up_best_ask'] - df['pm_up_best_bid']
    
    if 'pm_down_best_bid' in df.columns and 'pm_down_best_ask' in df.columns:
        df['pm_down_spread'] = df['pm_down_best_ask'] - df['pm_down_best_bid']
    
    # Complete-set no-arb checks
    if 'pm_up_best_bid' in df.columns and 'pm_down_best_bid' in df.columns:
        df['sum_bids'] = df['pm_up_best_bid'] + df['pm_down_best_bid']
    
    if 'pm_up_best_ask' in df.columns and 'pm_down_best_ask' in df.columns:
        df['sum_asks'] = df['pm_up_best_ask'] + df['pm_down_best_ask']
    
    # Distance to strike
    if 'cl_mid' in df.columns and not pd.isna(K):
        df['delta'] = df['cl_mid'] - K
        df['delta_bps'] = df['delta'] / K * 10000  # basis points
    
    return df


# =============================================================================
# MISSINGNESS TRACKING (Fixed Coverage Math)
# =============================================================================

def compute_coverage_stats(
    cl_observed_seconds: set,
    pm_observed_seconds: set,
    total_seconds: int = MARKET_DURATION_SECONDS
) -> Dict:
    """
    Compute coverage statistics with CORRECT math.
    
    Key insight: "both present" is intersection, not some average.
    """
    cl_count = len(cl_observed_seconds)
    pm_count = len(pm_observed_seconds)
    both_count = len(cl_observed_seconds & pm_observed_seconds)  # Intersection
    either_count = len(cl_observed_seconds | pm_observed_seconds)  # Union
    
    return {
        'cl_observed_seconds': cl_count,
        'pm_observed_seconds': pm_count,
        'both_observed_seconds': both_count,
        'either_observed_seconds': either_count,
        'cl_coverage_pct': cl_count / total_seconds * 100,
        'pm_coverage_pct': pm_count / total_seconds * 100,
        'both_coverage_pct': both_count / total_seconds * 100,
        'either_coverage_pct': either_count / total_seconds * 100,
        'cl_missing_seconds': total_seconds - cl_count,
        'pm_missing_seconds': total_seconds - pm_count,
    }


def compute_raw_data_stats(
    df_cl_raw: pd.DataFrame, 
    df_pm_raw: pd.DataFrame,
    market_start: datetime
) -> Dict:
    """Compute statistics about raw data before processing."""
    stats = {
        'cl_raw_rows': len(df_cl_raw),
        'pm_raw_rows': len(df_pm_raw),
        'cl_multi_obs_seconds': 0,
        'pm_multi_obs_seconds': 0,
    }
    
    if not df_cl_raw.empty:
        df = df_cl_raw.copy()
        # Use pre-calculated seconds or timestamp (actual data time)
        if 'seconds' in df.columns:
            df['t'] = df['seconds'].astype(int)
        elif 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
            df['t'] = (df['timestamp'] - market_start).dt.total_seconds().astype(int)
        else:
            df['collected_at'] = pd.to_datetime(df['collected_at'], format='ISO8601')
            df['t'] = (df['collected_at'] - market_start).dt.total_seconds().astype(int)
        df = df[(df['t'] >= 0) & (df['t'] < MARKET_DURATION_SECONDS)]
        counts = df.groupby('t').size()
        stats['cl_multi_obs_seconds'] = (counts > 1).sum()
    
    if not df_pm_raw.empty:
        df = df_pm_raw.copy()
        # Use pre-calculated seconds or timestamp (actual data time)
        if 'seconds' in df.columns:
            df['t'] = df['seconds'].astype(int)
        elif 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
            df['t'] = (df['timestamp'] - market_start).dt.total_seconds().astype(int)
        else:
            df['collected_at'] = pd.to_datetime(df['collected_at'], format='ISO8601')
            df['t'] = (df['collected_at'] - market_start).dt.total_seconds().astype(int)
        df = df[(df['t'] >= 0) & (df['t'] < MARKET_DURATION_SECONDS)]
        counts = df.groupby('t').size()
        stats['pm_multi_obs_seconds'] = (counts > 1).sum()
    
    return stats


# =============================================================================
# PROCESS SINGLE MARKET
# =============================================================================

def process_market(folder: Path, ground_truth_repo: Optional[GroundTruthRepository] = None) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Process a single market into canonical format.
    
    Returns:
        (canonical_df, market_info_dict)
    """
    market_info = parse_market_info_from_folder(folder)
    market_id = market_info['market_id']
    market_start = market_info['market_start']
    market_end = market_info['market_end']
    asset = market_info['asset']
    
    print(f"\nProcessing: {market_id}")
    
    # Load raw data
    df_cl, df_pm, summary = load_market_data(folder)
    
    # Get ground truth if available
    ground_truth = None
    if ground_truth_repo:
        # Try to find matching ground truth
        gt_data = ground_truth_repo.load_all()
        for gt_id, gt in gt_data.items():
            if market_id in gt_id or gt_id in market_id:
                ground_truth = gt
                break
    
    # Compute raw data stats
    raw_stats = compute_raw_data_stats(df_cl, df_pm, market_start)
    
    # Collapse to seconds and track observed
    # Note: collapse_to_second_with_observed uses 'collected_at' to determine observed seconds,
    # not the value columns, so missing value columns won't affect coverage calculation
    cl_cols = ['mid', 'bid', 'ask']
    
    # Polymarket columns - check what's actually available and map accordingly
    # Raw data has: up_bid, up_ask (without "best_")
    # Processed data should have: up_best_bid, up_best_ask (with "best_")
    # Build script expects: up_best_bid, up_best_ask, etc.
    pm_cols_base = ['up_mid', 'down_mid']
    pm_cols_optional = ['up_best_bid', 'up_best_ask', 'up_best_bid_size', 'up_best_ask_size',
                        'down_best_bid', 'down_best_ask', 'down_best_bid_size', 'down_best_ask_size']
    
    # L2-L6 depth columns (48 columns: 4 outcomes × 2 sides × 6 levels for price+size)
    for level in range(2, 7):
        pm_cols_optional.extend([
            f'up_bid_{level}', f'up_bid_{level}_size',
            f'up_ask_{level}', f'up_ask_{level}_size',
            f'down_bid_{level}', f'down_bid_{level}_size',
            f'down_ask_{level}', f'down_ask_{level}_size',
        ])
    
    # Also check for non-"best_" versions
    pm_cols_alt = ['up_bid', 'up_ask', 'down_bid', 'down_ask']
    
    # Build list of columns that actually exist
    pm_cols = pm_cols_base.copy()
    for col in pm_cols_optional:
        if col in df_pm.columns:
            pm_cols.append(col)
    # If "best_" versions don't exist, check for non-"best_" versions
    for alt_col in pm_cols_alt:
        best_col = alt_col.replace('_bid', '_best_bid').replace('_ask', '_best_ask')
        if alt_col in df_pm.columns and best_col not in pm_cols:
            pm_cols.append(alt_col)  # Will need to rename later
    
    df_cl_sec, cl_observed = collapse_to_second_with_observed(df_cl, market_start, cl_cols)
    df_pm_sec, pm_observed = collapse_to_second_with_observed(df_pm, market_start, pm_cols)
    
    # Compute coverage stats with CORRECT math
    coverage_stats = compute_coverage_stats(cl_observed, pm_observed)
    print(f"  Coverage: CL={coverage_stats['cl_coverage_pct']:.1f}%, "
          f"PM={coverage_stats['pm_coverage_pct']:.1f}%, "
          f"Both={coverage_stats['both_coverage_pct']:.1f}%")
    
    # Create grid
    grid = create_second_grid(market_start)
    
    # Forward-fill to grid WITH FFILL FLAGS
    canonical = forward_fill_to_grid_with_flags(grid, df_cl_sec, cl_cols, cl_observed, prefix='cl_')
    canonical = forward_fill_to_grid_with_flags(canonical, df_pm_sec, pm_cols, pm_observed, prefix='pm_')
    
    # Compute strike, settlement, outcome
    K, settlement, Y, k_meta = compute_strike_and_settlement(df_cl, market_start, market_end, ground_truth)
    
    # Ensure K and settlement are floats (may come as strings from CSV)
    try:
        K = float(K) if not pd.isna(K) else np.nan
    except (ValueError, TypeError):
        K = np.nan
    
    try:
        settlement = float(settlement) if not pd.isna(settlement) else np.nan
    except (ValueError, TypeError):
        settlement = np.nan
    
    # Handle K and settlement formatting for display
    try:
        K_val = float(K) if not pd.isna(K) else None
        K_str = f"${K_val:.2f}" if K_val is not None else "N/A"
    except (ValueError, TypeError):
        K_str = str(K) if not pd.isna(K) else "N/A"
    
    try:
        settlement_val = float(settlement) if not pd.isna(settlement) else None
        settlement_str = f"${settlement_val:.2f}" if settlement_val is not None else "N/A"
    except (ValueError, TypeError):
        settlement_str = str(settlement) if not pd.isna(settlement) else "N/A"
    
    print(f"  Strike (K): {K_str}")
    print(f"  Settlement: {settlement_str}")
    print(f"  Outcome (Y): {'UP' if Y == 1 else 'DOWN' if Y == 0 else 'N/A'} (source: {k_meta['source']})")
    
    # Add market identifiers
    canonical['market_id'] = market_id
    canonical['asset'] = asset
    canonical['market_start'] = market_start
    canonical['K'] = K
    canonical['settlement'] = settlement
    canonical['Y'] = Y
    
    # Add derived features
    canonical = add_derived_features(canonical, K)
    
    # Compile market info
    full_info = {
        'market_id': market_id,
        'asset': asset,
        'market_start': market_start.isoformat(),
        'market_end': market_end.isoformat(),
        'K': K,
        'settlement': settlement,
        'Y': Y,
        'price_to_beat_from_folder': market_info['price_to_beat_from_folder'],
        'k_source': k_meta['source'],
        'k_offset_seconds': k_meta['k_offset_seconds'],
        'settlement_offset_seconds': k_meta['settlement_offset_seconds'],
        'has_exact_k': k_meta['has_exact_k'],
        'has_exact_settlement': k_meta['has_exact_settlement'],
        **raw_stats,
        **coverage_stats
    }
    
    return canonical, full_info


# =============================================================================
# QC STATISTICS
# =============================================================================

def compute_qc_stats(df: pd.DataFrame, market_infos: List[Dict]) -> Dict:
    """Compute aggregate QC statistics."""
    n_markets = len(market_infos)
    
    stats = {
        'n_markets': n_markets,
        'n_rows': len(df),
        'n_up_outcomes': sum(1 for m in market_infos if m['Y'] == 1),
        'n_down_outcomes': sum(1 for m in market_infos if m['Y'] == 0),
    }
    
    # Coverage stats (averaged correctly)
    if market_infos:
        stats['avg_cl_coverage_pct'] = np.mean([m['cl_coverage_pct'] for m in market_infos])
        stats['avg_pm_coverage_pct'] = np.mean([m['pm_coverage_pct'] for m in market_infos])
        stats['avg_both_coverage_pct'] = np.mean([m['both_coverage_pct'] for m in market_infos])
    
    # Forward-fill stats
    if 'cl_ffill' in df.columns:
        stats['cl_ffill_pct'] = df['cl_ffill'].mean() * 100
    if 'pm_ffill' in df.columns:
        stats['pm_ffill_pct'] = df['pm_ffill'].mean() * 100
    
    # Spread statistics
    if 'pm_up_spread' in df.columns:
        valid_spreads = df[(df['pm_up_spread'] > 0) & (df['pm_up_spread'] < 0.5)]['pm_up_spread']
        if len(valid_spreads) > 0:
            stats['avg_up_spread'] = valid_spreads.mean()
            stats['median_up_spread'] = valid_spreads.median()
            stats['p95_up_spread'] = valid_spreads.quantile(0.95)
    
    if 'pm_down_spread' in df.columns:
        valid_spreads = df[(df['pm_down_spread'] > 0) & (df['pm_down_spread'] < 0.5)]['pm_down_spread']
        if len(valid_spreads) > 0:
            stats['avg_down_spread'] = valid_spreads.mean()
            stats['median_down_spread'] = valid_spreads.median()
            stats['p95_down_spread'] = valid_spreads.quantile(0.95)
    
    # No-arb violations
    if 'sum_bids' in df.columns:
        valid = df['sum_bids'].notna()
        stats['overround_pct'] = (df.loc[valid, 'sum_bids'] > 1).mean() * 100
    
    if 'sum_asks' in df.columns:
        valid = df['sum_asks'].notna()
        stats['underround_pct'] = (df.loc[valid, 'sum_asks'] < 1).mean() * 100
    
    # K accuracy (compare folder price to computed K)
    k_matches = [m for m in market_infos if m['price_to_beat_from_folder'] is not None and not pd.isna(m['K'])]
    if k_matches:
        diffs = [abs(m['K'] - m['price_to_beat_from_folder']) for m in k_matches]
        stats['avg_k_folder_diff'] = np.mean(diffs)
        stats['k_within_10'] = sum(1 for d in diffs if d <= 10) / len(diffs) * 100
    
    return stats


# =============================================================================
# QC PLOTS
# =============================================================================

def generate_qc_plots(df: pd.DataFrame, market_infos: List[Dict], output_dir: Path):
    """Generate QC diagnostic plots."""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, skipping QC plots")
        return
    
    print("\nGenerating QC plots...")
    plots_dir = output_dir / "qc_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # PLOT 1: Coverage Distribution with FFill breakdown
    # ==========================================================================
    fig1 = make_subplots(rows=2, cols=2, subplot_titles=(
        "CL Coverage %", "PM Coverage %",
        "CL FFill % per Market", "PM FFill % per Market"
    ))
    
    cl_coverage = [m['cl_coverage_pct'] for m in market_infos]
    pm_coverage = [m['pm_coverage_pct'] for m in market_infos]
    
    fig1.add_trace(go.Histogram(x=cl_coverage, nbinsx=20, name="CL"), row=1, col=1)
    fig1.add_trace(go.Histogram(x=pm_coverage, nbinsx=20, name="PM"), row=1, col=2)
    
    # FFill percentage by market
    if 'cl_ffill' in df.columns:
        cl_ffill_by_market = df.groupby('market_id')['cl_ffill'].mean() * 100
        fig1.add_trace(go.Bar(x=list(range(len(cl_ffill_by_market))), y=cl_ffill_by_market.values, name="CL FFill"), row=2, col=1)
    
    if 'pm_ffill' in df.columns:
        pm_ffill_by_market = df.groupby('market_id')['pm_ffill'].mean() * 100
        fig1.add_trace(go.Bar(x=list(range(len(pm_ffill_by_market))), y=pm_ffill_by_market.values, name="PM FFill"), row=2, col=2)
    
    fig1.update_layout(title="Data Coverage and Forward-Fill Analysis", template="plotly_dark", height=600, showlegend=False)
    fig1.write_html(str(plots_dir / "qc_01_coverage.html"))
    print(f"  Saved: qc_01_coverage.html")
    
    # ==========================================================================
    # PLOT 2: Spread Analysis
    # ==========================================================================
    fig2 = make_subplots(rows=2, cols=2, subplot_titles=(
        "UP Spread Distribution", "DOWN Spread Distribution",
        "UP Spread vs tau", "DOWN Spread vs tau"
    ))
    
    if 'pm_up_spread' in df.columns and 'pm_down_spread' in df.columns:
        df_valid = df[(df['pm_up_spread'] > 0) & (df['pm_up_spread'] < 0.5) & 
                      (df['pm_down_spread'] > 0) & (df['pm_down_spread'] < 0.5)].copy()
        
        if not df_valid.empty:
            fig2.add_trace(go.Histogram(x=df_valid['pm_up_spread'], nbinsx=50), row=1, col=1)
            fig2.add_trace(go.Histogram(x=df_valid['pm_down_spread'], nbinsx=50), row=1, col=2)
            
            sample = df_valid.sample(min(5000, len(df_valid)))
            fig2.add_trace(go.Scatter(x=sample['tau'], y=sample['pm_up_spread'], mode='markers',
                                       marker=dict(size=2, opacity=0.3)), row=2, col=1)
            fig2.add_trace(go.Scatter(x=sample['tau'], y=sample['pm_down_spread'], mode='markers',
                                       marker=dict(size=2, opacity=0.3)), row=2, col=2)
    else:
        print("  [WARN] Spread columns not found, skipping spread plots")
    
    fig2.update_layout(title="Spread Analysis", template="plotly_dark", height=600, showlegend=False)
    fig2.write_html(str(plots_dir / "qc_02_spreads.html"))
    print(f"  Saved: qc_02_spreads.html")
    
    # ==========================================================================
    # PLOT 3: No-Arb Analysis
    # ==========================================================================
    fig3 = make_subplots(rows=1, cols=2, subplot_titles=("Sum of Bids", "Sum of Asks"))
    
    df_noarb = df[df['sum_bids'].notna() & df['sum_asks'].notna()].copy()
    
    if not df_noarb.empty:
        fig3.add_trace(go.Histogram(x=df_noarb['sum_bids'], nbinsx=50), row=1, col=1)
        fig3.add_trace(go.Histogram(x=df_noarb['sum_asks'], nbinsx=50), row=1, col=2)
        fig3.add_vline(x=1, line_dash="dash", line_color="red", row=1, col=1)
        fig3.add_vline(x=1, line_dash="dash", line_color="red", row=1, col=2)
    
    fig3.update_layout(title="Complete-Set No-Arb Check", template="plotly_dark", height=400, showlegend=False)
    fig3.write_html(str(plots_dir / "qc_03_noarb.html"))
    print(f"  Saved: qc_03_noarb.html")
    
    # ==========================================================================
    # PLOT 4: Outcome Distribution
    # ==========================================================================
    fig4 = go.Figure()
    
    outcomes = {'UP': sum(1 for m in market_infos if m['Y'] == 1),
                'DOWN': sum(1 for m in market_infos if m['Y'] == 0),
                'N/A': sum(1 for m in market_infos if pd.isna(m['Y']))}
    
    fig4.add_trace(go.Bar(x=list(outcomes.keys()), y=list(outcomes.values())))
    fig4.update_layout(title="Outcome Distribution", template="plotly_dark", height=400)
    fig4.write_html(str(plots_dir / "qc_04_outcomes.html"))
    print(f"  Saved: qc_04_outcomes.html")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Build canonical research dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build canonical research dataset v2")
    parser.add_argument("--markets-dir", type=str, default=None, help="Markets directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--use-ground-truth", action="store_true", help="Use ground truth for K")
    parser.add_argument("--min-coverage", type=float, default=70.0, 
                        help="Minimum BOTH coverage percentage (CL+PM intersection) to include market (0-100, default: 70.0)")
    
    args = parser.parse_args()
    
    global MARKETS_DIR, OUTPUT_DIR
    if args.markets_dir:
        MARKETS_DIR = Path(args.markets_dir)
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Building Canonical Research Dataset v2")
    print("=" * 60)
    
    # Find market folders
    print(f"\nLooking for markets in: {MARKETS_DIR}")
    folders = find_market_folders()
    print(f"Found {len(folders)} valid market folders")
    
    if not folders:
        print("No market data found. Exiting.")
        return
    
    # Load ground truth if requested
    ground_truth_repo = None
    if args.use_ground_truth:
        ground_truth_repo = GroundTruthRepository()
        print(f"Using ground truth from: {ground_truth_repo.storage_dir}")
    
    print(f"\nMinimum coverage threshold: {args.min_coverage}%")
    print("Markets below this threshold will be excluded.\n")
    
    # Process all markets, organized by asset
    all_dfs = []
    market_infos = []
    markets_by_asset: Dict[str, List] = {}
    skipped_low_coverage = []
    
    for folder in folders:
        try:
            df, info = process_market(folder, ground_truth_repo)
            if df is not None:
                # Filter by minimum coverage
                both_coverage = info.get('both_coverage_pct', 0)
                
                if both_coverage >= args.min_coverage:
                    all_dfs.append(df)
                    market_infos.append(info)
                    
                    # Group by asset
                    asset = info.get('asset', 'UNKNOWN')
                    if asset not in markets_by_asset:
                        markets_by_asset[asset] = []
                    markets_by_asset[asset].append((df, info))
                else:
                    skipped_low_coverage.append({
                        'market_id': info.get('market_id'),
                        'coverage': both_coverage
                    })
                    print(f"  [WARN] Skipping {info.get('market_id')}: coverage too low ({both_coverage:.1f}% < {args.min_coverage}%)")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Report skipped markets
    if skipped_low_coverage:
        print(f"\n[WARN] Skipped {len(skipped_low_coverage)} market(s) due to low coverage:")
        for skipped in skipped_low_coverage:
            print(f"    - {skipped['market_id']}: {skipped['coverage']:.1f}% coverage")
    
    if not all_dfs:
        print("\nNo markets processed successfully. Exiting.")
        return
    
    # Combine all markets
    print(f"\n{'=' * 60}")
    print("Combining all markets...")
    
    if not all_dfs:
        print("\n❌ No markets passed the coverage threshold. Exiting.")
        if skipped_low_coverage:
            print(f"\nAll {len(skipped_low_coverage)} market(s) were below {args.min_coverage}% coverage.")
        return
    
    canonical_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows: {len(canonical_df):,}")
    print(f"Total markets included: {len(market_infos)}")
    if skipped_low_coverage:
        print(f"Markets excluded (low coverage): {len(skipped_low_coverage)}")
    print(f"Assets: {', '.join(sorted(markets_by_asset.keys()))}")
    
    # Compute QC stats
    print("\nComputing QC statistics...")
    qc_stats = compute_qc_stats(canonical_df, market_infos)
    
    # Save outputs organized by asset
    print("\nSaving outputs...")
    
    # Save combined dataset (all assets)
    canonical_df.to_parquet(OUTPUT_DIR / "canonical_dataset_all_assets.parquet", index=False)
    print(f"  Saved: canonical_dataset_all_assets.parquet")
    
    canonical_df.to_csv(OUTPUT_DIR / "canonical_dataset_all_assets.csv", index=False)
    print(f"  Saved: canonical_dataset_all_assets.csv")
    
    # Save per-asset datasets
    for asset, asset_markets in markets_by_asset.items():
        asset_dir = OUTPUT_DIR / asset.upper()
        asset_dir.mkdir(parents=True, exist_ok=True)
        
        asset_dfs = [df for df, _ in asset_markets]
        asset_info = [info for _, info in asset_markets]
        
        if asset_dfs:
            asset_df = pd.concat(asset_dfs, ignore_index=True)
            asset_df.to_parquet(asset_dir / f"canonical_dataset_{asset.upper()}.parquet", index=False)
            asset_df.to_csv(asset_dir / f"canonical_dataset_{asset.upper()}.csv", index=False)
            
            with open(asset_dir / f"market_info_{asset.upper()}.json", 'w') as f:
                json.dump(asset_info, f, indent=2, default=str)
            
            asset_qc_stats = compute_qc_stats(asset_df, asset_info)
            with open(asset_dir / f"qc_stats_{asset.upper()}.json", 'w') as f:
                json.dump(asset_qc_stats, f, indent=2, default=str)
            
            print(f"  Saved {asset}: {len(asset_df):,} rows, {len(asset_info)} markets")
    
    # Save combined market info
    with open(OUTPUT_DIR / "market_info_all_assets.json", 'w') as f:
        json.dump(market_infos, f, indent=2, default=str)
    print(f"  Saved: market_info_all_assets.json")
    
    # Save combined QC stats
    with open(OUTPUT_DIR / "qc_stats_all_assets.json", 'w') as f:
        json.dump(qc_stats, f, indent=2, default=str)
    print(f"  Saved: qc_stats_all_assets.json")
    
    # Generate QC plots (combined)
    plots_dir = OUTPUT_DIR / "qc_plots"
    generate_qc_plots(canonical_df, market_infos, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "=" * 60)
    print("QC SUMMARY")
    print("=" * 60)
    print(f"Minimum coverage threshold: {args.min_coverage}%")
    if skipped_low_coverage:
        print(f"Markets excluded (low coverage): {len(skipped_low_coverage)}")
    print(f"Markets included: {qc_stats['n_markets']}")
    print(f"Outcomes: {qc_stats['n_up_outcomes']} UP, {qc_stats['n_down_outcomes']} DOWN")
    print(f"Avg CL Coverage: {qc_stats.get('avg_cl_coverage_pct', 0):.1f}%")
    print(f"Avg PM Coverage: {qc_stats.get('avg_pm_coverage_pct', 0):.1f}%")
    print(f"Avg Both Coverage: {qc_stats.get('avg_both_coverage_pct', 0):.1f}%")
    print(f"CL FFill %: {qc_stats.get('cl_ffill_pct', 0):.1f}%")
    print(f"PM FFill %: {qc_stats.get('pm_ffill_pct', 0):.1f}%")
    print(f"Median UP Spread: {qc_stats.get('median_up_spread', 0):.3f}")
    print(f"Overround %: {qc_stats.get('overround_pct', 0):.2f}%")
    print(f"Underround %: {qc_stats.get('underround_pct', 0):.2f}%")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

