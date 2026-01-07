#!/usr/bin/env python3
"""Deep data quality check - examine raw and processed data for accuracy issues."""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import STORAGE

def check_raw_data(asset: str):
    """Check raw data files for collection issues."""
    print(f"\n{'='*80}")
    print(f"RAW DATA CHECK: {asset}")
    print(f"{'='*80}")
    
    raw_dir = Path(STORAGE.raw_dir)
    cl_path = raw_dir / "chainlink" / asset / f"chainlink_{asset}_continuous.csv"
    pm_path = raw_dir / "polymarket" / asset / f"polymarket_{asset}_continuous.csv"
    
    issues = []
    warnings = []
    
    # Check Chainlink data
    if cl_path.exists():
        print(f"\n[CHAINLINK] Loading {cl_path.name}...")
        try:
            df_cl = pd.read_csv(cl_path, on_bad_lines='skip')
            print(f"  Total rows: {len(df_cl)}")
            
            # Parse timestamps
            if 'timestamp' in df_cl.columns:
                df_cl['ts'] = pd.to_datetime(df_cl['timestamp'], format='mixed', utc=True, errors='coerce')
                df_cl = df_cl[df_cl['ts'].notna()].copy()
                
                # Check for duplicates
                df_cl['ts_second'] = df_cl['ts'].dt.floor('s')
                duplicates = df_cl[df_cl.duplicated(subset=['ts_second'], keep=False)]
                if len(duplicates) > 0:
                    issues.append(f"CL: {len(duplicates)} duplicate timestamps (same second)")
                    print(f"  [ERROR] Found {len(duplicates)} duplicate timestamps!")
                else:
                    print(f"  [OK] No duplicate timestamps")
                
                # Check for stuck timestamps
                unique_seconds = df_cl['ts_second'].nunique()
                total_seconds = len(df_cl)
                if unique_seconds < total_seconds * 0.9:
                    warnings.append(f"CL: Only {unique_seconds}/{total_seconds} unique seconds ({unique_seconds/total_seconds*100:.1f}%)")
                    print(f"  [WARN] Low unique seconds ratio: {unique_seconds}/{total_seconds} ({unique_seconds/total_seconds*100:.1f}%)")
                else:
                    print(f"  [OK] Unique seconds: {unique_seconds}/{total_seconds} ({unique_seconds/total_seconds*100:.1f}%)")
                
                # Check timestamp progression
                df_cl_sorted = df_cl.sort_values('ts')
                time_diffs = df_cl_sorted['ts'].diff().dt.total_seconds()
                negative_diffs = (time_diffs < 0).sum()
                if negative_diffs > 0:
                    issues.append(f"CL: {negative_diffs} timestamp regressions (going backward)")
                    print(f"  [ERROR] Found {negative_diffs} timestamp regressions!")
                else:
                    print(f"  [OK] Timestamps progress forward")
                
                # Check for large gaps
                large_gaps = (time_diffs > 10).sum()  # >10 seconds
                if large_gaps > 0:
                    warnings.append(f"CL: {large_gaps} gaps >10 seconds")
                    print(f"  [WARN] Found {large_gaps} gaps >10 seconds")
                else:
                    print(f"  [OK] No large gaps in timestamps")
                
                # Check price data
                if 'mid' in df_cl.columns:
                    prices = df_cl['mid'].dropna()
                    if len(prices) > 0:
                        # Check for stuck prices
                        price_changes = prices.diff().abs()
                        stuck = (price_changes < 0.01).sum()  # <$0.01 change
                        stuck_pct = stuck / len(price_changes) * 100
                        if stuck_pct > 50:
                            warnings.append(f"CL: {stuck_pct:.1f}% of prices unchanged (<$0.01)")
                            print(f"  [WARN] {stuck_pct:.1f}% of prices unchanged")
                        else:
                            print(f"  [OK] Price changes: {stuck_pct:.1f}% unchanged")
                        
                        # Check for unrealistic prices
                        if asset == 'BTC':
                            reasonable_range = (20000, 150000)
                        elif asset == 'ETH':
                            reasonable_range = (1000, 10000)
                        elif asset == 'SOL':
                            reasonable_range = (50, 500)
                        elif asset == 'XRP':
                            reasonable_range = (0.1, 10)
                        else:
                            reasonable_range = (0, float('inf'))
                        
                        out_of_range = ((prices < reasonable_range[0]) | (prices > reasonable_range[1])).sum()
                        if out_of_range > 0:
                            issues.append(f"CL: {out_of_range} prices out of reasonable range")
                            print(f"  [ERROR] {out_of_range} prices out of range {reasonable_range}")
                        else:
                            print(f"  [OK] All prices in reasonable range")
                
                # Check collected_at vs timestamp
                if 'collected_at' in df_cl.columns:
                    df_cl['collected_ts'] = pd.to_datetime(df_cl['collected_at'], format='mixed', utc=True, errors='coerce')
                    df_cl_valid = df_cl[df_cl['collected_ts'].notna() & df_cl['ts'].notna()]
                    if len(df_cl_valid) > 0:
                        delays = (df_cl_valid['collected_ts'] - df_cl_valid['ts']).dt.total_seconds()
                        avg_delay = delays.mean()
                        if avg_delay < 30 or avg_delay > 120:
                            warnings.append(f"CL: Unusual avg delay {avg_delay:.1f}s (expected 60-70s)")
                            print(f"  [WARN] Average delay: {avg_delay:.1f}s (expected ~65s)")
                        else:
                            print(f"  [OK] Average delay: {avg_delay:.1f}s (expected ~65s)")
            
        except Exception as e:
            issues.append(f"CL: Error loading data: {e}")
            print(f"  [ERROR] Failed to load: {e}")
    else:
        warnings.append(f"CL: File not found")
        print(f"  [WARN] Chainlink file not found")
    
    # Check Polymarket data
    if pm_path.exists():
        print(f"\n[POLYMARKET] Loading {pm_path.name}...")
        try:
            df_pm = pd.read_csv(pm_path, on_bad_lines='skip')
            print(f"  Total rows: {len(df_pm)}")
            
            # Parse timestamps
            if 'timestamp' in df_pm.columns:
                df_pm['ts'] = pd.to_datetime(df_pm['timestamp'], format='mixed', utc=True, errors='coerce')
                df_pm = df_pm[df_pm['ts'].notna()].copy()
                
                # Check for duplicates
                df_pm['ts_second'] = df_pm['ts'].dt.floor('s')
                duplicates = df_pm[df_pm.duplicated(subset=['ts_second'], keep=False)]
                if len(duplicates) > 0:
                    issues.append(f"PM: {len(duplicates)} duplicate timestamps")
                    print(f"  [ERROR] Found {len(duplicates)} duplicate timestamps!")
                else:
                    print(f"  [OK] No duplicate timestamps")
                
                # Check for stuck timestamps
                unique_seconds = df_pm['ts_second'].nunique()
                total_seconds = len(df_pm)
                if unique_seconds < total_seconds * 0.9:
                    warnings.append(f"PM: Only {unique_seconds}/{total_seconds} unique seconds")
                    print(f"  [WARN] Low unique seconds: {unique_seconds}/{total_seconds} ({unique_seconds/total_seconds*100:.1f}%)")
                else:
                    print(f"  [OK] Unique seconds: {unique_seconds}/{total_seconds} ({unique_seconds/total_seconds*100:.1f}%)")
                
                # Check timestamp progression
                df_pm_sorted = df_pm.sort_values('ts')
                time_diffs = df_pm_sorted['ts'].diff().dt.total_seconds()
                negative_diffs = (time_diffs < 0).sum()
                if negative_diffs > 0:
                    issues.append(f"PM: {negative_diffs} timestamp regressions")
                    print(f"  [ERROR] Found {negative_diffs} timestamp regressions!")
                else:
                    print(f"  [OK] Timestamps progress forward")
                
                # Check spreads
                if 'up_best_bid' in df_pm.columns and 'up_best_ask' in df_pm.columns:
                    up_spreads = df_pm['up_best_ask'] - df_pm['up_best_bid']
                    negative_spreads = (up_spreads < 0).sum()
                    if negative_spreads > 0:
                        issues.append(f"PM: {negative_spreads} negative UP spreads")
                        print(f"  [ERROR] Found {negative_spreads} negative UP spreads!")
                    else:
                        print(f"  [OK] No negative UP spreads")
                
                if 'down_best_bid' in df_pm.columns and 'down_best_ask' in df_pm.columns:
                    down_spreads = df_pm['down_best_ask'] - df_pm['down_best_bid']
                    negative_spreads = (down_spreads < 0).sum()
                    if negative_spreads > 0:
                        issues.append(f"PM: {negative_spreads} negative DOWN spreads")
                        print(f"  [ERROR] Found {negative_spreads} negative DOWN spreads!")
                    else:
                        print(f"  [OK] No negative DOWN spreads")
                
                # Check no-arb bounds
                if all(col in df_pm.columns for col in ['up_best_bid', 'down_best_bid', 'up_best_ask', 'down_best_ask']):
                    sum_bids = df_pm['up_best_bid'] + df_pm['down_best_bid']
                    sum_asks = df_pm['up_best_ask'] + df_pm['down_best_ask']
                    arb_violations = ((sum_bids >= 1.0) | (sum_asks <= 1.0)).sum()
                    if arb_violations > len(df_pm) * 0.05:  # >5% violations
                        warnings.append(f"PM: {arb_violations} no-arb violations ({arb_violations/len(df_pm)*100:.1f}%)")
                        print(f"  [WARN] {arb_violations} no-arb violations ({arb_violations/len(df_pm)*100:.1f}%)")
                    else:
                        print(f"  [OK] No-arb violations: {arb_violations} ({arb_violations/len(df_pm)*100:.1f}%)")
            
        except Exception as e:
            issues.append(f"PM: Error loading data: {e}")
            print(f"  [ERROR] Failed to load: {e}")
    else:
        warnings.append(f"PM: File not found")
        print(f"  [WARN] Polymarket file not found")
    
    return issues, warnings

def check_processed_markets(asset: str):
    """Check processed market data for quality issues."""
    print(f"\n{'='*80}")
    print(f"PROCESSED MARKETS CHECK: {asset}")
    print(f"{'='*80}")
    
    markets_dir = Path(STORAGE.markets_dir) / asset
    if not markets_dir.exists():
        print(f"  [WARN] No markets directory found")
        return [], []
    
    issues = []
    warnings = []
    
    market_dirs = [d for d in markets_dir.iterdir() if d.is_dir()]
    print(f"\nFound {len(market_dirs)} markets")
    
    for market_dir in sorted(market_dirs)[:5]:  # Check top 5 markets
        summary_path = market_dir / 'summary.json'
        if not summary_path.exists():
            continue
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        market_id = summary.get('market_id', market_dir.name)
        both_cov = summary.get('both_coverage', 0)
        
        print(f"\n  Market: {market_id} (Both coverage: {both_cov:.1f}%)")
        
        # Load market data
        cl_path = market_dir / 'chainlink.csv'
        pm_path = market_dir / 'polymarket.csv'
        
        if cl_path.exists() and pm_path.exists():
            df_cl = pd.read_csv(cl_path)
            df_pm = pd.read_csv(pm_path)
            
            # Check for time alignment
            if 'seconds' in df_cl.columns and 'seconds' in df_pm.columns:
                cl_seconds = set(df_cl['seconds'].astype(int).unique())
                pm_seconds = set(df_pm['seconds'].astype(int).unique())
                matched = cl_seconds & pm_seconds
                
                if len(matched) < 800:  # <90% of 900
                    warnings.append(f"{market_id}: Only {len(matched)} matched seconds")
                    print(f"    [WARN] Only {len(matched)} matched seconds")
                else:
                    print(f"    [OK] {len(matched)} matched seconds")
            
            # Check CL price continuity
            if 'mid' in df_cl.columns:
                prices = df_cl['mid'].dropna()
                if len(prices) > 0:
                    price_changes = prices.diff().abs()
                    large_jumps = (price_changes > prices.mean() * 0.1).sum()  # >10% jumps
                    if large_jumps > 0:
                        warnings.append(f"{market_id}: {large_jumps} large CL price jumps")
                        print(f"    [WARN] {large_jumps} large price jumps")
                    else:
                        print(f"    [OK] No large price jumps")
            
            # Check PM spread consistency
            if all(col in df_pm.columns for col in ['up_best_bid', 'up_best_ask']):
                spreads = df_pm['up_best_ask'] - df_pm['up_best_bid']
                negative = (spreads < 0).sum()
                if negative > 0:
                    issues.append(f"{market_id}: {negative} negative spreads")
                    print(f"    [ERROR] {negative} negative spreads")
                else:
                    print(f"    [OK] No negative spreads")
    
    return issues, warnings

def main():
    """Run comprehensive data quality checks."""
    print("="*80)
    print("DEEP DATA QUALITY CHECK")
    print("="*80)
    
    # Find assets
    raw_dir = Path(STORAGE.raw_dir)
    assets = []
    if (raw_dir / "chainlink").exists():
        for d in (raw_dir / "chainlink").iterdir():
            if d.is_dir() and d.name in ['BTC', 'ETH', 'SOL', 'XRP']:
                assets.append(d.name)
    
    if not assets:
        print("No assets found to check")
        return
    
    all_issues = []
    all_warnings = []
    
    for asset in assets:
        issues, warnings = check_raw_data(asset)
        all_issues.extend(issues)
        all_warnings.extend(warnings)
        
        issues, warnings = check_processed_markets(asset)
        all_issues.extend(issues)
        all_warnings.extend(warnings)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal Issues: {len(all_issues)}")
    for issue in all_issues:
        print(f"  [ERROR] {issue}")
    
    print(f"\nTotal Warnings: {len(all_warnings)}")
    for warning in all_warnings:
        print(f"  [WARN] {warning}")
    
    if len(all_issues) == 0 and len(all_warnings) == 0:
        print("\n[OK] No data quality issues detected!")
    elif len(all_issues) == 0:
        print("\n[OK] No critical issues, but review warnings")
    else:
        print("\n[ERROR] Critical issues found - review above")

if __name__ == "__main__":
    main()

