#!/usr/bin/env python3
"""Comprehensive dataset quality analysis to detect biases and collection issues."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import STORAGE


def analyze_dataset_quality():
    """Comprehensive quality analysis of the built dataset."""
    
    dataset_path = Path(STORAGE.research_dir) / "canonical_dataset_all_assets.parquet"
    market_info_path = Path(STORAGE.research_dir) / "market_info_all_assets.json"
    
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return
    
    print("=" * 80)
    print("DATASET QUALITY ANALYSIS")
    print("=" * 80)
    print()
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_parquet(dataset_path)
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print()
    
    # Load market info
    with open(market_info_path, 'r') as f:
        market_infos = json.load(f)
    
    print(f"Markets: {len(market_infos)}")
    print()
    
    # ============================================================================
    # 1. TIMESTAMP ANALYSIS
    # ============================================================================
    print("=" * 80)
    print("1. TIMESTAMP ANALYSIS")
    print("=" * 80)
    
    if 't' in df.columns:
        print(f"\nTime column 't' (seconds from market start):")
        print(f"  Min: {df['t'].min()}, Max: {df['t'].max()}")
        print(f"  Expected range: 0 to 899 (900 seconds = 15 minutes)")
        
        # Check for gaps
        expected_seconds = set(range(900))
        actual_seconds = set(df['t'].astype(int).unique())
        missing_seconds = expected_seconds - actual_seconds
        print(f"  Missing seconds: {len(missing_seconds)}")
        if len(missing_seconds) > 0:
            print(f"  Missing ranges: {sorted(list(missing_seconds))[:20]}..." if len(missing_seconds) > 20 else f"  Missing: {sorted(list(missing_seconds))}")
        
        # Check for duplicates
        duplicates = df[df.duplicated(subset=['t'], keep=False)]
        if len(duplicates) > 0:
            print(f"  [WARN] Found {len(duplicates)} duplicate timestamps!")
            print(f"  Duplicate 't' values: {sorted(duplicates['t'].unique())}")
        else:
            print(f"  [OK] No duplicate timestamps")
        
        # Check time progression
        df_sorted = df.sort_values('t')
        time_diffs = df_sorted['t'].diff().dropna()
        if (time_diffs != 1).any():
            gaps = time_diffs[time_diffs != 1]
            print(f"  [WARN] Found {len(gaps)} time gaps (expected diff=1)")
            print(f"  Gap sizes: {sorted(gaps.unique())}")
        else:
            print(f"  [OK] Continuous time progression (no gaps)")
    
    # ============================================================================
    # 2. CHAINLINK DATA QUALITY
    # ============================================================================
    print("\n" + "=" * 80)
    print("2. CHAINLINK DATA QUALITY")
    print("=" * 80)
    
    cl_price_col = 'cl_mid'
    cl_observed_col = 'cl_ffill'  # 0 = observed, 1 = forward-filled
    
    if cl_price_col in df.columns:
        cl_prices = df[cl_price_col].dropna()
        print(f"\nCL Price Statistics:")
        print(f"  Total rows: {len(df)}")
        print(f"  Non-null CL prices: {len(cl_prices)} ({len(cl_prices)/len(df)*100:.1f}%)")
        print(f"  Null/NaN CL prices: {df[cl_price_col].isna().sum()} ({df[cl_price_col].isna().sum()/len(df)*100:.1f}%)")
        
        if len(cl_prices) > 0:
            print(f"  Min price: ${cl_prices.min():.2f}")
            print(f"  Max price: ${cl_prices.max():.2f}")
            print(f"  Mean price: ${cl_prices.mean():.2f}")
            print(f"  Std dev: ${cl_prices.std():.2f}")
            
            # Check for stuck prices
            price_changes = cl_prices.diff().abs()
            stuck_threshold = 0.01  # $0.01
            stuck_periods = (price_changes < stuck_threshold).sum()
            print(f"  Price changes < ${stuck_threshold}: {stuck_periods} ({stuck_periods/len(cl_prices)*100:.1f}%)")
            
            # Check for large jumps
            large_jumps = (price_changes > cl_prices.mean() * 0.05).sum()  # >5% of mean
            print(f"  Large price jumps (>5% of mean): {large_jumps}")
            if large_jumps > 0:
                jump_indices = price_changes[price_changes > cl_prices.mean() * 0.05].index
                print(f"  [WARN] Large jumps at indices: {list(jump_indices[:10])}")
            
            # Check price range reasonableness
            price_range = cl_prices.max() - cl_prices.min()
            price_range_pct = (price_range / cl_prices.mean()) * 100
            print(f"  Price range: ${price_range:.2f} ({price_range_pct:.2f}% of mean)")
            
            if price_range_pct > 20:
                print(f"  [WARN] Large price range (>20% of mean) - possible data issues")
            else:
                print(f"  [OK] Price range is reasonable")
        
        # Check observed vs forward-filled
        if cl_observed_col in df.columns:
            observed_count = (df[cl_observed_col] == 0).sum()  # 0 = observed
            ffill_count = (df[cl_observed_col] == 1).sum()  # 1 = forward-filled
            print(f"\nCL Data Source:")
            print(f"  Observed (real data): {observed_count} ({observed_count/len(df)*100:.1f}%)")
            print(f"  Forward-filled: {ffill_count} ({ffill_count/len(df)*100:.1f}%)")
            
            if ffill_count / len(df) > 0.1:
                print(f"  [WARN] High forward-fill percentage (>10%)")
            else:
                print(f"  [OK] Low forward-fill percentage")
    
    # ============================================================================
    # 3. POLYMARKET DATA QUALITY
    # ============================================================================
    print("\n" + "=" * 80)
    print("3. POLYMARKET DATA QUALITY")
    print("=" * 80)
    
    pm_cols = ['pm_up_best_bid', 'pm_up_best_ask', 'pm_down_best_bid', 'pm_down_best_ask']
    pm_observed_col = 'pm_ffill'  # 0 = observed, 1 = forward-filled
    
    for col in pm_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"\n{col}:")
            print(f"  Non-null: {non_null} ({non_null/len(df)*100:.1f}%)")
            if non_null > 0:
                print(f"  Min: ${df[col].min():.4f}, Max: ${df[col].max():.4f}, Mean: ${df[col].mean():.4f}")
    
    # Check spreads
    if all(col in df.columns for col in ['pm_up_best_bid', 'pm_up_best_ask']):
        up_spreads = df['pm_up_best_ask'] - df['pm_up_best_bid']
        up_spreads_valid = up_spreads.dropna()
        if len(up_spreads_valid) > 0:
            print(f"\nUP Spread Statistics:")
            print(f"  Mean: {up_spreads_valid.mean():.4f}")
            print(f"  Median: {up_spreads_valid.median():.4f}")
            print(f"  Min: {up_spreads_valid.min():.4f}")
            print(f"  Max: {up_spreads_valid.max():.4f}")
            
            negative_spreads = (up_spreads_valid < 0).sum()
            if negative_spreads > 0:
                print(f"  [ERROR] Found {negative_spreads} negative spreads (bid > ask)!")
            else:
                print(f"  [OK] No negative spreads")
    
    if all(col in df.columns for col in ['pm_down_best_bid', 'pm_down_best_ask']):
        down_spreads = df['pm_down_best_ask'] - df['pm_down_best_bid']
        down_spreads_valid = down_spreads.dropna()
        if len(down_spreads_valid) > 0:
            print(f"\nDOWN Spread Statistics:")
            print(f"  Mean: {down_spreads_valid.mean():.4f}")
            print(f"  Median: {down_spreads_valid.median():.4f}")
            print(f"  Min: {down_spreads_valid.min():.4f}")
            print(f"  Max: {down_spreads_valid.max():.4f}")
            
            negative_spreads = (down_spreads_valid < 0).sum()
            if negative_spreads > 0:
                print(f"  [ERROR] Found {negative_spreads} negative spreads (bid > ask)!")
            else:
                print(f"  [OK] No negative spreads")
    
    # Check observed vs forward-filled
    if pm_observed_col in df.columns:
        observed_count = (df[pm_observed_col] == 0).sum()  # 0 = observed
        ffill_count = (df[pm_observed_col] == 1).sum()  # 1 = forward-filled
        print(f"\nPM Data Source:")
        print(f"  Observed (real data): {observed_count} ({observed_count/len(df)*100:.1f}%)")
        print(f"  Forward-filled: {ffill_count} ({ffill_count/len(df)*100:.1f}%)")
        
        if ffill_count / len(df) > 0.1:
            print(f"  [WARN] High forward-fill percentage (>10%)")
        else:
            print(f"  [OK] Low forward-fill percentage")
    
    # ============================================================================
    # 4. COVERAGE ANALYSIS
    # ============================================================================
    print("\n" + "=" * 80)
    print("4. COVERAGE ANALYSIS")
    print("=" * 80)
    
    for market_info in market_infos:
        market_id = market_info.get('market_id', 'unknown')
        print(f"\nMarket: {market_id}")
        print(f"  CL Coverage: {market_info.get('cl_coverage_pct', 0):.1f}%")
        print(f"  PM Coverage: {market_info.get('pm_coverage_pct', 0):.1f}%")
        print(f"  Both Coverage: {market_info.get('both_coverage_pct', 0):.1f}%")
        
        # Check if coverage is uniform across time
        if 't' in df.columns and cl_observed_col in df.columns:
            df_market = df[df.get('market_id', '') == market_id] if 'market_id' in df.columns else df
            
            # Split into 3 time periods
            period1 = df_market[df_market['t'] < 300]  # First 5 minutes
            period2 = df_market[(df_market['t'] >= 300) & (df_market['t'] < 600)]  # Middle 5 minutes
            period3 = df_market[df_market['t'] >= 600]  # Last 5 minutes
            
            for period_name, period_df in [("First 5min", period1), ("Middle 5min", period2), ("Last 5min", period3)]:
                if len(period_df) > 0:
                    cl_obs = (period_df[cl_observed_col] == 0).sum() if cl_observed_col in period_df.columns else 0  # 0 = observed
                    pm_obs = (period_df[pm_observed_col] == 0).sum() if pm_observed_col in period_df.columns else 0  # 0 = observed
                    print(f"    {period_name}: CL={cl_obs}/{len(period_df)} ({cl_obs/len(period_df)*100:.1f}%), PM={pm_obs}/{len(period_df)} ({pm_obs/len(period_df)*100:.1f}%)")
    
    # ============================================================================
    # 5. DATA CONSISTENCY CHECKS
    # ============================================================================
    print("\n" + "=" * 80)
    print("5. DATA CONSISTENCY CHECKS")
    print("=" * 80)
    
    # Check no-arb bounds
    if all(col in df.columns for col in ['pm_up_best_bid', 'pm_up_best_ask', 'pm_down_best_bid', 'pm_down_best_ask']):
        sum_bids = df['pm_up_best_bid'] + df['pm_down_best_bid']
        sum_asks = df['pm_up_best_ask'] + df['pm_down_best_ask']
        
        sum_bids_valid = sum_bids.dropna()
        sum_asks_valid = sum_asks.dropna()
        
        print(f"\nNo-Arb Bounds:")
        print(f"  Sum of bids (should be < 1): mean={sum_bids_valid.mean():.4f}, min={sum_bids_valid.min():.4f}, max={sum_bids_valid.max():.4f}")
        print(f"  Sum of asks (should be > 1): mean={sum_asks_valid.mean():.4f}, min={sum_asks_valid.min():.4f}, max={sum_asks_valid.max():.4f}")
        
        violations_bids = (sum_bids_valid >= 1.0).sum()
        violations_asks = (sum_asks_valid <= 1.0).sum()
        
        if violations_bids > 0:
            print(f"  [WARN] {violations_bids} rows with sum_bids >= 1.0 (arbitrage opportunity)")
        if violations_asks > 0:
            print(f"  [WARN] {violations_asks} rows with sum_asks <= 1.0 (arbitrage opportunity)")
        if violations_bids == 0 and violations_asks == 0:
            print(f"  [OK] No-arb bounds respected")
    
    # Check strike consistency
    if 'K' in df.columns and 'strike_price' in df.columns:
        strike_diff = (df['K'] - df['strike_price']).abs()
        print(f"\nStrike Consistency:")
        print(f"  Mean difference: ${strike_diff.mean():.2f}")
        print(f"  Max difference: ${strike_diff.max():.2f}")
        
        if strike_diff.max() > 50:
            print(f"  [WARN] Large strike price differences detected")
        else:
            print(f"  [OK] Strike prices consistent")
    
    # ============================================================================
    # 6. TEMPORAL PATTERNS (BIAS DETECTION)
    # ============================================================================
    print("\n" + "=" * 80)
    print("6. TEMPORAL PATTERNS (BIAS DETECTION)")
    print("=" * 80)
    
    if 't' in df.columns:
        # Check if data collection was uniform over time
        time_bins = np.linspace(0, 900, 10)  # 10 bins of 90 seconds each
        df['time_bin'] = pd.cut(df['t'], bins=time_bins, labels=False)
        
        if cl_observed_col in df.columns:
            cl_by_bin = df.groupby('time_bin')[cl_observed_col].apply(lambda x: (x == 0).sum())  # 0 = observed
            print(f"\nCL Data Collection by Time Bin:")
            for bin_idx, count in cl_by_bin.items():
                bin_start = int(bin_idx * 90) if not pd.isna(bin_idx) else 0
                bin_end = int((bin_idx + 1) * 90) if not pd.isna(bin_idx) else 90
                print(f"  {bin_start}-{bin_end}s: {count} observed points")
            
            # Check for systematic bias
            if len(cl_by_bin) > 1:
                cl_std = cl_by_bin.std()
                cl_mean = cl_by_bin.mean()
                cv = cl_std / cl_mean if cl_mean > 0 else 0
                print(f"  Coefficient of variation: {cv:.3f}")
                if cv > 0.3:
                    print(f"  [WARN] High variation in collection rate across time (possible bias)")
                else:
                    print(f"  [OK] Uniform collection rate across time")
        
        if pm_observed_col in df.columns:
            pm_by_bin = df.groupby('time_bin')[pm_observed_col].apply(lambda x: (x == 0).sum())  # 0 = observed
            print(f"\nPM Data Collection by Time Bin:")
            for bin_idx, count in pm_by_bin.items():
                bin_start = int(bin_idx * 90) if not pd.isna(bin_idx) else 0
                bin_end = int((bin_idx + 1) * 90) if not pd.isna(bin_idx) else 90
                print(f"  {bin_start}-{bin_end}s: {count} observed points")
            
            # Check for systematic bias
            if len(pm_by_bin) > 1:
                pm_std = pm_by_bin.std()
                pm_mean = pm_by_bin.mean()
                cv = pm_std / pm_mean if pm_mean > 0 else 0
                print(f"  Coefficient of variation: {cv:.3f}")
                if cv > 0.3:
                    print(f"  [WARN] High variation in collection rate across time (possible bias)")
                else:
                    print(f"  [OK] Uniform collection rate across time")
    
    # ============================================================================
    # 7. SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("7. SUMMARY")
    print("=" * 80)
    
    issues = []
    warnings = []
    
    # Collect issues
    if 't' in df.columns:
        expected_seconds = set(range(900))
        actual_seconds = set(df['t'].astype(int).unique())
        missing = len(expected_seconds - actual_seconds)
        if missing > 0:
            warnings.append(f"Missing {missing} seconds in time series")
        
        duplicates = df[df.duplicated(subset=['t'], keep=False)]
        if len(duplicates) > 0:
            issues.append(f"Found {len(duplicates)} duplicate timestamps")
    
    if cl_price_col in df.columns:
        cl_prices = df[cl_price_col].dropna()
        if len(cl_prices) > 0:
            price_range_pct = ((cl_prices.max() - cl_prices.min()) / cl_prices.mean()) * 100
            if price_range_pct > 20:
                warnings.append(f"CL price range is {price_range_pct:.1f}% of mean (unusually large)")
    
    if cl_observed_col in df.columns:
        ffill_pct = ((df[cl_observed_col] == 1).sum() / len(df)) * 100  # 1 = forward-filled
        if ffill_pct > 10:
            warnings.append(f"CL forward-fill is {ffill_pct:.1f}% (high)")
    
    if pm_observed_col in df.columns:
        ffill_pct = ((df[pm_observed_col] == 1).sum() / len(df)) * 100  # 1 = forward-filled
        if ffill_pct > 10:
            warnings.append(f"PM forward-fill is {ffill_pct:.1f}% (high)")
    
    print(f"\nIssues Found: {len(issues)}")
    for issue in issues:
        print(f"  [ERROR] {issue}")
    
    print(f"\nWarnings: {len(warnings)}")
    for warning in warnings:
        print(f"  [WARN] {warning}")
    
    if len(issues) == 0 and len(warnings) == 0:
        print("\n[OK] No significant data quality issues detected!")
    elif len(issues) == 0:
        print("\n[OK] No critical issues, but review warnings above")
    else:
        print("\n[ERROR] Critical issues detected - review above")
    
    print()


if __name__ == "__main__":
    analyze_dataset_quality()

