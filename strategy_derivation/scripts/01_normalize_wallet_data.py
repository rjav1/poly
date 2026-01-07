#!/usr/bin/env python3
"""
Step 0: Normalize Wallet Data

Filters to 15-minute updown markets and computes canonical time coordinates (t, tau).

Input: polymarket_multi_activity.csv/parquet
Output: wallet_data_normalized.parquet
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "profitable_traders_wallet_data"
OUTPUT_DIR = BASE_DIR / "data"
MARKET_DURATION_SECONDS = 900  # 15 minutes


def parse_start_ts_from_slug(slug: str) -> int:
    """
    Extract start timestamp from slug.
    
    Example: 'btc-updown-15m-1767726000' -> 1767726000
    """
    if pd.isna(slug):
        return None
    
    match = re.search(r'-(\d{10})$', slug)
    if match:
        return int(match.group(1))
    return None


def extract_asset_from_slug(slug: str) -> str:
    """
    Extract asset symbol from slug prefix.
    
    Example: 'btc-updown-15m-1767726000' -> 'BTC'
    """
    if pd.isna(slug):
        return None
    
    match = re.match(r'^([a-z]+)-updown-15m', slug)
    if match:
        return match.group(1).upper()
    return None


def main():
    print("=" * 60)
    print("Step 0: Normalize Wallet Data")
    print("=" * 60)
    
    # Load data (prefer parquet for speed)
    parquet_path = INPUT_DIR / "polymarket_multi_activity.parquet"
    csv_path = INPUT_DIR / "polymarket_multi_activity.csv"
    
    if parquet_path.exists():
        print(f"\nLoading from parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    else:
        print(f"\nLoading from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
    
    print(f"Total rows loaded: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Handles: {df['_handle'].unique().tolist()}")
    
    # Step 1: Filter to 15-minute updown markets
    print("\n--- Step 1: Filter to 15m updown markets ---")
    df_15m = df[df['slug'].str.contains('updown-15m', na=False)].copy()
    print(f"Rows after filter: {len(df_15m):,} ({len(df_15m)/len(df)*100:.1f}%)")
    
    # Step 2: Parse start_ts from slug
    print("\n--- Step 2: Parse start_ts from slug ---")
    df_15m['start_ts'] = df_15m['slug'].apply(parse_start_ts_from_slug)
    
    # Check for parsing failures
    null_start_ts = df_15m['start_ts'].isna().sum()
    if null_start_ts > 0:
        print(f"  WARNING: {null_start_ts} rows could not parse start_ts")
        print(f"  Sample failed slugs: {df_15m[df_15m['start_ts'].isna()]['slug'].head().tolist()}")
    
    # Remove rows without valid start_ts
    df_15m = df_15m[df_15m['start_ts'].notna()].copy()
    df_15m['start_ts'] = df_15m['start_ts'].astype(int)
    print(f"Rows with valid start_ts: {len(df_15m):,}")
    
    # Step 3: Compute t and tau
    print("\n--- Step 3: Compute t and tau ---")
    # timestamp column is Unix epoch seconds
    df_15m['t'] = df_15m['timestamp'] - df_15m['start_ts']
    df_15m['tau'] = MARKET_DURATION_SECONDS - df_15m['t']
    
    print(f"  t range: {df_15m['t'].min():.0f} to {df_15m['t'].max():.0f}")
    print(f"  tau range: {df_15m['tau'].min():.0f} to {df_15m['tau'].max():.0f}")
    
    # Step 4: Extract asset from slug
    print("\n--- Step 4: Extract asset from slug ---")
    df_15m['asset'] = df_15m['slug'].apply(extract_asset_from_slug)
    print(f"Assets: {df_15m['asset'].value_counts().to_dict()}")
    
    # Step 5: Filter to within-window trades (0 <= t < 900)
    print("\n--- Step 5: Filter to within-window trades ---")
    before_filter = len(df_15m)
    df_15m = df_15m[(df_15m['t'] >= 0) & (df_15m['t'] < MARKET_DURATION_SECONDS)].copy()
    after_filter = len(df_15m)
    print(f"Rows before: {before_filter:,}")
    print(f"Rows after: {after_filter:,}")
    print(f"Excluded: {before_filter - after_filter:,} ({(before_filter - after_filter)/before_filter*100:.1f}%)")
    
    # Summary statistics
    print("\n--- Summary Statistics ---")
    print(f"\nBy Handle:")
    for handle in df_15m['_handle'].unique():
        handle_df = df_15m[df_15m['_handle'] == handle]
        print(f"  {handle}: {len(handle_df):,} trades")
    
    print(f"\nBy Asset:")
    for asset in df_15m['asset'].unique():
        asset_df = df_15m[df_15m['asset'] == asset]
        print(f"  {asset}: {len(asset_df):,} trades")
    
    print(f"\nBy Type:")
    print(df_15m['type'].value_counts().to_dict())
    
    print(f"\nBy Side:")
    print(df_15m['side'].value_counts().to_dict())
    
    # Timing distribution
    print(f"\nTiming Distribution (tau buckets):")
    tau_buckets = pd.cut(df_15m['tau'], bins=[0, 60, 120, 300, 600, 900], 
                         labels=['0-60s', '60-120s', '120-300s', '300-600s', '600-900s'])
    print(tau_buckets.value_counts().sort_index().to_dict())
    
    # Step 6: Save output
    print("\n--- Step 6: Save output ---")
    output_path = OUTPUT_DIR / "wallet_data_normalized.parquet"
    df_15m.to_parquet(output_path, index=False)
    print(f"Saved to: {output_path}")
    print(f"Output shape: {df_15m.shape}")
    
    # Also save a small CSV sample for inspection
    sample_path = OUTPUT_DIR / "wallet_data_normalized_sample.csv"
    df_15m.head(1000).to_csv(sample_path, index=False)
    print(f"Sample saved to: {sample_path}")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    
    return df_15m


if __name__ == "__main__":
    main()

