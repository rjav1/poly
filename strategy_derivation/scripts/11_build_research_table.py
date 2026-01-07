#!/usr/bin/env python3
"""
Phase 1: Build Research Table (State → Action Dataset)

Creates a canonical research table where each row = (wallet, market, second, action).
Joins wallet trades to full market state features at trade time.

Input:
- wallet_data_normalized.parquet
- canonical_dataset_all_assets.parquet

Output:
- research_table.parquet
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESEARCH_DIR = BASE_DIR.parent / "data_v2" / "research"
MARKET_DURATION_SECONDS = 900


def load_wallet_data() -> pd.DataFrame:
    """Load normalized wallet data."""
    path = DATA_DIR / "wallet_data_normalized.parquet"
    print(f"Loading wallet data from: {path}")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Handles: {df['_handle'].unique().tolist()}")
    return df


def load_market_data() -> Tuple[pd.DataFrame, Dict]:
    """Load canonical market dataset and market info."""
    # Load canonical dataset
    parquet_path = RESEARCH_DIR / "canonical_dataset_all_assets.parquet"
    print(f"\nLoading market data from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df):,} rows")
    print(f"  Markets: {df['market_id'].nunique()}")
    
    # Load market info
    info_path = RESEARCH_DIR / "market_info_all_assets.json"
    with open(info_path, 'r') as f:
        market_info = json.load(f)
    print(f"  Market info loaded for {len(market_info)} markets")
    
    return df, market_info


def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additional microstructure features per market.
    
    Features:
    - Short-horizon returns (momentum)
    - Spread dynamics
    - Quote staleness
    - Orderbook imbalance
    - Volatility proxies
    - Underround/overround magnitude
    """
    print("\nComputing microstructure features...")
    df = df.copy()
    
    # Ensure sorted by market and time
    df = df.sort_values(['market_id', 't']).reset_index(drop=True)
    
    # -------------------------------------------------------------------------
    # Short-horizon returns (momentum)
    # -------------------------------------------------------------------------
    for k in [1, 5, 10, 30]:
        # PM momentum (using UP mid as proxy)
        if 'pm_up_mid' in df.columns:
            df[f'pm_momentum_{k}s'] = df.groupby('market_id')['pm_up_mid'].transform(
                lambda x: x - x.shift(k)
            )
        
        # CL momentum
        if 'cl_mid' in df.columns:
            df[f'cl_momentum_{k}s'] = df.groupby('market_id')['cl_mid'].transform(
                lambda x: x - x.shift(k)
            )
    
    # -------------------------------------------------------------------------
    # Spread dynamics
    # -------------------------------------------------------------------------
    if 'pm_up_spread' in df.columns:
        df['pm_up_spread_change'] = df.groupby('market_id')['pm_up_spread'].transform(
            lambda x: x - x.shift(1)
        )
    if 'pm_down_spread' in df.columns:
        df['pm_down_spread_change'] = df.groupby('market_id')['pm_down_spread'].transform(
            lambda x: x - x.shift(1)
        )
    
    # Average spread
    if 'pm_up_spread' in df.columns and 'pm_down_spread' in df.columns:
        df['avg_spread'] = (df['pm_up_spread'] + df['pm_down_spread']) / 2
    
    # -------------------------------------------------------------------------
    # Quote staleness (seconds since last quote change)
    # -------------------------------------------------------------------------
    # Detect quote changes in PM mid prices
    if 'pm_up_mid' in df.columns:
        df['pm_up_changed'] = df.groupby('market_id')['pm_up_mid'].transform(
            lambda x: (x != x.shift(1)).astype(int)
        )
        # Cumulative sum of changes, then rank within each group
        df['pm_up_staleness'] = df.groupby('market_id')['pm_up_changed'].transform(
            lambda x: x.groupby((x == 1).cumsum()).cumcount()
        )
    
    if 'pm_down_mid' in df.columns:
        df['pm_down_changed'] = df.groupby('market_id')['pm_down_mid'].transform(
            lambda x: (x != x.shift(1)).astype(int)
        )
        df['pm_down_staleness'] = df.groupby('market_id')['pm_down_changed'].transform(
            lambda x: x.groupby((x == 1).cumsum()).cumcount()
        )
    
    # Average staleness
    if 'pm_up_staleness' in df.columns and 'pm_down_staleness' in df.columns:
        df['quote_staleness'] = (df['pm_up_staleness'] + df['pm_down_staleness']) / 2
    
    # -------------------------------------------------------------------------
    # Orderbook imbalance
    # -------------------------------------------------------------------------
    if 'sum_bids' in df.columns and 'sum_asks' in df.columns:
        # Imbalance: (bids - asks) / (bids + asks)
        # Positive = more buying pressure (sum_bids > sum_asks is rare)
        denom = df['sum_bids'] + df['sum_asks']
        df['orderbook_imbalance'] = np.where(
            denom > 0,
            (df['sum_bids'] - df['sum_asks']) / denom,
            0
        )
    
    # -------------------------------------------------------------------------
    # Volatility proxies (rolling std of returns)
    # -------------------------------------------------------------------------
    if 'cl_mid' in df.columns:
        # Compute returns first
        df['cl_return'] = df.groupby('market_id')['cl_mid'].transform(
            lambda x: x.pct_change()
        )
        # Rolling std (10-second window)
        df['realized_vol_10s'] = df.groupby('market_id')['cl_return'].transform(
            lambda x: x.rolling(window=10, min_periods=2).std()
        )
        # Rolling std (30-second window)
        df['realized_vol_30s'] = df.groupby('market_id')['cl_return'].transform(
            lambda x: x.rolling(window=30, min_periods=5).std()
        )
    
    # -------------------------------------------------------------------------
    # Underround/overround magnitude
    # -------------------------------------------------------------------------
    if 'sum_asks' in df.columns:
        df['underround'] = 1 - df['sum_asks']  # Positive when sum_asks < 1
        df['underround_positive'] = df['underround'].clip(lower=0)
    
    if 'sum_bids' in df.columns:
        df['overround'] = df['sum_bids'] - 1  # Positive when sum_bids > 1 (rare)
        df['overround_positive'] = df['overround'].clip(lower=0)
    
    # -------------------------------------------------------------------------
    # Capacity constraints (if size data available)
    # -------------------------------------------------------------------------
    if 'pm_up_best_ask_size' in df.columns and 'pm_down_best_ask_size' in df.columns:
        df['achievable_capacity'] = np.minimum(
            df['pm_up_best_ask_size'].fillna(0),
            df['pm_down_best_ask_size'].fillna(0)
        )
    else:
        df['achievable_capacity'] = np.nan
    
    # -------------------------------------------------------------------------
    # Edge-weighted capacity (underround * capacity)
    # -------------------------------------------------------------------------
    if 'underround_positive' in df.columns and 'achievable_capacity' in df.columns:
        df['edge_weighted_capacity'] = df['underround_positive'] * df['achievable_capacity']
    
    # -------------------------------------------------------------------------
    # Temporal features
    # -------------------------------------------------------------------------
    # tau buckets
    df['tau_bucket'] = pd.cut(
        df['tau'],
        bins=[0, 60, 120, 300, 600, 900],
        labels=['0-60', '60-120', '120-300', '300-600', '600-900'],
        include_lowest=True
    )
    
    df['is_late_window'] = (df['tau'] <= 300).astype(int)
    df['is_very_late'] = (df['tau'] <= 60).astype(int)
    df['is_early_window'] = (df['tau'] > 600).astype(int)
    
    # Clean up temporary columns
    drop_cols = ['pm_up_changed', 'pm_down_changed', 'cl_return']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    print(f"  Added features. New shape: {df.shape}")
    return df


def parse_start_ts_from_slug(slug: str) -> Optional[int]:
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


def build_market_lookup(market_df: pd.DataFrame) -> Dict[Tuple[int, int], pd.Series]:
    """
    Build lookup dictionary for fast trade-to-quote matching.
    Key: (market_start_ts, t) -> quote row
    """
    print("\nBuilding market state lookup...")
    
    # Compute market_start_ts from market_start column
    market_df = market_df.copy()
    market_df['market_start_ts'] = market_df['market_start'].apply(
        lambda x: int(pd.Timestamp(x).timestamp()) if pd.notna(x) else None
    )
    
    # Build lookup
    lookup = {}
    for _, row in market_df.iterrows():
        if pd.notna(row['market_start_ts']):
            key = (int(row['market_start_ts']), int(row['t']))
            lookup[key] = row
    
    print(f"  Built lookup with {len(lookup):,} entries")
    return lookup


def join_trades_to_market_state(
    wallet_df: pd.DataFrame,
    market_lookup: Dict[Tuple[int, int], pd.Series]
) -> pd.DataFrame:
    """
    Join each wallet trade to market state at trade time.
    """
    print("\nJoining trades to market state...")
    
    # Filter to TRADE type only
    trades = wallet_df[wallet_df['type'] == 'TRADE'].copy()
    print(f"  Trades to match: {len(trades):,}")
    
    # Parse start_ts from slug (should already be in normalized data)
    if 'start_ts' not in trades.columns:
        trades['start_ts'] = trades['slug'].apply(parse_start_ts_from_slug)
    
    # Market state columns to extract
    market_state_cols = [
        # Price columns
        'cl_mid', 'pm_up_mid', 'pm_down_mid',
        'pm_up_best_bid', 'pm_up_best_ask', 'pm_down_best_bid', 'pm_down_best_ask',
        # Spread columns
        'pm_up_spread', 'pm_down_spread', 'avg_spread',
        # No-arb columns
        'sum_bids', 'sum_asks', 'underround', 'overround',
        'underround_positive', 'overround_positive',
        # Delta columns
        'delta', 'delta_bps',
        # Momentum columns
        'pm_momentum_1s', 'pm_momentum_5s', 'pm_momentum_10s', 'pm_momentum_30s',
        'cl_momentum_1s', 'cl_momentum_5s', 'cl_momentum_10s', 'cl_momentum_30s',
        # Spread dynamics
        'pm_up_spread_change', 'pm_down_spread_change',
        # Staleness
        'quote_staleness', 'pm_up_staleness', 'pm_down_staleness',
        # Imbalance
        'orderbook_imbalance',
        # Volatility
        'realized_vol_10s', 'realized_vol_30s',
        # Capacity
        'achievable_capacity', 'edge_weighted_capacity',
        # Temporal
        'tau_bucket', 'is_late_window', 'is_very_late', 'is_early_window',
        # Market identifiers
        'K', 'settlement', 'Y',
    ]
    
    # Match trades to market state
    matched_rows = []
    skipped_no_start_ts = 0
    skipped_no_match = 0
    
    for idx, trade in trades.iterrows():
        start_ts = trade.get('start_ts')
        if pd.isna(start_ts):
            skipped_no_start_ts += 1
            continue
        
        t = int(trade['t'])
        key = (int(start_ts), t)
        
        if key not in market_lookup:
            skipped_no_match += 1
            continue
        
        market_state = market_lookup[key]
        
        # Build matched row
        row = {
            # Wallet/trade identifiers
            'wallet': trade['_handle'],
            'trade_timestamp': trade.get('timestamp', trade.get('timestamp_utc')),
            'market_id': market_state.get('market_id', f"{start_ts}"),
            'asset': trade.get('asset', market_state.get('asset')),
            't': t,
            'tau': trade['tau'],
            # Trade details
            'action_type': trade['type'],
            'side': trade['side'],
            'outcome_token': trade['outcome'],
            'price': trade['price'],
            'size': trade['size'],
            'notional': trade['price'] * trade['size'],
            # Slug for reference
            'slug': trade['slug'],
            'conditionId': trade.get('conditionId'),
        }
        
        # Add market state columns
        for col in market_state_cols:
            if col in market_state.index:
                row[f'mkt_{col}'] = market_state[col]
        
        matched_rows.append(row)
    
    print(f"  Matched: {len(matched_rows):,}")
    print(f"  Skipped (no start_ts): {skipped_no_start_ts:,}")
    print(f"  Skipped (no market match): {skipped_no_match:,}")
    
    # Convert to DataFrame
    result_df = pd.DataFrame(matched_rows)
    
    # Add direction labels based on within-second grouping
    result_df = add_direction_labels(result_df)
    
    return result_df


def add_direction_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add direction labels: UP, DOWN, or BOTH (for paired trades in same second).
    """
    print("\nAdding direction labels...")
    df = df.copy()
    
    # Group by (wallet, market_id, t) to detect paired trades
    df['direction'] = df['outcome_token']  # Default to outcome token
    
    # Identify seconds where wallet traded both sides
    for (wallet, market_id, t), group in df.groupby(['wallet', 'market_id', 't']):
        outcomes = set(group['outcome_token'].unique())
        if 'Up' in outcomes and 'Down' in outcomes:
            # Both sides traded in this second - mark as BOTH
            df.loc[group.index, 'direction'] = 'BOTH'
    
    # Summary
    direction_counts = df['direction'].value_counts()
    print(f"  Direction distribution:")
    for d, c in direction_counts.items():
        print(f"    {d}: {c:,}")
    
    return df


def compute_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics for the research table."""
    stats = {
        'total_rows': len(df),
        'unique_wallets': df['wallet'].nunique(),
        'unique_markets': df['market_id'].nunique(),
        'wallets': df['wallet'].value_counts().to_dict(),
        'direction_distribution': df['direction'].value_counts().to_dict(),
        'side_distribution': df['side'].value_counts().to_dict(),
        'asset_distribution': df['asset'].value_counts().to_dict() if 'asset' in df.columns else {},
        'tau_bucket_distribution': df['mkt_tau_bucket'].value_counts().to_dict() if 'mkt_tau_bucket' in df.columns else {},
    }
    return stats


def main():
    print("=" * 70)
    print("Phase 1: Build Research Table (State -> Action Dataset)")
    print("=" * 70)
    
    # Step 1: Load wallet data
    wallet_df = load_wallet_data()
    
    # Step 2: Load market data
    market_df, market_info = load_market_data()
    
    # Step 3: Compute microstructure features
    market_df = compute_microstructure_features(market_df)
    
    # Step 4: Build market lookup
    market_lookup = build_market_lookup(market_df)
    
    # Step 5: Join trades to market state
    research_table = join_trades_to_market_state(wallet_df, market_lookup)
    
    if len(research_table) == 0:
        print("\nWARNING: No trades matched to market state!")
        print("This may be because wallet data and market data are from different markets.")
        print("Proceeding with research table generation using wallet data only...")
        
        # Create research table from wallet data without market state
        trades = wallet_df[wallet_df['type'] == 'TRADE'].copy()
        research_table = pd.DataFrame({
            'wallet': trades['_handle'],
            'trade_timestamp': trades.get('timestamp', trades.get('timestamp_utc')),
            'market_id': trades['slug'],
            'asset': trades.get('asset'),
            't': trades['t'],
            'tau': trades['tau'],
            'action_type': trades['type'],
            'side': trades['side'],
            'outcome_token': trades['outcome'],
            'price': trades['price'],
            'size': trades['size'],
            'notional': trades['price'] * trades['size'],
            'slug': trades['slug'],
            'conditionId': trades.get('conditionId'),
        })
        research_table = add_direction_labels(research_table)
        
        # Add tau bucket manually
        research_table['mkt_tau_bucket'] = pd.cut(
            research_table['tau'],
            bins=[0, 60, 120, 300, 600, 900],
            labels=['0-60', '60-120', '120-300', '300-600', '600-900'],
            include_lowest=True
        )
    
    # Step 6: Compute summary stats
    stats = compute_summary_stats(research_table)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total rows: {stats['total_rows']:,}")
    print(f"Unique wallets: {stats['unique_wallets']}")
    print(f"Unique markets: {stats['unique_markets']}")
    print(f"\nPer-wallet counts:")
    for wallet, count in stats['wallets'].items():
        print(f"  {wallet}: {count:,}")
    print(f"\nDirection distribution:")
    for d, c in stats['direction_distribution'].items():
        print(f"  {d}: {c:,}")
    
    # Step 7: Save outputs
    print("\n" + "=" * 70)
    print("Saving outputs...")
    
    # Save research table
    output_path = DATA_DIR / "research_table.parquet"
    research_table.to_parquet(output_path, index=False)
    print(f"  Research table saved to: {output_path}")
    print(f"  Shape: {research_table.shape}")
    
    # Save CSV sample
    sample_path = DATA_DIR / "research_table_sample.csv"
    research_table.head(1000).to_csv(sample_path, index=False)
    print(f"  Sample saved to: {sample_path}")
    
    # Save stats
    stats_path = RESULTS_DIR / "research_table_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Stats saved to: {stats_path}")
    
    print("\n" + "=" * 70)
    print("DONE - Phase 1 Complete")
    print("=" * 70)
    
    return research_table


if __name__ == "__main__":
    main()

