"""
Data Loader for ETH Lead-Lag Backtest

Loads and filters ETH markets with sufficient coverage for backtesting.
Implements chronological train/test split for proper out-of-sample evaluation.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


# Default paths (relative to project root)
DEFAULT_RESEARCH_DIR = Path('data_v2/research')
DEFAULT_RESEARCH_6LEVELS_DIR = Path('data_v2/research_6levels')


def load_eth_markets(
    min_coverage: float = 90.0,
    research_dir: Optional[Path] = None,
    use_6levels: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load ETH markets with minimum combined coverage.
    
    Args:
        min_coverage: Minimum combined coverage percentage (default 90%)
        research_dir: Path to research directory
        use_6levels: If True, load from research_6levels with all 6 orderbook levels
        
    Returns:
        Tuple of (DataFrame with all data, dict of market info)
    """
    if use_6levels:
        research_dir = research_dir or DEFAULT_RESEARCH_6LEVELS_DIR
    else:
        research_dir = research_dir or DEFAULT_RESEARCH_DIR
    research_dir = Path(research_dir)
    
    # Load canonical dataset
    parquet_path = research_dir / 'canonical_dataset_all_assets.parquet'
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        # Fall back to CSV
        csv_path = research_dir / 'canonical_dataset_all_assets.csv'
        df = pd.read_csv(csv_path)
    
    # Load market info (list format)
    with open(research_dir / 'market_info_all_assets.json') as f:
        market_info_list = json.load(f)
    
    # Convert to dict keyed by market_id
    all_market_info = {m['market_id']: m for m in market_info_list}
    
    # Filter to ETH only
    df = df[df['asset'] == 'ETH'].copy()
    
    # Find valid markets (ETH with sufficient coverage)
    # Use 'both_coverage_pct' as combined coverage
    valid_markets = []
    for mid, info in all_market_info.items():
        if info.get('asset') == 'ETH':
            coverage = info.get('both_coverage_pct', info.get('combined_coverage', 0))
            if coverage >= min_coverage:
                valid_markets.append(mid)
    
    # Filter DataFrame
    df = df[df['market_id'].isin(valid_markets)].copy()
    
    # Sort markets by start time for chronological ordering
    market_start_times = df.groupby('market_id')['timestamp'].min().sort_values()
    market_order = {m: i for i, m in enumerate(market_start_times.index)}
    df['market_order'] = df['market_id'].map(market_order)
    
    # Filter market info
    market_info = {m: all_market_info[m] for m in valid_markets}
    
    print(f"Loaded {len(valid_markets)} ETH markets with >= {min_coverage}% coverage")
    print(f"Total observations: {len(df):,}")
    if use_6levels:
        # Check for 6-level columns
        l2_cols = [c for c in df.columns if 'bid_2' in c or 'ask_2' in c]
        print(f"6-level data: {len(l2_cols)} L2+ columns present")
    
    return df, market_info


def load_unified_eth_markets(
    min_coverage: float = 90.0,
    prefer_6levels: bool = True,
    research_dir_l1: Optional[Path] = None,
    research_dir_l6: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load unified ETH markets combining L1-era and L6-era datasets.
    
    This merges:
    - L1 dataset: Original dataset with L1-only orderbook (more markets, older)
    - L6 dataset: New dataset with L2-L6 depth (subset of L1, newer)
    
    Strategy:
    - For markets in both: prefer L6 data (has depth)
    - For L1-only markets: add empty L2-L6 columns (NaN = zero depth)
    - All markets use same execution model (walk_the_book handles NaN gracefully)
    
    Args:
        min_coverage: Minimum combined coverage percentage
        prefer_6levels: If True, use L6 data where available; else use L1
        research_dir_l1: Path to L1 research directory
        research_dir_l6: Path to L6 research directory
        
    Returns:
        Tuple of (unified DataFrame, dict of market info)
    """
    research_dir_l1 = Path(research_dir_l1 or DEFAULT_RESEARCH_DIR)
    research_dir_l6 = Path(research_dir_l6 or DEFAULT_RESEARCH_6LEVELS_DIR)
    
    # Load both datasets
    print("Loading L1 dataset...")
    df_l1 = pd.read_parquet(research_dir_l1 / 'canonical_dataset_all_assets.parquet')
    df_l1 = df_l1[df_l1['asset'] == 'ETH'].copy()
    
    print("Loading L6 dataset...")
    df_l6 = pd.read_parquet(research_dir_l6 / 'canonical_dataset_all_assets.parquet')
    df_l6 = df_l6[df_l6['asset'] == 'ETH'].copy()
    
    # Get L6 column names (L2-L6 depth columns)
    l6_cols = [c for c in df_l6.columns if any(f'_{i}_' in c for i in range(2, 7))]
    
    # Add empty L2-L6 columns to L1 dataset if missing
    for col in l6_cols:
        if col not in df_l1.columns:
            df_l1[col] = np.nan
    
    # Get market IDs
    markets_l1 = set(df_l1['market_id'].unique())
    markets_l6 = set(df_l6['market_id'].unique())
    
    print(f"L1 markets: {len(markets_l1)}")
    print(f"L6 markets: {len(markets_l6)}")
    print(f"Overlap: {len(markets_l1 & markets_l6)}")
    print(f"L1-only: {len(markets_l1 - markets_l6)}")
    
    # Merge: prefer L6 where available, otherwise use L1
    if prefer_6levels:
        # Use L6 data for overlapping markets
        markets_to_use_l6 = markets_l1 & markets_l6
        markets_to_use_l1 = markets_l1 - markets_l6
        
        df_l6_subset = df_l6[df_l6['market_id'].isin(markets_to_use_l6)]
        df_l1_subset = df_l1[df_l1['market_id'].isin(markets_to_use_l1)]
        
        # Ensure both have same columns
        all_cols = set(df_l1.columns) | set(df_l6.columns)
        for col in all_cols:
            if col not in df_l6_subset.columns:
                df_l6_subset = df_l6_subset.copy()
                df_l6_subset[col] = np.nan
            if col not in df_l1_subset.columns:
                df_l1_subset = df_l1_subset.copy()
                df_l1_subset[col] = np.nan
        
        # Combine
        df_unified = pd.concat([df_l6_subset, df_l1_subset], ignore_index=True)
        print(f"Using L6 data for {len(markets_to_use_l6)} markets")
        print(f"Using L1 data for {len(markets_to_use_l1)} markets")
    else:
        # Use L1 for all (but add L6 columns as NaN)
        df_unified = df_l1.copy()
        print(f"Using L1 data for all {len(markets_l1)} markets (L6 columns as NaN)")
    
    # Load market info (prefer L6, fallback to L1)
    market_info = {}
    
    # Try L6 first
    info_path_l6 = research_dir_l6 / 'market_info_all_assets.json'
    if info_path_l6.exists():
        with open(info_path_l6) as f:
            info_l6 = {m['market_id']: m for m in json.load(f)}
        market_info.update(info_l6)
    
    # Fill gaps from L1
    info_path_l1 = research_dir_l1 / 'market_info_all_assets.json'
    if info_path_l1.exists():
        with open(info_path_l1) as f:
            info_l1 = {m['market_id']: m for m in json.load(f)}
        for mid, info in info_l1.items():
            if mid not in market_info:
                market_info[mid] = info
    
    # Filter by coverage
    valid_markets = []
    for mid, info in market_info.items():
        if info.get('asset') == 'ETH':
            coverage = info.get('both_coverage_pct', info.get('combined_coverage', 0))
            if coverage >= min_coverage:
                valid_markets.append(mid)
    
    df_unified = df_unified[df_unified['market_id'].isin(valid_markets)].copy()
    
    # Sort markets by start time for chronological ordering
    market_start_times = df_unified.groupby('market_id')['timestamp'].min().sort_values()
    market_order = {m: i for i, m in enumerate(market_start_times.index)}
    df_unified['market_order'] = df_unified['market_id'].map(market_order)
    
    # Filter market info
    market_info = {m: market_info[m] for m in valid_markets}
    
    print(f"\nUnified dataset:")
    print(f"  Markets: {len(valid_markets)}")
    print(f"  Total observations: {len(df_unified):,}")
    
    # Check L6 column presence
    l6_cols_present = [c for c in df_unified.columns if any(f'_{i}_' in c for i in range(2, 7))]
    markets_with_l6 = df_unified[df_unified[l6_cols_present].notna().any(axis=1)]['market_id'].nunique()
    print(f"  Markets with L6 data: {markets_with_l6}")
    print(f"  Markets with L1 only: {len(valid_markets) - markets_with_l6}")
    
    return df_unified, market_info


def load_6level_markets(
    min_coverage: float = 90.0,
    research_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load ETH markets with 6-level orderbook data.
    
    This is a convenience wrapper around load_eth_markets with use_6levels=True.
    
    The 6-level data includes:
    - Level 1: pm_{up|down}_best_bid/ask, pm_{up|down}_best_bid_size/ask_size
    - Levels 2-6: pm_{up|down}_bid_2..6, pm_{up|down}_ask_2..6
    - Level 2-6 sizes: pm_{up|down}_bid_2_size..6, pm_{up|down}_ask_2_size..6
    
    Args:
        min_coverage: Minimum combined coverage percentage (default 90%)
        research_dir: Path to research directory (defaults to data_v2/research_6levels)
        
    Returns:
        Tuple of (DataFrame with all 6-level data, dict of market info)
    """
    return load_eth_markets(
        min_coverage=min_coverage,
        research_dir=research_dir or DEFAULT_RESEARCH_6LEVELS_DIR,
        use_6levels=True
    )


def validate_6level_columns(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate that all 6-level orderbook columns are present.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dict with validation results
    """
    checks = {}
    
    # Expected column patterns
    tokens = ['up', 'down']
    sides = ['bid', 'ask']
    
    for token in tokens:
        for side in sides:
            # Level 1
            l1_px = f'pm_{token}_best_{side}'
            l1_sz = f'pm_{token}_best_{side}_size'
            checks[f'{token}_{side}_L1'] = l1_px in df.columns and l1_sz in df.columns
            
            # Levels 2-6
            for level in range(2, 7):
                px_col = f'pm_{token}_{side}_{level}'
                sz_col = f'pm_{token}_{side}_{level}_size'
                checks[f'{token}_{side}_L{level}'] = px_col in df.columns and sz_col in df.columns
    
    # Summary
    all_present = all(checks.values())
    checks['all_6_levels_present'] = all_present
    
    if not all_present:
        missing = [k for k, v in checks.items() if not v and k != 'all_6_levels_present']
        print(f"[WARN] Missing 6-level columns: {missing}")
    
    return checks


def get_train_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Split data chronologically into train and test sets.
    
    Args:
        df: Full DataFrame with market_order column
        train_frac: Fraction of markets for training (default 0.7)
        
    Returns:
        Tuple of (train_df, test_df, train_market_ids, test_market_ids)
    """
    # Get unique markets sorted by order
    markets = df.groupby('market_id')['market_order'].first().sort_values()
    n_markets = len(markets)
    n_train = int(n_markets * train_frac)
    
    train_markets = markets.iloc[:n_train].index.tolist()
    test_markets = markets.iloc[n_train:].index.tolist()
    
    train_df = df[df['market_id'].isin(train_markets)].copy()
    test_df = df[df['market_id'].isin(test_markets)].copy()
    
    print(f"Train: {len(train_markets)} markets ({len(train_df):,} obs)")
    print(f"Test: {len(test_markets)} markets ({len(test_df):,} obs)")
    
    return train_df, test_df, train_markets, test_markets


def get_walk_forward_splits(
    df: pd.DataFrame,
    train_size: int = 10,
    test_size: int = 2,
    step_size: int = 2
) -> List[Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]]:
    """
    Generate walk-forward validation splits.
    
    Args:
        df: Full DataFrame
        train_size: Number of markets in training window
        test_size: Number of markets in test window
        step_size: How many markets to step forward each iteration
        
    Returns:
        List of (train_df, test_df, train_ids, test_ids) tuples
    """
    markets = df.groupby('market_id')['market_order'].first().sort_values()
    market_list = markets.index.tolist()
    n_markets = len(market_list)
    
    splits = []
    start = 0
    
    while start + train_size + test_size <= n_markets:
        train_markets = market_list[start:start + train_size]
        test_markets = market_list[start + train_size:start + train_size + test_size]
        
        train_df = df[df['market_id'].isin(train_markets)].copy()
        test_df = df[df['market_id'].isin(test_markets)].copy()
        
        splits.append((train_df, test_df, train_markets, test_markets))
        start += step_size
    
    print(f"Generated {len(splits)} walk-forward splits")
    return splits


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add useful derived columns for analysis.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        DataFrame with additional columns including:
        - Mid prices (pm_up_mid, pm_down_mid)
        - CL returns (cl_return_bps, cl_cum_return_bps)
        - PM changes (pm_up_change, pm_down_change)
        - Spreads (up_spread, down_spread)
        - No-arb sums (bid_sum, ask_sum)
        - Above/below strike indicator
        - Realized volatility (realized_vol_bps, realized_vol_15s, realized_vol_60s, cl_vol_30s)
    """
    df = df.copy()
    
    # Mid prices
    if 'pm_up_mid' not in df.columns:
        df['pm_up_mid'] = (df['pm_up_best_bid'] + df['pm_up_best_ask']) / 2
    if 'pm_down_mid' not in df.columns:
        df['pm_down_mid'] = (df['pm_down_best_bid'] + df['pm_down_best_ask']) / 2
    
    # CL returns (per market)
    df['cl_return_bps'] = df.groupby('market_id')['cl_mid'].pct_change() * 10000
    
    # CL cumulative return from start
    def calc_cum_return(group):
        first_price = group.iloc[0]
        return (group / first_price - 1) * 10000
    df['cl_cum_return_bps'] = df.groupby('market_id')['cl_mid'].transform(calc_cum_return)
    
    # PM changes
    df['pm_up_change'] = df.groupby('market_id')['pm_up_mid'].diff()
    df['pm_down_change'] = df.groupby('market_id')['pm_down_mid'].diff()
    
    # Spreads (raw)
    df['up_spread'] = df['pm_up_best_ask'] - df['pm_up_best_bid']
    df['down_spread'] = df['pm_down_best_ask'] - df['pm_down_best_bid']
    
    # No-arb check
    df['bid_sum'] = df['pm_up_best_bid'] + df['pm_down_best_bid']
    df['ask_sum'] = df['pm_up_best_ask'] + df['pm_down_best_ask']
    
    # Above/below strike
    df['above_strike'] = df['cl_mid'] > df['K']
    
    # Rolling volatility (30s window) - legacy column for compatibility
    df['cl_vol_30s'] = df.groupby('market_id')['cl_return_bps'].transform(
        lambda x: x.rolling(30, min_periods=5).std()
    )
    
    # =========================================================================
    # REALIZED VOLATILITY (backward-looking, observed data only)
    # =========================================================================
    # These use the proper methodology: only observed (non-forward-filled) data
    
    # Determine if we should filter for observed data
    use_observed_only = 'cl_ffill' in df.columns
    
    # 30s window (default for mispricing strategy)
    df['realized_vol_bps'] = _compute_realized_vol_column(
        df, window_size=30, min_periods=5, use_observed_only=use_observed_only
    )
    
    # 15s window (for sensitivity analysis)
    df['realized_vol_15s'] = _compute_realized_vol_column(
        df, window_size=15, min_periods=3, use_observed_only=use_observed_only
    )
    
    # 60s window (for sensitivity analysis)
    df['realized_vol_60s'] = _compute_realized_vol_column(
        df, window_size=60, min_periods=10, use_observed_only=use_observed_only
    )
    
    return df


def _compute_realized_vol_column(
    df: pd.DataFrame, 
    window_size: int, 
    min_periods: int,
    use_observed_only: bool
) -> pd.Series:
    """
    Helper to compute realized volatility column.
    
    Args:
        df: DataFrame with cl_return_bps and optionally cl_ffill
        window_size: Rolling window size
        min_periods: Minimum observations required
        use_observed_only: If True, mask forward-filled data
        
    Returns:
        Series of realized volatility values
    """
    # Create a copy of returns
    returns = df['cl_return_bps'].copy()
    
    # Mask forward-filled data if requested
    if use_observed_only and 'cl_ffill' in df.columns:
        returns = returns.where(df['cl_ffill'] == 0, np.nan)
    
    # Compute rolling std per market
    return df.groupby('market_id').apply(
        lambda g: returns.loc[g.index].rolling(window=window_size, min_periods=min_periods).std()
    ).reset_index(level=0, drop=True)


def get_market_summary(df: pd.DataFrame, market_info: Dict) -> pd.DataFrame:
    """
    Create summary DataFrame with one row per market.
    
    Args:
        df: Full DataFrame
        market_info: Dict of market info
        
    Returns:
        Summary DataFrame
    """
    summaries = []
    
    for market_id in df['market_id'].unique():
        market_df = df[df['market_id'] == market_id]
        info = market_info.get(market_id, {})
        
        summaries.append({
            'market_id': market_id,
            'K': info.get('K'),
            'settlement': info.get('settlement'),
            'Y': info.get('Y'),
            'start_time': market_df['timestamp'].min(),
            'end_time': market_df['timestamp'].max(),
            'n_obs': len(market_df),
            'combined_coverage': info.get('both_coverage_pct', info.get('combined_coverage')),
            'cl_coverage': info.get('cl_coverage_pct', info.get('cl_coverage')),
            'pm_coverage': info.get('pm_coverage_pct', info.get('pm_coverage')),
            'cl_range_bps': (market_df['cl_mid'].max() - market_df['cl_mid'].min()) / market_df['cl_mid'].iloc[0] * 10000,
            'avg_up_spread': (market_df['pm_up_best_ask'] - market_df['pm_up_best_bid']).mean(),
            'avg_down_spread': (market_df['pm_down_best_ask'] - market_df['pm_down_best_bid']).mean(),
        })
    
    return pd.DataFrame(summaries)


def validate_data(df: pd.DataFrame, market_info: Dict) -> Dict[str, bool]:
    """
    Run validation checks on loaded data.
    
    Args:
        df: Loaded DataFrame
        market_info: Market info dict
        
    Returns:
        Dict of check names to pass/fail
    """
    checks = {}
    
    # Check 1: All markets have K and Y
    checks['all_markets_have_K'] = all(
        market_info.get(m, {}).get('K') is not None 
        for m in df['market_id'].unique()
    )
    checks['all_markets_have_Y'] = all(
        market_info.get(m, {}).get('Y') is not None 
        for m in df['market_id'].unique()
    )
    
    # Check 2: t ranges from 0 to ~900
    t_ranges = df.groupby('market_id')['t'].agg(['min', 'max'])
    checks['t_starts_at_0'] = (t_ranges['min'] == 0).all()
    checks['t_ends_near_900'] = (t_ranges['max'] >= 850).all()
    
    # Check 3: No duplicate (market_id, t) pairs
    checks['no_duplicates'] = not df.duplicated(['market_id', 't']).any()
    
    # Check 4: tau is consistent (tau = 900 - t)
    checks['tau_consistent'] = np.allclose(df['tau'], 900 - df['t'], atol=1)
    
    # Check 5: delta_bps is reasonable
    checks['delta_bps_reasonable'] = df['delta_bps'].abs().max() < 5000  # <50%
    
    # Check 6: PM prices in [0, 1]
    checks['pm_prices_valid'] = (
        (df['pm_up_best_bid'] >= 0).all() and 
        (df['pm_up_best_ask'] <= 1).all() and
        (df['pm_down_best_bid'] >= 0).all() and 
        (df['pm_down_best_ask'] <= 1).all()
    )
    
    # Print results
    print("\nData Validation:")
    for check, passed in checks.items():
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {check}")
    
    return checks


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Load ETH markets for backtesting')
    parser.add_argument('--min-coverage', type=float, default=90.0,
                        help='Minimum combined coverage %%')
    parser.add_argument('--train-frac', type=float, default=0.7,
                        help='Fraction of markets for training')
    
    args = parser.parse_args()
    
    # Load data
    df, market_info = load_eth_markets(min_coverage=args.min_coverage)
    
    # Validate
    validate_data(df, market_info)
    
    # Add derived columns
    df = add_derived_columns(df)
    
    # Create train/test split
    train_df, test_df, train_ids, test_ids = get_train_test_split(df, args.train_frac)
    
    # Market summary
    summary = get_market_summary(df, market_info)
    print("\nMarket Summary:")
    print(summary.to_string())

