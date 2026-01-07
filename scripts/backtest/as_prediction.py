#!/usr/bin/env python3
"""
Adverse Selection Prediction Model

This module builds a model to predict adverse selection (AS) at fill time
and identifies hard filters that can remove the worst fills.

Goal: Find 1-3 filters that remove worst 10-30% of fills while keeping
most spread capture.

Features used:
- spread: Spread width at fill time
- tau: Time-to-expiry at fill time  
- quote_update_rate: How often quotes updated in last N seconds
- pm_mid_return_Ns: PM mid return in last N seconds
- cl_mid_return_Ns: CL mid return in last N seconds
- delta_bps: Distance to strike at fill time
- after_quote_change: Fill happened right after quote moved
- volatility: PM mid volatility over last N seconds
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class FilterResult:
    """Result of applying a filter."""
    filter_name: str
    filter_condition: str
    fills_removed: int
    fills_removed_pct: float
    fills_remaining: int
    as_removed: float  # Total AS removed (positive = cost removed)
    as_remaining: float
    spread_captured_remaining: float
    net_edge_remaining: float  # spread - AS
    net_edge_improvement: float  # Change in net edge per remaining fill


def compute_fill_features(
    fills_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute features for each fill.
    
    Args:
        fills_df: DataFrame with fills (from run_maker_backtest)
        market_df: Full market data for context
        
    Returns:
        DataFrame with features per fill
    """
    if fills_df.empty:
        return pd.DataFrame()
    
    features = fills_df.copy()
    
    # Ensure we have the required columns
    if 'market_id' not in features.columns:
        # Try to extract from order_id or assign
        features['market_id'] = 'unknown'
    
    # Get market context at fill time
    # Create lookup for market data by (market_id, t)
    market_lookup = market_df.set_index(['market_id', 't'])
    
    # Add market features at fill time
    for idx, row in features.iterrows():
        fill_t = row.get('fill_time', 0)
        market_id = row.get('market_id', '')
        
        # Try to get market data at fill time
        try:
            if (market_id, fill_t) in market_lookup.index:
                mkt_row = market_lookup.loc[(market_id, fill_t)]
                
                # Tau
                features.at[idx, 'tau'] = mkt_row.get('tau', 0)
                
                # CL features
                features.at[idx, 'cl_mid'] = mkt_row.get('cl_mid', np.nan)
                features.at[idx, 'strike'] = mkt_row.get('strike', np.nan)
                
                # Compute delta_bps (distance to strike)
                cl_mid = mkt_row.get('cl_mid', np.nan)
                strike = mkt_row.get('strike', np.nan)
                if not pd.isna(cl_mid) and not pd.isna(strike) and strike > 0:
                    features.at[idx, 'delta_bps'] = abs((cl_mid - strike) / strike * 10000)
                
        except (KeyError, TypeError):
            pass
    
    # Compute spread at fill (if not already present)
    if 'spread_at_fill' not in features.columns or features['spread_at_fill'].isna().all():
        # Approximate from mid and fill price
        # For a maker fill, spread ~= 2 * |mid - fill_price|
        features['spread_at_fill'] = 2 * abs(features['mid_at_fill'] - features['fill_price'])
    
    # Compute AS from fill data (already computed by engine)
    if 'adverse_selection_1s' not in features.columns:
        features['adverse_selection_1s'] = features.get('as_1s_cost', np.nan)
    if 'adverse_selection_5s' not in features.columns:
        features['adverse_selection_5s'] = features.get('as_5s_cost', np.nan)
    
    # Compute spread captured
    if 'spread_captured' not in features.columns:
        # Spread captured = |mid - fill_price| * size (for makers)
        features['spread_captured'] = abs(features['mid_at_fill'] - features['fill_price']) * features.get('fill_size', 1.0)
    
    # Net edge per fill
    features['net_edge_1s'] = features['spread_captured'] - features['adverse_selection_1s'].fillna(0)
    features['net_edge_5s'] = features['spread_captured'] - features['adverse_selection_5s'].fillna(0)
    
    # Categorize AS outcome
    features['as_1s_positive'] = (features['adverse_selection_1s'] > 0).astype(int)  # Cost
    features['as_1s_negative'] = (features['adverse_selection_1s'] < 0).astype(int)  # Gain
    
    return features


def analyze_as_by_feature(
    features_df: pd.DataFrame,
    feature_col: str,
    n_bins: int = 5,
) -> pd.DataFrame:
    """
    Analyze AS distribution by feature bins.
    
    Args:
        features_df: DataFrame with fill features
        feature_col: Column to bin by
        n_bins: Number of bins
        
    Returns:
        DataFrame with AS stats per bin
    """
    if feature_col not in features_df.columns:
        return pd.DataFrame()
    
    valid = features_df[features_df[feature_col].notna()].copy()
    if len(valid) == 0:
        return pd.DataFrame()
    
    # Create bins
    try:
        valid['bin'] = pd.qcut(valid[feature_col], n_bins, duplicates='drop')
    except ValueError:
        # Not enough unique values
        valid['bin'] = pd.cut(valid[feature_col], n_bins, duplicates='drop')
    
    # Aggregate by bin
    agg = valid.groupby('bin', observed=True).agg({
        'adverse_selection_1s': ['mean', 'sum', 'count'],
        'adverse_selection_5s': ['mean', 'sum'],
        'spread_captured': ['mean', 'sum'],
        'net_edge_1s': ['mean', 'sum'],
        feature_col: ['min', 'max', 'mean'],
    })
    
    # Flatten column names
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()
    
    return agg


def evaluate_filter(
    features_df: pd.DataFrame,
    filter_mask: pd.Series,
    filter_name: str,
    filter_condition: str,
) -> FilterResult:
    """
    Evaluate the impact of applying a filter.
    
    Args:
        features_df: DataFrame with fill features
        filter_mask: Boolean mask (True = KEEP, False = REMOVE)
        filter_name: Name of the filter
        filter_condition: Description of filter condition
        
    Returns:
        FilterResult with statistics
    """
    total = len(features_df)
    removed = (~filter_mask).sum()
    remaining = filter_mask.sum()
    
    removed_df = features_df[~filter_mask]
    remaining_df = features_df[filter_mask]
    
    as_removed = removed_df['adverse_selection_1s'].sum() if len(removed_df) > 0 else 0
    as_remaining = remaining_df['adverse_selection_1s'].sum() if len(remaining_df) > 0 else 0
    spread_remaining = remaining_df['spread_captured'].sum() if len(remaining_df) > 0 else 0
    net_edge_remaining = remaining_df['net_edge_1s'].sum() if len(remaining_df) > 0 else 0
    
    # Original stats
    total_net_edge_orig = features_df['net_edge_1s'].sum()
    
    # Improvement per remaining fill
    if remaining > 0 and total > 0:
        orig_per_fill = total_net_edge_orig / total
        new_per_fill = net_edge_remaining / remaining
        improvement = new_per_fill - orig_per_fill
    else:
        improvement = 0
    
    return FilterResult(
        filter_name=filter_name,
        filter_condition=filter_condition,
        fills_removed=removed,
        fills_removed_pct=removed / total * 100 if total > 0 else 0,
        fills_remaining=remaining,
        as_removed=as_removed,
        as_remaining=as_remaining,
        spread_captured_remaining=spread_remaining,
        net_edge_remaining=net_edge_remaining,
        net_edge_improvement=improvement,
    )


def find_optimal_filters(
    features_df: pd.DataFrame,
    max_removal_pct: float = 30.0,
) -> List[FilterResult]:
    """
    Find optimal filters that improve net edge.
    
    Args:
        features_df: DataFrame with fill features
        max_removal_pct: Maximum percentage of fills to remove
        
    Returns:
        List of FilterResult, sorted by net edge improvement
    """
    results = []
    
    # Filter 1: Spread threshold
    for spread_thresh in [0.005, 0.01, 0.015, 0.02, 0.025]:
        mask = features_df['spread_at_fill'] >= spread_thresh
        if mask.sum() > 0 and (~mask).sum() / len(features_df) * 100 <= max_removal_pct:
            result = evaluate_filter(
                features_df, mask,
                f"spread >= {spread_thresh}",
                f"Only quote when spread >= ${spread_thresh:.3f}"
            )
            results.append(result)
    
    # Filter 2: Tau threshold
    if 'tau' in features_df.columns:
        for tau_thresh in [60, 90, 120, 150]:
            mask = features_df['tau'] >= tau_thresh
            if mask.sum() > 0 and (~mask).sum() / len(features_df) * 100 <= max_removal_pct:
                result = evaluate_filter(
                    features_df, mask,
                    f"tau >= {tau_thresh}",
                    f"Don't quote when tau < {tau_thresh}s"
                )
                results.append(result)
    
    # Filter 3: Delta (distance to strike)
    if 'delta_bps' in features_df.columns:
        for delta_thresh in [10, 20, 30, 50]:
            mask = features_df['delta_bps'] >= delta_thresh
            if mask.sum() > 0 and (~mask).sum() / len(features_df) * 100 <= max_removal_pct:
                result = evaluate_filter(
                    features_df, mask,
                    f"delta >= {delta_thresh}bps",
                    f"Don't quote when |price - strike| < {delta_thresh}bps"
                )
                results.append(result)
    
    # Filter 4: Only keep fills with positive net edge (oracle filter)
    mask = features_df['net_edge_1s'] > 0
    if mask.sum() > 0:
        result = evaluate_filter(
            features_df, mask,
            "net_edge > 0 (oracle)",
            "Only fills that would have been profitable (look-ahead)"
        )
        results.append(result)
    
    # Filter 5: Remove fills with high AS
    as_90 = features_df['adverse_selection_1s'].quantile(0.90)
    mask = features_df['adverse_selection_1s'] <= as_90
    if mask.sum() > 0:
        result = evaluate_filter(
            features_df, mask,
            "AS <= 90th percentile",
            f"Remove top 10% worst AS fills (AS > ${as_90:.4f})"
        )
        results.append(result)
    
    # Filter 6: Remove fills with extreme AS (top 20%)
    as_80 = features_df['adverse_selection_1s'].quantile(0.80)
    mask = features_df['adverse_selection_1s'] <= as_80
    if mask.sum() > 0:
        result = evaluate_filter(
            features_df, mask,
            "AS <= 80th percentile",
            f"Remove top 20% worst AS fills (AS > ${as_80:.4f})"
        )
        results.append(result)
    
    # Sort by net edge improvement (descending)
    results.sort(key=lambda x: x.net_edge_improvement, reverse=True)
    
    return results


def build_as_prediction_model(
    features_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Build a simple prediction model for AS.
    
    Uses binned averages to estimate E[AS | features].
    
    Args:
        features_df: DataFrame with fill features
        
    Returns:
        Dictionary with model parameters
    """
    model = {
        'type': 'binned_average',
        'bins': {},
    }
    
    # Spread bins
    spread_bins = analyze_as_by_feature(features_df, 'spread_at_fill', n_bins=5)
    if len(spread_bins) > 0:
        model['bins']['spread'] = spread_bins.to_dict('records')
    
    # Tau bins
    if 'tau' in features_df.columns:
        tau_bins = analyze_as_by_feature(features_df, 'tau', n_bins=5)
        if len(tau_bins) > 0:
            model['bins']['tau'] = tau_bins.to_dict('records')
    
    # Delta bins
    if 'delta_bps' in features_df.columns and features_df['delta_bps'].notna().any():
        delta_bins = analyze_as_by_feature(features_df, 'delta_bps', n_bins=5)
        if len(delta_bins) > 0:
            model['bins']['delta_bps'] = delta_bins.to_dict('records')
    
    # Overall stats
    model['overall'] = {
        'mean_as_1s': features_df['adverse_selection_1s'].mean(),
        'std_as_1s': features_df['adverse_selection_1s'].std(),
        'mean_spread_captured': features_df['spread_captured'].mean(),
        'mean_net_edge': features_df['net_edge_1s'].mean(),
    }
    
    return model


def print_as_report(
    features_df: pd.DataFrame,
    filter_results: List[FilterResult],
    model: Dict[str, Any],
):
    """Print the AS prediction report."""
    print("\n" + "="*70)
    print("ADVERSE SELECTION PREDICTION REPORT")
    print("="*70)
    
    print("\n1. OVERALL STATISTICS")
    print("-"*50)
    
    n_fills = len(features_df)
    total_as = features_df['adverse_selection_1s'].sum()
    total_spread = features_df['spread_captured'].sum()
    total_net_edge = features_df['net_edge_1s'].sum()
    
    print(f"  Total fills: {n_fills}")
    print(f"  Total spread captured: ${total_spread:.4f}")
    print(f"  Total adverse selection (1s): ${total_as:.4f}")
    print(f"  Total net edge: ${total_net_edge:.4f}")
    
    n_positive_as = (features_df['adverse_selection_1s'] > 0).sum()
    n_negative_as = (features_df['adverse_selection_1s'] < 0).sum()
    print(f"\n  Fills with AS > 0 (cost): {n_positive_as} ({n_positive_as/n_fills*100:.1f}%)")
    print(f"  Fills with AS < 0 (gain): {n_negative_as} ({n_negative_as/n_fills*100:.1f}%)")
    
    print("\n2. AS BY FEATURE BINS")
    print("-"*50)
    
    # Spread bins
    spread_analysis = analyze_as_by_feature(features_df, 'spread_at_fill', n_bins=4)
    if len(spread_analysis) > 0:
        print("\n  BY SPREAD:")
        for _, row in spread_analysis.iterrows():
            bin_label = str(row['bin'])
            mean_as = row['adverse_selection_1s_mean']
            count = int(row['adverse_selection_1s_count'])
            mean_net = row['net_edge_1s_mean']
            print(f"    {bin_label}: mean_AS=${mean_as:.4f}, count={count}, mean_net=${mean_net:.4f}")
    
    # Tau bins
    if 'tau' in features_df.columns:
        tau_analysis = analyze_as_by_feature(features_df, 'tau', n_bins=4)
        if len(tau_analysis) > 0:
            print("\n  BY TAU:")
            for _, row in tau_analysis.iterrows():
                bin_label = str(row['bin'])
                mean_as = row['adverse_selection_1s_mean']
                count = int(row['adverse_selection_1s_count'])
                print(f"    {bin_label}: mean_AS=${mean_as:.4f}, count={count}")
    
    print("\n3. FILTER ANALYSIS")
    print("-"*50)
    
    print("\n  TOP FILTERS (by net edge improvement):")
    for i, fr in enumerate(filter_results[:10]):
        print(f"\n  {i+1}. {fr.filter_name}")
        print(f"     Condition: {fr.filter_condition}")
        print(f"     Removes: {fr.fills_removed} fills ({fr.fills_removed_pct:.1f}%)")
        print(f"     AS removed: ${fr.as_removed:.4f}")
        print(f"     Net edge remaining: ${fr.net_edge_remaining:.4f}")
        print(f"     Edge improvement per fill: ${fr.net_edge_improvement:.4f}")
    
    print("\n4. RECOMMENDED FILTERS")
    print("-"*50)
    
    # Find best practical filters (not oracle, reasonable removal %)
    practical_filters = [
        f for f in filter_results 
        if 'oracle' not in f.filter_name.lower() 
        and f.fills_removed_pct <= 30
        and f.net_edge_improvement > 0
    ]
    
    if practical_filters:
        print("\n  Best practical filters:")
        for i, fr in enumerate(practical_filters[:3]):
            print(f"\n  {i+1}. {fr.filter_name}")
            print(f"     {fr.filter_condition}")
            print(f"     Removes {fr.fills_removed_pct:.1f}% of fills")
            print(f"     Improves edge by ${fr.net_edge_improvement:.4f} per fill")
    else:
        print("\n  No practical filters found that improve edge.")
    
    print("\n" + "="*70)


def run_as_analysis(
    df: pd.DataFrame,
    strategy: Any = None,
    volume_markets_only: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run full AS analysis pipeline.
    
    Args:
        df: Market data
        strategy: Strategy to use (or default)
        volume_markets_only: Only use volume markets
        verbose: Print report
        
    Returns:
        Dictionary with analysis results
    """
    from scripts.backtest.strategies import SpreadCaptureStrategy
    from scripts.backtest.backtest_engine import run_maker_backtest
    from scripts.backtest.maker_execution import MakerExecutionConfig, FillModel
    
    if strategy is None:
        strategy = SpreadCaptureStrategy(
            spread_min=0.01,
            tau_min=60,
            tau_max=600,
            inventory_limit_up=10.0,
            inventory_limit_down=10.0,
        )
    
    config = MakerExecutionConfig(
        place_latency_ms=100,
        cancel_latency_ms=50,
        fill_model=FillModel.TOUCH_SIZE_PROXY,
        touch_trade_rate_per_second=0.03,  # Calibrated rate
    )
    
    # Run backtest
    result = run_maker_backtest(df, strategy, config, 
                                volume_markets_only=volume_markets_only, 
                                verbose=verbose)
    
    # Get fills as DataFrame
    fills = result.get('fills', [])
    if not fills:
        return {'error': 'No fills'}
    
    fills_df = pd.DataFrame(fills)
    
    # Add market_id from market_results
    market_results = result.get('market_results', {})
    
    # For now, use the fills directly since they already have the needed fields
    features_df = fills_df.copy()
    
    # Compute derived features
    if 'spread_at_fill' not in features_df.columns:
        features_df['spread_at_fill'] = 2 * abs(
            features_df['mid_at_fill'] - features_df['fill_price']
        )
    
    if 'spread_captured' not in features_df.columns:
        features_df['spread_captured'] = abs(
            features_df['mid_at_fill'] - features_df['fill_price']
        ) * features_df.get('fill_size', 1.0)
    
    features_df['adverse_selection_1s'] = features_df.get('adverse_selection_1s', 0)
    features_df['adverse_selection_5s'] = features_df.get('adverse_selection_5s', 0)
    
    features_df['net_edge_1s'] = features_df['spread_captured'] - features_df['adverse_selection_1s'].fillna(0)
    features_df['net_edge_5s'] = features_df['spread_captured'] - features_df['adverse_selection_5s'].fillna(0)
    
    # Find filters
    filter_results = find_optimal_filters(features_df)
    
    # Build prediction model
    model = build_as_prediction_model(features_df)
    
    # Print report
    if verbose:
        print_as_report(features_df, filter_results, model)
    
    return {
        'features_df': features_df,
        'filter_results': filter_results,
        'model': model,
        'backtest_result': result,
    }


def main():
    """Run AS analysis."""
    from scripts.backtest.data_loader import load_eth_markets, add_derived_columns
    
    print("Loading data...")
    df, _ = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    print(f"Loaded {len(df):,} rows, {df['market_id'].nunique()} markets")
    
    # Run analysis
    results = run_as_analysis(df, volume_markets_only=True)


if __name__ == '__main__':
    main()

