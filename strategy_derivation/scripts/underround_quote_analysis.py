#!/usr/bin/env python3
"""
Underround Quote-Level Analysis

Analyzes quote-level triggers for underround harvesting strategy on markets with size data.
Computes frequency, magnitude, and achievable capacity of underround opportunities.

Output:
- underround_quote_analysis.json: Per-market and aggregate statistics
- underround_opportunity_log.csv: Every triggered opportunity with details
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Configuration
BASE_DIR = Path(__file__).parent.parent
RESEARCH_DIR = BASE_DIR.parent / "data_v2" / "research"
OUTPUT_DIR = BASE_DIR / "results"

# Strategy parameters to test
EPSILON_VALUES = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]  # Underround threshold
MIN_CAPACITY_VALUES = [10, 25, 50, 100]  # Minimum achievable capacity
MIN_TAU = 60  # Avoid last 60 seconds (settlement risk)
MIN_PRICE = 0.05  # Minimum valid price (filter out dust quotes like $0.01)
MAX_PRICE = 0.95  # Maximum valid price


@dataclass
class UnderroundOpportunity:
    """Single underround opportunity."""
    market_id: str
    t: int
    tau: int
    timestamp: str
    sum_asks: float
    underround: float  # 1 - sum_asks
    up_ask: float
    down_ask: float
    up_ask_size: float
    down_ask_size: float
    achievable_capacity: float  # min(up_ask_size, down_ask_size)
    capacity_weighted_edge: float  # underround * achievable_capacity


def load_volume_markets() -> pd.DataFrame:
    """Load markets with size data (the 12 volume markets)."""
    df = pd.read_parquet(RESEARCH_DIR / "canonical_dataset_all_assets.parquet")
    
    # Filter to volume markets (16:30-19:15 on Jan 6)
    volume_market_ids = [
        '20260106_1630', '20260106_1645', '20260106_1700', '20260106_1715',
        '20260106_1730', '20260106_1745', '20260106_1800', '20260106_1815',
        '20260106_1830', '20260106_1845', '20260106_1900', '20260106_1915'
    ]
    
    df = df[df['market_id'].isin(volume_market_ids)].copy()
    
    # Check size data availability
    has_both_sizes = (
        df['pm_up_best_ask_size'].notna() & 
        df['pm_down_best_ask_size'].notna()
    )
    
    print(f"Loaded {len(df)} rows from {df['market_id'].nunique()} volume markets")
    print(f"Rows with both ask sizes: {has_both_sizes.sum()} ({has_both_sizes.mean()*100:.1f}%)")
    
    return df


def compute_underround_stats(
    df: pd.DataFrame, 
    epsilon: float,
    min_capacity: float = 0,
    min_tau: int = 60
) -> Tuple[Dict[str, Any], List[UnderroundOpportunity]]:
    """
    Compute underround statistics for given parameters.
    
    Args:
        df: DataFrame with quote data
        epsilon: Underround threshold (trigger when sum_asks < 1 - epsilon)
        min_capacity: Minimum achievable capacity to count as tradeable
        min_tau: Minimum time to expiry (skip last N seconds)
    
    Returns:
        Tuple of (aggregate stats dict, list of opportunities)
    """
    opportunities = []
    market_stats = {}
    
    for market_id, mdf in df.groupby('market_id'):
        mdf = mdf.sort_values('t').copy()
        
        # Compute underround
        mdf['underround'] = 1 - mdf['sum_asks']
        
        # Compute achievable capacity (min of both ask sizes)
        mdf['achievable_capacity'] = np.minimum(
            mdf['pm_up_best_ask_size'].fillna(0),
            mdf['pm_down_best_ask_size'].fillna(0)
        )
        
        # Apply filters
        valid = (
            (mdf['underround'] >= epsilon) &  # Underround exceeds threshold
            (mdf['tau'] >= min_tau) &  # Not too close to expiry
            (mdf['achievable_capacity'] >= min_capacity) &  # Sufficient capacity
            (mdf['pm_up_best_ask_size'].notna()) &  # Have size data
            (mdf['pm_down_best_ask_size'].notna()) &
            # Filter out extreme/dust quotes (unrealistic prices)
            (mdf['pm_up_best_ask'] >= MIN_PRICE) &
            (mdf['pm_up_best_ask'] <= MAX_PRICE) &
            (mdf['pm_down_best_ask'] >= MIN_PRICE) &
            (mdf['pm_down_best_ask'] <= MAX_PRICE)
        )
        
        triggered = mdf[valid]
        
        # Record opportunities
        for _, row in triggered.iterrows():
            opp = UnderroundOpportunity(
                market_id=market_id,
                t=int(row['t']),
                tau=int(row['tau']),
                timestamp=str(row['timestamp']),
                sum_asks=float(row['sum_asks']),
                underround=float(row['underround']),
                up_ask=float(row['pm_up_best_ask']),
                down_ask=float(row['pm_down_best_ask']),
                up_ask_size=float(row['pm_up_best_ask_size']),
                down_ask_size=float(row['pm_down_best_ask_size']),
                achievable_capacity=float(row['achievable_capacity']),
                capacity_weighted_edge=float(row['underround'] * row['achievable_capacity'])
            )
            opportunities.append(opp)
        
        # Compute per-market stats
        total_seconds = len(mdf[mdf['tau'] >= min_tau])  # Seconds in valid window
        n_triggered = len(triggered)
        
        market_stats[market_id] = {
            'total_seconds': total_seconds,
            'triggered_seconds': n_triggered,
            'trigger_pct': n_triggered / total_seconds * 100 if total_seconds > 0 else 0,
            'mean_underround': triggered['underround'].mean() if len(triggered) > 0 else None,
            'median_underround': triggered['underround'].median() if len(triggered) > 0 else None,
            'max_underround': triggered['underround'].max() if len(triggered) > 0 else None,
            'mean_capacity': triggered['achievable_capacity'].mean() if len(triggered) > 0 else None,
            'total_capacity_weighted_edge': (triggered['underround'] * triggered['achievable_capacity']).sum() if len(triggered) > 0 else 0,
        }
    
    # Aggregate stats
    all_triggered = [o for o in opportunities]
    total_seconds = sum(m['total_seconds'] for m in market_stats.values())
    total_triggered = sum(m['triggered_seconds'] for m in market_stats.values())
    
    aggregate = {
        'epsilon': epsilon,
        'min_capacity': min_capacity,
        'min_tau': min_tau,
        'n_markets': len(market_stats),
        'markets_with_opportunities': sum(1 for m in market_stats.values() if m['triggered_seconds'] > 0),
        'total_seconds': total_seconds,
        'total_triggered_seconds': total_triggered,
        'trigger_pct': total_triggered / total_seconds * 100 if total_seconds > 0 else 0,
        'total_opportunities': len(opportunities),
    }
    
    if opportunities:
        underrounds = [o.underround for o in opportunities]
        capacities = [o.achievable_capacity for o in opportunities]
        cw_edges = [o.capacity_weighted_edge for o in opportunities]
        
        aggregate.update({
            'mean_underround': np.mean(underrounds),
            'median_underround': np.median(underrounds),
            'p25_underround': np.percentile(underrounds, 25),
            'p75_underround': np.percentile(underrounds, 75),
            'max_underround': np.max(underrounds),
            'mean_capacity': np.mean(capacities),
            'median_capacity': np.median(capacities),
            'min_capacity_observed': np.min(capacities),
            'total_capacity_weighted_edge': np.sum(cw_edges),
            'mean_capacity_weighted_edge': np.mean(cw_edges),
        })
    
    return {'aggregate': aggregate, 'per_market': market_stats}, opportunities


def compute_duration_analysis(df: pd.DataFrame, epsilon: float) -> Dict[str, Any]:
    """
    Analyze how long underround opportunities persist.
    """
    durations = []
    
    for market_id, mdf in df.groupby('market_id'):
        mdf = mdf.sort_values('t').copy()
        mdf['underround'] = 1 - mdf['sum_asks']
        
        # Apply same price filters as main analysis
        valid_prices = (
            (mdf['pm_up_best_ask'] >= MIN_PRICE) &
            (mdf['pm_up_best_ask'] <= MAX_PRICE) &
            (mdf['pm_down_best_ask'] >= MIN_PRICE) &
            (mdf['pm_down_best_ask'] <= MAX_PRICE)
        )
        mdf['is_underround'] = (mdf['underround'] >= epsilon) & valid_prices
        
        # Find consecutive runs of underround
        mdf['run_id'] = (mdf['is_underround'] != mdf['is_underround'].shift()).cumsum()
        
        for run_id, run_df in mdf[mdf['is_underround']].groupby('run_id'):
            duration = len(run_df)
            mean_edge = run_df['underround'].mean()
            durations.append({
                'market_id': market_id,
                'start_t': int(run_df['t'].min()),
                'duration_seconds': duration,
                'mean_underround': mean_edge
            })
    
    if not durations:
        return {'n_episodes': 0}
    
    dur_values = [d['duration_seconds'] for d in durations]
    
    return {
        'n_episodes': len(durations),
        'mean_duration': np.mean(dur_values),
        'median_duration': np.median(dur_values),
        'max_duration': np.max(dur_values),
        'p25_duration': np.percentile(dur_values, 25),
        'p75_duration': np.percentile(dur_values, 75),
        'duration_distribution': {
            '1s': sum(1 for d in dur_values if d == 1),
            '2-5s': sum(1 for d in dur_values if 2 <= d <= 5),
            '6-10s': sum(1 for d in dur_values if 6 <= d <= 10),
            '11-30s': sum(1 for d in dur_values if 11 <= d <= 30),
            '>30s': sum(1 for d in dur_values if d > 30),
        }
    }


def compute_tau_distribution(df: pd.DataFrame, epsilon: float) -> Dict[str, float]:
    """
    Analyze when underround opportunities occur relative to expiry.
    """
    df = df.copy()
    df['underround'] = 1 - df['sum_asks']
    
    # Apply same price filters as main analysis
    valid_prices = (
        (df['pm_up_best_ask'] >= MIN_PRICE) &
        (df['pm_up_best_ask'] <= MAX_PRICE) &
        (df['pm_down_best_ask'] >= MIN_PRICE) &
        (df['pm_down_best_ask'] <= MAX_PRICE)
    )
    df['is_underround'] = (df['underround'] >= epsilon) & valid_prices
    
    # Bin by tau
    tau_bins = [
        (0, 60, 'last_1min'),
        (60, 120, 'last_2min'),
        (120, 300, 'last_5min'),
        (300, 600, 'mid_market'),
        (600, 900, 'early_market'),
    ]
    
    distribution = {}
    for tau_min, tau_max, label in tau_bins:
        in_bin = (df['tau'] >= tau_min) & (df['tau'] < tau_max)
        total_in_bin = in_bin.sum()
        underround_in_bin = (in_bin & df['is_underround']).sum()
        
        distribution[label] = {
            'total_seconds': int(total_in_bin),
            'underround_seconds': int(underround_in_bin),
            'underround_pct': underround_in_bin / total_in_bin * 100 if total_in_bin > 0 else 0
        }
    
    return distribution


def compute_overround_stats(df: pd.DataFrame, epsilon: float) -> Dict[str, Any]:
    """
    Compute overround statistics (sum_bids > 1 + epsilon).
    This is the opposite arbitrage - selling both sides.
    """
    df = df.copy()
    df['overround'] = df['sum_bids'] - 1
    
    # Apply same price filters (using bid prices for overround)
    valid_prices = (
        (df['pm_up_best_bid'] >= MIN_PRICE) &
        (df['pm_up_best_bid'] <= MAX_PRICE) &
        (df['pm_down_best_bid'] >= MIN_PRICE) &
        (df['pm_down_best_bid'] <= MAX_PRICE)
    )
    
    # Check overround condition with valid prices
    triggered = df[(df['overround'] >= epsilon) & valid_prices]
    total = len(df)
    
    if len(triggered) == 0:
        return {
            'epsilon': epsilon,
            'total_seconds': total,
            'triggered_seconds': 0,
            'trigger_pct': 0,
        }
    
    return {
        'epsilon': epsilon,
        'total_seconds': total,
        'triggered_seconds': len(triggered),
        'trigger_pct': len(triggered) / total * 100,
        'mean_overround': triggered['overround'].mean(),
        'median_overround': triggered['overround'].median(),
        'max_overround': triggered['overround'].max(),
    }


def main():
    print("=" * 70)
    print("UNDERROUND QUOTE-LEVEL ANALYSIS")
    print("=" * 70)
    print()
    
    # Load data
    df = load_volume_markets()
    
    results = {
        'generated_at': datetime.now().isoformat(),
        'n_markets': df['market_id'].nunique(),
        'total_rows': len(df),
        'parameter_sweeps': [],
        'best_params': None,
        'duration_analysis': None,
        'tau_distribution': None,
        'overround_stats': None,
    }
    
    all_opportunities = []
    
    # Parameter sweep
    print("\n" + "=" * 70)
    print("PARAMETER SWEEP")
    print("=" * 70)
    print(f"\n{'Epsilon':>8} {'MinCap':>8} {'Triggers':>10} {'Pct':>8} {'MedEdge':>10} {'MedCap':>10} {'TotalCWE':>12}")
    print("-" * 78)
    
    best_cwe = 0
    best_params = None
    
    for epsilon in EPSILON_VALUES:
        for min_cap in MIN_CAPACITY_VALUES:
            stats, opps = compute_underround_stats(df, epsilon, min_cap, MIN_TAU)
            agg = stats['aggregate']
            
            results['parameter_sweeps'].append(stats)
            
            # Track best by total capacity-weighted edge
            cwe = agg.get('total_capacity_weighted_edge', 0)
            if cwe > best_cwe:
                best_cwe = cwe
                best_params = {'epsilon': epsilon, 'min_capacity': min_cap, 'stats': stats}
            
            # Collect opportunities for the baseline epsilon=0.02
            if epsilon == 0.02 and min_cap == 10:
                all_opportunities.extend(opps)
            
            med_edge = agg.get('median_underround', 0) or 0
            med_cap = agg.get('median_capacity', 0) or 0
            total_cwe = agg.get('total_capacity_weighted_edge', 0) or 0
            
            print(f"{epsilon:>8.2f} {min_cap:>8} {agg['total_opportunities']:>10} "
                  f"{agg['trigger_pct']:>7.2f}% {med_edge:>10.4f} {med_cap:>10.1f} ${total_cwe:>11.2f}")
    
    results['best_params'] = best_params
    
    # Duration analysis (using epsilon=0.02)
    print("\n" + "=" * 70)
    print("DURATION ANALYSIS (epsilon=0.02)")
    print("=" * 70)
    
    dur_stats = compute_duration_analysis(df, 0.02)
    results['duration_analysis'] = dur_stats
    
    if dur_stats['n_episodes'] > 0:
        print(f"\nTotal episodes: {dur_stats['n_episodes']}")
        print(f"Mean duration: {dur_stats['mean_duration']:.1f}s")
        print(f"Median duration: {dur_stats['median_duration']:.1f}s")
        print(f"Max duration: {dur_stats['max_duration']}s")
        print(f"\nDuration distribution:")
        for label, count in dur_stats['duration_distribution'].items():
            print(f"  {label}: {count}")
    else:
        print("\nNo underround episodes found.")
    
    # Tau distribution
    print("\n" + "=" * 70)
    print("TAU DISTRIBUTION (epsilon=0.02)")
    print("=" * 70)
    
    tau_dist = compute_tau_distribution(df, 0.02)
    results['tau_distribution'] = tau_dist
    
    print(f"\n{'Time Period':>15} {'Total':>10} {'Underround':>12} {'Pct':>8}")
    print("-" * 50)
    for period, stats in tau_dist.items():
        print(f"{period:>15} {stats['total_seconds']:>10} {stats['underround_seconds']:>12} {stats['underround_pct']:>7.2f}%")
    
    # Overround analysis
    print("\n" + "=" * 70)
    print("OVERROUND ANALYSIS (sum_bids > 1)")
    print("=" * 70)
    
    overround_results = []
    print(f"\n{'Epsilon':>8} {'Triggers':>10} {'Pct':>8} {'MedOver':>10}")
    print("-" * 40)
    
    for epsilon in EPSILON_VALUES:
        over_stats = compute_overround_stats(df, epsilon)
        overround_results.append(over_stats)
        med_over = over_stats.get('median_overround', 0) or 0
        print(f"{epsilon:>8.2f} {over_stats['triggered_seconds']:>10} {over_stats['trigger_pct']:>7.2f}% {med_over:>10.4f}")
    
    results['overround_stats'] = overround_results
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if best_params:
        bp = best_params
        agg = bp['stats']['aggregate']
        print(f"\nBest parameters (by total capacity-weighted edge):")
        print(f"  Epsilon: {bp['epsilon']}")
        print(f"  Min Capacity: {bp['min_capacity']}")
        print(f"  Total opportunities: {agg['total_opportunities']}")
        print(f"  Trigger percentage: {agg['trigger_pct']:.2f}%")
        print(f"  Median underround: ${agg.get('median_underround', 0):.4f}")
        print(f"  Total capacity-weighted edge: ${agg.get('total_capacity_weighted_edge', 0):.2f}")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    output_path = OUTPUT_DIR / "underround_quote_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {output_path}")
    
    # Save opportunity log
    if all_opportunities:
        opp_df = pd.DataFrame([asdict(o) for o in all_opportunities])
        opp_path = OUTPUT_DIR / "underround_opportunity_log.csv"
        opp_df.to_csv(opp_path, index=False)
        print(f"Saved: {opp_path}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()

