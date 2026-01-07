#!/usr/bin/env python3
"""
Step 5: Market Contribution Analysis

For Strategy B:
1. Plot per-market PnL distribution
2. Identify top markets by PnL contribution
3. Leave-one-out robustness test
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest.data_loader import load_eth_markets, add_derived_columns
from scripts.backtest.backtest_engine import run_backtest, ExecutionConfig
from scripts.backtest.strategies import LateDirectionalTakerStrategy

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "results"


def get_per_market_pnl(df: pd.DataFrame, strategy) -> pd.DataFrame:
    """
    Run backtest and get per-market PnL breakdown.
    """
    config = ExecutionConfig()
    result = run_backtest(df, strategy, config)
    
    # Get per-market summary from trades
    trades = result['trades']
    
    if len(trades) == 0:
        return pd.DataFrame()
    
    trades_df = pd.DataFrame(trades)
    
    # Aggregate by market
    market_pnl = trades_df.groupby('market_id').agg({
        'pnl': ['sum', 'count', 'mean'],
    }).round(4)
    
    market_pnl.columns = ['total_pnl', 'n_trades', 'mean_pnl']
    market_pnl = market_pnl.reset_index()
    market_pnl = market_pnl.sort_values('total_pnl', ascending=False)
    
    return market_pnl


def run_leave_one_out(df: pd.DataFrame, strategy, top_markets: List[str]) -> Dict:
    """
    Leave-one-out robustness test.
    
    Remove each top market and recompute t-stat.
    """
    config = ExecutionConfig()
    
    # Get baseline
    baseline_result = run_backtest(df, strategy, config)
    baseline_t = baseline_result['metrics']['t_stat']
    baseline_pnl = baseline_result['metrics']['total_pnl']
    
    results = {
        'baseline': {
            't_stat': baseline_t,
            'total_pnl': baseline_pnl,
            'n_markets': df['market_id'].nunique(),
        },
        'leave_one_out': []
    }
    
    for market_id in top_markets:
        df_loo = df[df['market_id'] != market_id].copy()
        
        try:
            loo_result = run_backtest(df_loo, strategy, config)
            loo_t = loo_result['metrics']['t_stat']
            loo_pnl = loo_result['metrics']['total_pnl']
            
            results['leave_one_out'].append({
                'removed_market': market_id,
                't_stat': loo_t,
                'total_pnl': loo_pnl,
                't_stat_change': loo_t - baseline_t,
                't_stat_change_pct': (loo_t - baseline_t) / baseline_t * 100 if baseline_t != 0 else 0,
            })
        except Exception as e:
            results['leave_one_out'].append({
                'removed_market': market_id,
                'error': str(e),
            })
    
    return results


def main():
    print("=" * 70)
    print("STEP 5: MARKET CONTRIBUTION ANALYSIS")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    df, market_info = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    # Create Strategy B
    strategy = LateDirectionalTakerStrategy(
        tau_max=420,
        delta_threshold_bps=10,
        hold_seconds=120,
    )
    
    print(f"Strategy: {strategy.name}")
    print(f"Markets: {df['market_id'].nunique()}")
    
    # Get per-market PnL
    print("\n" + "=" * 60)
    print("Per-Market PnL Distribution")
    print("=" * 60)
    
    market_pnl = get_per_market_pnl(df, strategy)
    
    if len(market_pnl) == 0:
        print("No trades generated!")
        return
    
    print(f"\nMarkets with trades: {len(market_pnl)}")
    print(f"Total PnL: ${market_pnl['total_pnl'].sum():.2f}")
    
    # Distribution stats
    print(f"\nPnL distribution:")
    print(f"  Mean: ${market_pnl['total_pnl'].mean():.3f}")
    print(f"  Std:  ${market_pnl['total_pnl'].std():.3f}")
    print(f"  Min:  ${market_pnl['total_pnl'].min():.3f}")
    print(f"  Max:  ${market_pnl['total_pnl'].max():.3f}")
    
    # Positive vs negative markets
    n_positive = (market_pnl['total_pnl'] > 0).sum()
    n_negative = (market_pnl['total_pnl'] < 0).sum()
    n_zero = (market_pnl['total_pnl'] == 0).sum()
    
    print(f"\nMarket breakdown:")
    print(f"  Positive PnL: {n_positive} ({n_positive/len(market_pnl)*100:.1f}%)")
    print(f"  Negative PnL: {n_negative} ({n_negative/len(market_pnl)*100:.1f}%)")
    print(f"  Zero PnL:     {n_zero} ({n_zero/len(market_pnl)*100:.1f}%)")
    
    # Top and bottom markets
    print("\n" + "-" * 60)
    print("Top 5 Markets by PnL:")
    print("-" * 60)
    for i, row in market_pnl.head(5).iterrows():
        pct_of_total = row['total_pnl'] / market_pnl['total_pnl'].sum() * 100 if market_pnl['total_pnl'].sum() != 0 else 0
        print(f"  {row['market_id'][:30]:30s}: ${row['total_pnl']:+.3f} ({pct_of_total:+.1f}% of total)")
    
    print("\n" + "-" * 60)
    print("Bottom 5 Markets by PnL:")
    print("-" * 60)
    for i, row in market_pnl.tail(5).iterrows():
        pct_of_total = row['total_pnl'] / market_pnl['total_pnl'].sum() * 100 if market_pnl['total_pnl'].sum() != 0 else 0
        print(f"  {row['market_id'][:30]:30s}: ${row['total_pnl']:+.3f} ({pct_of_total:+.1f}% of total)")
    
    # Concentration check
    print("\n" + "=" * 60)
    print("PnL Concentration Analysis")
    print("=" * 60)
    
    total_pnl = market_pnl['total_pnl'].sum()
    if total_pnl > 0:
        top1_share = market_pnl.iloc[0]['total_pnl'] / total_pnl * 100
        top3_share = market_pnl.head(3)['total_pnl'].sum() / total_pnl * 100
        top5_share = market_pnl.head(5)['total_pnl'].sum() / total_pnl * 100
        
        print(f"Top 1 market: {top1_share:.1f}% of total PnL")
        print(f"Top 3 markets: {top3_share:.1f}% of total PnL")
        print(f"Top 5 markets: {top5_share:.1f}% of total PnL")
        
        if top1_share > 50:
            print("\nWARNING: Single market dominates PnL - edge is not robust")
        elif top3_share > 80:
            print("\nWARNING: Top 3 markets dominate PnL - edge may not be robust")
    
    # Leave-one-out test
    print("\n" + "=" * 60)
    print("Leave-One-Out Robustness Test")
    print("=" * 60)
    
    top_markets = market_pnl.head(5)['market_id'].tolist()
    loo_results = run_leave_one_out(df, strategy, top_markets)
    
    print(f"\nBaseline: t={loo_results['baseline']['t_stat']:.2f}, PnL=${loo_results['baseline']['total_pnl']:.2f}")
    print("\nRemoving each top market:")
    
    for loo in loo_results['leave_one_out']:
        if 'error' in loo:
            print(f"  {loo['removed_market'][:25]:25s}: ERROR - {loo['error']}")
        else:
            print(f"  {loo['removed_market'][:25]:25s}: t={loo['t_stat']:.2f} (change: {loo['t_stat_change']:+.2f}, {loo['t_stat_change_pct']:+.1f}%)")
    
    # Check for fragility
    max_drop = max([abs(loo.get('t_stat_change_pct', 0)) for loo in loo_results['leave_one_out']])
    
    if max_drop > 50:
        print(f"\nWARNING: Removing one market causes >{max_drop:.0f}% t-stat change - strategy is fragile")
    
    # Summary
    print("\n" + "=" * 70)
    print("MARKET CONTRIBUTION SUMMARY")
    print("=" * 70)
    
    summary = {
        'n_markets_traded': len(market_pnl),
        'n_positive': n_positive,
        'n_negative': n_negative,
        'hit_rate_markets': n_positive / len(market_pnl) * 100 if len(market_pnl) > 0 else 0,
        'top1_share': top1_share if total_pnl > 0 else 0,
        'top3_share': top3_share if total_pnl > 0 else 0,
        'max_loo_drop_pct': max_drop,
    }
    
    print(f"\n1. Market hit rate: {summary['hit_rate_markets']:.1f}%")
    print(f"2. Top 3 concentration: {summary['top3_share']:.1f}%")
    print(f"3. Max LOO t-stat drop: {summary['max_loo_drop_pct']:.1f}%")
    
    # Save results
    all_results = {
        'market_pnl': market_pnl.to_dict('records'),
        'loo_results': loo_results,
        'summary': summary,
    }
    
    with open(OUTPUT_DIR / 'market_contribution_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    market_pnl.to_csv(OUTPUT_DIR / 'per_market_pnl.csv', index=False)
    
    print(f"\nResults saved to market_contribution_results.json and per_market_pnl.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()

