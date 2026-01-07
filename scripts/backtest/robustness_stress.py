#!/usr/bin/env python3
"""
Phase 6.3: Robustness Tests (Sample Size Independence)

Tests to ensure the strategy isn't driven by a few markets or outlier trades:
- Leave-One-Market-Out (LOMO) analysis
- Trade-level influence stress (winsorization)

These tests identify if edge is fragile (driven by 1-2 markets) or
robust (distributed across markets).
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest.data_loader import load_eth_markets, add_derived_columns, get_train_test_split
from scripts.backtest.fair_value import BinnedFairValueModel
from scripts.backtest.strategies import MispricingBasedStrategy
from scripts.backtest.backtest_engine import run_backtest, ExecutionConfig


# ==============================================================================
# 6.3.1: LEAVE-ONE-MARKET-OUT (LOMO) ANALYSIS
# ==============================================================================

def run_lomo_analysis(
    test_df: pd.DataFrame,
    fair_value_model,
    strategy_params: Dict,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Leave-One-Market-Out analysis on test set.
    
    Compute test performance N times (where N = number of test markets),
    each time dropping one test market. Reports min/median/max t-stat
    to identify if edge is driven by 1-2 markets.
    
    Args:
        test_df: Test data
        fair_value_model: Fitted fair value model
        strategy_params: Strategy parameters
        verbose: Print progress
        
    Returns:
        Dict with:
        - per_market_results: t-stat when each market is dropped
        - min_t_stat: Worst-case t-stat
        - median_t_stat: Typical t-stat
        - max_t_stat: Best-case t-stat
        - vulnerable_markets: Markets whose removal significantly impacts edge
    """
    if verbose:
        print("\nRunning Leave-One-Market-Out (LOMO) Analysis...")
    
    market_ids = test_df['market_id'].unique().tolist()
    n_markets = len(market_ids)
    
    if verbose:
        print(f"  Test set has {n_markets} markets")
    
    # First get baseline (all markets)
    strategy = MispricingBasedStrategy(
        fair_value_model=fair_value_model,
        **strategy_params
    )
    baseline_result = run_backtest(test_df, strategy, ExecutionConfig(), verbose=False)
    baseline_t = baseline_result['metrics']['t_stat']
    baseline_pnl = baseline_result['metrics']['total_pnl']
    
    if verbose:
        print(f"  Baseline: t={baseline_t:.2f}, PnL=${baseline_pnl:.2f}")
    
    # Run LOMO
    lomo_results = []
    
    for i, market_to_drop in enumerate(market_ids):
        # Filter out this market
        df_subset = test_df[test_df['market_id'] != market_to_drop]
        
        # Run strategy on subset
        result = run_backtest(df_subset, strategy, ExecutionConfig(), verbose=False)
        metrics = result['metrics']
        
        lomo_results.append({
            'dropped_market': market_to_drop,
            'n_markets': metrics['n_markets'],
            'n_trades': metrics['n_trades'],
            'total_pnl': metrics['total_pnl'],
            't_stat': metrics['t_stat'],
            't_stat_change': baseline_t - metrics['t_stat'],
        })
        
        if verbose and (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{n_markets} LOMO runs")
    
    lomo_df = pd.DataFrame(lomo_results)
    
    # Identify vulnerable markets (dropping them significantly impacts edge)
    t_stats = lomo_df['t_stat'].values
    t_stat_changes = lomo_df['t_stat_change'].values
    
    # Market is "vulnerable" if dropping it changes t-stat by more than 0.5
    vulnerable_threshold = 0.5
    vulnerable_markets = lomo_df[lomo_df['t_stat_change'].abs() > vulnerable_threshold][
        ['dropped_market', 't_stat', 't_stat_change']
    ].to_dict('records')
    
    # Markets that HELP the edge (dropping them decreases t-stat)
    helpful_markets = lomo_df.nsmallest(3, 't_stat')[
        ['dropped_market', 't_stat', 't_stat_change']
    ].to_dict('records')
    
    # Markets that HURT the edge (dropping them increases t-stat)
    hurting_markets = lomo_df.nlargest(3, 't_stat')[
        ['dropped_market', 't_stat', 't_stat_change']
    ].to_dict('records')
    
    results = {
        'n_markets': n_markets,
        'baseline': {
            't_stat': baseline_t,
            'total_pnl': baseline_pnl,
        },
        'lomo_stats': {
            'min_t_stat': float(t_stats.min()),
            'median_t_stat': float(np.median(t_stats)),
            'max_t_stat': float(t_stats.max()),
            'mean_t_stat': float(t_stats.mean()),
            'std_t_stat': float(t_stats.std()),
        },
        'vulnerable_markets': vulnerable_markets,
        'helpful_markets': helpful_markets,  # Dropping hurts edge
        'hurting_markets': hurting_markets,  # Dropping helps edge
        'per_market_results': lomo_df.to_dict('records'),
        'robustness_score': 1 - (t_stats.std() / max(baseline_t, 0.01)),  # Lower variance = more robust
    }
    
    if verbose:
        print(f"\n  LOMO Results:")
        print(f"    Min t-stat: {results['lomo_stats']['min_t_stat']:.2f}")
        print(f"    Median t-stat: {results['lomo_stats']['median_t_stat']:.2f}")
        print(f"    Max t-stat: {results['lomo_stats']['max_t_stat']:.2f}")
        print(f"    Vulnerable markets: {len(vulnerable_markets)}")
        
        if vulnerable_markets:
            print(f"    Markets whose removal significantly changes edge:")
            for m in vulnerable_markets:
                print(f"      {m['dropped_market']}: t-stat change = {m['t_stat_change']:.2f}")
    
    return results


# ==============================================================================
# 6.3.2: TRADE-LEVEL INFLUENCE STRESS (WINSORIZATION)
# ==============================================================================

def run_winsorization_analysis(
    test_df: pd.DataFrame,
    fair_value_model,
    strategy_params: Dict,
    percentiles: List[float] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test if edge is driven by rare tail events.
    
    Winsorize per-trade PnL at various percentiles and re-compute metrics.
    If edge disappears with winsorization, it's driven by outliers.
    
    Args:
        test_df: Test data
        fair_value_model: Fitted fair value model
        strategy_params: Strategy parameters
        percentiles: Percentiles to cap at (default [90, 95, 99])
        verbose: Print progress
        
    Returns:
        Dict with results at each percentile cap
    """
    percentiles = percentiles or [90, 95, 99, 100]
    
    if verbose:
        print("\nRunning Winsorization Analysis...")
    
    # Run baseline strategy to get trades
    strategy = MispricingBasedStrategy(
        fair_value_model=fair_value_model,
        **strategy_params
    )
    result = run_backtest(test_df, strategy, ExecutionConfig(), verbose=False)
    trades = result['trades']
    baseline_metrics = result['metrics']
    
    if len(trades) == 0:
        return {'error': 'No trades to analyze'}
    
    if verbose:
        print(f"  Baseline: {len(trades)} trades, t={baseline_metrics['t_stat']:.2f}")
    
    # Extract trade PnLs
    trade_pnls = np.array([t['pnl'] for t in trades])
    market_ids = [t['market_id'] for t in trades]
    
    # Get unique markets
    unique_markets = list(set(market_ids))
    
    results = {
        'baseline': {
            'n_trades': len(trades),
            't_stat': baseline_metrics['t_stat'],
            'total_pnl': baseline_metrics['total_pnl'],
            'mean_pnl_per_trade': trade_pnls.mean(),
            'max_pnl': trade_pnls.max(),
            'min_pnl': trade_pnls.min(),
        },
        'percentile_results': []
    }
    
    for pct in percentiles:
        if pct >= 100:
            # No winsorization
            pnls_capped = trade_pnls.copy()
            cap_value = trade_pnls.max()
        else:
            cap_value = np.percentile(trade_pnls, pct)
            pnls_capped = np.minimum(trade_pnls, cap_value)
        
        # Also cap losses symmetrically
        floor_value = np.percentile(trade_pnls, 100 - pct)
        pnls_capped = np.maximum(pnls_capped, floor_value)
        
        # Recompute metrics with capped PnLs
        market_pnls = {}
        for i, (market_id, pnl) in enumerate(zip(market_ids, pnls_capped)):
            if market_id not in market_pnls:
                market_pnls[market_id] = 0
            market_pnls[market_id] += pnl
        
        total_pnl = sum(market_pnls.values())
        mean_pnl = np.mean(list(market_pnls.values()))
        std_pnl = np.std(list(market_pnls.values()), ddof=1) if len(market_pnls) > 1 else 1.0
        n_markets = len(market_pnls)
        t_stat = mean_pnl / (std_pnl / np.sqrt(n_markets)) if std_pnl > 0 else 0
        
        pct_result = {
            'percentile': pct,
            'cap_value': cap_value,
            'floor_value': floor_value,
            'total_pnl': total_pnl,
            'mean_pnl_per_market': mean_pnl,
            't_stat': t_stat,
            't_stat_change': baseline_metrics['t_stat'] - t_stat,
            'pnl_change_pct': (total_pnl - baseline_metrics['total_pnl']) / abs(baseline_metrics['total_pnl']) * 100 if baseline_metrics['total_pnl'] != 0 else 0,
        }
        
        results['percentile_results'].append(pct_result)
        
        if verbose:
            print(f"  Percentile {pct}: t={t_stat:.2f}, PnL=${total_pnl:.2f} "
                  f"(cap at ${cap_value:.3f})")
    
    # Assess robustness
    pct_95_result = next((r for r in results['percentile_results'] if r['percentile'] == 95), None)
    if pct_95_result:
        results['robust_to_95th'] = pct_95_result['t_stat'] > 1.5
        results['t_stat_at_95th'] = pct_95_result['t_stat']
    
    return results


# ==============================================================================
# COMBINED ROBUSTNESS SUITE
# ==============================================================================

def run_robustness_stress_suite(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    strategy_params: Dict,
    output_dir: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete robustness stress test suite.
    
    Args:
        test_df: Test data
        train_df: Training data
        strategy_params: Strategy parameters
        output_dir: Directory to save results
        verbose: Print progress
        
    Returns:
        Dict with all robustness test results
    """
    print("\n" + "="*70)
    print("PHASE 6.3: ROBUSTNESS STRESS TESTS")
    print("="*70)
    
    # Fit model
    model = BinnedFairValueModel(sample_every=5)
    model.fit(train_df)
    
    results = {}
    
    # 6.3.1: LOMO Analysis
    print("\n--- 6.3.1: Leave-One-Market-Out Analysis ---")
    lomo_results = run_lomo_analysis(
        test_df, model, strategy_params, verbose=verbose
    )
    results['lomo'] = lomo_results
    
    # 6.3.2: Winsorization Analysis
    print("\n--- 6.3.2: Trade-Level Influence (Winsorization) ---")
    winsor_results = run_winsorization_analysis(
        test_df, model, strategy_params,
        percentiles=[80, 90, 95, 99, 100],
        verbose=verbose
    )
    results['winsorization'] = winsor_results
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        with open(output_dir / 'robustness_stress_results.json', 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)
        
        # Save LOMO details
        if 'per_market_results' in lomo_results:
            lomo_df = pd.DataFrame(lomo_results['per_market_results'])
            lomo_df.to_csv(output_dir / 'lomo_per_market.csv', index=False)
        
        print(f"\nResults saved to {output_dir}")
    
    # Summary
    print("\n" + "="*70)
    print("ROBUSTNESS STRESS TEST SUMMARY")
    print("="*70)
    
    print("\nLOMO Analysis:")
    print(f"  Min t-stat: {lomo_results['lomo_stats']['min_t_stat']:.2f}")
    print(f"  Median t-stat: {lomo_results['lomo_stats']['median_t_stat']:.2f}")
    print(f"  Max t-stat: {lomo_results['lomo_stats']['max_t_stat']:.2f}")
    print(f"  Vulnerable markets: {len(lomo_results['vulnerable_markets'])}")
    
    if lomo_results['lomo_stats']['min_t_stat'] > 1.0:
        print("  [PASS] Edge survives dropping any single market")
    else:
        print("  [WARN] Edge fragile - dropping some markets kills it")
    
    print("\nWinsorization Analysis:")
    if 'percentile_results' in winsor_results:
        for r in winsor_results['percentile_results']:
            status = "PASS" if r['t_stat'] > 1.5 else "WARN"
            print(f"  {r['percentile']}th pct: t={r['t_stat']:.2f} [{status}]")
    
    if winsor_results.get('robust_to_95th'):
        print("  [PASS] Edge survives 95th percentile winsorization")
    else:
        print("  [WARN] Edge may be driven by outlier trades")
    
    return results


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run robustness stress tests')
    parser.add_argument('--output-dir', type=str,
                       default=str(PROJECT_ROOT / 'data_v2' / 'backtest_results' / 'stress_tests'))
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    df, _ = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    train_df, test_df, _, _ = get_train_test_split(df, train_frac=0.7)
    
    # Default strategy params
    strategy_params = {
        'buffer': 0.02,
        'tau_max': 420,
        'min_tau': 0,
        'cooldown': 30,
        'exit_rule': 'expiry',
    }
    
    # Run suite
    results = run_robustness_stress_suite(
        test_df=test_df,
        train_df=train_df,
        strategy_params=strategy_params,
        output_dir=Path(args.output_dir)
    )

