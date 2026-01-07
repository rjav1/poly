#!/usr/bin/env python3
"""
Step 6: Execution Realism Stress Tests

Test Strategy B under realistic execution conditions:
1. Slippage buffers (+1c entry, -1c exit)
2. Spread filters (only trade if spread < threshold)
3. Combined stress test
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


def run_slippage_test(
    df: pd.DataFrame,
    strategy,
    slippage_cents: List[float]
) -> pd.DataFrame:
    """
    Test strategy with various slippage levels.
    
    Simulates slippage by adjusting entry/exit prices.
    Slippage is applied symmetrically: +X on entry, -X on exit.
    """
    results = []
    config = ExecutionConfig()
    
    for slip in slippage_cents:
        slip_frac = slip / 100.0  # Convert cents to fraction
        
        # Create df with adjusted prices to simulate slippage
        df_slip = df.copy()
        
        # Worsen all ask prices by +slippage (more expensive to buy)
        df_slip['pm_up_best_ask'] = df_slip['pm_up_best_ask'] + slip_frac
        df_slip['pm_down_best_ask'] = df_slip['pm_down_best_ask'] + slip_frac
        
        # Worsen all bid prices by -slippage (less received when selling)
        df_slip['pm_up_best_bid'] = df_slip['pm_up_best_bid'] - slip_frac
        df_slip['pm_down_best_bid'] = df_slip['pm_down_best_bid'] - slip_frac
        
        # Recalculate derived columns
        df_slip = add_derived_columns(df_slip)
        
        try:
            result = run_backtest(df_slip, strategy, config)
            metrics = result['metrics']
            
            results.append({
                'slippage_cents': slip,
                'n_trades': metrics['n_trades'],
                'total_pnl': metrics['total_pnl'],
                't_stat': metrics['t_stat'],
                'mean_pnl_per_trade': metrics.get('mean_pnl_per_trade', 0),
            })
        except Exception as e:
            results.append({
                'slippage_cents': slip,
                'error': str(e),
            })
    
    return pd.DataFrame(results)


def run_spread_filter_test(
    df: pd.DataFrame,
    strategy,
    max_spreads: List[float]
) -> pd.DataFrame:
    """
    Test strategy with spread filters.
    Only take trades where spread is below threshold.
    """
    results = []
    
    for max_spread in max_spreads:
        # Filter data to only include rows with acceptable spread
        df_filtered = df.copy()
        
        # Use the up_spread and down_spread columns
        if 'up_spread' not in df_filtered.columns:
            df_filtered['up_spread'] = df_filtered['pm_up_best_ask'] - df_filtered['pm_up_best_bid']
        if 'down_spread' not in df_filtered.columns:
            df_filtered['down_spread'] = df_filtered['pm_down_best_ask'] - df_filtered['pm_down_best_bid']
        
        df_filtered['max_spread'] = df_filtered[['up_spread', 'down_spread']].max(axis=1)
        
        # Only keep rows where spread is acceptable
        df_filtered = df_filtered[df_filtered['max_spread'] <= max_spread].copy()
        
        if len(df_filtered) == 0:
            results.append({
                'max_spread_cents': max_spread * 100,
                'rows_remaining': 0,
                'pct_remaining': 0,
                'error': 'No data after filter',
            })
            continue
        
        pct_remaining = len(df_filtered) / len(df) * 100
        
        config = ExecutionConfig()
        
        try:
            result = run_backtest(df_filtered, strategy, config)
            metrics = result['metrics']
            
            results.append({
                'max_spread_cents': max_spread * 100,
                'rows_remaining': len(df_filtered),
                'pct_remaining': pct_remaining,
                'n_trades': metrics['n_trades'],
                'total_pnl': metrics['total_pnl'],
                't_stat': metrics['t_stat'],
            })
        except Exception as e:
            results.append({
                'max_spread_cents': max_spread * 100,
                'rows_remaining': len(df_filtered),
                'pct_remaining': pct_remaining,
                'error': str(e),
            })
    
    return pd.DataFrame(results)


def apply_slippage(df: pd.DataFrame, slip_frac: float) -> pd.DataFrame:
    """Apply slippage to orderbook prices."""
    df_slip = df.copy()
    df_slip['pm_up_best_ask'] = df_slip['pm_up_best_ask'] + slip_frac
    df_slip['pm_down_best_ask'] = df_slip['pm_down_best_ask'] + slip_frac
    df_slip['pm_up_best_bid'] = df_slip['pm_up_best_bid'] - slip_frac
    df_slip['pm_down_best_bid'] = df_slip['pm_down_best_bid'] - slip_frac
    return add_derived_columns(df_slip)


def run_combined_stress_test(df: pd.DataFrame, strategy) -> Dict:
    """
    Run a combined stress test with realistic conditions.
    """
    config = ExecutionConfig()
    
    # Get baseline
    base_result = run_backtest(df, strategy, config)
    base_t = base_result['metrics']['t_stat']
    base_pnl = base_result['metrics']['total_pnl']
    
    results = {
        'baseline': {
            't_stat': base_t,
            'total_pnl': base_pnl,
        },
        'stress_tests': []
    }
    
    # Test 1: 0.5c slippage
    df_slip = apply_slippage(df, 0.005)
    result = run_backtest(df_slip, strategy, config)
    results['stress_tests'].append({
        'name': '0.5c slippage',
        't_stat': result['metrics']['t_stat'],
        'total_pnl': result['metrics']['total_pnl'],
    })
    
    # Test 2: 1c slippage
    df_slip = apply_slippage(df, 0.01)
    result = run_backtest(df_slip, strategy, config)
    results['stress_tests'].append({
        'name': '1c slippage',
        't_stat': result['metrics']['t_stat'],
        'total_pnl': result['metrics']['total_pnl'],
    })
    
    # Test 3: 2c slippage
    df_slip = apply_slippage(df, 0.02)
    result = run_backtest(df_slip, strategy, config)
    results['stress_tests'].append({
        'name': '2c slippage',
        't_stat': result['metrics']['t_stat'],
        'total_pnl': result['metrics']['total_pnl'],
    })
    
    # Test 4: Spread filter < 5c only
    df_tight = df.copy()
    df_tight['up_spread'] = df_tight['pm_up_best_ask'] - df_tight['pm_up_best_bid']
    df_tight['down_spread'] = df_tight['pm_down_best_ask'] - df_tight['pm_down_best_bid']
    df_tight['max_spread'] = df_tight[['up_spread', 'down_spread']].max(axis=1)
    df_tight = df_tight[df_tight['max_spread'] <= 0.05].copy()
    
    if len(df_tight) > 0:
        result = run_backtest(df_tight, strategy, config)
        results['stress_tests'].append({
            'name': 'Spread < 5c only',
            't_stat': result['metrics']['t_stat'],
            'total_pnl': result['metrics']['total_pnl'],
            'rows_used': len(df_tight),
        })
    
    # Test 5: Combined (1c slippage + 5c spread filter)
    if len(df_tight) > 0:
        df_combined = apply_slippage(df_tight, 0.01)
        result = run_backtest(df_combined, strategy, config)
        results['stress_tests'].append({
            'name': '1c slip + 5c spread',
            't_stat': result['metrics']['t_stat'],
            'total_pnl': result['metrics']['total_pnl'],
        })
    
    return results


def main():
    print("=" * 70)
    print("STEP 6: EXECUTION REALISM STRESS TESTS")
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
    
    # Get baseline
    config_base = ExecutionConfig()
    base_result = run_backtest(df, strategy, config_base)
    base_t = base_result['metrics']['t_stat']
    base_pnl = base_result['metrics']['total_pnl']
    
    print(f"Baseline: t={base_t:.2f}, PnL=${base_pnl:.2f}")
    
    # Test 1: Slippage impact
    print("\n" + "=" * 60)
    print("TEST 1: Slippage Impact")
    print("=" * 60)
    
    slippages = [0, 0.5, 1.0, 1.5, 2.0, 3.0]
    slip_results = run_slippage_test(df, strategy, slippages)
    
    print("\nSlippage impact on Strategy B:")
    if 't_stat' in slip_results.columns:
        print(slip_results[['slippage_cents', 't_stat', 'total_pnl']].to_string(index=False))
    else:
        print("All slippage tests failed:")
        print(slip_results.to_string(index=False))
    
    # Find break-even slippage
    for i, row in slip_results.iterrows():
        if row.get('t_stat', 0) < 1.5:
            print(f"\nBreak-even slippage: ~{row['slippage_cents']}c (t-stat drops below 1.5)")
            break
    
    # Test 2: Spread filter impact
    print("\n" + "=" * 60)
    print("TEST 2: Spread Filter Impact")
    print("=" * 60)
    
    max_spreads = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 1.0]  # As fractions
    spread_results = run_spread_filter_test(df, strategy, max_spreads)
    
    print("\nSpread filter impact:")
    cols = ['max_spread_cents', 'pct_remaining', 't_stat', 'total_pnl']
    print(spread_results[[c for c in cols if c in spread_results.columns]].to_string(index=False))
    
    # Test 3: Combined stress
    print("\n" + "=" * 60)
    print("TEST 3: Combined Stress Test")
    print("=" * 60)
    
    combined_results = run_combined_stress_test(df, strategy)
    
    print(f"\nBaseline: t={combined_results['baseline']['t_stat']:.2f}, PnL=${combined_results['baseline']['total_pnl']:.2f}")
    print("\nStress scenarios:")
    for test in combined_results['stress_tests']:
        print(f"  {test['name']:25s}: t={test['t_stat']:.2f}, PnL=${test['total_pnl']:.2f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("EXECUTION STRESS SUMMARY")
    print("=" * 70)
    
    # Find slippage tolerance
    slip_at_t15 = None
    for i, row in slip_results.iterrows():
        if row.get('t_stat', 0) < 1.5 and slip_at_t15 is None:
            slip_at_t15 = slip_results.iloc[i-1]['slippage_cents'] if i > 0 else 0
    
    if slip_at_t15 is None:
        slip_at_t15 = slip_results.iloc[-1]['slippage_cents']
    
    print(f"\n1. Slippage tolerance: ~{slip_at_t15}c before t-stat < 1.5")
    
    # Average trade PnL
    avg_trade_pnl = base_result['metrics'].get('mean_pnl_per_trade', 0)
    print(f"2. Average PnL per trade: ${avg_trade_pnl:.3f}")
    print(f"   (Slippage must be < {avg_trade_pnl*100:.1f}c to remain profitable)")
    
    # Minimum spread regime
    tight_test = [t for t in combined_results['stress_tests'] if 'Spread' in t['name']]
    if tight_test:
        print(f"3. Performance in tight-spread regime: t={tight_test[0]['t_stat']:.2f}")
    
    # Final assessment
    print("\n" + "-" * 70)
    print("ASSESSMENT")
    print("-" * 70)
    
    if avg_trade_pnl > 0.01:  # > 1 cent per trade
        print("\nPOSITIVE: Average trade PnL > 1c, can absorb some slippage")
    else:
        print("\nWARNING: Average trade PnL < 1c, very sensitive to execution costs")
    
    if slip_at_t15 >= 1.0:
        print("POSITIVE: Strategy tolerates 1c+ slippage")
    else:
        print("WARNING: Strategy breaks with < 1c slippage - execution-sensitive")
    
    # Save results
    all_results = {
        'baseline': {'t_stat': base_t, 'total_pnl': base_pnl},
        'slippage_results': slip_results.to_dict('records'),
        'spread_results': spread_results.to_dict('records'),
        'combined_results': combined_results,
    }
    
    with open(OUTPUT_DIR / 'execution_stress_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to execution_stress_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()

