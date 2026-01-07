#!/usr/bin/env python3
"""
Step 3: Enhanced Placebo Suite

Run comprehensive placebo tests to understand Strategy B's edge:
1. Multi-shift grid test (+5s to +120s)
2. Market-level permutation test
3. CL sign-flip test
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest.data_loader import load_eth_markets, add_derived_columns
from scripts.backtest.backtest_engine import run_backtest, ExecutionConfig
from scripts.backtest.strategies import LateDirectionalTakerStrategy

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "results"


def run_multi_shift_placebo(
    df: pd.DataFrame,
    strategy,
    shifts: List[int]
) -> pd.DataFrame:
    """
    Run placebo tests with multiple shift values.
    
    Args:
        df: Market data
        strategy: Strategy instance
        shifts: List of shift values (seconds)
        
    Returns:
        DataFrame with results for each shift
    """
    results = []
    
    for shift in shifts:
        print(f"  Testing shift = {shift}s...", end=" ")
        
        # Create shifted dataset
        df_shifted = df.copy()
        
        if shift > 0:
            # Shift CL columns to be staler
            cl_cols = ['cl_mid', 'cl_bid', 'cl_ask', 'delta', 'delta_bps']
            for col in cl_cols:
                if col in df_shifted.columns:
                    df_shifted[col] = df_shifted.groupby('market_id')[col].shift(shift)
            
            # Drop NaN rows
            df_shifted = df_shifted.dropna(subset=['cl_mid'])
            
            # Recalculate derived columns
            df_shifted = add_derived_columns(df_shifted)
        
        # Run backtest
        config = ExecutionConfig()
        try:
            result = run_backtest(df_shifted, strategy, config)
            metrics = result['metrics']
            
            results.append({
                'shift_seconds': shift,
                'n_trades': metrics['n_trades'],
                'n_markets': metrics['n_markets'],
                'total_pnl': metrics['total_pnl'],
                'mean_pnl_per_market': metrics['mean_pnl_per_market'],
                't_stat': metrics['t_stat'],
                'hit_rate': metrics['hit_rate_per_trade'],
            })
            print(f"t={metrics['t_stat']:.2f}, PnL=${metrics['total_pnl']:.2f}")
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'shift_seconds': shift,
                'error': str(e)
            })
    
    return pd.DataFrame(results)


def run_permutation_test(
    df: pd.DataFrame,
    strategy,
    n_permutations: int = 100,
    seed: int = 42
) -> Dict:
    """
    Market-level permutation test.
    
    Randomly shuffle time within each market to destroy temporal structure,
    then run the strategy. This tests if the edge comes from temporal patterns.
    
    Args:
        df: Market data
        strategy: Strategy instance
        n_permutations: Number of permutation iterations
        seed: Random seed
        
    Returns:
        Dict with permutation test results
    """
    np.random.seed(seed)
    
    # Get original t-stat
    config = ExecutionConfig()
    original_result = run_backtest(df, strategy, config)
    original_t = original_result['metrics']['t_stat']
    
    print(f"\n  Original t-stat: {original_t:.3f}")
    print(f"  Running {n_permutations} permutations...")
    
    perm_t_stats = []
    
    for i in range(n_permutations):
        # Create permuted dataset
        df_perm = df.copy()
        
        # Shuffle time within each market (destroys temporal order)
        for market_id in df_perm['market_id'].unique():
            mask = df_perm['market_id'] == market_id
            market_indices = df_perm[mask].index
            
            # Shuffle the CL-related columns
            cl_cols = ['cl_mid', 'cl_bid', 'cl_ask', 'delta', 'delta_bps']
            for col in cl_cols:
                if col in df_perm.columns:
                    shuffled_values = df_perm.loc[market_indices, col].values.copy()
                    np.random.shuffle(shuffled_values)
                    df_perm.loc[market_indices, col] = shuffled_values
        
        # Recalculate derived columns
        df_perm = add_derived_columns(df_perm)
        
        # Run backtest
        try:
            result = run_backtest(df_perm, strategy, config)
            perm_t = result['metrics']['t_stat']
            perm_t_stats.append(perm_t)
        except:
            pass
        
        if (i + 1) % 20 == 0:
            print(f"    Completed {i + 1}/{n_permutations}")
    
    # Compute p-value
    perm_t_stats = np.array(perm_t_stats)
    p_value = (perm_t_stats >= original_t).mean()
    
    return {
        'original_t_stat': original_t,
        'permutation_mean': perm_t_stats.mean(),
        'permutation_std': perm_t_stats.std(),
        'permutation_max': perm_t_stats.max(),
        'p_value': p_value,
        'n_permutations': len(perm_t_stats),
        'permutation_distribution': perm_t_stats.tolist()
    }


def run_sign_flip_test(
    df: pd.DataFrame,
    strategy
) -> Dict:
    """
    Test if flipping the sign of delta_bps affects the strategy.
    
    If Strategy B exploits CL direction, flipping should invert signals.
    If it exploits PM patterns, flipping might not matter.
    
    Args:
        df: Market data
        strategy: Strategy instance
        
    Returns:
        Dict with sign-flip test results
    """
    print("\n  Running sign-flip test...")
    
    # Get original result
    config = ExecutionConfig()
    original_result = run_backtest(df, strategy, config)
    original_t = original_result['metrics']['t_stat']
    original_pnl = original_result['metrics']['total_pnl']
    
    # Create sign-flipped dataset
    df_flipped = df.copy()
    df_flipped['delta_bps'] = -df_flipped['delta_bps']
    df_flipped['delta'] = -df_flipped['delta']
    
    # Also flip cl_mid relative to K (conceptually)
    # This makes CL appear to be on opposite side of strike
    df_flipped = add_derived_columns(df_flipped)
    
    # Run backtest
    flipped_result = run_backtest(df_flipped, strategy, config)
    flipped_t = flipped_result['metrics']['t_stat']
    flipped_pnl = flipped_result['metrics']['total_pnl']
    
    return {
        'original_t_stat': original_t,
        'original_pnl': original_pnl,
        'flipped_t_stat': flipped_t,
        'flipped_pnl': flipped_pnl,
        't_stat_change': flipped_t - original_t,
        'pnl_change': flipped_pnl - original_pnl,
    }


def main():
    print("=" * 70)
    print("STEP 3: ENHANCED PLACEBO SUITE")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    df, market_info = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    # Create Strategy B with best parameters
    strategy = LateDirectionalTakerStrategy(
        tau_max=420,
        delta_threshold_bps=10,
        hold_seconds=120,
    )
    
    print(f"Testing: {strategy.name}")
    print(f"Markets: {df['market_id'].nunique()}")
    
    # Test 1: Multi-shift grid
    print("\n" + "=" * 60)
    print("TEST 1: Multi-Shift Placebo Grid")
    print("=" * 60)
    
    shifts = [0, 5, 15, 30, 60, 120]
    shift_results = run_multi_shift_placebo(df, strategy, shifts)
    
    print("\nResults:")
    print(shift_results[['shift_seconds', 't_stat', 'total_pnl', 'n_trades']].to_string(index=False))
    
    # Calculate decay rate
    if 't_stat' in shift_results.columns and len(shift_results) > 1:
        t0 = shift_results[shift_results['shift_seconds'] == 0]['t_stat'].values[0]
        t30 = shift_results[shift_results['shift_seconds'] == 30]['t_stat'].values[0]
        decay = (t0 - t30) / t0 * 100
        print(f"\nt-stat decay from 0s to 30s: {decay:.1f}%")
        
        if decay < 20:
            print("WARNING: Edge decays slowly - suggests persistence strategy, not lead-lag")
    
    # Save shift results
    shift_results.to_csv(OUTPUT_DIR / 'multi_shift_placebo_results.csv', index=False)
    
    # Test 2: Permutation test (reduced iterations for speed)
    print("\n" + "=" * 60)
    print("TEST 2: Market-Level Permutation Test")
    print("=" * 60)
    
    perm_results = run_permutation_test(df, strategy, n_permutations=50, seed=42)
    
    print(f"\n  Original t-stat: {perm_results['original_t_stat']:.3f}")
    print(f"  Permutation mean: {perm_results['permutation_mean']:.3f}")
    print(f"  Permutation std: {perm_results['permutation_std']:.3f}")
    print(f"  Permutation max: {perm_results['permutation_max']:.3f}")
    print(f"  p-value: {perm_results['p_value']:.4f}")
    
    if perm_results['p_value'] < 0.05:
        print("\n  RESULT: Strategy exploits temporal structure (p < 0.05)")
    else:
        print("\n  RESULT: Edge may not require temporal structure")
    
    # Test 3: Sign-flip test
    print("\n" + "=" * 60)
    print("TEST 3: Delta Sign-Flip Test")
    print("=" * 60)
    
    flip_results = run_sign_flip_test(df, strategy)
    
    print(f"\n  Original: t={flip_results['original_t_stat']:.2f}, PnL=${flip_results['original_pnl']:.2f}")
    print(f"  Flipped:  t={flip_results['flipped_t_stat']:.2f}, PnL=${flip_results['flipped_pnl']:.2f}")
    
    if abs(flip_results['flipped_t_stat']) > 1.5:
        print("\n  RESULT: Sign-flip creates opposite/similar edge - direction matters")
    else:
        print("\n  RESULT: Sign-flip destroys edge - strategy is directional")
    
    # Summary
    print("\n" + "=" * 70)
    print("PLACEBO SUITE SUMMARY")
    print("=" * 70)
    
    summary = {
        'shift_decay_pct': (shift_results[shift_results['shift_seconds'] == 0]['t_stat'].values[0] - 
                          shift_results[shift_results['shift_seconds'] == 30]['t_stat'].values[0]) / 
                          shift_results[shift_results['shift_seconds'] == 0]['t_stat'].values[0] * 100,
        'permutation_p_value': perm_results['p_value'],
        'sign_flip_t_change': flip_results['t_stat_change'],
    }
    
    print(f"\n1. Shift Decay (0s -> 30s): {summary['shift_decay_pct']:.1f}%")
    if summary['shift_decay_pct'] < 20:
        print("   -> LOW DECAY: Edge persists with stale data (persistence strategy)")
    else:
        print("   -> HIGH DECAY: Edge requires fresh data (lead-lag strategy)")
    
    print(f"\n2. Permutation p-value: {summary['permutation_p_value']:.4f}")
    if summary['permutation_p_value'] < 0.05:
        print("   -> SIGNIFICANT: Strategy exploits temporal patterns")
    else:
        print("   -> NOT SIGNIFICANT: Edge may be spurious")
    
    print(f"\n3. Sign-flip t-stat change: {summary['sign_flip_t_change']:.2f}")
    if abs(flip_results['flipped_t_stat']) > 1.5:
        print("   -> Flipping creates edge: Strategy is symmetric to direction")
    else:
        print("   -> Flipping destroys edge: Strategy is directional")
    
    # Save all results
    all_results = {
        'shift_results': shift_results.to_dict('records'),
        'permutation_results': {k: v for k, v in perm_results.items() if k != 'permutation_distribution'},
        'sign_flip_results': flip_results,
        'summary': summary,
    }
    
    with open(OUTPUT_DIR / 'placebo_suite_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to placebo_suite_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()

