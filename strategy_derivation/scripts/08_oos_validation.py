#!/usr/bin/env python3
"""
Step 4: Out-of-Sample Validation

1. Train/test split by market (chronological)
2. Bootstrap reality check for Strategy B
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


def train_test_split(df: pd.DataFrame, train_ratio: float = 0.67) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test by market (chronological order).
    
    Args:
        df: Market data with market_id column
        train_ratio: Fraction of markets for training
        
    Returns:
        (train_df, test_df)
    """
    # Get unique markets sorted by market_start
    market_starts = df.groupby('market_id')['market_start'].first().sort_values()
    markets_sorted = market_starts.index.tolist()
    
    n_train = int(len(markets_sorted) * train_ratio)
    train_markets = set(markets_sorted[:n_train])
    test_markets = set(markets_sorted[n_train:])
    
    train_df = df[df['market_id'].isin(train_markets)].copy()
    test_df = df[df['market_id'].isin(test_markets)].copy()
    
    return train_df, test_df


def run_train_test_experiment(df: pd.DataFrame) -> Dict:
    """
    Run train/test split experiment for Strategy B.
    
    Tests if optimal parameters on train data work on test data.
    """
    print("\n--- Train/Test Split Experiment ---")
    
    # Split data
    train_df, test_df = train_test_split(df, train_ratio=0.67)
    
    print(f"Train markets: {train_df['market_id'].nunique()}")
    print(f"Test markets: {test_df['market_id'].nunique()}")
    
    # Parameter grid (focused on best region from original sweep)
    param_grid = [
        {'tau_max': 300, 'delta_threshold_bps': 10, 'hold_seconds': 120},
        {'tau_max': 420, 'delta_threshold_bps': 10, 'hold_seconds': 120},
        {'tau_max': 300, 'delta_threshold_bps': 20, 'hold_seconds': 120},
        {'tau_max': 420, 'delta_threshold_bps': 20, 'hold_seconds': 120},
        {'tau_max': 300, 'delta_threshold_bps': 10, 'hold_seconds': 180},
        {'tau_max': 420, 'delta_threshold_bps': 10, 'hold_seconds': 180},
    ]
    
    # Find best params on train set
    print("\nOptimizing on train set...")
    train_results = []
    config = ExecutionConfig()
    
    for params in param_grid:
        strategy = LateDirectionalTakerStrategy(**params)
        try:
            result = run_backtest(train_df, strategy, config)
            metrics = result['metrics']
            train_results.append({
                **params,
                't_stat': metrics['t_stat'],
                'total_pnl': metrics['total_pnl'],
            })
        except Exception as e:
            print(f"  Error with {params}: {e}")
    
    train_results_df = pd.DataFrame(train_results)
    
    if len(train_results_df) == 0:
        return {'error': 'No valid train results'}
    
    best_idx = train_results_df['t_stat'].idxmax()
    best_params = {
        'tau_max': int(train_results_df.loc[best_idx, 'tau_max']),
        'delta_threshold_bps': float(train_results_df.loc[best_idx, 'delta_threshold_bps']),
        'hold_seconds': int(train_results_df.loc[best_idx, 'hold_seconds']),
    }
    best_train_t = train_results_df.loc[best_idx, 't_stat']
    
    print(f"Best train params: {best_params}")
    print(f"Best train t-stat: {best_train_t:.2f}")
    
    # Test on held-out set
    print("\nTesting on held-out set...")
    strategy = LateDirectionalTakerStrategy(**best_params)
    
    try:
        test_result = run_backtest(test_df, strategy, config)
        test_metrics = test_result['metrics']
        test_t = test_metrics['t_stat']
        test_pnl = test_metrics['total_pnl']
    except Exception as e:
        return {'error': f'Test failed: {e}'}
    
    print(f"Test t-stat: {test_t:.2f}")
    print(f"Test PnL: ${test_pnl:.2f}")
    
    # Compute degradation
    degradation = (best_train_t - test_t) / best_train_t * 100 if best_train_t > 0 else float('nan')
    
    return {
        'train_n_markets': train_df['market_id'].nunique(),
        'test_n_markets': test_df['market_id'].nunique(),
        'best_params': best_params,
        'train_t_stat': best_train_t,
        'test_t_stat': test_t,
        'test_pnl': test_pnl,
        'degradation_pct': degradation,
        'train_results': train_results,
    }


def run_bootstrap_test(
    df: pd.DataFrame,
    strategy,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict:
    """
    Bootstrap reality check for Strategy B.
    
    Resample markets with replacement, compute t-stat for each bootstrap.
    Report 95% CI and bootstrap p-value.
    """
    print("\n--- Bootstrap Reality Check ---")
    
    np.random.seed(seed)
    
    config = ExecutionConfig()
    markets = df['market_id'].unique()
    
    # Get original result
    original_result = run_backtest(df, strategy, config)
    original_t = original_result['metrics']['t_stat']
    original_pnl = original_result['metrics']['total_pnl']
    
    print(f"Original: t={original_t:.2f}, PnL=${original_pnl:.2f}")
    print(f"Running {n_bootstraps} bootstraps...")
    
    bootstrap_t_stats = []
    bootstrap_pnls = []
    
    for i in range(n_bootstraps):
        # Resample markets with replacement
        sampled_markets = np.random.choice(markets, size=len(markets), replace=True)
        
        # Build bootstrapped dataset
        # For each sampled market, add a unique suffix to avoid duplicate market_ids
        dfs = []
        for j, mid in enumerate(sampled_markets):
            mdf = df[df['market_id'] == mid].copy()
            mdf['market_id'] = f"{mid}_boot_{j}"  # Make unique
            dfs.append(mdf)
        
        df_boot = pd.concat(dfs, ignore_index=True)
        
        try:
            result = run_backtest(df_boot, strategy, config)
            bootstrap_t_stats.append(result['metrics']['t_stat'])
            bootstrap_pnls.append(result['metrics']['total_pnl'])
        except:
            pass
        
        if (i + 1) % 200 == 0:
            print(f"  Completed {i + 1}/{n_bootstraps}")
    
    bootstrap_t_stats = np.array(bootstrap_t_stats)
    bootstrap_pnls = np.array(bootstrap_pnls)
    
    # Compute statistics
    t_mean = bootstrap_t_stats.mean()
    t_std = bootstrap_t_stats.std()
    t_ci_low = np.percentile(bootstrap_t_stats, 2.5)
    t_ci_high = np.percentile(bootstrap_t_stats, 97.5)
    
    pnl_mean = bootstrap_pnls.mean()
    pnl_std = bootstrap_pnls.std()
    pnl_ci_low = np.percentile(bootstrap_pnls, 2.5)
    pnl_ci_high = np.percentile(bootstrap_pnls, 97.5)
    
    # Bootstrap p-value (proportion of bootstraps with t <= 0)
    p_value = (bootstrap_t_stats <= 0).mean()
    
    print(f"\nBootstrap results (n={len(bootstrap_t_stats)}):")
    print(f"  t-stat: {t_mean:.2f} +/- {t_std:.2f} (95% CI: [{t_ci_low:.2f}, {t_ci_high:.2f}])")
    print(f"  PnL: ${pnl_mean:.2f} +/- ${pnl_std:.2f} (95% CI: [${pnl_ci_low:.2f}, ${pnl_ci_high:.2f}])")
    print(f"  Bootstrap p-value: {p_value:.4f}")
    
    return {
        'original_t_stat': original_t,
        'original_pnl': original_pnl,
        'bootstrap_n': len(bootstrap_t_stats),
        't_stat_mean': t_mean,
        't_stat_std': t_std,
        't_stat_ci_low': t_ci_low,
        't_stat_ci_high': t_ci_high,
        'pnl_mean': pnl_mean,
        'pnl_std': pnl_std,
        'pnl_ci_low': pnl_ci_low,
        'pnl_ci_high': pnl_ci_high,
        'p_value': p_value,
    }


def main():
    print("=" * 70)
    print("STEP 4: OUT-OF-SAMPLE VALIDATION")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    df, market_info = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    print(f"Total markets: {df['market_id'].nunique()}")
    
    # Test 1: Train/test split
    print("\n" + "=" * 60)
    print("TEST 1: Train/Test Split")
    print("=" * 60)
    
    train_test_results = run_train_test_experiment(df)
    
    print(f"\nTrain/test degradation: {train_test_results.get('degradation_pct', 'N/A'):.1f}%")
    
    if train_test_results.get('degradation_pct', 0) > 50:
        print("WARNING: High degradation - possible overfitting")
    
    # Test 2: Bootstrap
    print("\n" + "=" * 60)
    print("TEST 2: Bootstrap Reality Check")
    print("=" * 60)
    
    strategy = LateDirectionalTakerStrategy(
        tau_max=420,
        delta_threshold_bps=10,
        hold_seconds=120,
    )
    
    bootstrap_results = run_bootstrap_test(df, strategy, n_bootstraps=500, seed=42)
    
    # Summary
    print("\n" + "=" * 70)
    print("OOS VALIDATION SUMMARY")
    print("=" * 70)
    
    print(f"\n1. Train/Test Split:")
    print(f"   Train t-stat: {train_test_results.get('train_t_stat', 'N/A'):.2f}")
    print(f"   Test t-stat:  {train_test_results.get('test_t_stat', 'N/A'):.2f}")
    print(f"   Degradation:  {train_test_results.get('degradation_pct', 'N/A'):.1f}%")
    
    print(f"\n2. Bootstrap Reality Check:")
    print(f"   t-stat 95% CI: [{bootstrap_results['t_stat_ci_low']:.2f}, {bootstrap_results['t_stat_ci_high']:.2f}]")
    print(f"   Bootstrap p-value: {bootstrap_results['p_value']:.4f}")
    
    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    
    degradation = train_test_results.get('degradation_pct', 0)
    ci_low = bootstrap_results['t_stat_ci_low']
    
    if degradation < 30 and ci_low > 1.5:
        print("\nSTRONG: Low degradation + CI excludes weak edge")
        print("Strategy B appears robust to out-of-sample testing")
    elif degradation < 50 and ci_low > 0:
        print("\nMODERATE: Some degradation but CI still positive")
        print("Strategy B shows promise but needs more validation")
    else:
        print("\nWEAK: High degradation or CI includes zero")
        print("Strategy B may be overfit to in-sample data")
    
    # Save results
    all_results = {
        'train_test_results': train_test_results,
        'bootstrap_results': bootstrap_results,
    }
    
    with open(OUTPUT_DIR / 'oos_validation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to oos_validation_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()

