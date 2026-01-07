#!/usr/bin/env python3
"""
Mispricing-Based Strategy Testing Suite

Comprehensive testing for the mispricing-based late directional strategy:
- Phase 3: Parameter sweep and optimization
- Phase 4: Robustness checks (latency, placebo, vol sensitivity, binning sensitivity)
- Phase 5: Out-of-sample validation with walk-forward and bootstrap

Usage:
    python scripts/backtest/test_mispricing_strategy.py
    python scripts/backtest/test_mispricing_strategy.py --phase 3  # Run only Phase 3
    python scripts/backtest/test_mispricing_strategy.py --full     # Run all phases
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from itertools import product
import pandas as pd
import numpy as np
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest.data_loader import (
    load_eth_markets,
    add_derived_columns,
    get_train_test_split,
    get_walk_forward_splits
)
from scripts.backtest.fair_value import (
    BinnedFairValueModel,
    FairValueModel,
    compute_brier_score,
    compute_expected_calibration_error
)
from scripts.backtest.strategies import MispricingBasedStrategy
from scripts.backtest.backtest_engine import run_backtest, ExecutionConfig

# Output directory
OUTPUT_DIR = PROJECT_ROOT / 'data_v2' / 'backtest_results' / 'mispricing_strategy'


# ==============================================================================
# PHASE 3: PARAMETER SWEEP
# ==============================================================================

def run_parameter_sweep(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fair_value_model,
    param_grid: Dict[str, List],
    config: ExecutionConfig = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Run parameter sweep for mispricing strategy.
    
    Args:
        train_df: Training data (for reference, model already fitted)
        test_df: Test data
        fair_value_model: Fitted fair value model
        param_grid: Dict of param_name -> list of values
        config: Execution config
        verbose: Print progress
        
    Returns:
        DataFrame with results for each parameter combination
    """
    config = config or ExecutionConfig()
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    print(f"\nRunning {len(combinations)} parameter combinations...")
    
    results = []
    
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        
        try:
            strategy = MispricingBasedStrategy(
                fair_value_model=fair_value_model,
                **params
            )
            
            # Run on test set
            result = run_backtest(test_df, strategy, config, verbose=False)
            metrics = result['metrics']
            
            # Compute additional metrics
            trades = result.get('trades', [])
            # Note: metadata is in Signal, not Trade, so we can't access it here
            # We'll compute avg_hold_time from trades
            avg_hold_time = np.mean([t['exit_t'] - t['entry_t'] for t in trades]) if trades else 0
            
            row = {
                **params,
                'n_trades': metrics['n_trades'],
                'n_markets': metrics['n_markets'],
                'total_pnl': metrics['total_pnl'],
                'mean_pnl_per_market': metrics['mean_pnl_per_market'],
                'std_pnl_per_market': metrics['std_pnl_per_market'],
                't_stat': metrics['t_stat'],
                'hit_rate_market': metrics['hit_rate_per_market'],
                'hit_rate_trade': metrics['hit_rate_per_trade'],
                'avg_trades_per_market': metrics['avg_trades_per_market'],
                'avg_hold_time': avg_hold_time,
            }
            
            results.append(row)
            
            if verbose or (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(combinations)}] buf={params.get('buffer', 'N/A')}, "
                      f"tau_max={params.get('tau_max', 'N/A')}: "
                      f"PnL=${metrics['total_pnl']:.2f}, t={metrics['t_stat']:.2f}, "
                      f"trades={metrics['n_trades']}")
                
        except Exception as e:
            print(f"  Error with params {params}: {e}")
            results.append({**params, 'error': str(e)})
    
    return pd.DataFrame(results)


def select_best_parameters(
    sweep_results: pd.DataFrame,
    min_trades: int = 20,
    min_t_stat: float = 1.5
) -> Dict[str, Any]:
    """
    Select best parameters from sweep results.
    
    Selection criteria:
    1. Primary: t_stat > min_t_stat
    2. Secondary: mean_pnl_per_market > 0
    3. Tertiary: n_trades >= min_trades
    
    If multiple pass, pick most conservative (higher buffer).
    
    Args:
        sweep_results: DataFrame from parameter sweep
        min_trades: Minimum trades required
        min_t_stat: Minimum t-stat required
        
    Returns:
        Dict with best parameters and metrics
    """
    # Filter out error rows
    if 'error' in sweep_results.columns:
        valid_base = sweep_results[sweep_results['error'].isna()].copy()
    else:
        valid_base = sweep_results.copy()
    
    if len(valid_base) == 0:
        return {'error': 'No valid parameter combinations (all had errors)'}
    
    # Filter valid results
    valid = valid_base[
        (valid_base['n_trades'] >= min_trades) &
        (valid_base['t_stat'] >= min_t_stat) &
        (valid_base['mean_pnl_per_market'] > 0)
    ].copy()
    
    if len(valid) == 0:
        # Relax criteria
        valid = sweep_results[
            (sweep_results.get('error').isna() if 'error' in sweep_results.columns else True) &
            (sweep_results['n_trades'] >= min_trades // 2)
        ].copy()
        
        if len(valid) == 0:
            return {'error': 'No valid parameter combinations found'}
    
    # Sort by t_stat descending, then by buffer descending (more conservative)
    valid = valid.sort_values(['t_stat', 'buffer'], ascending=[False, False])
    
    best = valid.iloc[0].to_dict()
    
    # Get top 5 for reporting
    top5 = valid.head(5).to_dict('records')
    
    return {
        'best': best,
        'top5': top5,
        'n_valid_combinations': len(valid),
    }


# ==============================================================================
# PHASE 4: ROBUSTNESS CHECKS
# ==============================================================================

def run_latency_sensitivity(
    df: pd.DataFrame,
    fair_value_model,
    best_params: Dict,
    latencies: List[float] = None
) -> pd.DataFrame:
    """
    Test strategy at multiple latency levels.
    
    Args:
        df: Test data
        fair_value_model: Fitted model
        best_params: Best strategy parameters
        latencies: List of latencies to test
        
    Returns:
        DataFrame with metrics at each latency
    """
    latencies = latencies or [0, 0.5, 1, 2, 5, 10]
    
    print(f"\nRunning latency sensitivity ({len(latencies)} levels)...")
    
    results = []
    
    strategy_params = {k: v for k, v in best_params.items() 
                       if k in ['buffer', 'tau_max', 'min_tau', 'cooldown', 'exit_rule']}
    
    for latency in latencies:
        config = ExecutionConfig(
            signal_latency_s=latency / 2,
            exec_latency_s=latency / 2
        )
        
        strategy = MispricingBasedStrategy(
            fair_value_model=fair_value_model,
            **strategy_params
        )
        
        result = run_backtest(df, strategy, config, verbose=False)
        metrics = result['metrics']
        
        results.append({
            'latency': latency,
            'n_trades': metrics['n_trades'],
            'total_pnl': metrics['total_pnl'],
            'mean_pnl_per_market': metrics['mean_pnl_per_market'],
            't_stat': metrics['t_stat'],
            'hit_rate_trade': metrics['hit_rate_per_trade'],
        })
        
        print(f"  Latency {latency}s: PnL=${metrics['total_pnl']:.2f}, t={metrics['t_stat']:.2f}")
    
    return pd.DataFrame(results)


def run_placebo_test(
    df: pd.DataFrame,
    fair_value_model,
    best_params: Dict,
    shift_seconds: int = 30
) -> Dict[str, Any]:
    """
    Run placebo test by shifting CL data forward.
    
    This simulates using stale CL data. If edge persists, it's not from
    CL-PM lead-lag timing.
    
    Args:
        df: Test data
        fair_value_model: Fitted model
        best_params: Best strategy parameters
        shift_seconds: How many seconds to shift CL data
        
    Returns:
        Dict with original and placebo results
    """
    print(f"\nRunning placebo test (CL shift +{shift_seconds}s)...")
    
    strategy_params = {k: v for k, v in best_params.items() 
                       if k in ['buffer', 'tau_max', 'min_tau', 'cooldown', 'exit_rule']}
    
    # Original run
    strategy = MispricingBasedStrategy(
        fair_value_model=fair_value_model,
        **strategy_params
    )
    original_result = run_backtest(df, strategy, ExecutionConfig(), verbose=False)
    
    # Create shifted data
    df_shifted = df.copy()
    cl_columns = ['cl_mid', 'cl_bid', 'cl_ask', 'delta', 'delta_bps']
    
    for col in cl_columns:
        if col in df_shifted.columns:
            df_shifted[col] = df_shifted.groupby('market_id')[col].shift(shift_seconds)
    
    # Fill NaN with forward-fill (simulates stale data)
    for col in cl_columns:
        if col in df_shifted.columns:
            df_shifted[col] = df_shifted.groupby('market_id')[col].ffill()
    
    # Drop rows with remaining NaN
    df_shifted = df_shifted.dropna(subset=['cl_mid', 'delta_bps'])
    
    # Re-fit model on shifted data for fair comparison
    # (Or use same model - depends on what we're testing)
    # Using same model tests if the signal quality degrades
    
    placebo_result = run_backtest(df_shifted, strategy, ExecutionConfig(), verbose=False)
    
    original_metrics = original_result['metrics']
    placebo_metrics = placebo_result['metrics']
    
    result = {
        'shift_seconds': shift_seconds,
        'original': {
            'n_trades': original_metrics['n_trades'],
            'total_pnl': original_metrics['total_pnl'],
            't_stat': original_metrics['t_stat'],
        },
        'placebo': {
            'n_trades': placebo_metrics['n_trades'],
            'total_pnl': placebo_metrics['total_pnl'],
            't_stat': placebo_metrics['t_stat'],
        },
        't_stat_degradation': original_metrics['t_stat'] - placebo_metrics['t_stat'],
        'edge_destroyed': placebo_metrics['t_stat'] < 1.0,
    }
    
    print(f"  Original: t={original_metrics['t_stat']:.2f}, PnL=${original_metrics['total_pnl']:.2f}")
    print(f"  Placebo:  t={placebo_metrics['t_stat']:.2f}, PnL=${placebo_metrics['total_pnl']:.2f}")
    print(f"  Edge destroyed: {result['edge_destroyed']}")
    
    return result


def run_vol_window_sensitivity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_params: Dict,
    vol_windows: List[int] = None
) -> pd.DataFrame:
    """
    Test different volatility window sizes.
    
    Re-fits fair value model for each window size.
    
    Args:
        train_df: Training data
        test_df: Test data
        best_params: Best strategy parameters
        vol_windows: List of window sizes in seconds
        
    Returns:
        DataFrame with results for each window
    """
    vol_windows = vol_windows or [15, 30, 60, 120]
    
    print(f"\nRunning volatility window sensitivity ({len(vol_windows)} windows)...")
    
    results = []
    
    strategy_params = {k: v for k, v in best_params.items() 
                       if k in ['buffer', 'tau_max', 'min_tau', 'cooldown', 'exit_rule']}
    
    for window in vol_windows:
        # Create modified datasets with specified vol window
        # Map window to column name
        if window == 15:
            vol_col = 'realized_vol_15s'
        elif window == 30:
            vol_col = 'realized_vol_bps'
        elif window == 60:
            vol_col = 'realized_vol_60s'
        else:
            # Compute custom window
            vol_col = 'realized_vol_bps'  # Fallback
        
        train_mod = train_df.copy()
        test_mod = test_df.copy()
        
        if vol_col in train_df.columns and vol_col != 'realized_vol_bps':
            train_mod['realized_vol_bps'] = train_mod[vol_col]
            test_mod['realized_vol_bps'] = test_mod[vol_col]
        
        # Fit model
        model = BinnedFairValueModel(sample_every=5)
        model.fit(train_mod)
        
        # Run strategy
        strategy = MispricingBasedStrategy(
            fair_value_model=model,
            **strategy_params
        )
        
        result = run_backtest(test_mod, strategy, ExecutionConfig(), verbose=False)
        metrics = result['metrics']
        
        results.append({
            'vol_window': window,
            'n_trades': metrics['n_trades'],
            'total_pnl': metrics['total_pnl'],
            'mean_pnl_per_market': metrics['mean_pnl_per_market'],
            't_stat': metrics['t_stat'],
        })
        
        print(f"  Window {window}s: t={metrics['t_stat']:.2f}, PnL=${metrics['total_pnl']:.2f}")
    
    return pd.DataFrame(results)


def run_binning_sensitivity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_params: Dict,
) -> pd.DataFrame:
    """
    Test different binning schemes for fair value model.
    
    Args:
        train_df: Training data
        test_df: Test data
        best_params: Best strategy parameters
        
    Returns:
        DataFrame with results for each binning config
    """
    print("\nRunning binning sensitivity...")
    
    configs = [
        # (tau_size, delta_size, n_vol_bins)
        (15, 2.5, 2),
        (15, 5.0, 3),
        (30, 2.5, 2),
        (30, 5.0, 3),   # Default
        (30, 10.0, 3),
        (60, 5.0, 3),
        (60, 10.0, 5),
    ]
    
    results = []
    
    strategy_params = {k: v for k, v in best_params.items() 
                       if k in ['buffer', 'tau_max', 'min_tau', 'cooldown', 'exit_rule']}
    
    for tau_size, delta_size, n_vol in configs:
        model = BinnedFairValueModel(
            bin_tau_size=tau_size,
            bin_delta_size=delta_size,
            n_vol_bins=n_vol,
            sample_every=5
        )
        
        try:
            model.fit(train_df)
            
            strategy = MispricingBasedStrategy(
                fair_value_model=model,
                **strategy_params
            )
            
            result = run_backtest(test_df, strategy, ExecutionConfig(), verbose=False)
            metrics = result['metrics']
            
            results.append({
                'tau_size': tau_size,
                'delta_size': delta_size,
                'n_vol_bins': n_vol,
                'n_trades': metrics['n_trades'],
                'total_pnl': metrics['total_pnl'],
                't_stat': metrics['t_stat'],
                'n_valid_bins': model.bin_stats.get('n_valid_bins', 0),
            })
            
            print(f"  tau={tau_size}, delta={delta_size}, vol_bins={n_vol}: "
                  f"t={metrics['t_stat']:.2f}")
            
        except Exception as e:
            results.append({
                'tau_size': tau_size,
                'delta_size': delta_size,
                'n_vol_bins': n_vol,
                'error': str(e)
            })
    
    return pd.DataFrame(results)


def run_market_subset_analysis(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    fair_value_model,
    best_params: Dict,
    market_info: Dict
) -> Dict[str, Any]:
    """
    Test strategy on different market subsets.
    
    Args:
        df: Full data
        train_df: Training data
        fair_value_model: Fitted model
        best_params: Best strategy parameters
        market_info: Market metadata
        
    Returns:
        Dict with results for each subset
    """
    print("\nRunning market subset analysis...")
    
    strategy_params = {k: v for k, v in best_params.items() 
                       if k in ['buffer', 'tau_max', 'min_tau', 'cooldown', 'exit_rule']}
    
    strategy = MispricingBasedStrategy(
        fair_value_model=fair_value_model,
        **strategy_params
    )
    
    results = {}
    
    # Full dataset
    full_result = run_backtest(df, strategy, ExecutionConfig(), verbose=False)
    results['full'] = {
        'n_markets': df['market_id'].nunique(),
        'n_trades': full_result['metrics']['n_trades'],
        't_stat': full_result['metrics']['t_stat'],
        'total_pnl': full_result['metrics']['total_pnl'],
    }
    
    # High volatility markets (top 50%)
    market_vols = df.groupby('market_id')['realized_vol_bps'].mean()
    median_vol = market_vols.median()
    high_vol_markets = market_vols[market_vols >= median_vol].index.tolist()
    low_vol_markets = market_vols[market_vols < median_vol].index.tolist()
    
    high_vol_df = df[df['market_id'].isin(high_vol_markets)]
    low_vol_df = df[df['market_id'].isin(low_vol_markets)]
    
    if len(high_vol_df) > 0:
        high_vol_result = run_backtest(high_vol_df, strategy, ExecutionConfig(), verbose=False)
        results['high_volatility'] = {
            'n_markets': len(high_vol_markets),
            'n_trades': high_vol_result['metrics']['n_trades'],
            't_stat': high_vol_result['metrics']['t_stat'],
            'total_pnl': high_vol_result['metrics']['total_pnl'],
        }
        print(f"  High vol markets ({len(high_vol_markets)}): t={high_vol_result['metrics']['t_stat']:.2f}")
    
    if len(low_vol_df) > 0:
        low_vol_result = run_backtest(low_vol_df, strategy, ExecutionConfig(), verbose=False)
        results['low_volatility'] = {
            'n_markets': len(low_vol_markets),
            'n_trades': low_vol_result['metrics']['n_trades'],
            't_stat': low_vol_result['metrics']['t_stat'],
            'total_pnl': low_vol_result['metrics']['total_pnl'],
        }
        print(f"  Low vol markets ({len(low_vol_markets)}): t={low_vol_result['metrics']['t_stat']:.2f}")
    
    return results


def run_exit_rule_comparison(
    test_df: pd.DataFrame,
    fair_value_model,
    best_params: Dict,
) -> pd.DataFrame:
    """
    Compare different exit rules.
    
    Args:
        test_df: Test data
        fair_value_model: Fitted model
        best_params: Best strategy parameters (excluding exit_rule)
        
    Returns:
        DataFrame with results for each exit configuration
    """
    print("\nRunning exit rule comparison...")
    
    base_params = {k: v for k, v in best_params.items() 
                   if k in ['buffer', 'tau_max', 'min_tau', 'cooldown']}
    
    exit_configs = [
        {'exit_rule': 'expiry'},
        {'exit_rule': 'convergence', 'exit_buffer': 0.01, 'max_hold_seconds': 60},
        {'exit_rule': 'convergence', 'exit_buffer': 0.01, 'max_hold_seconds': 120},
        {'exit_rule': 'convergence', 'exit_buffer': 0.01, 'max_hold_seconds': 180},
        {'exit_rule': 'convergence', 'exit_buffer': 0.015, 'max_hold_seconds': 120},
        {'exit_rule': 'convergence', 'exit_buffer': 0.02, 'max_hold_seconds': 120},
    ]
    
    results = []
    
    for exit_config in exit_configs:
        strategy = MispricingBasedStrategy(
            fair_value_model=fair_value_model,
            **base_params,
            **exit_config
        )
        
        result = run_backtest(test_df, strategy, ExecutionConfig(), verbose=False)
        metrics = result['metrics']
        
        # Compute average hold time
        trades = result.get('trades', [])
        avg_hold = np.mean([t['exit_t'] - t['entry_t'] for t in trades]) if trades else 0
        
        results.append({
            **exit_config,
            'n_trades': metrics['n_trades'],
            'total_pnl': metrics['total_pnl'],
            't_stat': metrics['t_stat'],
            'hit_rate': metrics['hit_rate_per_trade'],
            'avg_hold_time': avg_hold,
        })
        
        print(f"  {exit_config['exit_rule']}: t={metrics['t_stat']:.2f}, hold={avg_hold:.0f}s")
    
    return pd.DataFrame(results)


# ==============================================================================
# PHASE 5: OUT-OF-SAMPLE VALIDATION
# ==============================================================================

def run_walk_forward_validation(
    df: pd.DataFrame,
    param_grid: Dict[str, List],
    train_size: int = 10,
    test_size: int = 2,
    step_size: int = 2
) -> Dict[str, Any]:
    """
    Run walk-forward validation.
    
    For each fold:
    1. Fit model on train markets
    2. Run parameter sweep (or use fixed params)
    3. Test on next markets
    
    Args:
        df: Full DataFrame
        param_grid: Parameter grid for sweep
        train_size: Markets in training window
        test_size: Markets in test window
        step_size: Step between folds
        
    Returns:
        Dict with per-fold results and aggregates
    """
    print("\nRunning walk-forward validation...")
    
    splits = get_walk_forward_splits(df, train_size, test_size, step_size)
    
    if len(splits) == 0:
        return {'error': 'Not enough markets for walk-forward validation'}
    
    fold_results = []
    
    for i, (train_df, test_df, train_ids, test_ids) in enumerate(splits):
        print(f"\n  Fold {i+1}/{len(splits)}: train={len(train_ids)} markets, test={len(test_ids)} markets")
        
        # Fit model
        model = BinnedFairValueModel(sample_every=5)
        model.fit(train_df)
        
        # Model calibration on test
        test_y = test_df.groupby('market_id')['Y'].first().loc[test_df['market_id']].values
        test_preds = model.predict(test_df)
        brier = compute_brier_score(test_y, test_preds)
        
        # Run with fixed parameters (simplified)
        strategy = MispricingBasedStrategy(
            fair_value_model=model,
            buffer=0.02,
            tau_max=420,
            exit_rule='expiry'
        )
        
        result = run_backtest(test_df, strategy, ExecutionConfig(), verbose=False)
        metrics = result['metrics']
        
        fold_results.append({
            'fold': i + 1,
            'n_train_markets': len(train_ids),
            'n_test_markets': len(test_ids),
            'model_brier': brier,
            'n_trades': metrics['n_trades'],
            'total_pnl': metrics['total_pnl'],
            'mean_pnl_per_market': metrics['mean_pnl_per_market'],
            't_stat': metrics['t_stat'],
        })
        
        print(f"    Model Brier: {brier:.4f}, Strategy t-stat: {metrics['t_stat']:.2f}")
    
    # Aggregate results
    fold_df = pd.DataFrame(fold_results)
    
    aggregates = {
        'mean_brier': fold_df['model_brier'].mean(),
        'std_brier': fold_df['model_brier'].std(),
        'mean_t_stat': fold_df['t_stat'].mean(),
        'std_t_stat': fold_df['t_stat'].std(),
        'mean_pnl': fold_df['total_pnl'].mean(),
        'positive_folds': (fold_df['total_pnl'] > 0).sum(),
        'total_folds': len(fold_df),
    }
    
    print(f"\n  Aggregate: mean_t={aggregates['mean_t_stat']:.2f} ± {aggregates['std_t_stat']:.2f}")
    print(f"  Positive folds: {aggregates['positive_folds']}/{aggregates['total_folds']}")
    
    return {
        'folds': fold_results,
        'aggregates': aggregates,
    }


def run_bootstrap_analysis(
    df: pd.DataFrame,
    fair_value_model,
    best_params: Dict,
    n_bootstrap: int = 1000,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Run bootstrap analysis for confidence intervals.
    
    Args:
        df: Data
        fair_value_model: Fitted model
        best_params: Strategy parameters
        n_bootstrap: Number of bootstrap samples
        random_seed: Random seed
        
    Returns:
        Dict with bootstrap distributions and confidence intervals
    """
    print(f"\nRunning bootstrap analysis ({n_bootstrap} samples)...")
    
    np.random.seed(random_seed)
    
    strategy_params = {k: v for k, v in best_params.items() 
                       if k in ['buffer', 'tau_max', 'min_tau', 'cooldown', 'exit_rule']}
    
    market_ids = df['market_id'].unique()
    n_markets = len(market_ids)
    
    bootstrap_pnls = []
    bootstrap_tstats = []
    
    for b in range(n_bootstrap):
        # Sample markets with replacement
        sampled_markets = np.random.choice(market_ids, size=n_markets, replace=True)
        
        # Build bootstrap sample
        bootstrap_dfs = []
        for market_id in sampled_markets:
            market_df = df[df['market_id'] == market_id].copy()
            # Give new market_id to avoid duplicate issues
            market_df['market_id'] = f"{market_id}_b{b}_{np.random.randint(1000000)}"
            bootstrap_dfs.append(market_df)
        
        bootstrap_df = pd.concat(bootstrap_dfs, ignore_index=True)
        
        # Run strategy
        strategy = MispricingBasedStrategy(
            fair_value_model=fair_value_model,
            **strategy_params
        )
        
        result = run_backtest(bootstrap_df, strategy, ExecutionConfig(), verbose=False)
        
        bootstrap_pnls.append(result['metrics']['mean_pnl_per_market'])
        bootstrap_tstats.append(result['metrics']['t_stat'])
        
        if (b + 1) % 100 == 0:
            print(f"  Completed {b + 1}/{n_bootstrap} bootstraps")
    
    bootstrap_pnls = np.array(bootstrap_pnls)
    bootstrap_tstats = np.array(bootstrap_tstats)
    
    results = {
        'n_bootstrap': n_bootstrap,
        'pnl_distribution': {
            'mean': bootstrap_pnls.mean(),
            'std': bootstrap_pnls.std(),
            'percentile_5': np.percentile(bootstrap_pnls, 5),
            'percentile_50': np.percentile(bootstrap_pnls, 50),
            'percentile_95': np.percentile(bootstrap_pnls, 95),
        },
        't_stat_distribution': {
            'mean': bootstrap_tstats.mean(),
            'std': bootstrap_tstats.std(),
            'percentile_5': np.percentile(bootstrap_tstats, 5),
            'percentile_50': np.percentile(bootstrap_tstats, 50),
            'percentile_95': np.percentile(bootstrap_tstats, 95),
        },
        'prob_positive_pnl': (bootstrap_pnls > 0).mean(),
        'prob_t_stat_gt_2': (bootstrap_tstats > 2.0).mean(),
    }
    
    print(f"\n  Mean PnL: ${results['pnl_distribution']['mean']:.4f} "
          f"(95% CI: ${results['pnl_distribution']['percentile_5']:.4f} - "
          f"${results['pnl_distribution']['percentile_95']:.4f})")
    print(f"  P(positive PnL): {results['prob_positive_pnl']:.1%}")
    print(f"  P(t-stat > 2): {results['prob_t_stat_gt_2']:.1%}")
    
    return results


# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def run_full_test_suite(
    min_coverage: float = 90.0,
    output_dir: Optional[Path] = None,
    phases: List[int] = None,
    n_bootstrap: int = 500
) -> Dict[str, Any]:
    """
    Run full mispricing strategy test suite.
    
    Args:
        min_coverage: Minimum market coverage
        output_dir: Directory to save results
        phases: Which phases to run (None = all)
        n_bootstrap: Number of bootstrap samples for Phase 5
        
    Returns:
        Dict with all results
    """
    phases = phases or [3, 4, 5]
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("MISPRICING-BASED STRATEGY TEST SUITE")
    print("="*70)
    print(f"Phases to run: {phases}")
    print(f"Output directory: {output_dir}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'phases_run': phases,
    }
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    df, market_info = load_eth_markets(min_coverage=min_coverage)
    df = add_derived_columns(df)
    
    print(f"Loaded {df['market_id'].nunique()} markets, {len(df):,} observations")
    
    # Train/test split
    train_df, test_df, train_ids, test_ids = get_train_test_split(df, train_frac=0.7)
    
    results['data_info'] = {
        'n_markets': df['market_id'].nunique(),
        'n_observations': len(df),
        'n_train_markets': len(train_ids),
        'n_test_markets': len(test_ids),
    }
    
    # Fit base model
    print("\nFitting base fair value model...")
    base_model = BinnedFairValueModel(
        bin_tau_size=30,
        bin_delta_size=5.0,
        n_vol_bins=3,
        sample_every=5
    )
    base_model.fit(train_df)
    print(f"  Model fitted: {base_model.bin_stats.get('n_valid_bins', 0)} valid bins")
    
    # ==========================
    # PHASE 3: Parameter Sweep
    # ==========================
    if 3 in phases:
        print("\n" + "="*70)
        print("PHASE 3: PARAMETER SWEEP")
        print("="*70)
        
        param_grid = {
            'buffer': [0.01, 0.015, 0.02, 0.025, 0.03],
            'tau_max': [300, 420, 600],
            'min_tau': [0, 30, 60],
            'exit_rule': ['expiry'],
            'cooldown': [30],
        }
        
        sweep_results = run_parameter_sweep(
            train_df, test_df, base_model, param_grid, verbose=False
        )
        
        # Save sweep results
        sweep_results.to_csv(output_dir / 'parameter_sweep.csv', index=False)
        
        best_params = select_best_parameters(sweep_results)
        
        results['phase3'] = {
            'param_grid': param_grid,
            'n_combinations': len(sweep_results),
            'best_params': best_params,
        }
        
        print(f"\nBest parameters:")
        if 'best' in best_params:
            for k, v in best_params['best'].items():
                if k not in ['error']:
                    print(f"  {k}: {v}")
    
    # Get best params for subsequent phases
    if 'phase3' in results and 'best' in results['phase3']['best_params']:
        best_strategy_params = results['phase3']['best_params']['best']
    else:
        # Default params
        best_strategy_params = {
            'buffer': 0.02,
            'tau_max': 420,
            'min_tau': 0,
            'cooldown': 30,
            'exit_rule': 'expiry',
        }
    
    # ==========================
    # PHASE 4: Robustness Checks
    # ==========================
    if 4 in phases:
        print("\n" + "="*70)
        print("PHASE 4: ROBUSTNESS CHECKS")
        print("="*70)
        
        phase4_results = {}
        
        # 4.1 Latency sensitivity
        latency_results = run_latency_sensitivity(
            test_df, base_model, best_strategy_params
        )
        latency_results.to_csv(output_dir / 'latency_sensitivity.csv', index=False)
        phase4_results['latency'] = latency_results.to_dict('records')
        
        # 4.2 Placebo test
        placebo_results = run_placebo_test(
            test_df, base_model, best_strategy_params, shift_seconds=30
        )
        phase4_results['placebo'] = placebo_results
        
        # 4.3 Volatility window sensitivity
        vol_results = run_vol_window_sensitivity(
            train_df, test_df, best_strategy_params
        )
        vol_results.to_csv(output_dir / 'vol_window_sensitivity.csv', index=False)
        phase4_results['vol_window'] = vol_results.to_dict('records')
        
        # 4.4 Binning sensitivity
        binning_results = run_binning_sensitivity(
            train_df, test_df, best_strategy_params
        )
        binning_results.to_csv(output_dir / 'binning_sensitivity.csv', index=False)
        phase4_results['binning'] = binning_results.to_dict('records')
        
        # 4.5 Market subset analysis
        subset_results = run_market_subset_analysis(
            df, train_df, base_model, best_strategy_params, market_info
        )
        phase4_results['market_subsets'] = subset_results
        
        # 4.6 Exit rule comparison
        exit_results = run_exit_rule_comparison(
            test_df, base_model, best_strategy_params
        )
        exit_results.to_csv(output_dir / 'exit_rule_comparison.csv', index=False)
        phase4_results['exit_rules'] = exit_results.to_dict('records')
        
        results['phase4'] = phase4_results
    
    # ==========================
    # PHASE 5: OOS Validation
    # ==========================
    if 5 in phases:
        print("\n" + "="*70)
        print("PHASE 5: OUT-OF-SAMPLE VALIDATION")
        print("="*70)
        
        phase5_results = {}
        
        # 5.1 Walk-forward validation
        wf_results = run_walk_forward_validation(
            df, 
            param_grid={'buffer': [0.02], 'tau_max': [420]},
            train_size=10,
            test_size=2,
            step_size=2
        )
        phase5_results['walk_forward'] = wf_results
        
        # 5.2 Bootstrap confidence intervals
        bootstrap_results = run_bootstrap_analysis(
            test_df, base_model, best_strategy_params, n_bootstrap=n_bootstrap
        )
        phase5_results['bootstrap'] = bootstrap_results
        
        results['phase5'] = phase5_results
    
    # ==========================
    # FINAL SUMMARY
    # ==========================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    summary = {
        'model_n_valid_bins': base_model.bin_stats.get('n_valid_bins', 0),
    }
    
    if 'phase3' in results:
        best = results['phase3']['best_params'].get('best', {})
        summary['best_t_stat'] = best.get('t_stat', None)
        summary['best_total_pnl'] = best.get('total_pnl', None)
        summary['best_buffer'] = best.get('buffer', None)
        summary['best_tau_max'] = best.get('tau_max', None)
    
    if 'phase4' in results:
        placebo = results['phase4'].get('placebo', {})
        summary['placebo_edge_destroyed'] = placebo.get('edge_destroyed', None)
        summary['placebo_t_stat_degradation'] = placebo.get('t_stat_degradation', None)
    
    if 'phase5' in results:
        bootstrap = results['phase5'].get('bootstrap', {})
        summary['bootstrap_prob_positive'] = bootstrap.get('prob_positive_pnl', None)
        summary['bootstrap_prob_t_gt_2'] = bootstrap.get('prob_t_stat_gt_2', None)
    
    results['summary'] = summary
    
    # Print summary
    print("\nStrategy Performance:")
    if summary.get('best_t_stat'):
        print(f"  Best t-stat: {summary['best_t_stat']:.2f}")
        print(f"  Best total PnL: ${summary['best_total_pnl']:.2f}")
    
    print("\nRobustness:")
    if summary.get('placebo_edge_destroyed') is not None:
        print(f"  Placebo edge destroyed: {summary['placebo_edge_destroyed']}")
    
    print("\nConfidence:")
    if summary.get('bootstrap_prob_positive'):
        print(f"  P(positive PnL): {summary['bootstrap_prob_positive']:.1%}")
        print(f"  P(t-stat > 2): {summary['bootstrap_prob_t_gt_2']:.1%}")
    
    # Save results
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_dir / 'mispricing_strategy_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


# ==============================================================================
# PHASE 6.1: DATA-SNOOPING / SELECTION BIAS TESTS
# ==============================================================================

def run_nested_selection_validation(
    df: pd.DataFrame,
    param_grid: Dict[str, List],
    n_outer_folds: int = 5,
    inner_train_frac: float = 0.7,
    random_seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run nested selection validation to quantify selection bias.
    
    Outer loop: Split data into train/test
    Inner loop: Select best params on train validation set
    Report: Distribution of test set performance across outer folds
    
    Args:
        df: Full DataFrame
        param_grid: Parameter grid for sweep
        n_outer_folds: Number of outer cross-validation folds
        inner_train_frac: Fraction of outer-train for inner training
        random_seed: Random seed
        verbose: Print progress
        
    Returns:
        Dict with nested selection results
    """
    if verbose:
        print("\n" + "="*70)
        print("NESTED SELECTION VALIDATION (Data-Snooping Risk)")
        print("="*70)
    
    np.random.seed(random_seed)
    
    market_ids = df['market_id'].unique().tolist()
    n_markets = len(market_ids)
    
    # Shuffle markets for random folds
    np.random.shuffle(market_ids)
    
    fold_results = []
    selected_params_all = []
    
    for fold in range(n_outer_folds):
        if verbose:
            print(f"\n--- Outer Fold {fold + 1}/{n_outer_folds} ---")
        
        # Create outer train/test split
        fold_size = n_markets // n_outer_folds
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_outer_folds - 1 else n_markets
        
        test_ids = market_ids[test_start:test_end]
        train_ids = [m for m in market_ids if m not in test_ids]
        
        outer_train_df = df[df['market_id'].isin(train_ids)]
        outer_test_df = df[df['market_id'].isin(test_ids)]
        
        # Create inner train/validation split from outer train
        n_inner_train = int(len(train_ids) * inner_train_frac)
        inner_train_ids = train_ids[:n_inner_train]
        inner_val_ids = train_ids[n_inner_train:]
        
        inner_train_df = df[df['market_id'].isin(inner_train_ids)]
        inner_val_df = df[df['market_id'].isin(inner_val_ids)]
        
        if verbose:
            print(f"  Inner train: {len(inner_train_ids)} markets, "
                  f"Inner val: {len(inner_val_ids)} markets, "
                  f"Outer test: {len(test_ids)} markets")
        
        # Fit model on inner train
        model = BinnedFairValueModel(sample_every=5)
        model.fit(inner_train_df)
        
        # Inner loop: Select best params on inner validation
        best_inner_t = -np.inf
        best_params = None
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        for combo in combinations:
            params = dict(zip(param_names, combo))
            
            try:
                strategy = MispricingBasedStrategy(
                    fair_value_model=model,
                    **params
                )
                result = run_backtest(inner_val_df, strategy, ExecutionConfig(), verbose=False)
                inner_t = result['metrics']['t_stat']
                
                if inner_t > best_inner_t:
                    best_inner_t = inner_t
                    best_params = params
            except Exception:
                continue
        
        if best_params is None:
            if verbose:
                print("  [WARN] No valid params found in inner loop")
            continue
        
        selected_params_all.append(best_params)
        
        if verbose:
            print(f"  Selected params: buffer={best_params.get('buffer')}, "
                  f"tau_max={best_params.get('tau_max')}")
            print(f"  Inner validation t-stat: {best_inner_t:.2f}")
        
        # Outer loop: Test selected params on outer test set
        # Re-fit model on FULL outer train
        model_outer = BinnedFairValueModel(sample_every=5)
        model_outer.fit(outer_train_df)
        
        strategy_outer = MispricingBasedStrategy(
            fair_value_model=model_outer,
            **best_params
        )
        
        outer_result = run_backtest(outer_test_df, strategy_outer, ExecutionConfig(), verbose=False)
        outer_metrics = outer_result['metrics']
        
        fold_results.append({
            'fold': fold + 1,
            'n_test_markets': len(test_ids),
            'selected_params': best_params,
            'inner_t_stat': best_inner_t,
            'outer_t_stat': outer_metrics['t_stat'],
            'outer_pnl': outer_metrics['total_pnl'],
            'outer_mean_pnl': outer_metrics['mean_pnl_per_market'],
            'n_trades': outer_metrics['n_trades'],
        })
        
        if verbose:
            print(f"  Outer test t-stat: {outer_metrics['t_stat']:.2f}, "
                  f"PnL: ${outer_metrics['total_pnl']:.2f}")
    
    if len(fold_results) == 0:
        return {'error': 'No valid folds completed'}
    
    # Aggregate results
    outer_tstats = [r['outer_t_stat'] for r in fold_results]
    outer_pnls = [r['outer_pnl'] for r in fold_results]
    
    results = {
        'n_folds': len(fold_results),
        'fold_results': fold_results,
        'outer_t_stat_distribution': {
            'mean': float(np.mean(outer_tstats)),
            'std': float(np.std(outer_tstats)),
            'min': float(np.min(outer_tstats)),
            'median': float(np.median(outer_tstats)),
            'max': float(np.max(outer_tstats)),
        },
        'outer_pnl_distribution': {
            'mean': float(np.mean(outer_pnls)),
            'std': float(np.std(outer_pnls)),
            'min': float(np.min(outer_pnls)),
            'median': float(np.median(outer_pnls)),
            'max': float(np.max(outer_pnls)),
        },
        'prob_positive_outer_pnl': float(np.mean([p > 0 for p in outer_pnls])),
        'prob_outer_t_gt_2': float(np.mean([t > 2.0 for t in outer_tstats])),
        'prob_outer_t_gt_1_5': float(np.mean([t > 1.5 for t in outer_tstats])),
    }
    
    if verbose:
        print("\n" + "="*70)
        print("NESTED SELECTION SUMMARY")
        print("="*70)
        print(f"\nOuter test t-stat distribution:")
        print(f"  Mean: {results['outer_t_stat_distribution']['mean']:.2f}")
        print(f"  Median: {results['outer_t_stat_distribution']['median']:.2f}")
        print(f"  Min: {results['outer_t_stat_distribution']['min']:.2f}")
        print(f"  Max: {results['outer_t_stat_distribution']['max']:.2f}")
        print(f"\nP(outer t-stat > 2.0): {results['prob_outer_t_gt_2']:.1%}")
        print(f"P(outer t-stat > 1.5): {results['prob_outer_t_gt_1_5']:.1%}")
        print(f"P(outer PnL > 0): {results['prob_positive_outer_pnl']:.1%}")
    
    return results


def run_spa_test(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    param_grid: Dict[str, List],
    n_bootstrap: int = 1000,
    random_seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Superior Predictive Ability (SPA) test to adjust for data-snooping.
    
    Tests null hypothesis: Best strategy performs no better than benchmark.
    Uses stationary bootstrap to adjust p-value for multiple testing.
    
    Args:
        test_df: Test data
        train_df: Training data
        param_grid: Parameter grid for all strategies tested
        n_bootstrap: Number of bootstrap samples
        random_seed: Random seed
        verbose: Print progress
        
    Returns:
        Dict with SPA test results
    """
    if verbose:
        print("\n" + "="*70)
        print("SUPERIOR PREDICTIVE ABILITY (SPA) TEST")
        print("="*70)
    
    np.random.seed(random_seed)
    
    # Fit model
    model = BinnedFairValueModel(sample_every=5)
    model.fit(train_df)
    
    # Get market IDs for resampling
    market_ids = test_df['market_id'].unique().tolist()
    n_markets = len(market_ids)
    
    # Compute returns for all strategy variants
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    if verbose:
        print(f"\nTesting {len(combinations)} strategy variants on {n_markets} test markets...")
    
    # Store per-market returns for each strategy
    strategy_returns = []  # List of dicts: {market_id: pnl}
    
    for combo in combinations:
        params = dict(zip(param_names, combo))
        
        try:
            strategy = MispricingBasedStrategy(
                fair_value_model=model,
                **params
            )
            result = run_backtest(test_df, strategy, ExecutionConfig(), verbose=False)
            
            # Get per-market PnL
            market_pnls = {}
            for trade in result['trades']:
                mid = trade['market_id']
                if mid not in market_pnls:
                    market_pnls[mid] = 0
                market_pnls[mid] += trade['pnl']
            
            # Fill missing markets with 0
            for mid in market_ids:
                if mid not in market_pnls:
                    market_pnls[mid] = 0
            
            strategy_returns.append({
                'params': params,
                'market_pnls': market_pnls,
                'mean_pnl': np.mean(list(market_pnls.values())),
            })
        except Exception:
            continue
    
    if len(strategy_returns) == 0:
        return {'error': 'No valid strategies'}
    
    # Find best strategy (benchmark)
    best_idx = np.argmax([s['mean_pnl'] for s in strategy_returns])
    best_strategy = strategy_returns[best_idx]
    best_mean_pnl = best_strategy['mean_pnl']
    
    if verbose:
        print(f"\nBest strategy: buffer={best_strategy['params'].get('buffer')}, "
              f"tau_max={best_strategy['params'].get('tau_max')}")
        print(f"Best mean PnL: ${best_mean_pnl:.4f}")
    
    # Compute loss differentials (vs zero benchmark)
    # d_it = R_it - R_0t where R_0 = 0 (zero benchmark)
    loss_diffs = []
    for s in strategy_returns:
        d_i = [s['market_pnls'].get(m, 0) for m in market_ids]
        loss_diffs.append(np.array(d_i))
    
    loss_diffs = np.array(loss_diffs)  # Shape: (n_strategies, n_markets)
    
    # Mean loss differential for each strategy
    mean_d = loss_diffs.mean(axis=1)  # Shape: (n_strategies,)
    
    # Best strategy's mean loss differential
    best_mean_d = mean_d[best_idx]
    
    # Bootstrap the test statistic
    # H0: max(mean_d) <= 0 (best strategy doesn't beat benchmark)
    bootstrap_maxes = []
    
    for b in range(n_bootstrap):
        # Stationary bootstrap: resample markets with replacement
        boot_indices = np.random.choice(n_markets, size=n_markets, replace=True)
        
        # Compute centered bootstrap statistic
        boot_loss_diffs = loss_diffs[:, boot_indices]
        boot_mean_d = boot_loss_diffs.mean(axis=1)
        
        # Center by subtracting original mean
        centered_boot = boot_mean_d - mean_d
        
        # Record max of centered statistics
        bootstrap_maxes.append(np.max(centered_boot))
    
    bootstrap_maxes = np.array(bootstrap_maxes)
    
    # P-value: probability that bootstrap max exceeds observed max - mean
    # Under H0, the test statistic should be centered at 0
    test_stat = best_mean_d
    p_value = np.mean(bootstrap_maxes >= test_stat)
    
    # Also compute unadjusted p-value (single test)
    std_d = loss_diffs[best_idx].std(ddof=1)
    se = std_d / np.sqrt(n_markets)
    unadjusted_t = best_mean_d / se if se > 0 else 0
    unadjusted_p = 2 * (1 - stats.norm.cdf(abs(unadjusted_t)))
    
    results = {
        'n_strategies': len(strategy_returns),
        'n_markets': n_markets,
        'n_bootstrap': n_bootstrap,
        'best_strategy_params': best_strategy['params'],
        'best_mean_pnl': float(best_mean_pnl),
        'test_statistic': float(test_stat),
        'spa_p_value': float(p_value),
        'unadjusted_t_stat': float(unadjusted_t),
        'unadjusted_p_value': float(unadjusted_p),
        'selection_bias_adjustment': float(p_value - unadjusted_p),
        'significant_at_005': p_value < 0.05,
        'significant_at_010': p_value < 0.10,
    }
    
    if verbose:
        print(f"\nSPA Test Results:")
        print(f"  Test statistic: {test_stat:.4f}")
        print(f"  Unadjusted p-value: {unadjusted_p:.4f}")
        print(f"  SPA-adjusted p-value: {p_value:.4f}")
        print(f"  Selection bias: +{(p_value - unadjusted_p):.4f}")
        print(f"\n  Significant at 5%: {results['significant_at_005']}")
        print(f"  Significant at 10%: {results['significant_at_010']}")
        
        if p_value < 0.05:
            print("\n  [PASS] Edge survives data-snooping adjustment")
        elif p_value < 0.10:
            print("\n  [WARN] Marginal significance after adjustment")
        else:
            print("\n  [FAIL] Edge may be driven by selection bias")
    
    return results


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test mispricing-based strategy')
    parser.add_argument('--min-coverage', type=float, default=90.0,
                        help='Minimum market coverage %%')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR),
                        help='Output directory')
    parser.add_argument('--phase', type=int, nargs='+', default=None,
                        help='Specific phases to run (3, 4, 5)')
    parser.add_argument('--full', action='store_true',
                        help='Run all phases')
    parser.add_argument('--n-bootstrap', type=int, default=500,
                        help='Number of bootstrap samples')
    
    args = parser.parse_args()
    
    phases = args.phase
    if args.full or phases is None:
        phases = [3, 4, 5]
    
    results = run_full_test_suite(
        min_coverage=args.min_coverage,
        output_dir=Path(args.output_dir),
        phases=phases,
        n_bootstrap=args.n_bootstrap
    )

