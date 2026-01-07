#!/usr/bin/env python3
"""
Phase 5: Run Backtests

Run Strategy A/B/C with parameter sweeps, latency sensitivity, and placebo tests.

Uses existing backtest infrastructure from scripts/backtest/.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from itertools import product

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest.data_loader import load_eth_markets, add_derived_columns
from scripts.backtest.backtest_engine import run_backtest, ExecutionConfig
from scripts.backtest.strategies import (
    UnderroundHarvesterStrategy,
    LateDirectionalTakerStrategy,
    TwoSidedEarlyTiltLateStrategy,
)

# Output directory
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "results"


def load_strategy_params() -> Dict:
    """Load extracted strategy parameters."""
    params_path = OUTPUT_DIR / "strategy_params.json"
    with open(params_path) as f:
        return json.load(f)


def run_strategy_sweep(
    df: pd.DataFrame,
    strategy_class,
    param_grid: Dict[str, List],
    base_config: ExecutionConfig,
    strategy_name: str,
    verbose: bool = False
) -> List[Dict]:
    """
    Run parameter sweep for a strategy.
    
    Args:
        df: Market data
        strategy_class: Strategy class
        param_grid: Dict of param_name -> list of values
        base_config: Execution config
        strategy_name: Name for logging
        verbose: Print progress
        
    Returns:
        List of result dicts
    """
    results = []
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    print(f"\nRunning {len(combinations)} parameter combinations for {strategy_name}...")
    
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        
        try:
            strategy = strategy_class(**params)
            result = run_backtest(df, strategy, base_config, verbose=verbose)
            
            # Extract key metrics (metrics is a dict from to_dict())
            metrics = result['metrics']
            results.append({
                'strategy': strategy_name,
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
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{len(combinations)}")
                
        except Exception as e:
            print(f"  Error with params {params}: {e}")
            results.append({
                'strategy': strategy_name,
                **params,
                'error': str(e)
            })
    
    return results


def run_latency_sweep(
    df: pd.DataFrame,
    strategy,
    latencies: List[float],
    strategy_name: str
) -> List[Dict]:
    """
    Run latency sensitivity sweep for a strategy.
    
    Args:
        df: Market data
        strategy: Strategy instance
        latencies: List of latency values to test
        strategy_name: Name for logging
        
    Returns:
        List of result dicts
    """
    results = []
    
    print(f"\nRunning latency sweep for {strategy_name}...")
    
    for latency in latencies:
        config = ExecutionConfig(
            signal_latency_s=latency / 2,
            exec_latency_s=latency / 2,
        )
        
        try:
            result = run_backtest(df, strategy, config)
            metrics = result['metrics']
            
            results.append({
                'strategy': strategy_name,
                'total_latency_s': latency,
                'n_trades': metrics['n_trades'],
                'total_pnl': metrics['total_pnl'],
                'mean_pnl_per_market': metrics['mean_pnl_per_market'],
                't_stat': metrics['t_stat'],
                'hit_rate_trade': metrics['hit_rate_per_trade'],
            })
            
        except Exception as e:
            print(f"  Error with latency {latency}: {e}")
    
    return results


def run_placebo_test(
    df: pd.DataFrame,
    strategy,
    cl_shift_seconds: int,
    strategy_name: str
) -> Dict:
    """
    Run placebo test by shifting CL data.
    
    Args:
        df: Market data
        strategy: Strategy instance
        cl_shift_seconds: How many seconds to shift CL forward
        strategy_name: Name for logging
        
    Returns:
        Result dict
    """
    print(f"\nRunning placebo test for {strategy_name} (CL shift +{cl_shift_seconds}s)...")
    
    # Create shifted dataset
    df_shifted = df.copy()
    
    # Shift CL columns forward by cl_shift_seconds within each market
    cl_cols = ['cl_mid', 'cl_bid', 'cl_ask', 'delta', 'delta_bps']
    for col in cl_cols:
        if col in df_shifted.columns:
            df_shifted[col] = df_shifted.groupby('market_id')[col].shift(cl_shift_seconds)
    
    # Drop rows with NaN CL data
    df_shifted = df_shifted.dropna(subset=['cl_mid'])
    
    # Recalculate derived columns
    df_shifted = add_derived_columns(df_shifted)
    
    # Run backtest on shifted data
    config = ExecutionConfig()
    try:
        result = run_backtest(df_shifted, strategy, config)
        metrics = result['metrics']
        
        return {
            'strategy': strategy_name,
            'cl_shift_seconds': cl_shift_seconds,
            'n_trades': metrics['n_trades'],
            'total_pnl': metrics['total_pnl'],
            'mean_pnl_per_market': metrics['mean_pnl_per_market'],
            't_stat': metrics['t_stat'],
            'placebo_status': 'PASS' if metrics['t_stat'] < 1.5 else 'FAIL (edge persists)',
        }
    except Exception as e:
        return {
            'strategy': strategy_name,
            'cl_shift_seconds': cl_shift_seconds,
            'error': str(e)
        }


def main():
    print("=" * 60)
    print("Phase 5: Run Backtests")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df, market_info = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    # Identify market subsets
    all_market_ids = df['market_id'].unique().tolist()
    volume_market_ids = [
        "20260106_1630", "20260106_1645", "20260106_1700", "20260106_1715",
        "20260106_1730", "20260106_1745", "20260106_1800", "20260106_1815",
        "20260106_1830", "20260106_1845", "20260106_1900", "20260106_1915"
    ]
    volume_market_ids = [m for m in volume_market_ids if m in all_market_ids]
    
    print(f"\nMarket subsets:")
    print(f"  All markets: {len(all_market_ids)}")
    print(f"  Volume markets: {len(volume_market_ids)}")
    
    # Create volume subset
    df_volume = df[df['market_id'].isin(volume_market_ids)].copy()
    
    # Load strategy parameters
    strategy_params = load_strategy_params()
    
    # Define parameter grids for sweeps
    param_grids = {
        'Strategy_A': {
            'epsilon': [0.01, 0.02, 0.03, 0.04],
            'min_tau': [30, 60, 120],
            'max_tau': [720, 840],
        },
        'Strategy_B': {
            'tau_max': [120, 180, 300, 420],
            'delta_threshold_bps': [5, 10, 15, 20],
            'hold_seconds': [120, 180, 240],
        },
        'Strategy_C': {
            'inventory_phase_end': [180, 300, 420],
            'tilt_phase_start': [120, 180],
            'inventory_epsilon': [0.01, 0.02, 0.03],
            'tilt_delta_threshold_bps': [10, 15, 20],
        },
    }
    
    # Strategy classes
    strategy_classes = {
        'Strategy_A': UnderroundHarvesterStrategy,
        'Strategy_B': LateDirectionalTakerStrategy,
        'Strategy_C': TwoSidedEarlyTiltLateStrategy,
    }
    
    # Run parameter sweeps on all markets
    all_results = []
    base_config = ExecutionConfig()
    
    print("\n" + "=" * 60)
    print("PARAMETER SWEEPS (All Markets)")
    print("=" * 60)
    
    for strategy_key, param_grid in param_grids.items():
        results = run_strategy_sweep(
            df, 
            strategy_classes[strategy_key],
            param_grid,
            base_config,
            strategy_key
        )
        all_results.extend(results)
    
    # Save sweep results
    sweep_df = pd.DataFrame(all_results)
    sweep_path = OUTPUT_DIR / "parameter_sweep_results.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\nParameter sweep results saved to: {sweep_path}")
    
    # Find best parameters for each strategy
    print("\n" + "=" * 60)
    print("BEST PARAMETERS BY STRATEGY")
    print("=" * 60)
    
    best_params = {}
    for strategy_key in param_grids.keys():
        strategy_results = sweep_df[sweep_df['strategy'] == strategy_key].copy()
        if len(strategy_results) == 0:
            continue
        
        # Filter to valid results
        strategy_results = strategy_results[~strategy_results['total_pnl'].isna()]
        if len(strategy_results) == 0:
            continue
        
        # Best by total PnL
        best_idx = strategy_results['total_pnl'].idxmax()
        best_row = strategy_results.loc[best_idx]
        
        print(f"\n{strategy_key}:")
        print(f"  Best PnL: ${best_row['total_pnl']:.2f}")
        print(f"  t-stat: {best_row['t_stat']:.2f}")
        print(f"  Trades: {best_row['n_trades']}")
        
        # Extract params
        param_cols = list(param_grids[strategy_key].keys())
        best_params[strategy_key] = {col: best_row[col] for col in param_cols}
        print(f"  Params: {best_params[strategy_key]}")
    
    # Run latency sensitivity on best strategies
    print("\n" + "=" * 60)
    print("LATENCY SENSITIVITY")
    print("=" * 60)
    
    latencies = [0, 0.5, 1, 2, 5]
    latency_results = []
    
    for strategy_key, params in best_params.items():
        strategy = strategy_classes[strategy_key](**params)
        results = run_latency_sweep(df, strategy, latencies, strategy_key)
        latency_results.extend(results)
    
    latency_df = pd.DataFrame(latency_results)
    latency_path = OUTPUT_DIR / "latency_sensitivity_results.csv"
    latency_df.to_csv(latency_path, index=False)
    print(f"\nLatency sensitivity results saved to: {latency_path}")
    
    # Print latency summary
    print("\nLatency Impact Summary:")
    for strategy_key in best_params.keys():
        strategy_lat = latency_df[latency_df['strategy'] == strategy_key]
        if len(strategy_lat) > 0:
            print(f"\n  {strategy_key}:")
            for _, row in strategy_lat.iterrows():
                print(f"    Latency {row['total_latency_s']:.1f}s: "
                      f"PnL=${row['total_pnl']:.2f}, t={row['t_stat']:.2f}")
    
    # Run placebo tests
    print("\n" + "=" * 60)
    print("PLACEBO TESTS (CL shift +30s)")
    print("=" * 60)
    
    placebo_results = []
    for strategy_key, params in best_params.items():
        strategy = strategy_classes[strategy_key](**params)
        result = run_placebo_test(df, strategy, cl_shift_seconds=30, strategy_name=strategy_key)
        placebo_results.append(result)
        
        print(f"\n  {strategy_key}:")
        print(f"    Original t-stat: {latency_df[latency_df['strategy'] == strategy_key].iloc[0]['t_stat']:.2f}")
        print(f"    Placebo t-stat: {result.get('t_stat', 'N/A')}")
        print(f"    Status: {result.get('placebo_status', 'N/A')}")
    
    placebo_df = pd.DataFrame(placebo_results)
    placebo_path = OUTPUT_DIR / "placebo_test_results.csv"
    placebo_df.to_csv(placebo_path, index=False)
    print(f"\nPlacebo test results saved to: {placebo_path}")
    
    # Run on volume markets subset
    if len(volume_market_ids) > 0:
        print("\n" + "=" * 60)
        print("VOLUME MARKETS SUBSET")
        print("=" * 60)
        
        volume_results = []
        for strategy_key, params in best_params.items():
            strategy = strategy_classes[strategy_key](**params)
            result = run_backtest(df_volume, strategy, base_config)
            metrics = result['metrics']
            
            volume_results.append({
                'strategy': strategy_key,
                'subset': 'volume_markets',
                'n_markets': metrics['n_markets'],
                'n_trades': metrics['n_trades'],
                'total_pnl': metrics['total_pnl'],
                'mean_pnl_per_market': metrics['mean_pnl_per_market'],
                't_stat': metrics['t_stat'],
                'hit_rate_trade': metrics['hit_rate_per_trade'],
            })
            
            print(f"\n  {strategy_key}:")
            print(f"    Markets: {metrics['n_markets']}")
            print(f"    Trades: {metrics['n_trades']}")
            print(f"    Total PnL: ${metrics['total_pnl']:.2f}")
            print(f"    t-stat: {metrics['t_stat']:.2f}")
        
        volume_df = pd.DataFrame(volume_results)
        volume_path = OUTPUT_DIR / "volume_subset_results.csv"
        volume_df.to_csv(volume_path, index=False)
        print(f"\nVolume subset results saved to: {volume_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    print("\nBest Strategy Performance (All Markets):")
    print("-" * 60)
    print(f"{'Strategy':<25} {'PnL':>10} {'t-stat':>8} {'Trades':>8}")
    print("-" * 60)
    
    for strategy_key, params in best_params.items():
        best_row = sweep_df[(sweep_df['strategy'] == strategy_key)].nlargest(1, 'total_pnl').iloc[0]
        print(f"{strategy_key:<25} ${best_row['total_pnl']:>9.2f} {best_row['t_stat']:>8.2f} {int(best_row['n_trades']):>8}")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    
    return sweep_df, latency_df, placebo_df


if __name__ == "__main__":
    main()

