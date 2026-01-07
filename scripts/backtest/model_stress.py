#!/usr/bin/env python3
"""
Phase 6.4: Model Risk Tests (Fair Value Model Stability)

Tests to ensure the model isn't overfitting or using artifacts:
- Observed-only training and trading
- Coarser information stress (5s, 10s updates)
- Calibration diagnostics (reliability diagram, Brier by tau)

These tests verify the edge doesn't require forward-filled segments
or micro-timing precision.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest.data_loader import load_eth_markets, add_derived_columns, get_train_test_split
from scripts.backtest.fair_value import (
    BinnedFairValueModel, 
    compute_brier_score, 
    compute_expected_calibration_error,
    compute_calibration_curve
)
from scripts.backtest.strategies import MispricingBasedStrategy
from scripts.backtest.backtest_engine import run_backtest, ExecutionConfig


# ==============================================================================
# 6.4.1: OBSERVED-ONLY TRAINING AND TRADING
# ==============================================================================

def run_observed_only_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    strategy_params: Dict,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test strategy using only observed data (no forward-fill).
    
    Filters to rows where cl_ffill == 0 AND pm_ffill == 0 for both
    training and testing. Compares to baseline with full data.
    
    Args:
        train_df: Training data
        test_df: Test data
        strategy_params: Strategy parameters
        verbose: Print progress
        
    Returns:
        Dict with baseline and observed-only results
    """
    if verbose:
        print("\nRunning Observed-Only Test...")
    
    # Baseline: full data
    model_full = BinnedFairValueModel(sample_every=5)
    model_full.fit(train_df)
    
    strategy_full = MispricingBasedStrategy(
        fair_value_model=model_full,
        **strategy_params
    )
    baseline_result = run_backtest(test_df, strategy_full, ExecutionConfig(), verbose=False)
    
    if verbose:
        print(f"  Baseline (full data):")
        print(f"    Train rows: {len(train_df):,}")
        print(f"    Test rows: {len(test_df):,}")
        print(f"    t-stat: {baseline_result['metrics']['t_stat']:.2f}")
        print(f"    PnL: ${baseline_result['metrics']['total_pnl']:.2f}")
    
    # Filter to observed-only
    train_observed = train_df.copy()
    test_observed = test_df.copy()
    
    if 'cl_ffill' in train_observed.columns:
        train_observed = train_observed[train_observed['cl_ffill'] == 0]
    if 'pm_ffill' in train_observed.columns:
        train_observed = train_observed[train_observed['pm_ffill'] == 0]
    
    if 'cl_ffill' in test_observed.columns:
        test_observed = test_observed[test_observed['cl_ffill'] == 0]
    if 'pm_ffill' in test_observed.columns:
        test_observed = test_observed[test_observed['pm_ffill'] == 0]
    
    if verbose:
        print(f"\n  Observed-only:")
        print(f"    Train rows: {len(train_observed):,} ({len(train_observed)/len(train_df)*100:.1f}%)")
        print(f"    Test rows: {len(test_observed):,} ({len(test_observed)/len(test_df)*100:.1f}%)")
    
    # Check if we have enough data
    if len(train_observed) < 1000 or len(test_observed) < 500:
        return {
            'error': 'Insufficient observed-only data',
            'train_observed_rows': len(train_observed),
            'test_observed_rows': len(test_observed),
        }
    
    # Fit model on observed-only
    model_observed = BinnedFairValueModel(sample_every=5)
    model_observed.fit(train_observed)
    
    # Run strategy on observed-only test
    strategy_observed = MispricingBasedStrategy(
        fair_value_model=model_observed,
        **strategy_params
    )
    observed_result = run_backtest(test_observed, strategy_observed, ExecutionConfig(), verbose=False)
    
    if verbose:
        print(f"    t-stat: {observed_result['metrics']['t_stat']:.2f}")
        print(f"    PnL: ${observed_result['metrics']['total_pnl']:.2f}")
        print(f"    Trades: {observed_result['metrics']['n_trades']}")
    
    results = {
        'baseline': {
            'train_rows': len(train_df),
            'test_rows': len(test_df),
            'n_trades': baseline_result['metrics']['n_trades'],
            't_stat': baseline_result['metrics']['t_stat'],
            'total_pnl': baseline_result['metrics']['total_pnl'],
        },
        'observed_only': {
            'train_rows': len(train_observed),
            'test_rows': len(test_observed),
            'train_pct': len(train_observed) / len(train_df) * 100,
            'test_pct': len(test_observed) / len(test_df) * 100,
            'n_trades': observed_result['metrics']['n_trades'],
            't_stat': observed_result['metrics']['t_stat'],
            'total_pnl': observed_result['metrics']['total_pnl'],
        },
        't_stat_change': baseline_result['metrics']['t_stat'] - observed_result['metrics']['t_stat'],
        'edge_persists': observed_result['metrics']['t_stat'] > 1.5,
    }
    
    return results


# ==============================================================================
# 6.4.2: COARSER INFORMATION STRESS
# ==============================================================================

def run_coarser_info_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    strategy_params: Dict,
    update_intervals: List[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test strategy with coarser prediction updates.
    
    Computes p_hat only every N seconds and holds constant between updates.
    Tests if strategy requires micro-timing (bad) or works with slower updates.
    
    Args:
        train_df: Training data
        test_df: Test data
        strategy_params: Strategy parameters
        update_intervals: List of update intervals in seconds
        verbose: Print progress
        
    Returns:
        Dict with results for each update interval
    """
    update_intervals = update_intervals or [1, 5, 10, 15, 30]
    
    if verbose:
        print("\nRunning Coarser Information Stress Test...")
    
    # Fit model on full training data
    model = BinnedFairValueModel(sample_every=5)
    model.fit(train_df)
    
    results = {
        'update_intervals': [],
    }
    
    for interval in update_intervals:
        # Create modified test data with stale predictions
        test_coarse = test_df.copy()
        
        if interval > 1:
            # Sample predictions only at update intervals
            # For rows not at update intervals, forward-fill p_hat
            test_coarse['_update_flag'] = (test_coarse['t'] % interval == 0).astype(int)
            
            # Compute p_hat for all rows first
            p_hat_full = model.predict(test_coarse)
            
            # Mask non-update rows
            p_hat_masked = np.where(test_coarse['_update_flag'] == 1, p_hat_full, np.nan)
            
            # Forward fill within each market
            test_coarse['p_hat_coarse'] = p_hat_masked
            test_coarse['p_hat_coarse'] = test_coarse.groupby('market_id')['p_hat_coarse'].ffill()
            test_coarse['p_hat_coarse'] = test_coarse['p_hat_coarse'].fillna(0.5)
            
            # Create a modified model that uses pre-computed coarse predictions
            # Store the coarse predictions indexed by market_id and t
            coarse_preds_dict = {}
            for _, row in test_coarse.iterrows():
                key = (row['market_id'], int(row['t']))
                coarse_preds_dict[key] = row['p_hat_coarse']
            
            class CoarseModel:
                def __init__(self, preds_dict, global_p=0.5):
                    self.preds_dict = preds_dict
                    self.fitted = True
                    self.global_p = global_p
                    self.bin_stats = {}
                
                def predict(self, df):
                    # Return predictions for the given DataFrame
                    preds = []
                    for _, row in df.iterrows():
                        key = (row['market_id'], int(row['t']))
                        preds.append(self.preds_dict.get(key, self.global_p))
                    return np.array(preds)
            
            coarse_model = CoarseModel(coarse_preds_dict)
        else:
            coarse_model = model
        
        # Run strategy with coarse model
        strategy = MispricingBasedStrategy(
            fair_value_model=coarse_model if interval > 1 else model,
            **strategy_params
        )
        
        result = run_backtest(test_coarse if interval > 1 else test_df, 
                             strategy, ExecutionConfig(), verbose=False)
        metrics = result['metrics']
        
        interval_result = {
            'interval': interval,
            'n_trades': metrics['n_trades'],
            't_stat': metrics['t_stat'],
            'total_pnl': metrics['total_pnl'],
            'mean_pnl': metrics['mean_pnl_per_market'],
        }
        results['update_intervals'].append(interval_result)
        
        if verbose:
            print(f"  Update every {interval}s: t={metrics['t_stat']:.2f}, "
                  f"trades={metrics['n_trades']}")
    
    # Assess degradation
    baseline = results['update_intervals'][0]  # 1s interval
    results['baseline_t_stat'] = baseline['t_stat']
    
    # Check if edge survives with 5s and 10s updates
    for r in results['update_intervals']:
        if r['interval'] == 5:
            results['survives_5s'] = r['t_stat'] > 1.5
            results['t_stat_at_5s'] = r['t_stat']
        if r['interval'] == 10:
            results['survives_10s'] = r['t_stat'] > 1.5
            results['t_stat_at_10s'] = r['t_stat']
    
    return results


# ==============================================================================
# 6.4.3: CALIBRATION DIAGNOSTICS
# ==============================================================================

def run_calibration_diagnostics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    strategy_params: Dict = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive calibration diagnostics on test set.
    
    Includes:
    - Reliability diagram (predicted vs actual)
    - Brier score by tau bucket
    - Calibration by delta_bps deciles
    - Calibration by volatility deciles
    
    Args:
        train_df: Training data
        test_df: Test data
        strategy_params: Strategy parameters (optional)
        verbose: Print progress
        
    Returns:
        Dict with calibration diagnostics
    """
    if verbose:
        print("\nRunning Calibration Diagnostics...")
    
    # Fit model
    model = BinnedFairValueModel(sample_every=5)
    model.fit(train_df)
    
    # Get predictions on test set
    p_hat = model.predict(test_df)
    
    # Get true outcomes (Y is same for all rows in a market)
    # Try to get Y from dataframe, or derive from K and settlement if available
    if 'Y' in test_df.columns:
        market_outcomes = test_df.groupby('market_id')['Y'].first()
        Y = test_df['market_id'].map(market_outcomes).values
    else:
        # Y column doesn't exist - try to derive from K and settlement
        if 'K' in test_df.columns and 'settlement' in test_df.columns:
            # Derive Y: Y=1 if settlement >= K, else Y=0
            market_k = test_df.groupby('market_id')['K'].first()
            market_settlement = test_df.groupby('market_id')['settlement'].first()
            market_outcomes = pd.Series(
                np.where(
                    (market_k.notna()) & (market_settlement.notna()),
                    (market_settlement >= market_k).astype(float),
                    np.nan
                ),
                index=market_k.index
            )
            Y = test_df['market_id'].map(market_outcomes).values
        else:
            # No way to get Y
            market_outcomes = pd.Series(index=test_df['market_id'].unique(), dtype=float)
            Y = np.full(len(test_df), np.nan)
    
    # Filter out NaN
    valid = ~(np.isnan(p_hat) | np.isnan(Y))
    p_hat_valid = p_hat[valid]
    Y_valid = Y[valid]
    
    if len(p_hat_valid) == 0:
        if verbose:
            print(f"  [WARN] No valid predictions (p_hat NaN: {np.isnan(p_hat).sum()}, Y NaN: {np.isnan(Y).sum()})")
        return {'error': 'No valid predictions', 'p_hat_nan': int(np.isnan(p_hat).sum()), 'Y_nan': int(np.isnan(Y).sum())}
    
    if verbose:
        print(f"  Valid predictions: {len(p_hat_valid):,}")
    
    results = {}
    
    # 1. Overall calibration metrics
    brier = compute_brier_score(Y_valid, p_hat_valid)
    ece = compute_expected_calibration_error(Y_valid, p_hat_valid)
    
    results['overall'] = {
        'brier_score': float(brier),
        'ece': float(ece),
        'mean_prediction': float(p_hat_valid.mean()),
        'mean_outcome': float(Y_valid.mean()),
    }
    
    if verbose:
        print(f"  Overall Brier: {brier:.4f}, ECE: {ece:.4f}")
    
    # 2. Reliability diagram
    bin_centers, actual_rates, bin_counts = compute_calibration_curve(Y_valid, p_hat_valid, n_bins=10)
    results['reliability_curve'] = {
        'bin_centers': bin_centers.tolist(),
        'actual_rates': actual_rates.tolist(),
        'bin_counts': bin_counts.tolist(),
    }
    
    # 3. Brier score by tau bucket
    test_valid = test_df[valid].copy()
    test_valid['p_hat'] = p_hat_valid
    test_valid['Y_label'] = Y_valid
    
    tau_buckets = [
        (0, 60, '0-60s'),
        (60, 120, '60-120s'),
        (120, 180, '120-180s'),
        (180, 240, '180-240s'),
        (240, 300, '240-300s'),
    ]
    
    brier_by_tau = []
    for tau_min, tau_max, label in tau_buckets:
        mask = (test_valid['tau'] >= tau_min) & (test_valid['tau'] < tau_max)
        if mask.sum() > 100:
            bucket_brier = compute_brier_score(
                test_valid.loc[mask, 'Y_label'].values,
                test_valid.loc[mask, 'p_hat'].values
            )
            brier_by_tau.append({
                'tau_bucket': label,
                'tau_min': tau_min,
                'tau_max': tau_max,
                'brier_score': float(bucket_brier),
                'n_samples': int(mask.sum()),
            })
            if verbose:
                print(f"  Tau {label}: Brier = {bucket_brier:.4f}, n = {mask.sum()}")
    
    results['brier_by_tau'] = brier_by_tau
    
    # 4. Calibration by delta_bps deciles
    test_valid['delta_decile'] = pd.qcut(
        test_valid['delta_bps'].clip(-200, 200), 
        q=10, 
        labels=False,
        duplicates='drop'
    )
    
    calibration_by_delta = []
    for decile in test_valid['delta_decile'].dropna().unique():
        mask = test_valid['delta_decile'] == decile
        if mask.sum() > 100:
            decile_brier = compute_brier_score(
                test_valid.loc[mask, 'Y_label'].values,
                test_valid.loc[mask, 'p_hat'].values
            )
            mean_delta = test_valid.loc[mask, 'delta_bps'].mean()
            calibration_by_delta.append({
                'decile': int(decile),
                'mean_delta_bps': float(mean_delta),
                'brier_score': float(decile_brier),
                'n_samples': int(mask.sum()),
            })
    
    results['calibration_by_delta'] = sorted(calibration_by_delta, key=lambda x: x['decile'])
    
    # 5. Calibration by volatility deciles
    if 'realized_vol_bps' in test_valid.columns:
        vol_valid = test_valid['realized_vol_bps'].dropna()
        if len(vol_valid) > 100:
            test_valid['vol_decile'] = pd.qcut(
                test_valid['realized_vol_bps'].fillna(vol_valid.median()),
                q=5,
                labels=False,
                duplicates='drop'
            )
            
            calibration_by_vol = []
            for decile in test_valid['vol_decile'].dropna().unique():
                mask = test_valid['vol_decile'] == decile
                if mask.sum() > 100:
                    decile_brier = compute_brier_score(
                        test_valid.loc[mask, 'Y_label'].values,
                        test_valid.loc[mask, 'p_hat'].values
                    )
                    mean_vol = test_valid.loc[mask, 'realized_vol_bps'].mean()
                    calibration_by_vol.append({
                        'decile': int(decile),
                        'mean_vol_bps': float(mean_vol),
                        'brier_score': float(decile_brier),
                        'n_samples': int(mask.sum()),
                    })
            
            results['calibration_by_vol'] = sorted(calibration_by_vol, key=lambda x: x['decile'])
    
    # 6. Check if model is "always predict near 0.5" or actually varying
    results['prediction_stats'] = {
        'mean': float(p_hat_valid.mean()),
        'std': float(p_hat_valid.std()),
        'min': float(p_hat_valid.min()),
        'max': float(p_hat_valid.max()),
        'pct_below_0.3': float((p_hat_valid < 0.3).mean()),
        'pct_above_0.7': float((p_hat_valid > 0.7).mean()),
    }
    
    if verbose:
        print(f"\n  Prediction distribution:")
        print(f"    Mean: {results['prediction_stats']['mean']:.3f}")
        print(f"    Std: {results['prediction_stats']['std']:.3f}")
        print(f"    % below 0.3: {results['prediction_stats']['pct_below_0.3']:.1%}")
        print(f"    % above 0.7: {results['prediction_stats']['pct_above_0.7']:.1%}")
    
    return results


# ==============================================================================
# COMBINED MODEL STRESS SUITE
# ==============================================================================

def run_model_stress_suite(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    strategy_params: Dict,
    output_dir: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete model stress test suite.
    
    Args:
        train_df: Training data
        test_df: Test data
        strategy_params: Strategy parameters
        output_dir: Directory to save results
        verbose: Print progress
        
    Returns:
        Dict with all model stress test results
    """
    print("\n" + "="*70)
    print("PHASE 6.4: MODEL RISK TESTS")
    print("="*70)
    
    results = {}
    
    # 6.4.1: Observed-Only Test
    print("\n--- 6.4.1: Observed-Only Training and Trading ---")
    observed_results = run_observed_only_test(
        train_df, test_df, strategy_params, verbose=verbose
    )
    results['observed_only'] = observed_results
    
    # 6.4.2: Coarser Information Test
    print("\n--- 6.4.2: Coarser Information Stress ---")
    coarse_results = run_coarser_info_test(
        train_df, test_df, strategy_params,
        update_intervals=[1, 5, 10, 15, 30],
        verbose=verbose
    )
    results['coarser_info'] = coarse_results
    
    # 6.4.3: Calibration Diagnostics
    print("\n--- 6.4.3: Calibration Diagnostics ---")
    calibration_results = run_calibration_diagnostics(
        train_df, test_df, strategy_params, verbose=verbose
    )
    results['calibration'] = calibration_results
    
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
        
        with open(output_dir / 'model_stress_results.json', 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)
        
        print(f"\nResults saved to {output_dir}")
    
    # Summary
    print("\n" + "="*70)
    print("MODEL STRESS TEST SUMMARY")
    print("="*70)
    
    print("\nObserved-Only Test:")
    if 'error' in observed_results:
        print(f"  {observed_results['error']}")
    else:
        print(f"  Baseline t-stat: {observed_results['baseline']['t_stat']:.2f}")
        print(f"  Observed-only t-stat: {observed_results['observed_only']['t_stat']:.2f}")
        if observed_results['edge_persists']:
            print("  [PASS] Edge persists with observed-only data")
        else:
            print("  [WARN] Edge may require forward-filled data")
    
    print("\nCoarser Information:")
    print(f"  Baseline (1s): t={coarse_results['baseline_t_stat']:.2f}")
    if coarse_results.get('survives_5s'):
        print(f"  5s updates: t={coarse_results.get('t_stat_at_5s', 0):.2f} [PASS]")
    else:
        print(f"  5s updates: t={coarse_results.get('t_stat_at_5s', 0):.2f} [WARN]")
    if coarse_results.get('survives_10s'):
        print(f"  10s updates: t={coarse_results.get('t_stat_at_10s', 0):.2f} [PASS]")
    else:
        print(f"  10s updates: t={coarse_results.get('t_stat_at_10s', 0):.2f} [WARN]")
    
    print("\nCalibration:")
    if 'overall' in calibration_results:
        print(f"  Overall Brier: {calibration_results['overall']['brier_score']:.4f}")
        print(f"  ECE: {calibration_results['overall']['ece']:.4f}")
        if calibration_results['overall']['brier_score'] < 0.25:
            print("  [PASS] Model is reasonably well-calibrated")
        else:
            print("  [WARN] Model calibration could be improved")
    else:
        print("  [WARN] Calibration data not available")
    
    return results


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run model stress tests')
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
    results = run_model_stress_suite(
        train_df=train_df,
        test_df=test_df,
        strategy_params=strategy_params,
        output_dir=Path(args.output_dir)
    )

