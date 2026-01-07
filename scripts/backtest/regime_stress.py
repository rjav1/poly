#!/usr/bin/env python3
"""
Phase 6.5: Regime Stress Tests (Conditional Performance)

Tests to understand regime dependence:
- Conditional performance surfaces (vol/delta/tau)
- Regime transition analysis

These tests identify if edge exists broadly or only in narrow corners.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest.data_loader import load_eth_markets, add_derived_columns, get_train_test_split
from scripts.backtest.fair_value import BinnedFairValueModel
from scripts.backtest.strategies import MispricingBasedStrategy
from scripts.backtest.backtest_engine import run_backtest, ExecutionConfig


# ==============================================================================
# 6.5.1: CONDITIONAL PERFORMANCE SURFACES
# ==============================================================================

def run_conditional_performance_analysis(
    test_df: pd.DataFrame,
    fair_value_model,
    strategy_params: Dict,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute test PnL/trade as function of regime variables.
    
    Analyzes performance by:
    - Realized vol deciles
    - |delta_bps| deciles
    - tau buckets
    
    Args:
        test_df: Test data
        fair_value_model: Fitted fair value model
        strategy_params: Strategy parameters
        verbose: Print progress
        
    Returns:
        Dict with conditional performance surfaces
    """
    if verbose:
        print("\nRunning Conditional Performance Analysis...")
    
    # Run strategy to get trades
    strategy = MispricingBasedStrategy(
        fair_value_model=fair_value_model,
        **strategy_params
    )
    result = run_backtest(test_df, strategy, ExecutionConfig(), verbose=False)
    trades = result['trades']
    
    if len(trades) == 0:
        return {'error': 'No trades to analyze'}
    
    if verbose:
        print(f"  Analyzing {len(trades)} trades")
    
    # Create trades DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Get entry row data for each trade
    for i, trade in enumerate(trades):
        market_df = test_df[test_df['market_id'] == trade['market_id']]
        entry_row = market_df[market_df['t'] == trade['entry_t']]
        
        if not entry_row.empty:
            entry_row = entry_row.iloc[0]
            trades_df.loc[i, 'tau'] = entry_row.get('tau', np.nan)
            trades_df.loc[i, 'delta_bps'] = entry_row.get('delta_bps', np.nan)
            trades_df.loc[i, 'abs_delta_bps'] = abs(entry_row.get('delta_bps', np.nan))
            trades_df.loc[i, 'realized_vol'] = entry_row.get('realized_vol_bps', np.nan)
    
    results = {}
    
    # 1. Performance by tau bucket
    tau_buckets = [
        (0, 60, '0-60s'),
        (60, 120, '60-120s'),
        (120, 180, '120-180s'),
        (180, 240, '180-240s'),
        (240, 300, '240-300s'),
        (300, 420, '300-420s'),
    ]
    
    perf_by_tau = []
    for tau_min, tau_max, label in tau_buckets:
        mask = (trades_df['tau'] >= tau_min) & (trades_df['tau'] < tau_max)
        if mask.sum() > 0:
            bucket_pnl = trades_df.loc[mask, 'pnl'].sum()
            bucket_mean = trades_df.loc[mask, 'pnl'].mean()
            bucket_std = trades_df.loc[mask, 'pnl'].std()
            n_trades = mask.sum()
            t_stat = bucket_mean / (bucket_std / np.sqrt(n_trades)) if bucket_std > 0 and n_trades > 1 else 0
            
            perf_by_tau.append({
                'tau_bucket': label,
                'tau_min': tau_min,
                'tau_max': tau_max,
                'n_trades': int(n_trades),
                'total_pnl': float(bucket_pnl),
                'mean_pnl': float(bucket_mean),
                't_stat': float(t_stat) if n_trades > 3 else np.nan,
                'hit_rate': float((trades_df.loc[mask, 'pnl'] > 0).mean()),
            })
            
            if verbose:
                print(f"  Tau {label}: n={n_trades}, mean_pnl=${bucket_mean:.3f}, t={t_stat:.2f}")
    
    results['perf_by_tau'] = perf_by_tau
    
    # 2. Performance by |delta_bps| deciles
    valid_delta = trades_df['abs_delta_bps'].dropna()
    if len(valid_delta) > 10:
        try:
            trades_df['delta_decile'] = pd.qcut(
                trades_df['abs_delta_bps'].fillna(valid_delta.median()),
                q=5,
                labels=False,
                duplicates='drop'
            )
            
            perf_by_delta = []
            for decile in sorted(trades_df['delta_decile'].dropna().unique()):
                mask = trades_df['delta_decile'] == decile
                if mask.sum() > 0:
                    bucket_mean = trades_df.loc[mask, 'pnl'].mean()
                    n_trades = mask.sum()
                    mean_delta = trades_df.loc[mask, 'abs_delta_bps'].mean()
                    
                    perf_by_delta.append({
                        'decile': int(decile),
                        'mean_abs_delta_bps': float(mean_delta),
                        'n_trades': int(n_trades),
                        'mean_pnl': float(bucket_mean),
                        'total_pnl': float(trades_df.loc[mask, 'pnl'].sum()),
                        'hit_rate': float((trades_df.loc[mask, 'pnl'] > 0).mean()),
                    })
            
            results['perf_by_delta'] = perf_by_delta
            
            if verbose:
                print("\n  By |delta_bps| quintile:")
                for r in perf_by_delta:
                    print(f"    Q{r['decile']}: delta={r['mean_abs_delta_bps']:.1f}bps, "
                          f"n={r['n_trades']}, mean=${r['mean_pnl']:.3f}")
        except Exception as e:
            results['perf_by_delta'] = {'error': str(e)}
    
    # 3. Performance by realized vol deciles
    valid_vol = trades_df['realized_vol'].dropna()
    if len(valid_vol) > 10:
        try:
            trades_df['vol_decile'] = pd.qcut(
                trades_df['realized_vol'].fillna(valid_vol.median()),
                q=5,
                labels=False,
                duplicates='drop'
            )
            
            perf_by_vol = []
            for decile in sorted(trades_df['vol_decile'].dropna().unique()):
                mask = trades_df['vol_decile'] == decile
                if mask.sum() > 0:
                    bucket_mean = trades_df.loc[mask, 'pnl'].mean()
                    n_trades = mask.sum()
                    mean_vol = trades_df.loc[mask, 'realized_vol'].mean()
                    
                    perf_by_vol.append({
                        'decile': int(decile),
                        'mean_vol_bps': float(mean_vol),
                        'n_trades': int(n_trades),
                        'mean_pnl': float(bucket_mean),
                        'total_pnl': float(trades_df.loc[mask, 'pnl'].sum()),
                        'hit_rate': float((trades_df.loc[mask, 'pnl'] > 0).mean()),
                    })
            
            results['perf_by_vol'] = perf_by_vol
            
            if verbose:
                print("\n  By realized vol quintile:")
                for r in perf_by_vol:
                    print(f"    Q{r['decile']}: vol={r['mean_vol_bps']:.1f}bps, "
                          f"n={r['n_trades']}, mean=${r['mean_pnl']:.3f}")
        except Exception as e:
            results['perf_by_vol'] = {'error': str(e)}
    
    # 4. 2D heatmap: tau x vol
    if 'vol_decile' in trades_df.columns and len(perf_by_tau) > 0:
        heatmap_data = []
        for tau_min, tau_max, tau_label in tau_buckets:
            tau_mask = (trades_df['tau'] >= tau_min) & (trades_df['tau'] < tau_max)
            for vol_decile in sorted(trades_df['vol_decile'].dropna().unique()):
                vol_mask = trades_df['vol_decile'] == vol_decile
                combined_mask = tau_mask & vol_mask
                
                if combined_mask.sum() > 0:
                    heatmap_data.append({
                        'tau_bucket': tau_label,
                        'vol_decile': int(vol_decile),
                        'n_trades': int(combined_mask.sum()),
                        'mean_pnl': float(trades_df.loc[combined_mask, 'pnl'].mean()),
                        'total_pnl': float(trades_df.loc[combined_mask, 'pnl'].sum()),
                    })
        
        results['heatmap_tau_vol'] = heatmap_data
    
    # Summary statistics
    results['summary'] = {
        'n_trades': len(trades),
        'total_pnl': float(trades_df['pnl'].sum()),
        'mean_pnl': float(trades_df['pnl'].mean()),
        'std_pnl': float(trades_df['pnl'].std()),
        'best_tau_bucket': max(perf_by_tau, key=lambda x: x['mean_pnl'])['tau_bucket'] if perf_by_tau else None,
        'worst_tau_bucket': min(perf_by_tau, key=lambda x: x['mean_pnl'])['tau_bucket'] if perf_by_tau else None,
    }
    
    return results


# ==============================================================================
# 6.5.2: REGIME TRANSITION ANALYSIS
# ==============================================================================

def run_regime_transition_analysis(
    test_df: pd.DataFrame,
    fair_value_model,
    strategy_params: Dict,
    vol_lookback: int = 60,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze performance by volatility regime transitions.
    
    Compares:
    - Stable vol vs increasing vol vs decreasing vol
    - Low->high transitions vs high->low transitions
    
    Args:
        test_df: Test data
        fair_value_model: Fitted fair value model
        strategy_params: Strategy parameters
        vol_lookback: Lookback for volatility trend calculation
        verbose: Print progress
        
    Returns:
        Dict with regime transition analysis
    """
    if verbose:
        print("\nRunning Regime Transition Analysis...")
    
    # Run strategy to get trades
    strategy = MispricingBasedStrategy(
        fair_value_model=fair_value_model,
        **strategy_params
    )
    result = run_backtest(test_df, strategy, ExecutionConfig(), verbose=False)
    trades = result['trades']
    
    if len(trades) == 0:
        return {'error': 'No trades to analyze'}
    
    # Compute volatility trend for each market
    test_df = test_df.copy()
    if 'realized_vol_bps' in test_df.columns:
        test_df['vol_change'] = test_df.groupby('market_id')['realized_vol_bps'].diff(vol_lookback)
    else:
        test_df['vol_change'] = 0
    
    # Create trades DataFrame with regime info
    trades_df = pd.DataFrame(trades)
    
    for i, trade in enumerate(trades):
        market_df = test_df[test_df['market_id'] == trade['market_id']]
        entry_row = market_df[market_df['t'] == trade['entry_t']]
        
        if not entry_row.empty:
            entry_row = entry_row.iloc[0]
            vol_change = entry_row.get('vol_change', 0)
            
            # Classify regime
            if pd.isna(vol_change) or abs(vol_change) < 1:  # Threshold for "stable"
                regime = 'stable'
            elif vol_change > 0:
                regime = 'increasing'
            else:
                regime = 'decreasing'
            
            trades_df.loc[i, 'vol_change'] = vol_change
            trades_df.loc[i, 'vol_regime'] = regime
    
    results = {}
    
    # Performance by regime
    perf_by_regime = []
    for regime in ['stable', 'increasing', 'decreasing']:
        mask = trades_df['vol_regime'] == regime
        if mask.sum() > 0:
            bucket_pnl = trades_df.loc[mask, 'pnl'].sum()
            bucket_mean = trades_df.loc[mask, 'pnl'].mean()
            bucket_std = trades_df.loc[mask, 'pnl'].std()
            n_trades = mask.sum()
            t_stat = bucket_mean / (bucket_std / np.sqrt(n_trades)) if bucket_std > 0 and n_trades > 1 else 0
            
            perf_by_regime.append({
                'regime': regime,
                'n_trades': int(n_trades),
                'total_pnl': float(bucket_pnl),
                'mean_pnl': float(bucket_mean),
                't_stat': float(t_stat) if n_trades > 3 else np.nan,
                'hit_rate': float((trades_df.loc[mask, 'pnl'] > 0).mean()),
            })
            
            if verbose:
                print(f"  {regime.capitalize()} vol: n={n_trades}, "
                      f"mean_pnl=${bucket_mean:.3f}, hit={perf_by_regime[-1]['hit_rate']:.1%}")
    
    results['perf_by_regime'] = perf_by_regime
    
    # Best/worst regime
    if perf_by_regime:
        best = max(perf_by_regime, key=lambda x: x['mean_pnl'])
        worst = min(perf_by_regime, key=lambda x: x['mean_pnl'])
        results['best_regime'] = best['regime']
        results['worst_regime'] = worst['regime']
        results['regime_spread'] = best['mean_pnl'] - worst['mean_pnl']
    
    return results


# ==============================================================================
# COMBINED REGIME STRESS SUITE
# ==============================================================================

def run_regime_stress_suite(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    strategy_params: Dict,
    output_dir: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete regime stress test suite.
    
    Args:
        test_df: Test data
        train_df: Training data
        strategy_params: Strategy parameters
        output_dir: Directory to save results
        verbose: Print progress
        
    Returns:
        Dict with all regime stress test results
    """
    print("\n" + "="*70)
    print("PHASE 6.5: REGIME STRESS TESTS")
    print("="*70)
    
    # Fit model
    model = BinnedFairValueModel(sample_every=5)
    model.fit(train_df)
    
    results = {}
    
    # 6.5.1: Conditional Performance Surfaces
    print("\n--- 6.5.1: Conditional Performance Surfaces ---")
    conditional_results = run_conditional_performance_analysis(
        test_df, model, strategy_params, verbose=verbose
    )
    results['conditional_performance'] = conditional_results
    
    # 6.5.2: Regime Transition Analysis
    print("\n--- 6.5.2: Regime Transition Analysis ---")
    transition_results = run_regime_transition_analysis(
        test_df, model, strategy_params, verbose=verbose
    )
    results['regime_transitions'] = transition_results
    
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
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            elif pd.isna(obj):
                return None
            return obj
        
        with open(output_dir / 'regime_stress_results.json', 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)
        
        # Save conditional performance as CSV
        if 'perf_by_tau' in conditional_results:
            pd.DataFrame(conditional_results['perf_by_tau']).to_csv(
                output_dir / 'perf_by_tau.csv', index=False
            )
        
        print(f"\nResults saved to {output_dir}")
    
    # Summary
    print("\n" + "="*70)
    print("REGIME STRESS TEST SUMMARY")
    print("="*70)
    
    print("\nConditional Performance:")
    if 'perf_by_tau' in conditional_results:
        print("  By tau bucket:")
        for r in conditional_results['perf_by_tau']:
            status = "+" if r['mean_pnl'] > 0 else "-"
            print(f"    {r['tau_bucket']}: {status}${abs(r['mean_pnl']):.3f}/trade, "
                  f"n={r['n_trades']}")
    
    if 'summary' in conditional_results:
        print(f"\n  Best tau: {conditional_results['summary'].get('best_tau_bucket', 'N/A')}")
        print(f"  Worst tau: {conditional_results['summary'].get('worst_tau_bucket', 'N/A')}")
    
    print("\nRegime Transitions:")
    if 'perf_by_regime' in transition_results:
        for r in transition_results['perf_by_regime']:
            status = "+" if r['mean_pnl'] > 0 else "-"
            print(f"  {r['regime'].capitalize()}: {status}${abs(r['mean_pnl']):.3f}/trade, "
                  f"n={r['n_trades']}")
    
    if 'best_regime' in transition_results:
        print(f"\n  Best regime: {transition_results['best_regime']}")
        print(f"  Regime spread: ${transition_results.get('regime_spread', 0):.3f}/trade")
    
    return results


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run regime stress tests')
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
    results = run_regime_stress_suite(
        test_df=test_df,
        train_df=train_df,
        strategy_params=strategy_params,
        output_dir=Path(args.output_dir)
    )

