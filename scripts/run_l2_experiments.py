#!/usr/bin/env python3
"""
L2 Queue Model Experiments

This script runs the 3 key experiments for the L2 queue model upgrade:

1. PnL Sign Robustness: Run conservative/base/optimistic L2 models to check if sign is consistent
2. L1 vs L2 Quoting: Compare quoting at L1 (best) vs L2 (second best) for fill rate and AS
3. Imbalance Filter On/Off: Test if imbalance filter reduces AS without killing fills

These experiments validate whether the L2 upgrade produces more robust results.

Success Criteria:
- PnL sign robust: Conservative/base models agree on sign
- PnL range narrow: ±$285 → ±$10-20 (or better)
- Fill rate reasonable: 1-5% (not 0% or 50%)
- AS filters work: Imbalance filter reduces AS without killing fills
- L2 quoting viable: L2 quotes have lower AS than L1
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    experiment_name: str
    variant: str
    total_pnl: float
    t_stat: float
    n_fills: int
    fill_rate: float
    spread_captured: float
    adverse_selection: float
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment_name': self.experiment_name,
            'variant': self.variant,
            'total_pnl': self.total_pnl,
            't_stat': self.t_stat,
            'n_fills': self.n_fills,
            'fill_rate': self.fill_rate,
            'spread_captured': self.spread_captured,
            'adverse_selection': self.adverse_selection,
            'parameters': self.parameters,
        }


def run_experiment_1_pnl_sign_robustness(
    df: pd.DataFrame,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Experiment 1: PnL Sign Robustness
    
    Run same strategy with conservative/base/optimistic L2 models.
    Check if PnL sign is consistent across all three.
    
    Args:
        df: Market data with 6-level columns
        verbose: Print progress
        
    Returns:
        Dictionary with experiment results
    """
    from scripts.backtest.maker_execution import MakerExecutionConfig, FillModel
    from scripts.backtest.backtest_engine import run_maker_backtest
    from scripts.backtest.strategies import SpreadCaptureStrategy
    
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 1: PNL SIGN ROBUSTNESS")
        print("="*70)
    
    # Base strategy
    strategy = SpreadCaptureStrategy(
        spread_min=0.01,
        tau_min=60,
        tau_max=600,
        inventory_limit_up=10.0,
        inventory_limit_down=10.0,
        quote_size=1.0,
    )
    
    results = []
    
    # L2 Conservative
    config_cons = MakerExecutionConfig(
        place_latency_ms=100,
        cancel_latency_ms=50,
        fill_model=FillModel.L2_QUEUE,
        l2_conservative_mode=True,
        l2_allow_level_drift=False,
        l2_optimistic_disappear=False,
    )
    
    if verbose:
        print("\n  Running L2 Conservative...")
    result_cons = run_maker_backtest(df, strategy, config_cons, verbose=False)
    results.append(ExperimentResult(
        experiment_name='PnL_Sign_Robustness',
        variant='L2_CONSERVATIVE',
        total_pnl=result_cons['metrics']['total_pnl'],
        t_stat=result_cons['metrics']['t_stat'],
        n_fills=result_cons['metrics']['n_fills'],
        fill_rate=result_cons['metrics']['fill_rate'],
        spread_captured=result_cons['metrics']['spread_captured_total'],
        adverse_selection=result_cons['metrics']['adverse_selection_total'],
        parameters={'l2_mode': 'conservative'},
    ))
    
    # L2 Base
    config_base = MakerExecutionConfig(
        place_latency_ms=100,
        cancel_latency_ms=50,
        fill_model=FillModel.L2_QUEUE,
        l2_conservative_mode=False,
        l2_allow_level_drift=True,
        l2_optimistic_disappear=False,
    )
    
    if verbose:
        print("  Running L2 Base...")
    result_base = run_maker_backtest(df, strategy, config_base, verbose=False)
    results.append(ExperimentResult(
        experiment_name='PnL_Sign_Robustness',
        variant='L2_BASE',
        total_pnl=result_base['metrics']['total_pnl'],
        t_stat=result_base['metrics']['t_stat'],
        n_fills=result_base['metrics']['n_fills'],
        fill_rate=result_base['metrics']['fill_rate'],
        spread_captured=result_base['metrics']['spread_captured_total'],
        adverse_selection=result_base['metrics']['adverse_selection_total'],
        parameters={'l2_mode': 'base'},
    ))
    
    # L2 Optimistic
    config_opt = MakerExecutionConfig(
        place_latency_ms=100,
        cancel_latency_ms=50,
        fill_model=FillModel.L2_QUEUE,
        l2_conservative_mode=False,
        l2_allow_level_drift=True,
        l2_optimistic_disappear=True,
    )
    
    if verbose:
        print("  Running L2 Optimistic...")
    result_opt = run_maker_backtest(df, strategy, config_opt, verbose=False)
    results.append(ExperimentResult(
        experiment_name='PnL_Sign_Robustness',
        variant='L2_OPTIMISTIC',
        total_pnl=result_opt['metrics']['total_pnl'],
        t_stat=result_opt['metrics']['t_stat'],
        n_fills=result_opt['metrics']['n_fills'],
        fill_rate=result_opt['metrics']['fill_rate'],
        spread_captured=result_opt['metrics']['spread_captured_total'],
        adverse_selection=result_opt['metrics']['adverse_selection_total'],
        parameters={'l2_mode': 'optimistic'},
    ))
    
    # Analyze results
    pnls = [r.total_pnl for r in results]
    signs = [np.sign(p) for p in pnls]
    sign_robust = len(set(signs)) == 1
    conservative_base_agree = np.sign(pnls[0]) == np.sign(pnls[1])
    
    analysis = {
        'pnl_range': (min(pnls), max(pnls)),
        'pnl_range_abs': max(pnls) - min(pnls),
        'sign_robust': sign_robust,
        'conservative_base_agree': conservative_base_agree,
        'all_positive': min(pnls) > 0,
        'all_negative': max(pnls) < 0,
    }
    
    if verbose:
        print("\n  RESULTS:")
        print(f"    {'Variant':<20} {'PnL':>10} {'t-stat':>8} {'Fills':>8} {'Fill%':>8}")
        print("    " + "-"*60)
        for r in results:
            print(f"    {r.variant:<20} ${r.total_pnl:>9.2f} {r.t_stat:>8.2f} {r.n_fills:>8} {r.fill_rate*100:>7.1f}%")
        
        print(f"\n  ANALYSIS:")
        print(f"    PnL Range: ${analysis['pnl_range'][0]:.2f} to ${analysis['pnl_range'][1]:.2f}")
        print(f"    PnL Range (abs): ${analysis['pnl_range_abs']:.2f}")
        print(f"    Sign Robust: {analysis['sign_robust']}")
        print(f"    Conservative/Base Agree: {analysis['conservative_base_agree']}")
        
        if analysis['sign_robust']:
            status = "[PASS]" if analysis['all_positive'] else "[FAIL]"
            outcome = "positive" if analysis['all_positive'] else "negative"
            print(f"\n  {status} PnL sign is ROBUST ({outcome} under all models)")
        else:
            print(f"\n  [WARN] PnL sign NOT robust - varies by fill model")
    
    return {
        'results': [r.to_dict() for r in results],
        'analysis': analysis,
    }


def run_experiment_2_l1_vs_l2_quoting(
    df: pd.DataFrame,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Experiment 2: L1 vs L2 Quoting
    
    Compare quoting at L1 (best bid/ask) vs L2 (second best).
    L2 quoting should have:
    - Lower fill rate (less favorable price)
    - Lower adverse selection (less informed flow)
    
    Args:
        df: Market data with 6-level columns
        verbose: Print progress
        
    Returns:
        Dictionary with experiment results
    """
    from scripts.backtest.maker_execution import MakerExecutionConfig, FillModel
    from scripts.backtest.backtest_engine import run_maker_backtest
    from scripts.backtest.strategies import SpreadCaptureStrategy
    
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 2: L1 VS L2 QUOTING")
        print("="*70)
    
    # Use L2_QUEUE with base settings for consistency
    config = MakerExecutionConfig(
        place_latency_ms=100,
        cancel_latency_ms=50,
        fill_model=FillModel.L2_QUEUE,
        l2_conservative_mode=False,
        l2_allow_level_drift=True,
        l2_optimistic_disappear=False,
    )
    
    results = []
    
    # L1 quoting (quote_level=1)
    strategy_l1 = SpreadCaptureStrategy(
        spread_min=0.01,
        tau_min=60,
        tau_max=600,
        inventory_limit_up=10.0,
        inventory_limit_down=10.0,
        quote_level=1,  # L1 (best)
    )
    
    if verbose:
        print("\n  Running L1 Quoting (best bid/ask)...")
    result_l1 = run_maker_backtest(df, strategy_l1, config, verbose=False)
    results.append(ExperimentResult(
        experiment_name='L1_vs_L2_Quoting',
        variant='L1_QUOTING',
        total_pnl=result_l1['metrics']['total_pnl'],
        t_stat=result_l1['metrics']['t_stat'],
        n_fills=result_l1['metrics']['n_fills'],
        fill_rate=result_l1['metrics']['fill_rate'],
        spread_captured=result_l1['metrics']['spread_captured_total'],
        adverse_selection=result_l1['metrics']['adverse_selection_total'],
        parameters={'quote_level': 1},
    ))
    
    # L2 quoting (quote_level=2)
    strategy_l2 = SpreadCaptureStrategy(
        spread_min=0.01,
        tau_min=60,
        tau_max=600,
        inventory_limit_up=10.0,
        inventory_limit_down=10.0,
        quote_level=2,  # L2 (second best)
    )
    
    if verbose:
        print("  Running L2 Quoting (second best)...")
    result_l2 = run_maker_backtest(df, strategy_l2, config, verbose=False)
    results.append(ExperimentResult(
        experiment_name='L1_vs_L2_Quoting',
        variant='L2_QUOTING',
        total_pnl=result_l2['metrics']['total_pnl'],
        t_stat=result_l2['metrics']['t_stat'],
        n_fills=result_l2['metrics']['n_fills'],
        fill_rate=result_l2['metrics']['fill_rate'],
        spread_captured=result_l2['metrics']['spread_captured_total'],
        adverse_selection=result_l2['metrics']['adverse_selection_total'],
        parameters={'quote_level': 2},
    ))
    
    # Analysis
    l1_res = results[0]
    l2_res = results[1]
    
    fill_rate_delta = l2_res.fill_rate - l1_res.fill_rate
    as_delta = l2_res.adverse_selection - l1_res.adverse_selection
    pnl_delta = l2_res.total_pnl - l1_res.total_pnl
    
    # Calculate per-fill metrics
    l1_as_per_fill = l1_res.adverse_selection / l1_res.n_fills if l1_res.n_fills > 0 else 0
    l2_as_per_fill = l2_res.adverse_selection / l2_res.n_fills if l2_res.n_fills > 0 else 0
    
    analysis = {
        'l1_pnl': l1_res.total_pnl,
        'l2_pnl': l2_res.total_pnl,
        'pnl_delta': pnl_delta,
        'l1_fill_rate': l1_res.fill_rate,
        'l2_fill_rate': l2_res.fill_rate,
        'fill_rate_delta': fill_rate_delta,
        'l1_as_total': l1_res.adverse_selection,
        'l2_as_total': l2_res.adverse_selection,
        'as_delta': as_delta,
        'l1_as_per_fill': l1_as_per_fill,
        'l2_as_per_fill': l2_as_per_fill,
        'l2_lower_as_per_fill': l2_as_per_fill < l1_as_per_fill,
        'l2_viable': l2_res.n_fills > 0 and l2_res.total_pnl > 0,
    }
    
    if verbose:
        print("\n  RESULTS:")
        print(f"    {'Variant':<15} {'PnL':>10} {'Fills':>8} {'Fill%':>8} {'AS Total':>10} {'AS/Fill':>10}")
        print("    " + "-"*65)
        for r in results:
            as_per = r.adverse_selection / r.n_fills if r.n_fills > 0 else 0
            print(f"    {r.variant:<15} ${r.total_pnl:>9.2f} {r.n_fills:>8} {r.fill_rate*100:>7.1f}% ${r.adverse_selection:>9.2f} ${as_per:>9.4f}")
        
        print(f"\n  ANALYSIS:")
        print(f"    L2 vs L1 PnL Delta: ${pnl_delta:.2f}")
        print(f"    L2 vs L1 Fill Rate Delta: {fill_rate_delta*100:.2f}%")
        print(f"    L2 vs L1 AS/Fill: ${l2_as_per_fill:.4f} vs ${l1_as_per_fill:.4f}")
        
        if analysis['l2_lower_as_per_fill']:
            print(f"\n  [GOOD] L2 quoting has LOWER adverse selection per fill")
        else:
            print(f"\n  [INFO] L2 quoting does NOT have lower AS per fill")
        
        if analysis['l2_viable']:
            print(f"  [PASS] L2 quoting is viable (fills > 0, PnL > 0)")
        else:
            print(f"  [WARN] L2 quoting may not be viable")
    
    return {
        'results': [r.to_dict() for r in results],
        'analysis': analysis,
    }


def run_experiment_3_imbalance_filter(
    df: pd.DataFrame,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Experiment 3: Imbalance Filter On/Off
    
    Test if order book imbalance filter reduces adverse selection
    without killing fill rate.
    
    Pre-registered threshold: |imbalance| < 0.3
    
    Args:
        df: Market data with 6-level columns
        verbose: Print progress
        
    Returns:
        Dictionary with experiment results
    """
    from scripts.backtest.maker_execution import MakerExecutionConfig, FillModel
    from scripts.backtest.backtest_engine import run_maker_backtest
    from scripts.backtest.strategies import SpreadCaptureStrategy
    
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 3: IMBALANCE FILTER ON/OFF")
        print("="*70)
    
    # Use L2_QUEUE with base settings
    config = MakerExecutionConfig(
        place_latency_ms=100,
        cancel_latency_ms=50,
        fill_model=FillModel.L2_QUEUE,
        l2_conservative_mode=False,
        l2_allow_level_drift=True,
        l2_optimistic_disappear=False,
    )
    
    results = []
    
    # Imbalance filter OFF
    strategy_off = SpreadCaptureStrategy(
        spread_min=0.01,
        tau_min=60,
        tau_max=600,
        inventory_limit_up=10.0,
        inventory_limit_down=10.0,
        imbalance_filter_enabled=False,
    )
    
    if verbose:
        print("\n  Running WITHOUT imbalance filter...")
    result_off = run_maker_backtest(df, strategy_off, config, verbose=False)
    results.append(ExperimentResult(
        experiment_name='Imbalance_Filter',
        variant='FILTER_OFF',
        total_pnl=result_off['metrics']['total_pnl'],
        t_stat=result_off['metrics']['t_stat'],
        n_fills=result_off['metrics']['n_fills'],
        fill_rate=result_off['metrics']['fill_rate'],
        spread_captured=result_off['metrics']['spread_captured_total'],
        adverse_selection=result_off['metrics']['adverse_selection_total'],
        parameters={'imbalance_filter_enabled': False},
    ))
    
    # Imbalance filter ON (threshold = 0.3)
    strategy_on = SpreadCaptureStrategy(
        spread_min=0.01,
        tau_min=60,
        tau_max=600,
        inventory_limit_up=10.0,
        inventory_limit_down=10.0,
        imbalance_filter_enabled=True,
        imbalance_threshold=0.3,
        imbalance_levels=3,  # Use 3-level imbalance
    )
    
    if verbose:
        print("  Running WITH imbalance filter (threshold=0.3)...")
    result_on = run_maker_backtest(df, strategy_on, config, verbose=False)
    results.append(ExperimentResult(
        experiment_name='Imbalance_Filter',
        variant='FILTER_ON',
        total_pnl=result_on['metrics']['total_pnl'],
        t_stat=result_on['metrics']['t_stat'],
        n_fills=result_on['metrics']['n_fills'],
        fill_rate=result_on['metrics']['fill_rate'],
        spread_captured=result_on['metrics']['spread_captured_total'],
        adverse_selection=result_on['metrics']['adverse_selection_total'],
        parameters={'imbalance_filter_enabled': True, 'imbalance_threshold': 0.3},
    ))
    
    # Analysis
    off_res = results[0]
    on_res = results[1]
    
    fill_reduction = (off_res.n_fills - on_res.n_fills) / off_res.n_fills if off_res.n_fills > 0 else 0
    as_reduction = (off_res.adverse_selection - on_res.adverse_selection) / abs(off_res.adverse_selection) if off_res.adverse_selection != 0 else 0
    pnl_delta = on_res.total_pnl - off_res.total_pnl
    
    # Per-fill AS
    off_as_per = off_res.adverse_selection / off_res.n_fills if off_res.n_fills > 0 else 0
    on_as_per = on_res.adverse_selection / on_res.n_fills if on_res.n_fills > 0 else 0
    
    analysis = {
        'off_pnl': off_res.total_pnl,
        'on_pnl': on_res.total_pnl,
        'pnl_delta': pnl_delta,
        'off_fills': off_res.n_fills,
        'on_fills': on_res.n_fills,
        'fill_reduction_pct': fill_reduction * 100,
        'off_as': off_res.adverse_selection,
        'on_as': on_res.adverse_selection,
        'as_reduction_pct': as_reduction * 100,
        'off_as_per_fill': off_as_per,
        'on_as_per_fill': on_as_per,
        'filter_effective': on_as_per < off_as_per,
        'filter_not_too_aggressive': on_res.n_fills > off_res.n_fills * 0.5,  # Keeps >50% of fills
    }
    
    if verbose:
        print("\n  RESULTS:")
        print(f"    {'Variant':<15} {'PnL':>10} {'Fills':>8} {'AS Total':>10} {'AS/Fill':>10}")
        print("    " + "-"*55)
        for r in results:
            as_per = r.adverse_selection / r.n_fills if r.n_fills > 0 else 0
            print(f"    {r.variant:<15} ${r.total_pnl:>9.2f} {r.n_fills:>8} ${r.adverse_selection:>9.2f} ${as_per:>9.4f}")
        
        print(f"\n  ANALYSIS:")
        print(f"    Fill Reduction: {analysis['fill_reduction_pct']:.1f}%")
        print(f"    AS Reduction: {analysis['as_reduction_pct']:.1f}%")
        print(f"    AS/Fill (off): ${off_as_per:.4f}")
        print(f"    AS/Fill (on): ${on_as_per:.4f}")
        print(f"    PnL Delta: ${pnl_delta:.2f}")
        
        if analysis['filter_effective']:
            print(f"\n  [GOOD] Imbalance filter REDUCES adverse selection per fill")
        else:
            print(f"\n  [INFO] Imbalance filter does NOT reduce AS per fill")
        
        if analysis['filter_not_too_aggressive']:
            print(f"  [PASS] Filter keeps {100-analysis['fill_reduction_pct']:.1f}% of fills (not too aggressive)")
        else:
            print(f"  [WARN] Filter is too aggressive (>50% fill reduction)")
    
    return {
        'results': [r.to_dict() for r in results],
        'analysis': analysis,
    }


def run_all_experiments(verbose: bool = True) -> Dict[str, Any]:
    """
    Run all 3 L2 experiments and produce comprehensive report.
    
    Args:
        verbose: Print progress
        
    Returns:
        Dictionary with all experiment results
    """
    from scripts.backtest.data_loader import load_6level_markets, add_derived_columns
    
    print("\n" + "="*70)
    print("L2 QUEUE MODEL EXPERIMENTS")
    print("="*70)
    
    # Load 6-level data
    print("\nLoading 6-level data...")
    try:
        df, market_info = load_6level_markets(min_coverage=90.0)
    except Exception as e:
        print(f"[ERROR] Failed to load 6-level data: {e}")
        return {'error': str(e)}
    
    df = add_derived_columns(df)
    
    print(f"Loaded {len(df):,} rows, {df['market_id'].nunique()} markets")
    
    # Check for 6-level columns
    l2_cols = [c for c in df.columns if 'bid_2' in c or 'ask_2' in c]
    print(f"6-level columns: {len(l2_cols)}")
    
    if len(l2_cols) == 0:
        print("[ERROR] No 6-level data found!")
        return {'error': 'No 6-level data'}
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'n_markets': df['market_id'].nunique(),
        'n_rows': len(df),
    }
    
    # Experiment 1: PnL Sign Robustness
    exp1 = run_experiment_1_pnl_sign_robustness(df, verbose=verbose)
    all_results['experiment_1_pnl_sign_robustness'] = exp1
    
    # Experiment 2: L1 vs L2 Quoting
    exp2 = run_experiment_2_l1_vs_l2_quoting(df, verbose=verbose)
    all_results['experiment_2_l1_vs_l2_quoting'] = exp2
    
    # Experiment 3: Imbalance Filter
    exp3 = run_experiment_3_imbalance_filter(df, verbose=verbose)
    all_results['experiment_3_imbalance_filter'] = exp3
    
    # Overall Summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    exp1_analysis = exp1.get('analysis', {})
    exp2_analysis = exp2.get('analysis', {})
    exp3_analysis = exp3.get('analysis', {})
    
    print(f"\n  Experiment 1 - PnL Sign Robustness:")
    print(f"    Sign Robust: {exp1_analysis.get('sign_robust', False)}")
    print(f"    PnL Range: ${exp1_analysis.get('pnl_range', (0,0))[0]:.2f} to ${exp1_analysis.get('pnl_range', (0,0))[1]:.2f}")
    
    print(f"\n  Experiment 2 - L1 vs L2 Quoting:")
    print(f"    L2 Lower AS/Fill: {exp2_analysis.get('l2_lower_as_per_fill', False)}")
    print(f"    L2 Viable: {exp2_analysis.get('l2_viable', False)}")
    
    print(f"\n  Experiment 3 - Imbalance Filter:")
    print(f"    Filter Effective: {exp3_analysis.get('filter_effective', False)}")
    print(f"    Not Too Aggressive: {exp3_analysis.get('filter_not_too_aggressive', False)}")
    
    # Success criteria check
    success = {
        'pnl_sign_robust': exp1_analysis.get('conservative_base_agree', False),
        'pnl_range_narrow': exp1_analysis.get('pnl_range_abs', float('inf')) < 50,
        'l2_quoting_viable': exp2_analysis.get('l2_viable', False),
        'imbalance_filter_works': (
            exp3_analysis.get('filter_effective', False) and 
            exp3_analysis.get('filter_not_too_aggressive', False)
        ),
    }
    
    all_results['success_criteria'] = success
    
    print(f"\n  SUCCESS CRITERIA:")
    for criterion, passed in success.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"    {status} {criterion}")
    
    overall_pass = sum(success.values()) >= 2  # At least 2/4 pass
    all_results['overall_pass'] = overall_pass
    
    if overall_pass:
        print(f"\n  [PASS] L2 upgrade shows promising results")
    else:
        print(f"\n  [WARN] L2 upgrade needs more investigation")
    
    return all_results


def save_results(results: Dict[str, Any], output_dir: Path = None):
    """Save experiment results to file."""
    if output_dir is None:
        output_dir = project_root / 'data_v2' / 'backtest_results' / 'l2_experiments'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_dir / 'l2_experiment_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {json_path}")


def main():
    """Run all experiments and save results."""
    results = run_all_experiments(verbose=True)
    save_results(results)


if __name__ == '__main__':
    main()

