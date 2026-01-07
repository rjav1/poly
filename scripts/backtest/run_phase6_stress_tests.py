#!/usr/bin/env python3
"""
Phase 6: High-Value Stress Tests Master Runner

Runs all Phase 6 stress tests in sequence:
- 6.1: Data-snooping / selection bias
- 6.2: Execution realism (slippage, quote fade, capacity)
- 6.3: Robustness (LOMO, winsorization)
- 6.4: Model risk (observed-only, coarser info, calibration)
- 6.5: Regime stress (conditional performance, transitions)

Generates consolidated report.

Usage:
    python scripts/backtest/run_phase6_stress_tests.py
    python scripts/backtest/run_phase6_stress_tests.py --quick  # Fewer simulations
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest.data_loader import (
    load_unified_eth_markets,
    add_derived_columns, 
    get_train_test_split
)
from scripts.backtest.fair_value import BinnedFairValueModel
from scripts.backtest.test_mispricing_strategy import (
    run_nested_selection_validation,
    run_spa_test
)
from scripts.backtest.execution_stress import run_execution_stress_suite
from scripts.backtest.robustness_stress import run_robustness_stress_suite
from scripts.backtest.model_stress import run_model_stress_suite
from scripts.backtest.regime_stress import run_regime_stress_suite

# Output directory
OUTPUT_DIR = PROJECT_ROOT / 'data_v2' / 'backtest_results' / 'stress_tests'


def convert_to_serializable(obj):
    """Convert numpy types to Python natives for JSON serialization."""
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
    elif pd.isna(obj):
        return None
    return obj


def run_phase6_stress_tests(
    min_coverage: float = 90.0,
    output_dir: Optional[Path] = None,
    n_slippage_sims: int = 1000,
    n_spa_bootstrap: int = 1000,
    n_nested_folds: int = 5,
    quick_mode: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run all Phase 6 stress tests.
    
    Args:
        min_coverage: Minimum market coverage
        output_dir: Output directory
        n_slippage_sims: Number of slippage Monte Carlo simulations
        n_spa_bootstrap: Number of SPA bootstrap samples
        n_nested_folds: Number of nested selection folds
        quick_mode: Run fewer simulations for speed
        verbose: Print progress
        
    Returns:
        Dict with all Phase 6 results
    """
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if quick_mode:
        n_slippage_sims = 200
        n_spa_bootstrap = 200
        n_nested_folds = 3
    
    print("="*80)
    print("PHASE 6: HIGH-VALUE STRESS TESTS")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Output directory: {output_dir}")
    print(f"Quick mode: {quick_mode}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'quick_mode': quick_mode,
        'config': {
            'n_slippage_sims': n_slippage_sims,
            'n_spa_bootstrap': n_spa_bootstrap,
            'n_nested_folds': n_nested_folds,
        }
    }
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Use unified loader: merges L1-era and L6-era datasets
    # Markets with L6 use depth-aware execution; L1-only markets treat L2-L6 as empty
    df, market_info = load_unified_eth_markets(min_coverage=min_coverage, prefer_6levels=True)
    df = add_derived_columns(df)
    
    train_df, test_df, train_ids, test_ids = get_train_test_split(df, train_frac=0.7)
    
    print(f"Total markets: {df['market_id'].nunique()}")
    print(f"Train markets: {len(train_ids)}")
    print(f"Test markets: {len(test_ids)}")
    
    results['data_info'] = {
        'n_total_markets': int(df['market_id'].nunique()),
        'n_train_markets': len(train_ids),
        'n_test_markets': len(test_ids),
        'n_observations': len(df),
    }
    
    # Default strategy params
    strategy_params = {
        'buffer': 0.02,
        'tau_max': 420,
        'min_tau': 0,
        'cooldown': 30,
        'exit_rule': 'expiry',
    }
    
    # Parameter grid for selection bias tests
    param_grid = {
        'buffer': [0.01, 0.015, 0.02, 0.025, 0.03],
        'tau_max': [300, 420, 600],
        'min_tau': [0],
        'cooldown': [30],
        'exit_rule': ['expiry'],
    }
    
    # =========================================================================
    # 6.1: DATA-SNOOPING / SELECTION BIAS
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 6.1: DATA-SNOOPING / SELECTION BIAS")
    print("="*70)
    
    # Nested selection
    print("\n--- 6.1a: Nested Selection Validation ---")
    nested_results = run_nested_selection_validation(
        df=df,
        param_grid=param_grid,
        n_outer_folds=n_nested_folds,
        verbose=verbose
    )
    results['nested_selection'] = nested_results
    
    # SPA test
    print("\n--- 6.1b: Superior Predictive Ability (SPA) Test ---")
    spa_results = run_spa_test(
        test_df=test_df,
        train_df=train_df,
        param_grid=param_grid,
        n_bootstrap=n_spa_bootstrap,
        verbose=verbose
    )
    results['spa_test'] = spa_results
    
    # =========================================================================
    # 6.2: EXECUTION REALISM
    # =========================================================================
    exec_results = run_execution_stress_suite(
        test_df=test_df,
        train_df=train_df,
        strategy_params=strategy_params,
        output_dir=output_dir,
        n_slippage_sims=n_slippage_sims,
        verbose=verbose
    )
    results['execution_stress'] = exec_results
    
    # =========================================================================
    # 6.3: ROBUSTNESS (LOMO, WINSORIZATION)
    # =========================================================================
    robust_results = run_robustness_stress_suite(
        test_df=test_df,
        train_df=train_df,
        strategy_params=strategy_params,
        output_dir=output_dir,
        verbose=verbose
    )
    results['robustness_stress'] = robust_results
    
    # =========================================================================
    # 6.4: MODEL RISK
    # =========================================================================
    model_results = run_model_stress_suite(
        train_df=train_df,
        test_df=test_df,
        strategy_params=strategy_params,
        output_dir=output_dir,
        verbose=verbose
    )
    results['model_stress'] = model_results
    
    # =========================================================================
    # 6.5: REGIME STRESS
    # =========================================================================
    regime_results = run_regime_stress_suite(
        test_df=test_df,
        train_df=train_df,
        strategy_params=strategy_params,
        output_dir=output_dir,
        verbose=verbose
    )
    results['regime_stress'] = regime_results
    
    # =========================================================================
    # CONSOLIDATED SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 6 CONSOLIDATED SUMMARY")
    print("="*80)
    
    # Extract key metrics
    summary = {
        'selection_bias': {
            'nested_median_t': nested_results.get('outer_t_stat_distribution', {}).get('median', None),
            'spa_p_value': spa_results.get('spa_p_value', None),
            'spa_significant_005': spa_results.get('significant_at_005', None),
        },
        'execution': {
            'slippage_prob_t_gt_2': exec_results.get('slippage_uniform', {}).get('prob_t_stat_gt_2', None),
            'quote_fade_1s_t_stat': next((r['t_stat'] for r in exec_results.get('quote_fade', []) if r['fade_delay'] == 1), None),
            'quote_fade_2s_t_stat': next((r['t_stat'] for r in exec_results.get('quote_fade', []) if r['fade_delay'] == 2), None),
        },
        'robustness': {
            'lomo_min_t': robust_results.get('lomo', {}).get('lomo_stats', {}).get('min_t_stat', None),
            'lomo_median_t': robust_results.get('lomo', {}).get('lomo_stats', {}).get('median_t_stat', None),
            'winsor_95_t': next((r['t_stat'] for r in robust_results.get('winsorization', {}).get('percentile_results', []) if r['percentile'] == 95), None),
        },
        'model': {
            'observed_only_t': model_results.get('observed_only', {}).get('observed_only', {}).get('t_stat', None),
            'coarse_5s_t': model_results.get('coarser_info', {}).get('t_stat_at_5s', None),
            'brier_score': model_results.get('calibration', {}).get('overall', {}).get('brier_score', None),
        },
    }
    
    results['summary'] = summary
    
    # Print summary
    print("\n1. SELECTION BIAS:")
    if summary['selection_bias']['nested_median_t']:
        status = "PASS" if summary['selection_bias']['nested_median_t'] > 2.0 else "WARN"
        print(f"   Nested selection median t: {summary['selection_bias']['nested_median_t']:.2f} [{status}]")
    if summary['selection_bias']['spa_p_value']:
        status = "PASS" if summary['selection_bias']['spa_significant_005'] else "WARN"
        print(f"   SPA p-value: {summary['selection_bias']['spa_p_value']:.4f} [{status}]")
    
    print("\n2. EXECUTION REALISM:")
    if summary['execution']['slippage_prob_t_gt_2']:
        status = "PASS" if summary['execution']['slippage_prob_t_gt_2'] > 0.7 else "WARN"
        print(f"   P(t>2) with slippage: {summary['execution']['slippage_prob_t_gt_2']:.1%} [{status}]")
    if summary['execution']['quote_fade_2s_t_stat']:
        status = "PASS" if summary['execution']['quote_fade_2s_t_stat'] > 1.5 else "WARN"
        print(f"   2s quote fade t-stat: {summary['execution']['quote_fade_2s_t_stat']:.2f} [{status}]")
    
    print("\n3. ROBUSTNESS:")
    if summary['robustness']['lomo_min_t']:
        status = "PASS" if summary['robustness']['lomo_min_t'] > 1.0 else "WARN"
        print(f"   LOMO min t-stat: {summary['robustness']['lomo_min_t']:.2f} [{status}]")
    if summary['robustness']['lomo_median_t']:
        print(f"   LOMO median t-stat: {summary['robustness']['lomo_median_t']:.2f}")
    if summary['robustness']['winsor_95_t']:
        status = "PASS" if summary['robustness']['winsor_95_t'] > 1.5 else "WARN"
        print(f"   Winsorized (95th) t-stat: {summary['robustness']['winsor_95_t']:.2f} [{status}]")
    
    print("\n4. MODEL RISK:")
    if summary['model']['observed_only_t']:
        status = "PASS" if summary['model']['observed_only_t'] > 1.5 else "WARN"
        print(f"   Observed-only t-stat: {summary['model']['observed_only_t']:.2f} [{status}]")
    if summary['model']['coarse_5s_t']:
        status = "PASS" if summary['model']['coarse_5s_t'] > 1.5 else "WARN"
        print(f"   5s update t-stat: {summary['model']['coarse_5s_t']:.2f} [{status}]")
    if summary['model']['brier_score']:
        status = "PASS" if summary['model']['brier_score'] < 0.25 else "WARN"
        print(f"   Brier score: {summary['model']['brier_score']:.4f} [{status}]")
    
    # Overall verdict
    print("\n" + "="*70)
    print("OVERALL VERDICT")
    print("="*70)
    
    passes = 0
    warns = 0
    
    if summary['selection_bias']['spa_significant_005']:
        passes += 1
    else:
        warns += 1
    
    if summary['execution']['slippage_prob_t_gt_2'] and summary['execution']['slippage_prob_t_gt_2'] > 0.7:
        passes += 1
    else:
        warns += 1
    
    if summary['robustness']['lomo_min_t'] and summary['robustness']['lomo_min_t'] > 1.0:
        passes += 1
    else:
        warns += 1
    
    if summary['model']['observed_only_t'] and summary['model']['observed_only_t'] > 1.5:
        passes += 1
    else:
        warns += 1
    
    print(f"\nCritical tests passed: {passes}/4")
    print(f"Warnings: {warns}/4")
    
    if passes >= 3:
        print("\n[STRONG] Strategy passes most critical stress tests")
        results['overall_verdict'] = 'STRONG'
    elif passes >= 2:
        print("\n[MODERATE] Strategy passes some stress tests - use caution")
        results['overall_verdict'] = 'MODERATE'
    else:
        print("\n[WEAK] Strategy fails most stress tests - high risk of false positive")
        results['overall_verdict'] = 'WEAK'
    
    # Save results
    with open(output_dir / 'phase6_consolidated_results.json', 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    # Generate markdown report
    generate_phase6_report(results, output_dir)
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


def generate_phase6_report(results: Dict, output_dir: Path):
    """Generate markdown report for Phase 6 results."""
    
    report_lines = [
        "# Phase 6: High-Value Stress Tests Report",
        "",
        f"**Generated:** {results['timestamp']}",
        f"**Quick Mode:** {results['quick_mode']}",
        "",
        "## Executive Summary",
        "",
        f"**Overall Verdict:** {results.get('overall_verdict', 'N/A')}",
        "",
        "## 1. Data-Snooping / Selection Bias (6.1)",
        "",
        "### Nested Selection Validation",
        "",
    ]
    
    if 'nested_selection' in results:
        ns = results['nested_selection']
        if 'outer_t_stat_distribution' in ns:
            dist = ns['outer_t_stat_distribution']
            report_lines.extend([
                f"- **Median outer t-stat:** {dist.get('median', 'N/A'):.2f}",
                f"- **Min outer t-stat:** {dist.get('min', 'N/A'):.2f}",
                f"- **Max outer t-stat:** {dist.get('max', 'N/A'):.2f}",
                f"- **P(t > 2.0):** {ns.get('prob_outer_t_gt_2', 0):.1%}",
                "",
            ])
    
    report_lines.extend([
        "### SPA Test",
        "",
    ])
    
    if 'spa_test' in results:
        spa = results['spa_test']
        report_lines.extend([
            f"- **SPA p-value:** {spa.get('spa_p_value', 'N/A'):.4f}",
            f"- **Unadjusted p-value:** {spa.get('unadjusted_p_value', 'N/A'):.4f}",
            f"- **Selection bias adjustment:** +{spa.get('selection_bias_adjustment', 0):.4f}",
            f"- **Significant at 5%:** {spa.get('significant_at_005', 'N/A')}",
            "",
        ])
    
    report_lines.extend([
        "## 2. Execution Realism (6.2)",
        "",
        "### Slippage Monte Carlo",
        "",
    ])
    
    if 'execution_stress' in results:
        exec_s = results['execution_stress']
        if 'slippage_uniform' in exec_s:
            slip = exec_s['slippage_uniform']
            report_lines.extend([
                f"- **P(t > 2.0) with uniform slippage:** {slip.get('prob_t_stat_gt_2', 0):.1%}",
                f"- **Median t-stat:** {slip.get('t_stat_distribution', {}).get('percentile_50', 'N/A'):.2f}",
                "",
            ])
        
        if 'quote_fade' in exec_s:
            report_lines.append("### Quote Fade\n")
            for row in exec_s['quote_fade']:
                report_lines.append(f"- **{row['fade_delay']}s delay:** t={row['t_stat']:.2f}")
            report_lines.append("")
    
    report_lines.extend([
        "## 3. Robustness (6.3)",
        "",
        "### Leave-One-Market-Out",
        "",
    ])
    
    if 'robustness_stress' in results:
        rob = results['robustness_stress']
        if 'lomo' in rob:
            lomo = rob['lomo']
            stats = lomo.get('lomo_stats', {})
            report_lines.extend([
                f"- **Min t-stat:** {stats.get('min_t_stat', 'N/A'):.2f}",
                f"- **Median t-stat:** {stats.get('median_t_stat', 'N/A'):.2f}",
                f"- **Max t-stat:** {stats.get('max_t_stat', 'N/A'):.2f}",
                "",
            ])
        
        if 'winsorization' in rob:
            report_lines.append("### Winsorization\n")
            for r in rob['winsorization'].get('percentile_results', []):
                report_lines.append(f"- **{r['percentile']}th percentile:** t={r['t_stat']:.2f}")
            report_lines.append("")
    
    report_lines.extend([
        "## 4. Model Risk (6.4)",
        "",
    ])
    
    if 'model_stress' in results:
        mod = results['model_stress']
        if 'observed_only' in mod and 'observed_only' in mod['observed_only']:
            obs = mod['observed_only']['observed_only']
            report_lines.extend([
                "### Observed-Only Test",
                "",
                f"- **Observed-only t-stat:** {obs.get('t_stat', 'N/A'):.2f}",
                f"- **Edge persists:** {mod['observed_only'].get('edge_persists', 'N/A')}",
                "",
            ])
        
        if 'calibration' in mod:
            cal = mod['calibration']
            brier = cal.get('overall', {}).get('brier_score', None)
            ece = cal.get('overall', {}).get('ece', None)
            report_lines.extend([
                "### Calibration",
                "",
                f"- **Brier score:** {brier:.4f}" if brier is not None else "- **Brier score:** N/A",
                f"- **ECE:** {ece:.4f}" if ece is not None else "- **ECE:** N/A",
                "",
            ])
    
    report_lines.extend([
        "## 5. Regime Stress (6.5)",
        "",
    ])
    
    if 'regime_stress' in results:
        reg = results['regime_stress']
        if 'conditional_performance' in reg:
            cond = reg['conditional_performance']
            if 'summary' in cond:
                report_lines.extend([
                    f"- **Best tau bucket:** {cond['summary'].get('best_tau_bucket', 'N/A')}",
                    f"- **Worst tau bucket:** {cond['summary'].get('worst_tau_bucket', 'N/A')}",
                    "",
                ])
    
    report_lines.extend([
        "## Conclusion",
        "",
        f"The strategy receives an overall verdict of **{results.get('overall_verdict', 'N/A')}**.",
        "",
        "### Recommendations",
        "",
    ])
    
    if results.get('overall_verdict') == 'STRONG':
        report_lines.extend([
            "- Strategy appears robust to major stress tests",
            "- Consider proceeding with live testing with small size",
            "- Monitor for regime changes not captured in test data",
        ])
    elif results.get('overall_verdict') == 'MODERATE':
        report_lines.extend([
            "- Strategy shows promise but has weaknesses",
            "- Collect more market data before scaling",
            "- Focus on improving areas that failed tests",
        ])
    else:
        report_lines.extend([
            "- Strategy may be a false positive",
            "- Significant risk of data-snooping or overfitting",
            "- Do not deploy without substantial improvements",
        ])
    
    report_path = output_dir / 'PHASE6_REPORT.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Report saved to: {report_path}")


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Phase 6 stress tests')
    parser.add_argument('--min-coverage', type=float, default=90.0,
                       help='Minimum market coverage')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR),
                       help='Output directory')
    parser.add_argument('--quick', action='store_true',
                       help='Run in quick mode (fewer simulations)')
    parser.add_argument('--n-slippage-sims', type=int, default=1000,
                       help='Number of slippage simulations')
    parser.add_argument('--n-spa-bootstrap', type=int, default=1000,
                       help='Number of SPA bootstrap samples')
    
    args = parser.parse_args()
    
    results = run_phase6_stress_tests(
        min_coverage=args.min_coverage,
        output_dir=Path(args.output_dir),
        n_slippage_sims=args.n_slippage_sims,
        n_spa_bootstrap=args.n_spa_bootstrap,
        quick_mode=args.quick
    )

