#!/usr/bin/env python3
"""
Comprehensive Spread Capture Strategy Analysis

This script runs the full analysis pipeline for the spread capture / maker strategy:
1. Data audit and validation
2. Backtest with diagnostics
3. Parameter sweep
4. Latency cliff analysis
5. Placebo tests
6. Stress tests
7. Report generation
8. Visualizations

Usage:
    python scripts/run_spread_capture_analysis.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.backtest.data_loader import load_eth_markets, add_derived_columns
from scripts.backtest.strategies import SpreadCaptureStrategy
from scripts.backtest.backtest_engine import run_maker_backtest
from scripts.backtest.maker_execution import MakerExecutionConfig, FillModel
from scripts.backtest.maker_diagnostics import generate_diagnostics_summary, print_diagnostics_report
from scripts.backtest.parameter_sweep import run_maker_parameter_sweep, analyze_maker_sweep_results
from scripts.backtest.latency_cliff import run_maker_latency_sweep, print_maker_latency_report
from scripts.backtest.placebo_tests import run_all_maker_placebo_tests, print_maker_placebo_report
from scripts.backtest.stress_tests import run_all_stress_tests, print_stress_test_report
from scripts.backtest.generate_report import generate_maker_report
from scripts.backtest.visualizations import save_maker_plots


def main():
    print("="*80)
    print("SPREAD CAPTURE / MAKER STRATEGY ANALYSIS")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Output directory
    output_dir = project_root / 'data_v2' / 'backtest_results' / 'maker_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    print("\n" + "-"*60)
    print("1. LOADING DATA")
    print("-"*60)
    
    df, market_info = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    n_markets = df['market_id'].nunique()
    n_obs = len(df)
    
    print(f"  Markets: {n_markets}")
    print(f"  Observations: {n_obs:,}")
    
    # Check for volume markets
    volume_prefixes = ['20260106_1630', '20260106_1645', '20260106_1700', '20260106_1715',
                       '20260106_1730', '20260106_1745', '20260106_1800', '20260106_1815',
                       '20260106_1830', '20260106_1845', '20260106_1900', '20260106_1915']
    volume_markets = [m for m in df['market_id'].unique() if any(m.startswith(p) for p in volume_prefixes)]
    print(f"  Volume markets: {len(volume_markets)}")
    
    data_summary = {
        'n_markets': n_markets,
        'n_obs': n_obs,
        'volume_markets': len(volume_markets),
    }
    
    # =========================================================================
    # 2. Define Strategy
    # =========================================================================
    print("\n" + "-"*60)
    print("2. STRATEGY CONFIGURATION")
    print("-"*60)
    
    strategy = SpreadCaptureStrategy(
        spread_min=0.01,  # 1 cent minimum spread
        tau_min=60,       # Don't quote in last 60 seconds
        tau_max=600,      # Don't quote too early
        inventory_limit_up=10.0,
        inventory_limit_down=10.0,
        tau_flatten=60,
        quote_size=1.0,
        two_sided=True,
        adverse_selection_filter=True,
        cl_jump_threshold_bps=10.0,
    )
    
    config = MakerExecutionConfig(
        place_latency_ms=100,
        cancel_latency_ms=50,
        fill_model=FillModel.TOUCH_SIZE_PROXY,
        touch_trade_rate_per_second=0.10,
    )
    
    print(f"  Strategy: {strategy.name}")
    print(f"  Config: {config.describe()}")
    for k, v in strategy.get_params().items():
        print(f"    {k}: {v}")
    
    # =========================================================================
    # 3. Run Backtest
    # =========================================================================
    print("\n" + "-"*60)
    print("3. RUNNING BACKTEST")
    print("-"*60)
    
    backtest_result = run_maker_backtest(
        df, strategy, config,
        verbose=True,
        volume_markets_only=True
    )
    
    # =========================================================================
    # 4. Generate Diagnostics
    # =========================================================================
    print("\n" + "-"*60)
    print("4. GENERATING DIAGNOSTICS")
    print("-"*60)
    
    diagnostics = generate_diagnostics_summary(backtest_result, df)
    print_diagnostics_report(diagnostics)
    
    # =========================================================================
    # 5. Parameter Sweep
    # =========================================================================
    print("\n" + "-"*60)
    print("5. PARAMETER SWEEP")
    print("-"*60)
    
    # Small sweep for quick results
    param_grid = [
        {'strategy': 'SpreadCapture', 'spread_min': s, 'tau_min': tm, 'tau_max': 600,
         'inventory_limit_up': 10.0, 'inventory_limit_down': 10.0, 'tau_flatten': 60,
         'two_sided': True, 'quote_size': 1.0, 'adverse_selection_filter': True}
        for s in [0.01, 0.015, 0.02]
        for tm in [60, 120]
    ]
    
    configs = [
        MakerExecutionConfig(place_latency_ms=0, cancel_latency_ms=0, 
                            fill_model=FillModel.TOUCH_SIZE_PROXY, touch_trade_rate_per_second=0.10),
        MakerExecutionConfig(place_latency_ms=100, cancel_latency_ms=50, 
                            fill_model=FillModel.TOUCH_SIZE_PROXY, touch_trade_rate_per_second=0.10),
    ]
    
    sweep_df = run_maker_parameter_sweep(
        df, param_grid=param_grid, configs=configs,
        volume_markets_only=True, verbose=True
    )
    
    sweep_analysis = analyze_maker_sweep_results(sweep_df)
    
    print(f"\n  Combinations tested: {sweep_analysis.get('n_combinations_tested', 0)}")
    print(f"  Profitable: {sweep_analysis.get('n_profitable', 0)}")
    print(f"  Significant (t>1.96): {sweep_analysis.get('n_significant', 0)}")
    
    if 'best_by_tstat' in sweep_analysis:
        best = sweep_analysis['best_by_tstat']
        print(f"\n  Best by t-stat:")
        print(f"    spread_min: {best.get('spread_min')}")
        print(f"    tau_min: {best.get('tau_min')}")
        print(f"    Total PnL: ${best.get('total_pnl', 0):.4f}")
        print(f"    t-stat: {best.get('t_stat', 0):.2f}")
    
    # Save sweep results
    sweep_df.to_csv(output_dir / 'parameter_sweep.csv', index=False)
    
    # =========================================================================
    # 6. Latency Cliff Analysis
    # =========================================================================
    print("\n" + "-"*60)
    print("6. LATENCY CLIFF ANALYSIS")
    print("-"*60)
    
    latency_results = run_maker_latency_sweep(
        df, strategy,
        place_latencies_ms=[0, 50, 100, 200, 500],
        cancel_latencies_ms=[0, 25, 50],
        volume_markets_only=True,
        verbose=True
    )
    
    print_maker_latency_report(latency_results)
    
    # =========================================================================
    # 7. Placebo Tests
    # =========================================================================
    print("\n" + "-"*60)
    print("7. PLACEBO TESTS")
    print("-"*60)
    
    placebo_results = run_all_maker_placebo_tests(
        df, strategy, config,
        volume_markets_only=True
    )
    
    print_maker_placebo_report(placebo_results)
    
    # =========================================================================
    # 8. Stress Tests
    # =========================================================================
    print("\n" + "-"*60)
    print("8. STRESS TESTS")
    print("-"*60)
    
    stress_results = run_all_stress_tests(
        df, strategy, config,
        volume_markets_only=True
    )
    
    print_stress_test_report(stress_results)
    
    # =========================================================================
    # 9. Generate Report
    # =========================================================================
    print("\n" + "-"*60)
    print("9. GENERATING REPORT")
    print("-"*60)
    
    report_path = generate_maker_report(
        data_summary=data_summary,
        strategy_name=strategy.name,
        strategy_params=strategy.get_params(),
        backtest_result=backtest_result,
        diagnostics=diagnostics,
        latency_results=latency_results,
        placebo_results=placebo_results,
        stress_results=stress_results,
        parameter_sweep=sweep_df,
        output_dir=output_dir,
    )
    
    print(f"  Report saved to: {report_path}")
    
    # =========================================================================
    # 10. Generate Visualizations
    # =========================================================================
    print("\n" + "-"*60)
    print("10. GENERATING VISUALIZATIONS")
    print("-"*60)
    
    plots_dir = output_dir / 'plots'
    saved_plots = save_maker_plots(
        plots_dir,
        backtest_result=backtest_result,
        diagnostics=diagnostics,
        latency_results=latency_results,
        stress_results=stress_results,
    )
    
    print(f"  Saved {len(saved_plots)} plots to {plots_dir}")
    for p in saved_plots:
        print(f"    - {Path(p).name}")
    
    # =========================================================================
    # 11. Save Results
    # =========================================================================
    print("\n" + "-"*60)
    print("11. SAVING RESULTS")
    print("-"*60)
    
    # Save full results as JSON
    def convert_for_json(obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict('records') if isinstance(obj, pd.DataFrame) else obj.to_dict()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        elif pd.isna(obj):
            return None
        return obj
    
    import pandas as pd
    import numpy as np
    
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'strategy': strategy.name,
        'strategy_params': strategy.get_params(),
        'config': {
            'place_latency_ms': config.place_latency_ms,
            'cancel_latency_ms': config.cancel_latency_ms,
            'fill_model': config.fill_model.value,
            'touch_trade_rate': config.touch_trade_rate_per_second,
        },
        'metrics': backtest_result.get('metrics', {}),
        'latency_cliff_place_ms': latency_results.get('cliff_place_latency_ms'),
        'latency_cliff_cancel_ms': latency_results.get('cliff_cancel_latency_ms'),
        'placebo_validation': placebo_results.get('overall_validation'),
        'placebo_tests_passed': placebo_results.get('tests_passed'),
        'stress_robustness_score': stress_results.get('robustness_score'),
    }
    
    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(convert_for_json(results_summary), f, indent=2)
    
    print(f"  Summary saved to: {output_dir / 'analysis_summary.json'}")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    metrics = backtest_result.get('metrics', {})
    print(f"\n  Strategy: {strategy.name}")
    print(f"  Total PnL: ${metrics.get('total_pnl', 0):.4f}")
    print(f"  t-statistic: {metrics.get('t_stat', 0):.2f}")
    print(f"  Fill rate: {metrics.get('fill_rate', 0)*100:.2f}%")
    print(f"  Latency cliff: {latency_results.get('cliff_place_latency_ms', 'N/A')}ms")
    print(f"  Placebo validation: {placebo_results.get('overall_validation', 'N/A')}")
    print(f"  Stress robustness: {stress_results.get('robustness_score', 0)*100:.0f}%")
    
    print(f"\n  Output directory: {output_dir}")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == '__main__':
    main()

