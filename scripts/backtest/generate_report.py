"""
Generate Comprehensive Backtest Report

Creates a markdown report with all findings from the backtest analysis.
Each section explicitly states the strategy, parameters, and trade counts
to ensure internal consistency.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np


def generate_report(
    data_summary: Dict[str, Any],
    strategy_name: str,
    strategy_params: Dict[str, Any],
    strategy_results: Dict[str, Any],
    latency_analysis: Dict[str, Any],
    control_results: Optional[Dict[str, Any]] = None,
    parameter_sweep: Optional[pd.DataFrame] = None,
    fair_value_analysis: Optional[Dict[str, Any]] = None,
    placebo_results: Optional[Dict[str, Any]] = None,
    output_dir: Path = None
) -> str:
    """
    Generate comprehensive markdown report.
    
    IMPORTANT: All sections use the same strategy for consistency.
    The strategy_name and strategy_params are displayed in every relevant section.
    
    Args:
        data_summary: Data overview stats
        strategy_name: Name of strategy being analyzed
        strategy_params: Strategy parameters
        strategy_results: Results from backtest_engine
        latency_analysis: Results from latency_cliff (must use same strategy)
        control_results: Results from control strategies (DoNothing, AlwaysTrade, Random)
        parameter_sweep: Parameter sweep DataFrame
        fair_value_analysis: Fair value model results
        placebo_results: Placebo test results
        output_dir: Where to save report
    
    Returns:
        Path to saved report
    """
    report = []
    
    # Header
    report.append("# ETH Lead-Lag Backtest Report")
    report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Primary Strategy**: {strategy_name}")
    report.append("\n---\n")
    
    # Strategy Configuration (shown upfront)
    report.append("## Strategy Configuration\n")
    report.append("All analysis in this report uses the following strategy:\n")
    report.append(f"- **Strategy**: {strategy_name}")
    for k, v in strategy_params.items():
        if v is not None and k != 'strategy':
            report.append(f"- **{k}**: {v}")
    report.append("\n---\n")
    
    # Executive Summary
    report.append("## Executive Summary\n")
    
    metrics = strategy_results.get('metrics', {})
    n_trades = metrics.get('n_trades', 0)
    total_pnl = metrics.get('total_pnl', 0)
    t_stat = metrics.get('t_stat', 0)
    
    report.append(f"| Metric | Value |")
    report.append(f"|--------|-------|")
    report.append(f"| Strategy | {strategy_name} |")
    report.append(f"| Total Trades | {n_trades} |")
    report.append(f"| Total PnL | ${total_pnl:.4f} |")
    report.append(f"| t-statistic | {t_stat:.2f} |")
    report.append(f"| Markets with positive PnL | {metrics.get('hit_rate_per_market', 0)*100:.1f}% |")
    
    if placebo_results:
        report.append(f"| Placebo p-value | {placebo_results.get('p_value_pnl', 'N/A'):.3f} |")
    
    # Latency cliff summary
    if latency_analysis and 'cliff_latency' in latency_analysis:
        report.append(f"| Latency cliff | {latency_analysis['cliff_latency']}s |")
    
    report.append("\n---\n")
    
    # Control Strategy Validation
    if control_results:
        report.append("## 1. Control Strategy Validation\n")
        report.append("Before trusting any results, we verify the simulator behaves correctly:\n")
        
        report.append("| Control Strategy | Expected | Actual | Status |")
        report.append("|------------------|----------|--------|--------|")
        
        for control_name, result in control_results.items():
            expected = result.get('expected', 'N/A')
            actual_pnl = result.get('metrics', {}).get('total_pnl', 0)
            n = result.get('metrics', {}).get('n_trades', 0)
            status = result.get('status', 'UNKNOWN')
            report.append(f"| {control_name} | {expected} | ${actual_pnl:.4f} (N={n}) | {status} |")
        
        report.append("\n")
    
    # Data Overview
    report.append("## 2. Data Overview\n")
    report.append(f"- **Asset**: ETH")
    report.append(f"- **Markets**: {data_summary.get('n_markets', 'N/A')}")
    report.append(f"- **Total Observations**: {data_summary.get('n_obs', 'N/A'):,}")
    report.append(f"- **Train/Test Split**: {data_summary.get('train_markets', 'N/A')} / {data_summary.get('test_markets', 'N/A')} markets")
    report.append(f"- **Min Coverage**: {data_summary.get('min_coverage', 90)}%")
    report.append("\n")
    
    # Latency Cliff Analysis
    report.append("## 3. Latency Cliff Analysis\n")
    report.append(f"**Strategy**: {strategy_name}\n")
    report.append("**Key Question**: At what latency does edge disappear?\n")
    
    if latency_analysis and 'summary_df' in latency_analysis:
        summary_df = latency_analysis['summary_df']
        if summary_df is not None and not summary_df.empty:
            report.append("| Latency | Total PnL | Avg PnL | Hit Rate | N Trades | Conv % |")
            report.append("|---------|-----------|---------|----------|----------|--------|")
            for _, row in summary_df.iterrows():
                report.append(
                    f"| {row['latency']:.0f}s | ${row['total_pnl']:.4f} | "
                    f"${row['avg_pnl']:.4f} | {row['hit_rate']*100:.1f}% | "
                    f"{row['n_trades']:.0f} | {row.get('conversion_rate', 0):.1f}% |"
                )
        
        cliff = latency_analysis.get('cliff_latency', 'N/A')
        report.append(f"\n**Cliff Point**: {cliff}s (first latency where avg PnL drops to 0 or below)")
        
        # Clustered statistics
        if 'stats_df' in latency_analysis and not latency_analysis['stats_df'].empty:
            report.append("\n### Per-Market Clustered Statistics\n")
            stats_df = latency_analysis['stats_df']
            report.append("| Latency | Mean PnL | Std | t-stat | Hit Rate | Worst |")
            report.append("|---------|----------|-----|--------|----------|-------|")
            for _, row in stats_df.iterrows():
                report.append(
                    f"| {row['latency']:.0f}s | ${row['mean_pnl']:.4f} | "
                    f"${row['std_pnl']:.4f} | {row['t_stat']:.2f} | "
                    f"{row['hit_rate']*100:.1f}% | ${row['worst_market']:.4f} |"
                )
    
    report.append("\n")
    
    # Strategy Results
    report.append("## 4. Strategy Performance\n")
    report.append(f"**Strategy**: {strategy_name}\n")
    
    params = strategy_results.get('params', strategy_params)
    report.append("### Parameters\n")
    for k, v in params.items():
        if v is not None and k != 'strategy':
            report.append(f"- {k}: {v}")
    
    report.append("\n### Metrics\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Total PnL | ${metrics.get('total_pnl', 0):.4f} |")
    report.append(f"| N Trades | {metrics.get('n_trades', 0)} |")
    report.append(f"| N Markets | {metrics.get('n_markets', 0)} |")
    report.append(f"| Mean PnL per market | ${metrics.get('mean_pnl_per_market', 0):.4f} |")
    report.append(f"| Std PnL per market | ${metrics.get('std_pnl_per_market', 0):.4f} |")
    report.append(f"| t-statistic | {metrics.get('t_stat', 0):.2f} |")
    report.append(f"| Hit rate (markets) | {metrics.get('hit_rate_per_market', 0)*100:.1f}% |")
    report.append(f"| Hit rate (trades) | {metrics.get('hit_rate_per_trade', 0)*100:.1f}% |")
    report.append(f"| Worst market | ${metrics.get('worst_market_pnl', 0):.4f} |")
    report.append(f"| Best market | ${metrics.get('best_market_pnl', 0):.4f} |")
    report.append(f"| Conversion rate (entry) | {metrics.get('conversion_entry_rate', 0):.1f}% |")
    report.append(f"| Conversion rate (exit) | {metrics.get('conversion_exit_rate', 0):.1f}% |")
    report.append("\n")
    
    # Parameter Sweep
    if parameter_sweep is not None and not parameter_sweep.empty:
        report.append("## 5. Parameter Sweep Summary\n")
        report.append("**Note**: All strategies tested with same execution model.\n")
        
        top5 = parameter_sweep.nlargest(5, 'train_t_stat')
        report.append("\n### Top 5 Strategies by Train t-stat\n")
        report.append("| Strategy | Params | Latency | Train PnL | Train t | Test PnL | N Train |")
        report.append("|----------|--------|---------|-----------|---------|----------|---------|")
        for _, row in top5.iterrows():
            params_str = f"tau={row.get('tau_max', 'N/A')}" if 'tau_max' in row else ""
            report.append(
                f"| {row['strategy']} | {params_str} | {row['latency']:.0f}s | "
                f"${row['train_total_pnl']:.3f} | {row['train_t_stat']:.2f} | "
                f"${row['test_total_pnl']:.3f} | {row['train_n_trades']:.0f} |"
            )
        report.append("\n")
    
    # Fair Value Analysis
    if fair_value_analysis:
        report.append("## 6. Fair Value Analysis\n")
        report.append(f"**Strategy**: {strategy_name}\n")
        report.append("**Purpose**: Determine if edge comes from latency or momentum\n")
        
        train_analysis = fair_value_analysis.get('train_analysis', {})
        test_analysis = fair_value_analysis.get('test_analysis', {})
        
        report.append("\n| Metric | Train | Test |")
        report.append("|--------|-------|------|")
        report.append(
            f"| PnL-Mispricing Correlation | "
            f"{train_analysis.get('pnl_mispricing_correlation', 0):.4f} | "
            f"{test_analysis.get('pnl_mispricing_correlation', 0):.4f} |"
        )
        report.append(
            f"| Avg Mispricing | "
            f"${train_analysis.get('avg_mispricing', 0):.4f} | "
            f"${test_analysis.get('avg_mispricing', 0):.4f} |"
        )
        report.append(
            f"| N Trades | "
            f"{train_analysis.get('n_trades', 0)} | "
            f"{test_analysis.get('n_trades', 0)} |"
        )
        
        interp = train_analysis.get('interpretation', 'N/A')
        report.append(f"\n**Interpretation**: {interp}")
        report.append("\n")
    
    # Placebo Tests
    if placebo_results:
        report.append("## 7. Placebo Test Results\n")
        report.append(f"**Strategy**: {strategy_name}\n")
        
        report.append("\n### Randomization Test\n")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| Real strategy PnL | ${placebo_results.get('real_result', {}).get('total_pnl', 0):.4f} |")
        report.append(f"| Real strategy N trades | {placebo_results.get('real_result', {}).get('n_trades', 0)} |")
        report.append(f"| Placebo mean PnL | ${placebo_results.get('placebo_mean', {}).get('total_pnl', 0):.4f} |")
        report.append(f"| Placebo std PnL | ${placebo_results.get('placebo_std', {}).get('total_pnl', 0):.4f} |")
        report.append(f"| P-value | {placebo_results.get('p_value_pnl', 'N/A'):.3f} |")
        
        report.append(f"\n**Result**: {placebo_results.get('interpretation', 'N/A')}")
        report.append("\n")
    
    # Conclusions
    report.append("## 8. Conclusions & Recommendations\n")
    
    # Determine overall conclusion
    t_stat = metrics.get('t_stat', 0)
    p_value = placebo_results.get('p_value_pnl', 1) if placebo_results else 1
    
    if t_stat > 1.96 and p_value < 0.05:
        conclusion = "STRONG EVIDENCE of tradeable edge"
    elif t_stat > 1.65 or p_value < 0.1:
        conclusion = "MODERATE EVIDENCE of potential edge (needs more data)"
    else:
        conclusion = "WEAK EVIDENCE - edge may not be robust"
    
    report.append(f"### Overall Assessment: {conclusion}\n")
    
    report.append("### Key Findings\n")
    report.append(f"1. {strategy_name} shows {'positive' if total_pnl > 0 else 'negative'} returns (${total_pnl:.4f})")
    
    if latency_analysis and 'cliff_latency' in latency_analysis:
        cliff = latency_analysis['cliff_latency']
        report.append(f"2. Edge persists up to {cliff}s execution latency")
    
    if fair_value_analysis:
        corr = fair_value_analysis.get('train_analysis', {}).get('pnl_mispricing_correlation', 0)
        if abs(corr) < 0.1:
            report.append("3. Fair value analysis: Edge is from latency, not momentum")
        else:
            report.append(f"3. Fair value analysis: Some momentum component (corr={corr:.2f})")
    
    if placebo_results and placebo_results.get('p_value_pnl', 1) < 0.05:
        report.append("4. Placebo tests: Signal is statistically significant")
    
    report.append("\n### Recommendations\n")
    report.append("1. Collect more data (100+ markets) for statistical confidence")
    report.append("2. Test on other assets (BTC, SOL) to verify generalization")
    report.append("3. Validate timestamp semantics (source vs received times)")
    report.append("4. Monitor execution quality and actual fill rates")
    
    report.append("\n---\n")
    report.append("### Report Consistency Check\n")
    report.append(f"- All sections analyze: **{strategy_name}**")
    lat_trades = latency_analysis['summary_df'].iloc[0]['n_trades'] if latency_analysis and 'summary_df' in latency_analysis else 'N/A'
    strat_trades = metrics.get('n_trades', 'N/A')
    report.append(f"- Latency cliff N trades at 0s: {lat_trades} (all {data_summary.get('n_markets', 'N/A')} markets)")
    report.append(f"- Strategy results N trades: {strat_trades} (train set: {data_summary.get('train_markets', 'N/A')} markets)")
    report.append("\n**Note**: Trade counts differ because latency cliff runs on all markets, while strategy metrics are on train set only.")
    report.append("\n*Report generated by scripts/backtest/generate_report.py*")
    
    # Write report
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / 'BACKTEST_REPORT.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        return str(report_path)
    
    return '\n'.join(report)


# ==============================================================================
# MAKER STRATEGY REPORT
# ==============================================================================

def generate_maker_report(
    data_summary: Dict[str, Any],
    strategy_name: str,
    strategy_params: Dict[str, Any],
    backtest_result: Dict[str, Any],
    diagnostics: Optional[Dict[str, Any]] = None,
    latency_results: Optional[Dict[str, Any]] = None,
    placebo_results: Optional[Dict[str, Any]] = None,
    stress_results: Optional[Dict[str, Any]] = None,
    parameter_sweep: Optional[pd.DataFrame] = None,
    output_dir: Path = None,
) -> str:
    """
    Generate comprehensive markdown report for maker/spread capture strategy.
    
    Args:
        data_summary: Data overview stats
        strategy_name: Name of strategy being analyzed
        strategy_params: Strategy parameters
        backtest_result: Results from run_maker_backtest
        diagnostics: Results from maker_diagnostics
        latency_results: Results from maker latency sweep
        placebo_results: Results from maker placebo tests
        stress_results: Results from stress tests
        parameter_sweep: Parameter sweep DataFrame
        output_dir: Where to save report
        
    Returns:
        Report content or path to saved report
    """
    report = []
    
    # Header
    report.append("# Spread Capture / Maker Strategy Report")
    report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Strategy**: {strategy_name}")
    report.append("\n---\n")
    
    # Strategy Configuration
    report.append("## Strategy Configuration\n")
    report.append("```")
    for k, v in strategy_params.items():
        if v is not None and k != 'strategy':
            report.append(f"{k}: {v}")
    report.append("```\n")
    
    # Executive Summary
    report.append("## Executive Summary\n")
    
    metrics = backtest_result.get('metrics', {})
    
    report.append("### Performance Metrics\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Total PnL | ${metrics.get('total_pnl', 0):.4f} |")
    report.append(f"| Mean PnL/Market | ${metrics.get('mean_pnl_per_market', 0):.4f} |")
    report.append(f"| t-statistic | {metrics.get('t_stat', 0):.2f} |")
    report.append(f"| Markets with positive PnL | {metrics.get('hit_rate_per_market', 0)*100:.1f}% |")
    report.append(f"| Number of Markets | {metrics.get('n_markets', 0)} |")
    report.append(f"| Total Fills | {metrics.get('n_fills', 0)} |")
    report.append(f"| Fill Rate | {metrics.get('fill_rate', 0)*100:.2f}% |")
    
    report.append("\n### PnL Decomposition\n")
    report.append("| Component | Value |")
    report.append("|-----------|-------|")
    report.append(f"| Spread Captured | ${metrics.get('spread_captured_total', 0):.4f} |")
    report.append(f"| Adverse Selection | -${metrics.get('adverse_selection_total', 0):.4f} |")
    report.append(f"| Inventory Carry | ${metrics.get('inventory_carry_total', 0):.4f} |")
    report.append(f"| Realized PnL | ${metrics.get('realized_pnl_total', 0):.4f} |")
    
    report.append("\n### Fill Statistics\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Orders Placed | {metrics.get('orders_placed_total', 0)} |")
    report.append(f"| Orders Filled | {metrics.get('orders_filled_total', 0)} |")
    report.append(f"| Orders Cancelled | {metrics.get('orders_cancelled_total', 0)} |")
    report.append(f"| Orders Expired | {metrics.get('orders_expired_total', 0)} |")
    report.append(f"| Fill Rate | {metrics.get('fill_rate', 0)*100:.2f}% |")
    report.append(f"| Avg Time to Fill | {metrics.get('avg_time_to_fill', 0):.1f}s |")
    
    report.append("\n---\n")
    
    # Diagnostics
    if diagnostics:
        report.append("## Detailed Diagnostics\n")
        
        # Fill rate by tau
        fill_by_tau = diagnostics.get('fill_rate_by_tau', [])
        if fill_by_tau:
            report.append("### Fill Rate by Time-to-Expiry (tau)\n")
            report.append("| Tau Window | Fills | % of Total |")
            report.append("|------------|-------|------------|")
            total_fills = sum(r['n_fills'] for r in fill_by_tau)
            for row in fill_by_tau:
                pct = row['n_fills'] / total_fills * 100 if total_fills > 0 else 0
                report.append(f"| {row['tau_label']} | {row['n_fills']} | {pct:.1f}% |")
        
        # Fill rate by spread
        fill_by_spread = diagnostics.get('fill_rate_by_spread', [])
        if fill_by_spread:
            report.append("\n### Fill Rate by Spread Width\n")
            report.append("| Spread | Fills | % of Total |")
            report.append("|--------|-------|------------|")
            for row in fill_by_spread:
                report.append(f"| {row['spread_label']} | {row['n_fills']} | {row['pct_of_fills']*100:.1f}% |")
        
        # Adverse selection
        as_analysis = diagnostics.get('adverse_selection', {})
        if as_analysis:
            report.append("\n### Adverse Selection Analysis\n")
            report.append(f"- Avg adverse selection (1s): {as_analysis.get('avg_adverse_selection_1s', 0)*100:.3f}c")
            report.append(f"- Avg adverse selection (5s): {as_analysis.get('avg_adverse_selection_5s', 0)*100:.3f}c")
            report.append(f"- % fills with gain (1s): {as_analysis.get('pct_negative_as_1s', 0)*100:.1f}%")
            report.append(f"- % fills with gain (5s): {as_analysis.get('pct_negative_as_5s', 0)*100:.1f}%")
        
        report.append("\n---\n")
    
    # Latency Analysis
    if latency_results:
        report.append("## Latency Sensitivity\n")
        
        report.append(f"**Placement Latency Cliff**: {latency_results.get('cliff_place_latency_ms', 'N/A')}ms\n")
        report.append(f"**Cancel Latency Cliff**: {latency_results.get('cliff_cancel_latency_ms', 'N/A')}ms\n")
        
        place_summary = latency_results.get('place_latency_summary')
        if place_summary is not None and len(place_summary) > 0:
            report.append("\n### PnL by Placement Latency\n")
            report.append("| Place Latency | Total PnL | t-stat | Fill Rate |")
            report.append("|---------------|-----------|--------|-----------|")
            if isinstance(place_summary, pd.DataFrame):
                for _, row in place_summary.iterrows():
                    report.append(f"| {row['place_latency_ms']}ms | ${row['total_pnl']:.4f} | {row['t_stat']:.2f} | {row['fill_rate']*100:.2f}% |")
            else:
                for row in place_summary:
                    report.append(f"| {row['place_latency_ms']}ms | ${row['total_pnl']:.4f} | {row['t_stat']:.2f} | {row['fill_rate']*100:.2f}% |")
        
        report.append("\n---\n")
    
    # Placebo Tests
    if placebo_results:
        report.append("## Placebo Test Results\n")
        
        # Random test
        rand = placebo_results.get('random_test', {})
        if rand:
            report.append("### Randomized Timing Test\n")
            report.append(f"- Real PnL: ${rand.get('real_pnl', 0):.4f}")
            report.append(f"- Placebo Mean: ${rand.get('placebo_mean', 0):.4f}")
            report.append(f"- P-value: {rand.get('p_value', 1):.3f}")
            report.append(f"- **Result**: {'PASS' if rand.get('passed', False) else 'FAIL'}")
            report.append(f"- {rand.get('interpretation', '')}\n")
        
        # Stale data test
        stale = placebo_results.get('stale_test', {})
        if stale:
            report.append("### Stale Data Test\n")
            report.append(f"- **Result**: {'PASS' if stale.get('passed', False) else 'FAIL'}\n")
        
        # Flipped sides test
        flip = placebo_results.get('flipped_test', {})
        if flip:
            report.append("### Flipped Sides Test\n")
            report.append(f"- Real PnL: ${flip.get('real_pnl', 0):.4f}")
            report.append(f"- Flipped PnL: ${flip.get('flipped_pnl', 0):.4f}")
            report.append(f"- **Result**: {'PASS' if flip.get('passed', False) else 'FAIL'}")
            report.append(f"- {flip.get('interpretation', '')}\n")
        
        # Overall
        report.append(f"\n**Overall Validation**: {placebo_results.get('overall_validation', 'N/A')}")
        report.append(f"\n**Tests Passed**: {placebo_results.get('tests_passed', 0)}/{placebo_results.get('total_tests', 0)}")
        
        report.append("\n---\n")
    
    # Stress Tests
    if stress_results:
        report.append("## Stress Test Results\n")
        report.append(f"**Robustness Score**: {stress_results.get('robustness_score', 0)*100:.0f}%\n")
        
        # Slippage
        slip = stress_results.get('slippage_test', {})
        if slip:
            report.append("### Slippage Tolerance\n")
            report.append(f"- Tolerance: {slip.get('tolerance_bps', 'N/A')}bps")
            report.append(f"- {slip.get('interpretation', '')}\n")
        
        # Spread widening
        spr = stress_results.get('spread_test', {})
        if spr:
            report.append("### Spread Widening\n")
            report.append(f"- Robust at 1.5x spreads: {'Yes' if spr.get('robust_at_15x', False) else 'No'}")
            report.append(f"- {spr.get('interpretation', '')}\n")
        
        # Volatility
        vol = stress_results.get('volatility_test', {})
        if vol:
            report.append("### Volatility Dependence\n")
            report.append(f"- Robust without top 10% volatile: {'Yes' if vol.get('robust_without_top10pct', False) else 'No'}")
            report.append(f"- {vol.get('interpretation', '')}\n")
        
        # Fill rate sensitivity
        fr = stress_results.get('fill_rate_test', {})
        if fr:
            report.append("### Fill Rate Sensitivity\n")
            report.append(f"- Robust across rates: {'Yes' if fr.get('robust_across_rates', False) else 'No'}")
            report.append(f"- {fr.get('interpretation', '')}\n")
        
        report.append("\n---\n")
    
    # Parameter Sweep
    if parameter_sweep is not None and len(parameter_sweep) > 0:
        report.append("## Parameter Sweep Results\n")
        
        # Top 10 by t-stat
        if 't_stat' in parameter_sweep.columns:
            top10 = parameter_sweep.nlargest(10, 't_stat')
            report.append("### Top 10 Parameter Combinations by t-stat\n")
            report.append("| Spread Min | Tau Min | Place Lat | PnL | t-stat | Fill Rate |")
            report.append("|------------|---------|-----------|-----|--------|-----------|")
            for _, row in top10.iterrows():
                report.append(f"| {row.get('spread_min', 'N/A')} | {row.get('tau_min', 'N/A')} | {row.get('place_latency_ms', 'N/A')}ms | ${row.get('total_pnl', 0):.4f} | {row.get('t_stat', 0):.2f} | {row.get('fill_rate', 0)*100:.2f}% |")
        
        report.append("\n---\n")
    
    # Conclusions
    report.append("## Conclusions and Recommendations\n")
    
    # Build recommendations based on results
    recs = []
    
    # Check t-stat
    t_stat = metrics.get('t_stat', 0)
    if t_stat > 1.96:
        recs.append("✅ Statistically significant positive PnL (t > 1.96)")
    elif t_stat > 1.0:
        recs.append("⚠️ Marginally positive PnL (1.0 < t < 1.96) - more data needed")
    else:
        recs.append("❌ Not statistically significant (t < 1.0)")
    
    # Check fill rate
    fill_rate = metrics.get('fill_rate', 0)
    if fill_rate > 0.05:
        recs.append(f"✅ Reasonable fill rate ({fill_rate*100:.1f}%)")
    elif fill_rate > 0.01:
        recs.append(f"⚠️ Low fill rate ({fill_rate*100:.1f}%) - may need more aggressive quoting")
    else:
        recs.append(f"❌ Very low fill rate ({fill_rate*100:.1f}%) - strategy may not be viable")
    
    # Check placebo
    if placebo_results:
        if placebo_results.get('overall_validation') == 'VALID':
            recs.append("✅ Passed placebo tests")
        else:
            recs.append("❌ Failed placebo tests - edge may be spurious")
    
    # Check stress tests
    if stress_results:
        if stress_results.get('robustness_score', 0) > 0.5:
            recs.append("✅ Passed majority of stress tests")
        else:
            recs.append("⚠️ Failed some stress tests - edge may not be robust")
    
    for rec in recs:
        report.append(f"- {rec}")
    
    report.append("\n")
    
    # Overall verdict
    verdict = "PROCEED WITH CAUTION"
    if t_stat > 1.96 and fill_rate > 0.03:
        if placebo_results and placebo_results.get('overall_validation') == 'VALID':
            if stress_results and stress_results.get('robustness_score', 0) > 0.5:
                verdict = "PROMISING - Consider paper trading"
            else:
                verdict = "NEEDS MORE TESTING"
        else:
            verdict = "QUESTIONABLE - Placebo tests not passed"
    elif t_stat < 0:
        verdict = "NOT VIABLE - Negative expected PnL"
    
    report.append(f"### Overall Verdict: **{verdict}**\n")
    
    report.append("\n---\n")
    report.append(f"\n*Report generated with spread capture testing framework v1.0*")
    
    # Save if output_dir specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / 'MAKER_BACKTEST_REPORT.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        return str(report_path)
    
    return '\n'.join(report)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    import sys
    
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from scripts.backtest.data_loader import load_eth_markets, add_derived_columns, get_train_test_split
    from scripts.backtest.strategies import StrikeCrossStrategy
    from scripts.backtest.backtest_engine import run_backtest, ExecutionConfig
    from scripts.backtest.latency_cliff import run_strategy_latency_analysis
    from scripts.backtest.fair_value import run_fair_value_analysis
    from scripts.backtest.placebo_tests import run_placebo_test_random
    
    print("Loading data...")
    df, market_info = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    train_df, test_df, train_ids, test_ids = get_train_test_split(df)
    
    # Data summary
    data_summary = {
        'n_markets': df['market_id'].nunique(),
        'n_obs': len(df),
        'train_markets': len(train_ids),
        'test_markets': len(test_ids),
        'min_coverage': 90,
    }
    
    # Define the strategy (used consistently throughout)
    strategy = StrikeCrossStrategy(tau_max=600, hold_to_expiry=True)
    strategy_name = strategy.name
    strategy_params = strategy.get_params()
    
    print(f"\nAnalyzing strategy: {strategy_name}")
    
    print("\n1. Running latency analysis...")
    latency_analysis = run_strategy_latency_analysis(df, strategy)
    
    print("\n2. Running backtest...")
    train_result = run_backtest(train_df, strategy, ExecutionConfig())
    test_result = run_backtest(test_df, strategy, ExecutionConfig())
    
    print("\n3. Running fair value analysis...")
    fair_value = run_fair_value_analysis(
        train_df, test_df,
        train_result['trades'], test_result['trades']
    )
    
    print("\n4. Running placebo tests...")
    placebo = run_placebo_test_random(df, strategy, n_iterations=10)
    
    # Load sweep results if available
    sweep_path = project_root / 'data_v2' / 'backtest_results' / 'parameter_sweep.csv'
    sweep_df = pd.read_csv(sweep_path) if sweep_path.exists() else None
    
    print("\n5. Generating report...")
    output_dir = project_root / 'data_v2' / 'backtest_results'
    
    report_path = generate_report(
        data_summary=data_summary,
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        strategy_results=train_result,
        latency_analysis=latency_analysis,
        control_results=None,  # Will add after implementing control strategies
        parameter_sweep=sweep_df,
        fair_value_analysis=fair_value,
        placebo_results=placebo,
        output_dir=output_dir
    )
    
    print(f"\nReport saved to: {report_path}")
