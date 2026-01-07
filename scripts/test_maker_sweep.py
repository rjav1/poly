#!/usr/bin/env python3
"""
Test maker parameter sweep functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backtest.data_loader import load_eth_markets, add_derived_columns
from scripts.backtest.parameter_sweep import (
    run_maker_parameter_sweep, 
    analyze_maker_sweep_results, 
    get_top_maker_strategies
)
from scripts.backtest.maker_execution import MakerExecutionConfig, FillModel

print("Loading data...")
df, market_info = load_eth_markets(min_coverage=90.0)
df = add_derived_columns(df)
n_markets = df['market_id'].nunique()
print(f"Loaded {len(df)} rows, {n_markets} markets")

# Small parameter grid for quick test
param_grid = [
    {'strategy': 'SpreadCapture', 'spread_min': s, 'tau_min': tm, 'tau_max': 600,
     'inventory_limit_up': 10.0, 'inventory_limit_down': 10.0, 'tau_flatten': 60,
     'two_sided': True, 'quote_size': 1.0, 'adverse_selection_filter': True}
    for s in [0.01, 0.02, 0.03]
    for tm in [60, 120]
]

# Small config set
configs = [
    MakerExecutionConfig(place_latency_ms=0, cancel_latency_ms=0, 
                        fill_model=FillModel.TOUCH_SIZE_PROXY, touch_trade_rate_per_second=0.10),
    MakerExecutionConfig(place_latency_ms=100, cancel_latency_ms=50, 
                        fill_model=FillModel.TOUCH_SIZE_PROXY, touch_trade_rate_per_second=0.10),
]

print(f"\nRunning maker parameter sweep...")
print(f"  Parameters: {len(param_grid)}")
print(f"  Configs: {len(configs)}")
print(f"  Total combinations: {len(param_grid) * len(configs)}")

results_df = run_maker_parameter_sweep(
    df, 
    param_grid=param_grid, 
    configs=configs, 
    volume_markets_only=True,
    verbose=True
)

print(f"\nResults: {len(results_df)} rows")

# Analyze
analysis = analyze_maker_sweep_results(results_df)

print("\n" + "=" * 60)
print("MAKER SWEEP ANALYSIS")
print("=" * 60)
print(f"Combinations tested: {analysis.get('n_combinations_tested', 0)}")
print(f"Profitable: {analysis.get('n_profitable', 0)}")
print(f"Significant (t>1.96): {analysis.get('n_significant', 0)}")
print(f"Avg fill rate: {analysis.get('avg_fill_rate', 0)*100:.2f}%")
print(f"Avg t-stat: {analysis.get('avg_t_stat', 0):.2f}")

if 'best_by_tstat' in analysis:
    best = analysis['best_by_tstat']
    print(f"\n--- Best by t-stat ---")
    print(f"  spread_min: {best.get('spread_min', 'N/A')}")
    print(f"  tau_min: {best.get('tau_min', 'N/A')}")
    print(f"  place_latency_ms: {best.get('place_latency_ms', 0)}")
    print(f"  Total PnL: ${best.get('total_pnl', 0):.4f}")
    print(f"  t-stat: {best.get('t_stat', 0):.2f}")
    print(f"  n_fills: {best.get('n_fills', 0)}")
    print(f"  fill_rate: {best.get('fill_rate', 0)*100:.2f}%")

# Top 5
print("\n--- Top 5 by t-stat ---")
top5 = get_top_maker_strategies(results_df, n=5, min_fills=1, sort_by='t_stat')
display_cols = ['spread_min', 'tau_min', 'place_latency_ms', 'total_pnl', 't_stat', 'n_fills', 'fill_rate']
print(top5[display_cols].to_string())

print("\n[OK] Maker sweep test completed")

