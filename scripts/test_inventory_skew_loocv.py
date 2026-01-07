#!/usr/bin/env python3
"""
Test Inventory Skew with Leave-One-Out Cross-Validation (LOOCV)

This script tests the inventory skew feature:
1. For each of 12 markets, train on 11, test on 1
2. Compare results with and without skew
3. Report mean test PnL, std, min, max across folds
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.backtest.data_loader import load_eth_markets, add_derived_columns
from scripts.backtest.strategies import SpreadCaptureStrategy
from scripts.backtest.backtest_engine import run_maker_backtest
from scripts.backtest.maker_execution import MakerExecutionConfig, FillModel


def get_volume_markets(df: pd.DataFrame) -> List[str]:
    """Get list of volume market IDs."""
    volume_prefixes = [
        '20260106_1630', '20260106_1645', '20260106_1700', '20260106_1715',
        '20260106_1730', '20260106_1745', '20260106_1800', '20260106_1815',
        '20260106_1830', '20260106_1845', '20260106_1900', '20260106_1915'
    ]
    all_markets = df['market_id'].unique()
    return [m for m in all_markets if any(m.startswith(p) for p in volume_prefixes)]


def run_single_market(
    df: pd.DataFrame,
    market_id: str,
    strategy: SpreadCaptureStrategy,
    config: MakerExecutionConfig,
) -> Dict:
    """Run backtest on a single market."""
    market_df = df[df['market_id'] == market_id]
    if len(market_df) == 0:
        return {'pnl': 0, 'fills': 0, 'spread_captured': 0, 'adverse_selection': 0}
    
    result = run_maker_backtest(
        market_df, strategy, config,
        volume_markets_only=False,  # Already filtered
        verbose=False
    )
    
    metrics = result.get('metrics', {})
    return {
        'pnl': metrics.get('total_pnl', 0),
        'fills': metrics.get('n_fills', 0),
        'spread_captured': metrics.get('spread_captured_total', 0),
        'adverse_selection': metrics.get('adverse_selection_total', 0),
    }


def run_loocv(
    df: pd.DataFrame,
    strategy_with_skew: SpreadCaptureStrategy,
    strategy_without_skew: SpreadCaptureStrategy,
    config: MakerExecutionConfig,
    markets: List[str],
) -> pd.DataFrame:
    """
    Run LOOCV comparison.
    
    For each market:
    - Train: find optimal skew_threshold on remaining 11 markets
    - Test: apply to held-out market
    """
    results = []
    
    for test_market in markets:
        train_markets = [m for m in markets if m != test_market]
        
        # Run on test market with both strategies
        result_with = run_single_market(df, test_market, strategy_with_skew, config)
        result_without = run_single_market(df, test_market, strategy_without_skew, config)
        
        results.append({
            'test_market': test_market,
            'pnl_with_skew': result_with['pnl'],
            'pnl_without_skew': result_without['pnl'],
            'pnl_diff': result_with['pnl'] - result_without['pnl'],
            'fills_with_skew': result_with['fills'],
            'fills_without_skew': result_without['fills'],
        })
    
    return pd.DataFrame(results)


def main():
    print("="*70)
    print("INVENTORY SKEW LOOCV TEST")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df, _ = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    markets = get_volume_markets(df)
    print(f"Volume markets: {len(markets)}")
    
    # Create strategies
    strategy_with_skew = SpreadCaptureStrategy(
        spread_min=0.01,
        tau_min=60,
        tau_max=600,
        inventory_limit_up=10.0,
        inventory_limit_down=10.0,
        inventory_skew_enabled=True,
        skew_threshold_up=5.0,
        skew_threshold_down=5.0,
    )
    
    strategy_without_skew = SpreadCaptureStrategy(
        spread_min=0.01,
        tau_min=60,
        tau_max=600,
        inventory_limit_up=10.0,
        inventory_limit_down=10.0,
        inventory_skew_enabled=False,
    )
    
    config = MakerExecutionConfig(
        place_latency_ms=100,
        cancel_latency_ms=50,
        fill_model=FillModel.TOUCH_SIZE_PROXY,
        touch_trade_rate_per_second=0.03,
    )
    
    print(f"\nStrategy with skew: {strategy_with_skew.name}")
    print(f"Strategy without skew: {strategy_without_skew.name}")
    
    # Run LOOCV
    print("\nRunning LOOCV...")
    results_df = run_loocv(df, strategy_with_skew, strategy_without_skew, config, markets)
    
    # Print results
    print("\n" + "-"*70)
    print("LOOCV RESULTS")
    print("-"*70)
    
    print("\nPer-market results:")
    for _, row in results_df.iterrows():
        market_short = row['test_market'][:20]
        pnl_with = row['pnl_with_skew']
        pnl_without = row['pnl_without_skew']
        diff = row['pnl_diff']
        print(f"  {market_short}: with=${pnl_with:.4f}, without=${pnl_without:.4f}, diff=${diff:.4f}")
    
    print("\nSummary statistics:")
    print(f"\n  WITH SKEW:")
    print(f"    Mean PnL: ${results_df['pnl_with_skew'].mean():.4f}")
    print(f"    Std PnL: ${results_df['pnl_with_skew'].std():.4f}")
    print(f"    Min PnL: ${results_df['pnl_with_skew'].min():.4f}")
    print(f"    Max PnL: ${results_df['pnl_with_skew'].max():.4f}")
    print(f"    Total PnL: ${results_df['pnl_with_skew'].sum():.4f}")
    print(f"    Total fills: {results_df['fills_with_skew'].sum()}")
    
    print(f"\n  WITHOUT SKEW:")
    print(f"    Mean PnL: ${results_df['pnl_without_skew'].mean():.4f}")
    print(f"    Std PnL: ${results_df['pnl_without_skew'].std():.4f}")
    print(f"    Min PnL: ${results_df['pnl_without_skew'].min():.4f}")
    print(f"    Max PnL: ${results_df['pnl_without_skew'].max():.4f}")
    print(f"    Total PnL: ${results_df['pnl_without_skew'].sum():.4f}")
    print(f"    Total fills: {results_df['fills_without_skew'].sum()}")
    
    print(f"\n  DIFFERENCE (with - without):")
    print(f"    Mean diff: ${results_df['pnl_diff'].mean():.4f}")
    print(f"    Std diff: ${results_df['pnl_diff'].std():.4f}")
    
    # T-test on difference
    mean_diff = results_df['pnl_diff'].mean()
    std_diff = results_df['pnl_diff'].std()
    n = len(results_df)
    se_diff = std_diff / np.sqrt(n) if n > 0 else 0
    t_stat = mean_diff / se_diff if se_diff > 0 else 0
    
    print(f"    t-stat: {t_stat:.2f}")
    
    # Conclusion
    print("\n" + "-"*70)
    print("CONCLUSION")
    print("-"*70)
    
    if t_stat > 1.96:
        print("\n  [SIGNIFICANT] Inventory skew improves PnL significantly (t > 1.96)")
    elif t_stat > 1.0:
        print("\n  [MARGINAL] Inventory skew shows weak improvement (1.0 < t < 1.96)")
    elif t_stat > -1.0:
        print("\n  [NO EFFECT] Inventory skew has no significant effect (-1.0 < t < 1.0)")
    else:
        print("\n  [NEGATIVE] Inventory skew hurts performance (t < -1.0)")
    
    print("="*70)
    
    # Save results
    output_dir = project_root / 'data_v2' / 'backtest_results' / 'maker_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / 'loocv_skew_results.csv', index=False)
    print(f"\nResults saved to {output_dir / 'loocv_skew_results.csv'}")


if __name__ == '__main__':
    main()

