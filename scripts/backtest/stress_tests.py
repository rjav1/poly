"""
Stress Tests for Maker Strategies

These tests check whether the strategy edge is robust to adverse conditions:

1. Extra Slippage on Exits: Add worst-case taker slippage when flattening
2. Widened Spreads: Artificially widen spreads to test if edge is just spread capture
3. Remove Volatile Seconds: Remove top X% volatile seconds to check if edge depends on jumps
4. Higher Fill Rate: What if our fill model is too pessimistic?
5. Lower Fill Rate: What if our fill model is too optimistic?

If strategy dies under any of these tests, it may not be robust for live trading.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from .backtest_engine import run_maker_backtest
    from .strategies import SpreadCaptureStrategy
    from .maker_execution import MakerExecutionConfig, FillModel
except ImportError:
    from backtest_engine import run_maker_backtest
    from strategies import SpreadCaptureStrategy
    from maker_execution import MakerExecutionConfig, FillModel


def stress_test_extra_slippage(
    df: pd.DataFrame,
    strategy: SpreadCaptureStrategy,
    config: MakerExecutionConfig = None,
    slippage_bps: List[int] = None,
    volume_markets_only: bool = True,
) -> pd.DataFrame:
    """
    Stress test: Add extra slippage on exit trades.
    
    In real trading, when we need to flatten inventory (taker exit),
    we may face worse prices than expected. This tests robustness to
    that scenario.
    
    Args:
        df: DataFrame with market data
        strategy: SpreadCaptureStrategy to test
        config: Maker execution config
        slippage_bps: List of additional slippage values (in bps)
        volume_markets_only: Only use volume markets
        
    Returns:
        DataFrame with results at each slippage level
    """
    if config is None:
        config = MakerExecutionConfig()
    
    if slippage_bps is None:
        slippage_bps = [0, 10, 25, 50, 100, 200]  # 0 = no extra slippage
    
    results = []
    
    for slip in slippage_bps:
        # Apply slippage by widening the exit costs
        # This affects the flatten trades which are taker orders
        df_stress = df.copy()
        
        # Reduce bid prices (worse for selling) and increase ask prices (worse for buying)
        slip_factor = slip / 10000.0  # Convert bps to decimal
        
        # Widen prices by slippage amount
        df_stress['pm_up_best_bid'] = df_stress['pm_up_best_bid'] * (1 - slip_factor)
        df_stress['pm_up_best_ask'] = df_stress['pm_up_best_ask'] * (1 + slip_factor)
        df_stress['pm_down_best_bid'] = df_stress['pm_down_best_bid'] * (1 - slip_factor)
        df_stress['pm_down_best_ask'] = df_stress['pm_down_best_ask'] * (1 + slip_factor)
        
        result = run_maker_backtest(
            df_stress, strategy, config,
            verbose=False,
            volume_markets_only=volume_markets_only
        )
        
        metrics = result.get('metrics', {})
        results.append({
            'slippage_bps': slip,
            'total_pnl': metrics.get('total_pnl', 0),
            't_stat': metrics.get('t_stat', 0),
            'hit_rate': metrics.get('hit_rate_per_market', 0),
            'n_fills': metrics.get('n_fills', 0),
            'fill_rate': metrics.get('fill_rate', 0),
            'spread_captured': metrics.get('spread_captured_total', 0),
            'adverse_selection': metrics.get('adverse_selection_total', 0),
        })
    
    return pd.DataFrame(results)


def stress_test_widened_spreads(
    df: pd.DataFrame,
    strategy: SpreadCaptureStrategy,
    config: MakerExecutionConfig = None,
    widen_factors: List[float] = None,
    volume_markets_only: bool = True,
) -> pd.DataFrame:
    """
    Stress test: Artificially widen spreads.
    
    If edge disappears when spreads are wider, the strategy is just
    "being paid for risk" rather than having true edge.
    
    Args:
        df: DataFrame with market data
        strategy: SpreadCaptureStrategy to test
        config: Maker execution config
        widen_factors: List of spread widening factors (1.0 = no change, 2.0 = double)
        volume_markets_only: Only use volume markets
        
    Returns:
        DataFrame with results at each widening level
    """
    if config is None:
        config = MakerExecutionConfig()
    
    if widen_factors is None:
        widen_factors = [1.0, 1.25, 1.5, 2.0, 3.0]  # 1.0 = baseline
    
    results = []
    
    for factor in widen_factors:
        df_stress = df.copy()
        
        # Calculate midpoints
        up_mid = (df_stress['pm_up_best_bid'] + df_stress['pm_up_best_ask']) / 2
        down_mid = (df_stress['pm_down_best_bid'] + df_stress['pm_down_best_ask']) / 2
        
        # Calculate current half-spreads
        up_half_spread = (df_stress['pm_up_best_ask'] - df_stress['pm_up_best_bid']) / 2
        down_half_spread = (df_stress['pm_down_best_ask'] - df_stress['pm_down_best_bid']) / 2
        
        # Widen spreads symmetrically around mid
        df_stress['pm_up_best_bid'] = up_mid - up_half_spread * factor
        df_stress['pm_up_best_ask'] = up_mid + up_half_spread * factor
        df_stress['pm_down_best_bid'] = down_mid - down_half_spread * factor
        df_stress['pm_down_best_ask'] = down_mid + down_half_spread * factor
        
        # Ensure prices stay in valid range [0, 1]
        df_stress['pm_up_best_bid'] = df_stress['pm_up_best_bid'].clip(0.01, 0.99)
        df_stress['pm_up_best_ask'] = df_stress['pm_up_best_ask'].clip(0.01, 0.99)
        df_stress['pm_down_best_bid'] = df_stress['pm_down_best_bid'].clip(0.01, 0.99)
        df_stress['pm_down_best_ask'] = df_stress['pm_down_best_ask'].clip(0.01, 0.99)
        
        result = run_maker_backtest(
            df_stress, strategy, config,
            verbose=False,
            volume_markets_only=volume_markets_only
        )
        
        metrics = result.get('metrics', {})
        
        # Calculate actual spreads
        actual_up_spread = (df_stress['pm_up_best_ask'] - df_stress['pm_up_best_bid']).mean()
        
        results.append({
            'widen_factor': factor,
            'avg_spread': actual_up_spread,
            'total_pnl': metrics.get('total_pnl', 0),
            't_stat': metrics.get('t_stat', 0),
            'hit_rate': metrics.get('hit_rate_per_market', 0),
            'n_fills': metrics.get('n_fills', 0),
            'fill_rate': metrics.get('fill_rate', 0),
            'spread_captured': metrics.get('spread_captured_total', 0),
        })
    
    return pd.DataFrame(results)


def stress_test_remove_volatile_seconds(
    df: pd.DataFrame,
    strategy: SpreadCaptureStrategy,
    config: MakerExecutionConfig = None,
    percentiles_to_remove: List[float] = None,
    volume_markets_only: bool = True,
) -> pd.DataFrame:
    """
    Stress test: Remove top X% most volatile seconds.
    
    If edge disappears when volatile periods are removed, the strategy
    might be gambling on jumps rather than having consistent edge.
    
    Args:
        df: DataFrame with market data
        strategy: SpreadCaptureStrategy to test
        config: Maker execution config
        percentiles_to_remove: List of percentiles (0 = none, 0.1 = top 10%)
        volume_markets_only: Only use volume markets
        
    Returns:
        DataFrame with results at each removal level
    """
    if config is None:
        config = MakerExecutionConfig()
    
    if percentiles_to_remove is None:
        percentiles_to_remove = [0, 0.05, 0.10, 0.20, 0.30]
    
    results = []
    
    # Calculate volatility measure for each second
    # Use CL return as proxy for volatility
    df_with_vol = df.copy()
    df_with_vol['cl_return'] = df_with_vol.groupby('market_id')['cl_mid'].pct_change().abs()
    
    for pct in percentiles_to_remove:
        df_stress = df_with_vol.copy()
        
        if pct > 0:
            # Calculate threshold for top X%
            threshold = df_stress['cl_return'].quantile(1 - pct)
            
            # Remove high-volatility seconds
            df_stress = df_stress[df_stress['cl_return'] <= threshold]
        
        n_rows_removed = len(df_with_vol) - len(df_stress)
        
        result = run_maker_backtest(
            df_stress, strategy, config,
            verbose=False,
            volume_markets_only=volume_markets_only
        )
        
        metrics = result.get('metrics', {})
        results.append({
            'pct_removed': pct,
            'rows_removed': n_rows_removed,
            'pct_rows_removed': n_rows_removed / len(df_with_vol) * 100,
            'total_pnl': metrics.get('total_pnl', 0),
            't_stat': metrics.get('t_stat', 0),
            'hit_rate': metrics.get('hit_rate_per_market', 0),
            'n_fills': metrics.get('n_fills', 0),
            'fill_rate': metrics.get('fill_rate', 0),
        })
    
    return pd.DataFrame(results)


def stress_test_fill_rate_sensitivity(
    df: pd.DataFrame,
    strategy: SpreadCaptureStrategy,
    touch_trade_rates: List[float] = None,
    volume_markets_only: bool = True,
) -> pd.DataFrame:
    """
    Stress test: Vary the assumed fill rate.
    
    Our fill model uses a calibrated touch_trade_rate parameter.
    This tests how sensitive results are to that assumption.
    
    Args:
        df: DataFrame with market data
        strategy: SpreadCaptureStrategy to test
        touch_trade_rates: List of rates to test
        volume_markets_only: Only use volume markets
        
    Returns:
        DataFrame with results at each rate
    """
    if touch_trade_rates is None:
        touch_trade_rates = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    
    results = []
    
    for rate in touch_trade_rates:
        config = MakerExecutionConfig(
            place_latency_ms=100,
            cancel_latency_ms=50,
            fill_model=FillModel.TOUCH_SIZE_PROXY,
            touch_trade_rate_per_second=rate,
        )
        
        result = run_maker_backtest(
            df, strategy, config,
            verbose=False,
            volume_markets_only=volume_markets_only
        )
        
        metrics = result.get('metrics', {})
        results.append({
            'touch_trade_rate': rate,
            'total_pnl': metrics.get('total_pnl', 0),
            't_stat': metrics.get('t_stat', 0),
            'hit_rate': metrics.get('hit_rate_per_market', 0),
            'n_fills': metrics.get('n_fills', 0),
            'fill_rate': metrics.get('fill_rate', 0),
            'spread_captured': metrics.get('spread_captured_total', 0),
            'adverse_selection': metrics.get('adverse_selection_total', 0),
        })
    
    return pd.DataFrame(results)


def run_all_stress_tests(
    df: pd.DataFrame,
    strategy: SpreadCaptureStrategy,
    config: MakerExecutionConfig = None,
    volume_markets_only: bool = True,
) -> Dict[str, Any]:
    """
    Run all stress tests and return comprehensive results.
    
    Args:
        df: DataFrame with market data
        strategy: SpreadCaptureStrategy to test
        config: Maker execution config
        volume_markets_only: Only use volume markets
        
    Returns:
        Dictionary with all stress test results
    """
    if config is None:
        config = MakerExecutionConfig()
    
    print("Running stress tests...")
    
    # 1. Extra slippage
    print("  1. Extra slippage test...")
    slippage_results = stress_test_extra_slippage(
        df, strategy, config, volume_markets_only=volume_markets_only
    )
    
    # 2. Widened spreads
    print("  2. Widened spreads test...")
    spread_results = stress_test_widened_spreads(
        df, strategy, config, volume_markets_only=volume_markets_only
    )
    
    # 3. Remove volatile seconds
    print("  3. Remove volatile seconds test...")
    volatility_results = stress_test_remove_volatile_seconds(
        df, strategy, config, volume_markets_only=volume_markets_only
    )
    
    # 4. Fill rate sensitivity
    print("  4. Fill rate sensitivity test...")
    fill_rate_results = stress_test_fill_rate_sensitivity(
        df, strategy, volume_markets_only=volume_markets_only
    )
    
    # Analyze results
    baseline_pnl = slippage_results[slippage_results['slippage_bps'] == 0]['total_pnl'].values[0]
    
    # Slippage tolerance: At what slippage does PnL go negative?
    negative_slip = slippage_results[slippage_results['total_pnl'] < 0]
    slip_tolerance = negative_slip['slippage_bps'].min() if len(negative_slip) > 0 else 999
    
    # Spread robustness: Does edge hold at 1.5x spreads?
    spread_15x = spread_results[spread_results['widen_factor'] == 1.5]
    spread_robust = spread_15x['total_pnl'].values[0] > 0 if len(spread_15x) > 0 else False
    
    # Volatility dependence: Does edge persist when removing top 10% volatile seconds?
    vol_10 = volatility_results[volatility_results['pct_removed'] == 0.10]
    vol_robust = vol_10['total_pnl'].values[0] > 0 if len(vol_10) > 0 else False
    
    # Fill rate sensitivity: Is PnL positive across reasonable fill rate assumptions?
    reasonable_rates = fill_rate_results[
        (fill_rate_results['touch_trade_rate'] >= 0.05) & 
        (fill_rate_results['touch_trade_rate'] <= 0.20)
    ]
    fill_rate_robust = (reasonable_rates['total_pnl'] > 0).mean() > 0.5
    
    return {
        'slippage_test': {
            'results': slippage_results.to_dict('records'),
            'tolerance_bps': slip_tolerance,
            'interpretation': f"Strategy tolerates up to {slip_tolerance}bps slippage",
        },
        'spread_test': {
            'results': spread_results.to_dict('records'),
            'robust_at_15x': spread_robust,
            'interpretation': "Robust to widened spreads" if spread_robust else "Edge disappears with wider spreads",
        },
        'volatility_test': {
            'results': volatility_results.to_dict('records'),
            'robust_without_top10pct': vol_robust,
            'interpretation': "Edge persists without volatile periods" if vol_robust else "Edge depends on volatility",
        },
        'fill_rate_test': {
            'results': fill_rate_results.to_dict('records'),
            'robust_across_rates': fill_rate_robust,
            'interpretation': "Robust to fill rate assumptions" if fill_rate_robust else "Sensitive to fill rate model",
        },
        'baseline_pnl': baseline_pnl,
        'robustness_score': sum([spread_robust, vol_robust, fill_rate_robust, slip_tolerance > 50]) / 4,
    }


def print_stress_test_report(results: Dict[str, Any]):
    """Print formatted stress test report."""
    print("\n" + "="*80)
    print("STRESS TEST REPORT")
    print("="*80)
    
    print(f"\nBaseline PnL: ${results['baseline_pnl']:.4f}")
    print(f"Robustness Score: {results['robustness_score']*100:.0f}%")
    
    # Slippage test
    print("\n--- 1. Extra Slippage Test ---")
    slip = results['slippage_test']
    print(f"  Slippage tolerance: {slip['tolerance_bps']}bps")
    print(f"  {slip['interpretation']}")
    print(f"\n  {'Slippage':>10} {'PnL':>12} {'t-stat':>10}")
    print(f"  {'-'*35}")
    for row in slip['results']:
        print(f"  {row['slippage_bps']:>8}bp ${row['total_pnl']:>10.4f} {row['t_stat']:>10.2f}")
    
    # Spread test
    print("\n--- 2. Widened Spreads Test ---")
    spr = results['spread_test']
    print(f"  Robust at 1.5x: {'Yes' if spr['robust_at_15x'] else 'No'}")
    print(f"  {spr['interpretation']}")
    print(f"\n  {'Factor':>8} {'Avg Spread':>12} {'PnL':>12} {'t-stat':>10}")
    print(f"  {'-'*45}")
    for row in spr['results']:
        print(f"  {row['widen_factor']:>8.2f}x {row['avg_spread']*100:>10.2f}c ${row['total_pnl']:>10.4f} {row['t_stat']:>10.2f}")
    
    # Volatility test
    print("\n--- 3. Volatile Seconds Removal Test ---")
    vol = results['volatility_test']
    print(f"  Robust without top 10%: {'Yes' if vol['robust_without_top10pct'] else 'No'}")
    print(f"  {vol['interpretation']}")
    print(f"\n  {'% Removed':>10} {'PnL':>12} {'t-stat':>10} {'Fills':>10}")
    print(f"  {'-'*45}")
    for row in vol['results']:
        print(f"  {row['pct_removed']*100:>8.0f}% ${row['total_pnl']:>10.4f} {row['t_stat']:>10.2f} {row['n_fills']:>10.0f}")
    
    # Fill rate test
    print("\n--- 4. Fill Rate Sensitivity Test ---")
    fr = results['fill_rate_test']
    print(f"  Robust across rates: {'Yes' if fr['robust_across_rates'] else 'No'}")
    print(f"  {fr['interpretation']}")
    print(f"\n  {'Rate':>8} {'PnL':>12} {'t-stat':>10} {'Fill%':>10}")
    print(f"  {'-'*45}")
    for row in fr['results']:
        print(f"  {row['touch_trade_rate']:>8.2f} ${row['total_pnl']:>10.4f} {row['t_stat']:>10.2f} {row['fill_rate']*100:>9.2f}%")
    
    print("\n" + "="*80)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    import sys
    
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from scripts.backtest.data_loader import load_eth_markets, add_derived_columns
    
    print("Loading data...")
    df, market_info = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    # Create strategy
    strategy = SpreadCaptureStrategy(
        spread_min=0.01,
        tau_min=60,
        tau_max=600,
        inventory_limit_up=10.0,
        inventory_limit_down=10.0,
        tau_flatten=60,
        two_sided=True,
    )
    
    print(f"\nTesting strategy: {strategy.name}")
    
    # Run stress tests
    results = run_all_stress_tests(
        df, strategy, volume_markets_only=True
    )
    
    print_stress_test_report(results)

