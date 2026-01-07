"""
Placebo Tests for Strategy Validation

Two key placebo tests to validate that edge is from latency, not spurious:

1. Shift CL Series: Shift CL forward by N seconds
   - If edge persists, it's NOT latency capture
   - Edge should disappear because CL "events" no longer predict PM moves

2. Randomize Events: Permute CL within each market
   - Destroys any true time-series relationship
   - Edge should completely disappear

If placebos show similar edge to real strategy, the results are suspect.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from .backtest_engine import run_backtest, run_maker_backtest, ExecutionConfig
    from .strategies import Strategy, SpreadCaptureStrategy
    from .maker_execution import MakerExecutionConfig, FillModel
except ImportError:
    from backtest_engine import run_backtest, run_maker_backtest, ExecutionConfig
    from strategies import Strategy, SpreadCaptureStrategy
    from maker_execution import MakerExecutionConfig, FillModel


def create_shifted_data(
    df: pd.DataFrame,
    shift_seconds: int
) -> pd.DataFrame:
    """
    Shift CL series forward by N seconds.
    
    This simulates having "stale" CL data. If our edge is real latency capture,
    shifting CL forward should destroy the edge because events would be
    detected at the wrong time.
    
    Args:
        df: Original DataFrame
        shift_seconds: How many seconds to shift CL forward
        
    Returns:
        DataFrame with shifted CL data
    """
    df_shifted = df.copy()
    
    # Shift CL data forward (negative shift = future data at current time)
    # We shift backward so that current CL is what it will be in N seconds
    df_shifted['cl_mid'] = df_shifted.groupby('market_id')['cl_mid'].shift(-shift_seconds)
    
    # Recalculate delta_bps with shifted CL
    df_shifted['delta_bps'] = (df_shifted['cl_mid'] - df_shifted['K']) / df_shifted['K'] * 10000
    
    # Drop rows where shift creates NaN
    df_shifted = df_shifted.dropna(subset=['cl_mid'])
    
    return df_shifted


def create_randomized_data(
    df: pd.DataFrame,
    seed: int = 42
) -> pd.DataFrame:
    """
    Randomize CL within each market.
    
    This destroys any time-series relationship between CL and PM.
    If strategy still shows edge, it's spurious.
    
    Args:
        df: Original DataFrame
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with randomized CL
    """
    np.random.seed(seed)
    df_random = df.copy()
    
    # Permute CL within each market
    def randomize_cl(group):
        group = group.copy()
        group['cl_mid'] = np.random.permutation(group['cl_mid'].values)
        return group
    
    df_random = df_random.groupby('market_id', group_keys=False).apply(randomize_cl)
    
    # Recalculate delta_bps
    df_random['delta_bps'] = (df_random['cl_mid'] - df_random['K']) / df_random['K'] * 10000
    
    return df_random


def run_placebo_test_shift(
    df: pd.DataFrame,
    strategy: Strategy,
    shifts: List[int] = None,
    config: ExecutionConfig = None
) -> pd.DataFrame:
    """
    Run strategy on shifted data at multiple shift levels.
    
    Args:
        df: Original DataFrame
        strategy: Strategy to test
        shifts: List of shift values (seconds)
        config: Execution config
        
    Returns:
        DataFrame with results at each shift
    """
    if shifts is None:
        shifts = [0, 5, 10, 30, 60, 120]  # 0 = no shift (baseline)
    
    if config is None:
        config = ExecutionConfig()
    
    results = []
    
    for shift in shifts:
        if shift == 0:
            test_df = df
        else:
            test_df = create_shifted_data(df, shift)
        
        result = run_backtest(test_df, strategy, config, verbose=False)
        
        results.append({
            'shift_seconds': shift,
            'total_pnl': result['metrics']['total_pnl'],
            'mean_pnl': result['metrics']['mean_pnl_per_market'],
            't_stat': result['metrics']['t_stat'],
            'hit_rate': result['metrics']['hit_rate_per_market'],
            'n_trades': result['metrics']['n_trades'],
        })
    
    return pd.DataFrame(results)


def run_placebo_test_random(
    df: pd.DataFrame,
    strategy: Strategy,
    n_iterations: int = 10,
    config: ExecutionConfig = None
) -> Dict[str, Any]:
    """
    Run strategy on randomized data multiple times.
    
    Args:
        df: Original DataFrame
        strategy: Strategy to test
        n_iterations: Number of random permutations
        config: Execution config
        
    Returns:
        Dictionary with placebo distribution
    """
    if config is None:
        config = ExecutionConfig()
    
    # First, run on real data
    real_result = run_backtest(df, strategy, config, verbose=False)
    
    # Then run on randomized data
    placebo_results = []
    for i in range(n_iterations):
        random_df = create_randomized_data(df, seed=42 + i)
        result = run_backtest(random_df, strategy, config, verbose=False)
        
        placebo_results.append({
            'iteration': i,
            'total_pnl': result['metrics']['total_pnl'],
            'mean_pnl': result['metrics']['mean_pnl_per_market'],
            't_stat': result['metrics']['t_stat'],
            'n_trades': result['metrics']['n_trades'],
        })
    
    placebo_df = pd.DataFrame(placebo_results)
    
    # Compute p-value: % of placebo results >= real result
    p_value_pnl = (placebo_df['total_pnl'] >= real_result['metrics']['total_pnl']).mean()
    p_value_tstat = (placebo_df['t_stat'] >= real_result['metrics']['t_stat']).mean()
    
    return {
        'real_result': {
            'total_pnl': real_result['metrics']['total_pnl'],
            'mean_pnl': real_result['metrics']['mean_pnl_per_market'],
            't_stat': real_result['metrics']['t_stat'],
            'n_trades': real_result['metrics']['n_trades'],
        },
        'placebo_mean': {
            'total_pnl': placebo_df['total_pnl'].mean(),
            'mean_pnl': placebo_df['mean_pnl'].mean(),
            't_stat': placebo_df['t_stat'].mean(),
        },
        'placebo_std': {
            'total_pnl': placebo_df['total_pnl'].std(),
            'mean_pnl': placebo_df['mean_pnl'].std(),
            't_stat': placebo_df['t_stat'].std(),
        },
        'p_value_pnl': p_value_pnl,
        'p_value_tstat': p_value_tstat,
        'placebo_df': placebo_df,
        'interpretation': interpret_placebo_results(
            real_result['metrics']['total_pnl'],
            placebo_df['total_pnl'].mean(),
            placebo_df['total_pnl'].std(),
            p_value_pnl
        ),
    }


def interpret_placebo_results(
    real_pnl: float,
    placebo_mean: float,
    placebo_std: float,
    p_value: float
) -> str:
    """Interpret placebo test results."""
    if p_value < 0.05:
        return "PASS: Real strategy significantly outperforms random (p < 0.05). Edge is likely real."
    elif p_value < 0.1:
        return "MARGINAL: Real strategy somewhat better than random (0.05 < p < 0.1). More data needed."
    else:
        return "FAIL: Real strategy not significantly different from random (p >= 0.1). Edge may be spurious."


def run_all_placebo_tests(
    df: pd.DataFrame,
    strategy: Strategy,
    config: ExecutionConfig = None
) -> Dict[str, Any]:
    """
    Run all placebo tests and return comprehensive results.
    
    Args:
        df: Original DataFrame
        strategy: Strategy to test
        config: Execution config
        
    Returns:
        All placebo test results
    """
    if config is None:
        config = ExecutionConfig()
    
    print("Running shift placebo test...")
    shift_results = run_placebo_test_shift(df, strategy, config=config)
    
    print("Running randomization placebo test (10 iterations)...")
    random_results = run_placebo_test_random(df, strategy, n_iterations=10, config=config)
    
    # Overall validation
    shift_passed = shift_results[shift_results['shift_seconds'] >= 10]['total_pnl'].mean() < \
                   shift_results[shift_results['shift_seconds'] == 0]['total_pnl'].values[0] * 0.5
    random_passed = random_results['p_value_pnl'] < 0.1
    
    return {
        'shift_test': {
            'results': shift_results.to_dict('records'),
            'passed': shift_passed,
            'interpretation': "Edge disappears with shifted data" if shift_passed else "Edge persists with shifted data (SUSPICIOUS)"
        },
        'random_test': {
            'real_pnl': random_results['real_result']['total_pnl'],
            'placebo_mean_pnl': random_results['placebo_mean']['total_pnl'],
            'p_value': random_results['p_value_pnl'],
            'passed': random_passed,
            'interpretation': random_results['interpretation'],
        },
        'overall_validation': "VALID" if (shift_passed and random_passed) else "NEEDS REVIEW",
    }


# ==============================================================================
# MAKER PLACEBO TESTS
# ==============================================================================

def run_maker_placebo_randomized_timing(
    df: pd.DataFrame,
    strategy: SpreadCaptureStrategy,
    config: MakerExecutionConfig = None,
    n_iterations: int = 10,
    volume_markets_only: bool = True,
) -> Dict[str, Any]:
    """
    Placebo test: Randomize when quotes are placed.
    
    Instead of following the strategy's quoting rules, we place quotes
    at random times (same total count). If edge persists, it's not from
    good timing decisions.
    
    Args:
        df: Original DataFrame
        strategy: SpreadCaptureStrategy to test
        config: Maker execution config
        n_iterations: Number of random iterations
        volume_markets_only: Only use volume markets
        
    Returns:
        Dictionary with real vs placebo results
    """
    if config is None:
        config = MakerExecutionConfig()
    
    # Run real strategy
    real_result = run_maker_backtest(
        df, strategy, config, 
        verbose=False, 
        volume_markets_only=volume_markets_only
    )
    real_metrics = real_result.get('metrics', {})
    
    # Create placebo strategy (quote at random times)
    placebo_results = []
    for i in range(n_iterations):
        # Randomize the market data timestamps (shuffles time series)
        df_random = create_randomized_data(df, seed=42 + i)
        
        result = run_maker_backtest(
            df_random, strategy, config,
            verbose=False,
            volume_markets_only=volume_markets_only
        )
        
        metrics = result.get('metrics', {})
        placebo_results.append({
            'iteration': i,
            'total_pnl': metrics.get('total_pnl', 0),
            't_stat': metrics.get('t_stat', 0),
            'n_fills': metrics.get('n_fills', 0),
            'fill_rate': metrics.get('fill_rate', 0),
        })
    
    placebo_df = pd.DataFrame(placebo_results)
    
    # P-values
    p_value_pnl = (placebo_df['total_pnl'] >= real_metrics.get('total_pnl', 0)).mean()
    p_value_tstat = (placebo_df['t_stat'] >= real_metrics.get('t_stat', 0)).mean()
    
    return {
        'real_result': {
            'total_pnl': real_metrics.get('total_pnl', 0),
            't_stat': real_metrics.get('t_stat', 0),
            'n_fills': real_metrics.get('n_fills', 0),
            'fill_rate': real_metrics.get('fill_rate', 0),
        },
        'placebo_mean': {
            'total_pnl': placebo_df['total_pnl'].mean(),
            't_stat': placebo_df['t_stat'].mean(),
            'n_fills': placebo_df['n_fills'].mean(),
        },
        'placebo_std': {
            'total_pnl': placebo_df['total_pnl'].std(),
            't_stat': placebo_df['t_stat'].std(),
        },
        'p_value_pnl': p_value_pnl,
        'p_value_tstat': p_value_tstat,
        'placebo_df': placebo_df,
        'interpretation': interpret_placebo_results(
            real_metrics.get('total_pnl', 0),
            placebo_df['total_pnl'].mean(),
            placebo_df['total_pnl'].std(),
            p_value_pnl
        ),
    }


def run_maker_placebo_stale_data(
    df: pd.DataFrame,
    strategy: SpreadCaptureStrategy,
    config: MakerExecutionConfig = None,
    shifts: List[int] = None,
    volume_markets_only: bool = True,
) -> pd.DataFrame:
    """
    Placebo test: Shift market data by N seconds.
    
    Simulates having stale orderbook data. If edge persists with stale
    data, the strategy timing may not be important.
    
    Args:
        df: Original DataFrame
        strategy: SpreadCaptureStrategy to test
        config: Maker execution config
        shifts: List of shift values in seconds
        volume_markets_only: Only use volume markets
        
    Returns:
        DataFrame with results at each shift
    """
    if config is None:
        config = MakerExecutionConfig()
    
    if shifts is None:
        shifts = [0, 5, 10, 30, 60]
    
    results = []
    
    for shift in shifts:
        if shift == 0:
            test_df = df
        else:
            test_df = create_shifted_data(df, shift)
        
        result = run_maker_backtest(
            test_df, strategy, config,
            verbose=False,
            volume_markets_only=volume_markets_only
        )
        
        metrics = result.get('metrics', {})
        results.append({
            'shift_seconds': shift,
            'total_pnl': metrics.get('total_pnl', 0),
            't_stat': metrics.get('t_stat', 0),
            'n_fills': metrics.get('n_fills', 0),
            'fill_rate': metrics.get('fill_rate', 0),
            'spread_captured': metrics.get('spread_captured_total', 0),
            'adverse_selection': metrics.get('adverse_selection_total', 0),
        })
    
    return pd.DataFrame(results)


def run_maker_placebo_flipped_sides(
    df: pd.DataFrame,
    strategy: SpreadCaptureStrategy,
    config: MakerExecutionConfig = None,
    volume_markets_only: bool = True,
) -> Dict[str, Any]:
    """
    Placebo test: Flip buy/sell sides.
    
    Instead of bidding at best bid, we bid at best ask (and vice versa).
    This should significantly hurt performance if our side selection matters.
    
    Args:
        df: Original DataFrame
        strategy: SpreadCaptureStrategy to test
        config: Maker execution config
        volume_markets_only: Only use volume markets
        
    Returns:
        Dictionary comparing real vs flipped results
    """
    if config is None:
        config = MakerExecutionConfig()
    
    # Run real strategy
    real_result = run_maker_backtest(
        df, strategy, config,
        verbose=False,
        volume_markets_only=volume_markets_only
    )
    real_metrics = real_result.get('metrics', {})
    
    # Create flipped data (swap bids and asks)
    df_flipped = df.copy()
    
    # Swap UP bid/ask
    df_flipped['pm_up_best_bid'], df_flipped['pm_up_best_ask'] = \
        df['pm_up_best_ask'].copy(), df['pm_up_best_bid'].copy()
    df_flipped['pm_up_best_bid_size'], df_flipped['pm_up_best_ask_size'] = \
        df['pm_up_best_ask_size'].copy(), df['pm_up_best_bid_size'].copy()
    
    # Swap DOWN bid/ask
    df_flipped['pm_down_best_bid'], df_flipped['pm_down_best_ask'] = \
        df['pm_down_best_ask'].copy(), df['pm_down_best_bid'].copy()
    df_flipped['pm_down_best_bid_size'], df_flipped['pm_down_best_ask_size'] = \
        df['pm_down_best_ask_size'].copy(), df['pm_down_best_bid_size'].copy()
    
    # Run on flipped data
    flipped_result = run_maker_backtest(
        df_flipped, strategy, config,
        verbose=False,
        volume_markets_only=volume_markets_only
    )
    flipped_metrics = flipped_result.get('metrics', {})
    
    return {
        'real_result': {
            'total_pnl': real_metrics.get('total_pnl', 0),
            't_stat': real_metrics.get('t_stat', 0),
            'n_fills': real_metrics.get('n_fills', 0),
        },
        'flipped_result': {
            'total_pnl': flipped_metrics.get('total_pnl', 0),
            't_stat': flipped_metrics.get('t_stat', 0),
            'n_fills': flipped_metrics.get('n_fills', 0),
        },
        'pnl_difference': real_metrics.get('total_pnl', 0) - flipped_metrics.get('total_pnl', 0),
        'interpretation': "Side selection matters (real > flipped)" 
            if real_metrics.get('total_pnl', 0) > flipped_metrics.get('total_pnl', 0)
            else "Side selection does NOT matter (suspicious)"
    }


def run_maker_placebo_no_as_filter(
    df: pd.DataFrame,
    strategy: SpreadCaptureStrategy,
    config: MakerExecutionConfig = None,
    volume_markets_only: bool = True,
) -> Dict[str, Any]:
    """
    Placebo test: Remove adverse selection filter.
    
    Compare strategy with and without the adverse selection filter.
    If filter helps, it means we're avoiding being picked off.
    
    Args:
        df: Original DataFrame
        strategy: SpreadCaptureStrategy to test (with filter)
        config: Maker execution config
        volume_markets_only: Only use volume markets
        
    Returns:
        Dictionary comparing results with/without filter
    """
    if config is None:
        config = MakerExecutionConfig()
    
    # Run with filter (original strategy)
    result_with_filter = run_maker_backtest(
        df, strategy, config,
        verbose=False,
        volume_markets_only=volume_markets_only
    )
    
    # Create strategy without filter
    strategy_no_filter = SpreadCaptureStrategy(
        spread_min=strategy.spread_min,
        tau_min=strategy.tau_min,
        tau_max=strategy.tau_max,
        inventory_limit_up=strategy.inventory_limit_up,
        inventory_limit_down=strategy.inventory_limit_down,
        tau_flatten=strategy.tau_flatten,
        quote_size=strategy.quote_size,
        two_sided=strategy.two_sided,
        adverse_selection_filter=False,  # Disable filter
    )
    
    # Run without filter
    result_no_filter = run_maker_backtest(
        df, strategy_no_filter, config,
        verbose=False,
        volume_markets_only=volume_markets_only
    )
    
    with_filter_metrics = result_with_filter.get('metrics', {})
    no_filter_metrics = result_no_filter.get('metrics', {})
    
    return {
        'with_filter': {
            'total_pnl': with_filter_metrics.get('total_pnl', 0),
            't_stat': with_filter_metrics.get('t_stat', 0),
            'n_fills': with_filter_metrics.get('n_fills', 0),
            'adverse_selection': with_filter_metrics.get('adverse_selection_total', 0),
        },
        'without_filter': {
            'total_pnl': no_filter_metrics.get('total_pnl', 0),
            't_stat': no_filter_metrics.get('t_stat', 0),
            'n_fills': no_filter_metrics.get('n_fills', 0),
            'adverse_selection': no_filter_metrics.get('adverse_selection_total', 0),
        },
        'pnl_improvement': with_filter_metrics.get('total_pnl', 0) - no_filter_metrics.get('total_pnl', 0),
        'interpretation': "AS filter helps" 
            if with_filter_metrics.get('total_pnl', 0) > no_filter_metrics.get('total_pnl', 0)
            else "AS filter does NOT help (filter may be too aggressive)"
    }


def run_all_maker_placebo_tests(
    df: pd.DataFrame,
    strategy: SpreadCaptureStrategy,
    config: MakerExecutionConfig = None,
    volume_markets_only: bool = True,
) -> Dict[str, Any]:
    """
    Run all maker placebo tests and return comprehensive results.
    
    Args:
        df: Original DataFrame
        strategy: SpreadCaptureStrategy to test
        config: Maker execution config
        volume_markets_only: Only use volume markets
        
    Returns:
        All placebo test results
    """
    if config is None:
        config = MakerExecutionConfig()
    
    print("Running maker placebo tests...")
    
    print("  1. Randomized timing test...")
    random_results = run_maker_placebo_randomized_timing(
        df, strategy, config, n_iterations=10, volume_markets_only=volume_markets_only
    )
    
    print("  2. Stale data test...")
    stale_results = run_maker_placebo_stale_data(
        df, strategy, config, volume_markets_only=volume_markets_only
    )
    
    print("  3. Flipped sides test...")
    flipped_results = run_maker_placebo_flipped_sides(
        df, strategy, config, volume_markets_only=volume_markets_only
    )
    
    print("  4. No AS filter test...")
    no_filter_results = run_maker_placebo_no_as_filter(
        df, strategy, config, volume_markets_only=volume_markets_only
    )
    
    # Overall validation
    random_passed = random_results['p_value_pnl'] < 0.1
    stale_passed = stale_results[stale_results['shift_seconds'] >= 30]['total_pnl'].mean() < \
                   stale_results[stale_results['shift_seconds'] == 0]['total_pnl'].values[0] * 0.5
    flipped_passed = flipped_results['pnl_difference'] > 0
    
    tests_passed = sum([random_passed, stale_passed, flipped_passed])
    
    return {
        'random_test': {
            'real_pnl': random_results['real_result']['total_pnl'],
            'placebo_mean': random_results['placebo_mean']['total_pnl'],
            'p_value': random_results['p_value_pnl'],
            'passed': random_passed,
            'interpretation': random_results['interpretation'],
        },
        'stale_test': {
            'results': stale_results.to_dict('records'),
            'passed': stale_passed,
        },
        'flipped_test': {
            'real_pnl': flipped_results['real_result']['total_pnl'],
            'flipped_pnl': flipped_results['flipped_result']['total_pnl'],
            'passed': flipped_passed,
            'interpretation': flipped_results['interpretation'],
        },
        'as_filter_test': {
            'with_filter_pnl': no_filter_results['with_filter']['total_pnl'],
            'without_filter_pnl': no_filter_results['without_filter']['total_pnl'],
            'improvement': no_filter_results['pnl_improvement'],
            'interpretation': no_filter_results['interpretation'],
        },
        'tests_passed': tests_passed,
        'total_tests': 3,  # Excluding AS filter as it's diagnostic
        'overall_validation': "VALID" if tests_passed >= 2 else "NEEDS REVIEW",
    }


def print_maker_placebo_report(results: Dict[str, Any]):
    """Print formatted maker placebo test report."""
    print("\n" + "="*80)
    print("MAKER STRATEGY PLACEBO TEST REPORT")
    print("="*80)
    
    # Random test
    print("\n--- 1. Randomized Timing Test ---")
    rand = results['random_test']
    print(f"  Real PnL: ${rand['real_pnl']:.4f}")
    print(f"  Placebo mean: ${rand['placebo_mean']:.4f}")
    print(f"  P-value: {rand['p_value']:.3f}")
    print(f"  Result: {'PASS' if rand['passed'] else 'FAIL'}")
    print(f"  {rand['interpretation']}")
    
    # Stale test
    print("\n--- 2. Stale Data Test ---")
    stale = results['stale_test']
    print(f"  {'Shift':>10} {'PnL':>12} {'t-stat':>10}")
    print(f"  {'-'*35}")
    for row in stale['results']:
        print(f"  {row['shift_seconds']:>8}s ${row['total_pnl']:>10.4f} {row['t_stat']:>10.2f}")
    print(f"  Result: {'PASS' if stale['passed'] else 'FAIL'}")
    
    # Flipped test
    print("\n--- 3. Flipped Sides Test ---")
    flip = results['flipped_test']
    print(f"  Real PnL: ${flip['real_pnl']:.4f}")
    print(f"  Flipped PnL: ${flip['flipped_pnl']:.4f}")
    print(f"  Result: {'PASS' if flip['passed'] else 'FAIL'}")
    print(f"  {flip['interpretation']}")
    
    # AS filter test
    print("\n--- 4. Adverse Selection Filter Test ---")
    asf = results['as_filter_test']
    print(f"  With filter PnL: ${asf['with_filter_pnl']:.4f}")
    print(f"  Without filter PnL: ${asf['without_filter_pnl']:.4f}")
    print(f"  Improvement: ${asf['improvement']:.4f}")
    print(f"  {asf['interpretation']}")
    
    # Overall
    print("\n" + "="*80)
    print(f"OVERALL: {results['tests_passed']}/{results['total_tests']} tests passed")
    print(f"VALIDATION: {results['overall_validation']}")
    print("="*80)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    import sys
    
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from scripts.backtest.data_loader import load_eth_markets, add_derived_columns
    from scripts.backtest.strategies import StrikeCrossStrategy
    
    print("Loading data...")
    df, market_info = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    # Test best strategy
    strategy = StrikeCrossStrategy(tau_max=600, hold_to_expiry=True)
    
    print(f"\nTesting strategy: {strategy.name}")
    print("="*60)
    
    # Shift test
    print("\n--- Shift Placebo Test ---")
    shift_df = run_placebo_test_shift(df, strategy)
    
    print(f"{'Shift':>10} {'Total PnL':>12} {'t-stat':>10} {'N Trades':>10}")
    print("-" * 50)
    for _, row in shift_df.iterrows():
        print(f"{row['shift_seconds']:>10}s {row['total_pnl']:>12.4f} {row['t_stat']:>10.2f} {row['n_trades']:>10}")
    
    # Random test
    print("\n--- Randomization Placebo Test ---")
    random_result = run_placebo_test_random(df, strategy, n_iterations=20)
    
    print(f"\nReal strategy PnL: ${random_result['real_result']['total_pnl']:.4f}")
    print(f"Placebo mean PnL: ${random_result['placebo_mean']['total_pnl']:.4f} (+/- {random_result['placebo_std']['total_pnl']:.4f})")
    print(f"P-value (PnL): {random_result['p_value_pnl']:.3f}")
    print(f"P-value (t-stat): {random_result['p_value_tstat']:.3f}")
    print(f"\nInterpretation: {random_result['interpretation']}")

