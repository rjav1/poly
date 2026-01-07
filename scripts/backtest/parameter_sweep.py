"""
Parameter Sweep with Train/Test Split

Systematically sweep strategy parameters with proper out-of-sample validation.
Uses chronological train/test split to avoid lookahead bias.
"""

from itertools import product
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import json

try:
    from .backtest_engine import run_backtest, run_maker_backtest, ExecutionConfig
    from .strategies import (
        LatencyCaptureStrategy, StrikeCrossStrategy,
        MomentumStrategy, NearStrikeStrategy, SpreadCaptureStrategy
    )
    from .data_loader import get_train_test_split, get_walk_forward_splits
    from .maker_execution import MakerExecutionConfig, FillModel
except ImportError:
    from backtest_engine import run_backtest, run_maker_backtest, ExecutionConfig
    from strategies import (
        LatencyCaptureStrategy, StrikeCrossStrategy,
        MomentumStrategy, NearStrikeStrategy, SpreadCaptureStrategy
    )
    from data_loader import get_train_test_split, get_walk_forward_splits
    from maker_execution import MakerExecutionConfig, FillModel


def create_parameter_grid() -> List[Dict[str, Any]]:
    """
    Define parameter grid for sweep.
    
    Returns:
        List of parameter dictionaries
    """
    grid = []
    
    # StrikeCross variations (showed best results)
    for tau_max in [60, 120, 180, 240, 300, 450, 600]:
        for hold_to_expiry in [True, False]:
            if hold_to_expiry:
                grid.append({
                    'strategy': 'StrikeCross',
                    'tau_max': tau_max,
                    'hold_to_expiry': True,
                    'hold_seconds': None,
                })
            else:
                for hold_seconds in [15, 30, 60, 120]:
                    grid.append({
                        'strategy': 'StrikeCross',
                        'tau_max': tau_max,
                        'hold_to_expiry': False,
                        'hold_seconds': hold_seconds,
                    })
    
    # LatencyCapture variations
    for threshold in [3, 5, 7, 10, 15]:
        for hold in [10, 15, 30, 60]:
            for tau_max in [300, 600, 900]:
                grid.append({
                    'strategy': 'LatencyCapture',
                    'threshold_bps': threshold,
                    'hold_seconds': hold,
                    'tau_max': tau_max,
                })
    
    # NearStrike variations
    for near_bps in [10, 20, 30, 50]:
        for min_move in [2, 3, 5, 8]:
            for hold in [10, 15, 30]:
                grid.append({
                    'strategy': 'NearStrike',
                    'near_strike_bps': near_bps,
                    'min_move_bps': min_move,
                    'hold_seconds': hold,
                })
    
    # Momentum variations
    for lookback in [3, 5, 10]:
        for min_move in [8, 10, 15, 20]:
            for hold in [10, 15, 30]:
                grid.append({
                    'strategy': 'Momentum',
                    'lookback': lookback,
                    'min_total_move_bps': min_move,
                    'hold_seconds': hold,
                })
    
    return grid


def params_to_strategy(params: Dict[str, Any]):
    """Convert parameter dict to Strategy object."""
    strategy_type = params['strategy']
    
    if strategy_type == 'StrikeCross':
        return StrikeCrossStrategy(
            tau_max=params['tau_max'],
            hold_to_expiry=params['hold_to_expiry'],
            hold_seconds=params.get('hold_seconds', 60),
        )
    elif strategy_type == 'LatencyCapture':
        return LatencyCaptureStrategy(
            threshold_bps=params['threshold_bps'],
            hold_seconds=params['hold_seconds'],
            tau_max=params.get('tau_max', 900),
        )
    elif strategy_type == 'NearStrike':
        return NearStrikeStrategy(
            near_strike_bps=params['near_strike_bps'],
            min_move_bps=params['min_move_bps'],
            hold_seconds=params['hold_seconds'],
        )
    elif strategy_type == 'Momentum':
        return MomentumStrategy(
            lookback=params['lookback'],
            min_total_move_bps=params['min_total_move_bps'],
            hold_seconds=params['hold_seconds'],
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_type}")


def run_parameter_sweep(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    param_grid: List[Dict[str, Any]] = None,
    latencies: List[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run parameter sweep on train data, evaluate on test.
    
    Args:
        train_df: Training data
        test_df: Test data
        param_grid: List of parameter combinations
        latencies: Latencies to test for each param combo
        verbose: Print progress
        
    Returns:
        DataFrame with results
    """
    if param_grid is None:
        param_grid = create_parameter_grid()
    
    if latencies is None:
        latencies = [0, 2, 5]  # Key latencies
    
    results = []
    total = len(param_grid) * len(latencies)
    
    for i, params in enumerate(param_grid):
        strategy = params_to_strategy(params)
        
        for latency in latencies:
            config = ExecutionConfig(
                signal_latency_s=latency // 2,
                exec_latency_s=latency - latency // 2
            )
            
            # Run on train
            train_result = run_backtest(train_df, strategy, config, verbose=False)
            
            # Run on test
            test_result = run_backtest(test_df, strategy, config, verbose=False)
            
            results.append({
                **params,
                'latency': latency,
                'train_total_pnl': train_result['metrics']['total_pnl'],
                'train_mean_pnl': train_result['metrics']['mean_pnl_per_market'],
                'train_t_stat': train_result['metrics']['t_stat'],
                'train_hit_rate': train_result['metrics']['hit_rate_per_market'],
                'train_n_trades': train_result['metrics']['n_trades'],
                'test_total_pnl': test_result['metrics']['total_pnl'],
                'test_mean_pnl': test_result['metrics']['mean_pnl_per_market'],
                'test_t_stat': test_result['metrics']['t_stat'],
                'test_hit_rate': test_result['metrics']['hit_rate_per_market'],
                'test_n_trades': test_result['metrics']['n_trades'],
            })
            
            if verbose and (i * len(latencies) + latencies.index(latency) + 1) % 50 == 0:
                print(f"Progress: {i * len(latencies) + latencies.index(latency) + 1}/{total}")
    
    return pd.DataFrame(results)


def analyze_sweep_results(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze sweep results to find best parameters.
    
    Args:
        results_df: Output from run_parameter_sweep
        
    Returns:
        Analysis dictionary
    """
    # Filter to strategies with at least some trades
    active = results_df[results_df['train_n_trades'] >= 5]
    
    if active.empty:
        return {'error': 'No strategies with sufficient trades'}
    
    # Best by train t-stat
    best_train_idx = active['train_t_stat'].idxmax()
    best_train = active.loc[best_train_idx].to_dict()
    
    # Best by test PnL (among positive train)
    positive_train = active[active['train_total_pnl'] > 0]
    if not positive_train.empty:
        best_test_idx = positive_train['test_total_pnl'].idxmax()
        best_test = positive_train.loc[best_test_idx].to_dict()
    else:
        best_test = None
    
    # Most robust (smallest gap between train and test)
    active = active.copy()
    active['train_test_gap'] = abs(active['train_mean_pnl'] - active['test_mean_pnl'])
    robust_idx = active[active['train_mean_pnl'] > 0]['train_test_gap'].idxmin()
    most_robust = active.loc[robust_idx].to_dict() if robust_idx is not None else None
    
    # Summary statistics
    strategies_tested = results_df['strategy'].nunique()
    total_combos = len(results_df)
    
    # Best by strategy type
    best_by_type = {}
    for strategy in results_df['strategy'].unique():
        type_df = results_df[results_df['strategy'] == strategy]
        type_active = type_df[type_df['train_n_trades'] >= 3]
        if not type_active.empty:
            best_idx = type_active['train_t_stat'].idxmax()
            best_by_type[strategy] = type_active.loc[best_idx].to_dict()
    
    return {
        'best_train': best_train,
        'best_test': best_test,
        'most_robust': most_robust,
        'best_by_type': best_by_type,
        'strategies_tested': strategies_tested,
        'total_combinations': total_combos,
        'profitable_train': (results_df['train_total_pnl'] > 0).sum(),
        'profitable_test': (results_df['test_total_pnl'] > 0).sum(),
    }


def get_top_strategies(
    results_df: pd.DataFrame,
    n: int = 10,
    min_trades: int = 5,
    sort_by: str = 'train_t_stat'
) -> pd.DataFrame:
    """
    Get top N strategies by specified metric.
    
    Args:
        results_df: Sweep results
        n: Number of top strategies
        min_trades: Minimum trades required
        sort_by: Column to sort by
        
    Returns:
        Top N strategies
    """
    filtered = results_df[results_df['train_n_trades'] >= min_trades]
    return filtered.nlargest(n, sort_by)


def run_walk_forward_sweep(
    df: pd.DataFrame,
    param_grid: List[Dict[str, Any]] = None,
    latencies: List[int] = None,
    train_size: int = 10,
    test_size: int = 4,
    step_size: int = 4,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run parameter sweep with walk-forward validation.
    
    This is the preferred method for robust out-of-sample evaluation.
    Runs each parameter combination across multiple train/test folds and
    aggregates results.
    
    Args:
        df: Full DataFrame with all markets
        param_grid: List of parameter combinations
        latencies: Latencies to test
        train_size: Number of markets in each training fold
        test_size: Number of markets in each test fold
        step_size: Step between folds
        verbose: Print progress
        
    Returns:
        Tuple of (per_fold_results, aggregated_results)
    """
    if param_grid is None:
        param_grid = create_parameter_grid()
    
    if latencies is None:
        latencies = [0, 2, 5]
    
    # Get walk-forward splits
    splits = get_walk_forward_splits(df, train_size, test_size, step_size)
    n_folds = len(splits)
    
    if n_folds == 0:
        print("Warning: No walk-forward splits possible with current parameters")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"Walk-forward: {n_folds} folds, {train_size} train / {test_size} test markets each")
    
    # Run sweep for each fold
    all_fold_results = []
    
    for fold_idx, (train_df, test_df, train_ids, test_ids) in enumerate(splits):
        if verbose:
            print(f"\nFold {fold_idx + 1}/{n_folds}...")
        
        for params in param_grid:
            strategy = params_to_strategy(params)
            
            for latency in latencies:
                config = ExecutionConfig(
                    signal_latency_s=latency // 2,
                    exec_latency_s=latency - latency // 2
                )
                
                # Run on train and test
                train_result = run_backtest(train_df, strategy, config, verbose=False)
                test_result = run_backtest(test_df, strategy, config, verbose=False)
                
                all_fold_results.append({
                    'fold': fold_idx,
                    **params,
                    'latency': latency,
                    'train_total_pnl': train_result['metrics']['total_pnl'],
                    'train_mean_pnl': train_result['metrics']['mean_pnl_per_market'],
                    'train_t_stat': train_result['metrics']['t_stat'],
                    'train_n_trades': train_result['metrics']['n_trades'],
                    'test_total_pnl': test_result['metrics']['total_pnl'],
                    'test_mean_pnl': test_result['metrics']['mean_pnl_per_market'],
                    'test_t_stat': test_result['metrics']['t_stat'],
                    'test_n_trades': test_result['metrics']['n_trades'],
                })
    
    per_fold_df = pd.DataFrame(all_fold_results)
    
    # Aggregate across folds
    # Group by strategy params and latency, compute mean/std across folds
    group_cols = [c for c in per_fold_df.columns if c not in [
        'fold', 'train_total_pnl', 'train_mean_pnl', 'train_t_stat', 'train_n_trades',
        'test_total_pnl', 'test_mean_pnl', 'test_t_stat', 'test_n_trades'
    ]]
    
    aggregated = per_fold_df.groupby(group_cols, dropna=False).agg({
        'train_total_pnl': ['mean', 'std'],
        'train_t_stat': ['mean', 'std'],
        'train_n_trades': 'sum',
        'test_total_pnl': ['mean', 'std'],
        'test_t_stat': ['mean', 'std'],
        'test_n_trades': 'sum',
    }).reset_index()
    
    # Flatten column names
    aggregated.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col 
        for col in aggregated.columns
    ]
    
    # Compute overall t-stat (more reliable than averaging t-stats)
    aggregated['n_folds'] = n_folds
    aggregated['test_pnl_se'] = aggregated['test_total_pnl_std'] / np.sqrt(n_folds)
    aggregated['wf_t_stat'] = aggregated['test_total_pnl_mean'] / aggregated['test_pnl_se']
    aggregated['wf_t_stat'] = aggregated['wf_t_stat'].fillna(0)
    
    return per_fold_df, aggregated


# ==============================================================================
# MAKER STRATEGY PARAMETER SWEEPS
# ==============================================================================

def create_maker_parameter_grid() -> List[Dict[str, Any]]:
    """
    Define parameter grid for maker/spread capture strategy sweep.
    
    Key parameters to sweep:
    - spread_min: Minimum spread to quote (filter for opportunities)
    - tau_window: (tau_min, tau_max) - when to quote
    - inventory_limit: Maximum position size
    - tau_flatten: When to flatten positions
    - touch_trade_rate: Fill model calibration (affects fill rate)
    
    Returns:
        List of parameter dictionaries for maker strategies
    """
    grid = []
    
    # Main parameter variations
    for spread_min in [0.01, 0.015, 0.02, 0.025, 0.03]:
        for tau_min, tau_max in [(60, 600), (90, 540), (120, 480), (120, 600), (60, 480)]:
            for inventory_limit in [5.0, 10.0, 15.0, 20.0]:
                for tau_flatten in [30, 60, 90]:
                    # Ensure tau_flatten < tau_min
                    if tau_flatten >= tau_min:
                        continue
                    
                    grid.append({
                        'strategy': 'SpreadCapture',
                        'spread_min': spread_min,
                        'tau_min': tau_min,
                        'tau_max': tau_max,
                        'inventory_limit_up': inventory_limit,
                        'inventory_limit_down': inventory_limit,
                        'tau_flatten': tau_flatten,
                        'two_sided': True,
                        'quote_size': 1.0,
                        'adverse_selection_filter': True,
                    })
    
    # Variations with adverse selection filter off
    for spread_min in [0.01, 0.02]:
        for tau_min, tau_max in [(60, 600), (120, 480)]:
            grid.append({
                'strategy': 'SpreadCapture',
                'spread_min': spread_min,
                'tau_min': tau_min,
                'tau_max': tau_max,
                'inventory_limit_up': 10.0,
                'inventory_limit_down': 10.0,
                'tau_flatten': 60,
                'two_sided': True,
                'quote_size': 1.0,
                'adverse_selection_filter': False,
            })
    
    return grid


def create_maker_execution_configs() -> List[MakerExecutionConfig]:
    """
    Create a list of maker execution configs to test.
    
    Tests different latency and fill model parameters.
    
    Returns:
        List of MakerExecutionConfig objects
    """
    configs = []
    
    # Different latency levels
    for place_latency in [0, 50, 100, 200]:
        for cancel_latency in [0, 25, 50]:
            # Different fill rate assumptions
            for touch_rate in [0.05, 0.10, 0.15, 0.20]:
                configs.append(MakerExecutionConfig(
                    place_latency_ms=place_latency,
                    cancel_latency_ms=cancel_latency,
                    fill_model=FillModel.TOUCH_SIZE_PROXY,
                    touch_trade_rate_per_second=touch_rate,
                ))
    
    return configs


def maker_params_to_strategy(params: Dict[str, Any]) -> SpreadCaptureStrategy:
    """Convert parameter dict to SpreadCaptureStrategy object."""
    return SpreadCaptureStrategy(
        spread_min=params.get('spread_min', 0.02),
        tau_min=params.get('tau_min', 120),
        tau_max=params.get('tau_max', 600),
        inventory_limit_up=params.get('inventory_limit_up', 10.0),
        inventory_limit_down=params.get('inventory_limit_down', 10.0),
        tau_flatten=params.get('tau_flatten', 60),
        two_sided=params.get('two_sided', True),
        quote_size=params.get('quote_size', 1.0),
        adverse_selection_filter=params.get('adverse_selection_filter', True),
    )


def run_maker_parameter_sweep(
    df: pd.DataFrame,
    param_grid: List[Dict[str, Any]] = None,
    configs: List[MakerExecutionConfig] = None,
    volume_markets_only: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run parameter sweep for maker strategies.
    
    Unlike taker strategies, maker strategies are evaluated on the same
    data (no train/test split) since they use size data which is limited.
    Statistical significance is assessed via per-market clustering.
    
    Args:
        df: DataFrame with market data
        param_grid: List of strategy parameter combinations
        configs: List of maker execution configs
        volume_markets_only: Only use markets with size data
        verbose: Print progress
        
    Returns:
        DataFrame with results
    """
    if param_grid is None:
        # Use a smaller default grid
        param_grid = [
            {'strategy': 'SpreadCapture', 'spread_min': s, 'tau_min': 120, 'tau_max': 600,
             'inventory_limit_up': 10.0, 'inventory_limit_down': 10.0, 'tau_flatten': 60,
             'two_sided': True, 'quote_size': 1.0, 'adverse_selection_filter': True}
            for s in [0.01, 0.015, 0.02, 0.025, 0.03]
        ]
    
    if configs is None:
        # Use a smaller default config set
        configs = [
            MakerExecutionConfig(place_latency_ms=0, cancel_latency_ms=0, 
                                fill_model=FillModel.TOUCH_SIZE_PROXY, touch_trade_rate_per_second=0.10),
            MakerExecutionConfig(place_latency_ms=50, cancel_latency_ms=25, 
                                fill_model=FillModel.TOUCH_SIZE_PROXY, touch_trade_rate_per_second=0.10),
            MakerExecutionConfig(place_latency_ms=100, cancel_latency_ms=50, 
                                fill_model=FillModel.TOUCH_SIZE_PROXY, touch_trade_rate_per_second=0.10),
        ]
    
    results = []
    total = len(param_grid) * len(configs)
    
    for i, params in enumerate(param_grid):
        strategy = maker_params_to_strategy(params)
        
        for j, config in enumerate(configs):
            # Run backtest
            result = run_maker_backtest(
                df, strategy, config, 
                verbose=False, 
                volume_markets_only=volume_markets_only
            )
            
            metrics = result.get('metrics', {})
            
            results.append({
                **params,
                'place_latency_ms': config.place_latency_ms,
                'cancel_latency_ms': config.cancel_latency_ms,
                'touch_trade_rate': config.touch_trade_rate_per_second,
                'total_pnl': metrics.get('total_pnl', 0),
                'mean_pnl_per_market': metrics.get('mean_pnl_per_market', 0),
                't_stat': metrics.get('t_stat', 0),
                'hit_rate': metrics.get('hit_rate_per_market', 0),
                'n_fills': metrics.get('n_fills', 0),
                'fill_rate': metrics.get('fill_rate', 0),
                'spread_captured': metrics.get('spread_captured_total', 0),
                'adverse_selection': metrics.get('adverse_selection_total', 0),
                'inventory_carry': metrics.get('inventory_carry_total', 0),
                'orders_placed': metrics.get('orders_placed_total', 0),
                'avg_time_to_fill': metrics.get('avg_time_to_fill', 0),
            })
            
            if verbose and (i * len(configs) + j + 1) % 10 == 0:
                print(f"Progress: {i * len(configs) + j + 1}/{total}")
    
    return pd.DataFrame(results)


def analyze_maker_sweep_results(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze maker sweep results to find best parameters.
    
    Args:
        results_df: Output from run_maker_parameter_sweep
        
    Returns:
        Analysis dictionary
    """
    if results_df.empty:
        return {'error': 'No results to analyze'}
    
    # Filter to strategies with fills
    active = results_df[results_df['n_fills'] > 0].copy()
    
    if active.empty:
        return {'error': 'No strategies with fills'}
    
    # Best by t-stat
    best_tstat_idx = active['t_stat'].idxmax()
    best_tstat = active.loc[best_tstat_idx].to_dict()
    
    # Best by total PnL
    best_pnl_idx = active['total_pnl'].idxmax()
    best_pnl = active.loc[best_pnl_idx].to_dict()
    
    # Best by spread captured / adverse selection ratio
    active['edge_ratio'] = active['spread_captured'] / (active['adverse_selection'].abs() + 0.001)
    best_ratio_idx = active['edge_ratio'].idxmax()
    best_ratio = active.loc[best_ratio_idx].to_dict()
    
    # Best by fill rate (among profitable)
    profitable = active[active['total_pnl'] > 0]
    if not profitable.empty:
        best_fill_rate_idx = profitable['fill_rate'].idxmax()
        best_fill_rate = profitable.loc[best_fill_rate_idx].to_dict()
    else:
        best_fill_rate = None
    
    # Summary statistics
    return {
        'best_by_tstat': best_tstat,
        'best_by_pnl': best_pnl,
        'best_by_edge_ratio': best_ratio,
        'best_by_fill_rate': best_fill_rate,
        'n_combinations_tested': len(results_df),
        'n_profitable': (results_df['total_pnl'] > 0).sum(),
        'n_significant': (results_df['t_stat'] > 1.96).sum(),
        'avg_fill_rate': results_df['fill_rate'].mean(),
        'avg_t_stat': results_df['t_stat'].mean(),
    }


def get_top_maker_strategies(
    results_df: pd.DataFrame,
    n: int = 10,
    min_fills: int = 5,
    sort_by: str = 't_stat'
) -> pd.DataFrame:
    """
    Get top N maker strategies by specified metric.
    
    Args:
        results_df: Sweep results
        n: Number of top strategies
        min_fills: Minimum fills required
        sort_by: Column to sort by
        
    Returns:
        Top N strategies
    """
    filtered = results_df[results_df['n_fills'] >= min_fills]
    return filtered.nlargest(n, sort_by)


def analyze_walk_forward_results(
    aggregated_df: pd.DataFrame,
    min_trades: int = 5
) -> Dict[str, Any]:
    """
    Analyze walk-forward sweep results.
    
    Args:
        aggregated_df: Aggregated results from run_walk_forward_sweep
        min_trades: Minimum total trades required
        
    Returns:
        Analysis dictionary
    """
    if aggregated_df.empty:
        return {'error': 'No results to analyze'}
    
    # Filter by minimum trades
    active = aggregated_df[aggregated_df['test_n_trades_sum'] >= min_trades].copy()
    
    if active.empty:
        return {'error': 'No strategies with sufficient trades'}
    
    # Best by walk-forward t-stat
    best_idx = active['wf_t_stat'].idxmax()
    best_wf = active.loc[best_idx].to_dict()
    
    # Best by average test PnL
    best_pnl_idx = active['test_total_pnl_mean'].idxmax()
    best_pnl = active.loc[best_pnl_idx].to_dict()
    
    # Most consistent (lowest std across folds)
    positive_avg = active[active['test_total_pnl_mean'] > 0]
    if not positive_avg.empty:
        most_consistent_idx = positive_avg['test_total_pnl_std'].idxmin()
        most_consistent = positive_avg.loc[most_consistent_idx].to_dict()
    else:
        most_consistent = None
    
    return {
        'best_wf_tstat': best_wf,
        'best_avg_pnl': best_pnl,
        'most_consistent': most_consistent,
        'n_strategies_tested': len(aggregated_df),
        'n_profitable_avg': (aggregated_df['test_total_pnl_mean'] > 0).sum(),
        'n_significant': (aggregated_df['wf_t_stat'] > 1.96).sum(),
    }


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
    
    # Create smaller grid for quick test
    quick_grid = []
    for tau_max in [120, 300, 600]:
        quick_grid.append({
            'strategy': 'StrikeCross',
            'tau_max': tau_max,
            'hold_to_expiry': True,
            'hold_seconds': None,
        })
    for threshold in [3, 5, 10]:
        quick_grid.append({
            'strategy': 'LatencyCapture',
            'threshold_bps': threshold,
            'hold_seconds': 15,
            'tau_max': 900,
        })
    
    print("\n" + "="*80)
    print("WALK-FORWARD PARAMETER SWEEP")
    print("="*80)
    
    # Run walk-forward sweep
    per_fold_df, aggregated_df = run_walk_forward_sweep(
        df,
        param_grid=quick_grid,
        latencies=[0, 2, 5],
        train_size=10,
        test_size=4,
        step_size=4,
        verbose=True
    )
    
    if not aggregated_df.empty:
        # Analyze
        wf_analysis = analyze_walk_forward_results(aggregated_df, min_trades=3)
        
        print("\n" + "-"*80)
        print("WALK-FORWARD RESULTS")
        print("-"*80)
        
        print(f"\nStrategies tested: {wf_analysis['n_strategies_tested']}")
        print(f"Profitable on average: {wf_analysis['n_profitable_avg']}")
        print(f"Statistically significant (t>1.96): {wf_analysis['n_significant']}")
        
        if 'best_wf_tstat' in wf_analysis:
            print("\n--- Best by Walk-Forward t-stat ---")
            bw = wf_analysis['best_wf_tstat']
            print(f"Strategy: {bw.get('strategy', 'N/A')}")
            print(f"  tau_max: {bw.get('tau_max', 'N/A')}")
            print(f"  latency: {bw.get('latency', 0)}s")
            print(f"  Avg test PnL: ${bw.get('test_total_pnl_mean', 0):.4f} +/- ${bw.get('test_total_pnl_std', 0):.4f}")
            print(f"  WF t-stat: {bw.get('wf_t_stat', 0):.2f}")
            print(f"  Total trades: {bw.get('test_n_trades_sum', 0)}")
        
        # Show top 5
        print("\n--- Top 5 by Walk-Forward t-stat ---")
        top5_cols = ['strategy', 'tau_max', 'latency', 'test_total_pnl_mean', 'test_total_pnl_std', 'wf_t_stat', 'test_n_trades_sum']
        available_cols = [c for c in top5_cols if c in aggregated_df.columns]
        top5 = aggregated_df.nlargest(5, 'wf_t_stat')[available_cols]
        print(top5.to_string())
        
        # Save results
        output_dir = project_root / 'data_v2' / 'backtest_results'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        per_fold_df.to_csv(output_dir / 'walk_forward_per_fold.csv', index=False)
        aggregated_df.to_csv(output_dir / 'walk_forward_aggregated.csv', index=False)
        
        with open(output_dir / 'walk_forward_analysis.json', 'w') as f:
            def convert(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif pd.isna(obj):
                    return None
                return obj
            
            json.dump(convert(wf_analysis), f, indent=2)
        
        print(f"\nResults saved to {output_dir}")

