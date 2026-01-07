"""
Latency Cliff Analysis

This module determines at what execution latency the edge disappears.
This is the key robustness check before implementing trading strategies.

The "latency cliff" is the point where increased execution latency causes
the strategy to become unprofitable. Understanding this cliff tells us:
1. Whether the edge is real (disappears at realistic latencies)
2. What execution infrastructure is needed
3. Whether the edge is exploitable in practice

IMPORTANT: This module now uses Strategy objects for consistency with
the backtest engine. All analysis uses the same signal generation logic.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np

try:
    from .execution_model import get_effective_prices, ExecutionConfig
    from .strategies import Strategy, Signal, SpreadCaptureStrategy
    from .backtest_engine import run_maker_backtest
    from .maker_execution import MakerExecutionConfig, FillModel
except ImportError:
    from execution_model import get_effective_prices, ExecutionConfig
    from strategies import Strategy, Signal, SpreadCaptureStrategy
    from backtest_engine import run_maker_backtest
    from maker_execution import MakerExecutionConfig, FillModel


@dataclass
class LatencyResult:
    """Results for a specific latency level."""
    latency: int
    strategy_name: str
    total_pnl: float
    n_trades: int
    avg_pnl: float
    hit_rate: float
    avg_entry_price: float
    avg_exit_price: float
    conversion_rate: float  # % of trades using conversion routing


def execute_signal_at_latency(
    market_df: pd.DataFrame,
    signal: Signal,
    latency: int
) -> Optional[Dict[str, Any]]:
    """
    Execute a signal with additional latency applied.
    
    Args:
        market_df: DataFrame for single market
        signal: Signal to execute
        latency: Additional latency in seconds
        
    Returns:
        Trade dict if successful, None otherwise
    """
    # Apply latency to entry/exit
    actual_entry_t = signal.entry_t + latency
    actual_exit_t = signal.exit_t + latency
    
    # Get entry row
    entry_row = market_df[market_df['t'] == actual_entry_t]
    if entry_row.empty:
        return None
    
    # Get exit row (or last available if beyond market end)
    max_t = market_df['t'].max()
    if actual_exit_t > max_t:
        actual_exit_t = max_t
    
    exit_row = market_df[market_df['t'] == actual_exit_t]
    if exit_row.empty:
        return None
    
    # Get effective prices (with conversion routing)
    entry_prices = get_effective_prices(entry_row.iloc[0])
    exit_prices = get_effective_prices(exit_row.iloc[0])
    
    # Execute based on side
    if signal.side == 'buy_up':
        entry_price = entry_prices.buy_up
        exit_price = exit_prices.sell_up
        entry_route = entry_prices.buy_up_route
        exit_route = exit_prices.sell_up_route
    elif signal.side == 'sell_up':
        entry_price = entry_prices.sell_up
        exit_price = exit_prices.buy_up
        entry_route = entry_prices.sell_up_route
        exit_route = exit_prices.buy_up_route
    elif signal.side == 'buy_down':
        entry_price = entry_prices.buy_down
        exit_price = exit_prices.sell_down
        entry_route = entry_prices.buy_down_route
        exit_route = exit_prices.sell_down_route
    elif signal.side == 'sell_down':
        entry_price = entry_prices.sell_down
        exit_price = exit_prices.buy_down
        entry_route = entry_prices.sell_down_route
        exit_route = exit_prices.buy_down_route
    else:
        return None
    
    # Handle NaN prices
    if pd.isna(entry_price) or pd.isna(exit_price):
        return None
    
    # Compute PnL
    if signal.side.startswith('buy'):
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price
    
    return {
        'market_id': signal.market_id,
        'signal_entry_t': signal.entry_t,
        'signal_exit_t': signal.exit_t,
        'actual_entry_t': actual_entry_t,
        'actual_exit_t': actual_exit_t,
        'side': signal.side,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'entry_route': entry_route,
        'exit_route': exit_route,
        'pnl': pnl,
        'latency_applied': latency,
        'tau_at_entry': 900 - actual_entry_t,
        'reason': signal.reason,
    }


def run_strategy_latency_analysis(
    df: pd.DataFrame,
    strategy: Strategy,
    latencies: List[int] = None
) -> Dict[str, Any]:
    """
    Run latency cliff analysis for a specific strategy.
    
    This is the primary function for latency analysis. It uses the SAME
    strategy and signal generation as the backtest engine for consistency.
    
    Args:
        df: Full DataFrame with all markets
        strategy: Strategy object to analyze
        latencies: List of latency values to test
        
    Returns:
        Dictionary with:
        - summary_df: PnL at each latency
        - trades_df: All individual trades
        - per_market_df: Per-market PnL at each latency
        - stats_df: Clustered statistics
        - strategy_name: Name of strategy analyzed
        - strategy_params: Strategy parameters
    """
    if latencies is None:
        latencies = [0, 1, 2, 3, 5, 10, 15, 20, 30]
    
    strategy_name = strategy.name
    strategy_params = strategy.get_params()
    
    # Generate all signals across all markets
    all_signals = []
    market_ids = df['market_id'].unique().tolist()
    
    for market_id in market_ids:
        market_df = df[df['market_id'] == market_id]
        signals = strategy.generate_signals(market_df)
        all_signals.extend(signals)
    
    print(f"Strategy: {strategy_name}")
    print(f"Generated {len(all_signals)} signals across {len(market_ids)} markets")
    
    if not all_signals:
        return {
            'error': 'No signals generated',
            'strategy_name': strategy_name,
            'strategy_params': strategy_params,
        }
    
    # Run at each latency level
    all_trades = []
    summary_results = []
    
    for latency in latencies:
        trades_at_latency = []
        
        for signal in all_signals:
            market_df = df[df['market_id'] == signal.market_id]
            trade = execute_signal_at_latency(market_df, signal, latency)
            
            if trade is not None:
                trades_at_latency.append(trade)
                all_trades.append(trade)
        
        # Compute summary for this latency
        if trades_at_latency:
            pnls = [t['pnl'] for t in trades_at_latency]
            conversion_count = sum(
                1 for t in trades_at_latency 
                if t['entry_route'] == 'conversion' or t['exit_route'] == 'conversion'
            )
            
            summary_results.append(LatencyResult(
                latency=latency,
                strategy_name=strategy_name,
                total_pnl=sum(pnls),
                n_trades=len(pnls),
                avg_pnl=np.mean(pnls),
                hit_rate=sum(1 for p in pnls if p > 0) / len(pnls),
                avg_entry_price=np.mean([t['entry_price'] for t in trades_at_latency]),
                avg_exit_price=np.mean([t['exit_price'] for t in trades_at_latency]),
                conversion_rate=conversion_count / len(trades_at_latency) * 100,
            ))
        else:
            summary_results.append(LatencyResult(
                latency=latency,
                strategy_name=strategy_name,
                total_pnl=0,
                n_trades=0,
                avg_pnl=0,
                hit_rate=0,
                avg_entry_price=0,
                avg_exit_price=0,
                conversion_rate=0,
            ))
    
    # Convert to DataFrames
    summary_df = pd.DataFrame([{
        'latency': r.latency,
        'strategy': r.strategy_name,
        'total_pnl': r.total_pnl,
        'n_trades': r.n_trades,
        'avg_pnl': r.avg_pnl,
        'hit_rate': r.hit_rate,
        'avg_entry_price': r.avg_entry_price,
        'avg_exit_price': r.avg_exit_price,
        'conversion_rate': r.conversion_rate,
    } for r in summary_results])
    
    trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    
    # Compute per-market PnL at each latency
    per_market_results = []
    for market_id in market_ids:
        market_trades = [t for t in all_trades if t['market_id'] == market_id]
        for latency in latencies:
            lat_trades = [t for t in market_trades if t['latency_applied'] == latency]
            per_market_results.append({
                'market_id': market_id,
                'latency': latency,
                'pnl': sum(t['pnl'] for t in lat_trades),
                'n_trades': len(lat_trades),
            })
    
    per_market_df = pd.DataFrame(per_market_results)
    
    # Compute clustered statistics
    stats_df = compute_latency_statistics(per_market_df)
    
    # Find cliff point
    cliff_latency = find_cliff_point(summary_df)
    
    return {
        'summary_df': summary_df,
        'trades_df': trades_df,
        'per_market_df': per_market_df,
        'stats_df': stats_df,
        'cliff_latency': cliff_latency,
        'n_signals': len(all_signals),
        'n_markets': len(market_ids),
        'strategy_name': strategy_name,
        'strategy_params': strategy_params,
    }


def find_cliff_point(summary_df: pd.DataFrame, threshold: float = 0) -> int:
    """
    Find the latency at which edge disappears.
    
    Args:
        summary_df: Output from run_strategy_latency_analysis
        threshold: PnL threshold below which edge is considered "gone"
        
    Returns:
        Latency at which avg_pnl first drops below threshold (or max latency)
    """
    # Check if there's any positive PnL at 0 latency
    zero_lat = summary_df[summary_df['latency'] == 0]
    if zero_lat.empty or zero_lat['avg_pnl'].iloc[0] <= threshold:
        return 0  # No edge even at 0 latency
    
    for _, row in summary_df.iterrows():
        if row['avg_pnl'] <= threshold:
            return int(row['latency'])
    
    return int(summary_df['latency'].max())


def compute_latency_statistics(per_market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute proper clustered statistics at each latency.
    
    Uses per-market PnL (not per-trade) for proper standard errors.
    """
    if per_market_df.empty:
        return pd.DataFrame()
    
    results = []
    
    for latency in sorted(per_market_df['latency'].unique()):
        lat_data = per_market_df[per_market_df['latency'] == latency]
        pnls = lat_data['pnl'].values
        n = len(pnls)
        
        if n == 0:
            continue
        
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls, ddof=1) if n > 1 else 0
        se = std_pnl / np.sqrt(n) if n > 0 else 0
        t_stat = mean_pnl / se if se > 0 else 0
        
        results.append({
            'latency': latency,
            'mean_pnl': mean_pnl,
            'std_pnl': std_pnl,
            'se': se,
            't_stat': t_stat,
            'n_markets': n,
            'hit_rate': (pnls > 0).mean(),
            'total_pnl': pnls.sum(),
            'worst_market': pnls.min() if n > 0 else 0,
            'best_market': pnls.max() if n > 0 else 0,
        })
    
    return pd.DataFrame(results)


def print_latency_report(results: Dict[str, Any]):
    """Print formatted latency analysis report."""
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    print("\n" + "="*80)
    print(f"LATENCY CLIFF ANALYSIS: {results['strategy_name']}")
    print("="*80)
    
    print(f"\nStrategy Parameters:")
    for k, v in results['strategy_params'].items():
        print(f"  {k}: {v}")
    
    print(f"\nSignals: {results['n_signals']} across {results['n_markets']} markets")
    
    # Summary table
    summary = results['summary_df']
    print("\nPnL by Execution Latency:")
    print("-" * 80)
    print(f"{'Latency':>10} {'Total PnL':>12} {'Avg PnL':>10} {'Hit Rate':>10} {'N Trades':>10} {'Conv %':>8}")
    print("-" * 80)
    for _, row in summary.iterrows():
        print(f"{row['latency']:>10}s {row['total_pnl']:>12.4f} {row['avg_pnl']:>10.4f} "
              f"{row['hit_rate']*100:>9.1f}% {row['n_trades']:>10} {row['conversion_rate']:>7.1f}%")
    
    # Clustered stats
    stats = results['stats_df']
    if not stats.empty:
        print("\nPer-Market Clustered Statistics:")
        print("-" * 80)
        print(f"{'Latency':>10} {'Mean PnL':>12} {'Std':>10} {'t-stat':>10} {'Hit Rate':>10} {'Worst':>10}")
        print("-" * 80)
        for _, row in stats.iterrows():
            print(f"{row['latency']:>10}s {row['mean_pnl']:>12.4f} {row['std_pnl']:>10.4f} "
                  f"{row['t_stat']:>10.2f} {row['hit_rate']*100:>9.1f}% {row['worst_market']:>10.4f}")
    
    print(f"\nLatency Cliff Point: {results['cliff_latency']}s")
    print("(First latency where avg PnL drops to 0 or below)")


# ==============================================================================
# MAKER LATENCY CLIFF ANALYSIS
# ==============================================================================

@dataclass
class MakerLatencyResult:
    """Results for a maker strategy at specific latency level."""
    place_latency_ms: int
    cancel_latency_ms: int
    total_pnl: float
    mean_pnl_per_market: float
    t_stat: float
    hit_rate: float
    n_fills: int
    fill_rate: float
    spread_captured: float
    adverse_selection: float
    

def run_maker_latency_sweep(
    df: pd.DataFrame,
    strategy: SpreadCaptureStrategy,
    place_latencies_ms: List[int] = None,
    cancel_latencies_ms: List[int] = None,
    touch_trade_rate: float = 0.10,
    volume_markets_only: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run latency cliff analysis for maker strategies.
    
    Sweeps place_latency_ms and cancel_latency_ms to determine:
    1. At what latency does the maker edge disappear?
    2. How sensitive is the strategy to placement vs cancellation latency?
    
    Args:
        df: Full DataFrame with all markets
        strategy: SpreadCaptureStrategy to analyze
        place_latencies_ms: List of order placement latencies (ms)
        cancel_latencies_ms: List of order cancellation latencies (ms)
        touch_trade_rate: Fill model parameter
        volume_markets_only: Only use markets with size data
        verbose: Print progress
        
    Returns:
        Dictionary with results, DataFrames, and cliff analysis
    """
    if place_latencies_ms is None:
        place_latencies_ms = [0, 25, 50, 100, 200, 300, 500, 750, 1000]
    
    if cancel_latencies_ms is None:
        cancel_latencies_ms = [0, 25, 50, 100, 200]
    
    results = []
    total = len(place_latencies_ms) * len(cancel_latencies_ms)
    count = 0
    
    for place_lat in place_latencies_ms:
        for cancel_lat in cancel_latencies_ms:
            count += 1
            
            config = MakerExecutionConfig(
                place_latency_ms=place_lat,
                cancel_latency_ms=cancel_lat,
                fill_model=FillModel.TOUCH_SIZE_PROXY,
                touch_trade_rate_per_second=touch_trade_rate,
            )
            
            result = run_maker_backtest(
                df, strategy, config,
                verbose=False,
                volume_markets_only=volume_markets_only
            )
            
            metrics = result.get('metrics', {})
            
            results.append(MakerLatencyResult(
                place_latency_ms=place_lat,
                cancel_latency_ms=cancel_lat,
                total_pnl=metrics.get('total_pnl', 0),
                mean_pnl_per_market=metrics.get('mean_pnl_per_market', 0),
                t_stat=metrics.get('t_stat', 0),
                hit_rate=metrics.get('hit_rate_per_market', 0),
                n_fills=metrics.get('n_fills', 0),
                fill_rate=metrics.get('fill_rate', 0),
                spread_captured=metrics.get('spread_captured_total', 0),
                adverse_selection=metrics.get('adverse_selection_total', 0),
            ))
            
            if verbose and count % 5 == 0:
                print(f"Progress: {count}/{total}")
    
    # Convert to DataFrame
    summary_df = pd.DataFrame([{
        'place_latency_ms': r.place_latency_ms,
        'cancel_latency_ms': r.cancel_latency_ms,
        'total_pnl': r.total_pnl,
        'mean_pnl_per_market': r.mean_pnl_per_market,
        't_stat': r.t_stat,
        'hit_rate': r.hit_rate,
        'n_fills': r.n_fills,
        'fill_rate': r.fill_rate,
        'spread_captured': r.spread_captured,
        'adverse_selection': r.adverse_selection,
    } for r in results])
    
    # Find cliff point (first latency where PnL goes negative)
    # Aggregate by place_latency (averaging across cancel latencies)
    place_lat_summary = summary_df.groupby('place_latency_ms').agg({
        'total_pnl': 'mean',
        't_stat': 'mean',
        'n_fills': 'mean',
        'fill_rate': 'mean',
    }).reset_index()
    
    cliff_place_latency = find_maker_cliff_point(place_lat_summary, 'total_pnl')
    
    # Also find cancel latency cliff
    cancel_lat_summary = summary_df.groupby('cancel_latency_ms').agg({
        'total_pnl': 'mean',
        't_stat': 'mean',
    }).reset_index()
    
    cliff_cancel_latency = find_maker_cliff_point(cancel_lat_summary, 'total_pnl')
    
    return {
        'summary_df': summary_df,
        'place_latency_summary': place_lat_summary,
        'cancel_latency_summary': cancel_lat_summary,
        'cliff_place_latency_ms': cliff_place_latency,
        'cliff_cancel_latency_ms': cliff_cancel_latency,
        'strategy_name': strategy.name,
        'strategy_params': strategy.get_params(),
        'touch_trade_rate': touch_trade_rate,
    }


def find_maker_cliff_point(
    summary_df: pd.DataFrame, 
    pnl_col: str = 'total_pnl'
) -> int:
    """
    Find the latency at which maker edge disappears.
    
    Args:
        summary_df: DataFrame with latency and PnL columns
        pnl_col: Column name for PnL
        
    Returns:
        First latency where PnL drops below 0
    """
    lat_col = [c for c in summary_df.columns if 'latency' in c.lower()][0]
    
    # Sort by latency
    df = summary_df.sort_values(lat_col)
    
    # Check if positive at 0/min latency
    min_lat_pnl = df[df[lat_col] == df[lat_col].min()][pnl_col].iloc[0]
    if min_lat_pnl <= 0:
        return int(df[lat_col].min())
    
    # Find first latency where PnL drops below 0
    for _, row in df.iterrows():
        if row[pnl_col] <= 0:
            return int(row[lat_col])
    
    # Never drops below 0
    return int(df[lat_col].max())


def print_maker_latency_report(results: Dict[str, Any]):
    """Print formatted maker latency cliff report."""
    print("\n" + "="*80)
    print(f"MAKER LATENCY CLIFF ANALYSIS: {results['strategy_name']}")
    print("="*80)
    
    print(f"\nStrategy Parameters:")
    for k, v in results['strategy_params'].items():
        print(f"  {k}: {v}")
    
    print(f"\nFill Model: TOUCH_SIZE_PROXY (rate={results['touch_trade_rate']})")
    
    # Place latency summary
    print("\n--- PnL by Order Placement Latency ---")
    print("-" * 60)
    print(f"{'Place Lat':>12} {'Total PnL':>12} {'t-stat':>10} {'Fill Rate':>12} {'N Fills':>10}")
    print("-" * 60)
    for _, row in results['place_latency_summary'].iterrows():
        print(f"{row['place_latency_ms']:>10}ms {row['total_pnl']:>12.4f} "
              f"{row['t_stat']:>10.2f} {row['fill_rate']*100:>11.2f}% {row['n_fills']:>10.0f}")
    
    # Cancel latency summary
    print("\n--- PnL by Order Cancel Latency ---")
    print("-" * 60)
    print(f"{'Cancel Lat':>12} {'Total PnL':>12} {'t-stat':>10}")
    print("-" * 60)
    for _, row in results['cancel_latency_summary'].iterrows():
        print(f"{row['cancel_latency_ms']:>10}ms {row['total_pnl']:>12.4f} "
              f"{row['t_stat']:>10.2f}")
    
    print(f"\n--- Cliff Points ---")
    print(f"  Place latency cliff: {results['cliff_place_latency_ms']}ms")
    print(f"  Cancel latency cliff: {results['cliff_cancel_latency_ms']}ms")
    print("  (First latency where avg PnL drops to 0 or below)")
    
    # Full matrix (top 10 by PnL)
    summary = results['summary_df']
    print("\n--- Top 10 Latency Combinations by PnL ---")
    print("-" * 80)
    top10 = summary.nlargest(10, 'total_pnl')
    print(f"{'Place':>8} {'Cancel':>8} {'Total PnL':>12} {'t-stat':>10} {'Fills':>8} {'Fill%':>8}")
    print("-" * 80)
    for _, row in top10.iterrows():
        print(f"{row['place_latency_ms']:>6}ms {row['cancel_latency_ms']:>6}ms "
              f"{row['total_pnl']:>12.4f} {row['t_stat']:>10.2f} "
              f"{row['n_fills']:>8.0f} {row['fill_rate']*100:>7.2f}%")


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from scripts.backtest.data_loader import load_eth_markets, add_derived_columns
    from scripts.backtest.strategies import StrikeCrossStrategy
    
    print("Loading ETH markets...")
    df, market_info = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    # Use the same strategy as in the backtest
    strategy = StrikeCrossStrategy(tau_max=600, hold_to_expiry=True)
    
    print("\nRunning latency cliff analysis...")
    results = run_strategy_latency_analysis(df, strategy)
    
    print_latency_report(results)
