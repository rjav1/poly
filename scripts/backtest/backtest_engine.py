"""
Backtest Engine

Runs strategies, simulates trades, and computes proper per-market clustered metrics.

This module supports two modes:
1. Taker strategies: Traditional signal-based execution using execute_signal()
2. Maker strategies: Quote-based execution using run_maker_backtest() with FillEngine
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from .execution_model import get_effective_prices, ExecutionConfig, FillResult
    from .strategies import Strategy, Signal, SpreadCaptureStrategy
    from .maker_execution import FillEngine, MakerExecutionConfig, FillModel, MakerOrder, FillEvent, Inventory
except ImportError:
    from execution_model import get_effective_prices, ExecutionConfig, FillResult
    from strategies import Strategy, Signal, SpreadCaptureStrategy
    from maker_execution import FillEngine, MakerExecutionConfig, FillModel, MakerOrder, FillEvent, Inventory


@dataclass
class Trade:
    """A completed trade."""
    market_id: str
    entry_t: int
    exit_t: int
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    entry_route: str
    exit_route: str
    signal_reason: str
    latency_applied: int
    tau_at_entry: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'market_id': self.market_id,
            'entry_t': self.entry_t,
            'exit_t': self.exit_t,
            'side': self.side,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'entry_route': self.entry_route,
            'exit_route': self.exit_route,
            'signal_reason': self.signal_reason,
            'latency_applied': self.latency_applied,
            'tau_at_entry': self.tau_at_entry,
        }


@dataclass
class BacktestMetrics:
    """Metrics computed with proper clustering."""
    n_markets: int
    n_trades: int
    total_pnl: float
    mean_pnl_per_market: float
    std_pnl_per_market: float
    se_pnl_per_market: float
    t_stat: float
    hit_rate_per_market: float  # % of markets with positive PnL
    hit_rate_per_trade: float   # % of trades with positive PnL
    worst_market_pnl: float
    best_market_pnl: float
    median_market_pnl: float
    avg_trades_per_market: float
    conversion_entry_rate: float
    conversion_exit_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_markets': self.n_markets,
            'n_trades': self.n_trades,
            'total_pnl': self.total_pnl,
            'mean_pnl_per_market': self.mean_pnl_per_market,
            'std_pnl_per_market': self.std_pnl_per_market,
            'se_pnl_per_market': self.se_pnl_per_market,
            't_stat': self.t_stat,
            'hit_rate_per_market': self.hit_rate_per_market,
            'hit_rate_per_trade': self.hit_rate_per_trade,
            'worst_market_pnl': self.worst_market_pnl,
            'best_market_pnl': self.best_market_pnl,
            'median_market_pnl': self.median_market_pnl,
            'avg_trades_per_market': self.avg_trades_per_market,
            'conversion_entry_rate': self.conversion_entry_rate,
            'conversion_exit_rate': self.conversion_exit_rate,
        }


@dataclass
class MakerBacktestMetrics:
    """
    Metrics for maker/spread capture strategies.
    
    These metrics are specific to maker strategies and include:
    - PnL decomposition (spread captured, adverse selection, inventory carry)
    - Fill statistics (fill rate, time to fill, etc.)
    - Order management stats (cancels, expirations)
    """
    # Basic metrics (same as BacktestMetrics)
    n_markets: int
    n_fills: int
    total_pnl: float
    mean_pnl_per_market: float
    std_pnl_per_market: float
    se_pnl_per_market: float
    t_stat: float
    hit_rate_per_market: float
    worst_market_pnl: float
    best_market_pnl: float
    median_market_pnl: float
    
    # PnL decomposition (maker-specific)
    spread_captured_total: float = 0.0  # Total spread captured
    adverse_selection_total: float = 0.0  # Total adverse selection cost
    inventory_carry_total: float = 0.0  # Total inventory carry PnL
    realized_pnl_total: float = 0.0  # Realized PnL from closed trades
    
    # Fill statistics
    fill_rate: float = 0.0  # % of placed orders that filled
    avg_time_to_fill: float = 0.0  # Average seconds from placement to fill
    cancel_to_fill_ratio: float = 0.0  # Cancels / fills
    orders_placed_total: int = 0
    orders_filled_total: int = 0
    orders_cancelled_total: int = 0
    orders_expired_total: int = 0
    total_fill_volume: float = 0.0
    
    # Quote statistics
    avg_quote_update_rate: float = 0.0  # How often quotes updated
    quote_seconds_total: int = 0  # Total seconds we were quoting
    
    # Per-market breakdown (stored separately)
    market_pnls: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_markets': self.n_markets,
            'n_fills': self.n_fills,
            'total_pnl': self.total_pnl,
            'mean_pnl_per_market': self.mean_pnl_per_market,
            'std_pnl_per_market': self.std_pnl_per_market,
            'se_pnl_per_market': self.se_pnl_per_market,
            't_stat': self.t_stat,
            'hit_rate_per_market': self.hit_rate_per_market,
            'worst_market_pnl': self.worst_market_pnl,
            'best_market_pnl': self.best_market_pnl,
            'median_market_pnl': self.median_market_pnl,
            'spread_captured_total': self.spread_captured_total,
            'adverse_selection_total': self.adverse_selection_total,
            'inventory_carry_total': self.inventory_carry_total,
            'realized_pnl_total': self.realized_pnl_total,
            'fill_rate': self.fill_rate,
            'avg_time_to_fill': self.avg_time_to_fill,
            'cancel_to_fill_ratio': self.cancel_to_fill_ratio,
            'orders_placed_total': self.orders_placed_total,
            'orders_filled_total': self.orders_filled_total,
            'orders_cancelled_total': self.orders_cancelled_total,
            'orders_expired_total': self.orders_expired_total,
            'total_fill_volume': self.total_fill_volume,
        }


def execute_signal(
    market_df: pd.DataFrame,
    signal: Signal,
    config: ExecutionConfig
) -> Optional[Trade]:
    """
    Execute a signal and return the resulting trade.
    
    Args:
        market_df: DataFrame for single market
        signal: Signal to execute
        config: Execution configuration
        
    Returns:
        Trade if successful, None if couldn't execute
    """
    # Apply latency to entry
    actual_entry_t = signal.entry_t + int(config.total_latency())
    actual_exit_t = signal.exit_t + int(config.exec_latency_s)  # Exit also has latency
    
    # Get entry row
    entry_row = market_df[market_df['t'] == actual_entry_t]
    if entry_row.empty:
        return None
    
    # Get exit row
    exit_row = market_df[market_df['t'] == actual_exit_t]
    if exit_row.empty:
        # Fall back to last available row
        exit_row = market_df[market_df['t'] <= actual_exit_t].iloc[-1:]
        if exit_row.empty:
            return None
    
    # Get effective prices
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
    
    # Handle NaN
    if pd.isna(entry_price):
        return None
    
    # Check if this is an expiry trade (exit_t at or near market end)
    max_t = market_df['t'].max()
    is_expiry = (actual_exit_t >= max_t - 1)  # Allow 1 second tolerance
    
    # For expiry trades, use settlement outcome (Y) instead of exit_price
    if is_expiry:
        Y = entry_row.iloc[0].get('Y', np.nan)
        if pd.isna(Y):
            # Try to get Y from market_df
            Y = market_df['Y'].iloc[0] if 'Y' in market_df.columns else np.nan
        
        if not pd.isna(Y):
            # Settlement payout: buy_up pays $1 if Y=1, else $0
            #                   buy_down pays $1 if Y=0, else $0
            if signal.side == 'buy_up':
                settlement_payout = float(Y)  # 1.0 if Y=1, 0.0 if Y=0
                exit_price = settlement_payout
            elif signal.side == 'buy_down':
                settlement_payout = 1.0 - float(Y)  # 1.0 if Y=0, 0.0 if Y=1
                exit_price = settlement_payout
            elif signal.side == 'sell_up':
                settlement_payout = float(Y)  # 1.0 if Y=1, 0.0 if Y=0
                exit_price = settlement_payout
            elif signal.side == 'sell_down':
                settlement_payout = 1.0 - float(Y)  # 1.0 if Y=0, 0.0 if Y=1
                exit_price = settlement_payout
            else:
                exit_price = np.nan
        else:
            # Y not available, fall back to exit_price from orderbook
            if pd.isna(exit_price):
                return None
    else:
        # Early exit: use orderbook exit_price
        if pd.isna(exit_price):
            return None
    
    # Compute PnL
    if signal.side.startswith('buy'):
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price
    
    return Trade(
        market_id=signal.market_id,
        entry_t=actual_entry_t,
        exit_t=actual_exit_t,
        side=signal.side,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=pnl,
        entry_route=entry_route,
        exit_route=exit_route,
        signal_reason=signal.reason,
        latency_applied=int(config.total_latency()),
        tau_at_entry=900 - actual_entry_t,
    )


def compute_metrics(
    trades: List[Trade],
    all_market_ids: List[str]
) -> BacktestMetrics:
    """
    Compute proper per-market clustered metrics.
    
    Args:
        trades: List of executed trades
        all_market_ids: All market IDs (to include markets with 0 trades)
        
    Returns:
        BacktestMetrics
    """
    if not trades:
        return BacktestMetrics(
            n_markets=len(all_market_ids),
            n_trades=0,
            total_pnl=0,
            mean_pnl_per_market=0,
            std_pnl_per_market=0,
            se_pnl_per_market=0,
            t_stat=0,
            hit_rate_per_market=0,
            hit_rate_per_trade=0,
            worst_market_pnl=0,
            best_market_pnl=0,
            median_market_pnl=0,
            avg_trades_per_market=0,
            conversion_entry_rate=0,
            conversion_exit_rate=0,
        )
    
    # Group trades by market
    market_pnls = {}
    market_trades = {}
    for market_id in all_market_ids:
        market_pnls[market_id] = 0
        market_trades[market_id] = []
    
    for trade in trades:
        market_pnls[trade.market_id] = market_pnls.get(trade.market_id, 0) + trade.pnl
        if trade.market_id not in market_trades:
            market_trades[trade.market_id] = []
        market_trades[trade.market_id].append(trade)
    
    pnls = list(market_pnls.values())
    n_markets = len(pnls)
    
    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls, ddof=1) if n_markets > 1 else 0
    se_pnl = std_pnl / np.sqrt(n_markets) if n_markets > 0 else 0
    t_stat = mean_pnl / se_pnl if se_pnl > 0 else 0
    
    # Conversion rates
    conv_entry = sum(1 for t in trades if t.entry_route == 'conversion') / len(trades) * 100
    conv_exit = sum(1 for t in trades if t.exit_route == 'conversion') / len(trades) * 100
    
    return BacktestMetrics(
        n_markets=n_markets,
        n_trades=len(trades),
        total_pnl=sum(pnls),
        mean_pnl_per_market=mean_pnl,
        std_pnl_per_market=std_pnl,
        se_pnl_per_market=se_pnl,
        t_stat=t_stat,
        hit_rate_per_market=sum(1 for p in pnls if p > 0) / n_markets if n_markets > 0 else 0,
        hit_rate_per_trade=sum(1 for t in trades if t.pnl > 0) / len(trades) if trades else 0,
        worst_market_pnl=min(pnls) if pnls else 0,
        best_market_pnl=max(pnls) if pnls else 0,
        median_market_pnl=np.median(pnls) if pnls else 0,
        avg_trades_per_market=len(trades) / n_markets if n_markets > 0 else 0,
        conversion_entry_rate=conv_entry,
        conversion_exit_rate=conv_exit,
    )


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    config: ExecutionConfig = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run backtest for a strategy.
    
    Args:
        df: Full DataFrame with all markets
        strategy: Strategy to test
        config: Execution configuration (default: no latency)
        verbose: Print progress
        
    Returns:
        Dictionary with trades, metrics, and per-market breakdown
    """
    if config is None:
        config = ExecutionConfig()
    
    all_trades = []
    all_signals = []
    market_ids = df['market_id'].unique().tolist()
    
    for market_id in market_ids:
        market_df = df[df['market_id'] == market_id]
        
        # Generate signals
        signals = strategy.generate_signals(market_df)
        all_signals.extend(signals)
        
        # Execute signals
        for signal in signals:
            trade = execute_signal(market_df, signal, config)
            if trade is not None:
                all_trades.append(trade)
    
    # Compute metrics
    metrics = compute_metrics(all_trades, market_ids)
    
    # Per-market breakdown
    market_breakdown = {}
    for market_id in market_ids:
        market_trades = [t for t in all_trades if t.market_id == market_id]
        market_breakdown[market_id] = {
            'n_trades': len(market_trades),
            'pnl': sum(t.pnl for t in market_trades),
            'trades': [t.to_dict() for t in market_trades],
        }
    
    if verbose:
        print(f"\n{strategy.name}")
        print(f"  Signals: {len(all_signals)}, Trades: {len(all_trades)}")
        print(f"  Total PnL: ${metrics.total_pnl:.4f}")
        print(f"  Mean PnL/market: ${metrics.mean_pnl_per_market:.4f} (t={metrics.t_stat:.2f})")
        print(f"  Hit rate: {metrics.hit_rate_per_market*100:.1f}% (markets), {metrics.hit_rate_per_trade*100:.1f}% (trades)")
    
    return {
        'strategy': strategy.name,
        'params': strategy.get_params(),
        'config': {
            'signal_latency_s': config.signal_latency_s,
            'exec_latency_s': config.exec_latency_s,
        },
        'n_signals': len(all_signals),
        'trades': [t.to_dict() for t in all_trades],
        'metrics': metrics.to_dict(),
        'market_breakdown': market_breakdown,
    }


def run_latency_sweep(
    df: pd.DataFrame,
    strategy: Strategy,
    latencies: List[int] = None
) -> pd.DataFrame:
    """
    Run strategy at multiple latency levels.
    
    Args:
        df: Full DataFrame
        strategy: Strategy to test
        latencies: List of total latencies to test
        
    Returns:
        DataFrame with metrics at each latency
    """
    if latencies is None:
        latencies = [0, 1, 2, 3, 5, 10, 15, 20, 30]
    
    results = []
    
    for latency in latencies:
        config = ExecutionConfig(
            signal_latency_s=latency // 2,  # Split between signal and exec
            exec_latency_s=latency - latency // 2
        )
        
        result = run_backtest(df, strategy, config, verbose=False)
        
        results.append({
            'latency': latency,
            **result['metrics']
        })
    
    return pd.DataFrame(results)


def compare_strategies(
    df: pd.DataFrame,
    strategies: List[Strategy],
    config: ExecutionConfig = None
) -> pd.DataFrame:
    """
    Compare multiple strategies.
    
    Args:
        df: Full DataFrame
        strategies: List of strategies to compare
        config: Execution configuration
        
    Returns:
        DataFrame with metrics for each strategy
    """
    results = []
    
    for strategy in strategies:
        result = run_backtest(df, strategy, config, verbose=False)
        results.append({
            'strategy': strategy.name,
            **result['metrics']
        })
    
    return pd.DataFrame(results)


# ==============================================================================
# MAKER BACKTEST ENGINE
# ==============================================================================

def run_maker_backtest(
    df: pd.DataFrame,
    strategy: SpreadCaptureStrategy,
    config: MakerExecutionConfig = None,
    verbose: bool = False,
    volume_markets_only: bool = True,
) -> Dict[str, Any]:
    """
    Run maker/spread capture strategy backtest.
    
    Unlike taker strategies, maker strategies:
    1. Place passive quotes rather than executing immediately
    2. Get filled probabilistically based on market activity
    3. Manage inventory continuously
    4. Must flatten before market expiry
    
    This function processes each market tick-by-tick using the FillEngine.
    
    Args:
        df: Full DataFrame with all markets
        strategy: SpreadCaptureStrategy instance
        config: MakerExecutionConfig (default if None)
        verbose: Print progress
        volume_markets_only: Only use markets with size data (recommended)
        
    Returns:
        Dictionary with results, metrics, and per-market breakdown
    """
    if config is None:
        config = MakerExecutionConfig()
    
    # Filter to volume markets if requested
    if volume_markets_only:
        # Volume market prefixes (12 ETH markets with size data)
        volume_prefixes = [
            '20260106_1630', '20260106_1645', '20260106_1700', '20260106_1715',
            '20260106_1730', '20260106_1745', '20260106_1800', '20260106_1815',
            '20260106_1830', '20260106_1845', '20260106_1900', '20260106_1915'
        ]
        # Filter markets that have size data
        market_ids = df['market_id'].unique()
        valid_markets = [m for m in market_ids if any(m.startswith(p) for p in volume_prefixes)]
        
        # Check if any volume markets exist
        if not valid_markets:
            # Fall back to all markets but warn
            if verbose:
                print("Warning: No volume markets found, using all markets (size data may be missing)")
            valid_markets = list(market_ids)
        
        df = df[df['market_id'].isin(valid_markets)]
    
    market_ids = df['market_id'].unique()
    
    if verbose:
        print(f"Running maker backtest: {strategy.name}")
        print(f"  Config: {config.describe()}")
        print(f"  Markets: {len(market_ids)}")
    
    # Results storage
    all_fills: List[FillEvent] = []
    market_results: Dict[str, Dict] = {}
    
    # Aggregated stats
    total_spread_captured = 0.0
    total_adverse_selection = 0.0
    total_inventory_carry = 0.0
    total_realized_pnl = 0.0
    total_orders_placed = 0
    total_orders_filled = 0
    total_orders_cancelled = 0
    total_orders_expired = 0
    total_fill_volume = 0.0
    total_time_to_fill = 0.0
    total_quote_seconds = 0
    
    # Process each market
    for market_id in market_ids:
        market_df = df[df['market_id'] == market_id].sort_values('t').reset_index(drop=True)
        
        if len(market_df) == 0:
            continue
        
        # Run maker backtest for this market
        market_result = _run_maker_backtest_single_market(
            market_df=market_df,
            market_id=market_id,
            strategy=strategy,
            config=config,
        )
        
        market_results[market_id] = market_result
        
        # Aggregate results
        all_fills.extend(market_result['fills'])
        total_spread_captured += market_result['spread_captured']
        total_adverse_selection += market_result['adverse_selection']
        total_inventory_carry += market_result['inventory_carry']
        total_realized_pnl += market_result['realized_pnl']
        total_orders_placed += market_result['orders_placed']
        total_orders_filled += market_result['orders_filled']
        total_orders_cancelled += market_result['orders_cancelled']
        total_orders_expired += market_result['orders_expired']
        total_fill_volume += market_result['fill_volume']
        total_time_to_fill += market_result['time_to_fill_sum']
        total_quote_seconds += market_result['quote_seconds']
    
    # Compute metrics with proper per-market clustering
    market_pnls = {mid: r['pnl'] for mid, r in market_results.items()}
    metrics = _compute_maker_metrics(
        market_pnls=market_pnls,
        n_fills=len(all_fills),
        spread_captured=total_spread_captured,
        adverse_selection=total_adverse_selection,
        inventory_carry=total_inventory_carry,
        realized_pnl=total_realized_pnl,
        orders_placed=total_orders_placed,
        orders_filled=total_orders_filled,
        orders_cancelled=total_orders_cancelled,
        orders_expired=total_orders_expired,
        fill_volume=total_fill_volume,
        time_to_fill_sum=total_time_to_fill,
        quote_seconds=total_quote_seconds,
    )
    
    if verbose:
        print(f"\n{strategy.name}")
        print(f"  Markets: {metrics.n_markets}, Fills: {metrics.n_fills}")
        print(f"  Total PnL: ${metrics.total_pnl:.4f}")
        print(f"  Mean PnL/market: ${metrics.mean_pnl_per_market:.4f} (t={metrics.t_stat:.2f})")
        print(f"  Hit rate: {metrics.hit_rate_per_market*100:.1f}% (markets)")
        print(f"  PnL Decomposition:")
        print(f"    Spread captured: ${metrics.spread_captured_total:.4f}")
        print(f"    Adverse selection: -${metrics.adverse_selection_total:.4f}")
        print(f"    Inventory carry: ${metrics.inventory_carry_total:.4f}")
        print(f"  Fill rate: {metrics.fill_rate*100:.1f}%")
        print(f"  Avg time to fill: {metrics.avg_time_to_fill:.1f}s")
    
    return {
        'strategy': strategy.name,
        'params': strategy.get_params(),
        'config': {
            'place_latency_ms': config.place_latency_ms,
            'cancel_latency_ms': config.cancel_latency_ms,
            'fill_model': config.fill_model.value,
        },
        'metrics': metrics.to_dict(),
        'market_results': {mid: _simplify_market_result(r) for mid, r in market_results.items()},
        'fills': [f.to_dict() for f in all_fills],
    }


def _run_maker_backtest_single_market(
    market_df: pd.DataFrame,
    market_id: str,
    strategy: SpreadCaptureStrategy,
    config: MakerExecutionConfig,
) -> Dict[str, Any]:
    """
    Run maker backtest for a single market.
    
    Processes tick-by-tick, managing quotes and fills.
    """
    engine = FillEngine(config)
    
    # Track active quotes (order_ids for current quotes)
    active_up_bid_id: Optional[str] = None
    active_up_ask_id: Optional[str] = None
    active_down_bid_id: Optional[str] = None
    active_down_ask_id: Optional[str] = None
    
    # Track for quote update rate calculation
    quote_updates = 0
    quote_seconds = 0
    time_to_fill_sum = 0.0
    
    # Keep last N rows for CL jump detection
    lookback_size = max(10, strategy.cl_lookback_seconds + 1)
    prev_rows = []
    
    max_t = int(market_df['t'].max())
    
    for idx, row in market_df.iterrows():
        t = int(row['t'])
        
        # Build prev_rows DataFrame
        prev_df = pd.DataFrame(prev_rows) if prev_rows else None
        
        # Get strategy decision
        quote_decision = strategy.should_quote(
            row=row,
            inventory_up=engine.inventory.up_position,
            inventory_down=engine.inventory.down_position,
            prev_rows=prev_df,
        )
        
        # Process tick in engine (updates market state, checks fills)
        fills = engine.process_tick(row, t)
        
        # Track time to fill
        for fill in fills:
            order = engine.orders[fill.order_id]
            if order.time_active is not None:
                time_to_fill_sum += (fill.fill_time - order.time_active)
        
        # Handle flatten
        if quote_decision['flatten']:
            # Cancel all active quotes
            for oid in [active_up_bid_id, active_up_ask_id, active_down_bid_id, active_down_ask_id]:
                if oid:
                    engine.cancel_order(oid, t)
            active_up_bid_id = active_up_ask_id = active_down_bid_id = active_down_ask_id = None
            
            # Generate flatten trades (aggressive exit)
            flatten_trades = strategy.get_flatten_trades(
                engine.inventory.up_position,
                engine.inventory.down_position,
                row,
            )
            
            # Execute flatten trades (taker style - instant fill at market)
            for trade in flatten_trades:
                if trade['price'] is not None and not pd.isna(trade['price']):
                    # Direct inventory update for taker flatten
                    side = 'BID' if trade['side'] == 'buy' else 'ASK'
                    engine.inventory.update_position(
                        trade['token'],
                        side,
                        trade['price'],
                        trade['size']
                    )
        else:
            # Update quotes based on strategy decision
            quote_updates += _update_quotes(
                engine=engine,
                t=t,
                quote_decision=quote_decision,
                active_up_bid_id=active_up_bid_id,
                active_up_ask_id=active_up_ask_id,
                active_down_bid_id=active_down_bid_id,
                active_down_ask_id=active_down_ask_id,
            )
            
            # Track if we're quoting
            if any([quote_decision['quote_up_bid'], quote_decision['quote_up_ask'],
                    quote_decision['quote_down_bid'], quote_decision['quote_down_ask']]):
                quote_seconds += 1
            
            # Update active order IDs
            active_up_bid_id, active_up_ask_id, active_down_bid_id, active_down_ask_id = \
                _get_current_quote_ids(engine, quote_decision)
        
        # Update prev_rows
        prev_rows.append(row.to_dict())
        if len(prev_rows) > lookback_size:
            prev_rows.pop(0)
    
    # Expire remaining orders at market end
    engine.expire_all_orders(max_t)
    
    # Get PnL decomposition
    pnl_decomp = engine.get_pnl_decomposition()
    stats = engine.get_stats()
    
    return {
        'market_id': market_id,
        'pnl': pnl_decomp['total_pnl'],
        'spread_captured': pnl_decomp['spread_captured'],
        'adverse_selection': pnl_decomp['adverse_selection'],
        'inventory_carry': pnl_decomp['inventory_carry'],
        'realized_pnl': pnl_decomp['realized_pnl'],
        'orders_placed': stats['orders_placed'],
        'orders_filled': stats['orders_filled'],
        'orders_cancelled': stats['orders_cancelled'],
        'orders_expired': stats['orders_expired'],
        'fill_volume': stats['total_fill_volume'],
        'time_to_fill_sum': time_to_fill_sum,
        'quote_seconds': quote_seconds,
        'quote_updates': quote_updates,
        'fills': engine.get_fills(),
        'final_inventory': engine.inventory.to_dict(),
    }


def _update_quotes(
    engine: FillEngine,
    t: int,
    quote_decision: Dict[str, Any],
    active_up_bid_id: Optional[str],
    active_up_ask_id: Optional[str],
    active_down_bid_id: Optional[str],
    active_down_ask_id: Optional[str],
) -> int:
    """
    Update quotes based on strategy decision.
    
    Returns number of quote updates (for tracking).
    """
    updates = 0
    size = quote_decision['quote_size']
    
    # UP BID
    target_up_bid = quote_decision['quote_up_bid']
    if target_up_bid is not None:
        # Check if we need to update
        if active_up_bid_id:
            order = engine.orders.get(active_up_bid_id)
            if order and order.status.value == 'active':
                if abs(order.price - target_up_bid) > 0.001:
                    # Price changed, cancel and re-quote
                    engine.cancel_order(active_up_bid_id, t)
                    engine.place_order('UP', 'BID', target_up_bid, size, t)
                    updates += 1
            else:
                # Order no longer active, place new
                engine.place_order('UP', 'BID', target_up_bid, size, t)
                updates += 1
        else:
            # No active quote, place new
            engine.place_order('UP', 'BID', target_up_bid, size, t)
            updates += 1
    else:
        # Should not quote, cancel if active
        if active_up_bid_id:
            engine.cancel_order(active_up_bid_id, t)
    
    # UP ASK
    target_up_ask = quote_decision['quote_up_ask']
    if target_up_ask is not None:
        if active_up_ask_id:
            order = engine.orders.get(active_up_ask_id)
            if order and order.status.value == 'active':
                if abs(order.price - target_up_ask) > 0.001:
                    engine.cancel_order(active_up_ask_id, t)
                    engine.place_order('UP', 'ASK', target_up_ask, size, t)
                    updates += 1
            else:
                engine.place_order('UP', 'ASK', target_up_ask, size, t)
                updates += 1
        else:
            engine.place_order('UP', 'ASK', target_up_ask, size, t)
            updates += 1
    else:
        if active_up_ask_id:
            engine.cancel_order(active_up_ask_id, t)
    
    # DOWN BID
    target_down_bid = quote_decision['quote_down_bid']
    if target_down_bid is not None:
        if active_down_bid_id:
            order = engine.orders.get(active_down_bid_id)
            if order and order.status.value == 'active':
                if abs(order.price - target_down_bid) > 0.001:
                    engine.cancel_order(active_down_bid_id, t)
                    engine.place_order('DOWN', 'BID', target_down_bid, size, t)
                    updates += 1
            else:
                engine.place_order('DOWN', 'BID', target_down_bid, size, t)
                updates += 1
        else:
            engine.place_order('DOWN', 'BID', target_down_bid, size, t)
            updates += 1
    else:
        if active_down_bid_id:
            engine.cancel_order(active_down_bid_id, t)
    
    # DOWN ASK
    target_down_ask = quote_decision['quote_down_ask']
    if target_down_ask is not None:
        if active_down_ask_id:
            order = engine.orders.get(active_down_ask_id)
            if order and order.status.value == 'active':
                if abs(order.price - target_down_ask) > 0.001:
                    engine.cancel_order(active_down_ask_id, t)
                    engine.place_order('DOWN', 'ASK', target_down_ask, size, t)
                    updates += 1
            else:
                engine.place_order('DOWN', 'ASK', target_down_ask, size, t)
                updates += 1
        else:
            engine.place_order('DOWN', 'ASK', target_down_ask, size, t)
            updates += 1
    else:
        if active_down_ask_id:
            engine.cancel_order(active_down_ask_id, t)
    
    return updates


def _get_current_quote_ids(
    engine: FillEngine,
    quote_decision: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Get current active quote order IDs."""
    up_bid_id = None
    up_ask_id = None
    down_bid_id = None
    down_ask_id = None
    
    for oid in engine.active_orders + engine.pending_orders:
        order = engine.orders.get(oid)
        if order:
            if order.token == 'UP' and order.side == 'BID':
                up_bid_id = oid
            elif order.token == 'UP' and order.side == 'ASK':
                up_ask_id = oid
            elif order.token == 'DOWN' and order.side == 'BID':
                down_bid_id = oid
            elif order.token == 'DOWN' and order.side == 'ASK':
                down_ask_id = oid
    
    return up_bid_id, up_ask_id, down_bid_id, down_ask_id


def _compute_maker_metrics(
    market_pnls: Dict[str, float],
    n_fills: int,
    spread_captured: float,
    adverse_selection: float,
    inventory_carry: float,
    realized_pnl: float,
    orders_placed: int,
    orders_filled: int,
    orders_cancelled: int,
    orders_expired: int,
    fill_volume: float,
    time_to_fill_sum: float,
    quote_seconds: int,
) -> MakerBacktestMetrics:
    """Compute maker-specific metrics with proper clustering."""
    pnls = list(market_pnls.values())
    n_markets = len(pnls)
    
    if n_markets == 0:
        return MakerBacktestMetrics(
            n_markets=0, n_fills=0, total_pnl=0, mean_pnl_per_market=0,
            std_pnl_per_market=0, se_pnl_per_market=0, t_stat=0,
            hit_rate_per_market=0, worst_market_pnl=0, best_market_pnl=0,
            median_market_pnl=0, market_pnls={},
        )
    
    total_pnl = sum(pnls)
    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls, ddof=1) if n_markets > 1 else 0.0
    se_pnl = std_pnl / np.sqrt(n_markets) if n_markets > 1 else 0.0
    t_stat = mean_pnl / se_pnl if se_pnl > 0 else 0.0
    
    hit_rate = sum(1 for p in pnls if p > 0) / n_markets
    
    fill_rate = orders_filled / orders_placed if orders_placed > 0 else 0.0
    avg_time_to_fill = time_to_fill_sum / orders_filled if orders_filled > 0 else 0.0
    cancel_to_fill_ratio = orders_cancelled / orders_filled if orders_filled > 0 else 0.0
    
    return MakerBacktestMetrics(
        n_markets=n_markets,
        n_fills=n_fills,
        total_pnl=total_pnl,
        mean_pnl_per_market=mean_pnl,
        std_pnl_per_market=std_pnl,
        se_pnl_per_market=se_pnl,
        t_stat=t_stat,
        hit_rate_per_market=hit_rate,
        worst_market_pnl=min(pnls),
        best_market_pnl=max(pnls),
        median_market_pnl=float(np.median(pnls)),
        spread_captured_total=spread_captured,
        adverse_selection_total=adverse_selection,
        inventory_carry_total=inventory_carry,
        realized_pnl_total=realized_pnl,
        fill_rate=fill_rate,
        avg_time_to_fill=avg_time_to_fill,
        cancel_to_fill_ratio=cancel_to_fill_ratio,
        orders_placed_total=orders_placed,
        orders_filled_total=orders_filled,
        orders_cancelled_total=orders_cancelled,
        orders_expired_total=orders_expired,
        total_fill_volume=fill_volume,
        quote_seconds_total=quote_seconds,
        market_pnls=market_pnls,
    )


def _simplify_market_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Simplify market result for JSON serialization."""
    return {
        'pnl': result['pnl'],
        'spread_captured': result['spread_captured'],
        'adverse_selection': result['adverse_selection'],
        'inventory_carry': result['inventory_carry'],
        'orders_placed': result['orders_placed'],
        'orders_filled': result['orders_filled'],
        'orders_cancelled': result['orders_cancelled'],
        'orders_expired': result['orders_expired'],
        'fill_volume': result['fill_volume'],
        'quote_seconds': result['quote_seconds'],
    }


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from scripts.backtest.data_loader import load_eth_markets, add_derived_columns
    from scripts.backtest.strategies import (
        LatencyCaptureStrategy, StrikeCrossStrategy, 
        MomentumStrategy, NearStrikeStrategy
    )
    
    print("Loading data...")
    df, market_info = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    # Test strategies
    strategies = [
        LatencyCaptureStrategy(threshold_bps=5, hold_seconds=15),
        LatencyCaptureStrategy(threshold_bps=3, hold_seconds=30),
        StrikeCrossStrategy(tau_max=300),
        StrikeCrossStrategy(tau_max=120),
        NearStrikeStrategy(near_strike_bps=20, min_move_bps=3),
        MomentumStrategy(lookback=5, min_total_move_bps=10),
    ]
    
    print("\n" + "="*80)
    print("STRATEGY COMPARISON (0 latency)")
    print("="*80)
    
    comparison = compare_strategies(df, strategies)
    
    print(f"\n{'Strategy':<50} {'Total PnL':>10} {'Mean/Mkt':>10} {'t-stat':>8} {'Hit%':>8} {'Trades':>8}")
    print("-" * 100)
    
    for _, row in comparison.iterrows():
        print(f"{row['strategy']:<50} {row['total_pnl']:>10.4f} {row['mean_pnl_per_market']:>10.4f} "
              f"{row['t_stat']:>8.2f} {row['hit_rate_per_market']*100:>7.1f}% {row['n_trades']:>8}")
    
    # Latency sweep for best strategy
    print("\n" + "="*80)
    print("LATENCY SWEEP - StrikeCross(tau_max=120)")
    print("="*80)
    
    best_strategy = StrikeCrossStrategy(tau_max=120)
    latency_df = run_latency_sweep(df, best_strategy)
    
    print(f"\n{'Latency':>10} {'Total PnL':>12} {'Mean/Mkt':>12} {'t-stat':>10} {'Hit%':>10}")
    print("-" * 60)
    for _, row in latency_df.iterrows():
        print(f"{row['latency']:>10}s {row['total_pnl']:>12.4f} {row['mean_pnl_per_market']:>12.4f} "
              f"{row['t_stat']:>10.2f} {row['hit_rate_per_market']*100:>9.1f}%")

