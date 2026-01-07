"""
Execution Model with Split/Redeem Conversion Routing

This module implements the correct execution model for Polymarket Up/Down markets,
including the Split/Redeem mechanism that successful traders use.

Polymarket mechanics:
- Complete set = 1 UP token + 1 DOWN token = $1.00
- Split: Pay $1 to get 1 UP + 1 DOWN
- Redeem: Exchange 1 UP + 1 DOWN for $1

This means the true best execution price is NOT just the orderbook price:
- Buy UP = min(up_ask, 1 - down_bid)  # Either buy UP direct OR sell DOWN and keep UP
- Sell UP = max(up_bid, 1 - down_ask)  # Either sell UP direct OR buy DOWN and redeem
"""

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class ExecutionConfig:
    """Configuration for trade execution simulation.
    
    TIMESTAMP SEMANTICS:
    When use_received_timestamps=True, we model realistic observation delays:
    - CL data is typically ~60-65s behind real-time when using UI scraping
    - This means if you align on source_timestamp, you're "gifting" yourself
      knowledge you wouldn't have had in real-time
    
    To validate edge is real, run backtests with:
    - use_received_timestamps=False: Optimistic (event time = observation time)
    - use_received_timestamps=True: Realistic (add CL observation delay)
    
    If edge disappears with realistic timestamps, it wasn't tradeable.
    """
    signal_latency_s: float = 0.0  # Additional latency to observe signal (seconds)
    exec_latency_s: float = 0.0    # Time from decision to fill (seconds)
    
    # Realistic latency mode
    use_received_timestamps: bool = False  # If True, add CL observation delay
    cl_observation_delay_s: float = 65.0   # CL UI delay (source_ts -> received_ts)
    
    def total_latency(self) -> float:
        """Total latency from event to fill."""
        base_latency = self.signal_latency_s + self.exec_latency_s
        if self.use_received_timestamps:
            # Add CL observation delay - this is when we'd actually see the move
            return base_latency + self.cl_observation_delay_s
        return base_latency
    
    def describe(self) -> str:
        """Human-readable description of latency config."""
        if self.use_received_timestamps:
            return f"Realistic: {self.total_latency():.0f}s total (signal={self.signal_latency_s:.0f}s, exec={self.exec_latency_s:.0f}s, CL_delay={self.cl_observation_delay_s:.0f}s)"
        return f"Optimistic: {self.total_latency():.0f}s total (signal={self.signal_latency_s:.0f}s, exec={self.exec_latency_s:.0f}s)"


@dataclass
class EffectivePrices:
    """Best execution prices including conversion routing."""
    # Effective prices (best of direct or conversion)
    buy_up: float
    sell_up: float
    buy_down: float
    sell_down: float
    
    # Which route is optimal
    buy_up_route: Literal['direct', 'conversion']
    sell_up_route: Literal['direct', 'conversion']
    buy_down_route: Literal['direct', 'conversion']
    sell_down_route: Literal['direct', 'conversion']
    
    # Raw orderbook prices (for reference)
    up_bid: float
    up_ask: float
    down_bid: float
    down_ask: float
    
    # Spread information
    up_spread: float
    down_spread: float
    effective_up_spread: float  # After conversion routing
    effective_down_spread: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'buy_up': self.buy_up,
            'sell_up': self.sell_up,
            'buy_down': self.buy_down,
            'sell_down': self.sell_down,
            'buy_up_route': self.buy_up_route,
            'sell_up_route': self.sell_up_route,
            'buy_down_route': self.buy_down_route,
            'sell_down_route': self.sell_down_route,
            'up_bid': self.up_bid,
            'up_ask': self.up_ask,
            'down_bid': self.down_bid,
            'down_ask': self.down_ask,
            'up_spread': self.up_spread,
            'down_spread': self.down_spread,
            'effective_up_spread': self.effective_up_spread,
            'effective_down_spread': self.effective_down_spread,
        }


@dataclass
class FillResult:
    """Result of a simulated trade fill."""
    side: Literal['buy_up', 'sell_up', 'buy_down', 'sell_down']
    entry_price: float
    route: Literal['direct', 'conversion']
    fill_time: int  # Actual fill time (t)
    signal_time: int  # When the signal occurred
    observe_time: int  # When we observed the signal
    
    # Price improvement from conversion
    price_improvement: float  # vs direct route
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'side': self.side,
            'entry_price': self.entry_price,
            'route': self.route,
            'fill_time': self.fill_time,
            'signal_time': self.signal_time,
            'observe_time': self.observe_time,
            'price_improvement': self.price_improvement,
        }


def get_effective_prices(row: pd.Series) -> EffectivePrices:
    """
    Calculate best execution prices including conversion routing.
    
    Args:
        row: DataFrame row with PM orderbook columns
        
    Returns:
        EffectivePrices with optimal routes
    """
    # Raw orderbook prices
    up_ask = row['pm_up_best_ask']
    up_bid = row['pm_up_best_bid']
    down_ask = row['pm_down_best_ask']
    down_bid = row['pm_down_best_bid']
    
    # Handle NaN values
    if pd.isna(up_ask) or pd.isna(up_bid) or pd.isna(down_ask) or pd.isna(down_bid):
        # Return raw prices with direct routing if any NaN
        return EffectivePrices(
            buy_up=up_ask,
            sell_up=up_bid,
            buy_down=down_ask,
            sell_down=down_bid,
            buy_up_route='direct',
            sell_up_route='direct',
            buy_down_route='direct',
            sell_down_route='direct',
            up_bid=up_bid,
            up_ask=up_ask,
            down_bid=down_bid,
            down_ask=down_ask,
            up_spread=up_ask - up_bid if not pd.isna(up_ask - up_bid) else np.nan,
            down_spread=down_ask - down_bid if not pd.isna(down_ask - down_bid) else np.nan,
            effective_up_spread=np.nan,
            effective_down_spread=np.nan,
        )
    
    # Calculate conversion prices
    # Buy UP via conversion: split $1 into UP+DOWN, sell DOWN for down_bid, keep UP
    # Net cost = 1 - down_bid
    buy_up_via_conversion = 1 - down_bid
    
    # Sell UP via conversion: buy DOWN for down_ask, redeem UP+DOWN for $1
    # Net proceeds = 1 - down_ask
    sell_up_via_conversion = 1 - down_ask
    
    # Buy DOWN via conversion: split $1 into UP+DOWN, sell UP for up_bid, keep DOWN
    buy_down_via_conversion = 1 - up_bid
    
    # Sell DOWN via conversion: buy UP for up_ask, redeem UP+DOWN for $1
    sell_down_via_conversion = 1 - up_ask
    
    # Choose best route
    buy_up = min(up_ask, buy_up_via_conversion)
    buy_up_route = 'direct' if up_ask <= buy_up_via_conversion else 'conversion'
    
    sell_up = max(up_bid, sell_up_via_conversion)
    sell_up_route = 'direct' if up_bid >= sell_up_via_conversion else 'conversion'
    
    buy_down = min(down_ask, buy_down_via_conversion)
    buy_down_route = 'direct' if down_ask <= buy_down_via_conversion else 'conversion'
    
    sell_down = max(down_bid, sell_down_via_conversion)
    sell_down_route = 'direct' if down_bid >= sell_down_via_conversion else 'conversion'
    
    # Calculate spreads
    up_spread = up_ask - up_bid
    down_spread = down_ask - down_bid
    effective_up_spread = buy_up - sell_up
    effective_down_spread = buy_down - sell_down
    
    return EffectivePrices(
        buy_up=buy_up,
        sell_up=sell_up,
        buy_down=buy_down,
        sell_down=sell_down,
        buy_up_route=buy_up_route,
        sell_up_route=sell_up_route,
        buy_down_route=buy_down_route,
        sell_down_route=sell_down_route,
        up_bid=up_bid,
        up_ask=up_ask,
        down_bid=down_bid,
        down_ask=down_ask,
        up_spread=up_spread,
        down_spread=down_spread,
        effective_up_spread=effective_up_spread,
        effective_down_spread=effective_down_spread,
    )


def get_effective_prices_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate effective prices for entire DataFrame (vectorized).
    
    Args:
        df: DataFrame with PM orderbook columns
        
    Returns:
        DataFrame with effective price columns added
    """
    result = df.copy()
    
    up_ask = df['pm_up_best_ask']
    up_bid = df['pm_up_best_bid']
    down_ask = df['pm_down_best_ask']
    down_bid = df['pm_down_best_bid']
    
    # Conversion prices
    buy_up_via_conversion = 1 - down_bid
    sell_up_via_conversion = 1 - down_ask
    buy_down_via_conversion = 1 - up_bid
    sell_down_via_conversion = 1 - up_ask
    
    # Best execution prices
    result['eff_buy_up'] = np.minimum(up_ask, buy_up_via_conversion)
    result['eff_sell_up'] = np.maximum(up_bid, sell_up_via_conversion)
    result['eff_buy_down'] = np.minimum(down_ask, buy_down_via_conversion)
    result['eff_sell_down'] = np.maximum(down_bid, sell_down_via_conversion)
    
    # Route flags (True = conversion is better)
    result['buy_up_conv'] = up_ask > buy_up_via_conversion
    result['sell_up_conv'] = up_bid < sell_up_via_conversion
    result['buy_down_conv'] = down_ask > buy_down_via_conversion
    result['sell_down_conv'] = down_bid < sell_down_via_conversion
    
    # Effective spreads
    result['eff_up_spread'] = result['eff_buy_up'] - result['eff_sell_up']
    result['eff_down_spread'] = result['eff_buy_down'] - result['eff_sell_down']
    
    # Price improvement from conversion
    result['buy_up_improvement'] = up_ask - result['eff_buy_up']
    result['sell_up_improvement'] = result['eff_sell_up'] - up_bid
    result['buy_down_improvement'] = down_ask - result['eff_buy_down']
    result['sell_down_improvement'] = result['eff_sell_down'] - down_bid
    
    return result


def simulate_fill(
    market_df: pd.DataFrame,
    signal_time: int,
    side: Literal['buy_up', 'sell_up', 'buy_down', 'sell_down'],
    config: ExecutionConfig
) -> Optional[FillResult]:
    """
    Simulate a trade fill given signal time and execution config.
    
    Args:
        market_df: DataFrame for single market
        signal_time: Time (t) when signal occurred
        side: Direction of trade
        config: Execution configuration
        
    Returns:
        FillResult if fill possible, None if time doesn't exist
    """
    # Time when we observe the signal
    observe_time = signal_time + int(config.signal_latency_s)
    # Time when trade executes
    fill_time = observe_time + int(config.exec_latency_s)
    
    # Get market state at fill time
    fill_row = market_df[market_df['t'] == fill_time]
    if fill_row.empty:
        return None  # Can't fill - time doesn't exist
    
    row = fill_row.iloc[0]
    prices = get_effective_prices(row)
    
    # Get price and route for this side
    if side == 'buy_up':
        entry_price = prices.buy_up
        route = prices.buy_up_route
        direct_price = prices.up_ask
    elif side == 'sell_up':
        entry_price = prices.sell_up
        route = prices.sell_up_route
        direct_price = prices.up_bid
    elif side == 'buy_down':
        entry_price = prices.buy_down
        route = prices.buy_down_route
        direct_price = prices.down_ask
    elif side == 'sell_down':
        entry_price = prices.sell_down
        route = prices.sell_down_route
        direct_price = prices.down_bid
    else:
        raise ValueError(f"Unknown side: {side}")
    
    # Calculate price improvement
    if side.startswith('buy'):
        price_improvement = direct_price - entry_price  # Lower is better
    else:
        price_improvement = entry_price - direct_price  # Higher is better
    
    return FillResult(
        side=side,
        entry_price=entry_price,
        route=route,
        fill_time=fill_time,
        signal_time=signal_time,
        observe_time=observe_time,
        price_improvement=price_improvement,
    )


def compute_trade_pnl(
    entry_fill: FillResult,
    exit_fill: FillResult
) -> float:
    """
    Compute PnL for a completed trade.
    
    Args:
        entry_fill: Entry fill result
        exit_fill: Exit fill result
        
    Returns:
        PnL in dollars per token
    """
    if entry_fill.side.startswith('buy'):
        # Bought, then sold
        return exit_fill.entry_price - entry_fill.entry_price
    else:
        # Sold, then bought back (short)
        return entry_fill.entry_price - exit_fill.entry_price


def analyze_conversion_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze how often conversion routing is optimal.
    
    Args:
        df: DataFrame with effective prices calculated
        
    Returns:
        Statistics on conversion usage
    """
    df = get_effective_prices_vectorized(df)
    
    total_rows = len(df)
    
    return {
        'total_observations': total_rows,
        'buy_up_conversion_pct': df['buy_up_conv'].mean() * 100,
        'sell_up_conversion_pct': df['sell_up_conv'].mean() * 100,
        'buy_down_conversion_pct': df['buy_down_conv'].mean() * 100,
        'sell_down_conversion_pct': df['sell_down_conv'].mean() * 100,
        'avg_buy_up_improvement': df['buy_up_improvement'].mean(),
        'avg_sell_up_improvement': df['sell_up_improvement'].mean(),
        'avg_buy_down_improvement': df['buy_down_improvement'].mean(),
        'avg_sell_down_improvement': df['sell_down_improvement'].mean(),
        'avg_eff_up_spread': df['eff_up_spread'].mean(),
        'avg_eff_down_spread': df['eff_down_spread'].mean(),
        'avg_raw_up_spread': (df['pm_up_best_ask'] - df['pm_up_best_bid']).mean(),
        'avg_raw_down_spread': (df['pm_down_best_ask'] - df['pm_down_best_bid']).mean(),
    }


# ==============================================================================
# Tests
# ==============================================================================

# =============================================================================
# DEPTH-AWARE EXECUTION (L6 Order Book)
# =============================================================================

@dataclass
class DepthFillResult:
    """Result of walking the order book to fill an order."""
    vwap: float                    # Volume-weighted average price
    filled_qty: float              # Quantity actually filled
    requested_qty: float           # Quantity requested
    is_partial: bool               # True if couldn't fill full qty
    levels_used: int               # Number of levels consumed (1-6)
    total_depth_available: float   # Total size available across all levels
    fill_details: Dict[int, Dict]  # Per-level fill details
    
    @property
    def is_complete(self) -> bool:
        """True if order was completely filled."""
        return not self.is_partial
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'vwap': self.vwap,
            'filled_qty': self.filled_qty,
            'requested_qty': self.requested_qty,
            'is_partial': self.is_partial,
            'levels_used': self.levels_used,
            'total_depth_available': self.total_depth_available,
            'fill_details': self.fill_details,
        }


@dataclass
class DepthEffectivePrices:
    """Best execution prices with depth-aware VWAP fills."""
    # Effective prices at requested size (VWAP)
    buy_up_vwap: float
    sell_up_vwap: float
    buy_down_vwap: float
    sell_down_vwap: float
    
    # Which route is optimal at size
    buy_up_route: Literal['direct', 'conversion']
    sell_up_route: Literal['direct', 'conversion']
    buy_down_route: Literal['direct', 'conversion']
    sell_down_route: Literal['direct', 'conversion']
    
    # Fill completeness
    buy_up_complete: bool
    sell_up_complete: bool
    buy_down_complete: bool
    sell_down_complete: bool
    
    # Levels used
    buy_up_levels: int
    sell_up_levels: int
    buy_down_levels: int
    sell_down_levels: int
    
    # Requested size
    size: float
    
    # Comparison to L1-only
    buy_up_l1: float   # L1-only price for comparison
    sell_up_l1: float
    buy_down_l1: float
    sell_down_l1: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'buy_up_vwap': self.buy_up_vwap,
            'sell_up_vwap': self.sell_up_vwap,
            'buy_down_vwap': self.buy_down_vwap,
            'sell_down_vwap': self.sell_down_vwap,
            'buy_up_route': self.buy_up_route,
            'sell_up_route': self.sell_up_route,
            'buy_down_route': self.buy_down_route,
            'sell_down_route': self.sell_down_route,
            'buy_up_complete': self.buy_up_complete,
            'sell_up_complete': self.sell_up_complete,
            'buy_down_complete': self.buy_down_complete,
            'sell_down_complete': self.sell_down_complete,
            'buy_up_levels': self.buy_up_levels,
            'sell_up_levels': self.sell_up_levels,
            'buy_down_levels': self.buy_down_levels,
            'sell_down_levels': self.sell_down_levels,
            'size': self.size,
            'buy_up_l1': self.buy_up_l1,
            'sell_up_l1': self.sell_up_l1,
            'buy_down_l1': self.buy_down_l1,
            'sell_down_l1': self.sell_down_l1,
        }


def get_l6_column_names_canonical() -> Dict[str, list]:
    """
    Get canonical column names for L1-L6 data.
    
    Returns dict with keys mapping side/outcome to list of (price_col, size_col) tuples.
    """
    return {
        'up_ask': [
            ('pm_up_best_ask', 'pm_up_best_ask_size'),
            ('pm_up_ask_2', 'pm_up_ask_2_size'),
            ('pm_up_ask_3', 'pm_up_ask_3_size'),
            ('pm_up_ask_4', 'pm_up_ask_4_size'),
            ('pm_up_ask_5', 'pm_up_ask_5_size'),
            ('pm_up_ask_6', 'pm_up_ask_6_size'),
        ],
        'up_bid': [
            ('pm_up_best_bid', 'pm_up_best_bid_size'),
            ('pm_up_bid_2', 'pm_up_bid_2_size'),
            ('pm_up_bid_3', 'pm_up_bid_3_size'),
            ('pm_up_bid_4', 'pm_up_bid_4_size'),
            ('pm_up_bid_5', 'pm_up_bid_5_size'),
            ('pm_up_bid_6', 'pm_up_bid_6_size'),
        ],
        'down_ask': [
            ('pm_down_best_ask', 'pm_down_best_ask_size'),
            ('pm_down_ask_2', 'pm_down_ask_2_size'),
            ('pm_down_ask_3', 'pm_down_ask_3_size'),
            ('pm_down_ask_4', 'pm_down_ask_4_size'),
            ('pm_down_ask_5', 'pm_down_ask_5_size'),
            ('pm_down_ask_6', 'pm_down_ask_6_size'),
        ],
        'down_bid': [
            ('pm_down_best_bid', 'pm_down_best_bid_size'),
            ('pm_down_bid_2', 'pm_down_bid_2_size'),
            ('pm_down_bid_3', 'pm_down_bid_3_size'),
            ('pm_down_bid_4', 'pm_down_bid_4_size'),
            ('pm_down_bid_5', 'pm_down_bid_5_size'),
            ('pm_down_bid_6', 'pm_down_bid_6_size'),
        ],
    }


def walk_the_book(
    row: pd.Series,
    side: Literal['buy', 'sell'],
    outcome: Literal['up', 'down'],
    qty: float
) -> DepthFillResult:
    """
    Walk the order book to fill an order, computing VWAP.
    
    For buy orders, we consume asks (ascending price).
    For sell orders, we consume bids (descending price).
    
    Args:
        row: DataFrame row with L6 orderbook columns
        side: 'buy' or 'sell'
        outcome: 'up' or 'down'
        qty: Quantity to fill (in shares/tokens)
        
    Returns:
        DepthFillResult with VWAP, fill status, and details
    """
    cols = get_l6_column_names_canonical()
    
    # Get the appropriate book side
    if side == 'buy':
        # Buying = consuming asks
        book_key = f'{outcome}_ask'
    else:
        # Selling = consuming bids
        book_key = f'{outcome}_bid'
    
    levels = cols.get(book_key, [])
    
    remaining_qty = qty
    total_cost = 0.0
    filled_qty = 0.0
    levels_used = 0
    total_depth = 0.0
    fill_details = {}
    
    for i, (price_col, size_col) in enumerate(levels):
        level_num = i + 1
        
        # Get price and size at this level
        # Handle missing columns gracefully (L1-only markets)
        if price_col not in row.index:
            price = np.nan
        else:
            price = row.get(price_col, np.nan)
        
        if size_col not in row.index:
            size = np.nan
        else:
            size = row.get(size_col, np.nan)
        
        # Skip if level is empty (NaN or zero size)
        # This handles both missing columns (L1-only) and empty levels (L6 markets)
        if pd.isna(price) or pd.isna(size) or size <= 0:
            continue
        
        total_depth += size
        
        # Calculate how much we can fill at this level
        fill_at_level = min(remaining_qty, size)
        
        if fill_at_level > 0:
            total_cost += price * fill_at_level
            filled_qty += fill_at_level
            remaining_qty -= fill_at_level
            levels_used = level_num
            
            fill_details[level_num] = {
                'price': price,
                'size_available': size,
                'filled': fill_at_level,
            }
        
        # Check if we're done
        if remaining_qty <= 0:
            break
    
    # Calculate VWAP
    if filled_qty > 0:
        vwap = total_cost / filled_qty
    else:
        vwap = np.nan
    
    return DepthFillResult(
        vwap=vwap,
        filled_qty=filled_qty,
        requested_qty=qty,
        is_partial=remaining_qty > 0,
        levels_used=levels_used,
        total_depth_available=total_depth,
        fill_details=fill_details,
    )


def get_effective_prices_with_depth(
    row: pd.Series,
    size: float = 1.0
) -> DepthEffectivePrices:
    """
    Calculate best execution prices with depth-aware VWAP fills.
    
    This is the depth-aware version of get_effective_prices(). For each
    action (buy/sell UP/DOWN), it:
    1. Walks the direct book to get VWAP
    2. Walks the conversion book to get VWAP  
    3. Chooses the better route
    
    Args:
        row: DataFrame row with L6 orderbook columns
        size: Order size in shares/tokens
        
    Returns:
        DepthEffectivePrices with VWAP fills and routing info
    """
    # Get L1 prices for comparison
    l1_prices = get_effective_prices(row)
    
    # === BUY UP ===
    # Direct: walk UP asks
    direct_buy_up = walk_the_book(row, 'buy', 'up', size)
    # Conversion: sell DOWN into DOWN bids, keep UP from split
    # Cost = 1 - VWAP(sell DOWN)
    conv_sell_down = walk_the_book(row, 'sell', 'down', size)
    conv_buy_up_price = 1 - conv_sell_down.vwap if not pd.isna(conv_sell_down.vwap) else np.nan
    
    if pd.isna(direct_buy_up.vwap) and pd.isna(conv_buy_up_price):
        buy_up_vwap = np.nan
        buy_up_route = 'direct'
        buy_up_complete = False
        buy_up_levels = 0
    elif pd.isna(conv_buy_up_price) or (not pd.isna(direct_buy_up.vwap) and direct_buy_up.vwap <= conv_buy_up_price):
        buy_up_vwap = direct_buy_up.vwap
        buy_up_route = 'direct'
        buy_up_complete = direct_buy_up.is_complete
        buy_up_levels = direct_buy_up.levels_used
    else:
        buy_up_vwap = conv_buy_up_price
        buy_up_route = 'conversion'
        buy_up_complete = conv_sell_down.is_complete
        buy_up_levels = conv_sell_down.levels_used
    
    # === SELL UP ===
    # Direct: walk UP bids
    direct_sell_up = walk_the_book(row, 'sell', 'up', size)
    # Conversion: buy DOWN, redeem UP+DOWN for $1
    # Proceeds = 1 - VWAP(buy DOWN)
    conv_buy_down = walk_the_book(row, 'buy', 'down', size)
    conv_sell_up_price = 1 - conv_buy_down.vwap if not pd.isna(conv_buy_down.vwap) else np.nan
    
    if pd.isna(direct_sell_up.vwap) and pd.isna(conv_sell_up_price):
        sell_up_vwap = np.nan
        sell_up_route = 'direct'
        sell_up_complete = False
        sell_up_levels = 0
    elif pd.isna(conv_sell_up_price) or (not pd.isna(direct_sell_up.vwap) and direct_sell_up.vwap >= conv_sell_up_price):
        sell_up_vwap = direct_sell_up.vwap
        sell_up_route = 'direct'
        sell_up_complete = direct_sell_up.is_complete
        sell_up_levels = direct_sell_up.levels_used
    else:
        sell_up_vwap = conv_sell_up_price
        sell_up_route = 'conversion'
        sell_up_complete = conv_buy_down.is_complete
        sell_up_levels = conv_buy_down.levels_used
    
    # === BUY DOWN ===
    # Direct: walk DOWN asks
    direct_buy_down = walk_the_book(row, 'buy', 'down', size)
    # Conversion: sell UP into UP bids, keep DOWN from split
    conv_sell_up = walk_the_book(row, 'sell', 'up', size)
    conv_buy_down_price = 1 - conv_sell_up.vwap if not pd.isna(conv_sell_up.vwap) else np.nan
    
    if pd.isna(direct_buy_down.vwap) and pd.isna(conv_buy_down_price):
        buy_down_vwap = np.nan
        buy_down_route = 'direct'
        buy_down_complete = False
        buy_down_levels = 0
    elif pd.isna(conv_buy_down_price) or (not pd.isna(direct_buy_down.vwap) and direct_buy_down.vwap <= conv_buy_down_price):
        buy_down_vwap = direct_buy_down.vwap
        buy_down_route = 'direct'
        buy_down_complete = direct_buy_down.is_complete
        buy_down_levels = direct_buy_down.levels_used
    else:
        buy_down_vwap = conv_buy_down_price
        buy_down_route = 'conversion'
        buy_down_complete = conv_sell_up.is_complete
        buy_down_levels = conv_sell_up.levels_used
    
    # === SELL DOWN ===
    # Direct: walk DOWN bids
    direct_sell_down = walk_the_book(row, 'sell', 'down', size)
    # Conversion: buy UP, redeem UP+DOWN for $1
    conv_buy_up = walk_the_book(row, 'buy', 'up', size)
    conv_sell_down_price = 1 - conv_buy_up.vwap if not pd.isna(conv_buy_up.vwap) else np.nan
    
    if pd.isna(direct_sell_down.vwap) and pd.isna(conv_sell_down_price):
        sell_down_vwap = np.nan
        sell_down_route = 'direct'
        sell_down_complete = False
        sell_down_levels = 0
    elif pd.isna(conv_sell_down_price) or (not pd.isna(direct_sell_down.vwap) and direct_sell_down.vwap >= conv_sell_down_price):
        sell_down_vwap = direct_sell_down.vwap
        sell_down_route = 'direct'
        sell_down_complete = direct_sell_down.is_complete
        sell_down_levels = direct_sell_down.levels_used
    else:
        sell_down_vwap = conv_sell_down_price
        sell_down_route = 'conversion'
        sell_down_complete = conv_buy_up.is_complete
        sell_down_levels = conv_buy_up.levels_used
    
    return DepthEffectivePrices(
        buy_up_vwap=buy_up_vwap,
        sell_up_vwap=sell_up_vwap,
        buy_down_vwap=buy_down_vwap,
        sell_down_vwap=sell_down_vwap,
        buy_up_route=buy_up_route,
        sell_up_route=sell_up_route,
        buy_down_route=buy_down_route,
        sell_down_route=sell_down_route,
        buy_up_complete=buy_up_complete,
        sell_up_complete=sell_up_complete,
        buy_down_complete=buy_down_complete,
        sell_down_complete=sell_down_complete,
        buy_up_levels=buy_up_levels,
        sell_up_levels=sell_up_levels,
        buy_down_levels=buy_down_levels,
        sell_down_levels=sell_down_levels,
        size=size,
        buy_up_l1=l1_prices.buy_up,
        sell_up_l1=l1_prices.sell_up,
        buy_down_l1=l1_prices.buy_down,
        sell_down_l1=l1_prices.sell_down,
    )


def simulate_depth_fill(
    market_df: pd.DataFrame,
    signal_time: int,
    side: Literal['buy_up', 'sell_up', 'buy_down', 'sell_down'],
    size: float,
    config: ExecutionConfig = None
) -> Optional[Dict[str, Any]]:
    """
    Simulate a fill with depth-aware execution.
    
    Args:
        market_df: Market data with L6 columns
        signal_time: Time when signal occurred (t)
        side: Trade side
        size: Order size
        config: Execution config (for latency)
        
    Returns:
        Dict with fill details or None if fill not possible
    """
    if config is None:
        config = ExecutionConfig()
    
    # Calculate fill time with latency
    fill_time = signal_time + int(config.total_latency())
    
    # Get row at fill time
    fill_rows = market_df[market_df['t'] == fill_time]
    if fill_rows.empty:
        return None
    
    row = fill_rows.iloc[0]
    
    # Get depth-aware prices
    depth_prices = get_effective_prices_with_depth(row, size)
    
    # Extract relevant price based on side
    if side == 'buy_up':
        entry_price = depth_prices.buy_up_vwap
        route = depth_prices.buy_up_route
        is_complete = depth_prices.buy_up_complete
        levels = depth_prices.buy_up_levels
        l1_price = depth_prices.buy_up_l1
    elif side == 'sell_up':
        entry_price = depth_prices.sell_up_vwap
        route = depth_prices.sell_up_route
        is_complete = depth_prices.sell_up_complete
        levels = depth_prices.sell_up_levels
        l1_price = depth_prices.sell_up_l1
    elif side == 'buy_down':
        entry_price = depth_prices.buy_down_vwap
        route = depth_prices.buy_down_route
        is_complete = depth_prices.buy_down_complete
        levels = depth_prices.buy_down_levels
        l1_price = depth_prices.buy_down_l1
    else:  # sell_down
        entry_price = depth_prices.sell_down_vwap
        route = depth_prices.sell_down_route
        is_complete = depth_prices.sell_down_complete
        levels = depth_prices.sell_down_levels
        l1_price = depth_prices.sell_down_l1
    
    if pd.isna(entry_price):
        return None
    
    # Calculate slippage vs L1
    slippage = entry_price - l1_price if not pd.isna(l1_price) else 0
    
    return {
        'side': side,
        'size': size,
        'signal_time': signal_time,
        'fill_time': fill_time,
        'entry_price': entry_price,
        'l1_price': l1_price,
        'slippage': slippage,
        'route': route,
        'is_complete': is_complete,
        'levels_used': levels,
    }


# =============================================================================
# TESTS
# =============================================================================

def test_effective_prices():
    """Test effective price calculation."""
    # Test case: conversion is better for buying UP
    row = pd.Series({
        'pm_up_best_ask': 0.55,
        'pm_up_best_bid': 0.50,
        'pm_down_best_ask': 0.50,
        'pm_down_best_bid': 0.46,  # 1 - 0.46 = 0.54 < 0.55, conversion better
    })
    
    prices = get_effective_prices(row)
    
    assert prices.buy_up == 0.54, f"Expected 0.54, got {prices.buy_up}"
    assert prices.buy_up_route == 'conversion', f"Expected conversion, got {prices.buy_up_route}"
    
    # Sell UP: 1 - 0.50 = 0.50 == 0.50, tie goes to direct
    assert prices.sell_up == 0.50, f"Expected 0.50, got {prices.sell_up}"
    assert prices.sell_up_route == 'direct', f"Expected direct, got {prices.sell_up_route}"
    
    print("[OK] Effective prices test passed")


def test_latency():
    """Test latency application."""
    # Create test market
    data = {
        't': [0, 1, 2, 3, 4, 5],
        'pm_up_best_ask': [0.55, 0.55, 0.56, 0.57, 0.58, 0.59],
        'pm_up_best_bid': [0.50, 0.50, 0.51, 0.52, 0.53, 0.54],
        'pm_down_best_ask': [0.50, 0.50, 0.49, 0.48, 0.47, 0.46],
        'pm_down_best_bid': [0.45, 0.45, 0.44, 0.43, 0.42, 0.41],
    }
    market_df = pd.DataFrame(data)
    
    # Signal at t=0, no latency
    config = ExecutionConfig(signal_latency_s=0, exec_latency_s=0)
    fill = simulate_fill(market_df, signal_time=0, side='buy_up', config=config)
    assert fill is not None
    assert fill.fill_time == 0
    
    # Signal at t=0, 2s total latency
    config = ExecutionConfig(signal_latency_s=1, exec_latency_s=1)
    fill = simulate_fill(market_df, signal_time=0, side='buy_up', config=config)
    assert fill is not None
    assert fill.fill_time == 2
    
    # Signal at t=3, 5s latency -> fill at t=8 (doesn't exist)
    config = ExecutionConfig(signal_latency_s=2, exec_latency_s=3)
    fill = simulate_fill(market_df, signal_time=3, side='buy_up', config=config)
    assert fill is None
    
    print("[OK] Latency test passed")


if __name__ == '__main__':
    test_effective_prices()
    test_latency()
    print("\nAll execution model tests passed!")

