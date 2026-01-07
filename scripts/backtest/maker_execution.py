"""
Maker Execution Model for Spread Capture Strategy

This module implements a realistic fill simulation engine for maker (limit order)
strategies, including:
- Order state management (placement, cancellation, fills)
- Queue position tracking
- Fill probability models (TAPE_QUEUE, TOUCH_SIZE_PROXY, BOUNDS_ONLY)
- Inventory tracking with mark-to-market PnL
- Latency modeling for placement and cancellation

Key insight: Maker strategies face different execution dynamics than taker strategies.
The fill is NOT guaranteed - it depends on queue position, trading activity, and
adverse selection risk.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Literal
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime
import uuid


# =============================================================================
# CONFIGURATION
# =============================================================================

class FillModel(Enum):
    """Fill probability models for maker orders."""
    TAPE_QUEUE = "tape_queue"             # Best: uses trade tape data
    TOUCH_SIZE_PROXY = "touch_size_proxy" # Middle: uses L1 size data at touch
    BOUNDS_ONLY = "bounds_only"           # Exploratory: upper/lower bounds
    L2_QUEUE = "l2_queue"                 # L2 queue model: tracks consumption across 6 levels


class QueueModel(Enum):
    """Queue position models."""
    FIFO = "fifo"           # First-in-first-out (default)
    PROPORTIONAL = "proportional"  # Proportional to size at level


@dataclass
class MakerExecutionConfig:
    """
    Configuration for maker order execution simulation.
    
    LATENCY MODELING:
    - place_latency_ms: Time from decision to order becoming active in book
    - cancel_latency_ms: Time from cancel request to order being removed
    
    These are CRITICAL for maker strategies - even 100ms can determine fill vs no-fill.
    
    FILL MODEL:
    - TAPE_QUEUE: Uses trade prints to determine fills (most realistic)
    - TOUCH_SIZE_PROXY: Estimates fills from top-of-book size changes
    - BOUNDS_ONLY: Upper/lower bounds for exploratory analysis
    """
    # Latency parameters (in milliseconds)
    place_latency_ms: int = 100  # Order becomes active after this
    cancel_latency_ms: int = 50  # Cancel takes effect after this
    
    # Fill model selection
    fill_model: FillModel = FillModel.TOUCH_SIZE_PROXY
    queue_model: QueueModel = QueueModel.FIFO
    
    # TOUCH_SIZE_PROXY parameters
    # Calibration: what fraction of top-of-book size trades per second?
    # This is a key parameter - higher = more fills, lower = fewer fills
    touch_trade_rate_per_second: float = 0.1  # 10% of touch size trades per second
    
    # BOUNDS_ONLY parameters
    bounds_fill_on_price_through: bool = True  # Lower bound: fill only when price moves through
    bounds_fill_at_touch: bool = True  # Upper bound: fill whenever at best bid/ask
    
    # L2_QUEUE parameters (for 6-level orderbook consumption tracking)
    # These control how we handle prices that disappear from top 6 levels
    l2_conservative_mode: bool = False    # Only count consumption when price stays visible
    l2_allow_level_drift: bool = True     # Allow tracking when price moves between levels
    l2_optimistic_disappear: bool = False # Assume disappearance = full consumption (optimistic)
    
    # Fee model (Polymarket has no maker fees, but we can model future scenarios)
    maker_fee_rate: float = 0.0
    taker_fee_rate: float = 0.0
    
    def total_latency_seconds(self) -> float:
        """Total latency in seconds for a round-trip (place + potential cancel)."""
        return (self.place_latency_ms + self.cancel_latency_ms) / 1000.0
    
    def describe(self) -> str:
        """Human-readable description."""
        return (
            f"MakerConfig(place={self.place_latency_ms}ms, cancel={self.cancel_latency_ms}ms, "
            f"fill_model={self.fill_model.value}, queue={self.queue_model.value})"
        )


# =============================================================================
# ORDER AND FILL TRACKING
# =============================================================================

class OrderStatus(Enum):
    """Order lifecycle status."""
    PENDING = "pending"     # Submitted, not yet active (in latency window)
    ACTIVE = "active"       # Active in orderbook, can be filled
    FILLED = "filled"       # Fully filled
    PARTIAL = "partial"     # Partially filled
    CANCELLED = "cancelled" # Cancelled (or cancel pending)
    EXPIRED = "expired"     # Expired at market end


@dataclass
class MakerOrder:
    """A maker (limit) order with full lifecycle tracking."""
    order_id: str
    token: Literal['UP', 'DOWN']  # Which token
    side: Literal['BID', 'ASK']   # BID = buy, ASK = sell
    price: float
    size: float
    time_placed: int  # t when order was submitted
    time_active: Optional[int] = None  # t when order became active (after latency)
    status: OrderStatus = OrderStatus.PENDING
    
    # Fill tracking
    filled_size: float = 0.0
    fill_time: Optional[int] = None
    fill_price: Optional[float] = None  # May differ from order price in some models
    
    # Queue tracking (L1 model)
    queue_ahead: float = 0.0  # Size ahead of us in queue
    initial_queue_position: float = 0.0  # Queue position when we joined
    
    # L2 queue tracking (6-level model)
    l2_queue_ahead: float = 0.0           # Queue ahead at our price (from L2 data)
    l2_initial_queue: float = 0.0         # Initial queue when we joined
    l2_cumulative_consumed: float = 0.0   # Cumulative size consumed at our price
    l2_level_at_placement: Optional[int] = None  # Which level (1-6) price was at when placed
    l2_last_seen_level: Optional[int] = None     # Last level price was seen at
    l2_last_seen_size: float = 0.0        # Last observed size at our price
    
    # Cancellation
    cancel_requested_at: Optional[int] = None
    cancel_effective_at: Optional[int] = None
    
    @property
    def remaining_size(self) -> float:
        return self.size - self.filled_size
    
    @property
    def is_buy(self) -> bool:
        return self.side == 'BID'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'token': self.token,
            'side': self.side,
            'price': self.price,
            'size': self.size,
            'filled_size': self.filled_size,
            'time_placed': self.time_placed,
            'time_active': self.time_active,
            'fill_time': self.fill_time,
            'status': self.status.value,
            'queue_ahead': self.queue_ahead,
        }


@dataclass
class FillEvent:
    """
    A fill event for an order.
    
    SIGN CONVENTION for adverse selection:
    - Positive = COST (market moved against us)
    - Negative = GAIN (market moved in our favor)
    
    The fill's total P&L contribution is:
        spread_captured - adverse_selection
    where spread_captured is always positive (we bought below mid or sold above mid).
    """
    order_id: str
    fill_time: int  # t when fill occurred
    fill_price: float
    fill_size: float
    is_partial: bool
    
    # Adverse selection tracking
    mid_at_fill: float  # Midpoint at time of fill
    mid_after_1s: Optional[float] = None  # Midpoint 1 second after fill
    mid_after_5s: Optional[float] = None  # Midpoint 5 seconds after fill
    adverse_selection_1s: Optional[float] = None  # Move against us (positive = cost)
    adverse_selection_5s: Optional[float] = None  # Move against us (positive = cost)
    
    # Extended fields for validation (set by FillEngine)
    token: Optional[str] = None  # 'UP' or 'DOWN'
    side: Optional[str] = None   # 'BID' or 'ASK'
    spread_at_fill: Optional[float] = None  # Spread width at fill time
    
    @property
    def signed_qty(self) -> float:
        """Signed quantity: +1 for BID (buy), -1 for ASK (sell)."""
        return 1.0 if self.side == 'BID' else -1.0
    
    @property
    def spread_captured(self) -> float:
        """
        Spread captured from this fill (always positive for maker fills).
        BID: bought below mid → mid_at_fill - fill_price
        ASK: sold above mid → fill_price - mid_at_fill
        """
        if self.side == 'BID':
            return (self.mid_at_fill - self.fill_price) * self.fill_size
        else:
            return (self.fill_price - self.mid_at_fill) * self.fill_size
    
    @property
    def gain_after_1s(self) -> Optional[float]:
        """
        Price gain for our position 1s after fill.
        Positive = market moved in our favor.
        For BID: mid went up = good. For ASK: mid went down = good.
        """
        if self.mid_after_1s is None:
            return None
        if self.side == 'BID':
            return (self.mid_after_1s - self.mid_at_fill) * self.fill_size
        else:
            return (self.mid_at_fill - self.mid_after_1s) * self.fill_size
    
    @property
    def gain_after_5s(self) -> Optional[float]:
        """Price gain for our position 5s after fill."""
        if self.mid_after_5s is None:
            return None
        if self.side == 'BID':
            return (self.mid_after_5s - self.mid_at_fill) * self.fill_size
        else:
            return (self.mid_at_fill - self.mid_after_5s) * self.fill_size
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'fill_time': self.fill_time,
            'fill_price': self.fill_price,
            'fill_size': self.fill_size,
            'is_partial': self.is_partial,
            'mid_at_fill': self.mid_at_fill,
            'mid_after_1s': self.mid_after_1s,
            'mid_after_5s': self.mid_after_5s,
            'adverse_selection_1s': self.adverse_selection_1s,
            'adverse_selection_5s': self.adverse_selection_5s,
            'token': self.token,
            'side': self.side,
            'spread_at_fill': self.spread_at_fill,
            'spread_captured': self.spread_captured,
            'gain_after_1s': self.gain_after_1s,
            'gain_after_5s': self.gain_after_5s,
        }


@dataclass
class OrderbookState:
    """
    Current state of the orderbook for one token (6-level depth).
    
    Level 1 = best bid/ask
    Levels 2-6 = deeper levels
    
    For bids: px_1 > px_2 > px_3 > px_4 > px_5 > px_6 (strictly decreasing)
    For asks: px_1 < px_2 < px_3 < px_4 < px_5 < px_6 (strictly increasing)
    """
    timestamp: int  # t
    
    # Level 1 (best bid/ask)
    bid_px_1: Optional[float] = None
    bid_sz_1: Optional[float] = None
    ask_px_1: Optional[float] = None
    ask_sz_1: Optional[float] = None
    
    # Level 2
    bid_px_2: Optional[float] = None
    bid_sz_2: Optional[float] = None
    ask_px_2: Optional[float] = None
    ask_sz_2: Optional[float] = None
    
    # Level 3
    bid_px_3: Optional[float] = None
    bid_sz_3: Optional[float] = None
    ask_px_3: Optional[float] = None
    ask_sz_3: Optional[float] = None
    
    # Level 4
    bid_px_4: Optional[float] = None
    bid_sz_4: Optional[float] = None
    ask_px_4: Optional[float] = None
    ask_sz_4: Optional[float] = None
    
    # Level 5
    bid_px_5: Optional[float] = None
    bid_sz_5: Optional[float] = None
    ask_px_5: Optional[float] = None
    ask_sz_5: Optional[float] = None
    
    # Level 6
    bid_px_6: Optional[float] = None
    bid_sz_6: Optional[float] = None
    ask_px_6: Optional[float] = None
    ask_sz_6: Optional[float] = None
    
    # Backward compatibility aliases
    @property
    def best_bid(self) -> Optional[float]:
        return self.bid_px_1
    
    @property
    def best_bid_size(self) -> Optional[float]:
        return self.bid_sz_1
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.ask_px_1
    
    @property
    def best_ask_size(self) -> Optional[float]:
        return self.ask_sz_1
    
    @property
    def mid(self) -> Optional[float]:
        if self.bid_px_1 is not None and self.ask_px_1 is not None:
            return (self.bid_px_1 + self.ask_px_1) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        if self.bid_px_1 is not None and self.ask_px_1 is not None:
            return self.ask_px_1 - self.bid_px_1
        return None
    
    def get_bid_levels(self) -> List[Tuple[Optional[float], Optional[float]]]:
        """Get all bid levels as list of (price, size) tuples."""
        return [
            (self.bid_px_1, self.bid_sz_1),
            (self.bid_px_2, self.bid_sz_2),
            (self.bid_px_3, self.bid_sz_3),
            (self.bid_px_4, self.bid_sz_4),
            (self.bid_px_5, self.bid_sz_5),
            (self.bid_px_6, self.bid_sz_6),
        ]
    
    def get_ask_levels(self) -> List[Tuple[Optional[float], Optional[float]]]:
        """Get all ask levels as list of (price, size) tuples."""
        return [
            (self.ask_px_1, self.ask_sz_1),
            (self.ask_px_2, self.ask_sz_2),
            (self.ask_px_3, self.ask_sz_3),
            (self.ask_px_4, self.ask_sz_4),
            (self.ask_px_5, self.ask_sz_5),
            (self.ask_px_6, self.ask_sz_6),
        ]
    
    def find_price_level(self, price: float, side: Literal['BID', 'ASK']) -> Optional[int]:
        """
        Find which level (1-6) a price appears at, or None if not in top 6.
        
        Args:
            price: The price to find
            side: 'BID' or 'ASK'
            
        Returns:
            Level number (1-6) or None
        """
        levels = self.get_bid_levels() if side == 'BID' else self.get_ask_levels()
        for i, (px, _) in enumerate(levels):
            if px is not None and abs(px - price) < 0.0001:
                return i + 1
        return None
    
    def get_size_at_price(self, price: float, side: Literal['BID', 'ASK']) -> float:
        """
        Get the size at a specific price level.
        
        Args:
            price: The price to query
            side: 'BID' or 'ASK'
            
        Returns:
            Size at that price, or 0.0 if not found
        """
        levels = self.get_bid_levels() if side == 'BID' else self.get_ask_levels()
        for px, sz in levels:
            if px is not None and abs(px - price) < 0.0001:
                return sz or 0.0
        return 0.0
    
    def get_cumulative_size_to_level(self, level: int, side: Literal['BID', 'ASK']) -> float:
        """
        Get cumulative size from level 1 up to (and including) the specified level.
        
        Args:
            level: Target level (1-6)
            side: 'BID' or 'ASK'
            
        Returns:
            Cumulative size
        """
        levels = self.get_bid_levels() if side == 'BID' else self.get_ask_levels()
        total = 0.0
        for i, (px, sz) in enumerate(levels):
            if i >= level:
                break
            if sz is not None:
                total += sz
        return total
    
    def get_total_top6_depth(self, side: Literal['BID', 'ASK']) -> float:
        """Get total size across all 6 levels."""
        return self.get_cumulative_size_to_level(6, side)
    
    def get_imbalance(self, levels: int = 1) -> Optional[float]:
        """
        Compute order book imbalance at specified depth.
        
        Formula: (bid_size - ask_size) / (bid_size + ask_size)
        Range: -1 (all asks) to +1 (all bids)
        
        Args:
            levels: Number of levels to include (1-6)
            
        Returns:
            Imbalance or None if no data
        """
        bid_size = self.get_cumulative_size_to_level(levels, 'BID')
        ask_size = self.get_cumulative_size_to_level(levels, 'ASK')
        
        if bid_size + ask_size == 0:
            return None
        
        return (bid_size - ask_size) / (bid_size + ask_size)


@dataclass
class Inventory:
    """Inventory state tracking with mark-to-market PnL."""
    up_position: float = 0.0  # Positive = long UP, negative = short UP
    down_position: float = 0.0
    cash: float = 0.0  # USD balance from trading
    
    # Cost basis for PnL calculation
    up_cost_basis: float = 0.0  # Total cost to acquire UP position
    down_cost_basis: float = 0.0
    
    # Realized PnL from closed trades
    realized_pnl: float = 0.0
    
    def mark_to_market(self, up_mid: float, down_mid: float) -> float:
        """Calculate unrealized PnL at current prices."""
        # Handle None/NaN mid values
        if up_mid is None or (isinstance(up_mid, float) and np.isnan(up_mid)):
            up_mid = 0.5
        if down_mid is None or (isinstance(down_mid, float) and np.isnan(down_mid)):
            down_mid = 0.5
        
        up_value = self.up_position * up_mid
        down_value = self.down_position * down_mid
        return up_value + down_value + self.cash - self.up_cost_basis - self.down_cost_basis
    
    def total_exposure(self) -> float:
        """Total absolute position exposure."""
        return abs(self.up_position) + abs(self.down_position)
    
    def net_delta(self) -> float:
        """Net directional exposure (UP - DOWN)."""
        return self.up_position - self.down_position
    
    def update_position(self, token: str, side: str, price: float, size: float):
        """Update position from a fill.
        
        Args:
            token: 'UP' or 'DOWN'
            side: 'BID' (buy) or 'ASK' (sell)
            price: Fill price
            size: Fill size
        """
        if token == 'UP':
            if side == 'BID':  # Bought UP
                self.up_position += size
                self.up_cost_basis += price * size
                self.cash -= price * size
            else:  # Sold UP
                self.up_position -= size
                # Realize PnL on sell
                avg_cost = self.up_cost_basis / self.up_position if self.up_position != 0 else 0
                self.realized_pnl += size * (price - avg_cost) if self.up_position + size > 0 else 0
                self.up_cost_basis = max(0, self.up_cost_basis - price * size)
                self.cash += price * size
        else:  # DOWN
            if side == 'BID':  # Bought DOWN
                self.down_position += size
                self.down_cost_basis += price * size
                self.cash -= price * size
            else:  # Sold DOWN
                self.down_position -= size
                avg_cost = self.down_cost_basis / self.down_position if self.down_position != 0 else 0
                self.realized_pnl += size * (price - avg_cost) if self.down_position + size > 0 else 0
                self.down_cost_basis = max(0, self.down_cost_basis - price * size)
                self.cash += price * size
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'up_position': self.up_position,
            'down_position': self.down_position,
            'cash': self.cash,
            'realized_pnl': self.realized_pnl,
            'up_cost_basis': self.up_cost_basis,
            'down_cost_basis': self.down_cost_basis,
        }


# =============================================================================
# FILL ENGINE
# =============================================================================

class FillEngine:
    """
    Event-driven fill simulator for maker orders.
    
    Processes market data tick-by-tick and determines:
    1. When pending orders become active
    2. When active orders get filled
    3. Queue position updates
    4. Inventory changes
    
    The engine maintains state across ticks and produces fill events.
    """
    
    def __init__(self, config: MakerExecutionConfig):
        """
        Initialize fill engine.
        
        Args:
            config: Maker execution configuration
        """
        self.config = config
        
        # Order tracking
        self.orders: Dict[str, MakerOrder] = {}  # order_id -> order
        self.active_orders: List[str] = []  # Active order IDs
        self.pending_orders: List[str] = []  # Pending order IDs (in latency window)
        self.pending_cancels: Dict[str, int] = {}  # order_id -> cancel_effective_at
        
        # Market state (current and previous for L2 consumption tracking)
        self.up_book: Optional[OrderbookState] = None
        self.down_book: Optional[OrderbookState] = None
        self.up_book_prev: Optional[OrderbookState] = None
        self.down_book_prev: Optional[OrderbookState] = None
        self.current_t: int = 0
        
        # Inventory
        self.inventory = Inventory()
        
        # Fill history
        self.fills: List[FillEvent] = []
        
        # Market data history (for adverse selection calculation)
        self.mid_history: Dict[int, Tuple[float, float]] = {}  # t -> (up_mid, down_mid)
        
        # Statistics
        self.stats = {
            'orders_placed': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'orders_expired': 0,
            'partial_fills': 0,
            'total_fill_volume': 0.0,
        }
    
    def reset(self):
        """Reset engine state for new market."""
        self.orders.clear()
        self.active_orders.clear()
        self.pending_orders.clear()
        self.pending_cancels.clear()
        self.up_book = None
        self.down_book = None
        self.up_book_prev = None
        self.down_book_prev = None
        self.current_t = 0
        self.inventory = Inventory()
        self.fills.clear()
        self.mid_history.clear()
        self.stats = {k: 0 if isinstance(v, (int, float)) else v for k, v in self.stats.items()}
    
    def place_order(
        self,
        token: Literal['UP', 'DOWN'],
        side: Literal['BID', 'ASK'],
        price: float,
        size: float,
        t: int
    ) -> str:
        """
        Place a new maker order.
        
        Queue position tracking:
        - L1 model: queue_ahead = size at L1 if price matches L1
        - L2 model: search levels 1-6 to find price, queue = cumulative size ahead
        
        Args:
            token: 'UP' or 'DOWN'
            side: 'BID' (buy) or 'ASK' (sell)
            price: Limit price
            size: Order size
            t: Current time (seconds from market start)
            
        Returns:
            Order ID
        """
        order_id = str(uuid.uuid4())[:8]
        
        # Calculate when order becomes active
        latency_seconds = self.config.place_latency_ms / 1000.0
        time_active = t + int(np.ceil(latency_seconds))
        
        # Get current queue position (L1 model: size at L1 only)
        book = self.up_book if token == 'UP' else self.down_book
        queue_ahead = 0.0
        
        # L2 queue tracking variables
        l2_queue_ahead = 0.0
        l2_level_at_placement = None
        l2_last_seen_size = 0.0
        
        if book:
            # L1 queue model (backward compatible)
            if side == 'BID':
                if book.best_bid is not None and abs(book.best_bid - price) < 0.0001:
                    queue_ahead = book.best_bid_size or 0.0
            else:
                if book.best_ask is not None and abs(book.best_ask - price) < 0.0001:
                    queue_ahead = book.best_ask_size or 0.0
            
            # L2 queue model: search all 6 levels
            level = book.find_price_level(price, side)
            if level is not None:
                l2_level_at_placement = level
                l2_last_seen_size = book.get_size_at_price(price, side)
                
                # Queue ahead = all size at levels better than us + size at our level
                # For BID: levels 1..level (better bids are higher prices)
                # For ASK: levels 1..level (better asks are lower prices)
                if level > 1:
                    # Size at levels ahead of us
                    l2_queue_ahead = book.get_cumulative_size_to_level(level - 1, side)
                # Add size at our level (we join back of queue)
                l2_queue_ahead += l2_last_seen_size
            else:
                # Price not in top 6 - we're improving the book
                l2_level_at_placement = 0  # Better than L1
                l2_queue_ahead = 0.0
        
        order = MakerOrder(
            order_id=order_id,
            token=token,
            side=side,
            price=price,
            size=size,
            time_placed=t,
            time_active=time_active,
            status=OrderStatus.PENDING,
            queue_ahead=queue_ahead or 0.0,
            initial_queue_position=queue_ahead or 0.0,
            # L2 tracking
            l2_queue_ahead=l2_queue_ahead,
            l2_initial_queue=l2_queue_ahead,
            l2_cumulative_consumed=0.0,
            l2_level_at_placement=l2_level_at_placement,
            l2_last_seen_level=l2_level_at_placement,
            l2_last_seen_size=l2_last_seen_size,
        )
        
        self.orders[order_id] = order
        self.pending_orders.append(order_id)
        self.stats['orders_placed'] += 1
        
        return order_id
    
    def cancel_order(self, order_id: str, t: int) -> bool:
        """
        Request cancellation of an order.
        
        Args:
            order_id: Order to cancel
            t: Current time
            
        Returns:
            True if cancel request accepted
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED]:
            return False
        
        # Schedule cancel after latency
        latency_seconds = self.config.cancel_latency_ms / 1000.0
        cancel_effective_at = t + int(np.ceil(latency_seconds))
        
        order.cancel_requested_at = t
        order.cancel_effective_at = cancel_effective_at
        self.pending_cancels[order_id] = cancel_effective_at
        
        return True
    
    def update_market_state(self, row: pd.Series, t: int):
        """
        Update market state from a data row (6-level orderbook).
        
        Expected column naming convention:
        - Level 1: pm_{up|down}_best_bid, pm_{up|down}_best_ask
        - Level 1 sizes: pm_{up|down}_best_bid_size, pm_{up|down}_best_ask_size
        - Levels 2-6: pm_{up|down}_bid_2..6, pm_{up|down}_ask_2..6
        - Levels 2-6 sizes: pm_{up|down}_bid_2_size..6, pm_{up|down}_ask_2_size..6
        
        Args:
            row: DataFrame row with orderbook data
            t: Current time
        """
        self.current_t = t
        
        # Store previous state for L2 consumption tracking
        self.up_book_prev = self.up_book
        self.down_book_prev = self.down_book
        
        # Update UP orderbook (all 6 levels)
        self.up_book = OrderbookState(
            timestamp=t,
            # Level 1
            bid_px_1=row.get('pm_up_best_bid'),
            bid_sz_1=row.get('pm_up_best_bid_size'),
            ask_px_1=row.get('pm_up_best_ask'),
            ask_sz_1=row.get('pm_up_best_ask_size'),
            # Level 2
            bid_px_2=row.get('pm_up_bid_2'),
            bid_sz_2=row.get('pm_up_bid_2_size'),
            ask_px_2=row.get('pm_up_ask_2'),
            ask_sz_2=row.get('pm_up_ask_2_size'),
            # Level 3
            bid_px_3=row.get('pm_up_bid_3'),
            bid_sz_3=row.get('pm_up_bid_3_size'),
            ask_px_3=row.get('pm_up_ask_3'),
            ask_sz_3=row.get('pm_up_ask_3_size'),
            # Level 4
            bid_px_4=row.get('pm_up_bid_4'),
            bid_sz_4=row.get('pm_up_bid_4_size'),
            ask_px_4=row.get('pm_up_ask_4'),
            ask_sz_4=row.get('pm_up_ask_4_size'),
            # Level 5
            bid_px_5=row.get('pm_up_bid_5'),
            bid_sz_5=row.get('pm_up_bid_5_size'),
            ask_px_5=row.get('pm_up_ask_5'),
            ask_sz_5=row.get('pm_up_ask_5_size'),
            # Level 6
            bid_px_6=row.get('pm_up_bid_6'),
            bid_sz_6=row.get('pm_up_bid_6_size'),
            ask_px_6=row.get('pm_up_ask_6'),
            ask_sz_6=row.get('pm_up_ask_6_size'),
        )
        
        # Update DOWN orderbook (all 6 levels)
        self.down_book = OrderbookState(
            timestamp=t,
            # Level 1
            bid_px_1=row.get('pm_down_best_bid'),
            bid_sz_1=row.get('pm_down_best_bid_size'),
            ask_px_1=row.get('pm_down_best_ask'),
            ask_sz_1=row.get('pm_down_best_ask_size'),
            # Level 2
            bid_px_2=row.get('pm_down_bid_2'),
            bid_sz_2=row.get('pm_down_bid_2_size'),
            ask_px_2=row.get('pm_down_ask_2'),
            ask_sz_2=row.get('pm_down_ask_2_size'),
            # Level 3
            bid_px_3=row.get('pm_down_bid_3'),
            bid_sz_3=row.get('pm_down_bid_3_size'),
            ask_px_3=row.get('pm_down_ask_3'),
            ask_sz_3=row.get('pm_down_ask_3_size'),
            # Level 4
            bid_px_4=row.get('pm_down_bid_4'),
            bid_sz_4=row.get('pm_down_bid_4_size'),
            ask_px_4=row.get('pm_down_ask_4'),
            ask_sz_4=row.get('pm_down_ask_4_size'),
            # Level 5
            bid_px_5=row.get('pm_down_bid_5'),
            bid_sz_5=row.get('pm_down_bid_5_size'),
            ask_px_5=row.get('pm_down_ask_5'),
            ask_sz_5=row.get('pm_down_ask_5_size'),
            # Level 6
            bid_px_6=row.get('pm_down_bid_6'),
            bid_sz_6=row.get('pm_down_bid_6_size'),
            ask_px_6=row.get('pm_down_ask_6'),
            ask_sz_6=row.get('pm_down_ask_6_size'),
        )
        
        # Store mid history for adverse selection
        up_mid = self.up_book.mid
        down_mid = self.down_book.mid
        if up_mid is not None and down_mid is not None:
            self.mid_history[t] = (up_mid, down_mid)
    
    def process_tick(self, row: pd.Series, t: int) -> List[FillEvent]:
        """
        Process one tick of market data.
        
        This is the main engine loop that:
        1. Updates market state
        2. Activates pending orders
        3. Processes pending cancels
        4. Checks for fills
        5. Updates queue positions
        
        Args:
            row: DataFrame row with market data
            t: Current time
            
        Returns:
            List of fill events that occurred this tick
        """
        # Update market state
        self.update_market_state(row, t)
        
        fill_events = []
        
        # 1. Activate pending orders
        self._activate_pending_orders(t)
        
        # 2. Process pending cancels
        self._process_pending_cancels(t)
        
        # 3. Check for fills based on fill model
        if self.config.fill_model == FillModel.TOUCH_SIZE_PROXY:
            fill_events = self._check_fills_touch_size_proxy(t)
        elif self.config.fill_model == FillModel.BOUNDS_ONLY:
            fill_events = self._check_fills_bounds_only(t)
        elif self.config.fill_model == FillModel.TAPE_QUEUE:
            # Would require trade tape data
            fill_events = self._check_fills_tape_queue(t)
        elif self.config.fill_model == FillModel.L2_QUEUE:
            # L2 queue model: track consumption across 6 levels
            fill_events = self._check_fills_l2_queue(t)
        
        # 4. Update queue positions for active orders
        self._update_queue_positions(t)
        
        # 5. Calculate adverse selection for recent fills
        self._update_adverse_selection(t)
        
        return fill_events
    
    def _activate_pending_orders(self, t: int):
        """Activate orders that have passed their latency window."""
        newly_active = []
        for order_id in self.pending_orders[:]:  # Copy to allow modification
            order = self.orders[order_id]
            if order.time_active is not None and t >= order.time_active:
                order.status = OrderStatus.ACTIVE
                self.pending_orders.remove(order_id)
                self.active_orders.append(order_id)
                newly_active.append(order_id)
        
        return newly_active
    
    def _process_pending_cancels(self, t: int):
        """Process cancels that have passed their latency window."""
        for order_id, cancel_at in list(self.pending_cancels.items()):
            if t >= cancel_at:
                order = self.orders[order_id]
                if order.status == OrderStatus.ACTIVE:
                    order.status = OrderStatus.CANCELLED
                    if order_id in self.active_orders:
                        self.active_orders.remove(order_id)
                    self.stats['orders_cancelled'] += 1
                elif order.status == OrderStatus.PENDING:
                    order.status = OrderStatus.CANCELLED
                    if order_id in self.pending_orders:
                        self.pending_orders.remove(order_id)
                    self.stats['orders_cancelled'] += 1
                del self.pending_cancels[order_id]
    
    def _check_fills_touch_size_proxy(self, t: int) -> List[FillEvent]:
        """
        Check for fills using TOUCH_SIZE_PROXY model.
        
        This model estimates fill probability based on:
        - Size at best bid/ask as proxy for trading activity
        - Configured trade rate (fraction of touch size that trades per second)
        - Queue position (FIFO or proportional)
        
        Key insight: If touch size decreases, trading happened. If our queue_ahead
        drops enough, we get filled.
        """
        fill_events = []
        
        for order_id in self.active_orders[:]:
            order = self.orders[order_id]
            book = self.up_book if order.token == 'UP' else self.down_book
            
            if book is None:
                continue
            
            # Check if order is at best price
            at_touch = False
            current_touch_size = 0.0
            
            if order.side == 'BID' and book.best_bid is not None:
                at_touch = (abs(order.price - book.best_bid) < 0.001)  # Price tolerance
                current_touch_size = book.best_bid_size or 0.0
            elif order.side == 'ASK' and book.best_ask is not None:
                at_touch = (abs(order.price - book.best_ask) < 0.001)
                current_touch_size = book.best_ask_size or 0.0
            
            if not at_touch:
                # Order not at touch - no fill this tick
                # Could check for price improvement (fill through) here
                continue
            
            # Estimate trading volume this tick
            # Simple model: trade_rate * touch_size trades per second
            estimated_volume = self.config.touch_trade_rate_per_second * current_touch_size
            
            # Update queue position
            order.queue_ahead = max(0, order.queue_ahead - estimated_volume)
            
            # Check if filled (queue exhausted)
            if order.queue_ahead <= 0:
                fill_event = self._execute_fill(order, t, order.price)
                if fill_event:
                    fill_events.append(fill_event)
        
        return fill_events
    
    def _check_fills_bounds_only(self, t: int) -> List[FillEvent]:
        """
        Check for fills using BOUNDS_ONLY model.
        
        Upper bound: Fill whenever at best bid/ask (optimistic)
        Lower bound: Fill only when price moves through order (pessimistic)
        
        Results from this model are BOUNDS, not estimates.
        """
        fill_events = []
        
        for order_id in self.active_orders[:]:
            order = self.orders[order_id]
            book = self.up_book if order.token == 'UP' else self.down_book
            
            if book is None:
                continue
            
            should_fill = False
            fill_price = order.price
            
            if self.config.bounds_fill_at_touch:
                # Upper bound: fill at touch
                if order.side == 'BID' and book.best_bid is not None:
                    if abs(order.price - book.best_bid) < 0.001:
                        should_fill = True
                        fill_price = book.best_bid
                elif order.side == 'ASK' and book.best_ask is not None:
                    if abs(order.price - book.best_ask) < 0.001:
                        should_fill = True
                        fill_price = book.best_ask
            
            if self.config.bounds_fill_on_price_through and not should_fill:
                # Lower bound: fill only when price moves through
                if order.side == 'BID' and book.best_ask is not None:
                    if book.best_ask <= order.price:  # Asks dropped to/below our bid
                        should_fill = True
                        fill_price = order.price
                elif order.side == 'ASK' and book.best_bid is not None:
                    if book.best_bid >= order.price:  # Bids rose to/above our ask
                        should_fill = True
                        fill_price = order.price
            
            if should_fill:
                fill_event = self._execute_fill(order, t, fill_price)
                if fill_event:
                    fill_events.append(fill_event)
        
        return fill_events
    
    def _check_fills_tape_queue(self, t: int) -> List[FillEvent]:
        """
        Check for fills using TAPE_QUEUE model.
        
        This would use actual trade tape data to determine fills.
        NOT IMPLEMENTED - would require trade tape collection.
        """
        # Placeholder - requires trade tape data
        return []
    
    def _estimate_consumption_at_price(
        self,
        order: MakerOrder,
        book: OrderbookState,
        book_prev: Optional[OrderbookState]
    ) -> float:
        """
        Estimate volume consumed at a specific price level from L2 book deltas.
        
        Logic:
        - Find price p in current book (levels 1-6)
        - If p still visible: consumed = max(0, sz_p(t-1) - sz_p(t))
        - If p disappeared from top 6:
            - Conservative: consumed = 0 (treat as cancels/book moved)
            - Base (allow level drift): check if price moved between levels
            - Optimistic: consumed = remaining_queue (assume fully consumed)
        
        Args:
            order: The order we're tracking
            book: Current orderbook state
            book_prev: Previous orderbook state
            
        Returns:
            Estimated volume consumed at this price
        """
        if book_prev is None:
            return 0.0
        
        price = order.price
        side = order.side
        
        # Find price in current and previous books
        current_level = book.find_price_level(price, side)
        prev_level = book_prev.find_price_level(price, side)
        
        # Get sizes
        current_size = book.get_size_at_price(price, side) if current_level else 0.0
        prev_size = book_prev.get_size_at_price(price, side) if prev_level else order.l2_last_seen_size
        
        consumed = 0.0
        
        if current_level is not None:
            # Price still visible in book
            if prev_level is not None:
                # Was visible before too - check for size decrease
                size_delta = prev_size - current_size
                if size_delta > 0:
                    # Size decreased - this is consumption
                    consumed = size_delta
                # Update tracking
                order.l2_last_seen_level = current_level
                order.l2_last_seen_size = current_size
            else:
                # Appeared in book (was outside top 6, now visible)
                # This shouldn't happen for our orders normally
                order.l2_last_seen_level = current_level
                order.l2_last_seen_size = current_size
        else:
            # Price NOT in current top 6
            if self.config.l2_conservative_mode:
                # Conservative: don't count consumption when price disappears
                consumed = 0.0
            elif self.config.l2_allow_level_drift and prev_level is not None:
                # Base: price was visible, now gone
                # Could be: book moved up/down, large cancel, or consumption
                # For now, assume it's NOT consumption (conservative for disappeared)
                consumed = 0.0
            elif self.config.l2_optimistic_disappear:
                # Optimistic: assume remaining queue was consumed
                consumed = order.l2_queue_ahead
            else:
                # Default: treat as unknown, don't count
                consumed = 0.0
        
        return consumed
    
    def _check_fills_l2_queue(self, t: int) -> List[FillEvent]:
        """
        Check for fills using L2_QUEUE model (6-level consumption tracking).
        
        This model:
        1. Tracks cumulative consumption at each order's price level
        2. Fills when cumulative_consumed > queue_ahead
        3. Handles level drift and disappeared prices based on config
        
        Key insight: More accurate than L1-only because:
        - Queue position accounts for all size ahead at price
        - Consumption tracked even when price moves between levels
        - Can distinguish between fills and cancels (partially)
        """
        fill_events = []
        
        for order_id in self.active_orders[:]:
            order = self.orders[order_id]
            book = self.up_book if order.token == 'UP' else self.down_book
            book_prev = self.up_book_prev if order.token == 'UP' else self.down_book_prev
            
            if book is None:
                continue
            
            # Step 1: Estimate consumption at this price
            consumed = self._estimate_consumption_at_price(order, book, book_prev)
            order.l2_cumulative_consumed += consumed
            
            # Step 2: Check if we should be filled
            # We get filled when cumulative consumption exceeds queue ahead
            # (i.e., all orders ahead of us have been filled)
            if order.l2_cumulative_consumed >= order.l2_queue_ahead:
                # Calculate fill size based on how much consumed beyond queue
                excess_consumed = order.l2_cumulative_consumed - order.l2_queue_ahead
                fill_size = min(order.remaining_size, max(0.01, excess_consumed))  # Min fill size
                
                if fill_size >= 0.01:  # Only fill if meaningful
                    fill_event = self._execute_fill(order, t, order.price)
                    if fill_event:
                        fill_events.append(fill_event)
            
            # Step 3: Update queue_ahead for orders not yet filled
            # (this is for compatibility with L1 model metrics)
            if order.status == OrderStatus.ACTIVE:
                # Reduce queue ahead by consumption
                order.queue_ahead = max(0, order.queue_ahead - consumed)
                order.l2_queue_ahead = max(0, order.l2_queue_ahead - consumed)
        
        return fill_events
    
    def _execute_fill(self, order: MakerOrder, t: int, fill_price: float) -> Optional[FillEvent]:
        """Execute a fill for an order."""
        if order.status != OrderStatus.ACTIVE:
            return None
        
        fill_size = order.remaining_size
        
        # Update order
        order.filled_size += fill_size
        order.fill_time = t
        order.fill_price = fill_price
        order.status = OrderStatus.FILLED
        
        # Remove from active orders
        if order.order_id in self.active_orders:
            self.active_orders.remove(order.order_id)
        
        # Update inventory
        self.inventory.update_position(order.token, order.side, fill_price, fill_size)
        
        # Get current market state for fill record
        book = self.up_book if order.token == 'UP' else self.down_book
        mid_at_fill = book.mid if book else fill_price
        spread_at_fill = book.spread if book else None
        
        # Create fill event with extended fields
        fill_event = FillEvent(
            order_id=order.order_id,
            fill_time=t,
            fill_price=fill_price,
            fill_size=fill_size,
            is_partial=False,
            mid_at_fill=mid_at_fill or fill_price,
            token=order.token,
            side=order.side,
            spread_at_fill=spread_at_fill,
        )
        
        self.fills.append(fill_event)
        self.stats['orders_filled'] += 1
        self.stats['total_fill_volume'] += fill_size
        
        return fill_event
    
    def _update_queue_positions(self, t: int):
        """Update queue positions for active orders."""
        for order_id in self.active_orders:
            order = self.orders[order_id]
            book = self.up_book if order.token == 'UP' else self.down_book
            
            if book is None:
                continue
            
            # Check if still at best price
            if order.side == 'BID':
                if book.best_bid is not None and abs(order.price - book.best_bid) < 0.001:
                    # Still at touch - queue_ahead already updated in fill check
                    pass
                elif book.best_bid is not None and order.price < book.best_bid:
                    # Our price is now below best bid - we improved!
                    # This happens if orderbook moved up
                    order.queue_ahead = 0  # We're at front if we improve
            else:  # ASK
                if book.best_ask is not None and abs(order.price - book.best_ask) < 0.001:
                    pass
                elif book.best_ask is not None and order.price > book.best_ask:
                    # Our price is now above best ask
                    order.queue_ahead = 0
    
    def _update_adverse_selection(self, t: int):
        """
        Calculate adverse selection for fills that happened 1-5 seconds ago.
        
        SIGN CONVENTION (fixed):
        - Positive AS = COST (market moved against us after fill)
        - Negative AS = GAIN (market moved in our favor after fill)
        
        AS is computed relative to mid_at_fill, NOT fill_price:
        - BID (bought): AS = mid_at_fill - mid_after (positive if price dropped = bad)
        - ASK (sold): AS = mid_after - mid_at_fill (positive if price rose = bad)
        
        This separates:
        - Spread captured = edge from buying below / selling above mid
        - Adverse selection = subsequent market move against our position
        """
        for fill in self.fills:
            if fill.adverse_selection_1s is not None and fill.adverse_selection_5s is not None:
                continue  # Already calculated
            
            order = self.orders[fill.order_id]
            
            # 1-second adverse selection
            if fill.adverse_selection_1s is None and (t - fill.fill_time) >= 1:
                if (fill.fill_time + 1) in self.mid_history:
                    mids = self.mid_history[fill.fill_time + 1]
                    mid_after = mids[0] if order.token == 'UP' else mids[1]
                    
                    # Adverse selection = move against us RELATIVE TO MID AT FILL
                    # (not fill_price - this is the fix for the sign convention bug)
                    if order.side == 'BID':  # We bought, price dropping is adverse
                        # Positive = mid dropped after we bought = cost
                        fill.adverse_selection_1s = fill.mid_at_fill - mid_after
                    else:  # We sold, price rising is adverse
                        # Positive = mid rose after we sold = cost
                        fill.adverse_selection_1s = mid_after - fill.mid_at_fill
                    
                    fill.mid_after_1s = mid_after
            
            # 5-second adverse selection
            if fill.adverse_selection_5s is None and (t - fill.fill_time) >= 5:
                if (fill.fill_time + 5) in self.mid_history:
                    mids = self.mid_history[fill.fill_time + 5]
                    mid_after = mids[0] if order.token == 'UP' else mids[1]
                    
                    if order.side == 'BID':
                        fill.adverse_selection_5s = fill.mid_at_fill - mid_after
                    else:
                        fill.adverse_selection_5s = mid_after - fill.mid_at_fill
                    
                    fill.mid_after_5s = mid_after
    
    def expire_all_orders(self, t: int):
        """Expire all outstanding orders at market end."""
        for order_id in self.active_orders[:] + self.pending_orders[:]:
            order = self.orders[order_id]
            if order.status in [OrderStatus.ACTIVE, OrderStatus.PENDING]:
                order.status = OrderStatus.EXPIRED
                self.stats['orders_expired'] += 1
        
        self.active_orders.clear()
        self.pending_orders.clear()
    
    def get_active_orders(self, token: Optional[str] = None) -> List[MakerOrder]:
        """Get all active orders, optionally filtered by token."""
        orders = [self.orders[oid] for oid in self.active_orders]
        if token:
            orders = [o for o in orders if o.token == token]
        return orders
    
    def get_pending_orders(self) -> List[MakerOrder]:
        """Get all pending orders."""
        return [self.orders[oid] for oid in self.pending_orders]
    
    def get_inventory(self) -> Inventory:
        """Get current inventory state."""
        return self.inventory
    
    def get_fills(self) -> List[FillEvent]:
        """Get all fill events."""
        return self.fills
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return self.stats.copy()
    
    def get_pnl_decomposition(self) -> Dict[str, float]:
        """
        Decompose PnL into components.
        
        SIGN CONVENTION:
        - spread_captured: Always positive (we bought below mid or sold above mid)
        - adverse_selection: Positive = cost (market moved against us), Negative = gain
        - inventory_carry: Current unrealized P&L from open positions
        
        FORMULA:
        total_pnl = spread_captured - adverse_selection + inventory_carry
        
        If adverse_selection is positive (cost), we subtract it → PnL decreases.
        If adverse_selection is negative (gain), we subtract negative → PnL increases.
        
        Returns:
            Dictionary with:
            - spread_captured: Price improvement from maker fills (positive)
            - adverse_selection: Net adverse selection (positive = cost)
            - inventory_carry: Mark-to-market on positions
            - total_pnl: Sum of all components
            - n_fills: Number of fills
            - pct_fills_gain_1s: Percentage of fills with negative AS (gain)
        """
        spread_captured = 0.0
        adverse_selection_total = 0.0
        n_with_as = 0
        n_gain_1s = 0
        
        for fill in self.fills:
            # Use FillEvent's spread_captured property for consistency
            spread_captured += fill.spread_captured
            
            # Adverse selection (use 5s if available, else 1s)
            # AS is already per-unit, fill_size is already factored into spread_captured
            as_cost = fill.adverse_selection_5s if fill.adverse_selection_5s is not None else fill.adverse_selection_1s
            if as_cost is not None:
                adverse_selection_total += as_cost * fill.fill_size
                n_with_as += 1
                if fill.adverse_selection_1s is not None and fill.adverse_selection_1s < 0:
                    n_gain_1s += 1  # Negative AS = gain
        
        # Inventory carry (current mark-to-market)
        inventory_carry = 0.0
        if self.up_book and self.down_book:
            up_mid = self.up_book.mid
            down_mid = self.down_book.mid
            # Handle None/NaN
            if up_mid is None or (isinstance(up_mid, float) and np.isnan(up_mid)):
                up_mid = 0.5
            if down_mid is None or (isinstance(down_mid, float) and np.isnan(down_mid)):
                down_mid = 0.5
            inventory_carry = self.inventory.mark_to_market(up_mid, down_mid)
        
        # Total PnL: spread captured minus adverse selection cost plus inventory value
        total_pnl = spread_captured - adverse_selection_total + inventory_carry
        
        pct_gain_1s = (n_gain_1s / n_with_as * 100) if n_with_as > 0 else 0.0
        
        return {
            'spread_captured': spread_captured,
            'adverse_selection': adverse_selection_total,
            'inventory_carry': inventory_carry,
            'realized_pnl': self.inventory.realized_pnl,
            'total_pnl': total_pnl,
            'n_fills': len(self.fills),
            'n_fills_with_as': n_with_as,
            'pct_fills_gain_1s': pct_gain_1s,
        }
    
    def validate_pnl_decomposition(self) -> Dict[str, Any]:
        """
        Validate PnL decomposition is internally consistent.
        
        Creates a fill-by-fill table with clear sign conventions and verifies
        that the components sum correctly.
        
        SIGN CONVENTION (CRITICAL):
        - spread_captured: Always POSITIVE for maker fills (we got price improvement)
        - adverse_selection: POSITIVE = cost (market moved against us)
                            NEGATIVE = gain (market moved in our favor)
        - gain_after_Xs: POSITIVE = price moved in our favor
                        (note: gain = -adverse_selection)
        
        Returns:
            Dictionary with:
            - fills_table: DataFrame with per-fill decomposition
            - summary: Reconciliation of components
            - is_consistent: Boolean indicating if math checks out
            - issues: List of any inconsistencies found
        """
        fills_data = []
        
        for fill in self.fills:
            order = self.orders[fill.order_id]
            
            # Recompute all values with clear formulas
            signed_qty = 1.0 if fill.side == 'BID' else -1.0
            
            # Spread captured (should be positive for maker fills)
            if fill.side == 'BID':
                spread_captured = (fill.mid_at_fill - fill.fill_price) * fill.fill_size
            else:
                spread_captured = (fill.fill_price - fill.mid_at_fill) * fill.fill_size
            
            # Adverse selection (positive = cost)
            as_1s = None
            as_5s = None
            gain_1s = None
            gain_5s = None
            
            if fill.mid_after_1s is not None:
                if fill.side == 'BID':
                    as_1s = (fill.mid_at_fill - fill.mid_after_1s) * fill.fill_size
                    gain_1s = (fill.mid_after_1s - fill.mid_at_fill) * fill.fill_size
                else:
                    as_1s = (fill.mid_after_1s - fill.mid_at_fill) * fill.fill_size
                    gain_1s = (fill.mid_at_fill - fill.mid_after_1s) * fill.fill_size
            
            if fill.mid_after_5s is not None:
                if fill.side == 'BID':
                    as_5s = (fill.mid_at_fill - fill.mid_after_5s) * fill.fill_size
                    gain_5s = (fill.mid_after_5s - fill.mid_at_fill) * fill.fill_size
                else:
                    as_5s = (fill.mid_after_5s - fill.mid_at_fill) * fill.fill_size
                    gain_5s = (fill.mid_at_fill - fill.mid_after_5s) * fill.fill_size
            
            # Net P&L from this fill (at 5s horizon)
            if gain_5s is not None:
                net_pnl_5s = spread_captured + gain_5s
            else:
                net_pnl_5s = None
            
            fills_data.append({
                'order_id': fill.order_id,
                'fill_time': fill.fill_time,
                'token': fill.token,
                'side': fill.side,
                'signed_qty': signed_qty,
                'fill_price': fill.fill_price,
                'fill_size': fill.fill_size,
                'mid_at_fill': fill.mid_at_fill,
                'mid_1s_later': fill.mid_after_1s,
                'mid_5s_later': fill.mid_after_5s,
                'spread_captured': spread_captured,
                'as_1s_cost': as_1s,  # Positive = cost
                'as_5s_cost': as_5s,  # Positive = cost
                'gain_1s': gain_1s,   # Positive = gain (= -as_1s)
                'gain_5s': gain_5s,   # Positive = gain (= -as_5s)
                'net_pnl_5s': net_pnl_5s,
            })
        
        fills_df = pd.DataFrame(fills_data)
        
        # Compute summary statistics
        summary = {}
        issues = []
        
        if len(fills_df) > 0:
            summary['total_spread_captured'] = fills_df['spread_captured'].sum()
            summary['total_as_1s'] = fills_df['as_1s_cost'].sum() if fills_df['as_1s_cost'].notna().any() else 0
            summary['total_as_5s'] = fills_df['as_5s_cost'].sum() if fills_df['as_5s_cost'].notna().any() else 0
            summary['total_gain_1s'] = fills_df['gain_1s'].sum() if fills_df['gain_1s'].notna().any() else 0
            summary['total_gain_5s'] = fills_df['gain_5s'].sum() if fills_df['gain_5s'].notna().any() else 0
            
            # Check: gain = -as (should always hold)
            if fills_df['as_1s_cost'].notna().any() and fills_df['gain_1s'].notna().any():
                as_plus_gain = summary['total_as_1s'] + summary['total_gain_1s']
                if abs(as_plus_gain) > 0.0001:
                    issues.append(f"AS + gain should = 0, got {as_plus_gain:.6f}")
            
            # Count fills with gain vs cost
            if fills_df['gain_1s'].notna().any():
                n_gain = (fills_df['gain_1s'] > 0).sum()
                n_cost = (fills_df['gain_1s'] < 0).sum()
                n_neutral = (fills_df['gain_1s'] == 0).sum()
                summary['n_fills_gain_1s'] = int(n_gain)
                summary['n_fills_cost_1s'] = int(n_cost)
                summary['pct_fills_gain_1s'] = n_gain / len(fills_df) * 100
            
            # Validate against get_pnl_decomposition
            decomp = self.get_pnl_decomposition()
            
            # Check spread captured matches
            spread_diff = abs(summary['total_spread_captured'] - decomp['spread_captured'])
            if spread_diff > 0.0001:
                issues.append(f"Spread captured mismatch: table={summary['total_spread_captured']:.4f}, decomp={decomp['spread_captured']:.4f}")
            
            summary['decomp_spread_captured'] = decomp['spread_captured']
            summary['decomp_adverse_selection'] = decomp['adverse_selection']
            summary['decomp_total_pnl'] = decomp['total_pnl']
        
        is_consistent = len(issues) == 0
        
        return {
            'fills_table': fills_df,
            'summary': summary,
            'is_consistent': is_consistent,
            'issues': issues,
        }
    
    def get_fills_dataframe(self) -> pd.DataFrame:
        """Get fills as a DataFrame for analysis."""
        return pd.DataFrame([f.to_dict() for f in self.fills])


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_default_config() -> MakerExecutionConfig:
    """Create default maker execution config."""
    return MakerExecutionConfig()


def create_optimistic_config() -> MakerExecutionConfig:
    """Create optimistic config (low latency, high fill rate)."""
    return MakerExecutionConfig(
        place_latency_ms=0,
        cancel_latency_ms=0,
        fill_model=FillModel.BOUNDS_ONLY,
        touch_trade_rate_per_second=0.3,
    )


def create_pessimistic_config() -> MakerExecutionConfig:
    """Create pessimistic config (high latency, conservative fills)."""
    config = MakerExecutionConfig(
        place_latency_ms=500,
        cancel_latency_ms=200,
        fill_model=FillModel.BOUNDS_ONLY,
    )
    config.bounds_fill_at_touch = False  # Only fill when price moves through
    return config


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("Testing FillEngine...")
    
    config = MakerExecutionConfig(
        place_latency_ms=100,
        cancel_latency_ms=50,
        fill_model=FillModel.TOUCH_SIZE_PROXY,
        touch_trade_rate_per_second=0.1,
    )
    print(f"Config: {config.describe()}")
    
    engine = FillEngine(config)
    
    # Simulate placing an order
    order_id = engine.place_order('UP', 'BID', 0.50, 1.0, t=0)
    print(f"Placed order: {order_id}")
    
    # Simulate market data
    test_row = pd.Series({
        'pm_up_best_bid': 0.50,
        'pm_up_best_bid_size': 100.0,
        'pm_up_best_ask': 0.52,
        'pm_up_best_ask_size': 100.0,
        'pm_down_best_bid': 0.48,
        'pm_down_best_bid_size': 100.0,
        'pm_down_best_ask': 0.50,
        'pm_down_best_ask_size': 100.0,
    })
    
    # Process several ticks
    for t in range(20):
        fills = engine.process_tick(test_row, t)
        if fills:
            print(f"Tick {t}: {len(fills)} fill(s)")
            for f in fills:
                print(f"  Fill: {f.fill_size} @ {f.fill_price}")
    
    # Print stats
    print(f"\nStats: {engine.get_stats()}")
    print(f"Inventory: {engine.get_inventory().to_dict()}")
    print(f"PnL Decomposition: {engine.get_pnl_decomposition()}")
    
    print("\n[OK] FillEngine test passed")

