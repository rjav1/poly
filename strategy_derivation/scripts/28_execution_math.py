#!/usr/bin/env python3
"""
Phase 2: Depth-Aware Execution Math

Provides core functions for computing executable edge using 6-level orderbook depth:

1. vwap_from_ladder(level_prices, level_sizes, q)
   - Compute VWAP price to buy q contracts walking up the ladder

2. set_edge_from_books(book_up_asks, book_down_asks, q, slippage_buffer)
   - Compute edge for complete-set arb at quantity q

3. max_executable_size(book_up_asks, book_down_asks, epsilon, slippage_buffer)
   - Find maximum executable size that maintains positive edge

These functions are used by the depth-aware backtest and shadow trader.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class BookLadder:
    """6-level orderbook ladder for one side (ask or bid)."""
    prices: List[float]  # Level 1-6 prices
    sizes: List[float]   # Level 1-6 sizes
    
    def __post_init__(self):
        """Clean NaN values to 0."""
        self.prices = [p if not np.isnan(p) else 0.0 for p in self.prices]
        self.sizes = [s if not np.isnan(s) else 0.0 for s in self.sizes]
    
    @property
    def total_size(self) -> float:
        """Total size across all levels."""
        return sum(self.sizes)
    
    @property
    def best_price(self) -> float:
        """Best price (level 1)."""
        return self.prices[0] if self.prices else 0.0
    
    def is_valid(self) -> bool:
        """Check if ladder has at least one valid level."""
        return self.best_price > 0 and len(self.prices) > 0


def vwap_from_ladder(prices: List[float], sizes: List[float], q: float) -> Tuple[float, float, bool]:
    """
    Compute VWAP price to buy q contracts by walking up the ladder.
    
    Args:
        prices: List of prices for levels 1-6
        sizes: List of sizes for levels 1-6
        q: Desired quantity to fill
    
    Returns:
        (vwap_price, filled_qty, fully_filled)
        - vwap_price: Volume-weighted average price for the fill
        - filled_qty: Actual quantity filled (may be < q if not enough depth)
        - fully_filled: True if we could fill the entire quantity
    
    Example:
        prices = [0.45, 0.46, 0.47]
        sizes = [10, 20, 15]
        
        vwap_from_ladder(prices, sizes, 25) would:
        - Take 10 @ 0.45 = 4.50
        - Take 15 @ 0.46 = 6.90
        - Total: 25 contracts, cost = 11.40, VWAP = 0.456
    """
    if q <= 0:
        return 0.0, 0.0, True
    
    total_cost = 0.0
    filled = 0.0
    
    for price, size in zip(prices, sizes):
        if price <= 0 or size <= 0:
            continue
        
        remaining = q - filled
        if remaining <= 0:
            break
        
        fill_at_level = min(size, remaining)
        total_cost += price * fill_at_level
        filled += fill_at_level
    
    if filled <= 0:
        return 0.0, 0.0, False
    
    vwap = total_cost / filled
    fully_filled = filled >= q - 1e-9  # Small tolerance for float comparison
    
    return vwap, filled, fully_filled


def set_cost_from_books(
    up_asks: BookLadder,
    down_asks: BookLadder,
    q: float
) -> Tuple[float, float, bool, bool]:
    """
    Compute total cost to buy a complete set (UP + DOWN) at quantity q.
    
    Args:
        up_asks: Ask ladder for UP token
        down_asks: Ask ladder for DOWN token
        q: Quantity of complete sets to buy
    
    Returns:
        (total_cost, q_filled, up_fully_filled, down_fully_filled)
        - total_cost: VWAP_up(q) + VWAP_down(q) per set
        - q_filled: Minimum of UP and DOWN filled quantities
        - up_fully_filled: True if UP side fully filled
        - down_fully_filled: True if DOWN side fully filled
    """
    vwap_up, filled_up, up_full = vwap_from_ladder(up_asks.prices, up_asks.sizes, q)
    vwap_down, filled_down, down_full = vwap_from_ladder(down_asks.prices, down_asks.sizes, q)
    
    if filled_up <= 0 or filled_down <= 0:
        return 0.0, 0.0, False, False
    
    # Total cost per set is VWAP_up + VWAP_down
    total_cost = vwap_up + vwap_down
    
    # Actual fillable quantity is minimum of both sides
    q_filled = min(filled_up, filled_down)
    
    return total_cost, q_filled, up_full, down_full


def set_edge_from_books(
    up_asks: BookLadder,
    down_asks: BookLadder,
    q: float,
    slippage_buffer: float = 0.005
) -> Tuple[float, float, bool]:
    """
    Compute edge for complete-set arbitrage at quantity q.
    
    Args:
        up_asks: Ask ladder for UP token
        down_asks: Ask ladder for DOWN token
        q: Quantity of complete sets to buy
        slippage_buffer: Additional buffer to subtract from edge (default 0.5c per leg = 1c total)
    
    Returns:
        (edge, q_fillable, executable)
        - edge: 1 - set_cost - slippage_buffer (per set)
        - q_fillable: Maximum quantity that can actually be filled
        - executable: True if edge > 0 and both sides can fill
    
    Complete-set arb pays $1 at expiry regardless of outcome.
    Edge = $1 - cost_to_acquire_set - slippage_buffer
    """
    if not up_asks.is_valid() or not down_asks.is_valid():
        return 0.0, 0.0, False
    
    set_cost, q_filled, up_full, down_full = set_cost_from_books(up_asks, down_asks, q)
    
    if set_cost <= 0 or q_filled <= 0:
        return 0.0, 0.0, False
    
    # Edge = guaranteed payout - cost - buffer
    edge = 1.0 - set_cost - slippage_buffer
    
    # Executable if edge > 0 and we can fill
    executable = edge > 0 and q_filled >= min(1.0, q)
    
    return edge, q_filled, executable


def max_executable_size(
    up_asks: BookLadder,
    down_asks: BookLadder,
    epsilon: float = 0.005,
    slippage_buffer: float = 0.005,
    max_q: float = 100.0,
    step: float = 1.0
) -> Tuple[float, float]:
    """
    Find maximum executable size that maintains positive edge.
    
    Args:
        up_asks: Ask ladder for UP token
        down_asks: Ask ladder for DOWN token
        epsilon: Minimum required edge (default 0.5c = 0.005)
        slippage_buffer: Slippage buffer (default 0.5c = 0.005)
        max_q: Maximum quantity to check (default 100)
        step: Step size for search (default 1 contract)
    
    Returns:
        (q_max, edge_at_q_max)
        - q_max: Maximum executable quantity
        - edge_at_q_max: Edge at that quantity
    
    Algorithm:
        Binary search to find largest q where edge >= epsilon
    """
    if not up_asks.is_valid() or not down_asks.is_valid():
        return 0.0, 0.0
    
    # First check if ANY size is executable
    edge_at_1, _, executable = set_edge_from_books(up_asks, down_asks, 1.0, slippage_buffer)
    if not executable or edge_at_1 < epsilon:
        return 0.0, 0.0
    
    # Binary search for max executable size
    low, high = 1.0, max_q
    best_q = 1.0
    best_edge = edge_at_1
    
    while high - low > step:
        mid = (low + high) / 2
        edge, q_filled, executable = set_edge_from_books(up_asks, down_asks, mid, slippage_buffer)
        
        if executable and edge >= epsilon and q_filled >= mid - 0.1:
            # Can do this size, try larger
            best_q = mid
            best_edge = edge
            low = mid
        else:
            # Too large, try smaller
            high = mid
    
    # Final check at best_q
    edge, q_filled, executable = set_edge_from_books(up_asks, down_asks, best_q, slippage_buffer)
    
    return best_q, edge


def compute_pnl_at_size(
    up_asks: BookLadder,
    down_asks: BookLadder,
    q: float,
    slippage_buffer: float = 0.005
) -> Dict:
    """
    Compute detailed PnL metrics at a specific size.
    
    Args:
        up_asks: Ask ladder for UP token
        down_asks: Ask ladder for DOWN token
        q: Quantity
        slippage_buffer: Slippage buffer
    
    Returns:
        Dict with:
        - q_requested: Requested quantity
        - q_filled: Actual fillable quantity
        - vwap_up: VWAP for UP leg
        - vwap_down: VWAP for DOWN leg
        - set_cost: Total cost per set
        - edge_per_set: Edge per set
        - total_pnl: Total PnL (edge_per_set * q_filled)
        - executable: Whether this is executable
    """
    vwap_up, filled_up, up_full = vwap_from_ladder(up_asks.prices, up_asks.sizes, q)
    vwap_down, filled_down, down_full = vwap_from_ladder(down_asks.prices, down_asks.sizes, q)
    
    q_filled = min(filled_up, filled_down)
    set_cost = vwap_up + vwap_down if q_filled > 0 else 0.0
    edge_per_set = 1.0 - set_cost - slippage_buffer if set_cost > 0 else 0.0
    total_pnl = edge_per_set * q_filled
    executable = edge_per_set > 0 and q_filled >= min(1.0, q)
    
    return {
        'q_requested': q,
        'q_filled': q_filled,
        'vwap_up': vwap_up,
        'vwap_down': vwap_down,
        'set_cost': set_cost,
        'edge_per_set': edge_per_set,
        'total_pnl': total_pnl,
        'executable': executable,
        'up_fully_filled': up_full,
        'down_fully_filled': down_full
    }


def extract_ladder_from_row(row: dict, token: str, side: str) -> BookLadder:
    """
    Extract a BookLadder from a market data row.
    
    Args:
        row: Dict or pandas Series with market data
        token: 'up' or 'down'
        side: 'ask' or 'bid'
    
    Returns:
        BookLadder with 6 levels
    """
    if side == 'ask':
        price_cols = [
            f'{token}_best_ask',
            f'{token}_ask_2', f'{token}_ask_3', 
            f'{token}_ask_4', f'{token}_ask_5', f'{token}_ask_6'
        ]
        size_cols = [
            f'{token}_best_ask_size',
            f'{token}_ask_2_size', f'{token}_ask_3_size',
            f'{token}_ask_4_size', f'{token}_ask_5_size', f'{token}_ask_6_size'
        ]
    else:
        price_cols = [
            f'{token}_best_bid',
            f'{token}_bid_2', f'{token}_bid_3',
            f'{token}_bid_4', f'{token}_bid_5', f'{token}_bid_6'
        ]
        size_cols = [
            f'{token}_best_bid_size',
            f'{token}_bid_2_size', f'{token}_bid_3_size',
            f'{token}_bid_4_size', f'{token}_bid_5_size', f'{token}_bid_6_size'
        ]
    
    prices = []
    sizes = []
    
    for p_col, s_col in zip(price_cols, size_cols):
        price = row.get(p_col, np.nan)
        size = row.get(s_col, np.nan)
        
        prices.append(float(price) if not np.isnan(price) else 0.0)
        sizes.append(float(size) if not np.isnan(size) else 0.0)
    
    return BookLadder(prices=prices, sizes=sizes)


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_vwap_simple():
    """Test VWAP calculation with simple ladder."""
    prices = [0.45, 0.46, 0.47]
    sizes = [10.0, 20.0, 15.0]
    
    # Fill 10 at first level only
    vwap, filled, full = vwap_from_ladder(prices, sizes, 10.0)
    assert abs(vwap - 0.45) < 0.001, f"Expected 0.45, got {vwap}"
    assert abs(filled - 10.0) < 0.001, f"Expected 10, got {filled}"
    assert full, "Should be fully filled"
    
    # Fill 25 across two levels
    # 10 @ 0.45 + 15 @ 0.46 = 4.50 + 6.90 = 11.40 / 25 = 0.456
    vwap, filled, full = vwap_from_ladder(prices, sizes, 25.0)
    expected_vwap = (10 * 0.45 + 15 * 0.46) / 25
    assert abs(vwap - expected_vwap) < 0.001, f"Expected {expected_vwap}, got {vwap}"
    assert abs(filled - 25.0) < 0.001, f"Expected 25, got {filled}"
    assert full, "Should be fully filled"
    
    # Fill 50 (more than available)
    vwap, filled, full = vwap_from_ladder(prices, sizes, 50.0)
    assert abs(filled - 45.0) < 0.001, f"Expected 45 (total), got {filled}"
    assert not full, "Should NOT be fully filled"
    
    print("  test_vwap_simple: PASSED")


def test_set_edge():
    """Test set edge calculation."""
    # UP asks: 0.45, 0.46, 0.47 with sizes 10, 20, 15
    # DOWN asks: 0.50, 0.51, 0.52 with sizes 10, 20, 15
    up_asks = BookLadder(prices=[0.45, 0.46, 0.47], sizes=[10.0, 20.0, 15.0])
    down_asks = BookLadder(prices=[0.50, 0.51, 0.52], sizes=[10.0, 20.0, 15.0])
    
    # At q=1, cost = 0.45 + 0.50 = 0.95, edge = 1 - 0.95 - 0.005 = 0.045
    edge, q_filled, executable = set_edge_from_books(up_asks, down_asks, 1.0, 0.005)
    expected_edge = 1.0 - 0.45 - 0.50 - 0.005
    assert abs(edge - expected_edge) < 0.001, f"Expected {expected_edge}, got {edge}"
    assert executable, "Should be executable"
    
    # At q=10, cost = 0.45 + 0.50 = 0.95, edge = 0.045 (same since all at best)
    edge, q_filled, executable = set_edge_from_books(up_asks, down_asks, 10.0, 0.005)
    assert executable, "Should be executable"
    
    print("  test_set_edge: PASSED")


def test_max_executable():
    """Test max executable size finding."""
    # Ladder with decreasing edge as size increases
    up_asks = BookLadder(prices=[0.45, 0.48, 0.55], sizes=[5.0, 5.0, 5.0])
    down_asks = BookLadder(prices=[0.50, 0.52, 0.55], sizes=[5.0, 5.0, 5.0])
    
    # At small sizes, should have positive edge
    # At large sizes, edge becomes negative due to slippage
    q_max, edge = max_executable_size(up_asks, down_asks, epsilon=0.01, slippage_buffer=0.005)
    
    assert q_max >= 1.0, f"Should have at least 1 contract executable, got {q_max}"
    assert edge >= 0.01, f"Edge should be >= epsilon, got {edge}"
    
    print("  test_max_executable: PASSED")


def run_tests():
    """Run all unit tests."""
    print("\nRunning unit tests...")
    test_vwap_simple()
    test_set_edge()
    test_max_executable()
    print("\nAll tests passed!")


if __name__ == "__main__":
    run_tests()

