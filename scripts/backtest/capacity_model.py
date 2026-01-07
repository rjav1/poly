#!/usr/bin/env python3
"""
Capacity Model for Depth-Aware Strategy Analysis

This module computes:
1. Maximum tradable size (q*) per signal where edge remains positive
2. Capacity curves (PnL vs size)
3. Signal filtering by capacity constraints

The capacity model answers: "How much can I trade before my edge disappears?"
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json

try:
    from .execution_model import (
        walk_the_book, 
        get_effective_prices_with_depth,
        get_effective_prices,
        DepthFillResult,
        DepthEffectivePrices,
    )
    from .strategies import Signal
except ImportError:
    from execution_model import (
        walk_the_book,
        get_effective_prices_with_depth,
        get_effective_prices,
        DepthFillResult,
        DepthEffectivePrices,
    )
    from strategies import Signal


@dataclass
class CapacityResult:
    """Result of capacity analysis for a single signal."""
    signal: Signal
    
    # Maximum tradable size
    q_star: float              # Max size where edge > 0
    q_star_edge: float         # Edge at q_star
    
    # Fill info at q_star
    entry_price_at_qstar: float
    levels_used_at_qstar: int
    is_complete_at_qstar: bool
    
    # Total available depth
    total_depth: float
    
    # Edge breakdown
    fair_value: float          # p_hat from model
    l1_entry_price: float      # Entry at L1
    l1_edge: float             # Edge at L1 (size=1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'market_id': self.signal.market_id,
            'entry_t': self.signal.entry_t,
            'side': self.signal.side,
            'q_star': self.q_star,
            'q_star_edge': self.q_star_edge,
            'entry_price_at_qstar': self.entry_price_at_qstar,
            'levels_used_at_qstar': self.levels_used_at_qstar,
            'is_complete_at_qstar': self.is_complete_at_qstar,
            'total_depth': self.total_depth,
            'fair_value': self.fair_value,
            'l1_entry_price': self.l1_entry_price,
            'l1_edge': self.l1_edge,
        }


@dataclass
class CapacityCurvePoint:
    """Single point on a capacity curve."""
    size: float
    entry_price: float
    edge: float
    expected_pnl: float
    is_complete: bool
    levels_used: int


def compute_edge(
    p_hat: float,
    entry_price: float,
    side: str,
    buffer: float = 0.0
) -> float:
    """
    Compute expected edge for a trade.
    
    For buy_up: edge = p_hat - entry_price - buffer
    For buy_down: edge = (1 - p_hat) - entry_price - buffer
    
    Args:
        p_hat: Model's fair probability of UP outcome
        entry_price: Entry price (after walking book)
        side: 'buy_up' or 'buy_down'
        buffer: Execution buffer to subtract
        
    Returns:
        Expected edge (positive = profitable)
    """
    if pd.isna(p_hat) or pd.isna(entry_price):
        return np.nan
    
    if side == 'buy_up':
        # Expected value = p_hat * 1 + (1-p_hat) * 0 = p_hat
        return p_hat - entry_price - buffer
    elif side == 'buy_down':
        # Expected value = (1-p_hat) * 1 + p_hat * 0 = 1-p_hat
        return (1 - p_hat) - entry_price - buffer
    else:
        return np.nan


def compute_max_tradable_size(
    row: pd.Series,
    signal: Signal,
    p_hat: float,
    buffer: float = 0.0,
    size_grid: List[float] = None,
    min_edge: float = 0.0
) -> CapacityResult:
    """
    Compute maximum tradable size (q*) where edge remains positive.
    
    Args:
        row: DataFrame row with L6 orderbook data at signal entry time
        signal: Signal object with side, entry_t, etc.
        p_hat: Model's fair probability of UP outcome
        buffer: Execution buffer to subtract from edge
        size_grid: Grid of sizes to test (default: 1, 5, 10, 25, 50, 100, 200, 500)
        min_edge: Minimum edge to consider profitable (default: 0)
        
    Returns:
        CapacityResult with q_star and related info
    """
    if size_grid is None:
        size_grid = [1, 5, 10, 25, 50, 100, 200, 500, 1000]
    
    # Get L1 price for comparison
    l1_prices = get_effective_prices(row)
    if signal.side == 'buy_up':
        l1_entry = l1_prices.buy_up
    elif signal.side == 'buy_down':
        l1_entry = l1_prices.buy_down
    else:
        l1_entry = np.nan
    
    l1_edge = compute_edge(p_hat, l1_entry, signal.side, buffer)
    
    # Find q_star by testing each size
    q_star = 0
    q_star_edge = 0
    entry_price_at_qstar = l1_entry
    levels_used_at_qstar = 1
    is_complete_at_qstar = True
    total_depth = 0
    
    for size in size_grid:
        prices = get_effective_prices_with_depth(row, size)
        
        if signal.side == 'buy_up':
            entry_price = prices.buy_up_vwap
            is_complete = prices.buy_up_complete
            levels_used = prices.buy_up_levels
        elif signal.side == 'buy_down':
            entry_price = prices.buy_down_vwap
            is_complete = prices.buy_down_complete
            levels_used = prices.buy_down_levels
        else:
            continue
        
        if pd.isna(entry_price):
            break
        
        edge = compute_edge(p_hat, entry_price, signal.side, buffer)
        
        if edge > min_edge:
            q_star = size
            q_star_edge = edge
            entry_price_at_qstar = entry_price
            levels_used_at_qstar = levels_used
            is_complete_at_qstar = is_complete
        else:
            # Edge went negative, stop here
            break
    
    # Get total depth
    if signal.side == 'buy_up':
        fill_result = walk_the_book(row, 'buy', 'up', float('inf'))
        total_depth = fill_result.total_depth_available
    elif signal.side == 'buy_down':
        fill_result = walk_the_book(row, 'buy', 'down', float('inf'))
        total_depth = fill_result.total_depth_available
    
    return CapacityResult(
        signal=signal,
        q_star=q_star,
        q_star_edge=q_star_edge,
        entry_price_at_qstar=entry_price_at_qstar,
        levels_used_at_qstar=levels_used_at_qstar,
        is_complete_at_qstar=is_complete_at_qstar,
        total_depth=total_depth,
        fair_value=p_hat,
        l1_entry_price=l1_entry,
        l1_edge=l1_edge,
    )


def compute_capacity_curve(
    row: pd.Series,
    signal: Signal,
    p_hat: float,
    buffer: float = 0.0,
    size_grid: List[float] = None
) -> List[CapacityCurvePoint]:
    """
    Compute PnL vs size curve for a signal.
    
    Args:
        row: DataFrame row with L6 orderbook data
        signal: Signal object
        p_hat: Model's fair probability
        buffer: Execution buffer
        size_grid: Sizes to evaluate
        
    Returns:
        List of CapacityCurvePoint for each size
    """
    if size_grid is None:
        size_grid = [1, 5, 10, 25, 50, 100, 200, 500]
    
    curve = []
    
    for size in size_grid:
        prices = get_effective_prices_with_depth(row, size)
        
        if signal.side == 'buy_up':
            entry_price = prices.buy_up_vwap
            is_complete = prices.buy_up_complete
            levels_used = prices.buy_up_levels
        elif signal.side == 'buy_down':
            entry_price = prices.buy_down_vwap
            is_complete = prices.buy_down_complete
            levels_used = prices.buy_down_levels
        else:
            continue
        
        if pd.isna(entry_price):
            # Can't fill at this size
            curve.append(CapacityCurvePoint(
                size=size,
                entry_price=np.nan,
                edge=np.nan,
                expected_pnl=np.nan,
                is_complete=False,
                levels_used=0,
            ))
            continue
        
        edge = compute_edge(p_hat, entry_price, signal.side, buffer)
        expected_pnl = edge * size if not pd.isna(edge) else np.nan
        
        curve.append(CapacityCurvePoint(
            size=size,
            entry_price=entry_price,
            edge=edge,
            expected_pnl=expected_pnl,
            is_complete=is_complete,
            levels_used=levels_used,
        ))
    
    return curve


def filter_signals_by_capacity(
    signals: List[Signal],
    capacity_results: List[CapacityResult],
    min_q: float = 10
) -> Tuple[List[Signal], List[CapacityResult]]:
    """
    Filter signals by minimum capacity requirement.
    
    Args:
        signals: List of Signal objects
        capacity_results: Corresponding CapacityResult for each signal
        min_q: Minimum q_star required
        
    Returns:
        (filtered_signals, filtered_results)
    """
    filtered_signals = []
    filtered_results = []
    
    for signal, result in zip(signals, capacity_results):
        if result.q_star >= min_q:
            filtered_signals.append(signal)
            filtered_results.append(result)
    
    return filtered_signals, filtered_results


def run_capacity_analysis(
    df: pd.DataFrame,
    signals: List[Signal],
    model,
    buffer: float = 0.02,
    size_grid: List[float] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run capacity analysis on all signals.
    
    Args:
        df: DataFrame with L6 orderbook data
        signals: List of Signal objects
        model: Fair value model (with predict method)
        buffer: Execution buffer
        size_grid: Sizes to test
        verbose: Print progress
        
    Returns:
        Dict with capacity analysis results
    """
    if size_grid is None:
        size_grid = [1, 5, 10, 25, 50, 100, 200, 500]
    
    if verbose:
        print(f"\n{'='*70}")
        print("CAPACITY ANALYSIS")
        print(f"{'='*70}")
        print(f"Signals: {len(signals)}")
        print(f"Size grid: {size_grid}")
    
    capacity_results = []
    capacity_curves = []
    
    for i, signal in enumerate(signals):
        # Get row at signal entry time
        market_df = df[df['market_id'] == signal.market_id]
        entry_rows = market_df[market_df['t'] == signal.entry_t]
        
        if entry_rows.empty:
            continue
        
        row = entry_rows.iloc[0]
        
        # Get p_hat from model
        p_hat = model.predict(entry_rows)[0]
        
        # Compute capacity
        result = compute_max_tradable_size(
            row, signal, p_hat, buffer, size_grid
        )
        capacity_results.append(result)
        
        # Compute capacity curve
        curve = compute_capacity_curve(row, signal, p_hat, buffer, size_grid)
        capacity_curves.append(curve)
    
    # Compute statistics
    q_stars = [r.q_star for r in capacity_results]
    l1_edges = [r.l1_edge for r in capacity_results if not pd.isna(r.l1_edge)]
    total_depths = [r.total_depth for r in capacity_results]
    
    # Survival rates at different sizes
    survival_rates = {}
    for min_q in [5, 10, 25, 50, 100]:
        n_surviving = sum(1 for q in q_stars if q >= min_q)
        survival_rates[min_q] = n_surviving / len(q_stars) * 100 if q_stars else 0
    
    # Average PnL at different sizes
    avg_pnl_by_size = {}
    for size_idx, size in enumerate(size_grid):
        pnls = []
        for curve in capacity_curves:
            if size_idx < len(curve) and not pd.isna(curve[size_idx].expected_pnl):
                pnls.append(curve[size_idx].expected_pnl)
        avg_pnl_by_size[size] = np.mean(pnls) if pnls else np.nan
    
    results = {
        'n_signals': len(signals),
        'n_analyzed': len(capacity_results),
        'q_star_distribution': {
            'mean': np.mean(q_stars) if q_stars else 0,
            'median': np.median(q_stars) if q_stars else 0,
            'p25': np.percentile(q_stars, 25) if q_stars else 0,
            'p75': np.percentile(q_stars, 75) if q_stars else 0,
            'min': min(q_stars) if q_stars else 0,
            'max': max(q_stars) if q_stars else 0,
        },
        'l1_edge_distribution': {
            'mean': np.mean(l1_edges) if l1_edges else 0,
            'median': np.median(l1_edges) if l1_edges else 0,
        },
        'total_depth_distribution': {
            'mean': np.mean(total_depths) if total_depths else 0,
            'median': np.median(total_depths) if total_depths else 0,
        },
        'survival_rates': survival_rates,
        'avg_pnl_by_size': avg_pnl_by_size,
        'capacity_results': [r.to_dict() for r in capacity_results],
    }
    
    if verbose:
        print(f"\nResults:")
        print(f"  Signals analyzed: {results['n_analyzed']}")
        print(f"\nq* distribution:")
        qs = results['q_star_distribution']
        print(f"  Mean: {qs['mean']:.1f}")
        print(f"  Median: {qs['median']:.1f}")
        print(f"  25th-75th: [{qs['p25']:.1f}, {qs['p75']:.1f}]")
        print(f"  Min-Max: [{qs['min']:.1f}, {qs['max']:.1f}]")
        print(f"\nSurvival rates:")
        for min_q, rate in survival_rates.items():
            print(f"  q* >= {min_q}: {rate:.1f}%")
        print(f"\nAverage PnL by size:")
        for size, pnl in avg_pnl_by_size.items():
            pnl_str = f"${pnl:.2f}" if not pd.isna(pnl) else "N/A"
            print(f"  Size {size}: {pnl_str}")
    
    return results


def main():
    """Test capacity model."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from scripts.backtest.data_loader import (
        load_unified_eth_markets,
        add_derived_columns,
        get_train_test_split
    )
    from scripts.backtest.fair_value import BinnedFairValueModel
    from scripts.backtest.strategies import MispricingBasedStrategy
    
    print("Loading unified dataset (L1 + L6)...")
    df, market_info = load_unified_eth_markets(min_coverage=90.0, prefer_6levels=True)
    df = add_derived_columns(df)
    
    # Split using get_train_test_split if market_order exists, otherwise chronological
    if 'market_order' in df.columns:
        train_df, test_df, _, _ = get_train_test_split(df, train_frac=0.7)
    else:
        # Fallback: chronological split
        markets = sorted(df['market_id'].unique())
        n_train = int(len(markets) * 0.7)
        train_markets = markets[:n_train]
        test_markets = markets[n_train:]
        train_df = df[df['market_id'].isin(train_markets)].copy()
        test_df = df[df['market_id'].isin(test_markets)].copy()
    print(f"Train: {train_df['market_id'].nunique()} markets, Test: {test_df['market_id'].nunique()} markets")
    
    # Train model
    print("\nTraining fair value model...")
    model = BinnedFairValueModel(sample_every=5)
    model.fit(train_df)
    
    # Generate signals
    print("\nGenerating signals...")
    strategy = MispricingBasedStrategy(
        fair_value_model=model,
        buffer=0.02,
        tau_max=420,
        exit_rule='expiry'
    )
    
    all_signals = []
    for market_id in test_df['market_id'].unique()[:5]:  # Just 5 markets for test
        market_df = test_df[test_df['market_id'] == market_id]
        signals = strategy.generate_signals(market_df)
        all_signals.extend(signals)
    
    print(f"Generated {len(all_signals)} signals")
    
    if all_signals:
        # Run capacity analysis
        results = run_capacity_analysis(
            test_df, all_signals, model, buffer=0.02
        )
        
        print("\n[OK] Capacity analysis complete!")


if __name__ == '__main__':
    main()

