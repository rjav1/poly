"""
Maker Strategy Diagnostics

Diagnostic functions for analyzing maker/spread capture strategy performance:
- Fill rate analysis by tau and spread
- Time-to-fill distribution
- PnL vs quote staleness
- Adverse selection decomposition
- Quote update rate analysis

These diagnostics help understand:
1. Where fills are happening (time/spread buckets)
2. Why PnL is positive/negative (spread vs adverse selection)
3. Whether the strategy is sensitive to timing parameters
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from .maker_execution import FillEvent, MakerOrder
except ImportError:
    from maker_execution import FillEvent, MakerOrder


@dataclass
class FillAnalysis:
    """Analysis of fills for a maker strategy."""
    total_fills: int
    fill_rate: float  # fills / orders placed
    avg_time_to_fill: float
    median_time_to_fill: float
    fill_rate_by_tau_bucket: Dict[str, float]
    fill_rate_by_spread_bucket: Dict[str, float]
    avg_spread_at_fill: float
    avg_adverse_selection_1s: float
    avg_adverse_selection_5s: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_fills': self.total_fills,
            'fill_rate': self.fill_rate,
            'avg_time_to_fill': self.avg_time_to_fill,
            'median_time_to_fill': self.median_time_to_fill,
            'fill_rate_by_tau_bucket': self.fill_rate_by_tau_bucket,
            'fill_rate_by_spread_bucket': self.fill_rate_by_spread_bucket,
            'avg_spread_at_fill': self.avg_spread_at_fill,
            'avg_adverse_selection_1s': self.avg_adverse_selection_1s,
            'avg_adverse_selection_5s': self.avg_adverse_selection_5s,
        }


def analyze_fills(
    fills: List[Dict[str, Any]],
    orders_placed: int,
    market_data: pd.DataFrame,
) -> FillAnalysis:
    """
    Analyze fill patterns and quality.
    
    Args:
        fills: List of fill dictionaries (from backtest result)
        orders_placed: Total orders placed
        market_data: DataFrame with market data for context
        
    Returns:
        FillAnalysis with detailed breakdown
    """
    if not fills:
        return FillAnalysis(
            total_fills=0,
            fill_rate=0.0,
            avg_time_to_fill=0.0,
            median_time_to_fill=0.0,
            fill_rate_by_tau_bucket={},
            fill_rate_by_spread_bucket={},
            avg_spread_at_fill=0.0,
            avg_adverse_selection_1s=0.0,
            avg_adverse_selection_5s=0.0,
        )
    
    fills_df = pd.DataFrame(fills)
    
    # Basic stats
    total_fills = len(fills_df)
    fill_rate = total_fills / orders_placed if orders_placed > 0 else 0.0
    
    # Time to fill stats
    # Note: This requires joining with order data to get time_placed
    # For now, use fill_time as proxy if time data not available
    avg_time_to_fill = 0.0
    median_time_to_fill = 0.0
    
    # Merge with market data to get tau at fill time
    if 'fill_time' in fills_df.columns and 't' in market_data.columns:
        # Create tau lookup
        tau_lookup = market_data.set_index('t')['tau'].to_dict()
        fills_df['tau_at_fill'] = fills_df['fill_time'].map(tau_lookup)
        
        # Create spread lookup (approximate - use up_spread)
        if 'pm_up_spread' in market_data.columns:
            spread_lookup = market_data.set_index('t')['pm_up_spread'].to_dict()
            fills_df['spread_at_fill'] = fills_df['fill_time'].map(spread_lookup)
    
    # Fill rate by tau bucket
    tau_buckets = {
        '0-60s': (0, 60),
        '60-120s': (60, 120),
        '120-300s': (120, 300),
        '300-600s': (300, 600),
        '600+s': (600, 9999),
    }
    
    fill_rate_by_tau = {}
    if 'tau_at_fill' in fills_df.columns:
        for bucket_name, (low, high) in tau_buckets.items():
            bucket_fills = fills_df[(fills_df['tau_at_fill'] >= low) & (fills_df['tau_at_fill'] < high)]
            fill_rate_by_tau[bucket_name] = len(bucket_fills) / total_fills if total_fills > 0 else 0.0
    
    # Fill rate by spread bucket
    spread_buckets = {
        '0-1c': (0, 0.01),
        '1-2c': (0.01, 0.02),
        '2-3c': (0.02, 0.03),
        '3-5c': (0.03, 0.05),
        '5c+': (0.05, 1.0),
    }
    
    fill_rate_by_spread = {}
    avg_spread_at_fill = 0.0
    if 'spread_at_fill' in fills_df.columns:
        fills_with_spread = fills_df[fills_df['spread_at_fill'].notna()]
        if len(fills_with_spread) > 0:
            avg_spread_at_fill = fills_with_spread['spread_at_fill'].mean()
            
            for bucket_name, (low, high) in spread_buckets.items():
                bucket_fills = fills_with_spread[
                    (fills_with_spread['spread_at_fill'] >= low) & 
                    (fills_with_spread['spread_at_fill'] < high)
                ]
                fill_rate_by_spread[bucket_name] = len(bucket_fills) / len(fills_with_spread)
    
    # Adverse selection analysis
    avg_as_1s = 0.0
    avg_as_5s = 0.0
    if 'adverse_selection_1s' in fills_df.columns:
        as_1s = fills_df['adverse_selection_1s'].dropna()
        if len(as_1s) > 0:
            avg_as_1s = as_1s.mean()
    if 'adverse_selection_5s' in fills_df.columns:
        as_5s = fills_df['adverse_selection_5s'].dropna()
        if len(as_5s) > 0:
            avg_as_5s = as_5s.mean()
    
    return FillAnalysis(
        total_fills=total_fills,
        fill_rate=fill_rate,
        avg_time_to_fill=avg_time_to_fill,
        median_time_to_fill=median_time_to_fill,
        fill_rate_by_tau_bucket=fill_rate_by_tau,
        fill_rate_by_spread_bucket=fill_rate_by_spread,
        avg_spread_at_fill=avg_spread_at_fill,
        avg_adverse_selection_1s=avg_as_1s,
        avg_adverse_selection_5s=avg_as_5s,
    )


def analyze_fill_rate_by_tau(
    backtest_result: Dict[str, Any],
    market_data: pd.DataFrame,
    tau_bins: List[int] = None,
) -> pd.DataFrame:
    """
    Analyze fill rate as a function of time-to-expiry (tau).
    
    Args:
        backtest_result: Result from run_maker_backtest
        market_data: DataFrame with market data
        tau_bins: Custom tau bin edges (default: [0, 60, 120, 180, 300, 600, 900])
        
    Returns:
        DataFrame with fill rate by tau bucket
    """
    if tau_bins is None:
        tau_bins = [0, 60, 120, 180, 300, 600, 900]
    
    fills = backtest_result.get('fills', [])
    if not fills:
        return pd.DataFrame()
    
    fills_df = pd.DataFrame(fills)
    
    # Get tau at fill time
    tau_lookup = market_data.groupby('t')['tau'].first().to_dict()
    fills_df['tau'] = fills_df['fill_time'].map(tau_lookup)
    
    # Bin by tau
    fills_df['tau_bin'] = pd.cut(fills_df['tau'], bins=tau_bins, labels=False)
    
    # Compute stats per bin
    results = []
    for i, (low, high) in enumerate(zip(tau_bins[:-1], tau_bins[1:])):
        bin_fills = fills_df[fills_df['tau_bin'] == i]
        results.append({
            'tau_low': low,
            'tau_high': high,
            'tau_label': f'{low}-{high}s',
            'n_fills': len(bin_fills),
            'avg_fill_price': bin_fills['fill_price'].mean() if len(bin_fills) > 0 else np.nan,
            'avg_adverse_selection_1s': bin_fills['adverse_selection_1s'].mean() if 'adverse_selection_1s' in bin_fills.columns and len(bin_fills) > 0 else np.nan,
        })
    
    return pd.DataFrame(results)


def analyze_fill_rate_by_spread(
    backtest_result: Dict[str, Any],
    market_data: pd.DataFrame,
    spread_bins: List[float] = None,
) -> pd.DataFrame:
    """
    Analyze fill rate as a function of spread width.
    
    Args:
        backtest_result: Result from run_maker_backtest
        market_data: DataFrame with market data
        spread_bins: Custom spread bin edges (default: [0, 0.01, 0.02, 0.03, 0.05, 0.10])
        
    Returns:
        DataFrame with fill rate by spread bucket
    """
    if spread_bins is None:
        spread_bins = [0, 0.01, 0.02, 0.03, 0.05, 0.10]
    
    fills = backtest_result.get('fills', [])
    if not fills:
        return pd.DataFrame()
    
    fills_df = pd.DataFrame(fills)
    
    # Get spread at fill time
    if 'pm_up_spread' not in market_data.columns:
        market_data = market_data.copy()
        market_data['pm_up_spread'] = market_data['pm_up_best_ask'] - market_data['pm_up_best_bid']
    
    spread_lookup = market_data.groupby('t')['pm_up_spread'].first().to_dict()
    fills_df['spread'] = fills_df['fill_time'].map(spread_lookup)
    
    # Bin by spread
    fills_df['spread_bin'] = pd.cut(fills_df['spread'], bins=spread_bins, labels=False)
    
    results = []
    for i, (low, high) in enumerate(zip(spread_bins[:-1], spread_bins[1:])):
        bin_fills = fills_df[fills_df['spread_bin'] == i]
        results.append({
            'spread_low': low,
            'spread_high': high,
            'spread_label': f'{low*100:.0f}-{high*100:.0f}c',
            'n_fills': len(bin_fills),
            'avg_fill_price': bin_fills['fill_price'].mean() if len(bin_fills) > 0 else np.nan,
            'pct_of_fills': len(bin_fills) / len(fills_df) if len(fills_df) > 0 else 0.0,
        })
    
    return pd.DataFrame(results)


def time_to_fill_distribution(
    backtest_result: Dict[str, Any],
    bins: List[int] = None,
) -> pd.DataFrame:
    """
    Compute time-to-fill distribution.
    
    Args:
        backtest_result: Result from run_maker_backtest
        bins: Time bins in seconds (default: [0, 1, 2, 5, 10, 30, 60])
        
    Returns:
        DataFrame with time-to-fill distribution
    """
    if bins is None:
        bins = [0, 1, 2, 5, 10, 30, 60, 120]
    
    # Extract market results which have time-to-fill data
    market_results = backtest_result.get('market_results', {})
    
    # For now, return empty - would need to store time_to_fill per order
    # This requires enhancing the backtest to track order placement times
    return pd.DataFrame({
        'time_bin': [f'{bins[i]}-{bins[i+1]}s' for i in range(len(bins)-1)],
        'count': [0] * (len(bins)-1),
        'pct': [0.0] * (len(bins)-1),
    })


def pnl_decomposition_by_market(
    backtest_result: Dict[str, Any],
) -> pd.DataFrame:
    """
    Decompose PnL by market into spread/adverse selection/inventory components.
    
    Args:
        backtest_result: Result from run_maker_backtest
        
    Returns:
        DataFrame with PnL breakdown per market
    """
    market_results = backtest_result.get('market_results', {})
    
    rows = []
    for market_id, result in market_results.items():
        rows.append({
            'market_id': market_id,
            'total_pnl': result.get('pnl', 0.0),
            'spread_captured': result.get('spread_captured', 0.0),
            'adverse_selection': result.get('adverse_selection', 0.0),
            'inventory_carry': result.get('inventory_carry', 0.0),
            'n_fills': result.get('orders_filled', 0),
            'fill_volume': result.get('fill_volume', 0.0),
            'quote_seconds': result.get('quote_seconds', 0),
        })
    
    return pd.DataFrame(rows)


def adverse_selection_analysis(
    backtest_result: Dict[str, Any],
    market_data: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Detailed analysis of adverse selection costs.
    
    Analyzes:
    - When does adverse selection occur (by tau, spread, CL movement)
    - How large are adverse selection costs
    - Is there a pattern to which fills are picked off
    
    Args:
        backtest_result: Result from run_maker_backtest
        market_data: DataFrame with market data
        
    Returns:
        Dictionary with adverse selection analysis
    """
    fills = backtest_result.get('fills', [])
    if not fills:
        return {
            'total_fills': 0,
            'fills_with_adverse_selection_1s': 0,
            'fills_with_adverse_selection_5s': 0,
            'avg_adverse_selection_1s': 0.0,
            'avg_adverse_selection_5s': 0.0,
            'pct_negative_as_1s': 0.0,
            'pct_negative_as_5s': 0.0,
        }
    
    fills_df = pd.DataFrame(fills)
    
    # Count fills with adverse selection data
    has_as_1s = fills_df['adverse_selection_1s'].notna().sum() if 'adverse_selection_1s' in fills_df.columns else 0
    has_as_5s = fills_df['adverse_selection_5s'].notna().sum() if 'adverse_selection_5s' in fills_df.columns else 0
    
    # Compute averages
    avg_as_1s = 0.0
    avg_as_5s = 0.0
    pct_neg_1s = 0.0
    pct_neg_5s = 0.0
    
    if 'adverse_selection_1s' in fills_df.columns:
        as_1s = fills_df['adverse_selection_1s'].dropna()
        if len(as_1s) > 0:
            avg_as_1s = as_1s.mean()
            pct_neg_1s = (as_1s < 0).mean()  # Negative AS means we gained
    
    if 'adverse_selection_5s' in fills_df.columns:
        as_5s = fills_df['adverse_selection_5s'].dropna()
        if len(as_5s) > 0:
            avg_as_5s = as_5s.mean()
            pct_neg_5s = (as_5s < 0).mean()
    
    return {
        'total_fills': len(fills_df),
        'fills_with_adverse_selection_1s': has_as_1s,
        'fills_with_adverse_selection_5s': has_as_5s,
        'avg_adverse_selection_1s': avg_as_1s,
        'avg_adverse_selection_5s': avg_as_5s,
        'pct_negative_as_1s': pct_neg_1s,  # % where we gained after fill
        'pct_negative_as_5s': pct_neg_5s,
    }


def quote_staleness_analysis(
    backtest_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Analyze quote staleness and update patterns.
    
    Args:
        backtest_result: Result from run_maker_backtest
        
    Returns:
        Dictionary with quote staleness metrics
    """
    market_results = backtest_result.get('market_results', {})
    
    total_quote_updates = sum(r.get('quote_updates', 0) for r in market_results.values())
    total_quote_seconds = sum(r.get('quote_seconds', 0) for r in market_results.values())
    
    avg_updates_per_second = total_quote_updates / total_quote_seconds if total_quote_seconds > 0 else 0.0
    
    return {
        'total_quote_updates': total_quote_updates,
        'total_quote_seconds': total_quote_seconds,
        'avg_updates_per_second': avg_updates_per_second,
        'avg_seconds_between_updates': 1 / avg_updates_per_second if avg_updates_per_second > 0 else float('inf'),
    }


def generate_diagnostics_summary(
    backtest_result: Dict[str, Any],
    market_data: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Generate comprehensive diagnostics summary.
    
    Args:
        backtest_result: Result from run_maker_backtest
        market_data: DataFrame with market data
        
    Returns:
        Dictionary with all diagnostic analyses
    """
    metrics = backtest_result.get('metrics', {})
    
    # Fill analysis
    fill_analysis = analyze_fills(
        fills=backtest_result.get('fills', []),
        orders_placed=metrics.get('orders_placed_total', 0),
        market_data=market_data,
    )
    
    # Fill rate by tau
    fill_by_tau = analyze_fill_rate_by_tau(backtest_result, market_data)
    
    # Fill rate by spread
    fill_by_spread = analyze_fill_rate_by_spread(backtest_result, market_data)
    
    # PnL decomposition
    pnl_decomp = pnl_decomposition_by_market(backtest_result)
    
    # Adverse selection
    as_analysis = adverse_selection_analysis(backtest_result, market_data)
    
    # Quote staleness
    quote_analysis = quote_staleness_analysis(backtest_result)
    
    return {
        'strategy': backtest_result.get('strategy', 'Unknown'),
        'metrics': metrics,
        'fill_analysis': fill_analysis.to_dict(),
        'fill_rate_by_tau': fill_by_tau.to_dict('records') if len(fill_by_tau) > 0 else [],
        'fill_rate_by_spread': fill_by_spread.to_dict('records') if len(fill_by_spread) > 0 else [],
        'pnl_by_market': pnl_decomp.to_dict('records') if len(pnl_decomp) > 0 else [],
        'adverse_selection': as_analysis,
        'quote_staleness': quote_analysis,
    }


def fill_quality_by_quote_level(
    backtest_result: Dict[str, Any],
    market_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze fill quality (adverse selection) as a function of quote level.
    
    Compares L1 (best) vs L2 (second best) vs deeper levels.
    Key insight: Quoting at L2 typically has lower fill rate but also lower AS.
    
    Args:
        backtest_result: Result from run_maker_backtest
        market_data: DataFrame with market data
        
    Returns:
        DataFrame with fill quality metrics by level
    """
    fills = backtest_result.get('fills', [])
    if not fills:
        return pd.DataFrame()
    
    fills_df = pd.DataFrame(fills)
    
    # Check if we have quote level info in fills
    # This would require the backtest to track which level order was placed at
    # For now, infer from fill price vs L1/L2/L3 prices at fill time
    
    results = []
    
    # Get L1 and L2 prices from market data
    if 't' in market_data.columns and 'fill_time' in fills_df.columns:
        # Create price lookups for each level
        l1_bid = market_data.groupby('t')['pm_up_best_bid'].first().to_dict()
        l1_ask = market_data.groupby('t')['pm_up_best_ask'].first().to_dict()
        
        # Check if L2 columns exist
        has_l2 = 'pm_up_bid_2' in market_data.columns
        if has_l2:
            l2_bid = market_data.groupby('t')['pm_up_bid_2'].first().to_dict()
            l2_ask = market_data.groupby('t')['pm_up_ask_2'].first().to_dict()
        
        # Classify each fill by which level it was at
        for _, fill in fills_df.iterrows():
            t = fill.get('fill_time')
            price = fill.get('fill_price')
            side = fill.get('side')
            
            if t is None or price is None:
                continue
            
            # Determine level
            level = 0
            if side == 'BID':
                if t in l1_bid and abs(price - l1_bid[t]) < 0.001:
                    level = 1
                elif has_l2 and t in l2_bid and l2_bid[t] is not None and abs(price - l2_bid[t]) < 0.001:
                    level = 2
            else:  # ASK
                if t in l1_ask and abs(price - l1_ask[t]) < 0.001:
                    level = 1
                elif has_l2 and t in l2_ask and l2_ask[t] is not None and abs(price - l2_ask[t]) < 0.001:
                    level = 2
            
            if level == 0:
                level = 3  # Deeper or improved
        
        # Group fills by level
        fills_df['inferred_level'] = fills_df.apply(
            lambda row: _infer_fill_level(row, l1_bid, l1_ask, l2_bid if has_l2 else {}, l2_ask if has_l2 else {}),
            axis=1
        )
        
        # Compute stats per level
        for level in fills_df['inferred_level'].unique():
            level_fills = fills_df[fills_df['inferred_level'] == level]
            
            as_1s = level_fills['adverse_selection_1s'].dropna() if 'adverse_selection_1s' in level_fills.columns else pd.Series()
            as_5s = level_fills['adverse_selection_5s'].dropna() if 'adverse_selection_5s' in level_fills.columns else pd.Series()
            
            results.append({
                'quote_level': level,
                'n_fills': len(level_fills),
                'pct_of_fills': len(level_fills) / len(fills_df) if len(fills_df) > 0 else 0,
                'avg_as_1s': as_1s.mean() if len(as_1s) > 0 else np.nan,
                'avg_as_5s': as_5s.mean() if len(as_5s) > 0 else np.nan,
                'pct_gain_1s': (as_1s < 0).mean() if len(as_1s) > 0 else np.nan,
            })
    
    return pd.DataFrame(results)


def _infer_fill_level(
    fill: pd.Series,
    l1_bid: Dict,
    l1_ask: Dict,
    l2_bid: Dict,
    l2_ask: Dict,
) -> int:
    """Infer which level a fill was at based on price comparison."""
    t = fill.get('fill_time')
    price = fill.get('fill_price')
    side = fill.get('side')
    
    if t is None or price is None:
        return 0
    
    if side == 'BID':
        if t in l1_bid and l1_bid[t] is not None and abs(price - l1_bid[t]) < 0.001:
            return 1
        if l2_bid and t in l2_bid and l2_bid[t] is not None and abs(price - l2_bid[t]) < 0.001:
            return 2
    else:  # ASK
        if t in l1_ask and l1_ask[t] is not None and abs(price - l1_ask[t]) < 0.001:
            return 1
        if l2_ask and t in l2_ask and l2_ask[t] is not None and abs(price - l2_ask[t]) < 0.001:
            return 2
    
    return 3  # Unknown or deeper


def compute_capacity_proxy(
    backtest_result: Dict[str, Any],
    market_data: pd.DataFrame,
    target_as_budget: float = 0.005,  # Maximum AS cost we're willing to accept per fill
) -> Dict[str, Any]:
    """
    Estimate maximum notional capacity at target adverse selection budget.
    
    Key insight: How much could we trade before AS costs exceed acceptable threshold?
    This answers: "How scalable is this strategy?"
    
    Args:
        backtest_result: Result from run_maker_backtest
        market_data: DataFrame with market data
        target_as_budget: Maximum AS cost per fill (default 0.5 cents)
        
    Returns:
        Dictionary with capacity metrics
    """
    fills = backtest_result.get('fills', [])
    if not fills:
        return {
            'max_notional_at_target_as': 0.0,
            'current_fill_volume': 0.0,
            'avg_as_per_unit': 0.0,
            'capacity_multiple': 0.0,
            'notes': 'No fills to analyze',
        }
    
    fills_df = pd.DataFrame(fills)
    
    # Calculate current fill volume
    current_fill_volume = fills_df['fill_size'].sum() if 'fill_size' in fills_df.columns else 0.0
    
    # Calculate average AS per unit volume
    if 'adverse_selection_5s' in fills_df.columns and 'fill_size' in fills_df.columns:
        as_with_size = fills_df[['adverse_selection_5s', 'fill_size']].dropna()
        if len(as_with_size) > 0:
            # Total AS cost / total volume = AS per unit
            total_as = (as_with_size['adverse_selection_5s'] * as_with_size['fill_size']).sum()
            total_volume = as_with_size['fill_size'].sum()
            avg_as_per_unit = total_as / total_volume if total_volume > 0 else 0.0
        else:
            avg_as_per_unit = 0.0
    else:
        avg_as_per_unit = 0.0
    
    # Estimate max notional at target AS budget
    if avg_as_per_unit > 0 and avg_as_per_unit < target_as_budget:
        # We're within budget - capacity is not limited by AS
        capacity_multiple = float('inf')
        max_notional = float('inf')
        notes = 'AS cost within budget - capacity limited by other factors'
    elif avg_as_per_unit >= target_as_budget:
        # AS exceeds budget - capacity is 0 at this target
        capacity_multiple = 0.0
        max_notional = 0.0
        notes = f'AS cost ({avg_as_per_unit:.4f}) exceeds target ({target_as_budget:.4f})'
    else:
        # avg_as_per_unit <= 0 (we're gaining on average)
        capacity_multiple = float('inf')
        max_notional = float('inf')
        notes = 'Favorable AS (gain on average) - capacity unlimited by AS'
    
    return {
        'max_notional_at_target_as': max_notional,
        'current_fill_volume': current_fill_volume,
        'avg_as_per_unit': avg_as_per_unit,
        'target_as_budget': target_as_budget,
        'capacity_multiple': capacity_multiple,
        'notes': notes,
    }


def compute_expected_edge_per_fill(
    fills: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Compute expected edge for each fill.
    
    Expected edge = spread_captured - adverse_selection - latency_buffer
    
    This metric helps identify which fills were profitable and why.
    
    Args:
        fills: List of fill dictionaries
        
    Returns:
        DataFrame with per-fill edge calculations
    """
    if not fills:
        return pd.DataFrame()
    
    fills_df = pd.DataFrame(fills)
    
    # Add edge calculations
    results = []
    for _, fill in fills_df.iterrows():
        spread_captured = fill.get('spread_captured', 0)
        as_1s = fill.get('adverse_selection_1s', 0) or 0
        as_5s = fill.get('adverse_selection_5s', 0) or 0
        
        # Edge = spread captured - AS cost
        edge_1s = spread_captured - as_1s if not pd.isna(as_1s) else None
        edge_5s = spread_captured - as_5s if not pd.isna(as_5s) else None
        
        results.append({
            'order_id': fill.get('order_id'),
            'fill_time': fill.get('fill_time'),
            'fill_price': fill.get('fill_price'),
            'fill_size': fill.get('fill_size'),
            'spread_captured': spread_captured,
            'adverse_selection_1s': as_1s,
            'adverse_selection_5s': as_5s,
            'expected_edge_1s': edge_1s,
            'expected_edge_5s': edge_5s,
            'is_profitable_1s': edge_1s > 0 if edge_1s is not None else None,
            'is_profitable_5s': edge_5s > 0 if edge_5s is not None else None,
        })
    
    return pd.DataFrame(results)


def print_diagnostics_report(
    diagnostics: Dict[str, Any],
) -> None:
    """
    Print formatted diagnostics report.
    
    Args:
        diagnostics: Result from generate_diagnostics_summary
    """
    print("=" * 80)
    print(f"MAKER STRATEGY DIAGNOSTICS: {diagnostics['strategy']}")
    print("=" * 80)
    
    # Metrics summary
    metrics = diagnostics.get('metrics', {})
    print(f"\n{'PERFORMANCE SUMMARY':-^60}")
    print(f"  Total PnL: ${metrics.get('total_pnl', 0):.4f}")
    print(f"  Mean PnL/market: ${metrics.get('mean_pnl_per_market', 0):.4f}")
    print(f"  t-statistic: {metrics.get('t_stat', 0):.2f}")
    print(f"  Hit rate (markets): {metrics.get('hit_rate_per_market', 0)*100:.1f}%")
    
    # PnL decomposition
    print(f"\n{'PNL DECOMPOSITION':-^60}")
    print(f"  Spread captured: ${metrics.get('spread_captured_total', 0):.4f}")
    print(f"  Adverse selection: -${metrics.get('adverse_selection_total', 0):.4f}")
    print(f"  Inventory carry: ${metrics.get('inventory_carry_total', 0):.4f}")
    
    # Fill analysis
    fill_analysis = diagnostics.get('fill_analysis', {})
    print(f"\n{'FILL ANALYSIS':-^60}")
    print(f"  Total fills: {fill_analysis.get('total_fills', 0)}")
    print(f"  Fill rate: {fill_analysis.get('fill_rate', 0)*100:.2f}%")
    print(f"  Avg spread at fill: {fill_analysis.get('avg_spread_at_fill', 0)*100:.2f}c")
    
    # Fill rate by tau
    fill_by_tau = diagnostics.get('fill_rate_by_tau', [])
    if fill_by_tau:
        print(f"\n{'FILL RATE BY TAU':-^60}")
        for row in fill_by_tau:
            print(f"  {row['tau_label']}: {row['n_fills']} fills")
    
    # Fill rate by spread
    fill_by_spread = diagnostics.get('fill_rate_by_spread', [])
    if fill_by_spread:
        print(f"\n{'FILL RATE BY SPREAD':-^60}")
        for row in fill_by_spread:
            print(f"  {row['spread_label']}: {row['n_fills']} fills ({row['pct_of_fills']*100:.1f}%)")
    
    # Adverse selection
    as_analysis = diagnostics.get('adverse_selection', {})
    print(f"\n{'ADVERSE SELECTION ANALYSIS':-^60}")
    print(f"  Avg AS (1s): {as_analysis.get('avg_adverse_selection_1s', 0)*100:.3f}c")
    print(f"  Avg AS (5s): {as_analysis.get('avg_adverse_selection_5s', 0)*100:.3f}c")
    print(f"  % fills with gain (1s): {as_analysis.get('pct_negative_as_1s', 0)*100:.1f}%")
    print(f"  % fills with gain (5s): {as_analysis.get('pct_negative_as_5s', 0)*100:.1f}%")
    
    # Quote staleness
    quote_analysis = diagnostics.get('quote_staleness', {})
    print(f"\n{'QUOTE ANALYSIS':-^60}")
    print(f"  Total quote seconds: {quote_analysis.get('total_quote_seconds', 0)}")
    print(f"  Quote updates/second: {quote_analysis.get('avg_updates_per_second', 0):.2f}")
    
    print("\n" + "=" * 80)


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from scripts.backtest.data_loader import load_eth_markets, add_derived_columns
    from scripts.backtest.strategies import SpreadCaptureStrategy
    from scripts.backtest.backtest_engine import run_maker_backtest
    from scripts.backtest.maker_execution import MakerExecutionConfig, FillModel
    
    print("Loading data...")
    df, market_info = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    # Run backtest
    strategy = SpreadCaptureStrategy(
        spread_min=0.01,
        tau_min=60,
        tau_max=600,
        two_sided=True,
    )
    
    config = MakerExecutionConfig(
        place_latency_ms=50,
        cancel_latency_ms=25,
        fill_model=FillModel.TOUCH_SIZE_PROXY,
        touch_trade_rate_per_second=0.15,
    )
    
    print(f"\nRunning backtest: {strategy.name}")
    result = run_maker_backtest(df, strategy, config, verbose=False, volume_markets_only=True)
    
    # Generate and print diagnostics
    diagnostics = generate_diagnostics_summary(result, df)
    print_diagnostics_report(diagnostics)

