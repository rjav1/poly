#!/usr/bin/env python3
"""
Phase 3: Execution Style Inference (Maker vs Taker)

Classifies each trade as maker or taker based on trade price vs orderbook quotes.
Also infers conversion route usage.

Input:
- positions.parquet (enriched trades from Phase 2)

Output:
- execution_enriched.parquet (trades with execution labels)
- execution_summary.json (per-wallet execution style analysis)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Price proximity tolerance (in probability points, e.g., 0.005 = 0.5 cents)
PRICE_TOLERANCE = 0.005


def load_positions_data() -> pd.DataFrame:
    """Load enriched positions data from Phase 2."""
    path = DATA_DIR / "positions.parquet"
    print(f"Loading positions data from: {path}")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} rows")
    return df


def infer_maker_taker(
    trade_price: float,
    trade_side: str,
    outcome_token: str,
    up_bid: float,
    up_ask: float,
    down_bid: float,
    down_ask: float,
    tolerance: float = PRICE_TOLERANCE
) -> Tuple[str, str, float]:
    """
    Infer execution type (MAKER/TAKER) and route (DIRECT/CONVERSION) for a trade.
    
    Args:
        trade_price: Actual fill price
        trade_side: BUY or SELL
        outcome_token: Up or Down
        up_bid, up_ask, down_bid, down_ask: Orderbook quotes
        tolerance: Price proximity tolerance
        
    Returns:
        Tuple of (execution_type, execution_route, aggressiveness_score)
    """
    # Handle NaN quotes
    if pd.isna(up_bid) or pd.isna(up_ask) or pd.isna(down_bid) or pd.isna(down_ask):
        return 'UNKNOWN', 'UNKNOWN', 0.5
    
    # Compute conversion prices
    # Buy UP via conversion: split $1 into UP+DOWN, sell DOWN for down_bid, keep UP
    # Effective cost = 1 - down_bid
    buy_up_conversion = 1 - down_bid
    sell_up_conversion = 1 - down_ask  # Buy DOWN, redeem pair
    
    buy_down_conversion = 1 - up_bid
    sell_down_conversion = 1 - up_ask
    
    if outcome_token == 'Up':
        if trade_side == 'BUY':
            direct_price = up_ask
            conversion_price = buy_up_conversion
            best_price = min(direct_price, conversion_price)
            
            # Check if trade was at ask (taker) or better (maker/inside)
            if abs(trade_price - up_ask) <= tolerance:
                exec_type = 'TAKER'
                route = 'DIRECT'
            elif abs(trade_price - buy_up_conversion) <= tolerance:
                exec_type = 'TAKER'  # Using conversion but still aggressive
                route = 'CONVERSION'
            elif trade_price < up_bid - tolerance:
                exec_type = 'MAKER'  # Got filled inside spread
                route = 'DIRECT'
            elif trade_price >= up_bid - tolerance and trade_price <= up_ask + tolerance:
                # Inside spread
                spread = up_ask - up_bid
                if spread > 0:
                    position_in_spread = (trade_price - up_bid) / spread
                    if position_in_spread < 0.3:
                        exec_type = 'MAKER'  # Near bid
                    elif position_in_spread > 0.7:
                        exec_type = 'TAKER'  # Near ask
                    else:
                        exec_type = 'UNKNOWN'  # Mid-spread
                else:
                    exec_type = 'UNKNOWN'
                route = 'DIRECT'
            else:
                exec_type = 'UNKNOWN'
                route = 'UNKNOWN'
            
            # Aggressiveness: how close to worst price (ask for buy)
            if up_ask > up_bid:
                aggressiveness = (trade_price - up_bid) / (up_ask - up_bid)
            else:
                aggressiveness = 0.5
                
        else:  # SELL UP
            direct_price = up_bid
            conversion_price = sell_up_conversion
            
            if abs(trade_price - up_bid) <= tolerance:
                exec_type = 'TAKER'
                route = 'DIRECT'
            elif abs(trade_price - sell_up_conversion) <= tolerance:
                exec_type = 'TAKER'
                route = 'CONVERSION'
            elif trade_price > up_ask + tolerance:
                exec_type = 'MAKER'
                route = 'DIRECT'
            elif trade_price >= up_bid - tolerance and trade_price <= up_ask + tolerance:
                spread = up_ask - up_bid
                if spread > 0:
                    position_in_spread = (trade_price - up_bid) / spread
                    if position_in_spread > 0.7:
                        exec_type = 'MAKER'  # Near ask
                    elif position_in_spread < 0.3:
                        exec_type = 'TAKER'  # Near bid
                    else:
                        exec_type = 'UNKNOWN'
                else:
                    exec_type = 'UNKNOWN'
                route = 'DIRECT'
            else:
                exec_type = 'UNKNOWN'
                route = 'UNKNOWN'
            
            # Aggressiveness for sell: higher price is better (less aggressive)
            if up_ask > up_bid:
                aggressiveness = 1 - (trade_price - up_bid) / (up_ask - up_bid)
            else:
                aggressiveness = 0.5
                
    else:  # outcome_token == 'Down'
        if trade_side == 'BUY':
            direct_price = down_ask
            conversion_price = buy_down_conversion
            
            if abs(trade_price - down_ask) <= tolerance:
                exec_type = 'TAKER'
                route = 'DIRECT'
            elif abs(trade_price - buy_down_conversion) <= tolerance:
                exec_type = 'TAKER'
                route = 'CONVERSION'
            elif trade_price < down_bid - tolerance:
                exec_type = 'MAKER'
                route = 'DIRECT'
            elif trade_price >= down_bid - tolerance and trade_price <= down_ask + tolerance:
                spread = down_ask - down_bid
                if spread > 0:
                    position_in_spread = (trade_price - down_bid) / spread
                    if position_in_spread < 0.3:
                        exec_type = 'MAKER'
                    elif position_in_spread > 0.7:
                        exec_type = 'TAKER'
                    else:
                        exec_type = 'UNKNOWN'
                else:
                    exec_type = 'UNKNOWN'
                route = 'DIRECT'
            else:
                exec_type = 'UNKNOWN'
                route = 'UNKNOWN'
            
            if down_ask > down_bid:
                aggressiveness = (trade_price - down_bid) / (down_ask - down_bid)
            else:
                aggressiveness = 0.5
                
        else:  # SELL DOWN
            direct_price = down_bid
            conversion_price = sell_down_conversion
            
            if abs(trade_price - down_bid) <= tolerance:
                exec_type = 'TAKER'
                route = 'DIRECT'
            elif abs(trade_price - sell_down_conversion) <= tolerance:
                exec_type = 'TAKER'
                route = 'CONVERSION'
            elif trade_price > down_ask + tolerance:
                exec_type = 'MAKER'
                route = 'DIRECT'
            elif trade_price >= down_bid - tolerance and trade_price <= down_ask + tolerance:
                spread = down_ask - down_bid
                if spread > 0:
                    position_in_spread = (trade_price - down_bid) / spread
                    if position_in_spread > 0.7:
                        exec_type = 'MAKER'
                    elif position_in_spread < 0.3:
                        exec_type = 'TAKER'
                    else:
                        exec_type = 'UNKNOWN'
                else:
                    exec_type = 'UNKNOWN'
                route = 'DIRECT'
            else:
                exec_type = 'UNKNOWN'
                route = 'UNKNOWN'
            
            if down_ask > down_bid:
                aggressiveness = 1 - (trade_price - down_bid) / (down_ask - down_bid)
            else:
                aggressiveness = 0.5
    
    # Clip aggressiveness to [0, 1]
    aggressiveness = float(np.clip(aggressiveness, 0, 1))
    
    return exec_type, route, aggressiveness


def detect_execution_patterns(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Detect execution patterns like repeated fills at same price level.
    """
    print("\nDetecting execution patterns...")
    trades = trades.copy()
    
    # Group by (wallet, market_id, outcome_token) and look for patterns
    trades['repeated_price'] = False
    trades['fills_at_price'] = 1
    
    for (wallet, market_id, token), group in trades.groupby(['wallet', 'market_id', 'outcome_token']):
        idx = group.index
        subset = group.sort_values('t')
        
        # Check for repeated fills at same price
        prices = subset['price'].values
        
        # Count consecutive same-price fills
        same_price = np.concatenate([[False], prices[1:] == prices[:-1]])
        trades.loc[idx, 'repeated_price'] = same_price
        
        # Count fills at each price level
        price_counts = subset['price'].value_counts()
        trades.loc[idx, 'fills_at_price'] = subset['price'].map(price_counts).values
    
    # High frequency of repeated fills suggests maker/MM behavior
    trades['mm_like_pattern'] = trades['fills_at_price'] > 3
    
    print(f"  Repeated price fills: {trades['repeated_price'].sum():,}")
    print(f"  MM-like patterns (>3 fills at same price): {trades['mm_like_pattern'].sum():,}")
    
    return trades


def infer_all_execution_styles(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Infer execution style for all trades.
    """
    print("\nInferring execution styles...")
    trades = trades.copy()
    
    # Check for required market state columns
    has_quotes = all(col in trades.columns for col in [
        'mkt_pm_up_best_bid', 'mkt_pm_up_best_ask',
        'mkt_pm_down_best_bid', 'mkt_pm_down_best_ask'
    ])
    
    if not has_quotes:
        print("  WARNING: Market quote columns not found. Using simplified inference.")
        trades['execution_type'] = 'UNKNOWN'
        trades['execution_route'] = 'UNKNOWN'
        trades['aggressiveness_score'] = 0.5
        return trades
    
    # Initialize columns
    trades['execution_type'] = 'UNKNOWN'
    trades['execution_route'] = 'UNKNOWN'
    trades['aggressiveness_score'] = 0.5
    
    # Process each trade
    for idx, row in trades.iterrows():
        exec_type, route, aggr = infer_maker_taker(
            trade_price=row['price'],
            trade_side=row['side'],
            outcome_token=row['outcome_token'],
            up_bid=row.get('mkt_pm_up_best_bid', np.nan),
            up_ask=row.get('mkt_pm_up_best_ask', np.nan),
            down_bid=row.get('mkt_pm_down_best_bid', np.nan),
            down_ask=row.get('mkt_pm_down_best_ask', np.nan),
        )
        
        trades.loc[idx, 'execution_type'] = exec_type
        trades.loc[idx, 'execution_route'] = route
        trades.loc[idx, 'aggressiveness_score'] = aggr
    
    # Summary
    exec_type_counts = trades['execution_type'].value_counts()
    route_counts = trades['execution_route'].value_counts()
    
    print(f"\n  Execution type distribution:")
    for t, c in exec_type_counts.items():
        print(f"    {t}: {c:,} ({c/len(trades)*100:.1f}%)")
    
    print(f"\n  Execution route distribution:")
    for r, c in route_counts.items():
        print(f"    {r}: {c:,} ({c/len(trades)*100:.1f}%)")
    
    return trades


def compute_wallet_execution_summary(trades: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute execution style summary per wallet.
    """
    print("\nComputing wallet execution summaries...")
    
    summaries = {}
    
    for wallet in trades['wallet'].unique():
        wallet_trades = trades[trades['wallet'] == wallet]
        n_trades = len(wallet_trades)
        
        # Execution type breakdown
        exec_types = wallet_trades['execution_type'].value_counts(normalize=True).to_dict()
        
        # Route breakdown
        routes = wallet_trades['execution_route'].value_counts(normalize=True).to_dict()
        
        # Average aggressiveness
        avg_aggr = wallet_trades['aggressiveness_score'].mean()
        
        # MM-like pattern ratio
        mm_ratio = wallet_trades['mm_like_pattern'].mean() if 'mm_like_pattern' in wallet_trades.columns else 0
        
        # By side breakdown
        buy_trades = wallet_trades[wallet_trades['side'] == 'BUY']
        sell_trades = wallet_trades[wallet_trades['side'] == 'SELL']
        
        buy_aggr = buy_trades['aggressiveness_score'].mean() if len(buy_trades) > 0 else np.nan
        sell_aggr = sell_trades['aggressiveness_score'].mean() if len(sell_trades) > 0 else np.nan
        
        # Primary execution style determination
        if exec_types.get('TAKER', 0) > 0.6:
            primary_style = 'AGGRESSIVE_TAKER'
        elif exec_types.get('MAKER', 0) > 0.6:
            primary_style = 'PASSIVE_MAKER'
        elif exec_types.get('TAKER', 0) > exec_types.get('MAKER', 0):
            primary_style = 'MIXED_TAKER_BIAS'
        elif exec_types.get('MAKER', 0) > exec_types.get('TAKER', 0):
            primary_style = 'MIXED_MAKER_BIAS'
        else:
            primary_style = 'MIXED'
        
        summaries[wallet] = {
            'n_trades': n_trades,
            'execution_type_pct': {k: float(v) for k, v in exec_types.items()},
            'route_pct': {k: float(v) for k, v in routes.items()},
            'avg_aggressiveness': float(avg_aggr) if not np.isnan(avg_aggr) else None,
            'buy_aggressiveness': float(buy_aggr) if not np.isnan(buy_aggr) else None,
            'sell_aggressiveness': float(sell_aggr) if not np.isnan(sell_aggr) else None,
            'mm_like_ratio': float(mm_ratio),
            'primary_style': primary_style,
        }
        
        print(f"\n  {wallet}:")
        print(f"    Trades: {n_trades:,}")
        print(f"    Primary style: {primary_style}")
        print(f"    Avg aggressiveness: {avg_aggr:.3f}" if not np.isnan(avg_aggr) else "    Avg aggressiveness: N/A")
        print(f"    TAKER: {exec_types.get('TAKER', 0)*100:.1f}%, MAKER: {exec_types.get('MAKER', 0)*100:.1f}%")
        print(f"    Conversion usage: {routes.get('CONVERSION', 0)*100:.1f}%")
    
    return summaries


def compute_execution_by_conditions(trades: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze execution style conditional on market state.
    """
    print("\nAnalyzing execution by market conditions...")
    
    analysis = {}
    
    # By tau bucket
    if 'mkt_tau_bucket' in trades.columns:
        tau_analysis = {}
        for bucket in trades['mkt_tau_bucket'].dropna().unique():
            bucket_trades = trades[trades['mkt_tau_bucket'] == bucket]
            tau_analysis[str(bucket)] = {
                'n_trades': len(bucket_trades),
                'taker_pct': float((bucket_trades['execution_type'] == 'TAKER').mean()),
                'maker_pct': float((bucket_trades['execution_type'] == 'MAKER').mean()),
                'avg_aggressiveness': float(bucket_trades['aggressiveness_score'].mean()),
            }
        analysis['by_tau_bucket'] = tau_analysis
    
    # By underround presence
    if 'mkt_underround' in trades.columns:
        underround_trades = trades[trades['mkt_underround'] > 0.01]
        no_underround_trades = trades[trades['mkt_underround'] <= 0.01]
        
        analysis['with_underround'] = {
            'n_trades': len(underround_trades),
            'taker_pct': float((underround_trades['execution_type'] == 'TAKER').mean()) if len(underround_trades) > 0 else 0,
            'avg_aggressiveness': float(underround_trades['aggressiveness_score'].mean()) if len(underround_trades) > 0 else 0,
        }
        analysis['no_underround'] = {
            'n_trades': len(no_underround_trades),
            'taker_pct': float((no_underround_trades['execution_type'] == 'TAKER').mean()) if len(no_underround_trades) > 0 else 0,
            'avg_aggressiveness': float(no_underround_trades['aggressiveness_score'].mean()) if len(no_underround_trades) > 0 else 0,
        }
    
    # By spread width
    if 'mkt_avg_spread' in trades.columns:
        median_spread = trades['mkt_avg_spread'].median()
        if not pd.isna(median_spread):
            tight_spread = trades[trades['mkt_avg_spread'] <= median_spread]
            wide_spread = trades[trades['mkt_avg_spread'] > median_spread]
            
            analysis['tight_spread'] = {
                'n_trades': len(tight_spread),
                'taker_pct': float((tight_spread['execution_type'] == 'TAKER').mean()) if len(tight_spread) > 0 else 0,
                'avg_aggressiveness': float(tight_spread['aggressiveness_score'].mean()) if len(tight_spread) > 0 else 0,
            }
            analysis['wide_spread'] = {
                'n_trades': len(wide_spread),
                'taker_pct': float((wide_spread['execution_type'] == 'TAKER').mean()) if len(wide_spread) > 0 else 0,
                'avg_aggressiveness': float(wide_spread['aggressiveness_score'].mean()) if len(wide_spread) > 0 else 0,
            }
    
    return analysis


def main():
    print("=" * 70)
    print("Phase 3: Execution Style Inference (Maker vs Taker)")
    print("=" * 70)
    
    # Step 1: Load positions data
    trades = load_positions_data()
    
    # Step 2: Detect execution patterns
    trades = detect_execution_patterns(trades)
    
    # Step 3: Infer execution styles
    trades = infer_all_execution_styles(trades)
    
    # Step 4: Compute wallet summaries
    wallet_summaries = compute_wallet_execution_summary(trades)
    
    # Step 5: Analyze execution by conditions
    conditional_analysis = compute_execution_by_conditions(trades)
    
    # Step 6: Save outputs
    print("\n" + "=" * 70)
    print("Saving outputs...")
    
    # Save enriched trades
    output_path = DATA_DIR / "execution_enriched.parquet"
    trades.to_parquet(output_path, index=False)
    print(f"  Enriched trades saved to: {output_path}")
    print(f"  Shape: {trades.shape}")
    
    # Save wallet summaries
    summary_path = RESULTS_DIR / "execution_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'wallet_summaries': wallet_summaries,
            'conditional_analysis': conditional_analysis,
        }, f, indent=2, default=str)
    print(f"  Execution summary saved to: {summary_path}")
    
    print("\n" + "=" * 70)
    print("DONE - Phase 3 Complete")
    print("=" * 70)
    
    return trades, wallet_summaries


if __name__ == "__main__":
    main()

