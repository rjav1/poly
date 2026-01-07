#!/usr/bin/env python3
"""
Phase 2: Position & Inventory Reconstruction

Reconstructs signed positions over time per wallet x market x token to identify 
inventory patterns, holding times, and round-trip logic.

Input:
- research_table.parquet

Output:
- positions.parquet (per-trade with position context)
- position_summary.json (per-wallet x market analysis)
- hold_time_distributions.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MARKET_DURATION_SECONDS = 900


@dataclass
class LotMatch:
    """Represents a matched buy-sell lot."""
    buy_time: int
    sell_time: int
    hold_time: int
    buy_price: float
    sell_price: float
    quantity: float
    pnl: float
    pnl_per_unit: float


@dataclass
class PositionSummary:
    """Summary statistics for a wallet's position in a market."""
    wallet: str
    market_id: str
    outcome_token: str
    n_buys: int
    n_sells: int
    total_bought: float
    total_sold: float
    net_position: float
    peak_long: float
    peak_short: float
    end_position: float
    n_position_changes: int
    mean_reversion_score: float  # How often position returns toward zero
    hold_to_expiry: bool
    n_round_trips: int
    avg_hold_time: Optional[float]
    median_hold_time: Optional[float]
    total_realized_pnl: float


def load_research_table() -> pd.DataFrame:
    """Load the research table from Phase 1."""
    path = DATA_DIR / "research_table.parquet"
    print(f"Loading research table from: {path}")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} rows")
    return df


def build_position_time_series(
    trades: pd.DataFrame,
    wallet: str,
    market_id: str,
    outcome_token: str
) -> pd.Series:
    """
    Build signed position time series for a wallet/market/token combination.
    
    Returns:
        Series indexed by t (0-899) with position at each second.
    """
    # Filter to relevant trades
    mask = (
        (trades['wallet'] == wallet) &
        (trades['market_id'] == market_id) &
        (trades['outcome_token'] == outcome_token)
    )
    subset = trades[mask].copy()
    
    if len(subset) == 0:
        return pd.Series(dtype=float)
    
    # Sort by time
    subset = subset.sort_values('t')
    
    # Compute position changes
    # BUY = +size, SELL = -size
    subset['position_change'] = np.where(
        subset['side'] == 'BUY',
        subset['size'],
        -subset['size']
    )
    
    # Group by second (in case multiple trades at same second)
    changes_by_t = subset.groupby('t')['position_change'].sum()
    
    # Create full time series (0-899)
    full_index = pd.RangeIndex(0, MARKET_DURATION_SECONDS)
    position_changes = pd.Series(0.0, index=full_index)
    position_changes.loc[changes_by_t.index] = changes_by_t.values
    
    # Cumulative sum to get position
    position = position_changes.cumsum()
    
    return position


def compute_position_metrics(position: pd.Series) -> Dict[str, Any]:
    """Compute metrics from a position time series."""
    if len(position) == 0:
        return {
            'peak_long': 0.0,
            'peak_short': 0.0,
            'end_position': 0.0,
            'n_position_changes': 0,
            'mean_reversion_score': 0.0,
            'hold_to_expiry': False,
        }
    
    # Peak positions
    peak_long = float(position.max())
    peak_short = float(position.min())
    end_position = float(position.iloc[-1])
    
    # Count position changes (sign flips)
    signs = np.sign(position)
    sign_changes = (signs != signs.shift(1)).sum() - 1  # -1 for initial NaN
    n_position_changes = int(max(0, sign_changes))
    
    # Mean reversion score: how often does position move toward zero?
    pos_diff = position.diff().dropna()
    pos_sign = np.sign(position.shift(1)).dropna()
    
    if len(pos_diff) > 0 and len(pos_sign) > 0:
        # Reversion = position changed in opposite direction of current sign
        reversion_moves = ((pos_diff * pos_sign) < 0).sum()
        mean_reversion_score = float(reversion_moves / len(pos_diff)) if len(pos_diff) > 0 else 0.0
    else:
        mean_reversion_score = 0.0
    
    # Hold to expiry: did they still have position at end?
    hold_to_expiry = abs(end_position) > 0.001  # Small threshold for floating point
    
    return {
        'peak_long': peak_long,
        'peak_short': peak_short,
        'end_position': end_position,
        'n_position_changes': n_position_changes,
        'mean_reversion_score': mean_reversion_score,
        'hold_to_expiry': hold_to_expiry,
    }


def fifo_match_lots(
    trades: pd.DataFrame,
    wallet: str,
    market_id: str,
    outcome_token: str
) -> List[LotMatch]:
    """
    FIFO match buys to sells for a wallet/market/token combination.
    
    Returns:
        List of LotMatch objects representing matched round-trips.
    """
    # Filter to relevant trades
    mask = (
        (trades['wallet'] == wallet) &
        (trades['market_id'] == market_id) &
        (trades['outcome_token'] == outcome_token)
    )
    subset = trades[mask].copy()
    
    if len(subset) == 0:
        return []
    
    # Sort by time
    subset = subset.sort_values('t')
    
    # Separate buys and sells
    buys = subset[subset['side'] == 'BUY'][['t', 'price', 'size']].values.tolist()
    sells = subset[subset['side'] == 'SELL'][['t', 'price', 'size']].values.tolist()
    
    if len(buys) == 0 or len(sells) == 0:
        return []
    
    # FIFO matching
    matches = []
    buy_queue = []  # [(t, price, remaining_qty), ...]
    
    # Add all buys to queue
    for t, price, size in buys:
        buy_queue.append([t, price, size])
    
    # Match sells against buy queue
    for sell_t, sell_price, sell_qty in sells:
        remaining = sell_qty
        
        while remaining > 0 and buy_queue:
            buy_t, buy_price, buy_qty = buy_queue[0]
            
            # Skip if buy is after sell (can't match)
            if buy_t > sell_t:
                break
            
            # Match quantity
            match_qty = min(buy_qty, remaining)
            
            if match_qty > 0:
                hold_time = sell_t - buy_t
                pnl = (sell_price - buy_price) * match_qty
                
                matches.append(LotMatch(
                    buy_time=int(buy_t),
                    sell_time=int(sell_t),
                    hold_time=int(hold_time),
                    buy_price=buy_price,
                    sell_price=sell_price,
                    quantity=match_qty,
                    pnl=pnl,
                    pnl_per_unit=sell_price - buy_price,
                ))
            
            # Update remaining quantities
            buy_queue[0][2] -= match_qty
            remaining -= match_qty
            
            # Remove exhausted buy
            if buy_queue[0][2] <= 0:
                buy_queue.pop(0)
    
    return matches


def analyze_hold_times(matches: List[LotMatch]) -> Dict[str, Any]:
    """Analyze hold time distribution from matched lots."""
    if not matches:
        return {
            'n_matches': 0,
            'total_quantity': 0.0,
            'avg_hold_time': None,
            'median_hold_time': None,
            'p25_hold_time': None,
            'p75_hold_time': None,
            'min_hold_time': None,
            'max_hold_time': None,
            'total_pnl': 0.0,
            'avg_pnl_per_unit': None,
        }
    
    hold_times = [m.hold_time for m in matches]
    quantities = [m.quantity for m in matches]
    pnls = [m.pnl for m in matches]
    pnls_per_unit = [m.pnl_per_unit for m in matches]
    
    return {
        'n_matches': len(matches),
        'total_quantity': sum(quantities),
        'avg_hold_time': float(np.mean(hold_times)),
        'median_hold_time': float(np.median(hold_times)),
        'p25_hold_time': float(np.percentile(hold_times, 25)),
        'p75_hold_time': float(np.percentile(hold_times, 75)),
        'min_hold_time': int(min(hold_times)),
        'max_hold_time': int(max(hold_times)),
        'total_pnl': sum(pnls),
        'avg_pnl_per_unit': float(np.mean(pnls_per_unit)),
    }


def reconstruct_all_positions(trades: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Reconstruct positions for all wallet/market/token combinations.
    
    Returns:
        - enriched_trades: trades enriched with position context
        - position_summaries: per-wallet/market summaries
        - hold_time_stats: per-wallet hold time distributions
    """
    print("\nReconstructing positions...")
    
    # Get unique combinations
    combinations = trades.groupby(['wallet', 'market_id', 'outcome_token']).size().reset_index()
    print(f"  Found {len(combinations):,} wallet/market/token combinations")
    
    # Track results
    position_summaries = {}
    hold_time_stats = defaultdict(list)
    all_lot_matches = []
    
    # Process each combination
    for idx, row in combinations.iterrows():
        wallet = row['wallet']
        market_id = row['market_id']
        outcome_token = row['outcome_token']
        
        # Build position time series
        position = build_position_time_series(trades, wallet, market_id, outcome_token)
        
        # Compute position metrics
        pos_metrics = compute_position_metrics(position)
        
        # FIFO match lots
        lot_matches = fifo_match_lots(trades, wallet, market_id, outcome_token)
        
        # Analyze hold times
        hold_analysis = analyze_hold_times(lot_matches)
        
        # Get trade counts
        mask = (
            (trades['wallet'] == wallet) &
            (trades['market_id'] == market_id) &
            (trades['outcome_token'] == outcome_token)
        )
        subset = trades[mask]
        n_buys = (subset['side'] == 'BUY').sum()
        n_sells = (subset['side'] == 'SELL').sum()
        total_bought = subset[subset['side'] == 'BUY']['size'].sum()
        total_sold = subset[subset['side'] == 'SELL']['size'].sum()
        
        # Create summary
        summary = PositionSummary(
            wallet=wallet,
            market_id=market_id,
            outcome_token=outcome_token,
            n_buys=int(n_buys),
            n_sells=int(n_sells),
            total_bought=float(total_bought),
            total_sold=float(total_sold),
            net_position=float(total_bought - total_sold),
            peak_long=pos_metrics['peak_long'],
            peak_short=pos_metrics['peak_short'],
            end_position=pos_metrics['end_position'],
            n_position_changes=pos_metrics['n_position_changes'],
            mean_reversion_score=pos_metrics['mean_reversion_score'],
            hold_to_expiry=pos_metrics['hold_to_expiry'],
            n_round_trips=hold_analysis['n_matches'],
            avg_hold_time=hold_analysis['avg_hold_time'],
            median_hold_time=hold_analysis['median_hold_time'],
            total_realized_pnl=hold_analysis['total_pnl'],
        )
        
        key = f"{wallet}|{market_id}|{outcome_token}"
        position_summaries[key] = asdict(summary)
        
        # Track lot matches for enrichment
        for match in lot_matches:
            match_dict = asdict(match)
            match_dict['wallet'] = wallet
            match_dict['market_id'] = market_id
            match_dict['outcome_token'] = outcome_token
            all_lot_matches.append(match_dict)
        
        # Track hold times per wallet
        if lot_matches:
            hold_time_stats[wallet].extend([m.hold_time for m in lot_matches])
    
    # Aggregate hold time stats per wallet
    wallet_hold_stats = {}
    for wallet, hold_times in hold_time_stats.items():
        if hold_times:
            wallet_hold_stats[wallet] = {
                'n_round_trips': len(hold_times),
                'avg_hold_time': float(np.mean(hold_times)),
                'median_hold_time': float(np.median(hold_times)),
                'p25_hold_time': float(np.percentile(hold_times, 25)),
                'p75_hold_time': float(np.percentile(hold_times, 75)),
                'min_hold_time': int(min(hold_times)),
                'max_hold_time': int(max(hold_times)),
                'hold_time_distribution': {
                    '0-10s': sum(1 for h in hold_times if h <= 10),
                    '10-60s': sum(1 for h in hold_times if 10 < h <= 60),
                    '60-120s': sum(1 for h in hold_times if 60 < h <= 120),
                    '120-300s': sum(1 for h in hold_times if 120 < h <= 300),
                    '300-600s': sum(1 for h in hold_times if 300 < h <= 600),
                    '600-900s': sum(1 for h in hold_times if 600 < h <= 900),
                },
            }
        else:
            wallet_hold_stats[wallet] = {
                'n_round_trips': 0,
                'avg_hold_time': None,
                'median_hold_time': None,
            }
    
    print(f"  Processed {len(position_summaries):,} position summaries")
    print(f"  Found {len(all_lot_matches):,} lot matches across all wallets")
    
    return position_summaries, wallet_hold_stats, all_lot_matches


def enrich_trades_with_position(
    trades: pd.DataFrame,
    position_summaries: Dict
) -> pd.DataFrame:
    """
    Enrich trades with position-level features.
    """
    print("\nEnriching trades with position context...")
    trades = trades.copy()
    
    # Initialize position columns
    trades['inventory_state'] = 'UNKNOWN'
    trades['inventory_size'] = 0.0
    trades['inventory_age'] = 0
    trades['is_entry'] = False
    trades['is_exit'] = False
    
    # Group by wallet/market/token and compute position at each trade
    for (wallet, market_id, token), group in trades.groupby(['wallet', 'market_id', 'outcome_token']):
        idx = group.index
        subset = group.sort_values('t')
        
        # Compute running position
        position_change = np.where(subset['side'] == 'BUY', subset['size'], -subset['size'])
        cumulative_pos = np.cumsum(position_change)
        prev_pos = np.concatenate([[0], cumulative_pos[:-1]])
        
        # Inventory state
        inventory_state = np.where(
            prev_pos > 0.001, 'LONG',
            np.where(prev_pos < -0.001, 'SHORT', 'FLAT')
        )
        
        # Inventory size (absolute)
        inventory_size = np.abs(prev_pos)
        
        # Is this an entry (from flat) or exit (to flat)?
        current_pos = cumulative_pos
        is_entry = (np.abs(prev_pos) < 0.001) & (np.abs(current_pos) > 0.001)
        is_exit = (np.abs(prev_pos) > 0.001) & (np.abs(current_pos) < 0.001)
        
        # Assign back
        trades.loc[idx, 'inventory_state'] = inventory_state
        trades.loc[idx, 'inventory_size'] = inventory_size
        trades.loc[idx, 'is_entry'] = is_entry
        trades.loc[idx, 'is_exit'] = is_exit
    
    print(f"  Enriched {len(trades):,} trades")
    print(f"  Entry trades: {trades['is_entry'].sum():,}")
    print(f"  Exit trades: {trades['is_exit'].sum():,}")
    
    return trades


def compute_wallet_inventory_patterns(
    trades: pd.DataFrame,
    position_summaries: Dict
) -> Dict[str, Any]:
    """
    Compute high-level inventory patterns per wallet.
    """
    print("\nComputing wallet inventory patterns...")
    
    patterns = {}
    
    for wallet in trades['wallet'].unique():
        wallet_summaries = {k: v for k, v in position_summaries.items() if v['wallet'] == wallet}
        
        if not wallet_summaries:
            continue
        
        # Aggregate metrics
        total_markets = len(set(v['market_id'] for v in wallet_summaries.values()))
        
        # Hold-to-expiry ratio
        hold_to_expiry_count = sum(1 for v in wallet_summaries.values() if v['hold_to_expiry'])
        hold_to_expiry_ratio = hold_to_expiry_count / len(wallet_summaries) if wallet_summaries else 0
        
        # Two-sided vs one-sided
        market_tokens = defaultdict(set)
        for k, v in wallet_summaries.items():
            market_tokens[v['market_id']].add(v['outcome_token'])
        
        n_both_sides = sum(1 for tokens in market_tokens.values() if len(tokens) == 2)
        both_sides_ratio = n_both_sides / len(market_tokens) if market_tokens else 0
        
        # Mean reversion tendency
        avg_reversion_score = np.mean([v['mean_reversion_score'] for v in wallet_summaries.values()])
        
        # Net position bias
        net_positions = [v['net_position'] for v in wallet_summaries.values()]
        avg_net_position = np.mean(net_positions)
        net_position_bias = 'LONG' if avg_net_position > 1 else ('SHORT' if avg_net_position < -1 else 'NEUTRAL')
        
        # Round-trip frequency
        total_round_trips = sum(v['n_round_trips'] for v in wallet_summaries.values())
        
        patterns[wallet] = {
            'total_markets': total_markets,
            'total_positions': len(wallet_summaries),
            'hold_to_expiry_ratio': float(hold_to_expiry_ratio),
            'both_sides_ratio': float(both_sides_ratio),
            'avg_reversion_score': float(avg_reversion_score),
            'avg_net_position': float(avg_net_position),
            'net_position_bias': net_position_bias,
            'total_round_trips': total_round_trips,
            'scalp_ratio': total_round_trips / len(wallet_summaries) if wallet_summaries else 0,
        }
        
        print(f"\n  {wallet}:")
        print(f"    Markets: {total_markets}, Positions: {len(wallet_summaries)}")
        print(f"    Hold-to-expiry: {hold_to_expiry_ratio*100:.1f}%")
        print(f"    Both sides: {both_sides_ratio*100:.1f}%")
        print(f"    Reversion score: {avg_reversion_score:.3f}")
        print(f"    Round trips: {total_round_trips}")
    
    return patterns


def main():
    print("=" * 70)
    print("Phase 2: Position & Inventory Reconstruction")
    print("=" * 70)
    
    # Step 1: Load research table
    trades = load_research_table()
    
    # Step 2: Reconstruct all positions
    position_summaries, hold_time_stats, lot_matches = reconstruct_all_positions(trades)
    
    # Step 3: Enrich trades with position context
    enriched_trades = enrich_trades_with_position(trades, position_summaries)
    
    # Step 4: Compute wallet-level inventory patterns
    inventory_patterns = compute_wallet_inventory_patterns(enriched_trades, position_summaries)
    
    # Step 5: Save outputs
    print("\n" + "=" * 70)
    print("Saving outputs...")
    
    # Save enriched trades (positions.parquet)
    positions_path = DATA_DIR / "positions.parquet"
    enriched_trades.to_parquet(positions_path, index=False)
    print(f"  Enriched trades saved to: {positions_path}")
    print(f"  Shape: {enriched_trades.shape}")
    
    # Save position summaries
    summary_path = RESULTS_DIR / "position_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(position_summaries, f, indent=2, default=str)
    print(f"  Position summaries saved to: {summary_path}")
    
    # Save hold time distributions
    hold_time_path = RESULTS_DIR / "hold_time_distributions.json"
    with open(hold_time_path, 'w') as f:
        json.dump(hold_time_stats, f, indent=2, default=str)
    print(f"  Hold time distributions saved to: {hold_time_path}")
    
    # Save inventory patterns
    patterns_path = RESULTS_DIR / "inventory_patterns.json"
    with open(patterns_path, 'w') as f:
        json.dump(inventory_patterns, f, indent=2, default=str)
    print(f"  Inventory patterns saved to: {patterns_path}")
    
    # Save lot matches
    if lot_matches:
        lot_matches_df = pd.DataFrame(lot_matches)
        lot_matches_path = DATA_DIR / "lot_matches.parquet"
        lot_matches_df.to_parquet(lot_matches_path, index=False)
        print(f"  Lot matches saved to: {lot_matches_path}")
        print(f"  Shape: {lot_matches_df.shape}")
    
    print("\n" + "=" * 70)
    print("DONE - Phase 2 Complete")
    print("=" * 70)
    
    return enriched_trades, position_summaries, hold_time_stats


if __name__ == "__main__":
    main()

