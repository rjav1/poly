#!/usr/bin/env python3
"""
Shadow Trader for Complete-Set Arb

Paper trades the complete-set arbitrage strategy to answer:
"When a signal triggers in live conditions, would you actually have been able to fill both legs?"

Logs per signal:
- timestamp, market, tau
- underround magnitude
- L1/L2 book snapshot (UP asks + DOWN asks + sizes)
- simulated fills under taker/maker models
- post-signal evolution (did underround persist?)

Stop condition: 100+ signals observed

Output:
- shadow_trader_log.json (all signals with full details)
- shadow_trader_report.md (fill rates, realized edge, untradeable rate)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
RESEARCH_DIR = BASE_DIR.parent / "data_v2" / "research"
MARKET_DURATION_SECONDS = 900


@dataclass
class BookSnapshot:
    """L1/L2 book snapshot at signal time."""
    # UP side
    up_ask_1: float
    up_ask_size_1: float
    up_bid_1: float
    up_bid_size_1: float
    
    # DOWN side
    down_ask_1: float
    down_ask_size_1: float
    down_bid_1: float
    down_bid_size_1: float
    
    # Aggregated
    sum_asks: float
    underround: float
    min_capacity: float  # min(up_ask_size_1, down_ask_size_1)


@dataclass
class FillSimulation:
    """Simulated fill results for different execution models."""
    taker_filled: bool
    taker_fill_price: float
    taker_pnl: float
    
    maker_conservative_filled: bool
    maker_conservative_fill_price: float
    maker_conservative_pnl: float
    maker_conservative_fill_window_seconds: int
    
    maker_realistic_filled: bool
    maker_realistic_fill_price: float
    maker_realistic_pnl: float
    
    capacity_limited: bool  # True if min_capacity < 1


@dataclass
class PostSignalEvolution:
    """Track how the underround evolved after signal."""
    # Check at +5s, +10s, +30s, +60s
    persisted_5s: bool  # Underround still exists at t+5s
    persisted_10s: bool
    persisted_30s: bool
    persisted_60s: bool
    
    disappeared_at_seconds: Optional[int]  # When underround disappeared, or None
    max_persist_seconds: int  # How long it persisted


@dataclass
class SignalLog:
    """Complete log entry for a signal."""
    signal_id: int
    timestamp: str
    market_id: str
    t: int  # seconds since market start
    tau: int  # seconds until expiry
    
    # Signal details
    underround_magnitude: float
    book_snapshot: BookSnapshot
    
    # Fill simulation
    fill_simulation: FillSimulation
    
    # Post-signal evolution
    post_evolution: PostSignalEvolution
    
    # Outcome
    actual_outcome: int  # Y (0 or 1)
    realized_pnl_taker: float
    realized_pnl_maker_cons: float
    realized_pnl_maker_real: float


def load_market_data() -> Tuple[pd.DataFrame, Dict]:
    """Load canonical market dataset."""
    path = RESEARCH_DIR / "canonical_dataset_all_assets.parquet"
    print(f"Loading market data from: {path}")
    df = pd.read_parquet(path)
    
    info_path = RESEARCH_DIR / "market_info_all_assets.json"
    with open(info_path, 'r') as f:
        market_info = json.load(f)
    
    if isinstance(market_info, list):
        market_info_dict = {}
        for item in market_info:
            mid = item.get('market_id', item.get('condition_id', ''))
            market_info_dict[mid] = item
        market_info = market_info_dict
    
    return df, market_info


def load_shadow_spec() -> Dict[str, Any]:
    """Load shadow trader specification."""
    path = RESULTS_DIR / "shadow_trader_spec.json"
    with open(path, 'r') as f:
        return json.load(f)


def get_book_snapshot(row: pd.Series) -> BookSnapshot:
    """Extract L1/L2 book snapshot from a market data row."""
    # L1 (best level)
    up_ask_1 = row.get('pm_up_best_ask', np.nan)
    up_ask_size_1 = row.get('pm_up_best_ask_size', 0)
    up_bid_1 = row.get('pm_up_best_bid', np.nan)
    up_bid_size_1 = row.get('pm_up_best_bid_size', 0)
    
    down_ask_1 = row.get('pm_down_best_ask', np.nan)
    down_ask_size_1 = row.get('pm_down_best_ask_size', 0)
    down_bid_1 = row.get('pm_down_best_bid', np.nan)
    down_bid_size_1 = row.get('pm_down_best_bid_size', 0)
    
    # Handle NaN
    up_ask_1 = up_ask_1 if not pd.isna(up_ask_1) else 0.0
    down_ask_1 = down_ask_1 if not pd.isna(down_ask_1) else 0.0
    up_ask_size_1 = up_ask_size_1 if not pd.isna(up_ask_size_1) else 0.0
    down_ask_size_1 = down_ask_size_1 if not pd.isna(down_ask_size_1) else 0.0
    
    sum_asks = up_ask_1 + down_ask_1
    underround = max(0.0, 1.0 - sum_asks)
    min_capacity = min(up_ask_size_1, down_ask_size_1)
    
    return BookSnapshot(
        up_ask_1=float(up_ask_1),
        up_ask_size_1=float(up_ask_size_1),
        up_bid_1=float(up_bid_1) if not pd.isna(up_bid_1) else 0.0,
        up_bid_size_1=float(up_bid_size_1) if not pd.isna(up_bid_size_1) else 0.0,
        down_ask_1=float(down_ask_1),
        down_ask_size_1=float(down_ask_size_1),
        down_bid_1=float(down_bid_1) if not pd.isna(down_bid_1) else 0.0,
        down_bid_size_1=float(down_bid_size_1) if not pd.isna(down_bid_size_1) else 0.0,
        sum_asks=float(sum_asks),
        underround=float(underround),
        min_capacity=float(min_capacity)
    )


def simulate_taker_fill(book: BookSnapshot) -> Tuple[bool, float, float]:
    """
    Taker fill simulation: immediate fill at best ask.
    
    Returns: (filled, fill_price, pnl)
    """
    if book.min_capacity < 1.0:
        return False, 0.0, 0.0
    
    if book.underround <= 0:
        return False, 0.0, 0.0
    
    fill_price = book.up_ask_1 + book.down_ask_1
    pnl = 1.0 - fill_price  # Guaranteed $1 at expiry
    
    return True, fill_price, pnl


def simulate_maker_conservative_fill(
    signal_t: int,
    signal_tau: int,
    book: BookSnapshot,
    future_data: pd.DataFrame,
    fill_window_seconds: int = 30
) -> Tuple[bool, float, float, int]:
    """
    Conservative maker fill: limit order fills only if price crosses within window.
    
    Returns: (filled, fill_price, pnl, fill_time_seconds)
    """
    if book.underround <= 0:
        return False, 0.0, 0.0, 0
    
    # Post limit orders slightly below current ask
    limit_offset = 0.005  # $0.005 below ask
    up_limit = max(0.01, book.up_ask_1 - limit_offset)
    down_limit = max(0.01, book.down_ask_1 - limit_offset)
    
    # Check future data for fills
    for _, future_row in future_data.iterrows():
        future_t = int(future_row['t'])
        seconds_after_signal = future_t - signal_t
        
        if seconds_after_signal > fill_window_seconds:
            break
        
        # Check if both legs can fill
        future_up_ask = future_row.get('pm_up_best_ask', 999)
        future_down_ask = future_row.get('pm_down_best_ask', 999)
        
        up_filled = not pd.isna(future_up_ask) and future_up_ask <= up_limit
        down_filled = not pd.isna(future_down_ask) and future_down_ask <= down_limit
        
        if up_filled and down_filled:
            fill_price = up_limit + down_limit
            pnl = 1.0 - fill_price
            return True, fill_price, pnl, seconds_after_signal
    
    return False, 0.0, 0.0, 0


def simulate_maker_realistic_fill(
    signal_tau: int,
    book: BookSnapshot,
    base_fill_prob: float = 0.6
) -> Tuple[bool, float, float]:
    """
    Realistic maker fill: probabilistic with time/competition factors.
    
    Returns: (filled, fill_price, pnl)
    """
    if book.underround <= 0:
        return False, 0.0, 0.0
    
    # Estimate fill probability
    tau_factor = min(1.0, signal_tau / 300)  # More time = higher prob
    underround_factor = min(1.0, book.underround / 0.02)  # Larger underround = higher prob
    
    fill_prob = base_fill_prob * tau_factor * underround_factor
    
    # Deterministic threshold (in production would use Monte Carlo)
    if fill_prob > 0.5:
        # Assume fill with slight improvement over taker
        improvement = 0.003  # $0.003 per leg
        fill_price = book.up_ask_1 + book.down_ask_1 - 2 * improvement
        pnl = 1.0 - fill_price
        return True, fill_price, pnl
    else:
        return False, 0.0, 0.0


def track_post_signal_evolution(
    signal_t: int,
    signal_market_id: str,
    initial_underround: float,
    market_df: pd.DataFrame
) -> PostSignalEvolution:
    """Track how underround evolves after signal."""
    # Get future rows
    future_rows = market_df[
        (market_df['t'] > signal_t) & 
        (market_df['t'] <= signal_t + 60)
    ].sort_values('t')
    
    persisted_5s = False
    persisted_10s = False
    persisted_30s = False
    persisted_60s = False
    disappeared_at = None
    max_persist = 0
    
    check_times = [5, 10, 30, 60]
    check_results = {}
    
    for seconds in check_times:
        check_row = market_df[market_df['t'] == signal_t + seconds]
        if not check_row.empty:
            check_row = check_row.iloc[0]
            future_up_ask = check_row.get('pm_up_best_ask', np.nan)
            future_down_ask = check_row.get('pm_down_best_ask', np.nan)
            
            if not pd.isna(future_up_ask) and not pd.isna(future_down_ask):
                future_sum = future_up_ask + future_down_ask
                future_underround = 1.0 - future_sum
                
                check_results[seconds] = future_underround > 0.001  # Still has underround
                if seconds == 5:
                    persisted_5s = check_results[seconds]
                elif seconds == 10:
                    persisted_10s = check_results[seconds]
                elif seconds == 30:
                    persisted_30s = check_results[seconds]
                elif seconds == 60:
                    persisted_60s = check_results[seconds]
            else:
                check_results[seconds] = False
    
    # Find when underround disappeared
    for seconds in check_times:
        if seconds in check_results and not check_results[seconds]:
            disappeared_at = seconds
            max_persist = seconds - 5  # Approximate
            break
    else:
        max_persist = 60  # Persisted at least 60s
    
    return PostSignalEvolution(
        persisted_5s=persisted_5s,
        persisted_10s=persisted_10s,
        persisted_30s=persisted_30s,
        persisted_60s=persisted_60s,
        disappeared_at_seconds=disappeared_at,
        max_persist_seconds=max_persist
    )


def generate_signals(
    market_df: pd.DataFrame,
    spec: Dict[str, Any]
) -> List[Tuple[pd.Series, pd.DataFrame]]:
    """Generate signals based on specification."""
    strategy_spec = next(s for s in spec['strategies'] if s['name'] == 'complete_set_arb')
    conditions = strategy_spec['signal_conditions']
    
    epsilon = conditions['underround_threshold']
    min_tau = conditions['min_tau']
    max_tau = conditions['max_tau']
    cooldown = conditions['cooldown_seconds']
    min_capacity = conditions.get('min_capacity', 0)
    
    signals = []
    last_signal_t = -cooldown
    
    for _, row in market_df.iterrows():
        t = int(row['t'])
        tau = int(row['tau'])
        
        # Check conditions
        if tau < min_tau or tau > max_tau:
            continue
        
        if t - last_signal_t < cooldown:
            continue
        
        # Get book snapshot
        book = get_book_snapshot(row)
        
        # Check underround
        if book.underround <= epsilon:
            continue
        
        # Check capacity
        if book.min_capacity < min_capacity:
            continue
        
        signals.append((row, market_df))
        last_signal_t = t
    
    return signals


def run_shadow_trader(
    market_data: pd.DataFrame,
    market_info: Dict,
    spec: Dict[str, Any],
    max_signals: int = 100
) -> List[SignalLog]:
    """Run shadow trader across all markets."""
    all_signals = []
    signal_id = 0
    
    print(f"\nRunning shadow trader (target: {max_signals} signals)...")
    
    for market_id, market_df in market_data.groupby('market_id'):
        market_df = market_df.sort_values('t')
        
        # Get market Y
        if market_id in market_info:
            market_Y = market_info[market_id].get('Y', 0)
        else:
            Y_vals = market_df['Y'].dropna().unique()
            market_Y = int(Y_vals[0]) if len(Y_vals) > 0 else 0
        
        # Generate signals for this market
        signals = generate_signals(market_df, spec)
        
        for signal_row, future_df in signals:
            if signal_id >= max_signals:
                break
            
            signal_id += 1
            t = int(signal_row['t'])
            tau = int(signal_row['tau'])
            
            # Get book snapshot
            book = get_book_snapshot(signal_row)
            
            # Simulate fills
            taker_filled, taker_price, taker_pnl = simulate_taker_fill(book)
            
            maker_cons_filled, maker_cons_price, maker_cons_pnl, maker_cons_time = simulate_maker_conservative_fill(
                t, tau, book, future_df
            )
            
            maker_real_filled, maker_real_price, maker_real_pnl = simulate_maker_realistic_fill(
                tau, book
            )
            
            fill_sim = FillSimulation(
                taker_filled=taker_filled,
                taker_fill_price=taker_price if taker_filled else 0.0,
                taker_pnl=taker_pnl if taker_filled else 0.0,
                maker_conservative_filled=maker_cons_filled,
                maker_conservative_fill_price=maker_cons_price if maker_cons_filled else 0.0,
                maker_conservative_pnl=maker_cons_pnl if maker_cons_filled else 0.0,
                maker_conservative_fill_window_seconds=maker_cons_time,
                maker_realistic_filled=maker_real_filled,
                maker_realistic_fill_price=maker_real_price if maker_real_filled else 0.0,
                maker_realistic_pnl=maker_real_pnl if maker_real_filled else 0.0,
                capacity_limited=book.min_capacity < 1.0
            )
            
            # Track post-signal evolution
            post_evolution = track_post_signal_evolution(
                t, market_id, book.underround, market_df
            )
            
            # Realized PnL (all strategies hold to expiry for complete-set)
            realized_pnl_taker = taker_pnl if taker_filled else 0.0
            realized_pnl_maker_cons = maker_cons_pnl if maker_cons_filled else 0.0
            realized_pnl_maker_real = maker_real_pnl if maker_real_filled else 0.0
            
            # Create signal log
            signal_log = SignalLog(
                signal_id=signal_id,
                timestamp=signal_row.get('timestamp', f't={t}'),
                market_id=market_id,
                t=t,
                tau=tau,
                underround_magnitude=book.underround,
                book_snapshot=book,
                fill_simulation=fill_sim,
                post_evolution=post_evolution,
                actual_outcome=market_Y,
                realized_pnl_taker=realized_pnl_taker,
                realized_pnl_maker_cons=realized_pnl_maker_cons,
                realized_pnl_maker_real=realized_pnl_maker_real
            )
            
            all_signals.append(signal_log)
            
            if signal_id % 10 == 0:
                print(f"  Processed {signal_id} signals...")
            
            if signal_id >= max_signals:
                break
        
        if signal_id >= max_signals:
            break
    
    print(f"  Completed: {len(all_signals)} signals processed")
    return all_signals


def generate_shadow_report(signals: List[SignalLog]) -> str:
    """Generate shadow trader report."""
    report = ["# Shadow Trader Report - Complete-Set Arb\n\n"]
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report.append(f"**Total Signals Observed**: {len(signals)}\n\n")
    
    report.append("---\n\n")
    
    # Fill Rate Analysis
    report.append("## Fill Rate Analysis\n\n")
    
    taker_fills = sum(1 for s in signals if s.fill_simulation.taker_filled)
    maker_cons_fills = sum(1 for s in signals if s.fill_simulation.maker_conservative_filled)
    maker_real_fills = sum(1 for s in signals if s.fill_simulation.maker_realistic_filled)
    
    taker_fill_rate = taker_fills / len(signals) if signals else 0
    maker_cons_fill_rate = maker_cons_fills / len(signals) if signals else 0
    maker_real_fill_rate = maker_real_fills / len(signals) if signals else 0
    
    report.append("| Execution Model | Filled | Total | Fill Rate |\n")
    report.append("|----------------|--------|-------|-----------|\n")
    report.append(f"| Taker | {taker_fills} | {len(signals)} | {taker_fill_rate*100:.1f}% |\n")
    report.append(f"| Maker (Conservative) | {maker_cons_fills} | {len(signals)} | {maker_cons_fill_rate*100:.1f}% |\n")
    report.append(f"| Maker (Realistic) | {maker_real_fills} | {len(signals)} | {maker_real_fill_rate*100:.1f}% |\n\n")
    
    # Realized Edge Analysis
    report.append("## Realized Edge Analysis\n\n")
    
    taker_filled_signals = [s for s in signals if s.fill_simulation.taker_filled]
    maker_cons_filled_signals = [s for s in signals if s.fill_simulation.maker_conservative_filled]
    maker_real_filled_signals = [s for s in signals if s.fill_simulation.maker_realistic_filled]
    
    if taker_filled_signals:
        avg_edge_taker = np.mean([s.realized_pnl_taker for s in taker_filled_signals])
        total_pnl_taker = sum(s.realized_pnl_taker for s in taker_filled_signals)
    else:
        avg_edge_taker = 0.0
        total_pnl_taker = 0.0
    
    if maker_cons_filled_signals:
        avg_edge_maker_cons = np.mean([s.realized_pnl_maker_cons for s in maker_cons_filled_signals])
        total_pnl_maker_cons = sum(s.realized_pnl_maker_cons for s in maker_cons_filled_signals)
    else:
        avg_edge_maker_cons = 0.0
        total_pnl_maker_cons = 0.0
    
    if maker_real_filled_signals:
        avg_edge_maker_real = np.mean([s.realized_pnl_maker_real for s in maker_real_filled_signals])
        total_pnl_maker_real = sum(s.realized_pnl_maker_real for s in maker_real_filled_signals)
    else:
        avg_edge_maker_real = 0.0
        total_pnl_maker_real = 0.0
    
    report.append("| Execution Model | Filled Signals | Avg Edge/Fill | Total PnL |\n")
    report.append("|----------------|----------------|---------------|-----------|\n")
    report.append(f"| Taker | {len(taker_filled_signals)} | ${avg_edge_taker:.4f} | ${total_pnl_taker:.2f} |\n")
    report.append(f"| Maker (Conservative) | {len(maker_cons_filled_signals)} | ${avg_edge_maker_cons:.4f} | ${total_pnl_maker_cons:.2f} |\n")
    report.append(f"| Maker (Realistic) | {len(maker_real_filled_signals)} | ${avg_edge_maker_real:.4f} | ${total_pnl_maker_real:.2f} |\n\n")
    
    # Untradeable Analysis
    report.append("## Untradeable Rate Analysis\n\n")
    
    capacity_limited = sum(1 for s in signals if s.fill_simulation.capacity_limited)
    untradeable_pct = capacity_limited / len(signals) * 100 if signals else 0
    
    report.append(f"- **Capacity Limited (< 1 contract)**: {capacity_limited}/{len(signals)} ({untradeable_pct:.1f}%)\n\n")
    
    # Post-Signal Evolution
    report.append("## Post-Signal Evolution\n\n")
    
    persisted_5s = sum(1 for s in signals if s.post_evolution.persisted_5s)
    persisted_10s = sum(1 for s in signals if s.post_evolution.persisted_10s)
    persisted_30s = sum(1 for s in signals if s.post_evolution.persisted_30s)
    persisted_60s = sum(1 for s in signals if s.post_evolution.persisted_60s)
    
    report.append("| Time After Signal | Still Has Underround | % of Signals |\n")
    report.append("|-------------------|----------------------|--------------|\n")
    report.append(f"| +5 seconds | {persisted_5s} | {persisted_5s/len(signals)*100:.1f}% |\n")
    report.append(f"| +10 seconds | {persisted_10s} | {persisted_10s/len(signals)*100:.1f}% |\n")
    report.append(f"| +30 seconds | {persisted_30s} | {persisted_30s/len(signals)*100:.1f}% |\n")
    report.append(f"| +60 seconds | {persisted_60s} | {persisted_60s/len(signals)*100:.1f}% |\n\n")
    
    # Go/No-Go Decision
    report.append("## Go/No-Go Decision\n\n")
    
    report.append("### Criteria:\n")
    report.append("1. Fill rate > 30% (taker)\n")
    report.append("2. Average realized edge per fill > $0.01\n")
    report.append("3. Untradeable rate < 70%\n\n")
    
    criteria_met = []
    
    if taker_fill_rate > 0.3:
        criteria_met.append("[PASS] Fill rate > 30%")
        report.append(f"- **Fill rate**: {taker_fill_rate*100:.1f}% [PASS]\n")
    else:
        criteria_met.append("[FAIL] Fill rate < 30%")
        report.append(f"- **Fill rate**: {taker_fill_rate*100:.1f}% [FAIL]\n")
    
    if avg_edge_taker > 0.01:
        criteria_met.append("[PASS] Avg edge > $0.01")
        report.append(f"- **Average edge per fill**: ${avg_edge_taker:.4f} [PASS]\n")
    else:
        criteria_met.append("[FAIL] Avg edge < $0.01")
        report.append(f"- **Average edge per fill**: ${avg_edge_taker:.4f} [FAIL]\n")
    
    if untradeable_pct < 70:
        criteria_met.append("[PASS] Untradeable < 70%")
        report.append(f"- **Untradeable rate**: {untradeable_pct:.1f}% [PASS]\n")
    else:
        criteria_met.append("[FAIL] Untradeable > 70%")
        report.append(f"- **Untradeable rate**: {untradeable_pct:.1f}% [FAIL]\n")
    
    report.append("\n### Verdict:\n\n")
    
    if all("[PASS]" in c for c in criteria_met):
        report.append("**GO FOR SMALL CAPITAL DEPLOYMENT**\n\n")
        report.append("All criteria met. Strategy is tradeable with:\n")
        report.append(f"- {taker_fill_rate*100:.0f}% fill rate\n")
        report.append(f"- ${avg_edge_taker:.4f} average edge per fill\n")
        report.append(f"- {untradeable_pct:.0f}% untradeable rate\n")
    else:
        report.append("**NO-GO - NEEDS MORE WORK**\n\n")
        report.append("Criteria not met:\n")
        for c in criteria_met:
            if "[FAIL]" in c:
                report.append(f"- {c}\n")
    
    return ''.join(report)


def main():
    print("=" * 70)
    print("Shadow Trader - Complete-Set Arb")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    market_data, market_info = load_market_data()
    spec = load_shadow_spec()
    
    print(f"  Loaded {len(market_data['market_id'].unique())} markets")
    print(f"  Loaded shadow spec with {len(spec['strategies'])} strategies")
    
    # Run shadow trader
    signals = run_shadow_trader(market_data, market_info, spec, max_signals=100)
    
    # Generate report
    print("\nGenerating report...")
    report = generate_shadow_report(signals)
    
    # Save outputs
    print("\n" + "=" * 70)
    print("Saving outputs...")
    
    # Save signal log
    signals_dict = [asdict(s) for s in signals]
    log_path = RESULTS_DIR / "shadow_trader_log.json"
    with open(log_path, 'w') as f:
        json.dump({
            'n_signals': len(signals),
            'generated': datetime.now().isoformat(),
            'signals': signals_dict
        }, f, indent=2, default=str)
    print(f"  Signal log saved to: {log_path}")
    
    # Save report
    report_path = REPORTS_DIR / "shadow_trader_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Report saved to: {report_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SHADOW TRADER SUMMARY")
    print("=" * 70)
    
    taker_fills = sum(1 for s in signals if s.fill_simulation.taker_filled)
    maker_cons_fills = sum(1 for s in signals if s.fill_simulation.maker_conservative_filled)
    capacity_limited = sum(1 for s in signals if s.fill_simulation.capacity_limited)
    
    taker_filled_signals = [s for s in signals if s.fill_simulation.taker_filled]
    avg_edge = np.mean([s.realized_pnl_taker for s in taker_filled_signals]) if taker_filled_signals else 0.0
    
    print(f"\n  Signals observed: {len(signals)}")
    print(f"  Taker fill rate: {taker_fills/len(signals)*100:.1f}% ({taker_fills}/{len(signals)})")
    print(f"  Maker (conservative) fill rate: {maker_cons_fills/len(signals)*100:.1f}% ({maker_cons_fills}/{len(signals)})")
    print(f"  Capacity limited: {capacity_limited}/{len(signals)} ({capacity_limited/len(signals)*100:.1f}%)")
    
    if taker_filled_signals:
        print(f"\n  Average realized edge per fill: ${avg_edge:.4f}")
        print(f"  Total PnL (if all fills executed): ${sum(s.realized_pnl_taker for s in taker_filled_signals):.2f}")
    
    print("\n" + "=" * 70)
    print("DONE - Shadow Trader Complete")
    print("=" * 70)
    
    return signals


if __name__ == "__main__":
    main()

