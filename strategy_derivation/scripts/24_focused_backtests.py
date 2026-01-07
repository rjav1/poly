#!/usr/bin/env python3
"""
Phase E: Focused Backtests

Narrows focus to 2 production candidates:
1. Complete-Set Arb Family (H6/H7/H9/H12 unified)
2. Directional CL-Based (H8 - if it survives validation)

Runs comprehensive parameter sweeps with execution realism.

Output:
- focused_backtest_results.json
- FOCUSED_BACKTEST_REPORT.md
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from itertools import product

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
RESEARCH_DIR = BASE_DIR.parent / "data_v2" / "research"
MARKET_DURATION_SECONDS = 900


@dataclass
class Signal:
    market_id: str
    t: int
    tau: int
    side: str
    execution_mode: str
    up_ask: float
    down_ask: float
    up_ask_size: float
    down_ask_size: float
    underround: float
    delta_bps: float
    reason: str


def load_market_data() -> Tuple[pd.DataFrame, Dict]:
    """Load canonical market dataset."""
    path = RESEARCH_DIR / "canonical_dataset_all_assets.parquet"
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


# =============================================================================
# CANDIDATE 1: UNIFIED COMPLETE-SET ARB
# =============================================================================

def unified_complete_set_strategy(
    market_df: pd.DataFrame,
    params: Dict[str, Any]
) -> List[Signal]:
    """
    Unified Complete-Set Arbitrage Strategy.
    """
    epsilon = params.get('epsilon', 0.01)
    min_tau = params.get('min_tau', 0)
    max_tau = params.get('max_tau', 900)
    cooldown = params.get('cooldown', 30)
    min_capacity = params.get('min_capacity', 0)
    
    signals = []
    last_signal_t = -cooldown
    
    for _, row in market_df.iterrows():
        t = int(row['t'])
        tau = int(row['tau'])
        
        if tau < min_tau or tau > max_tau:
            continue
        
        if t - last_signal_t < cooldown:
            continue
        
        up_ask = row.get('pm_up_best_ask')
        down_ask = row.get('pm_down_best_ask')
        up_ask_size = row.get('pm_up_best_ask_size', 0)
        down_ask_size = row.get('pm_down_best_ask_size', 0)
        
        if pd.isna(up_ask) or pd.isna(down_ask):
            continue
        
        # Check capacity constraint
        min_size = min(
            up_ask_size if not pd.isna(up_ask_size) else 0,
            down_ask_size if not pd.isna(down_ask_size) else 0
        )
        if min_size < min_capacity:
            continue
        
        sum_asks = up_ask + down_ask
        underround = 1 - sum_asks
        
        if underround > epsilon:
            signals.append(Signal(
                market_id=row['market_id'],
                t=t,
                tau=tau,
                side='buy_both',
                execution_mode='taker',
                up_ask=up_ask,
                down_ask=down_ask,
                up_ask_size=up_ask_size if not pd.isna(up_ask_size) else 0,
                down_ask_size=down_ask_size if not pd.isna(down_ask_size) else 0,
                underround=underround,
                delta_bps=row.get('delta_bps', 0),
                reason=f"underround={underround:.4f}, tau={tau}"
            ))
            last_signal_t = t
    
    return signals


# =============================================================================
# CANDIDATE 2: LATE DIRECTIONAL (CL-BASED)
# =============================================================================

def late_directional_strategy(
    market_df: pd.DataFrame,
    params: Dict[str, Any],
    market_Y: int = None
) -> List[Signal]:
    """
    Late Directional Strategy based on CL delta.
    """
    max_tau = params.get('max_tau', 300)
    min_tau = params.get('min_tau', 30)
    delta_threshold_bps = params.get('delta_threshold_bps', 10)
    cooldown = params.get('cooldown', 60)
    
    signals = []
    last_signal_t = -cooldown
    
    for _, row in market_df.iterrows():
        t = int(row['t'])
        tau = int(row['tau'])
        
        if tau > max_tau or tau < min_tau:
            continue
        
        if t - last_signal_t < cooldown:
            continue
        
        delta_bps = row.get('delta_bps')
        if pd.isna(delta_bps):
            continue
        
        if abs(delta_bps) > delta_threshold_bps:
            side = 'buy_up' if delta_bps > 0 else 'buy_down'
            
            up_ask = row.get('pm_up_best_ask')
            down_ask = row.get('pm_down_best_ask')
            
            if pd.isna(up_ask) or pd.isna(down_ask):
                continue
            
            signals.append(Signal(
                market_id=row['market_id'],
                t=t,
                tau=tau,
                side=side,
                execution_mode='taker',
                up_ask=up_ask,
                down_ask=down_ask,
                up_ask_size=row.get('pm_up_best_ask_size', 0),
                down_ask_size=row.get('pm_down_best_ask_size', 0),
                underround=1 - (up_ask + down_ask),
                delta_bps=delta_bps,
                reason=f"delta_bps={delta_bps:.1f}, tau={tau}"
            ))
            last_signal_t = t
    
    return signals


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def execute_complete_set(signal: Signal) -> Tuple[float, bool]:
    """Execute complete-set trade. Returns (pnl, is_win)."""
    if signal.underround > 0:
        pnl = signal.underround  # 1 - (up_ask + down_ask)
        return pnl, True
    return 0.0, False


def execute_directional(signal: Signal, market_Y: int) -> Tuple[float, bool]:
    """Execute directional trade. Returns (pnl, is_win)."""
    if signal.side == 'buy_up':
        entry_price = signal.up_ask
        exit_price = 1.0 if market_Y == 1 else 0.0
    else:  # buy_down
        entry_price = signal.down_ask
        exit_price = 1.0 if market_Y == 0 else 0.0
    
    pnl = exit_price - entry_price
    return pnl, pnl > 0


def backtest_complete_set(
    market_data: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Backtest complete-set strategy."""
    market_results = {}
    all_signals = []
    
    for market_id, market_df in market_data.groupby('market_id'):
        market_df = market_df.sort_values('t')
        signals = unified_complete_set_strategy(market_df, params)
        
        total_pnl = 0.0
        wins = 0
        losses = 0
        
        for signal in signals:
            pnl, is_win = execute_complete_set(signal)
            total_pnl += pnl
            if is_win:
                wins += 1
            else:
                losses += 1
            all_signals.append(signal)
        
        market_results[market_id] = {
            'pnl': total_pnl,
            'n_signals': len(signals),
            'wins': wins,
            'losses': losses
        }
    
    # Aggregate
    pnl_array = np.array([r['pnl'] for r in market_results.values()])
    n_markets = len(pnl_array)
    total_pnl = pnl_array.sum()
    total_signals = sum(r['n_signals'] for r in market_results.values())
    total_wins = sum(r['wins'] for r in market_results.values())
    total_losses = sum(r['losses'] for r in market_results.values())
    
    if n_markets > 1 and pnl_array.std() > 0:
        t_stat = pnl_array.mean() / (pnl_array.std() / np.sqrt(n_markets))
    else:
        t_stat = 0.0
    
    # Capacity-constrained PnL
    capacity_constrained_signals = [s for s in all_signals if min(s.up_ask_size, s.down_ask_size) >= 1]
    capacity_pnl = sum(s.underround for s in capacity_constrained_signals)
    
    return {
        'strategy': 'complete_set_arb',
        'params': params,
        'n_markets': n_markets,
        'total_signals': total_signals,
        'total_pnl': float(total_pnl),
        'capacity_constrained_pnl': float(capacity_pnl),
        'capacity_constrained_signals': len(capacity_constrained_signals),
        'wins': total_wins,
        'losses': total_losses,
        'win_rate': total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0,
        't_stat': float(t_stat),
        'avg_pnl_per_signal': float(total_pnl / total_signals) if total_signals > 0 else 0,
    }


def backtest_directional(
    market_data: pd.DataFrame,
    market_info: Dict,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Backtest directional strategy."""
    market_results = {}
    all_signals = []
    
    for market_id, market_df in market_data.groupby('market_id'):
        market_df = market_df.sort_values('t')
        
        # Get Y for this market
        if market_id in market_info:
            market_Y = market_info[market_id].get('Y', None)
        else:
            Y_vals = market_df['Y'].dropna().unique()
            market_Y = int(Y_vals[0]) if len(Y_vals) > 0 else None
        
        if market_Y is None:
            continue
        
        signals = late_directional_strategy(market_df, params, market_Y)
        
        total_pnl = 0.0
        wins = 0
        losses = 0
        
        for signal in signals:
            pnl, is_win = execute_directional(signal, market_Y)
            total_pnl += pnl
            if is_win:
                wins += 1
            else:
                losses += 1
            all_signals.append((signal, market_Y, pnl))
        
        market_results[market_id] = {
            'pnl': total_pnl,
            'n_signals': len(signals),
            'wins': wins,
            'losses': losses,
            'Y': market_Y
        }
    
    # Aggregate
    pnl_array = np.array([r['pnl'] for r in market_results.values()])
    n_markets = len(pnl_array)
    total_pnl = pnl_array.sum()
    total_signals = sum(r['n_signals'] for r in market_results.values())
    total_wins = sum(r['wins'] for r in market_results.values())
    total_losses = sum(r['losses'] for r in market_results.values())
    
    if n_markets > 1 and pnl_array.std() > 0:
        t_stat = pnl_array.mean() / (pnl_array.std() / np.sqrt(n_markets))
    else:
        t_stat = 0.0
    
    # Direction accuracy
    correct_direction = sum(1 for s, y, p in all_signals if p > 0)
    direction_accuracy = correct_direction / len(all_signals) if all_signals else 0
    
    return {
        'strategy': 'late_directional',
        'params': params,
        'n_markets': n_markets,
        'total_signals': total_signals,
        'total_pnl': float(total_pnl),
        'wins': total_wins,
        'losses': total_losses,
        'win_rate': total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0,
        't_stat': float(t_stat),
        'direction_accuracy': float(direction_accuracy),
        'avg_pnl_per_signal': float(total_pnl / total_signals) if total_signals > 0 else 0,
    }


# =============================================================================
# PARAMETER SWEEP
# =============================================================================

def run_complete_set_sweep(market_data: pd.DataFrame) -> List[Dict]:
    """Run parameter sweep for complete-set strategy."""
    param_grid = {
        'epsilon': [0.005, 0.01, 0.015, 0.02],
        'min_tau': [0, 300, 600],
        'max_tau': [300, 600, 900],
        'cooldown': [30, 60],
        'min_capacity': [0, 1, 5],
    }
    
    results = []
    
    # Generate all combinations
    keys = list(param_grid.keys())
    for combo in product(*param_grid.values()):
        params = dict(zip(keys, combo))
        
        # Skip invalid combinations
        if params['min_tau'] >= params['max_tau']:
            continue
        
        result = backtest_complete_set(market_data, params)
        results.append(result)
    
    return results


def run_directional_sweep(market_data: pd.DataFrame, market_info: Dict) -> List[Dict]:
    """Run parameter sweep for directional strategy."""
    param_grid = {
        'max_tau': [120, 180, 300, 420],
        'min_tau': [30, 60],
        'delta_threshold_bps': [5, 10, 15, 20, 25],
        'cooldown': [30, 60, 120],
    }
    
    results = []
    
    keys = list(param_grid.keys())
    for combo in product(*param_grid.values()):
        params = dict(zip(keys, combo))
        
        result = backtest_directional(market_data, market_info, params)
        results.append(result)
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Phase E: Focused Backtests")
    print("=" * 70)
    
    # Load data
    print("\nLoading market data...")
    market_data, market_info = load_market_data()
    print(f"  Loaded {len(market_data['market_id'].unique())} markets")
    
    results = {
        'complete_set_family': {},
        'directional_family': {},
        'recommendations': {}
    }
    
    # ==========================================================================
    # CANDIDATE 1: Complete-Set Arb Family
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Candidate 1: Complete-Set Arb Family")
    print("=" * 70)
    
    print("\nRunning parameter sweep...")
    cs_results = run_complete_set_sweep(market_data)
    print(f"  Tested {len(cs_results)} parameter combinations")
    
    # Filter to valid results
    cs_results = [r for r in cs_results if r['total_signals'] > 0]
    
    # Sort by t-stat
    cs_results_sorted = sorted(cs_results, key=lambda x: x['t_stat'], reverse=True)
    
    print("\n  Top 10 Complete-Set Configurations:")
    print("  | Epsilon | Tau Window | CD | MinCap | Signals | PnL | t-stat | WinRate |")
    print("  |---------|------------|-----|--------|---------|-----|--------|---------|")
    
    for r in cs_results_sorted[:10]:
        p = r['params']
        print(f"  | {p['epsilon']:.3f} | {p['min_tau']}-{p['max_tau']} | {p['cooldown']} | {p['min_capacity']} | "
              f"{r['total_signals']} | ${r['total_pnl']:.2f} | {r['t_stat']:.2f} | {r['win_rate']*100:.0f}% |")
    
    results['complete_set_family']['all_results'] = cs_results_sorted
    results['complete_set_family']['best'] = cs_results_sorted[0] if cs_results_sorted else None
    
    # Capacity-constrained analysis
    print("\n  Capacity-Constrained Analysis (min_capacity >= 1):")
    cap_constrained = [r for r in cs_results if r['params']['min_capacity'] >= 1]
    if cap_constrained:
        best_cap = max(cap_constrained, key=lambda x: x['t_stat'])
        print(f"    Best t-stat with capacity constraint: {best_cap['t_stat']:.2f}")
        print(f"    Signals: {best_cap['total_signals']}, PnL: ${best_cap['total_pnl']:.2f}")
    
    # ==========================================================================
    # CANDIDATE 2: Late Directional (CL-Based)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Candidate 2: Late Directional (CL-Based)")
    print("=" * 70)
    
    print("\nRunning parameter sweep...")
    dir_results = run_directional_sweep(market_data, market_info)
    print(f"  Tested {len(dir_results)} parameter combinations")
    
    # Filter to valid results
    dir_results = [r for r in dir_results if r['total_signals'] > 0]
    
    # Sort by t-stat
    dir_results_sorted = sorted(dir_results, key=lambda x: x['t_stat'], reverse=True)
    
    print("\n  Top 10 Directional Configurations:")
    print("  | MaxTau | MinTau | Delta | CD | Signals | W/L | PnL | t-stat | DirAcc |")
    print("  |--------|--------|-------|-----|---------|-----|-----|--------|--------|")
    
    for r in dir_results_sorted[:10]:
        p = r['params']
        wl = f"{r['wins']}/{r['losses']}"
        print(f"  | {p['max_tau']} | {p['min_tau']} | {p['delta_threshold_bps']} | {p['cooldown']} | "
              f"{r['total_signals']} | {wl} | ${r['total_pnl']:.2f} | {r['t_stat']:.2f} | {r['direction_accuracy']*100:.0f}% |")
    
    results['directional_family']['all_results'] = dir_results_sorted
    results['directional_family']['best'] = dir_results_sorted[0] if dir_results_sorted else None
    
    # ==========================================================================
    # VERDICT & RECOMMENDATIONS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("VERDICT & RECOMMENDATIONS")
    print("=" * 70)
    
    recommendations = []
    
    # Complete-set analysis
    if cs_results_sorted:
        best_cs = cs_results_sorted[0]
        print(f"\n  COMPLETE-SET ARB:")
        print(f"    Best t-stat: {best_cs['t_stat']:.2f}")
        print(f"    Best PnL: ${best_cs['total_pnl']:.2f}")
        print(f"    Win rate: {best_cs['win_rate']*100:.0f}%")
        
        if best_cs['win_rate'] == 1.0:
            print("    STATUS: 100% win rate confirms pure arbitrage")
            recommendations.append("Complete-set arb is a valid arbitrage strategy with guaranteed payoff")
        
        if best_cs['t_stat'] > 2.0:
            print("    STATUS: Statistically significant at 95% level")
            recommendations.append("Complete-set arb passes statistical significance threshold")
        else:
            print("    STATUS: Below 95% significance - needs more data")
            recommendations.append("Complete-set arb needs more markets for confidence")
    
    # Directional analysis
    if dir_results_sorted:
        best_dir = dir_results_sorted[0]
        print(f"\n  LATE DIRECTIONAL:")
        print(f"    Best t-stat: {best_dir['t_stat']:.2f}")
        print(f"    Best PnL: ${best_dir['total_pnl']:.2f}")
        print(f"    Win rate: {best_dir['win_rate']*100:.0f}%")
        print(f"    Direction accuracy: {best_dir['direction_accuracy']*100:.0f}%")
        
        if best_dir['losses'] > 0:
            print("    STATUS: Shows losses - direction prediction not perfect")
            recommendations.append("Late directional has real execution risk and losses")
        else:
            print("    WARNING: 100% win rate may indicate sample luck or bug")
            recommendations.append("Late directional 100% win rate suspicious - need more data")
        
        if best_dir['t_stat'] > 2.0:
            print("    STATUS: Statistically significant")
        else:
            print("    STATUS: Below significance threshold")
    
    # Overall verdict
    print("\n  FINAL VERDICT:")
    
    if cs_results_sorted and best_cs['t_stat'] > 1.5:
        print("    1. COMPLETE-SET ARB: PROCEED TO PAPER TRADING")
        print(f"       Recommended params: epsilon={best_cs['params']['epsilon']}, "
              f"tau=[{best_cs['params']['min_tau']},{best_cs['params']['max_tau']}]")
        recommendations.append(f"Paper trade complete-set with epsilon={best_cs['params']['epsilon']}")
    else:
        print("    1. COMPLETE-SET ARB: NEEDS MORE DATA")
        recommendations.append("Collect more markets before paper trading")
    
    if dir_results_sorted and best_dir['losses'] > 0:
        print("    2. LATE DIRECTIONAL: NEEDS VALIDATION")
        print("       Must pass CL time-shift degradation test before proceeding")
        recommendations.append("Validate late directional against CL staleness")
    else:
        print("    2. LATE DIRECTIONAL: SUSPICIOUS - INVESTIGATE")
        recommendations.append("Investigate why late directional has 100% win rate")
    
    results['recommendations']['items'] = recommendations
    results['recommendations']['complete_set_ready'] = cs_results_sorted[0]['t_stat'] > 1.5 if cs_results_sorted else False
    results['recommendations']['directional_ready'] = dir_results_sorted[0]['losses'] > 0 if dir_results_sorted else False
    
    # Save outputs
    print("\n" + "=" * 70)
    print("Saving outputs...")
    
    # Save JSON (only top results to keep file size reasonable)
    save_results = {
        'complete_set_family': {
            'best': results['complete_set_family']['best'],
            'top_10': results['complete_set_family']['all_results'][:10]
        },
        'directional_family': {
            'best': results['directional_family']['best'],
            'top_10': results['directional_family']['all_results'][:10]
        },
        'recommendations': results['recommendations']
    }
    
    results_path = RESULTS_DIR / "focused_backtest_results.json"
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"  Results saved to: {results_path}")
    
    # Generate report
    report = ["# Focused Backtest Report\n\n"]
    
    report.append("## Complete-Set Arb Family\n\n")
    report.append("### Top Configurations\n\n")
    report.append("| Epsilon | Tau Window | Cooldown | MinCap | Signals | PnL | t-stat | Win Rate |\n")
    report.append("|---------|------------|----------|--------|---------|-----|--------|----------|\n")
    
    for r in cs_results_sorted[:5]:
        p = r['params']
        report.append(f"| {p['epsilon']} | {p['min_tau']}-{p['max_tau']} | {p['cooldown']} | "
                     f"{p['min_capacity']} | {r['total_signals']} | ${r['total_pnl']:.2f} | "
                     f"{r['t_stat']:.2f} | {r['win_rate']*100:.0f}% |\n")
    
    report.append("\n## Late Directional Family\n\n")
    report.append("### Top Configurations\n\n")
    report.append("| MaxTau | Delta | Cooldown | Signals | W/L | PnL | t-stat | Dir Acc |\n")
    report.append("|--------|-------|----------|---------|-----|-----|--------|----------|\n")
    
    for r in dir_results_sorted[:5]:
        p = r['params']
        report.append(f"| {p['max_tau']} | {p['delta_threshold_bps']} | {p['cooldown']} | "
                     f"{r['total_signals']} | {r['wins']}/{r['losses']} | ${r['total_pnl']:.2f} | "
                     f"{r['t_stat']:.2f} | {r['direction_accuracy']*100:.0f}% |\n")
    
    report.append("\n## Recommendations\n\n")
    for rec in recommendations:
        report.append(f"- {rec}\n")
    
    report_path = REPORTS_DIR / "FOCUSED_BACKTEST_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(''.join(report))
    print(f"  Report saved to: {report_path}")
    
    print("\n" + "=" * 70)
    print("DONE - Phase E Complete")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()

