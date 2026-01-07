#!/usr/bin/env python3
"""
Phase C: Strategy-Class Aware Validation

Fixes the validation suite to apply correct logic based on strategy category:
- PM_ONLY strategies: CL time-shift should NOT affect edge (test passes if edge persists)
- CL-dependent strategies: CL time-shift SHOULD degrade edge (test passes if edge degrades)

Also fixes permutation test logic for complete-set arbitrage strategies.

Output:
- validation_results_fixed.json
- VALIDATION_REPORT_FIXED.md
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Callable
from dataclasses import dataclass
import sys

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
    side: str
    size: float
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


def load_hypotheses() -> List[Dict]:
    """Load hypotheses with categories."""
    path = RESULTS_DIR / "hypotheses.json"
    with open(path, 'r') as f:
        return json.load(f)


def load_backtest_results() -> Dict:
    """Load backtest results."""
    path = RESULTS_DIR / "backtest_results.json"
    with open(path, 'r') as f:
        return json.load(f)


def get_strategy_category(hypotheses: List[Dict], hyp_id: str) -> str:
    """Get category for a hypothesis."""
    for hyp in hypotheses:
        if hyp['hypothesis_id'] == hyp_id:
            return hyp.get('category', 'UNKNOWN')
    return 'UNKNOWN'


# =============================================================================
# STRATEGY IMPLEMENTATIONS
# =============================================================================

def strategy_underround_harvest(market_df: pd.DataFrame, params: Dict) -> List[Signal]:
    epsilon = params.get('epsilon', 0.01)
    min_tau = params.get('min_tau', 60)
    max_tau = params.get('max_tau', 840)
    cooldown = params.get('cooldown', 30)
    
    signals = []
    last_signal_t = -cooldown
    
    for _, row in market_df.iterrows():
        t = int(row['t'])
        tau = int(row['tau'])
        if tau < min_tau or tau > max_tau:
            continue
        if t - last_signal_t < cooldown:
            continue
        sum_asks = row.get('sum_asks')
        if pd.isna(sum_asks):
            continue
        underround = 1 - sum_asks
        if underround > epsilon:
            signals.append(Signal(row['market_id'], t, 'buy_both', 1.0, f"underround={underround:.4f}"))
            last_signal_t = t
    return signals


def strategy_late_directional(market_df: pd.DataFrame, params: Dict) -> List[Signal]:
    max_tau = params.get('max_tau', 300)
    delta_threshold_bps = params.get('delta_threshold_bps', 10)
    cooldown = params.get('cooldown', 60)
    
    signals = []
    last_signal_t = -cooldown
    
    for _, row in market_df.iterrows():
        t = int(row['t'])
        tau = int(row['tau'])
        if tau > max_tau or tau < 30:
            continue
        if t - last_signal_t < cooldown:
            continue
        delta_bps = row.get('delta_bps')
        if pd.isna(delta_bps):
            continue
        if abs(delta_bps) > delta_threshold_bps:
            side = 'buy_up' if delta_bps > 0 else 'buy_down'
            signals.append(Signal(row['market_id'], t, side, 1.0, f"delta_bps={delta_bps:.1f}"))
            last_signal_t = t
    return signals


def strategy_early_inventory(market_df: pd.DataFrame, params: Dict) -> List[Signal]:
    min_tau = params.get('min_tau', 600)
    epsilon = params.get('epsilon', 0.015)
    cooldown = params.get('cooldown', 60)
    
    signals = []
    last_signal_t = -cooldown
    
    for _, row in market_df.iterrows():
        t = int(row['t'])
        tau = int(row['tau'])
        if tau < min_tau:
            continue
        if t - last_signal_t < cooldown:
            continue
        sum_asks = row.get('sum_asks')
        if pd.isna(sum_asks):
            continue
        underround = 1 - sum_asks
        if underround > epsilon:
            signals.append(Signal(row['market_id'], t, 'buy_both', 1.0, f"early_underround={underround:.4f}"))
            last_signal_t = t
    return signals


STRATEGY_REGISTRY = {
    'H6_underround_harvest': strategy_underround_harvest,
    'H8_late_directional': strategy_late_directional,
    'H9_early_inventory': strategy_early_inventory,
}


# =============================================================================
# EXECUTION AND BACKTEST
# =============================================================================

def execute_signals(market_df: pd.DataFrame, signals: List[Signal], market_Y: int = None) -> Tuple[float, int, int]:
    """
    Execute signals and return (total_pnl, wins, losses).
    
    Uses market_Y for directional trades to correctly compute losses.
    """
    total_pnl = 0.0
    wins = 0
    losses = 0
    
    for signal in signals:
        entry_row = market_df[market_df['t'] == signal.t]
        if entry_row.empty:
            continue
        entry_row = entry_row.iloc[0]
        
        if signal.side == 'buy_both':
            up_ask = entry_row.get('pm_up_best_ask')
            down_ask = entry_row.get('pm_down_best_ask')
            if pd.isna(up_ask) or pd.isna(down_ask):
                continue
            entry_price = up_ask + down_ask
            exit_price = 1.0
            pnl = exit_price - entry_price
            
        elif signal.side in ['buy_up', 'buy_down']:
            # Use market_Y (the final outcome) not per-row Y
            Y = market_Y if market_Y is not None else entry_row.get('Y', 0)
            
            if signal.side == 'buy_up':
                entry_price = entry_row.get('pm_up_best_ask')
                exit_price = 1.0 if Y == 1 else 0.0
            else:
                entry_price = entry_row.get('pm_down_best_ask')
                exit_price = 1.0 if Y == 0 else 0.0
            
            if pd.isna(entry_price):
                continue
            pnl = exit_price - entry_price
        else:
            continue
        
        total_pnl += pnl * signal.size
        if pnl > 0:
            wins += 1
        else:
            losses += 1
    
    return total_pnl, wins, losses


def backtest_single_strategy(
    market_data: pd.DataFrame,
    market_info: Dict,
    strategy_fn: Callable,
    params: Dict
) -> Tuple[float, int, Dict[str, float], int, int]:
    """
    Run backtest and return (total_pnl, n_trades, market_pnls, total_wins, total_losses).
    """
    market_pnls = {}
    total_trades = 0
    total_wins = 0
    total_losses = 0
    
    for market_id, market_df in market_data.groupby('market_id'):
        market_df = market_df.sort_values('t')
        
        # Get Y for this market
        if market_id in market_info:
            market_Y = market_info[market_id].get('Y', None)
        else:
            Y_vals = market_df['Y'].dropna().unique()
            market_Y = int(Y_vals[0]) if len(Y_vals) > 0 else None
        
        signals = strategy_fn(market_df, params)
        pnl, wins, losses = execute_signals(market_df, signals, market_Y)
        
        market_pnls[market_id] = pnl
        total_trades += len(signals)
        total_wins += wins
        total_losses += losses
    
    total_pnl = sum(market_pnls.values())
    return total_pnl, total_trades, market_pnls, total_wins, total_losses


# =============================================================================
# STRATEGY-CLASS AWARE PLACEBO TESTS
# =============================================================================

def run_placebo_time_shift_aware(
    market_data: pd.DataFrame,
    market_info: Dict,
    strategy_fn: Callable,
    params: Dict,
    category: str,
    shift_seconds: int = 30
) -> Dict[str, Any]:
    """
    Strategy-class aware time shift placebo.
    
    - PM_ONLY: CL shift should NOT affect edge (test PASSES if edge persists)
    - CL-dependent: CL shift SHOULD degrade edge (test PASSES if edge degrades)
    """
    shifted_data = market_data.copy()
    
    # Shift CL columns forward (making them stale)
    cl_cols = [c for c in shifted_data.columns if 'cl_' in c.lower() or 'delta' in c.lower()]
    
    for col in cl_cols:
        shifted_data[col] = shifted_data.groupby('market_id')[col].shift(shift_seconds)
    
    # Run backtests
    original_pnl, original_trades, original_market_pnls, _, _ = backtest_single_strategy(
        market_data, market_info, strategy_fn, params
    )
    
    shifted_pnl, shifted_trades, shifted_market_pnls, _, _ = backtest_single_strategy(
        shifted_data, market_info, strategy_fn, params
    )
    
    # Compute t-stats
    original_arr = np.array(list(original_market_pnls.values()))
    shifted_arr = np.array(list(shifted_market_pnls.values()))
    
    n = len(original_arr)
    if n > 1 and original_arr.std() > 0:
        original_t = original_arr.mean() / (original_arr.std() / np.sqrt(n))
    else:
        original_t = 0.0
    
    if n > 1 and shifted_arr.std() > 0:
        shifted_t = shifted_arr.mean() / (shifted_arr.std() / np.sqrt(n))
    else:
        shifted_t = 0.0
    
    # Strategy-class aware pass logic
    if category in ['PM_ONLY', 'INVENTORY']:
        # PM-only: edge should PERSIST (no CL dependency)
        # Pass if shifted_t is close to original_t
        t_ratio = shifted_t / original_t if original_t != 0 else 1.0
        passed = t_ratio > 0.5  # Edge persists within 50%
        expected_behavior = "Edge should persist (no CL dependency)"
    else:
        # CL-dependent: edge should DEGRADE
        # Pass if shifted_t is much lower than original_t
        passed = shifted_t < original_t * 0.5
        expected_behavior = "Edge should degrade under CL staleness"
    
    return {
        'shift_seconds': shift_seconds,
        'category': category,
        'expected_behavior': expected_behavior,
        'original_pnl': float(original_pnl),
        'shifted_pnl': float(shifted_pnl),
        'original_t_stat': float(original_t),
        'shifted_t_stat': float(shifted_t),
        'pnl_change_pct': float((shifted_pnl - original_pnl) / abs(original_pnl) * 100) if original_pnl != 0 else 0,
        'passed': passed,
    }


def run_placebo_outcome_shuffle(
    market_data: pd.DataFrame,
    market_info: Dict,
    strategy_fn: Callable,
    params: Dict,
    category: str,
    n_shuffles: int = 50
) -> Dict[str, Any]:
    """
    Outcome shuffle placebo - only meaningful for directional strategies.
    
    Randomly flips Y (outcome) for each market to test if direction prediction matters.
    """
    # Skip for PM_ONLY strategies (outcome doesn't affect complete-set arb)
    if category == 'PM_ONLY':
        return {
            'skipped': True,
            'reason': 'PM_ONLY strategies use complete-set arb, outcome does not affect PnL',
            'passed': 'N/A'
        }
    
    original_pnl, _, _, _, _ = backtest_single_strategy(
        market_data, market_info, strategy_fn, params
    )
    
    shuffled_pnls = []
    
    for _ in range(n_shuffles):
        # Create modified market_info with flipped Y
        shuffled_info = {}
        for mid, info in market_info.items():
            shuffled_info[mid] = info.copy()
            if np.random.random() > 0.5:
                shuffled_info[mid]['Y'] = 1 - info.get('Y', 0)
        
        shuf_pnl, _, _, _, _ = backtest_single_strategy(
            market_data, shuffled_info, strategy_fn, params
        )
        shuffled_pnls.append(shuf_pnl)
    
    shuffled_pnls = np.array(shuffled_pnls)
    pct_beaten = (shuffled_pnls >= original_pnl).mean()
    
    return {
        'n_shuffles': n_shuffles,
        'category': category,
        'original_pnl': float(original_pnl),
        'shuffled_mean': float(shuffled_pnls.mean()),
        'shuffled_std': float(shuffled_pnls.std()),
        'pct_shuffled_beats_original': float(pct_beaten),
        'passed': pct_beaten < 0.1,  # Original should beat 90%+ of shuffles
    }


def run_walk_forward_validation(
    market_data: pd.DataFrame,
    market_info: Dict,
    strategy_fn: Callable,
    params: Dict,
    train_pct: float = 0.7
) -> Dict[str, Any]:
    """Walk-forward validation with corrected Y handling."""
    # Sort markets by start time
    market_times = []
    for mid, info in market_info.items():
        market_start = info.get('market_start', '')
        market_times.append((mid, market_start))
    
    market_times.sort(key=lambda x: x[1])
    ordered_markets = [m[0] for m in market_times]
    
    # Split
    n_train = int(len(ordered_markets) * train_pct)
    train_markets = set(ordered_markets[:n_train])
    test_markets = set(ordered_markets[n_train:])
    
    # Filter market_info
    train_info = {k: v for k, v in market_info.items() if k in train_markets}
    test_info = {k: v for k, v in market_info.items() if k in test_markets}
    
    # Backtest
    train_data = market_data[market_data['market_id'].isin(train_markets)]
    test_data = market_data[market_data['market_id'].isin(test_markets)]
    
    train_pnl, _, train_market_pnls, train_wins, train_losses = backtest_single_strategy(
        train_data, train_info, strategy_fn, params
    )
    test_pnl, _, test_market_pnls, test_wins, test_losses = backtest_single_strategy(
        test_data, test_info, strategy_fn, params
    )
    
    # Compute t-stats
    train_arr = np.array(list(train_market_pnls.values()))
    test_arr = np.array(list(test_market_pnls.values()))
    
    n_train_markets = len(train_arr)
    n_test_markets = len(test_arr)
    
    if n_train_markets > 1 and train_arr.std() > 0:
        train_t = train_arr.mean() / (train_arr.std() / np.sqrt(n_train_markets))
    else:
        train_t = 0.0
    
    if n_test_markets > 1 and test_arr.std() > 0:
        test_t = test_arr.mean() / (test_arr.std() / np.sqrt(n_test_markets))
    else:
        test_t = 0.0
    
    total_wins = train_wins + test_wins
    total_losses = train_losses + test_losses
    total_trades = total_wins + total_losses
    win_rate = total_wins / total_trades if total_trades > 0 else 0.0
    
    return {
        'train_markets': n_train_markets,
        'test_markets': n_test_markets,
        'train_pnl': float(train_pnl),
        'test_pnl': float(test_pnl),
        'train_t_stat': float(train_t),
        'test_t_stat': float(test_t),
        'train_avg_pnl_per_market': float(train_arr.mean()) if len(train_arr) > 0 else 0,
        'test_avg_pnl_per_market': float(test_arr.mean()) if len(test_arr) > 0 else 0,
        'degradation_pct': float((train_t - test_t) / train_t * 100) if train_t > 0 else 0,
        'win_rate': float(win_rate),
        'total_wins': total_wins,
        'total_losses': total_losses,
        'passed': test_t > 0,
    }


def run_bootstrap_ci(
    market_data: pd.DataFrame,
    market_info: Dict,
    strategy_fn: Callable,
    params: Dict,
    n_bootstrap: int = 500
) -> Dict[str, Any]:
    """Bootstrap confidence intervals."""
    _, _, market_pnls, _, _ = backtest_single_strategy(
        market_data, market_info, strategy_fn, params
    )
    
    pnl_values = list(market_pnls.values())
    n_markets = len(pnl_values)
    
    if n_markets == 0:
        return {'error': 'no_trades'}
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_markets, size=n_markets, replace=True)
        sample_pnls = [pnl_values[i] for i in indices]
        bootstrap_means.append(np.mean(sample_pnls))
    
    bootstrap_means = np.array(bootstrap_means)
    lower = np.percentile(bootstrap_means, 2.5)
    upper = np.percentile(bootstrap_means, 97.5)
    prob_positive = (bootstrap_means > 0).mean()
    
    return {
        'n_bootstrap': n_bootstrap,
        'n_markets': n_markets,
        'original_mean': float(np.mean(pnl_values)),
        'bootstrap_mean': float(bootstrap_means.mean()),
        'bootstrap_std': float(bootstrap_means.std()),
        'ci_95_lower': float(lower),
        'ci_95_upper': float(upper),
        'prob_positive': float(prob_positive),
        'ci_excludes_zero': lower > 0,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Phase C: Strategy-Class Aware Validation")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    market_data, market_info = load_market_data()
    hypotheses = load_hypotheses()
    backtest_results = load_backtest_results()
    best_results = backtest_results.get('best_results', {})
    
    print(f"  Loaded {len(market_data['market_id'].unique())} markets")
    print(f"  Loaded {len(hypotheses)} hypotheses")
    
    # Run validation for each strategy
    print("\n" + "=" * 70)
    print("Running strategy-class aware validation...")
    print("=" * 70)
    
    validation_results = {}
    
    for hyp_id, best_result in best_results.items():
        if hyp_id not in STRATEGY_REGISTRY:
            print(f"\nSkipping {hyp_id} - no implementation")
            continue
        
        category = get_strategy_category(hypotheses, hyp_id)
        print(f"\n--- Validating: {hyp_id} (Category: {category}) ---")
        
        strategy_fn = STRATEGY_REGISTRY[hyp_id]
        params = best_result['params']
        
        results = {
            'hypothesis_id': hyp_id,
            'category': category,
            'params': params,
            'original_t_stat': best_result['t_stat'],
            'original_pnl': best_result['total_pnl'],
        }
        
        # Time shift placebo (strategy-class aware)
        print("  Running time shift placebo (strategy-class aware)...")
        for shift in [30, 60]:
            shift_result = run_placebo_time_shift_aware(
                market_data, market_info, strategy_fn, params, category, shift
            )
            results[f'placebo_time_shift_{shift}s'] = shift_result
            status = "PASS" if shift_result['passed'] else "FAIL"
            print(f"    Shift {shift}s: {status} ({shift_result['expected_behavior']})")
        
        # Outcome shuffle placebo (for directional only)
        print("  Running outcome shuffle placebo...")
        outcome_result = run_placebo_outcome_shuffle(
            market_data, market_info, strategy_fn, params, category
        )
        results['placebo_outcome_shuffle'] = outcome_result
        if outcome_result.get('skipped'):
            print(f"    Skipped: {outcome_result['reason']}")
        else:
            status = "PASS" if outcome_result['passed'] else "FAIL"
            print(f"    Outcome shuffle: {status}")
        
        # Walk-forward validation
        print("  Running walk-forward validation...")
        wf_result = run_walk_forward_validation(
            market_data, market_info, strategy_fn, params
        )
        results['walk_forward'] = wf_result
        status = "PASS" if wf_result['passed'] else "FAIL"
        print(f"    Walk-forward: {status} (train t={wf_result['train_t_stat']:.2f}, test t={wf_result['test_t_stat']:.2f})")
        print(f"    Win rate: {wf_result['win_rate']*100:.1f}% ({wf_result['total_wins']}W/{wf_result['total_losses']}L)")
        
        # Bootstrap CI
        print("  Running bootstrap CI...")
        bootstrap_result = run_bootstrap_ci(market_data, market_info, strategy_fn, params)
        results['bootstrap_ci'] = bootstrap_result
        print(f"    95% CI: [{bootstrap_result['ci_95_lower']:.4f}, {bootstrap_result['ci_95_upper']:.4f}]")
        print(f"    P(positive): {bootstrap_result['prob_positive']*100:.1f}%")
        
        validation_results[hyp_id] = results
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY (Strategy-Class Aware)")
    print("=" * 70)
    
    print("\n| Strategy | Category | Time Shift | Outcome | Walk-Fwd | Win Rate | P(pos) |")
    print("|----------|----------|------------|---------|----------|----------|--------|")
    
    for hyp_id, results in validation_results.items():
        ts30 = results.get('placebo_time_shift_30s', {})
        ts_status = "PASS" if ts30.get('passed') else "FAIL"
        
        outcome = results.get('placebo_outcome_shuffle', {})
        if outcome.get('skipped'):
            out_status = "N/A"
        else:
            out_status = "PASS" if outcome.get('passed') else "FAIL"
        
        wf = results.get('walk_forward', {})
        wf_status = "PASS" if wf.get('passed') else "FAIL"
        win_rate = wf.get('win_rate', 0) * 100
        
        prob_pos = results.get('bootstrap_ci', {}).get('prob_positive', 0) * 100
        
        print(f"| {hyp_id[:20]} | {results['category'][:8]} | {ts_status} | {out_status} | {wf_status} | {win_rate:.0f}% | {prob_pos:.0f}% |")
    
    # Save outputs
    print("\n" + "=" * 70)
    print("Saving outputs...")
    
    val_path = RESULTS_DIR / "validation_results_fixed.json"
    with open(val_path, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    print(f"  Validation results saved to: {val_path}")
    
    # Generate report
    report = ["# Validation Report (Strategy-Class Aware)\n\n"]
    report.append("## Summary\n\n")
    report.append("| Strategy | Category | Time Shift | Outcome Shuffle | Walk-Forward | Win Rate | P(positive) |\n")
    report.append("|----------|----------|------------|-----------------|--------------|----------|-------------|\n")
    
    for hyp_id, results in validation_results.items():
        ts30 = results.get('placebo_time_shift_30s', {})
        ts_status = "PASS" if ts30.get('passed') else "FAIL"
        
        outcome = results.get('placebo_outcome_shuffle', {})
        out_status = "N/A" if outcome.get('skipped') else ("PASS" if outcome.get('passed') else "FAIL")
        
        wf = results.get('walk_forward', {})
        wf_status = "PASS" if wf.get('passed') else "FAIL"
        win_rate = f"{wf.get('win_rate', 0)*100:.0f}%"
        
        prob_pos = f"{results.get('bootstrap_ci', {}).get('prob_positive', 0)*100:.0f}%"
        
        report.append(f"| {hyp_id} | {results['category']} | {ts_status} | {out_status} | {wf_status} | {win_rate} | {prob_pos} |\n")
    
    report.append("\n## Key Changes from Original Validation\n\n")
    report.append("1. **Time Shift Placebo**: Now strategy-class aware\n")
    report.append("   - PM_ONLY strategies: PASS if edge persists (no CL dependency expected)\n")
    report.append("   - CL-dependent: PASS if edge degrades (CL dependency confirmed)\n\n")
    report.append("2. **Outcome Shuffle**: Replaces timing permutation\n")
    report.append("   - Only applied to directional strategies\n")
    report.append("   - Tests if direction prediction matters\n")
    report.append("   - Skipped for complete-set arb (outcome doesn't affect PnL)\n\n")
    report.append("3. **Win Rate Tracking**: Now correctly tracks wins/losses\n")
    report.append("   - Uses market-level Y for directional trades\n")
    report.append("   - Exposes strategies with suspicious 100% win rates\n")
    
    report_path = REPORTS_DIR / "VALIDATION_REPORT_FIXED.md"
    with open(report_path, 'w') as f:
        f.write(''.join(report))
    print(f"  Report saved to: {report_path}")
    
    print("\n" + "=" * 70)
    print("DONE - Phase C Complete")
    print("=" * 70)
    
    return validation_results


if __name__ == "__main__":
    main()

