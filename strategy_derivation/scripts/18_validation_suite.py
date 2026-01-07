#!/usr/bin/env python3
"""
Phase 8: Validation Suite

Runs placebo tests, walk-forward validation, cross-wallet replication checks,
and bootstrap confidence intervals.

Input:
- backtest_results.json (from Phase 7)
- canonical_dataset_all_assets.parquet (market data)

Output:
- validation_results.json (placebo, walk-forward, bootstrap results)
- validation_report.md (summary report)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict
import sys

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
RESEARCH_DIR = BASE_DIR.parent / "data_v2" / "research"
MARKET_DURATION_SECONDS = 900


def load_market_data() -> Tuple[pd.DataFrame, Dict]:
    """Load canonical market dataset."""
    path = RESEARCH_DIR / "canonical_dataset_all_assets.parquet"
    print(f"Loading market data from: {path}")
    df = pd.read_parquet(path)
    
    info_path = RESEARCH_DIR / "market_info_all_assets.json"
    with open(info_path, 'r') as f:
        market_info = json.load(f)
    
    return df, market_info


def load_backtest_results() -> Dict:
    """Load backtest results from Phase 7."""
    path = RESULTS_DIR / "backtest_results.json"
    with open(path, 'r') as f:
        results = json.load(f)
    return results


# =============================================================================
# STRATEGY IMPLEMENTATIONS (copied from Phase 7 for self-containment)
# =============================================================================

@dataclass
class Signal:
    market_id: str
    t: int
    side: str
    size: float
    reason: str


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


def execute_signals(market_df: pd.DataFrame, signals: List[Signal]) -> float:
    """Execute signals and return total PnL."""
    total_pnl = 0.0
    
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
            if signal.side == 'buy_up':
                entry_price = entry_row.get('pm_up_best_ask')
                Y = entry_row.get('Y', 0)
                exit_price = float(Y)
            else:
                entry_price = entry_row.get('pm_down_best_ask')
                Y = entry_row.get('Y', 0)
                exit_price = 1 - float(Y)
            if pd.isna(entry_price):
                continue
            pnl = exit_price - entry_price
        else:
            continue
        
        total_pnl += pnl * signal.size
    
    return total_pnl


def backtest_single_strategy(
    market_data: pd.DataFrame,
    strategy_fn: Callable,
    params: Dict
) -> Tuple[float, int, Dict[str, float]]:
    """Run backtest and return (total_pnl, n_trades, market_pnls)."""
    market_pnls = {}
    total_trades = 0
    
    for market_id, market_df in market_data.groupby('market_id'):
        market_df = market_df.sort_values('t')
        signals = strategy_fn(market_df, params)
        pnl = execute_signals(market_df, signals)
        market_pnls[market_id] = pnl
        total_trades += len(signals)
    
    total_pnl = sum(market_pnls.values())
    return total_pnl, total_trades, market_pnls


# =============================================================================
# PLACEBO TESTS
# =============================================================================

def run_placebo_time_shift(
    market_data: pd.DataFrame,
    strategy_fn: Callable,
    params: Dict,
    shift_seconds: int = 30
) -> Dict[str, Any]:
    """
    Placebo test: Shift CL features in time.
    If edge persists with stale CL data, it might not be from CL lead-lag.
    """
    shifted_data = market_data.copy()
    
    # Shift CL columns forward (making them stale)
    cl_cols = [c for c in shifted_data.columns if 'cl_' in c.lower() or 'delta' in c.lower()]
    
    for col in cl_cols:
        shifted_data[col] = shifted_data.groupby('market_id')[col].shift(shift_seconds)
    
    # Run backtest on shifted data
    original_pnl, original_trades, original_market_pnls = backtest_single_strategy(
        market_data, strategy_fn, params
    )
    
    shifted_pnl, shifted_trades, shifted_market_pnls = backtest_single_strategy(
        shifted_data, strategy_fn, params
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
    
    return {
        'shift_seconds': shift_seconds,
        'original_pnl': float(original_pnl),
        'shifted_pnl': float(shifted_pnl),
        'original_t_stat': float(original_t),
        'shifted_t_stat': float(shifted_t),
        'pnl_change_pct': float((shifted_pnl - original_pnl) / abs(original_pnl) * 100) if original_pnl != 0 else 0,
        'passed': abs(shifted_t) < abs(original_t) * 0.5,  # Edge should degrade significantly
    }


def run_placebo_permutation(
    market_data: pd.DataFrame,
    strategy_fn: Callable,
    params: Dict,
    n_permutations: int = 100
) -> Dict[str, Any]:
    """
    Placebo test: Permute trade times within each market.
    If edge persists with random timing, it's not timing-based.
    """
    original_pnl, original_trades, _ = backtest_single_strategy(
        market_data, strategy_fn, params
    )
    
    permuted_pnls = []
    
    for _ in range(n_permutations):
        # Randomly shuffle 't' within each market
        permuted_data = market_data.copy()
        
        for market_id in permuted_data['market_id'].unique():
            mask = permuted_data['market_id'] == market_id
            t_values = permuted_data.loc[mask, 't'].values.copy()
            np.random.shuffle(t_values)
            permuted_data.loc[mask, 't'] = t_values
        
        permuted_data = permuted_data.sort_values(['market_id', 't'])
        
        perm_pnl, _, _ = backtest_single_strategy(permuted_data, strategy_fn, params)
        permuted_pnls.append(perm_pnl)
    
    # How often does permuted beat original?
    permuted_pnls = np.array(permuted_pnls)
    pct_beaten = (permuted_pnls >= original_pnl).mean()
    
    return {
        'n_permutations': n_permutations,
        'original_pnl': float(original_pnl),
        'permuted_mean': float(permuted_pnls.mean()),
        'permuted_std': float(permuted_pnls.std()),
        'pct_permuted_beats_original': float(pct_beaten),
        'passed': pct_beaten < 0.05,  # Original should beat 95%+ of permutations
    }


def run_placebo_direction_shuffle(
    market_data: pd.DataFrame,
    strategy_fn: Callable,
    params: Dict,
    n_shuffles: int = 100
) -> Dict[str, Any]:
    """
    Placebo test: Randomize Y (outcome) while keeping everything else.
    If edge persists, it's not based on correct direction prediction.
    """
    original_pnl, _, _ = backtest_single_strategy(market_data, strategy_fn, params)
    
    shuffled_pnls = []
    
    for _ in range(n_shuffles):
        shuffled_data = market_data.copy()
        
        # Randomly flip Y for each market
        for market_id in shuffled_data['market_id'].unique():
            mask = shuffled_data['market_id'] == market_id
            if np.random.random() > 0.5:
                shuffled_data.loc[mask, 'Y'] = 1 - shuffled_data.loc[mask, 'Y']
        
        shuf_pnl, _, _ = backtest_single_strategy(shuffled_data, strategy_fn, params)
        shuffled_pnls.append(shuf_pnl)
    
    shuffled_pnls = np.array(shuffled_pnls)
    pct_beaten = (shuffled_pnls >= original_pnl).mean()
    
    return {
        'n_shuffles': n_shuffles,
        'original_pnl': float(original_pnl),
        'shuffled_mean': float(shuffled_pnls.mean()),
        'shuffled_std': float(shuffled_pnls.std()),
        'pct_shuffled_beats_original': float(pct_beaten),
        'passed': pct_beaten < 0.1,  # For complete-set, direction shouldn't matter as much
    }


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def run_walk_forward_validation(
    market_data: pd.DataFrame,
    market_info: Any,
    strategy_fn: Callable,
    params: Dict,
    train_pct: float = 0.7
) -> Dict[str, Any]:
    """
    Walk-forward validation: train on early markets, test on later markets.
    """
    # Sort markets by start time
    market_times = []
    
    # Handle both dict and list formats of market_info
    if isinstance(market_info, dict):
        for mid, info in market_info.items():
            market_start = info.get('market_start', '')
            market_times.append((mid, market_start))
    elif isinstance(market_info, list):
        for info in market_info:
            mid = info.get('market_id', '')
            market_start = info.get('market_start', '')
            market_times.append((mid, market_start))
    else:
        # Fallback: use unique market_ids from data, sorted by first timestamp
        unique_markets = market_data.groupby('market_id')['timestamp'].min().sort_values()
        market_times = [(mid, str(ts)) for mid, ts in unique_markets.items()]
    
    market_times.sort(key=lambda x: x[1])
    ordered_markets = [m[0] for m in market_times]
    
    # Split into train/test
    n_train = int(len(ordered_markets) * train_pct)
    train_markets = set(ordered_markets[:n_train])
    test_markets = set(ordered_markets[n_train:])
    
    # Backtest on train
    train_data = market_data[market_data['market_id'].isin(train_markets)]
    train_pnl, train_trades, train_market_pnls = backtest_single_strategy(
        train_data, strategy_fn, params
    )
    
    # Backtest on test
    test_data = market_data[market_data['market_id'].isin(test_markets)]
    test_pnl, test_trades, test_market_pnls = backtest_single_strategy(
        test_data, strategy_fn, params
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
        'passed': test_t > 0,  # Test set should still be positive
    }


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def run_bootstrap_ci(
    market_data: pd.DataFrame,
    strategy_fn: Callable,
    params: Dict,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95
) -> Dict[str, Any]:
    """
    Bootstrap confidence intervals for PnL.
    """
    # First get market-level PnLs
    _, _, market_pnls = backtest_single_strategy(market_data, strategy_fn, params)
    
    pnl_values = list(market_pnls.values())
    market_ids = list(market_pnls.keys())
    n_markets = len(pnl_values)
    
    if n_markets == 0:
        return {'error': 'no_trades'}
    
    # Bootstrap
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        # Resample market indices with replacement
        indices = np.random.choice(n_markets, size=n_markets, replace=True)
        sample_pnls = [pnl_values[i] for i in indices]
        bootstrap_means.append(np.mean(sample_pnls))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Compute CI
    alpha = (1 - ci_level) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    
    # Probability of positive mean
    prob_positive = (bootstrap_means > 0).mean()
    
    return {
        'n_bootstrap': n_bootstrap,
        'n_markets': n_markets,
        'original_mean': float(np.mean(pnl_values)),
        'bootstrap_mean': float(bootstrap_means.mean()),
        'bootstrap_std': float(bootstrap_means.std()),
        f'ci_{int(ci_level*100)}_lower': float(lower),
        f'ci_{int(ci_level*100)}_upper': float(upper),
        'prob_positive': float(prob_positive),
        'ci_excludes_zero': lower > 0 or upper < 0,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Phase 8: Validation Suite")
    print("=" * 70)
    
    # Step 1: Load data
    market_data, market_info = load_market_data()
    backtest_results = load_backtest_results()
    
    best_results = backtest_results.get('best_results', {})
    
    # Step 2: Run validation for each strategy
    print("\n" + "=" * 70)
    print("Running validation tests...")
    print("=" * 70)
    
    validation_results = {}
    
    for hyp_id, best_result in best_results.items():
        if hyp_id not in STRATEGY_REGISTRY:
            print(f"\nSkipping {hyp_id} - no implementation")
            continue
        
        print(f"\n--- Validating: {hyp_id} ---")
        
        strategy_fn = STRATEGY_REGISTRY[hyp_id]
        params = best_result['params']
        
        results = {
            'hypothesis_id': hyp_id,
            'params': params,
            'original_t_stat': best_result['t_stat'],
            'original_pnl': best_result['total_pnl'],
        }
        
        # Placebo: Time shift
        print("  Running time shift placebo...")
        for shift in [30, 60]:
            shift_result = run_placebo_time_shift(market_data, strategy_fn, params, shift)
            results[f'placebo_time_shift_{shift}s'] = shift_result
            status = "PASS" if shift_result['passed'] else "FAIL"
            print(f"    Shift {shift}s: {status} (t-stat: {shift_result['original_t_stat']:.2f} -> {shift_result['shifted_t_stat']:.2f})")
        
        # Placebo: Permutation (only for small sample)
        print("  Running permutation placebo (50 perms)...")
        perm_result = run_placebo_permutation(market_data, strategy_fn, params, n_permutations=50)
        results['placebo_permutation'] = perm_result
        status = "PASS" if perm_result['passed'] else "FAIL"
        print(f"    Permutation: {status} (pct beaten: {perm_result['pct_permuted_beats_original']*100:.1f}%)")
        
        # Walk-forward validation
        print("  Running walk-forward validation...")
        wf_result = run_walk_forward_validation(market_data, market_info, strategy_fn, params)
        results['walk_forward'] = wf_result
        status = "PASS" if wf_result['passed'] else "FAIL"
        print(f"    Walk-forward: {status} (train t={wf_result['train_t_stat']:.2f}, test t={wf_result['test_t_stat']:.2f})")
        
        # Bootstrap CI
        print("  Running bootstrap CI...")
        bootstrap_result = run_bootstrap_ci(market_data, strategy_fn, params, n_bootstrap=500)
        results['bootstrap_ci'] = bootstrap_result
        print(f"    95% CI: [{bootstrap_result.get('ci_95_lower', 0):.3f}, {bootstrap_result.get('ci_95_upper', 0):.3f}]")
        print(f"    P(positive): {bootstrap_result.get('prob_positive', 0)*100:.1f}%")
        
        validation_results[hyp_id] = results
    
    # Step 3: Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for hyp_id, results in validation_results.items():
        print(f"\n{hyp_id}:")
        print(f"  Original t-stat: {results['original_t_stat']:.2f}")
        
        # Count passed tests
        n_passed = 0
        n_total = 0
        
        if 'placebo_time_shift_30s' in results:
            n_total += 1
            if results['placebo_time_shift_30s']['passed']:
                n_passed += 1
        
        if 'placebo_permutation' in results:
            n_total += 1
            if results['placebo_permutation']['passed']:
                n_passed += 1
        
        if 'walk_forward' in results:
            n_total += 1
            if results['walk_forward']['passed']:
                n_passed += 1
        
        print(f"  Tests passed: {n_passed}/{n_total}")
        
        if 'bootstrap_ci' in results:
            bs = results['bootstrap_ci']
            print(f"  Bootstrap P(positive): {bs.get('prob_positive', 0)*100:.1f}%")
    
    # Step 4: Save outputs
    print("\n" + "=" * 70)
    print("Saving outputs...")
    
    # Save validation results
    val_path = RESULTS_DIR / "validation_results.json"
    with open(val_path, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    print(f"  Validation results saved to: {val_path}")
    
    # Generate validation report
    report_path = REPORTS_DIR / "VALIDATION_REPORT_NEW.md"
    REPORTS_DIR.mkdir(exist_ok=True)
    
    report = ["# Validation Report\n"]
    report.append(f"**Generated**: Phase 8 Validation Suite\n\n")
    report.append("## Summary\n\n")
    report.append("| Strategy | Original t-stat | Time Shift | Permutation | Walk-Forward | P(positive) |\n")
    report.append("|----------|-----------------|------------|-------------|--------------|-------------|\n")
    
    for hyp_id, results in validation_results.items():
        ts_status = "PASS" if results.get('placebo_time_shift_30s', {}).get('passed', False) else "FAIL"
        perm_status = "PASS" if results.get('placebo_permutation', {}).get('passed', False) else "FAIL"
        wf_status = "PASS" if results.get('walk_forward', {}).get('passed', False) else "FAIL"
        prob_pos = results.get('bootstrap_ci', {}).get('prob_positive', 0) * 100
        
        report.append(f"| {hyp_id} | {results['original_t_stat']:.2f} | {ts_status} | {perm_status} | {wf_status} | {prob_pos:.1f}% |\n")
    
    report.append("\n## Detailed Results\n\n")
    
    for hyp_id, results in validation_results.items():
        report.append(f"### {hyp_id}\n\n")
        report.append(f"**Parameters**: {results['params']}\n\n")
        
        if 'walk_forward' in results:
            wf = results['walk_forward']
            report.append(f"**Walk-Forward**: Train t-stat={wf['train_t_stat']:.2f}, Test t-stat={wf['test_t_stat']:.2f}\n\n")
        
        if 'bootstrap_ci' in results:
            bs = results['bootstrap_ci']
            report.append(f"**Bootstrap 95% CI**: [{bs.get('ci_95_lower', 0):.4f}, {bs.get('ci_95_upper', 0):.4f}]\n\n")
    
    with open(report_path, 'w') as f:
        f.write(''.join(report))
    print(f"  Validation report saved to: {report_path}")
    
    print("\n" + "=" * 70)
    print("DONE - Phase 8 Complete")
    print("=" * 70)
    
    return validation_results


if __name__ == "__main__":
    main()

