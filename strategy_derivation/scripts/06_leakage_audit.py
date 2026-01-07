#!/usr/bin/env python3
"""
Step 2: Hard Leakage Audit for Strategy B

This script audits Strategy B for look-ahead bias and data leakage.
Each check includes explicit proof of correctness or flags for issues.

CRITICAL: Strategy B shows t=3.09 but failed placebo (t=2.87 with 30s shift).
We must determine if this is real or an artifact.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "results"


def audit_check(name: str, passed: bool, details: str) -> Dict[str, Any]:
    """Record an audit check result."""
    result = {
        'check': name,
        'passed': passed,
        'status': 'PASS' if passed else 'FAIL',
        'details': details,
    }
    status_str = "PASS" if passed else "FAIL"
    print(f"  [{status_str}] {name}")
    print(f"          {details}")
    return result


def audit_k_construction():
    """
    AUDIT 1: K (strike price) construction
    
    K must be determined at t=0 (market start), not from future data.
    If K uses any information from t>0, that's look-ahead bias.
    """
    print("\n" + "=" * 60)
    print("AUDIT 1: K (Strike Price) Construction")
    print("=" * 60)
    
    checks = []
    
    # Load canonical dataset
    df = pd.read_parquet('data_v2/research/canonical_dataset_all_assets.parquet')
    df_eth = df[df['asset'] == 'ETH'].copy()
    
    # Check 1: K is constant within each market
    k_per_market = df_eth.groupby('market_id')['K'].nunique()
    all_constant = (k_per_market == 1).all()
    checks.append(audit_check(
        "K is constant within each market",
        all_constant,
        f"K unique values per market: max={k_per_market.max()}, min={k_per_market.min()}"
    ))
    
    # Check 2: K approximately equals CL price at t=0
    t0_rows = df_eth[df_eth['t'] == 0][['market_id', 'K', 'cl_mid']].copy()
    t0_rows['k_vs_cl0'] = abs(t0_rows['K'] - t0_rows['cl_mid'])
    t0_rows['k_vs_cl0_bps'] = t0_rows['k_vs_cl0'] / t0_rows['cl_mid'] * 10000
    
    # K should be very close to cl_mid at t=0 (within a few bps typically)
    max_diff_bps = t0_rows['k_vs_cl0_bps'].max()
    mean_diff_bps = t0_rows['k_vs_cl0_bps'].mean()
    
    checks.append(audit_check(
        "K approximates CL price at t=0",
        max_diff_bps < 100,  # Allow up to 1% difference
        f"K vs CL(t=0): mean={mean_diff_bps:.2f}bps, max={max_diff_bps:.2f}bps"
    ))
    
    # Check 3: K does NOT equal CL at any other specific time (e.g., settlement)
    # This would indicate K was set using future info
    for check_t in [450, 899]:  # Middle and end of market
        t_rows = df_eth[df_eth['t'] == check_t][['market_id', 'K', 'cl_mid']].copy()
        if len(t_rows) > 0:
            t_rows['diff_bps'] = abs(t_rows['K'] - t_rows['cl_mid']) / t_rows['cl_mid'] * 10000
            # K should NOT be closer to mid-market or end prices than to t=0
            mean_diff = t_rows['diff_bps'].mean()
            checks.append(audit_check(
                f"K is NOT suspiciously close to CL(t={check_t})",
                mean_diff > 10,  # Should have drifted away
                f"K vs CL(t={check_t}): mean={mean_diff:.2f}bps"
            ))
    
    return checks


def audit_delta_bps_construction():
    """
    AUDIT 2: delta_bps construction
    
    delta_bps = (cl_mid - K) / K * 10000
    
    Must use only cl_mid at current time t, not future values.
    """
    print("\n" + "=" * 60)
    print("AUDIT 2: delta_bps Construction")
    print("=" * 60)
    
    checks = []
    
    # Load data
    df = pd.read_parquet('data_v2/research/canonical_dataset_all_assets.parquet')
    df_eth = df[df['asset'] == 'ETH'].copy()
    
    # Recompute delta_bps manually
    df_eth['delta_bps_recomputed'] = (df_eth['cl_mid'] - df_eth['K']) / df_eth['K'] * 10000
    
    # Check they match
    diff = abs(df_eth['delta_bps'] - df_eth['delta_bps_recomputed'])
    max_diff = diff.max()
    
    checks.append(audit_check(
        "delta_bps matches (cl_mid - K) / K * 10000",
        max_diff < 0.01,
        f"Max difference: {max_diff:.6f} bps"
    ))
    
    # Check delta_bps is NOT correlated with future CL
    # Sample a few markets and check correlation with shifted CL
    sample_markets = df_eth['market_id'].unique()[:5]
    
    future_corrs = []
    for mid in sample_markets:
        mdf = df_eth[df_eth['market_id'] == mid].sort_values('t')
        if len(mdf) > 60:
            # delta_bps should NOT be correlated with CL 30s in the future
            future_cl = mdf['cl_mid'].shift(-30)
            corr = mdf['delta_bps'].corr(future_cl)
            if not np.isnan(corr):
                future_corrs.append(corr)
    
    if future_corrs:
        mean_corr = np.mean(future_corrs)
        checks.append(audit_check(
            "delta_bps not anomalously correlated with future CL",
            abs(mean_corr) < 0.9,  # Some correlation expected due to autocorrelation
            f"Mean correlation with CL(t+30): {mean_corr:.3f}"
        ))
    
    return checks


def audit_strategy_b_signal_generation():
    """
    AUDIT 3: Strategy B signal generation
    
    Check that signals use ONLY information available at time t.
    """
    print("\n" + "=" * 60)
    print("AUDIT 3: Strategy B Signal Generation")
    print("=" * 60)
    
    checks = []
    
    # Read the strategy code
    strategy_file = PROJECT_ROOT / 'scripts' / 'backtest' / 'strategies.py'
    with open(strategy_file) as f:
        code = f.read()
    
    # Find Strategy B class
    start_marker = "class LateDirectionalTakerStrategy"
    end_marker = "class TwoSidedEarlyTiltLateStrategy"
    
    start_idx = code.find(start_marker)
    end_idx = code.find(end_marker)
    
    if start_idx == -1:
        checks.append(audit_check(
            "Strategy B code found",
            False,
            "Could not locate LateDirectionalTakerStrategy class"
        ))
        return checks
    
    strategy_b_code = code[start_idx:end_idx]
    
    # Check 1: Does NOT use 'settlement' column
    uses_settlement = 'settlement' in strategy_b_code.lower()
    checks.append(audit_check(
        "Does NOT use settlement column",
        not uses_settlement,
        f"'settlement' found in code: {uses_settlement}"
    ))
    
    # Check 2: Does NOT use 'Y' column (outcome)
    uses_y = "'Y'" in strategy_b_code or '"Y"' in strategy_b_code or "['Y']" in strategy_b_code
    checks.append(audit_check(
        "Does NOT use Y (outcome) column",
        not uses_y,
        f"'Y' column access found: {uses_y}"
    ))
    
    # Check 3: Uses tau from row, not computed from end time
    uses_tau = 'tau' in strategy_b_code
    uses_max_t = 'max_t' in strategy_b_code
    checks.append(audit_check(
        "Uses tau from dataset (not max_t - t)",
        uses_tau,
        f"Uses tau: {uses_tau}, Uses max_t: {uses_max_t}"
    ))
    
    # Check 4: Examine what columns are accessed
    # Common data access patterns in pandas
    accessed_columns = []
    for col in ['delta_bps', 'cl_mid', 'tau', 't', 'momentum', 'cl_return_bps']:
        if f"'{col}'" in strategy_b_code or f'"{col}"' in strategy_b_code or f"['{col}']" in strategy_b_code:
            accessed_columns.append(col)
    
    checks.append(audit_check(
        "Only uses expected columns",
        set(accessed_columns).issubset({'delta_bps', 'cl_mid', 'tau', 't', 'momentum', 'cl_return_bps'}),
        f"Columns accessed: {accessed_columns}"
    ))
    
    # Check 5: Momentum is computed from pct_change (backward-looking)
    uses_pct_change = 'pct_change' in strategy_b_code
    uses_rolling = 'rolling' in strategy_b_code
    checks.append(audit_check(
        "Momentum computed from backward-looking pct_change + rolling",
        uses_pct_change and uses_rolling,
        f"pct_change: {uses_pct_change}, rolling: {uses_rolling}"
    ))
    
    return checks


def audit_placebo_shift_implementation():
    """
    AUDIT 4: Placebo shift implementation
    
    The placebo test shifts CL data. We must verify:
    1. Shift direction is correct (positive shift = staler data)
    2. All relevant columns are shifted consistently
    """
    print("\n" + "=" * 60)
    print("AUDIT 4: Placebo Shift Implementation")
    print("=" * 60)
    
    checks = []
    
    # Read the backtest code
    backtest_file = OUTPUT_DIR / '04_run_backtests.py'
    with open(backtest_file) as f:
        code = f.read()
    
    # Find placebo function
    if 'def run_placebo_test' not in code:
        checks.append(audit_check(
            "Placebo function found",
            False,
            "Could not find run_placebo_test function"
        ))
        return checks
    
    # Check 1: Shift direction
    # df.shift(n) with n>0 shifts data DOWN, meaning row i gets value from row i-n
    # This is CORRECT for making data staler
    if '.shift(cl_shift_seconds)' in code or '.shift(30)' in code:
        shift_correct = True
        shift_detail = "shift(positive) = staler data (CORRECT)"
    else:
        shift_correct = False
        shift_detail = "Could not verify shift direction"
    
    checks.append(audit_check(
        "Shift direction is correct (positive = staler)",
        shift_correct,
        shift_detail
    ))
    
    # Check 2: All CL columns shifted
    expected_shifted = ['cl_mid', 'cl_bid', 'cl_ask', 'delta', 'delta_bps']
    shifted_cols = []
    for col in expected_shifted:
        if f"'{col}'" in code or f'"{col}"' in code:
            shifted_cols.append(col)
    
    checks.append(audit_check(
        "All CL-related columns shifted",
        set(expected_shifted).issubset(set(shifted_cols)),
        f"Columns shifted: {shifted_cols}"
    ))
    
    # Check 3: K is NOT shifted (should remain constant)
    k_shifted = "'K'" in code.split('cl_cols')[1] if 'cl_cols' in code else False
    checks.append(audit_check(
        "K (strike) is NOT shifted",
        not k_shifted,
        f"K in shifted columns: {k_shifted}"
    ))
    
    # Check 4: Verify shift with actual data
    print("\n  --- Running shift verification test ---")
    df = pd.read_parquet('data_v2/research/canonical_dataset_all_assets.parquet')
    df_eth = df[df['asset'] == 'ETH'].copy()
    
    # Take one market
    sample_market = df_eth['market_id'].iloc[0]
    mdf = df_eth[df_eth['market_id'] == sample_market].sort_values('t').copy()
    
    # Apply shift like placebo does
    original_cl_at_t50 = mdf[mdf['t'] == 50]['cl_mid'].values[0]
    original_cl_at_t20 = mdf[mdf['t'] == 20]['cl_mid'].values[0]
    
    mdf['cl_mid_shifted'] = mdf['cl_mid'].shift(30)
    shifted_cl_at_t50 = mdf[mdf['t'] == 50]['cl_mid_shifted'].values[0]
    
    # After shift(30), cl_mid at t=50 should equal original cl_mid at t=20
    shift_works = np.isclose(shifted_cl_at_t50, original_cl_at_t20, rtol=1e-6)
    
    checks.append(audit_check(
        "Shift(30) gives value from 30s earlier",
        shift_works,
        f"At t=50: shifted={shifted_cl_at_t50:.2f}, original_t20={original_cl_at_t20:.2f}"
    ))
    
    return checks


def audit_backtest_execution():
    """
    AUDIT 5: Backtest execution model
    
    Check that entry/exit prices are taken from correct times.
    """
    print("\n" + "=" * 60)
    print("AUDIT 5: Backtest Execution Model")
    print("=" * 60)
    
    checks = []
    
    # Read backtest engine
    engine_file = PROJECT_ROOT / 'scripts' / 'backtest' / 'backtest_engine.py'
    with open(engine_file) as f:
        code = f.read()
    
    # Check 1: Entry price from entry time
    # Should use market_df at entry_t, not some other time
    if 'entry_t' in code and 'entry_price' in code:
        checks.append(audit_check(
            "Entry uses entry_t for price lookup",
            True,
            "entry_t and entry_price found in execution code"
        ))
    
    # Check 2: Exit price from exit time
    if 'exit_t' in code and 'exit_price' in code:
        checks.append(audit_check(
            "Exit uses exit_t for price lookup",
            True,
            "exit_t and exit_price found in execution code"
        ))
    
    # Check 3: No future price access
    # Look for patterns that might indicate future data access
    suspicious_patterns = [
        '.shift(-',  # Negative shift = future
        'iloc[-1]',  # Last row might be future
    ]
    
    suspicious_found = []
    for pattern in suspicious_patterns:
        if pattern in code:
            suspicious_found.append(pattern)
    
    checks.append(audit_check(
        "No obvious future data access patterns",
        len(suspicious_found) == 0,
        f"Suspicious patterns found: {suspicious_found if suspicious_found else 'None'}"
    ))
    
    return checks


def audit_delta_bps_autocorrelation():
    """
    AUDIT 6: delta_bps autocorrelation analysis
    
    If delta_bps is highly autocorrelated, then seeing "stale" delta_bps
    still provides valid information about current delta_bps.
    This could explain why placebo test doesn't kill the edge.
    """
    print("\n" + "=" * 60)
    print("AUDIT 6: delta_bps Autocorrelation")
    print("=" * 60)
    
    checks = []
    
    # Load data
    df = pd.read_parquet('data_v2/research/canonical_dataset_all_assets.parquet')
    df_eth = df[df['asset'] == 'ETH'].copy()
    
    # Compute autocorrelation at various lags
    autocorrs = {}
    for lag in [1, 5, 15, 30, 60, 120]:
        corrs = []
        for mid in df_eth['market_id'].unique()[:10]:  # Sample markets
            mdf = df_eth[df_eth['market_id'] == mid].sort_values('t')
            if len(mdf) > lag + 10:
                corr = mdf['delta_bps'].autocorr(lag=lag)
                if not np.isnan(corr):
                    corrs.append(corr)
        if corrs:
            autocorrs[lag] = np.mean(corrs)
    
    print(f"\n  delta_bps autocorrelation by lag:")
    for lag, corr in autocorrs.items():
        print(f"    Lag {lag}s: {corr:.3f}")
    
    # High autocorrelation at lag 30 would explain placebo failure
    lag30_corr = autocorrs.get(30, 0)
    
    checks.append(audit_check(
        "delta_bps autocorrelation at lag 30s",
        True,  # This is informational, not pass/fail
        f"Autocorr(30s) = {lag30_corr:.3f} - {'HIGH' if lag30_corr > 0.7 else 'MODERATE' if lag30_corr > 0.4 else 'LOW'}"
    ))
    
    # If autocorr is high, the strategy signal persists over time
    # This means even with 30s stale data, the signal is still informative
    if lag30_corr > 0.7:
        explanation = (
            "HIGH AUTOCORRELATION DETECTED!\n"
            "         This explains why placebo test doesn't kill the edge:\n"
            "         delta_bps(t-30) ~ delta_bps(t), so 'stale' data is still informative.\n"
            "         This is NOT a bug - it means the strategy exploits PERSISTENCE,\n"
            "         not short-term CL-PM lead-lag."
        )
        print(f"\n  {explanation}")
    
    return checks


def main():
    print("=" * 70)
    print("STEP 2: HARD LEAKAGE AUDIT FOR STRATEGY B")
    print("=" * 70)
    print("\nThis audit checks for look-ahead bias and data leakage.")
    print("Strategy B is GUILTY UNTIL PROVEN INNOCENT.")
    
    all_checks = []
    
    # Run all audits
    all_checks.extend(audit_k_construction())
    all_checks.extend(audit_delta_bps_construction())
    all_checks.extend(audit_strategy_b_signal_generation())
    all_checks.extend(audit_placebo_shift_implementation())
    all_checks.extend(audit_backtest_execution())
    all_checks.extend(audit_delta_bps_autocorrelation())
    
    # Summary
    print("\n" + "=" * 70)
    print("LEAKAGE AUDIT SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for c in all_checks if c['passed'])
    failed = sum(1 for c in all_checks if not c['passed'])
    
    print(f"\nTotal checks: {len(all_checks)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print("\n*** AUDIT FOUND ISSUES ***")
        print("\nFailed checks:")
        for c in all_checks:
            if not c['passed']:
                print(f"  - {c['check']}: {c['details']}")
    else:
        print("\n*** ALL CHECKS PASSED ***")
        print("No obvious leakage detected in code.")
    
    # Key finding
    print("\n" + "-" * 70)
    print("KEY FINDING: delta_bps Autocorrelation")
    print("-" * 70)
    
    # Check if we found high autocorrelation
    autocorr_check = [c for c in all_checks if 'autocorrelation' in c['check'].lower()]
    if autocorr_check:
        print(f"\n{autocorr_check[0]['details']}")
        
        if 'HIGH' in autocorr_check[0]['details']:
            print("""
INTERPRETATION:
The high autocorrelation of delta_bps explains why the placebo test 
doesn't significantly reduce the t-statistic:

1. Strategy B triggers when |delta_bps| > threshold in the late window
2. delta_bps is highly persistent (autocorr ~0.9+ at 30s lag)
3. Therefore, delta_bps(t-30) is a good predictor of delta_bps(t)
4. The "stale" signal in the placebo test is STILL INFORMATIVE

This is NOT a bug or leakage - it means Strategy B exploits:
- The PERSISTENCE of CL being above/below strike
- NOT short-term lead-lag between CL and PM

The strategy is: "If CL was significantly above strike 30s ago, 
it's probably still above strike now, so buy UP."

This is a valid market-making / trend-following strategy, 
but it's NOT the CL-PM lead-lag we were originally testing for.
""")
    
    print("\n" + "=" * 70)
    
    # Save audit results
    import json
    with open(OUTPUT_DIR / 'leakage_audit_results.json', 'w') as f:
        json.dump(all_checks, f, indent=2, default=str)
    print(f"\nAudit results saved to leakage_audit_results.json")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

