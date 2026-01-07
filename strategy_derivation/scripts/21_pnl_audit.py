#!/usr/bin/env python3
"""
Phase B: PnL Accounting Audit

Verifies PnL calculations are correct:
1. Complete-set arb: entry = sum_asks, exit = 1.0, pnl = 1.0 - sum_asks (always positive)
2. Directional: entry = ask, exit = 1.0 if correct else 0.0, pnl can be NEGATIVE

This script traces through the PnL logic and reports any issues.

Output:
- pnl_audit_results.json
- PNL_AUDIT_REPORT.md
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict

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
    
    # Handle list format
    if isinstance(market_info, list):
        market_info_dict = {}
        for item in market_info:
            mid = item.get('market_id', item.get('condition_id', ''))
            market_info_dict[mid] = item
        market_info = market_info_dict
    
    return df, market_info


@dataclass
class Trade:
    """Trade with full audit trail."""
    market_id: str
    entry_t: int
    side: str
    entry_price: float
    exit_price: float
    Y: int  # Actual outcome
    pnl: float
    expected_pnl: float  # What PnL should be
    pnl_correct: bool


def audit_complete_set_pnl(
    market_df: pd.DataFrame,
    market_id: str
) -> List[Trade]:
    """
    Audit complete-set (buy_both) PnL calculation.
    
    Expected:
    - Entry price = up_ask + down_ask
    - Exit price = 1.0 (guaranteed)
    - PnL = 1.0 - entry_price (always positive if underround exists)
    """
    trades = []
    
    # Get a few sample rows with underround
    market_df = market_df.sort_values('t')
    
    for _, row in market_df.head(5).iterrows():
        up_ask = row.get('pm_up_best_ask')
        down_ask = row.get('pm_down_best_ask')
        
        if pd.isna(up_ask) or pd.isna(down_ask):
            continue
        
        entry_price = up_ask + down_ask
        exit_price = 1.0
        
        # Check if underround exists
        if entry_price >= 1.0:
            continue
        
        # Compute PnL
        computed_pnl = exit_price - entry_price
        expected_pnl = 1.0 - entry_price
        
        # This should always be positive for underround
        pnl_correct = computed_pnl > 0 and abs(computed_pnl - expected_pnl) < 0.0001
        
        trades.append(Trade(
            market_id=market_id,
            entry_t=int(row['t']),
            side='buy_both',
            entry_price=entry_price,
            exit_price=exit_price,
            Y=int(row.get('Y', -1)),
            pnl=computed_pnl,
            expected_pnl=expected_pnl,
            pnl_correct=pnl_correct
        ))
    
    return trades


def audit_directional_pnl(
    market_df: pd.DataFrame,
    market_id: str,
    market_Y: int
) -> List[Trade]:
    """
    Audit directional PnL calculation.
    
    Expected:
    - buy_up: entry = up_ask, exit = 1.0 if Y=1 else 0.0
    - buy_down: entry = down_ask, exit = 1.0 if Y=0 else 0.0
    - PnL = exit - entry (CAN BE NEGATIVE)
    """
    trades = []
    market_df = market_df.sort_values('t')
    
    # Sample some rows
    for _, row in market_df.head(5).iterrows():
        t = int(row['t'])
        
        # Test buy_up
        up_ask = row.get('pm_up_best_ask')
        if not pd.isna(up_ask):
            entry_price = up_ask
            exit_price = 1.0 if market_Y == 1 else 0.0
            computed_pnl = exit_price - entry_price
            
            # Expected PnL based on Y
            if market_Y == 1:
                expected_pnl = 1.0 - entry_price  # Win
            else:
                expected_pnl = 0.0 - entry_price  # Loss (NEGATIVE!)
            
            pnl_correct = abs(computed_pnl - expected_pnl) < 0.0001
            
            trades.append(Trade(
                market_id=market_id,
                entry_t=t,
                side='buy_up',
                entry_price=entry_price,
                exit_price=exit_price,
                Y=market_Y,
                pnl=computed_pnl,
                expected_pnl=expected_pnl,
                pnl_correct=pnl_correct
            ))
        
        # Test buy_down
        down_ask = row.get('pm_down_best_ask')
        if not pd.isna(down_ask):
            entry_price = down_ask
            exit_price = 1.0 if market_Y == 0 else 0.0
            computed_pnl = exit_price - entry_price
            
            # Expected PnL based on Y
            if market_Y == 0:
                expected_pnl = 1.0 - entry_price  # Win
            else:
                expected_pnl = 0.0 - entry_price  # Loss (NEGATIVE!)
            
            pnl_correct = abs(computed_pnl - expected_pnl) < 0.0001
            
            trades.append(Trade(
                market_id=market_id,
                entry_t=t,
                side='buy_down',
                entry_price=entry_price,
                exit_price=exit_price,
                Y=market_Y,
                pnl=computed_pnl,
                expected_pnl=expected_pnl,
                pnl_correct=pnl_correct
            ))
    
    return trades


def check_current_implementation() -> Dict[str, Any]:
    """
    Check the current implementation in 17_implement_strategies.py
    and 18_validation_suite.py for PnL bugs.
    """
    issues = []
    
    # Read the implementation file
    impl_path = BASE_DIR / "scripts" / "17_implement_strategies.py"
    if impl_path.exists():
        with open(impl_path, 'r') as f:
            impl_code = f.read()
        
        # Check for suspicious patterns
        
        # 1. Check if Y is read correctly
        if "entry_row.get('Y'" in impl_code:
            issues.append({
                'file': '17_implement_strategies.py',
                'issue': 'Y is read from entry_row, but Y should be the market outcome (constant for whole market)',
                'line_pattern': "entry_row.get('Y'",
                'fix': 'Y should come from market_info or be consistent across the market, not per-row'
            })
        
        # 2. Check for pnl clipping
        if "max(0, pnl)" in impl_code or "max(0," in impl_code:
            issues.append({
                'file': '17_implement_strategies.py',
                'issue': 'PnL might be clipped to non-negative values',
                'line_pattern': 'max(0,',
                'fix': 'Remove max(0, pnl) - directional losses must be negative'
            })
        
        # 3. Check if exit_price for directional uses Y
        if "exit_price = float(Y)" in impl_code:
            # This is correct IF Y is the actual outcome
            pass
    
    # Read validation suite
    val_path = BASE_DIR / "scripts" / "18_validation_suite.py"
    if val_path.exists():
        with open(val_path, 'r') as f:
            val_code = f.read()
        
        # Same checks
        if "entry_row.get('Y'" in val_code:
            issues.append({
                'file': '18_validation_suite.py',
                'issue': 'Y is read from entry_row in validation suite',
                'line_pattern': "entry_row.get('Y'",
                'fix': 'Y should be consistent market outcome'
            })
    
    return {
        'implementation_issues': issues
    }


def simulate_directional_strategy(
    market_data: pd.DataFrame,
    market_info: Dict
) -> Dict[str, Any]:
    """
    Simulate the late_directional strategy and check win rate.
    """
    results = {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'total_pnl': 0.0,
        'trades': []
    }
    
    for market_id, market_df in market_data.groupby('market_id'):
        market_df = market_df.sort_values('t')
        
        # Get Y for this market
        # Y should be in market_info or we can get it from the data
        if market_id in market_info:
            Y = market_info[market_id].get('Y', None)
        else:
            # Try to get from data - Y should be same across all rows
            Y_values = market_df['Y'].unique()
            Y = int(Y_values[0]) if len(Y_values) > 0 and not pd.isna(Y_values[0]) else None
        
        if Y is None:
            continue
        
        # Simulate late directional strategy
        # Look at late rows (tau < 300)
        late_rows = market_df[market_df['tau'] < 300]
        if len(late_rows) == 0:
            continue
        
        # Take first signal based on delta_bps
        for _, row in late_rows.head(1).iterrows():
            delta_bps = row.get('delta_bps')
            if pd.isna(delta_bps):
                continue
            
            if abs(delta_bps) <= 10:  # Below threshold
                continue
            
            # Determine side
            if delta_bps > 0:
                side = 'buy_up'
                entry_price = row.get('pm_up_best_ask')
                exit_price = 1.0 if Y == 1 else 0.0
            else:
                side = 'buy_down'
                entry_price = row.get('pm_down_best_ask')
                exit_price = 1.0 if Y == 0 else 0.0
            
            if pd.isna(entry_price):
                continue
            
            pnl = exit_price - entry_price
            
            results['total_trades'] += 1
            results['total_pnl'] += pnl
            
            if pnl > 0:
                results['winning_trades'] += 1
            else:
                results['losing_trades'] += 1
            
            results['trades'].append({
                'market_id': market_id,
                'Y': Y,
                'delta_bps': float(delta_bps),
                'side': side,
                'entry_price': float(entry_price),
                'exit_price': float(exit_price),
                'pnl': float(pnl),
                'correct_direction': (delta_bps > 0 and Y == 1) or (delta_bps < 0 and Y == 0)
            })
    
    if results['total_trades'] > 0:
        results['win_rate'] = results['winning_trades'] / results['total_trades']
    else:
        results['win_rate'] = 0.0
    
    return results


def generate_pnl_audit_report(
    complete_set_trades: List[Trade],
    directional_trades: List[Trade],
    impl_issues: Dict,
    directional_simulation: Dict
) -> str:
    """Generate PnL audit report."""
    report = ["# PnL Accounting Audit Report\n\n"]
    
    # Summary
    report.append("## Summary\n\n")
    
    # Complete-set audit
    report.append("### Complete-Set (buy_both) Trades\n\n")
    n_correct = sum(1 for t in complete_set_trades if t.pnl_correct)
    n_total = len(complete_set_trades)
    report.append(f"- Sampled trades: {n_total}\n")
    report.append(f"- PnL calculation correct: {n_correct}/{n_total}\n")
    
    if n_total > 0:
        avg_pnl = sum(t.pnl for t in complete_set_trades) / n_total
        report.append(f"- Average PnL (underround): ${avg_pnl:.4f}\n")
        report.append(f"- All positive (expected for underround): {'Yes' if all(t.pnl > 0 for t in complete_set_trades) else 'No'}\n")
    report.append("\n")
    
    # Directional audit
    report.append("### Directional Trades (Audit)\n\n")
    n_correct = sum(1 for t in directional_trades if t.pnl_correct)
    n_total = len(directional_trades)
    report.append(f"- Sampled trades: {n_total}\n")
    report.append(f"- PnL calculation correct: {n_correct}/{n_total}\n")
    
    if n_total > 0:
        n_negative = sum(1 for t in directional_trades if t.pnl < 0)
        report.append(f"- Trades with negative PnL: {n_negative}/{n_total}\n")
    report.append("\n")
    
    # Directional simulation
    report.append("### Late Directional Strategy Simulation\n\n")
    sim = directional_simulation
    report.append(f"- Total trades: {sim['total_trades']}\n")
    report.append(f"- Winning trades: {sim['winning_trades']}\n")
    report.append(f"- Losing trades: {sim['losing_trades']}\n")
    report.append(f"- Win rate: {sim['win_rate']*100:.1f}%\n")
    report.append(f"- Total PnL: ${sim['total_pnl']:.2f}\n\n")
    
    if sim['win_rate'] > 0.99:
        report.append("**WARNING**: 100% win rate on directional strategy is suspicious!\n\n")
    elif sim['losing_trades'] > 0:
        report.append("**GOOD**: Strategy correctly shows losing trades.\n\n")
    
    # Implementation issues
    report.append("## Implementation Issues Found\n\n")
    issues = impl_issues.get('implementation_issues', [])
    
    if not issues:
        report.append("No code-level issues found.\n\n")
    else:
        for issue in issues:
            report.append(f"### {issue['file']}\n\n")
            report.append(f"- **Issue**: {issue['issue']}\n")
            report.append(f"- **Pattern**: `{issue['line_pattern']}`\n")
            report.append(f"- **Fix**: {issue['fix']}\n\n")
    
    # Sample trades
    report.append("## Sample Directional Trades\n\n")
    report.append("| Market | Y | Delta | Side | Entry | Exit | PnL | Correct? |\n")
    report.append("|--------|---|-------|------|-------|------|-----|----------|\n")
    
    for trade in sim.get('trades', [])[:10]:
        correct = 'Yes' if trade['correct_direction'] else 'No'
        report.append(f"| {trade['market_id'][:20]} | {trade['Y']} | {trade['delta_bps']:.1f} | "
                     f"{trade['side']} | {trade['entry_price']:.3f} | {trade['exit_price']:.1f} | "
                     f"${trade['pnl']:.3f} | {correct} |\n")
    
    # Recommendations
    report.append("\n## Recommendations\n\n")
    
    if sim['win_rate'] > 0.99:
        report.append("1. **Investigate 100% win rate**: Directional strategies should have losing trades.\n")
        report.append("   - Check if Y (outcome) is being read correctly\n")
        report.append("   - Verify Y is the FINAL outcome, not a per-row signal\n\n")
    
    report.append("2. **Verify Y consistency**: Y should be the same for all rows in a market.\n")
    report.append("3. **Check for PnL clipping**: Ensure `max(0, pnl)` is NOT used.\n")
    report.append("4. **Validate exit logic**: For hold-to-expiry, exit = 1.0 or 0.0 based on Y.\n")
    
    return ''.join(report)


def main():
    print("=" * 70)
    print("Phase B: PnL Accounting Audit")
    print("=" * 70)
    
    # Load data
    print("\nLoading market data...")
    market_data, market_info = load_market_data()
    print(f"  Loaded {len(market_data):,} rows, {len(market_data['market_id'].unique())} markets")
    
    # Check Y consistency
    print("\nChecking Y consistency across markets...")
    y_check = market_data.groupby('market_id')['Y'].nunique()
    inconsistent = y_check[y_check > 1]
    if len(inconsistent) > 0:
        print(f"  WARNING: {len(inconsistent)} markets have inconsistent Y values")
    else:
        print("  Y is consistent within each market")
    
    # Audit complete-set trades
    print("\nAuditing complete-set (buy_both) PnL...")
    complete_set_trades = []
    sample_markets = list(market_data['market_id'].unique())[:5]
    for mid in sample_markets:
        market_df = market_data[market_data['market_id'] == mid]
        trades = audit_complete_set_pnl(market_df, mid)
        complete_set_trades.extend(trades)
    print(f"  Audited {len(complete_set_trades)} sample trades")
    
    # Audit directional trades
    print("\nAuditing directional PnL...")
    directional_trades = []
    for mid in sample_markets:
        market_df = market_data[market_data['market_id'] == mid]
        Y = int(market_df['Y'].iloc[0]) if not market_df['Y'].isna().all() else 0
        trades = audit_directional_pnl(market_df, mid, Y)
        directional_trades.extend(trades)
    print(f"  Audited {len(directional_trades)} sample trades")
    
    # Check for negative PnL in directional
    n_negative = sum(1 for t in directional_trades if t.pnl < 0)
    print(f"  Trades with negative PnL (losses): {n_negative}/{len(directional_trades)}")
    
    # Check implementation code
    print("\nChecking implementation code for issues...")
    impl_issues = check_current_implementation()
    print(f"  Found {len(impl_issues.get('implementation_issues', []))} potential issues")
    
    # Simulate directional strategy
    print("\nSimulating late_directional strategy...")
    directional_sim = simulate_directional_strategy(market_data, market_info)
    print(f"  Total trades: {directional_sim['total_trades']}")
    print(f"  Winning: {directional_sim['winning_trades']}, Losing: {directional_sim['losing_trades']}")
    print(f"  Win rate: {directional_sim['win_rate']*100:.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("PNL AUDIT SUMMARY")
    print("=" * 70)
    
    if directional_sim['win_rate'] > 0.99 and directional_sim['total_trades'] > 5:
        print("\nWARNING: 100% win rate on directional strategy!")
        print("  This is suspicious - directional trades should have losses.")
        print("  Check Y (outcome) handling in the implementation.")
    elif directional_sim['losing_trades'] > 0:
        print("\nGOOD: Directional strategy correctly shows losing trades.")
    
    # Save results
    print("\n" + "=" * 70)
    print("Saving outputs...")
    
    # Save JSON
    results = {
        'complete_set_trades': [asdict(t) for t in complete_set_trades],
        'directional_trades': [asdict(t) for t in directional_trades],
        'implementation_issues': impl_issues,
        'directional_simulation': {
            k: v for k, v in directional_sim.items() 
            if k != 'trades'  # Don't save all trades
        }
    }
    results['directional_simulation']['sample_trades'] = directional_sim.get('trades', [])[:20]
    
    results_path = RESULTS_DIR / "pnl_audit_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to: {results_path}")
    
    # Generate report
    report = generate_pnl_audit_report(
        complete_set_trades, 
        directional_trades, 
        impl_issues, 
        directional_sim
    )
    report_path = REPORTS_DIR / "PNL_AUDIT_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Report saved to: {report_path}")
    
    print("\n" + "=" * 70)
    print("DONE - Phase B Complete")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()

