#!/usr/bin/env python3
"""
Phase A: Consistency Audit

Verifies that:
1. Every metric in FINAL_RESEARCH_REPORT.md is traceable to JSON outputs
2. PASS/FAIL labels are strategy-class aware
3. Report and JSON are internally consistent

Output:
- audit_results.json (discrepancies found)
- AUDIT_REPORT.md (summary of issues)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Configuration
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"


def load_all_results() -> Dict[str, Any]:
    """Load all JSON result files."""
    results = {}
    
    # Load validation results
    val_path = RESULTS_DIR / "validation_results.json"
    if val_path.exists():
        with open(val_path, 'r') as f:
            results['validation'] = json.load(f)
    
    # Load backtest results
    bt_path = RESULTS_DIR / "backtest_results.json"
    if bt_path.exists():
        with open(bt_path, 'r') as f:
            results['backtest'] = json.load(f)
    
    # Load hypotheses
    hyp_path = RESULTS_DIR / "hypotheses.json"
    if hyp_path.exists():
        with open(hyp_path, 'r') as f:
            results['hypotheses'] = json.load(f)
    
    return results


def get_strategy_category(hypotheses: List[Dict], hyp_id: str) -> str:
    """Get the category for a hypothesis."""
    for hyp in hypotheses:
        if hyp['hypothesis_id'] == hyp_id:
            return hyp.get('category', 'UNKNOWN')
    return 'UNKNOWN'


def audit_placebo_logic(results: Dict) -> List[Dict]:
    """
    Audit placebo test logic for strategy-class correctness.
    
    Rules:
    - PM_ONLY/INVENTORY strategies: CL time-shift should NOT affect edge
      -> Test should PASS if edge persists (shifted_t_stat close to original)
    - TIMING/CL_PM_LEADLAG strategies: CL time-shift SHOULD degrade edge  
      -> Test should PASS if edge degrades (shifted_t_stat much lower)
    """
    issues = []
    hypotheses = results.get('hypotheses', [])
    validation = results.get('validation', {})
    
    for hyp_id, val_result in validation.items():
        category = get_strategy_category(hypotheses, hyp_id)
        
        # Check time-shift placebo logic
        for shift_key in ['placebo_time_shift_30s', 'placebo_time_shift_60s']:
            if shift_key not in val_result:
                continue
            
            placebo = val_result[shift_key]
            original_t = placebo.get('original_t_stat', 0)
            shifted_t = placebo.get('shifted_t_stat', 0)
            current_passed = str(placebo.get('passed', 'False'))
            
            # Determine correct logic based on category
            if category in ['PM_ONLY', 'INVENTORY']:
                # PM-only: edge should PERSIST under CL shift
                # So "passed" should be True if shifted_t is close to original_t
                correct_passed = abs(shifted_t - original_t) / max(abs(original_t), 0.001) < 0.5
                expected_behavior = "Edge should persist (no CL dependency)"
            else:
                # CL-dependent: edge should DEGRADE under CL shift
                # So "passed" should be True if shifted_t is much lower than original_t
                correct_passed = shifted_t < original_t * 0.5
                expected_behavior = "Edge should degrade (CL dependency)"
            
            if str(current_passed).lower() != str(correct_passed).lower():
                issues.append({
                    'type': 'PLACEBO_LOGIC_ERROR',
                    'hypothesis_id': hyp_id,
                    'category': category,
                    'test': shift_key,
                    'current_passed': current_passed,
                    'correct_passed': correct_passed,
                    'original_t_stat': original_t,
                    'shifted_t_stat': shifted_t,
                    'expected_behavior': expected_behavior,
                    'recommendation': f"PM-only strategies don't use CL data, so CL time-shift should have no effect. "
                                    f"The current logic expects edge to DEGRADE, but for {category} it should PERSIST."
                })
    
    return issues


def audit_win_rate_sanity(results: Dict) -> List[Dict]:
    """
    Audit win rates for sanity.
    
    Rules:
    - Complete-set arb (buy_both): 100% win rate is expected
    - Directional trades: 100% win rate is suspicious (likely bug)
    """
    issues = []
    hypotheses = results.get('hypotheses', [])
    backtest = results.get('backtest', {})
    best_results = backtest.get('best_results', {})
    
    for hyp_id, bt_result in best_results.items():
        win_rate = bt_result.get('win_rate', 0)
        category = get_strategy_category(hypotheses, hyp_id)
        
        # H8_late_directional is a directional strategy
        if hyp_id == 'H8_late_directional' and win_rate >= 0.99:
            issues.append({
                'type': 'WIN_RATE_SUSPICION',
                'hypothesis_id': hyp_id,
                'category': category,
                'win_rate': win_rate,
                'concern': "Directional strategy shows 100% win rate. "
                          "This is suspicious - directional trades should have losses when direction is wrong.",
                'recommendation': "Audit PnL calculation in execute_signal(). Check if Y (outcome) is being used correctly. "
                                 "Verify losses are not being clipped with max(0, pnl)."
            })
        
        # Complete-set strategies with <100% win rate would also be suspicious
        if category == 'PM_ONLY' and 'underround' in hyp_id.lower():
            if win_rate < 0.95:
                issues.append({
                    'type': 'WIN_RATE_SUSPICION',
                    'hypothesis_id': hyp_id,
                    'category': category,
                    'win_rate': win_rate,
                    'concern': "Complete-set arbitrage should have ~100% win rate when underround exists.",
                    'recommendation': "Check if strategy is correctly buying both sides."
                })
    
    return issues


def audit_permutation_test(results: Dict) -> List[Dict]:
    """
    Audit permutation test logic.
    
    Issues:
    - Permuting 't' breaks market structure for underround strategies
    - For complete-set arb, timing doesn't matter if underround exists
    """
    issues = []
    hypotheses = results.get('hypotheses', [])
    validation = results.get('validation', {})
    
    for hyp_id, val_result in validation.items():
        if 'placebo_permutation' not in val_result:
            continue
        
        category = get_strategy_category(hypotheses, hyp_id)
        perm = val_result['placebo_permutation']
        pct_beaten = perm.get('pct_permuted_beats_original', 0)
        
        # For underround strategies, permuting 't' breaks the strategy
        # because underround conditions at random times won't match
        if 'underround' in hyp_id.lower() or category in ['PM_ONLY', 'INVENTORY']:
            if pct_beaten > 0.5:
                issues.append({
                    'type': 'PERMUTATION_TEST_INAPPROPRIATE',
                    'hypothesis_id': hyp_id,
                    'category': category,
                    'pct_beaten': pct_beaten,
                    'concern': "Permuting 't' breaks market structure for underround strategies. "
                              "When timing is randomized, the strategy finds underround at 'wrong' times, "
                              "creating MORE signals (but meaningless ones).",
                    'recommendation': "For complete-set arb, permutation test is not appropriate. "
                                    "Consider: 1) Skip this test for PM_ONLY, or 2) Permute Y (outcome) instead."
                })
    
    return issues


def audit_report_json_consistency(results: Dict) -> List[Dict]:
    """
    Audit that report metrics match JSON exactly.
    """
    issues = []
    
    report_path = REPORTS_DIR / "FINAL_RESEARCH_REPORT.md"
    if not report_path.exists():
        return issues
    
    with open(report_path, 'r') as f:
        report_content = f.read()
    
    validation = results.get('validation', {})
    
    # Check validation summary table claims
    # The report says "PASS" for time shift tests, but JSON says "False"
    for hyp_id, val_result in validation.items():
        # Check each placebo test
        for test_key in ['placebo_time_shift_30s', 'placebo_time_shift_60s', 'placebo_permutation', 'walk_forward']:
            if test_key not in val_result:
                continue
            
            json_passed = val_result[test_key].get('passed')
            
            # The report shows a different status
            # Look for the hypothesis in the validation summary
            if hyp_id in report_content:
                # Check if report claims PASS but JSON says False
                if str(json_passed).lower() == 'false':
                    # See if report mentions this strategy with "PASS" 
                    lines = report_content.split('\n')
                    for line in lines:
                        if hyp_id in line and 'PASS' in line:
                            if 'Time Shift' in line or 'Permutation' in line:
                                issues.append({
                                    'type': 'REPORT_JSON_MISMATCH',
                                    'hypothesis_id': hyp_id,
                                    'test': test_key,
                                    'json_passed': str(json_passed),
                                    'report_claims': 'PASS',
                                    'concern': "Report shows PASS but JSON shows False",
                                    'recommendation': "Report generator should read directly from JSON and apply strategy-class aware logic"
                                })
                            break
    
    return issues


def generate_audit_report(all_issues: List[Dict]) -> str:
    """Generate markdown audit report."""
    report = ["# Consistency Audit Report\n\n"]
    
    # Summary
    report.append("## Summary\n\n")
    issue_types = {}
    for issue in all_issues:
        t = issue['type']
        issue_types[t] = issue_types.get(t, 0) + 1
    
    if not all_issues:
        report.append("**No issues found.**\n\n")
    else:
        report.append(f"**Total issues found: {len(all_issues)}**\n\n")
        report.append("| Issue Type | Count |\n")
        report.append("|------------|-------|\n")
        for t, count in issue_types.items():
            report.append(f"| {t} | {count} |\n")
        report.append("\n")
    
    # Details by type
    if all_issues:
        report.append("## Issues by Type\n\n")
        
        for issue_type in issue_types.keys():
            report.append(f"### {issue_type}\n\n")
            
            type_issues = [i for i in all_issues if i['type'] == issue_type]
            for issue in type_issues:
                report.append(f"**{issue.get('hypothesis_id', 'Unknown')}**\n\n")
                report.append(f"- Category: {issue.get('category', 'N/A')}\n")
                report.append(f"- Concern: {issue.get('concern', issue.get('recommendation', 'N/A'))}\n")
                
                if 'current_passed' in issue:
                    report.append(f"- Current PASS/FAIL: {issue['current_passed']}\n")
                    report.append(f"- Correct PASS/FAIL: {issue['correct_passed']}\n")
                
                if 'recommendation' in issue and 'concern' in issue:
                    report.append(f"- Recommendation: {issue['recommendation']}\n")
                
                report.append("\n")
    
    # Recommendations
    report.append("## Key Recommendations\n\n")
    report.append("1. **Fix placebo logic** to be strategy-class aware:\n")
    report.append("   - PM_ONLY strategies: CL time-shift should NOT affect edge (test passes if edge persists)\n")
    report.append("   - CL-dependent strategies: CL time-shift SHOULD degrade edge (test passes if edge degrades)\n\n")
    report.append("2. **Audit PnL calculation** for directional strategies:\n")
    report.append("   - Verify losses are correctly computed when direction is wrong\n")
    report.append("   - Check that Y (outcome) is properly used in exit price calculation\n\n")
    report.append("3. **Fix permutation test** or skip for PM_ONLY strategies:\n")
    report.append("   - Permuting 't' is not meaningful for underround strategies\n")
    report.append("   - Consider permuting Y (outcome) for directional strategies instead\n\n")
    report.append("4. **Ensure report reads directly from JSON**:\n")
    report.append("   - No manual interpretation of PASS/FAIL\n")
    report.append("   - Strategy-class determines which tests are relevant\n")
    
    return ''.join(report)


def main():
    print("=" * 70)
    print("Phase A: Consistency Audit")
    print("=" * 70)
    
    # Load all results
    print("\nLoading results...")
    results = load_all_results()
    print(f"  Loaded: validation={len(results.get('validation', {}))}, "
          f"backtest={len(results.get('backtest', {}).get('best_results', {}))}, "
          f"hypotheses={len(results.get('hypotheses', []))}")
    
    # Run audits
    print("\nRunning audits...")
    all_issues = []
    
    print("  Auditing placebo logic...")
    placebo_issues = audit_placebo_logic(results)
    all_issues.extend(placebo_issues)
    print(f"    Found {len(placebo_issues)} issues")
    
    print("  Auditing win rates...")
    win_rate_issues = audit_win_rate_sanity(results)
    all_issues.extend(win_rate_issues)
    print(f"    Found {len(win_rate_issues)} issues")
    
    print("  Auditing permutation tests...")
    perm_issues = audit_permutation_test(results)
    all_issues.extend(perm_issues)
    print(f"    Found {len(perm_issues)} issues")
    
    print("  Auditing report-JSON consistency...")
    consistency_issues = audit_report_json_consistency(results)
    all_issues.extend(consistency_issues)
    print(f"    Found {len(consistency_issues)} issues")
    
    # Summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    print(f"\nTotal issues found: {len(all_issues)}")
    
    issue_types = {}
    for issue in all_issues:
        t = issue['type']
        issue_types[t] = issue_types.get(t, 0) + 1
    
    for t, count in issue_types.items():
        print(f"  {t}: {count}")
    
    # Save results
    print("\n" + "=" * 70)
    print("Saving outputs...")
    
    # Save audit results JSON
    audit_path = RESULTS_DIR / "audit_results.json"
    with open(audit_path, 'w') as f:
        json.dump({
            'total_issues': len(all_issues),
            'issues_by_type': issue_types,
            'issues': all_issues
        }, f, indent=2)
    print(f"  Audit results saved to: {audit_path}")
    
    # Save audit report
    report = generate_audit_report(all_issues)
    report_path = REPORTS_DIR / "AUDIT_REPORT.md"
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Audit report saved to: {report_path}")
    
    print("\n" + "=" * 70)
    print("DONE - Phase A Complete")
    print("=" * 70)
    
    return all_issues


if __name__ == "__main__":
    main()

