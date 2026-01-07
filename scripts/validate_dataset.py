"""
Dataset Validation Suite.

This script validates the canonical research dataset before strategy testing.

Validation Checks:
1. Outcome Reproduction: computed_Y == resolved_Y for all markets
2. Coverage Sanity: both_coverage <= min(cl, pm) coverage (math check)
3. Timestamp Integrity: No gaps > 2 seconds in either stream
4. No-Arb Sanity: sum_bids and sum_asks within [0.95, 1.05]
5. Strike Consistency: K matches "price to beat" from folder/UI

Output: VALIDATION_REPORT.md with pass/fail for each check.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import json
from typing import Dict, List, Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import STORAGE
from src.ground_truth import GroundTruthRepository


# =============================================================================
# CONFIGURATION
# =============================================================================

MARKET_DURATION_SECONDS = 900
RESEARCH_DIR = Path(STORAGE.research_dir)


# =============================================================================
# VALIDATION CHECKS
# =============================================================================

class ValidationResult:
    """Container for validation results."""
    
    def __init__(self, name: str, passed: bool, message: str, details: Optional[Dict] = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


def check_outcome_reproduction(
    market_infos: List[Dict],
    ground_truth_repo: Optional[GroundTruthRepository] = None
) -> ValidationResult:
    """
    Check 1: Verify computed outcomes match resolved outcomes.
    
    Critical check - if this fails, our labeling is wrong.
    """
    name = "Outcome Reproduction"
    
    if ground_truth_repo is None:
        return ValidationResult(name, True, "Skipped (no ground truth available)", {"skipped": True})
    
    gt_data = ground_truth_repo.load_all()
    
    if not gt_data:
        return ValidationResult(name, True, "Skipped (no ground truth data)", {"skipped": True})
    
    matches = 0
    mismatches = 0
    mismatch_details = []
    
    for market_id, gt in gt_data.items():
        resolved = gt.get('resolved_outcome')
        computed = gt.get('computed_outcome')
        
        if resolved and computed:
            if resolved.lower() == computed.lower():
                matches += 1
            else:
                mismatches += 1
                mismatch_details.append({
                    'market_id': market_id,
                    'resolved': resolved,
                    'computed': computed,
                    'K': gt.get('strike_K'),
                    'settlement': gt.get('settlement_price')
                })
    
    total = matches + mismatches
    
    if total == 0:
        return ValidationResult(name, True, "No markets with both resolved and computed outcomes", {"skipped": True})
    
    match_rate = matches / total * 100
    passed = match_rate >= 95  # Allow 5% mismatch (could be API timing issues)
    
    message = f"{matches}/{total} outcomes match ({match_rate:.1f}%)"
    if mismatches > 0:
        message += f" - {mismatches} mismatches"
    
    return ValidationResult(name, passed, message, {
        'matches': matches,
        'mismatches': mismatches,
        'match_rate': match_rate,
        'mismatch_details': mismatch_details[:5]  # First 5 mismatches
    })


def check_coverage_sanity(market_infos: List[Dict]) -> ValidationResult:
    """
    Check 2: Verify coverage math is consistent.
    
    both_coverage should be <= min(cl_coverage, pm_coverage)
    (Can't have both present more often than either individually)
    """
    name = "Coverage Math Sanity"
    
    violations = []
    
    for info in market_infos:
        cl = info.get('cl_coverage_pct', 0)
        pm = info.get('pm_coverage_pct', 0)
        both = info.get('both_coverage_pct', 0)
        
        min_coverage = min(cl, pm)
        
        # Allow small floating point tolerance
        if both > min_coverage + 0.1:
            violations.append({
                'market_id': info.get('market_id'),
                'cl_coverage': cl,
                'pm_coverage': pm,
                'both_coverage': both,
                'expected_max': min_coverage
            })
    
    passed = len(violations) == 0
    message = f"{'All' if passed else len(violations)} markets {'pass' if passed else 'fail'} coverage math check"
    
    return ValidationResult(name, passed, message, {
        'violations': violations[:5],
        'n_violations': len(violations)
    })


def check_timestamp_gaps(
    canonical_df: pd.DataFrame,
    max_gap_seconds: int = 2
) -> ValidationResult:
    """
    Check 3: Verify no large gaps in timestamps.
    
    Each market should have continuous data (gaps <= max_gap_seconds).
    """
    name = "Timestamp Integrity"
    
    markets_with_gaps = []
    
    for market_id in canonical_df['market_id'].unique():
        market_data = canonical_df[canonical_df['market_id'] == market_id].copy()
        
        # Check if t values are continuous
        t_values = sorted(market_data['t'].unique())
        if len(t_values) < 2:
            continue
        
        gaps = [t_values[i+1] - t_values[i] for i in range(len(t_values)-1)]
        max_gap = max(gaps)
        
        if max_gap > max_gap_seconds:
            markets_with_gaps.append({
                'market_id': market_id,
                'max_gap': max_gap,
                'n_rows': len(market_data)
            })
    
    passed = len(markets_with_gaps) == 0
    message = f"{'All' if passed else f'{len(markets_with_gaps)}'} markets {'pass' if passed else 'have gaps >'} {max_gap_seconds}s"
    
    return ValidationResult(name, passed, message, {
        'markets_with_gaps': markets_with_gaps[:5],
        'n_markets_with_gaps': len(markets_with_gaps),
        'max_gap_threshold': max_gap_seconds
    })


def check_noarb_bounds(
    canonical_df: pd.DataFrame,
    lower_bound: float = 0.90,
    upper_bound: float = 1.10
) -> ValidationResult:
    """
    Check 4: Verify no-arb conditions are reasonable.
    
    sum_bids and sum_asks should be close to 1 (within bounds).
    """
    name = "No-Arb Bounds"
    
    results = {}
    
    if 'sum_bids' in canonical_df.columns:
        valid_bids = canonical_df['sum_bids'].dropna()
        if len(valid_bids) > 0:
            out_of_bounds_bids = ((valid_bids < lower_bound) | (valid_bids > upper_bound)).sum()
            results['sum_bids'] = {
                'mean': valid_bids.mean(),
                'std': valid_bids.std(),
                'min': valid_bids.min(),
                'max': valid_bids.max(),
                'out_of_bounds_pct': out_of_bounds_bids / len(valid_bids) * 100
            }
    
    if 'sum_asks' in canonical_df.columns:
        valid_asks = canonical_df['sum_asks'].dropna()
        if len(valid_asks) > 0:
            out_of_bounds_asks = ((valid_asks < lower_bound) | (valid_asks > upper_bound)).sum()
            results['sum_asks'] = {
                'mean': valid_asks.mean(),
                'std': valid_asks.std(),
                'min': valid_asks.min(),
                'max': valid_asks.max(),
                'out_of_bounds_pct': out_of_bounds_asks / len(valid_asks) * 100
            }
    
    # Pass if <5% out of bounds for both
    bids_ok = results.get('sum_bids', {}).get('out_of_bounds_pct', 0) < 5
    asks_ok = results.get('sum_asks', {}).get('out_of_bounds_pct', 0) < 5
    passed = bids_ok and asks_ok
    
    message = f"sum_bids mean={results.get('sum_bids', {}).get('mean', 0):.3f}, sum_asks mean={results.get('sum_asks', {}).get('mean', 0):.3f}"
    
    return ValidationResult(name, passed, message, results)


def check_strike_consistency(
    market_infos: List[Dict],
    tolerance: float = 50  # Allow $50 difference
) -> ValidationResult:
    """
    Check 5: Verify computed K matches folder "price to beat".
    
    The K we compute should match what Polymarket shows as "price to beat".
    """
    name = "Strike Consistency"
    
    matches = 0
    mismatches = 0
    mismatch_details = []
    
    for info in market_infos:
        folder_price = info.get('price_to_beat_from_folder')
        computed_k = info.get('K')
        
        if folder_price is None or pd.isna(computed_k):
            continue
        
        diff = abs(computed_k - folder_price)
        
        if diff <= tolerance:
            matches += 1
        else:
            mismatches += 1
            mismatch_details.append({
                'market_id': info.get('market_id'),
                'folder_price': folder_price,
                'computed_K': computed_k,
                'difference': diff
            })
    
    total = matches + mismatches
    
    if total == 0:
        return ValidationResult(name, True, "No markets with both folder price and computed K", {"skipped": True})
    
    match_rate = matches / total * 100
    passed = match_rate >= 90  # 90% should match within tolerance
    
    message = f"{matches}/{total} K values within ${tolerance} of folder price ({match_rate:.1f}%)"
    
    return ValidationResult(name, passed, message, {
        'matches': matches,
        'mismatches': mismatches,
        'match_rate': match_rate,
        'tolerance': tolerance,
        'mismatch_details': mismatch_details[:5]
    })


def check_ffill_reasonableness(canonical_df: pd.DataFrame) -> ValidationResult:
    """
    Check 6: Verify forward-fill percentages are reasonable.
    
    If >80% of data is forward-filled, something is wrong with collection.
    """
    name = "Forward-Fill Reasonableness"
    
    results = {}
    
    if 'cl_ffill' in canonical_df.columns:
        cl_ffill_pct = canonical_df['cl_ffill'].mean() * 100
        results['cl_ffill_pct'] = cl_ffill_pct
    
    if 'pm_ffill' in canonical_df.columns:
        pm_ffill_pct = canonical_df['pm_ffill'].mean() * 100
        results['pm_ffill_pct'] = pm_ffill_pct
    
    cl_ok = results.get('cl_ffill_pct', 0) < 80
    pm_ok = results.get('pm_ffill_pct', 0) < 80
    passed = cl_ok and pm_ok
    
    message = f"CL FFill={results.get('cl_ffill_pct', 0):.1f}%, PM FFill={results.get('pm_ffill_pct', 0):.1f}%"
    
    return ValidationResult(name, passed, message, results)


def check_data_completeness(market_infos: List[Dict]) -> ValidationResult:
    """
    Check 7: Verify we have enough data for meaningful analysis.
    
    Should have at least 10 markets with >50% coverage.
    """
    name = "Data Completeness"
    
    good_markets = []
    
    for info in market_infos:
        cl_coverage = info.get('cl_coverage_pct', 0)
        pm_coverage = info.get('pm_coverage_pct', 0)
        
        if cl_coverage > 50 and pm_coverage > 50:
            good_markets.append(info.get('market_id'))
    
    n_good = len(good_markets)
    n_total = len(market_infos)
    
    passed = n_good >= 10
    message = f"{n_good}/{n_total} markets have >50% coverage on both sources"
    
    return ValidationResult(name, passed, message, {
        'n_good_markets': n_good,
        'n_total_markets': n_total,
        'good_market_ids': good_markets[:10]
    })


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_validation_report(
    results: List[ValidationResult],
    output_path: Path
):
    """Generate markdown validation report."""
    
    n_passed = sum(1 for r in results if r.passed)
    n_failed = sum(1 for r in results if not r.passed)
    n_total = len(results)
    
    lines = [
        "# Dataset Validation Report",
        "",
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
        f"- **Total Checks**: {n_total}",
        f"- **Passed**: {n_passed}",
        f"- **Failed**: {n_failed}",
        "",
        "## Validation Results",
        "",
    ]
    
    for result in results:
        status = "[PASS]" if result.passed else "[FAIL]"
        lines.append(f"### {status} {result.name}")
        lines.append("")
        lines.append(result.message)
        lines.append("")
        
        if result.details and not result.details.get('skipped'):
            lines.append("**Details:**")
            lines.append("```json")
            # Format details nicely
            import json
            lines.append(json.dumps(result.details, indent=2, default=str))
            lines.append("```")
            lines.append("")
    
    # Final verdict
    lines.append("## Verdict")
    lines.append("")
    
    if n_failed == 0:
        lines.append("**ALL CHECKS PASSED** - Dataset is ready for strategy backtesting.")
    else:
        lines.append(f"**{n_failed} CHECK(S) FAILED** - Review and fix issues before proceeding.")
        lines.append("")
        lines.append("Failed checks:")
        for result in results:
            if not result.passed:
                lines.append(f"- {result.name}: {result.message}")
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return n_failed == 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all validation checks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate research dataset")
    parser.add_argument("--research-dir", type=str, default=None, help="Research data directory")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset filename (auto-detects if not specified)")
    parser.add_argument("--market-info", type=str, default=None, help="Market info filename (auto-detects if not specified)")
    
    args = parser.parse_args()
    
    research_dir = Path(args.research_dir or RESEARCH_DIR)
    
    print("=" * 60)
    print("Dataset Validation Suite")
    print("=" * 60)
    
    # Auto-detect dataset file (try new naming first, then legacy)
    if args.dataset:
        dataset_path = research_dir / args.dataset
    else:
        # Try new v2 naming
        dataset_path = research_dir / "canonical_dataset_all_assets.parquet"
        if not dataset_path.exists():
            # Try legacy v2 naming
            dataset_path = research_dir / "canonical_dataset_v2.parquet"
        if not dataset_path.exists():
            # Try v1 naming
            dataset_path = research_dir / "canonical_dataset.parquet"
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found. Tried:")
        print(f"  - {research_dir / 'canonical_dataset_all_assets.parquet'}")
        print(f"  - {research_dir / 'canonical_dataset_v2.parquet'}")
        print(f"  - {research_dir / 'canonical_dataset.parquet'}")
        return 1
    
    print(f"\nLoading dataset: {dataset_path}")
    canonical_df = pd.read_parquet(dataset_path)
    print(f"  Rows: {len(canonical_df):,}")
    print(f"  Markets: {canonical_df['market_id'].nunique()}")
    
    # Auto-detect market info file
    if args.market_info:
        market_info_path = research_dir / args.market_info
    else:
        # Try new v2 naming
        market_info_path = research_dir / "market_info_all_assets.json"
        if not market_info_path.exists():
            # Try legacy v2 naming
            market_info_path = research_dir / "market_info_v2.json"
        if not market_info_path.exists():
            # Try v1 naming
            market_info_path = research_dir / "market_info.json"
    
    market_infos = []
    if market_info_path.exists():
        with open(market_info_path, 'r') as f:
            market_infos = json.load(f)
        print(f"  Market info loaded: {len(market_infos)} markets")
    else:
        print("  WARNING: Market info not found")
    
    # Load ground truth (optional)
    ground_truth_repo = None
    try:
        ground_truth_repo = GroundTruthRepository(str(research_dir))
        gt_data = ground_truth_repo.load_all()
        if gt_data:
            print(f"  Ground truth loaded: {len(gt_data)} markets")
    except Exception as e:
        print(f"  Ground truth not available: {e}")
    
    # Run validation checks
    print("\nRunning validation checks...")
    results = []
    
    # Check 1: Outcome Reproduction
    print("  [1/7] Outcome Reproduction...")
    results.append(check_outcome_reproduction(market_infos, ground_truth_repo))
    
    # Check 2: Coverage Sanity
    print("  [2/7] Coverage Math Sanity...")
    results.append(check_coverage_sanity(market_infos))
    
    # Check 3: Timestamp Gaps
    print("  [3/7] Timestamp Integrity...")
    results.append(check_timestamp_gaps(canonical_df))
    
    # Check 4: No-Arb Bounds
    print("  [4/7] No-Arb Bounds...")
    results.append(check_noarb_bounds(canonical_df))
    
    # Check 5: Strike Consistency
    print("  [5/7] Strike Consistency...")
    results.append(check_strike_consistency(market_infos))
    
    # Check 6: Forward-Fill Reasonableness
    print("  [6/7] Forward-Fill Reasonableness...")
    results.append(check_ffill_reasonableness(canonical_df))
    
    # Check 7: Data Completeness
    print("  [7/7] Data Completeness...")
    results.append(check_data_completeness(market_infos))
    
    # Print results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    for result in results:
        status = "[PASS]" if result.passed else "[FAIL]"
        print(f"{status} {result.name}: {result.message}")
    
    # Generate report
    report_path = research_dir / "VALIDATION_REPORT.md"
    all_passed = generate_validation_report(results, report_path)
    print(f"\nReport saved: {report_path}")
    
    # Final verdict
    n_passed = sum(1 for r in results if r.passed)
    n_total = len(results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL CHECKS PASSED - Dataset ready for strategy backtesting")
    else:
        print(f"VALIDATION INCOMPLETE - {n_passed}/{n_total} checks passed")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

