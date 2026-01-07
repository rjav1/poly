#!/usr/bin/env python3
"""
Organize strategy_derivation directory.
"""

import shutil
from pathlib import Path

BASE = Path(__file__).parent

# Create directories
(BASE / 'scripts').mkdir(exist_ok=True)
(BASE / 'reports').mkdir(exist_ok=True)
(BASE / 'results').mkdir(exist_ok=True)
(BASE / 'data').mkdir(exist_ok=True)

# Move scripts
scripts = [
    '01_normalize_wallet_data.py',
    '02_hypothesis_tests.py',
    '03_extract_strategy_params.py',
    '04_run_backtests.py',
    '05_reproduce_pipeline.py',
    '06_leakage_audit.py',
    '07_placebo_suite.py',
    '08_oos_validation.py',
    '09_market_contribution.py',
    '10_execution_stress.py',
]

for script in scripts:
    src = BASE / script
    if src.exists():
        shutil.move(str(src), str(BASE / 'scripts' / script))
        print(f"Moved {script} -> scripts/")

# Move reports
reports = [
    'STRATEGY_DERIVATION_REPORT.md',
    'AUDIT_FINAL_VERDICT.md',
    'LEAKAGE_AUDIT_NOTE.md',
]

for report in reports:
    src = BASE / report
    if src.exists():
        shutil.move(str(src), str(BASE / 'reports' / report))
        print(f"Moved {report} -> reports/")

# Move results
results_json = [
    'hypothesis_results.json',
    'strategy_params.json',
    'leakage_audit_results.json',
    'placebo_suite_results.json',
    'oos_validation_results.json',
    'market_contribution_results.json',
    'execution_stress_results.json',
]

results_csv = [
    'parameter_sweep_results.csv',
    'latency_sensitivity_results.csv',
    'placebo_test_results.csv',
    'volume_subset_results.csv',
    'multi_shift_placebo_results.csv',
    'per_market_pnl.csv',
]

for result in results_json + results_csv:
    src = BASE / result
    if src.exists():
        shutil.move(str(src), str(BASE / 'results' / result))
        print(f"Moved {result} -> results/")

# Move data files
data_files = [
    'wallet_data_normalized.parquet',
]

for data_file in data_files:
    src = BASE / data_file
    if src.exists():
        shutil.move(str(src), str(BASE / 'data' / data_file))
        print(f"Moved {data_file} -> data/")

# Remove temporary files
temp_files = [
    '_output_hashes.json',
    '_organize.py',  # Remove this script itself
    'wallet_data_normalized_sample.csv',
]

for temp_file in temp_files:
    src = BASE / temp_file
    if src.exists():
        src.unlink()
        print(f"Removed {temp_file}")

# Remove backup directory
backup_dir = BASE / '_reproduce_backup'
if backup_dir.exists():
    shutil.rmtree(backup_dir)
    print(f"Removed {backup_dir.name}/")

# Handle check_volume_data.py - could keep or remove
check_script = BASE / 'check_volume_data.py'
if check_script.exists():
    # Move to scripts if it might be useful, otherwise delete
    # For now, let's remove it as it was a one-off check
    check_script.unlink()
    print(f"Removed check_volume_data.py")

print("\nOrganization complete!")
print("\nDirectory structure:")
print("  scripts/ - All pipeline scripts (01-10)")
print("  reports/ - Documentation and audit reports")
print("  results/ - All JSON/CSV result files")
print("  data/ - Processed data files")
print("  profitable_traders_wallet_data/ - Source wallet data (unchanged)")

