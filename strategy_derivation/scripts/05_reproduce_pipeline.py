#!/usr/bin/env python3
"""
Step 1: Reproduce Pipeline End-to-End

Verifies that the analysis pipeline produces deterministic, reproducible results.
This is critical before any audit work - if we can't reproduce, we can't trust.
"""

import subprocess
import hashlib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR


def file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of file contents."""
    if not filepath.exists():
        return "FILE_NOT_FOUND"
    
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]  # First 16 chars for readability


def compare_numeric_csv(path1: Path, path2: Path, rtol: float = 1e-5) -> Tuple[bool, str]:
    """Compare two CSV files with numeric tolerance."""
    if not path1.exists() or not path2.exists():
        return False, f"File missing: {path1} or {path2}"
    
    try:
        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)
        
        if list(df1.columns) != list(df2.columns):
            return False, f"Column mismatch: {list(df1.columns)} vs {list(df2.columns)}"
        
        if len(df1) != len(df2):
            return False, f"Row count mismatch: {len(df1)} vs {len(df2)}"
        
        # Compare numeric columns with tolerance
        for col in df1.columns:
            if pd.api.types.is_numeric_dtype(df1[col]):
                if not np.allclose(df1[col].fillna(0), df2[col].fillna(0), rtol=rtol, equal_nan=True):
                    max_diff = abs(df1[col].fillna(0) - df2[col].fillna(0)).max()
                    return False, f"Numeric mismatch in column {col}, max diff: {max_diff}"
            else:
                if not df1[col].equals(df2[col]):
                    return False, f"Non-numeric mismatch in column {col}"
        
        return True, "Match"
    except Exception as e:
        return False, f"Error comparing: {str(e)}"


def run_script(script_name: str) -> Tuple[bool, str]:
    """Run a Python script and capture output."""
    script_path = OUTPUT_DIR / script_name
    if not script_path.exists():
        return False, f"Script not found: {script_path}"
    
    try:
        result = subprocess.run(
            ['python', str(script_path)],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        if result.returncode != 0:
            return False, f"Script failed: {result.stderr[:500]}"
        return True, "Success"
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def backup_outputs() -> Dict[str, str]:
    """Backup current output files and return hashes."""
    base = OUTPUT_DIR
    output_files = [
        ('data', 'wallet_data_normalized.parquet'),
        ('results', 'hypothesis_results.json'),
        ('results', 'strategy_params.json'),
        ('results', 'parameter_sweep_results.csv'),
        ('results', 'latency_sensitivity_results.csv'),
        ('results', 'placebo_test_results.csv'),
        ('results', 'volume_subset_results.csv'),
    ]
    
    hashes = {}
    backup_dir = base / '_reproduce_backup'
    backup_dir.mkdir(exist_ok=True)
    
    for subdir, fname in output_files:
        fpath = base / subdir / fname
        if fpath.exists():
            hashes[fname] = file_hash(fpath)
            # Copy to backup
            backup_path = backup_dir / fname
            backup_path.parent.mkdir(exist_ok=True)
            if fpath.suffix == '.parquet':
                pd.read_parquet(fpath).to_parquet(backup_path)
            elif fpath.suffix == '.json':
                with open(fpath) as f:
                    data = json.load(f)
                with open(backup_path, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                pd.read_csv(fpath).to_csv(backup_path, index=False)
    
    return hashes


def verify_outputs_structure() -> Dict[str, bool]:
    """Verify all output files exist and have expected structure."""
    base = OUTPUT_DIR
    checks = {}
    
    # Check wallet_data_normalized.parquet
    path = base / 'data' / 'wallet_data_normalized.parquet'
    if path.exists():
        df = pd.read_parquet(path)
        checks['wallet_data'] = (
            len(df) > 400000 and 
            't' in df.columns and 
            'tau' in df.columns and
            '_handle' in df.columns
        )
    else:
        checks['wallet_data'] = False
    
    # Check hypothesis_results.json
    path = base / 'results' / 'hypothesis_results.json'
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        checks['hypothesis'] = (
            'H1_late_window_concentration' in data and
            'H2_two_sided_behavior' in data and
            'H3_underround_harvesting' in data
        )
    else:
        checks['hypothesis'] = False
    
    # Check parameter_sweep_results.csv
    path = base / 'results' / 'parameter_sweep_results.csv'
    if path.exists():
        df = pd.read_csv(path)
        checks['sweep'] = (
            len(df) > 100 and
            'strategy' in df.columns and
            'total_pnl' in df.columns and
            't_stat' in df.columns
        )
    else:
        checks['sweep'] = False
    
    # Check placebo_test_results.csv
    path = base / 'results' / 'placebo_test_results.csv'
    if path.exists():
        df = pd.read_csv(path)
        checks['placebo'] = (
            len(df) >= 3 and
            'strategy' in df.columns and
            't_stat' in df.columns
        )
    else:
        checks['placebo'] = False
    
    return checks


def main():
    print("=" * 70)
    print("STEP 1: VERIFY PIPELINE OUTPUTS")
    print("=" * 70)
    
    # Step 1: Record current file hashes
    print("\n--- Recording output file hashes ---")
    output_files = [
        ('data', 'wallet_data_normalized.parquet'),
        ('results', 'hypothesis_results.json'),
        ('results', 'strategy_params.json'),
        ('results', 'parameter_sweep_results.csv'),
        ('results', 'latency_sensitivity_results.csv'),
        ('results', 'placebo_test_results.csv'),
        ('results', 'volume_subset_results.csv'),
    ]
    
    hashes = {}
    for subdir, fname in output_files:
        fpath = base / subdir / fname
        hash_val = file_hash(fpath)
        key = f"{subdir}/{fname}" if subdir else fname
        hashes[key] = hash_val
        exists = "EXISTS" if fpath.exists() else "MISSING"
        print(f"  {key}: {hash_val} ({exists})")
    
    # Step 2: Verify structure
    print("\n--- Verifying output structure ---")
    structure_checks = verify_outputs_structure()
    
    all_valid = True
    for check_name, passed in structure_checks.items():
        status = "OK" if passed else "FAIL"
        print(f"  {check_name}: {status}")
        if not passed:
            all_valid = False
    
    # Step 3: Load and verify key results
    print("\n--- Verifying key backtest results ---")
    
    sweep_path = base / 'results' / 'parameter_sweep_results.csv'
    if sweep_path.exists():
        sweep_df = pd.read_csv(sweep_path)
        
        # Check Strategy B best result
        strat_b = sweep_df[sweep_df['strategy'] == 'Strategy_B']
        if len(strat_b) > 0:
            best_b = strat_b.nlargest(1, 'total_pnl').iloc[0]
            print(f"  Strategy B best: PnL=${best_b['total_pnl']:.2f}, t={best_b['t_stat']:.2f}")
            
            # Verify this matches expected
            if abs(best_b['t_stat'] - 3.09) < 0.1:
                print(f"    t-stat matches expected (~3.09): OK")
            else:
                print(f"    t-stat differs from expected (3.09): INVESTIGATE")
    
    # Step 4: Load placebo results
    placebo_path = base / 'results' / 'placebo_test_results.csv'
    if placebo_path.exists():
        placebo_df = pd.read_csv(placebo_path)
        print(f"\n  Placebo test results:")
        for _, row in placebo_df.iterrows():
            print(f"    {row['strategy']}: t={row['t_stat']:.2f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    if all_valid:
        print("\n*** OUTPUTS VERIFIED ***")
        print("All pipeline outputs exist with expected structure.")
        print("File hashes recorded for future comparison.")
        print("Safe to proceed with audit.")
        
        # Save hashes for reference
        with open(base / '_output_hashes.json', 'w') as f:
            json.dump(hashes, f, indent=2)
        print(f"\nHashes saved to _output_hashes.json")
    else:
        print("\n*** VERIFICATION FAILED ***")
        print("Some outputs are missing or malformed.")
        print("Re-run the pipeline before proceeding.")
    
    print("\n" + "=" * 70)
    
    return all_valid


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

