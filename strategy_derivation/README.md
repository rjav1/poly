# Strategy Derivation & Audit

This directory contains the complete pipeline for deriving and auditing trading strategies from wallet transaction history.

## Directory Structure

```
strategy_derivation/
├── scripts/              # Pipeline scripts (run in order 01-10)
├── reports/              # Documentation and audit reports
├── results/              # All output files (JSON/CSV)
├── data/                 # Processed data files
└── profitable_traders_wallet_data/  # Source wallet transaction data
```

## Pipeline Scripts

### Phase 1: Strategy Derivation
1. `01_normalize_wallet_data.py` - Normalize wallet activity data (t/tau coordinates)
2. `02_hypothesis_tests.py` - Run hypothesis tests on trader behavior
3. `03_extract_strategy_params.py` - Extract strategy parameters from hypotheses
4. `04_run_backtests.py` - Backtest strategies with parameter sweeps

### Phase 2: Audit & Validation
5. `05_reproduce_pipeline.py` - Verify pipeline reproducibility
6. `06_leakage_audit.py` - Hard audit for look-ahead bias
7. `07_placebo_suite.py` - Enhanced placebo tests (multi-shift, permutation)
8. `08_oos_validation.py` - Out-of-sample validation (train/test, bootstrap)
9. `09_market_contribution.py` - Per-market PnL analysis
10. `10_execution_stress.py` - Execution realism tests (slippage, spread filters)

## Key Reports

- **`reports/STRATEGY_DERIVATION_REPORT.md`** - Main findings from strategy derivation
- **`reports/AUDIT_FINAL_VERDICT.md`** - Complete audit conclusions
- **`reports/LEAKAGE_AUDIT_NOTE.md`** - Detailed leakage audit findings

## Results

All output files are in `results/`:
- `hypothesis_results.json` - Hypothesis test outcomes
- `strategy_params.json` - Extracted strategy parameters
- `parameter_sweep_results.csv` - Backtest parameter sweep results
- `*_results.json` - Results from audit scripts (leakage, placebo, OOS, etc.)
- `*_results.csv` - Additional CSV outputs

## Data

- `data/wallet_data_normalized.parquet` - Normalized wallet transaction data
- `profitable_traders_wallet_data/` - Raw source data (wallet activity CSVs)

## Quick Start

1. Run derivation pipeline:
   ```bash
   python scripts/01_normalize_wallet_data.py
   python scripts/02_hypothesis_tests.py
   python scripts/03_extract_strategy_params.py
   python scripts/04_run_backtests.py
   ```

2. Run audit pipeline:
   ```bash
   python scripts/05_reproduce_pipeline.py
   python scripts/06_leakage_audit.py
   python scripts/07_placebo_suite.py
   python scripts/08_oos_validation.py
   python scripts/09_market_contribution.py
   python scripts/10_execution_stress.py
   ```

## Audit Verdict

**Strategy B (Late Directional Taker): DO NOT TRADE**

See `reports/AUDIT_FINAL_VERDICT.md` for full details.

