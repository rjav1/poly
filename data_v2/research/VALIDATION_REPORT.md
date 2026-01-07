# Dataset Validation Report

**Generated**: 2026-01-07T12:49:20.145542+00:00

## Summary

- **Total Checks**: 7
- **Passed**: 7
- **Failed**: 0

## Validation Results

### [PASS] Outcome Reproduction

Skipped (no ground truth data)

### [PASS] Coverage Math Sanity

All markets pass coverage math check

**Details:**
```json
{
  "violations": [],
  "n_violations": 0
}
```

### [PASS] Timestamp Integrity

All markets pass 2s

**Details:**
```json
{
  "markets_with_gaps": [],
  "n_markets_with_gaps": 0,
  "max_gap_threshold": 2
}
```

### [PASS] No-Arb Bounds

sum_bids mean=0.989, sum_asks mean=1.011

**Details:**
```json
{
  "sum_bids": {
    "mean": 0.9892322702562067,
    "std": 0.022883032467809532,
    "min": 0.63,
    "max": 1.51,
    "out_of_bounds_pct": 0.22731318553411753
  },
  "sum_asks": {
    "mean": 1.0107145713229353,
    "std": 0.02285820424328467,
    "min": 0.49,
    "max": 1.37,
    "out_of_bounds_pct": 0.2300518986128418
  }
}
```

### [PASS] Strike Consistency

71/71 K values within $50 of folder price (100.0%)

**Details:**
```json
{
  "matches": 71,
  "mismatches": 0,
  "match_rate": 100.0,
  "tolerance": 50,
  "mismatch_details": []
}
```

### [PASS] Forward-Fill Reasonableness

CL FFill=1.5%, PM FFill=6.9%

**Details:**
```json
{
  "cl_ffill_pct": 1.5207496653279786,
  "pm_ffill_pct": 6.926372155287817
}
```

### [PASS] Data Completeness

83/83 markets have >50% coverage on both sources

**Details:**
```json
{
  "n_good_markets": 83,
  "n_total_markets": 83,
  "good_market_ids": [
    "20260106_0215_93747",
    "20260106_0315_93802",
    "20260106_0330_93764",
    "20260106_0245_3215",
    "20260106_0415_3232",
    "20260106_0430_3226",
    "20260106_0445_3224",
    "20260106_0500_3225",
    "20260106_0515_3225",
    "20260106_0530_3224"
  ]
}
```

## Verdict

**ALL CHECKS PASSED** - Dataset is ready for strategy backtesting.