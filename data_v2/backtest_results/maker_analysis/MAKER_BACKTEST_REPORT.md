# Spread Capture / Maker Strategy Report

**Generated**: 2026-01-06 20:56:28

**Strategy**: SpreadCapture(2sided,spread>0.01,tau[60,600])

---

## Strategy Configuration

```
spread_min: 0.01
tau_min: 60
tau_max: 600
inventory_limit_up: 10.0
inventory_limit_down: 10.0
tau_flatten: 60
quote_improvement_ticks: 0
adverse_selection_filter: True
cl_jump_threshold_bps: 10.0
quote_size: 1.0
two_sided: True
token: UP
```

## Executive Summary

### Performance Metrics

| Metric | Value |
|--------|-------|
| Total PnL | $12.3400 |
| Mean PnL/Market | $1.0283 |
| t-statistic | 0.98 |
| Markets with positive PnL | 58.3% |
| Number of Markets | 12 |
| Total Fills | 436 |
| Fill Rate | 4.64% |

### PnL Decomposition

| Component | Value |
|-----------|-------|
| Spread Captured | $2.6150 |
| Adverse Selection | -$-4.5150 |
| Inventory Carry | $5.2100 |
| Realized PnL | $26.3133 |

### Fill Statistics

| Metric | Value |
|--------|-------|
| Orders Placed | 9400 |
| Orders Filled | 436 |
| Orders Cancelled | 8964 |
| Orders Expired | 0 |
| Fill Rate | 4.64% |
| Avg Time to Fill | 4.1s |

---

## Detailed Diagnostics

### Fill Rate by Time-to-Expiry (tau)

| Tau Window | Fills | % of Total |
|------------|-------|------------|
| 0-60s | 0 | 0.0% |
| 60-120s | 38 | 8.7% |
| 120-180s | 23 | 5.3% |
| 180-300s | 124 | 28.4% |
| 300-600s | 251 | 57.6% |
| 600-900s | 0 | 0.0% |

### Fill Rate by Spread Width

| Spread | Fills | % of Total |
|--------|-------|------------|
| 0-1c | 117 | 26.8% |
| 1-2c | 168 | 38.5% |
| 2-3c | 84 | 19.3% |
| 3-5c | 6 | 1.4% |
| 5-10c | 0 | 0.0% |

### Adverse Selection Analysis

- Avg adverse selection (1s): -1.025c
- Avg adverse selection (5s): -1.021c
- % fills with gain (1s): 91.3%
- % fills with gain (5s): 77.3%

---

## Latency Sensitivity

**Placement Latency Cliff**: 500ms

**Cancel Latency Cliff**: 50ms


### PnL by Placement Latency

| Place Latency | Total PnL | t-stat | Fill Rate |
|---------------|-----------|--------|-----------|
| 0.0ms | $12.3400 | 0.98 | 4.64% |
| 50.0ms | $12.3400 | 0.98 | 4.64% |
| 100.0ms | $12.3400 | 0.98 | 4.64% |
| 200.0ms | $12.3400 | 0.98 | 4.64% |
| 500.0ms | $12.3400 | 0.98 | 4.64% |

---

## Placebo Test Results

### Randomized Timing Test

- Real PnL: $12.3400
- Placebo Mean: $12.3400
- P-value: 1.000
- **Result**: FAIL
- FAIL: Real strategy not significantly different from random (p >= 0.1). Edge may be spurious.

### Stale Data Test

- **Result**: FAIL

### Flipped Sides Test

- Real PnL: $12.3400
- Flipped PnL: $-0.6200
- **Result**: PASS
- Side selection matters (real > flipped)


**Overall Validation**: NEEDS REVIEW

**Tests Passed**: 1/3

---

## Stress Test Results

**Robustness Score**: 100%

### Slippage Tolerance

- Tolerance: 999bps
- Strategy tolerates up to 999bps slippage

### Spread Widening

- Robust at 1.5x spreads: Yes
- Robust to widened spreads

### Volatility Dependence

- Robust without top 10% volatile: Yes
- Edge persists without volatile periods

### Fill Rate Sensitivity

- Robust across rates: Yes
- Robust to fill rate assumptions


---

## Parameter Sweep Results

### Top 10 Parameter Combinations by t-stat

| Spread Min | Tau Min | Place Lat | PnL | t-stat | Fill Rate |
|------------|---------|-----------|-----|--------|-----------|
| 0.01 | 120 | 0ms | $13.6750 | 1.09 | 4.69% |
| 0.01 | 120 | 100ms | $13.6750 | 1.09 | 4.69% |
| 0.01 | 60 | 0ms | $12.3400 | 0.98 | 4.64% |
| 0.01 | 60 | 100ms | $12.3400 | 0.98 | 4.64% |
| 0.02 | 120 | 0ms | $-0.8350 | -0.28 | 0.83% |
| 0.02 | 120 | 100ms | $-0.8350 | -0.28 | 0.83% |
| 0.02 | 60 | 0ms | $-1.8100 | -0.59 | 0.89% |
| 0.02 | 60 | 100ms | $-1.8100 | -0.59 | 0.89% |
| 0.015 | 120 | 0ms | $-3.7150 | -1.11 | 0.83% |
| 0.015 | 120 | 100ms | $-3.7150 | -1.11 | 0.83% |

---

## Conclusions and Recommendations

- ❌ Not statistically significant (t < 1.0)
- ⚠️ Low fill rate (4.6%) - may need more aggressive quoting
- ❌ Failed placebo tests - edge may be spurious
- ✅ Passed majority of stress tests


### Overall Verdict: **PROCEED WITH CAUTION**


---


*Report generated with spread capture testing framework v1.0*