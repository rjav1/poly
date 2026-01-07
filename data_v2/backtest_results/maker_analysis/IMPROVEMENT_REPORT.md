# Spread Capture Strategy Improvement Report

**Date**: 2026-01-06  
**Status**: ALL TASKS COMPLETE

---

## Executive Summary

Completed all 7 improvement tasks for the spread capture / maker strategy:

| Task | Status | Key Finding |
|------|--------|-------------|
| 1. Fix AS Calculation | DONE | Sign convention fixed: positive = cost, negative = gain |
| 2. Calibrate Fill Model | DONE | PnL sign NOT robust ($-285 to $+67 range) |
| 3. AS Prediction Model | DONE | Only 15.3% of fills have AS cost; best filters ~$0.002/fill improvement |
| 4. Inventory Skew | DONE | No significant effect (t = -1.00) |
| 5. Net-Edge Threshold | DONE | Implemented, requires expected_as calibration |
| 6. Bayesian Analysis | DONE | P(mean > 0) = 68.3%, P(mean > $0.50) = 11.6% |
| 7. Optionality Check | DONE | No directional bias - PnL is from spread capture |

---

## 1. Adverse Selection Sign Convention Fix

**File**: `scripts/backtest/maker_execution.py`

**Problem Found**: AS was computed relative to fill_price instead of mid_at_fill, conflating spread capture with adverse selection.

**Fix Applied**:
- BID: `AS = mid_at_fill - mid_after` (positive = price dropped = cost)
- ASK: `AS = mid_after - mid_at_fill` (positive = price rose = cost)

**Validation**:
- Created `scripts/validate_as_fix.py`
- Verified AS + gain = 0 for all fills
- Verified PnL formula is internally consistent

---

## 2. Fill Model Calibration

**File**: `scripts/backtest/fill_model_calibration.py`

**Findings**:

| Fill Model | Total PnL | t-stat | Fill Rate |
|------------|-----------|--------|-----------|
| TOUCH_SIZE_PROXY (calibrated) | $1.76 | 0.51 | 1.3% |
| BOUNDS_OPTIMISTIC | $67.44 | 2.83 | 53.0% |
| BOUNDS_PESSIMISTIC | $-284.54 | -6.04 | 34.7% |

**Critical Finding**: **PnL sign is NOT robust** - varies from -$285 to +$67 depending on fill model assumptions.

**Calibrated Parameters**:
- `touch_trade_rate_per_second`: 0.0293 (from size depletion analysis)
- Upper bound: 0.0586 (all depletions are trades)
- Lower bound: 0.0147 (half are trades)

---

## 3. Adverse Selection Prediction Model

**File**: `scripts/backtest/as_prediction.py`

**Findings**:
- Total fills analyzed: 118
- Fills with AS cost (positive): 15.3%
- Fills with AS gain (negative): 33.1%
- Remaining fills: ~51% with AS near zero

**Best Filters** (but limited improvement):

| Filter | Fills Removed | Edge Improvement/Fill |
|--------|--------------|----------------------|
| AS <= 80th percentile | 15.3% | $0.0020 |
| AS <= 90th percentile | 8.5% | $0.0014 |

**Conclusion**: Hard filters provide minimal improvement. Most fills already have favorable or neutral AS.

---

## 4. Inventory Skew Quoting

**File**: `scripts/backtest/strategies.py`

**Implementation**: Added `inventory_skew_enabled` parameter:
- When UP inventory > `skew_threshold_up`: only quote ASK (sell to reduce)
- When UP inventory < -`skew_threshold_up`: only quote BID (buy to reduce short)

**LOOCV Results**:

| Metric | With Skew | Without Skew |
|--------|-----------|--------------|
| Total PnL | $1.68 | $1.78 |
| Total Fills | 112 | 118 |
| Mean diff | -$0.008 | - |
| t-stat | -1.00 | - |

**Conclusion**: **No significant effect**. Inventory rarely exceeds skew threshold in 15-minute markets.

---

## 5. Net-Edge Threshold

**File**: `scripts/backtest/strategies.py`

**Implementation**: Added `use_net_edge_threshold` parameter:
- Old: Quote if `spread >= spread_min`
- New: Quote if `expected_net_edge >= edge_threshold`

**Formula**:
```
expected_edge = 0.5 * spread - expected_as_per_spread * spread - edge_buffer
```

**Parameters**:
- `edge_threshold`: Minimum expected net edge (default: 0.0)
- `edge_buffer`: Safety buffer (default: 0.005 = 0.5c)
- `expected_as_per_spread`: E[AS] as fraction of spread (default: 0.5)

---

## 6. Bayesian Analysis

**File**: `scripts/backtest/bayesian_analysis.py`

**Results** (12 markets):

| Metric | Value |
|--------|-------|
| Sample mean | $0.14 |
| Sample std | $0.99 |
| P(mean > 0) | 68.3% |
| P(mean > $0.50) | 11.6% |
| 95% CI | [$-0.49, $0.77] |
| t-stat | 0.49 |
| p-value | 0.634 |

**Interpretation**:
- Weak/inconclusive evidence of positive edge (68% probability)
- 95% CI includes zero
- Only 12% chance edge exceeds $0.50/market

---

## 7. Optionality Check

**File**: `scripts/backtest/optionality_check.py`

**Correlation Analysis**:

| Correlation | Value | P-value |
|-------------|-------|---------|
| PnL vs Direction | N/A | N/A |
| PnL vs Max Exposure | 0.00 | 1.00 |
| PnL vs Spread Captured | 0.08 | 0.80 |

**Conclusion**: **No directional bias detected**. PnL appears to be from spread capture, not directional bets.

---

## Key Conclusions

### Strategy Assessment

1. **PnL is positive but not significant**: $1.68 total, t = 0.49
2. **68% probability of positive edge** (Bayesian)
3. **Fill model uncertainty is the biggest risk**: PnL ranges from -$285 to +$67
4. **No directional bias**: PnL is from spread capture
5. **Inventory skew doesn't help** in short-horizon markets
6. **Hard filters provide minimal improvement**

### Structural Issues

1. **Fill uncertainty dominates**: Need trade tape data to reduce model risk
2. **Small sample size**: 12 markets is insufficient for statistical significance
3. **Low fill rate**: 1.3% fill rate limits potential PnL

### Recommendations

1. **DO NOT TRADE** until fill model is better validated
2. **Collect more data**: Need 50+ markets for reliable statistics
3. **Investigate trade tape**: Check if Polymarket has WebSocket trade feed
4. **Consider taker hybrid**: Combine maker quotes with taker fills when spreads compress

---

## Files Created/Modified

**New Files**:
- `scripts/backtest/fill_model_calibration.py`
- `scripts/backtest/as_prediction.py`
- `scripts/backtest/bayesian_analysis.py`
- `scripts/backtest/optionality_check.py`
- `scripts/validate_as_fix.py`
- `scripts/test_inventory_skew_loocv.py`

**Modified Files**:
- `scripts/backtest/maker_execution.py`: Fixed AS calculation, added validate_pnl_decomposition()
- `scripts/backtest/strategies.py`: Added inventory skew, net-edge threshold parameters

---

## Next Steps

1. **Collect more markets with size data** to improve statistical power
2. **Investigate WebSocket APIs** for real-time trade tape
3. **Test taker-maker hybrid** that takes when spreads are tight
4. **Run out-of-sample validation** on new data as it becomes available

---

**Report Generated**: 2026-01-06  
**Framework Version**: 2.0 (with all improvements)

