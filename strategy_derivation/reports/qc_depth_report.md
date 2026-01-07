# 6-Level Orderbook Data QC Report
**Generated**: 2026-01-07 08:43:05
**Markets Analyzed**: 41
**Total Rows**: 30,807

---

## Summary

| Check | Status | Details |
|-------|--------|--------|
| Monotonic Ladders | [PASS] | Up asks: 0.00%, Down asks: 0.00% violations |
| Crossed Books | [PASS] | Up: 0.00%, Down: 0.00% crossed |
| Positive Sizes | [PASS] | 0 negative values found |
| Depth Coverage | [PASS] | UP full-depth: 88.6%, DOWN full-depth: 88.1% |

---

## 1. Monotonic Ladder Check

Verifies that ask prices increase and bid prices decrease as we move deeper into the book.

| Side | Violation % |
|------|-------------|
| UP Asks | 0.000% |
| UP Bids | 0.000% |
| DOWN Asks | 0.000% |
| DOWN Bids | 0.000% |

---

## 2. Crossed Books Check

Verifies that best_bid <= best_ask (no arbitrage within the book).

- **UP token crossed**: 0 rows (0.00%)
- **DOWN token crossed**: 0 rows (0.00%)

---

## 3. Size Validity Check

- **Negative sizes found**: 0
- **Zero sizes (avg)**: 0.00%

---

## 4. Depth Coverage

Percentage of snapshots with all 6 levels available (with non-zero size).

- **UP asks full depth**: 88.6%
- **DOWN asks full depth**: 88.1%

---

## 5. Size Consistency

Checks for prices without sizes or sizes without prices.

- **Price exists but no size**: 0.00%
- **Size exists but no price**: 0.00%

---

## Overall Verdict

**[PASS] Data quality sufficient for depth-aware execution modeling.**

The 6-level orderbook data passes all critical checks and can be used for:
- VWAP calculations
- Executable size estimation
- Slippage modeling
