# Phase 6: High-Value Stress Tests Report

**Generated:** 2026-01-07T10:01:24.568688
**Quick Mode:** True

## Executive Summary

**Overall Verdict:** WEAK

## 1. Data-Snooping / Selection Bias (6.1)

### Nested Selection Validation

- **Median outer t-stat:** 1.47
- **Min outer t-stat:** 0.86
- **Max outer t-stat:** 3.09
- **P(t > 2.0):** 33.3%

### SPA Test

- **SPA p-value:** 0.1250
- **Unadjusted p-value:** 0.2586
- **Selection bias adjustment:** +-0.1336
- **Significant at 5%:** False

## 2. Execution Realism (6.2)

### Slippage Monte Carlo

- **P(t > 2.0) with uniform slippage:** 0.0%
- **Median t-stat:** 0.99

### Quote Fade

- **0s delay:** t=1.25
- **1s delay:** t=1.31
- **2s delay:** t=1.29
- **3s delay:** t=1.22
- **5s delay:** t=1.49

## 3. Robustness (6.3)

### Leave-One-Market-Out

- **Min t-stat:** 0.74
- **Median t-stat:** 1.27
- **Max t-stat:** 1.40

### Winsorization

- **80th percentile:** t=1.48
- **90th percentile:** t=1.44
- **95th percentile:** t=1.38
- **99th percentile:** t=1.25
- **100th percentile:** t=1.25

## 4. Model Risk (6.4)

### Observed-Only Test

- **Observed-only t-stat:** 1.22
- **Edge persists:** False

### Calibration

- **Brier score:** 0.2536
- **ECE:** 0.0612

## 5. Regime Stress (6.5)

- **Best tau bucket:** 0-60s
- **Worst tau bucket:** 300-420s

## Conclusion

The strategy receives an overall verdict of **WEAK**.

### Recommendations

- Strategy may be a false positive
- Significant risk of data-snooping or overfitting
- Do not deploy without substantial improvements