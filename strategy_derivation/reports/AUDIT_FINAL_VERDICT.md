# Strategy B Audit: Final Verdict

**Date:** January 6, 2026  
**Updated:** January 6, 2026 (OOS test corrected)  
**Subject:** Comprehensive audit of Strategy B (Late Directional Taker)  
**Status:** AUDIT COMPLETE - CORRECTED

---

## Executive Summary

| Criterion | Result | Verdict |
|-----------|--------|---------|
| Leakage/Look-ahead | No bugs found | PASS |
| Placebo Test | Edge persists (7% decay at 30s) | EXPLAINED |
| Out-of-Sample | 65% degradation (t=0.94 on test) | **FAIL** |
| Market Robustness | 94% hit rate, no single market dominance | PASS |
| Execution Realism | Survives 1c slippage (t=2.43) | MARGINAL |

**OVERALL VERDICT: DO NOT TRADE**

**Data Fix Applied:** The original 12 test markets had corrupted data (column shift in chainlink.csv). After fixing all 15 broken markets and rebuilding:

- **Train (30 markets)**: t=2.70, PnL=$3.29
- **Test (13 markets)**: t=0.94, PnL=$1.77
- **Degradation: 65%**

Strategy B does NOT maintain significant edge out-of-sample. Test t-stat (0.94) is below threshold (1.5).

---

## Detailed Findings

### 1. Leakage Audit (PASSED)

No data leakage or look-ahead bias was found:

- K (strike price) is correctly set at t=0
- delta_bps computed correctly: (cl_mid - K) / K * 10000
- Strategy uses only: t, tau, delta_bps, momentum (all causal)
- Placebo shift implementation is correct (shift(30) = staler data)
- No access to settlement, Y, or future prices

**Key Finding:** The placebo test "failure" is explained by high autocorrelation in delta_bps (0.779 at 30s lag). This is not a bug - it means Strategy B exploits price **persistence**, not short-term CL-PM lead-lag.

### 2. Enhanced Placebo Suite

| Shift | t-stat | Decay from 0s |
|-------|--------|---------------|
| 0s    | 3.09   | 0% |
| 5s    | 3.04   | 2% |
| 15s   | 2.38   | 23% |
| 30s   | 2.87   | 7% |
| 60s   | -1.04  | 134% |
| 120s  | -1.61  | 152% |

**Observations:**
- Low decay at 30s confirms persistence-based edge
- Edge reverses at 60s+ (noise)
- Permutation p-value = 0.0000 (strategy exploits temporal patterns)
- Sign-flip creates opposite edge (t=-3.05), confirming directionality

**Interpretation:** Strategy B exploits the persistence of CL being above/below strike. If CL was 10+ bps above strike 30 seconds ago, it's likely still above strike now. This is momentum/trend-following, not information speed.

### 3. Out-of-Sample Validation (CORRECTED - FAIL)

**Data Fix Applied:** The original 12 test markets had corrupted chainlink.csv (column shift). After fixing:
- 15 markets repaired
- Canonical dataset rebuilt with 43 ETH markets
- All markets now have valid delta_bps

**Final OOS Result (all 43 fixed markets):**
| Set | Markets | Signals | Trades | t-stat | PnL |
|-----|---------|---------|--------|--------|-----|
| Train | 30 | 65 | 65 | **2.70** | $3.29 |
| Test | 13 | 35 | 31 | **0.94** | $1.77 |

**Degradation: 65%**

Test set had plenty of opportunities (3,306 rows with |delta|>=10bps) and generated 35 signals.
The test t-stat (0.94) is below significance threshold (1.5).

**Conclusion: Strategy B does NOT maintain significant edge out-of-sample.**

### 4. Market Contribution Analysis (PASSED)

| Metric | Value |
|--------|-------|
| Markets with trades | 17 |
| Hit rate (positive PnL markets) | 94.1% |
| Top 1 market share | 22.5% |
| Top 3 market share | 48.4% |
| Max leave-one-out t-drop | 7.9% |

**No single market dominates** - removing any top market only reduces t-stat by ~8%. The edge appears distributed across markets that did trade.

### 5. Execution Stress Tests (MARGINAL)

| Scenario | t-stat | PnL |
|----------|--------|-----|
| Baseline | 3.09 | $3.16 |
| 0.5c slippage | 2.78 | $2.73 |
| 1c slippage | 2.43 | $2.30 |
| 2c slippage | 1.60 | $1.44 |
| Spread < 5c only | 3.09 | $3.15 |

- Strategy tolerates ~1c slippage before becoming marginal
- Performance is similar whether filtering for tight spreads or not
- At 2c slippage, t-stat drops to 1.60 (barely significant)

---

## Root Cause Analysis

### Why does Strategy B fail OOS despite good in-sample metrics?

1. **Small sample size:** Only 36 ETH markets total, split into 24 train / 12 test. The test set has extremely low power.

2. **Time-period clustering:** All markets are from a narrow time window. Market microstructure may have been different during the train period.

3. **Regime sensitivity:** The persistence pattern that Strategy B exploits may not be stable across different market conditions.

4. **Parameter overfitting:** The optimal parameters (tau_max=420, delta_threshold=10bps) were selected on the same 36 markets they're tested on.

### What the strategy actually does

```
Strategy B logic:
IF in last 7 minutes (tau < 420)
AND CL is 10+ bps from strike
AND momentum confirms direction
THEN bet on that direction
```

This exploits: "Once CL moves away from strike, it tends to stay away."

This is a valid market pattern, but:
- It may not persist out-of-sample
- It requires specific market conditions (trending CL)
- With 36 markets, we can't reliably distinguish signal from noise

---

## Verdict Summary (FINAL)

### Strategy B FAILS out-of-sample validation

| Factor | Assessment |
|--------|------------|
| Is there leakage? | No |
| Is the in-sample edge real? | Yes (t=2.70 on 30 markets) |
| Does it generalize OOS? | **NO** (t=0.94 on 13 test markets) |
| Is it execution-robust? | Marginal (1c tolerance) |
| Was data issue fixed? | Yes (15 markets repaired, 43 total now valid) |

### Why does Strategy B fail OOS?

The strategy exploits delta_bps persistence (high autocorrelation at 30s lag = 0.78).
However, this pattern appears to be:
1. **Time-period specific** - worked on morning markets, failed on afternoon markets
2. **Regime dependent** - the test period may have had different volatility/conditions
3. **Overfitted** - in-sample optimization may have found parameters that don't generalize

---

## Recommendations

### Immediate Actions

1. **Do NOT deploy Strategy B** - it fails OOS validation
2. **Investigate regime differences** between train (morning) and test (afternoon) periods
3. **Consider alternative strategies** - Strategy A or C may have better OOS properties

### For Future Research

1. **Time-of-day analysis:** Is the persistence pattern only present in certain hours?
2. **Volatility regime filter:** Does the strategy work only in high/low volatility?
3. **Cross-asset validation:** Test on BTC/SOL to see if pattern is ETH-specific
4. **Rolling OOS:** Instead of single split, use walk-forward validation

### For the PhD Researcher

The final audit found:
- **No code bugs** - the implementation is correct
- **A real in-sample pattern** - delta_bps persistence (autocorr=0.78 at 30s)
- **Data issues fixed** - 15 markets repaired, 43 total now valid
- **OOS FAILURE** - t=0.94 on 13 test markets (threshold is 1.5)

The strategy does NOT generalize. This could be due to:
- Regime change between train/test periods
- Time-of-day effects
- Parameter overfitting

---

## Files Generated

| File | Description |
|------|-------------|
| `05_reproduce_pipeline.py` | Pipeline verification |
| `06_leakage_audit.py` | Hard leakage audit |
| `LEAKAGE_AUDIT_NOTE.md` | Leakage audit findings |
| `07_placebo_suite.py` | Enhanced placebo tests |
| `08_oos_validation.py` | Original train/test (invalid) |
| `09_market_contribution.py` | Per-market analysis |
| `10_execution_stress.py` | Execution stress tests |
| `11_oos_audit.py` | OOS test audit (found data issue) |
| `12_oos_corrected.py` | Corrected OOS validation |
| `oos_audit_results.json` | Audit diagnosis |
| `oos_corrected_results.json` | **Corrected OOS results** |

---

**Audit conducted:** January 6, 2026  
**Data fix applied:** January 6, 2026 (15 markets repaired)  
**Final OOS run:** January 6, 2026 (43 ETH markets)  
**Methodology:** 6-step falsification protocol + data repair + OOS audit  

**FINAL CONCLUSION: STRATEGY B FAILS OOS VALIDATION - DO NOT TRADE**

The strategy shows a real in-sample pattern (delta_bps persistence) but does not generalize to held-out markets. Test t-stat = 0.94 is below the 1.5 significance threshold.

