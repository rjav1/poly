# Depth-Aware Complete-Set Arb Backtest Report

**Generated**: 2026-01-07 09:41:24

## Parameters

- **Epsilon (min underround)**: 0.5%
- **Tau window**: [600, 900] seconds
- **Slippage buffer**: 0.5%
- **Max position size**: 10 contracts
- **Cooldown**: 30 seconds

---

## Summary

- **Total Signals**: 45
- **Markets with Signals**: 29
- **Total PnL**: $3.44
- **Avg PnL per Signal**: $0.0765
- **Avg Edge per Set**: $0.0083

---

## Capacity Analysis

Maximum executable size (contracts) at each signal:

| Percentile | q_max |
|------------|-------|
| p10 | 4.9 |
| p50 (median) | 16.5 |
| p90 | 99.2 |
| Mean | 34.4 |

---

## Executability by Size

| Target Size | Executable Signals | % |
|-------------|-------------------|---|
| q=1 | 45 | 100.0% |
| q=5 | 38 | 84.4% |
| q=10 | 30 | 66.7% |
| q=20 | 18 | 40.0% |

---

## Expected PnL and Edge by Size

| Size | Avg Edge/Set | Expected PnL/Signal |
|------|---------------|--------------------|
| q=1 | $0.0090 | $0.0090 |
| q=5 | $0.0084 | $0.0418 |
| q=10 | $0.0083 | $0.0831 |
| q=20 | $0.0076 | $0.1511 |

---

## VWAP Analysis

- **Avg VWAP UP**: $0.5102
- **Avg VWAP DOWN**: $0.4764
- **Avg Set Cost (VWAP)**: $0.9867
- **Avg Set Cost (L1 only)**: $0.9860
- **Slippage from L1**: $0.0007

---

## Comparison to L1-Only Baseline

| Metric | L1-Only | Depth-Aware |
|--------|---------|-------------|
| Executable at q>=1 | 45 (100.0%) | 45 (100.0%) |
| Avg Capacity | 31.0 | 34.4 |
| Improvement Factor | - | 1.00x |

**[PASS] >50% of signals executable at q>=1 with depth-aware sizing**

---

## Go/No-Go Assessment

- **Executability**: 100.0% >= 50% [PASS]
- **Avg Edge**: $0.0083 < $0.01 [FAIL]
- **Capacity Improvement**: 1.00x < 1.5x [INFO]

### Verdict

**CAUTION - Some criteria not met**

- [FAIL] Avg edge < $0.01
