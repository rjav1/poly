# Shadow Trader Report - Complete-Set Arb

**Generated**: 2026-01-06 21:56:35

**Total Signals Observed**: 78

---

## Fill Rate Analysis

| Execution Model | Filled | Total | Fill Rate |
|----------------|--------|-------|-----------|
| Taker | 14 | 78 | 17.9% |
| Maker (Conservative) | 0 | 78 | 0.0% |
| Maker (Realistic) | 39 | 78 | 50.0% |

## Realized Edge Analysis

| Execution Model | Filled Signals | Avg Edge/Fill | Total PnL |
|----------------|----------------|---------------|-----------|
| Taker | 14 | $0.0114 | $0.16 |
| Maker (Conservative) | 0 | $0.0000 | $0.00 |
| Maker (Realistic) | 39 | $0.7011 | $27.34 |

## Untradeable Rate Analysis

- **Capacity Limited (< 1 contract)**: 64/78 (82.1%)

## Post-Signal Evolution

| Time After Signal | Still Has Underround | % of Signals |
|-------------------|----------------------|--------------|
| +5 seconds | 2 | 2.6% |
| +10 seconds | 0 | 0.0% |
| +30 seconds | 0 | 0.0% |
| +60 seconds | 0 | 0.0% |

## Go/No-Go Decision

### Criteria:
1. Fill rate > 30% (taker)
2. Average realized edge per fill > $0.01
3. Untradeable rate < 70%

- **Fill rate**: 17.9% [FAIL]
- **Average edge per fill**: $0.0114 [PASS]
- **Untradeable rate**: 82.1% [FAIL]

### Verdict:

**NO-GO - NEEDS MORE WORK**

Criteria not met:
- [FAIL] Fill rate < 30%
- [FAIL] Untradeable > 70%
