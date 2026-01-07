# Shadow Trader with Depth Logging - Report

**Generated**: 2026-01-07 09:40:26

**Total Signals Observed**: 45

---

## Executability Dashboard

| Size | % Executable |
|------|-------------|
| q >= 1 | 100.0% |
| q >= 5 | 84.4% |
| q >= 10 | 66.7% |

## Capacity Metrics

- **Median q_max**: 16.5 contracts
- **Mean q_max**: 34.4 contracts

## Edge Metrics

- **Median edge at q=1**: $0.0050
- **Mean edge at q=1**: $0.0090

---

## Edge Persistence

How long does the arbitrage opportunity persist after detection?

| Time After Signal | % Still Executable |
|-------------------|-------------------|
| +0.5 second | 100.0% |
| +1 second | 0.0% |
| +2 seconds | 0.0% |
| +5 seconds | 0.0% |

## Capacity Over Time

How does available capacity change after signal?

| Time | Avg q_max |
|------|-----------|
| At signal | 34.4 |
| +0.5s | 34.4 |
| +1s | 0.0 |
| +2s | 0.0 |
| +5s | 0.0 |

---

## Go/No-Go Decision

- **Executability at q=1**: 100.0% [PASS]
- **Median edge**: $0.0050 [PASS]
- **Persistence at +1s**: 0.0% [WARN]

### Verdict

**PROCEED WITH CAUTION**

Core criteria met but some warnings:
- [WARN] <50% persist at +1s
