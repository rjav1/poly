# L6 Depth-Aware Validation Report

Generated: 2026-01-07T08:49:15.993638

## Summary

- Train markets: 24
- Test markets: 11
- Strategy params: {'buffer': 0.02, 'tau_max': 420, 'exit_rule': 'expiry'}

## Capacity Analysis

### q* Distribution

- Mean: 484.6
- Median: 500.0
- 25th-75th: [500.0, 500.0]

### Survival Rates

| Min q* | Survival Rate |
|--------|---------------|
| 5 | 100.0% |
| 10 | 99.4% |
| 25 | 99.4% |
| 50 | 98.7% |
| 100 | 98.7% |

## Tiered Validation

| Tier | Size | PnL | t-stat | Status |
|------|------|-----|--------|--------|
| tiny | 5 | $8.34 | 0.21 | FAIL |
| medium | 25 | $39.11 | 0.19 | FAIL |
| large | 100 | $121.30 | 0.15 | FAIL |

## Slippage Monte Carlo

| Tier | P(PnL>0) | P(t>2.0) | P(t>1.5) | Median t |
|------|----------|----------|----------|----------|
| tiny | 100.0% | 0.0% | 0.0% | 0.06 |
| medium | 100.0% | 0.0% | 0.0% | 0.05 |
| large | 88.0% | 0.0% | 0.0% | 0.01 |

## Verdict

**WEAK**: Edge doesn't survive even at tiny sizes. Review strategy.
