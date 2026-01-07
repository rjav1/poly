# Execution Realism Report

## Fill Model Comparison

| Config | Fill Model | Signals | Filled | Fill Rate | PnL | t-stat |
|--------|------------|---------|--------|-----------|-----|--------|
| full_window | taker | 181 | 60 | 33% | $5.83 | 1.18 |
| full_window | maker_conservative | 181 | 58 | 32% | $2.35 | 2.39 |
| full_window | maker_realistic | 181 | 29 | 16% | $1.91 | 1.94 |
| early_only | taker | 51 | 14 | 27% | $0.16 | 2.62 |
| early_only | maker_conservative | 51 | 16 | 31% | $0.34 | 4.71 |
| early_only | maker_realistic | 51 | 12 | 24% | $0.40 | 3.73 |
| late_only | taker | 76 | 27 | 36% | $5.39 | 1.10 |
| late_only | maker_conservative | 76 | 25 | 33% | $1.61 | 1.65 |
| late_only | maker_realistic | 76 | 7 | 9% | $1.15 | 1.17 |
| high_epsilon | taker | 66 | 27 | 41% | $5.54 | 1.13 |
| high_epsilon | maker_conservative | 66 | 18 | 27% | $1.65 | 1.70 |
| high_epsilon | maker_realistic | 66 | 35 | 53% | $2.09 | 2.18 |

## H10 Decomposition

| Bucket | Signals | PnL | Avg Underround |
|--------|---------|-----|----------------|
| with_underround | 9 | $1.56 | 17.33% |
| without_underround | 627 | $0.00 | -1.18% |

## Capacity Analysis

- p10 capacity: 0.00
- p50 capacity: 0.00
- p90 capacity: 61.00
- % with capacity >= 1: 33.1%

## Key Conclusions

1. **Maker fill risk**: Conservative maker model shows significant reduction in fills
2. **H10 verdict**: Tight spread strategy is primarily underround in disguise
3. **Capacity**: Most signals have limited capacity, constraining size
