# Focused Backtest Report

## Complete-Set Arb Family

### Top Configurations

| Epsilon | Tau Window | Cooldown | MinCap | Signals | PnL | t-stat | Win Rate |
|---------|------------|----------|--------|---------|-----|--------|----------|
| 0.005 | 600-900 | 30 | 0 | 51 | $0.72 | 6.53 | 100% |
| 0.01 | 600-900 | 30 | 0 | 51 | $0.72 | 6.53 | 100% |
| 0.005 | 600-900 | 60 | 0 | 46 | $0.63 | 6.48 | 100% |
| 0.01 | 600-900 | 60 | 0 | 46 | $0.63 | 6.48 | 100% |
| 0.005 | 300-900 | 30 | 0 | 109 | $2.00 | 4.00 | 100% |

## Late Directional Family

### Top Configurations

| MaxTau | Delta | Cooldown | Signals | W/L | PnL | t-stat | Dir Acc |
|--------|-------|----------|---------|-----|-----|--------|----------|
| 120 | 15 | 60 | 23 | 23/0 | $0.28 | 3.96 | 100% |
| 420 | 10 | 30 | 213 | 210/3 | $17.66 | 3.93 | 99% |
| 420 | 15 | 120 | 48 | 48/0 | $4.44 | 3.91 | 100% |
| 420 | 15 | 120 | 43 | 43/0 | $4.38 | 3.88 | 100% |
| 420 | 10 | 30 | 197 | 194/3 | $17.23 | 3.87 | 98% |

## Recommendations

- Complete-set arb is a valid arbitrage strategy with guaranteed payoff
- Complete-set arb passes statistical significance threshold
- Late directional 100% win rate suspicious - need more data
- Paper trade complete-set with epsilon=0.005
- Investigate why late directional has 100% win rate
