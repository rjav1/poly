# Strategy Card: Underround Harvesting

**Hypothesis ID**: H6_underround_harvest

**Category**: PM_ONLY

**Generated**: 2026-01-06

---

## Strategy Definition

**Condition**: sum_asks < 1 - epsilon (underround > 1%)

**Action**: Buy both UP and DOWN tokens simultaneously (complete set)

**Mechanism**: When sum_asks < 1, buying both sides for less than $1 guarantees $1 at expiry regardless of outcome. The edge equals the underround magnitude.

## Parameters

| Parameter | Suggested | Sweep Range |
|-----------|-----------|-------------|
| epsilon | 0.01 | [0.005, 0.01, 0.02, 0.03] |
| min_tau | 60 | [30, 60, 120] |
| max_tau | 840 | [720, 840] |
| min_capacity | 1 | [0, 1, 5, 10] |

## Backtest Results

**Total PnL**: $5.87

**Number of Trades**: 154

**Markets**: 47

**Win Rate**: 100.0%

**t-statistic**: 1.72

**Best Parameters**: {'epsilon': 0.005, 'min_tau': 120, 'max_tau': 840, 'min_capacity': 0}

## Validation Results

**Walk-Forward Validation**:
- Train t-stat: 5.02
- Test t-stat: 1.28
- Status: PASS

**Placebo (Time Shift 30s)**:
- Original t-stat: 1.72
- Shifted t-stat: 1.72
- Status: PASS

**Bootstrap Confidence Interval**:
- 95% CI: [0.0411, 0.2901]
- P(positive): 100.0%

## Wallet Evidence

- **n_trades_with_underround**: 1268
- **avg_underround_when_trading**: 0.01409305993690853
- **pct_both_sides**: 0.5331230283911672
- **wallets**: ['Account88888', 'PurpleThunderBicycleMountain', 'vidarx', 'Lkjhgfdna']

## Failure Modes

- Insufficient capacity (size limits)
- Execution slippage erodes edge
- Quote staleness leads to stale underround
- Competition from other arb traders

## Next Steps / Data Needed

- Collect more markets for larger sample size
- Add orderbook depth data for capacity modeling
- Test on other assets (BTC, SOL, XRP)
- Implement execution simulation with realistic slippage

