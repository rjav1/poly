# Strategy Card: Early Inventory Build

**Hypothesis ID**: H9_early_inventory

**Category**: INVENTORY

**Generated**: 2026-01-06

---

## Strategy Definition

**Condition**: tau > 600s AND underround exists

**Action**: Build matched inventory early, hold to expiry

**Mechanism**: Building inventory early when spreads are tighter allows accumulating position at better prices. Hold to expiry for guaranteed payoff.

## Parameters

| Parameter | Suggested | Sweep Range |
|-----------|-----------|-------------|
| min_tau | 600 | [500, 600, 700] |
| epsilon | 0.015 | [0.01, 0.015, 0.02] |

## Backtest Results

**Total PnL**: $0.75

**Number of Trades**: 58

**Markets**: 47

**Win Rate**: 100.0%

**t-statistic**: 7.94

**Best Parameters**: {'min_tau': 500, 'epsilon': 0.01}

## Validation Results

**Walk-Forward Validation**:
- Train t-stat: 6.56
- Test t-stat: 4.52
- Status: PASS

**Placebo (Time Shift 30s)**:
- Original t-stat: 7.94
- Shifted t-stat: 7.94
- Status: PASS

**Bootstrap Confidence Interval**:
- 95% CI: [0.0120, 0.0198]
- P(positive): 100.0%

## Wallet Evidence

- **n_early_both_trades**: 27544
- **early_underround_mean**: 0.00010632837134267077

## Failure Modes

- Early prices may not be better
- Capital tied up longer
- Missing late opportunities

## Next Steps / Data Needed

- Collect more markets for larger sample size
- Add orderbook depth data for capacity modeling
- Test on other assets (BTC, SOL, XRP)
- Implement execution simulation with realistic slippage

