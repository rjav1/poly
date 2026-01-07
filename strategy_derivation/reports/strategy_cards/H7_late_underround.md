# Strategy Card: Late Window Underround

**Hypothesis ID**: H7_late_underround

**Category**: PM_ONLY

**Generated**: 2026-01-06

---

## Strategy Definition

**Condition**: underround > 1% AND tau < 120s

**Action**: Buy complete set in last 2 minutes

**Mechanism**: Late-window underround opportunities may be more reliable as market-makers widen spreads near expiry, creating exploitable inefficiencies.

## Parameters

| Parameter | Suggested | Sweep Range |
|-----------|-----------|-------------|
| epsilon | 0.015 | [0.01, 0.015, 0.02] |
| max_tau | 120 | [60, 120, 180] |

## Backtest Results

**Total PnL**: $5.00

**Number of Trades**: 46

**Markets**: 47

**Win Rate**: 100.0%

**t-statistic**: 1.25

**Best Parameters**: {'epsilon': 0.01, 'max_tau': 180}

## Validation Results

No validation results available.

## Wallet Evidence

- **n_late_underround_trades**: 126
- **avg_underround**: 0.011825396825396836

## Failure Modes

- Time pressure increases execution risk
- Late spreads may be wider
- Lower liquidity near expiry

## Next Steps / Data Needed

- Collect more markets for larger sample size
- Add orderbook depth data for capacity modeling
- Test on other assets (BTC, SOL, XRP)
- Implement execution simulation with realistic slippage

