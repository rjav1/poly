# Strategy Card: Late Directional Taker

**Hypothesis ID**: H8_late_directional

**Category**: TIMING

**Generated**: 2026-01-06

---

## Strategy Definition

**Condition**: tau < 300s AND |delta_bps| > threshold

**Action**: Take directional position based on CL signal

**Mechanism**: In late window, CL price movement has high predictive value for final outcome. Taking directional position captures this information advantage.

## Parameters

| Parameter | Suggested | Sweep Range |
|-----------|-----------|-------------|
| max_tau | 300 | [120, 180, 300, 420] |
| delta_threshold_bps | 10 | [5, 10, 15, 20] |
| hold_seconds | 180 | [60, 120, 180, 240] |

## Backtest Results

**Total PnL**: $0.28

**Number of Trades**: 23

**Markets**: 47

**Win Rate**: 100.0%

**t-statistic**: 3.96

**Best Parameters**: {'max_tau': 120, 'delta_threshold_bps': 15, 'hold_seconds': 60}

## Validation Results

**Walk-Forward Validation**:
- Train t-stat: 3.78
- Test t-stat: 1.49
- Status: PASS

**Placebo (Time Shift 30s)**:
- Original t-stat: 3.96
- Shifted t-stat: 2.02
- Status: PASS

**Bootstrap Confidence Interval**:
- 95% CI: [0.0031, 0.0088]
- P(positive): 100.0%

## Wallet Evidence

- **n_late_directional_trades**: 4984
- **late_wallets**: []

## Failure Modes

- CL signal is noise not signal
- Spread cost exceeds expected edge
- Late liquidity insufficient
- Information already priced in

## Next Steps / Data Needed

- Collect more markets for larger sample size
- Add orderbook depth data for capacity modeling
- Test on other assets (BTC, SOL, XRP)
- Implement execution simulation with realistic slippage

