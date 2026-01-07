# Strategy Card: CL Momentum Following

**Hypothesis ID**: H11_momentum_follow

**Category**: CL_PM_LEADLAG

**Generated**: 2026-01-06

---

## Strategy Definition

**Condition**: |CL_momentum_10s| > threshold

**Action**: Trade in direction of CL momentum

**Mechanism**: CL price momentum indicates direction of underlying asset movement. PM prices may lag CL, creating directional opportunity.

## Parameters

| Parameter | Suggested | Sweep Range |
|-----------|-----------|-------------|
| momentum_threshold | 0.0001 | [5e-05, 0.0001, 0.0002] |
| momentum_window | 10 | [5, 10, 20] |

## Backtest Results

No backtest results available.

## Validation Results

No validation results available.

## Wallet Evidence

- **n_momentum_trades**: 105197
- **up_avg_momentum**: 0.07047206057893089
- **down_avg_momentum**: 0.09251416154099978
- **follows_momentum**: False

## Failure Modes

- Momentum reversal
- PM already priced in CL move
- Momentum is noise
- Spread cost exceeds momentum magnitude

## Next Steps / Data Needed

- Collect more markets for larger sample size
- Add orderbook depth data for capacity modeling
- Test on other assets (BTC, SOL, XRP)
- Implement execution simulation with realistic slippage

