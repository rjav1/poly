# Strategy Card: Tight Spread Entry

**Hypothesis ID**: H10_tight_spread

**Category**: PM_ONLY

**Generated**: 2026-01-06

---

## Strategy Definition

**Condition**: spread < median_spread * 0.8

**Action**: Enter position when spreads are unusually tight

**Mechanism**: Tight spreads indicate either high liquidity or stale quotes. If it's liquidity, execution is cheaper. If it's staleness, there may be information advantage.

## Parameters

| Parameter | Suggested | Sweep Range |
|-----------|-----------|-------------|
| spread_percentile | 20 | [10, 20, 30] |
| min_spread_bps | 5 | [0, 5, 10] |

## Backtest Results

**Total PnL**: $3.73

**Number of Trades**: 125

**Markets**: 47

**Win Rate**: 100.0%

**t-statistic**: 1.92

**Best Parameters**: {'spread_percentile': 10, 'min_spread_bps': 0}

## Validation Results

No validation results available.

## Wallet Evidence

- **n_tight_spread_trades**: 1403
- **median_spread**: 0.010000000000000009
- **tight_threshold**: 0.008000000000000007

## Failure Modes

- Tight spreads may widen immediately
- May indicate low volatility periods with no edge
- Staleness means quotes may not be executable

## Next Steps / Data Needed

- Collect more markets for larger sample size
- Add orderbook depth data for capacity modeling
- Test on other assets (BTC, SOL, XRP)
- Implement execution simulation with realistic slippage

