# Strategy Card: Passive Underround Harvesting

**Hypothesis ID**: H12_maker_underround

**Category**: PM_ONLY

**Generated**: 2026-01-06

---

## Strategy Definition

**Condition**: underround > 1% AND execution via limit orders

**Action**: Post limit orders to capture underround passively

**Mechanism**: Posting limit orders inside the underround allows earning the spread while still capturing the complete-set arbitrage. Lower execution risk than taker.

## Parameters

| Parameter | Suggested | Sweep Range |
|-----------|-----------|-------------|
| limit_offset | 0.005 | [0.002, 0.005, 0.01] |
| min_underround | 0.01 | [0.005, 0.01, 0.015] |

## Backtest Results

No backtest results available.

## Validation Results

No validation results available.

## Wallet Evidence

- **n_maker_underround_trades**: 608
- **maker_pct_overall**: 0.5216563844238171

## Failure Modes

- Orders may not fill
- Queue priority disadvantage
- Other makers crowd out
- Underround disappears before fill

## Next Steps / Data Needed

- Collect more markets for larger sample size
- Add orderbook depth data for capacity modeling
- Test on other assets (BTC, SOL, XRP)
- Implement execution simulation with realistic slippage

