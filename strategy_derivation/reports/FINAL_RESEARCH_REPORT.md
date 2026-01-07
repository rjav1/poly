# Strategy Discovery Pipeline - Final Research Report

**Generated**: 2026-01-06 20:35:14

**Pipeline Version**: Advanced Strategy Discovery v1.0

---

## Executive Summary

This report presents the results of a systematic strategy discovery pipeline that analyzed profitable trader wallet data to extract testable trading hypotheses for Polymarket 15-minute Up/Down markets.

### Top Performing Strategies

| Strategy | t-stat | Total PnL | Trades | Win Rate | Validation |
|----------|--------|-----------|--------|----------|------------|
| H9_early_inventory | 7.94 | $0.75 | 58 | 100% | PASS |
| H8_late_directional | 3.96 | $0.28 | 23 | 100% | PASS |
| H10_tight_spread | 1.92 | $3.73 | 125 | 100% | PARTIAL |
| H6_underround_harvest | 1.72 | $5.87 | 154 | 100% | PASS |
| H7_late_underround | 1.25 | $5.00 | 46 | 100% | PARTIAL |

### Key Findings

1. **Underround Harvesting (H6, H9)**: PM-only strategy that captures complete-set arbitrage when sum_asks < 1. Shows consistent positive edge.

2. **Late Directional (H8)**: CL-based strategy that takes directional positions in late window based on delta from strike. Higher t-stat but requires careful validation.

3. **Execution Style**: Most profitable traders show maker-bias execution, suggesting passive order placement is important.

---

## Pipeline Overview

The strategy discovery pipeline consisted of 9 phases:

1. **Research Table Construction**: Joined wallet trades to market state
2. **Position Reconstruction**: Built inventory time series per wallet/market
3. **Execution Style Inference**: Classified trades as maker/taker
4. **Feature Engineering**: Created 95 features for modeling
5. **Policy Inversion**: Extracted rules predicting trader actions
6. **Hypothesis Generation**: Formulated 7 testable hypotheses
7. **Strategy Implementation**: Implemented and backtested strategies
8. **Validation Suite**: Ran placebo, walk-forward, bootstrap tests
9. **Report Generation**: This report

---

## Data Summary

### Wallet Data

| Wallet | Markets | Hold-to-Expiry | Both Sides | Scalping |
|--------|---------|----------------|------------|----------|
| vidarx | 43 | 100% | 100% | 0.00 |
| tsaiTop | 8 | 100% | 0% | 14.38 |
| Account88888 | 41 | 100% | 100% | 0.00 |
| PurpleThunderBicycleMountain | 29 | 100% | 100% | 0.00 |
| Lkjhgfdna | 43 | 100% | 100% | 55.37 |
| FLO782 | 3 | 100% | 0% | 0.00 |

### Execution Style Summary

| Wallet | Primary Style | Taker % | Maker % | Avg Aggr |
|--------|---------------|---------|---------|----------|
| vidarx | MIXED_MAKER_BIAS | 4.5% | 49.4% | 0.50 |
| tsaiTop | MIXED_MAKER_BIAS | 4.8% | 56.4% | 0.42 |
| Account88888 | MIXED_MAKER_BIAS | 6.5% | 51.1% | 0.48 |
| PurpleThunderBicycleMountain | MIXED_MAKER_BIAS | 6.7% | 58.7% | 0.39 |
| Lkjhgfdna | MIXED_MAKER_BIAS | 10.5% | 52.4% | 0.45 |
| FLO782 | PASSIVE_MAKER | 0.0% | 75.7% | 0.24 |

---

## Hypothesis Details

### H6_underround_harvest: Underround Harvesting

**Category**: PM_ONLY

**Condition**: sum_asks < 1 - epsilon (underround > 1%)

**Action**: Buy both UP and DOWN tokens simultaneously (complete set)

**Mechanism**: When sum_asks < 1, buying both sides for less than $1 guarantees $1 at expiry regardless of outcome. The edge equals the underround magnitude.

**Backtest**: t-stat=1.72, PnL=$5.87, Trades=154

---

### H7_late_underround: Late Window Underround

**Category**: PM_ONLY

**Condition**: underround > 1% AND tau < 120s

**Action**: Buy complete set in last 2 minutes

**Mechanism**: Late-window underround opportunities may be more reliable as market-makers widen spreads near expiry, creating exploitable inefficiencies.

**Backtest**: t-stat=1.25, PnL=$5.00, Trades=46

---

### H12_maker_underround: Passive Underround Harvesting

**Category**: PM_ONLY

**Condition**: underround > 1% AND execution via limit orders

**Action**: Post limit orders to capture underround passively

**Mechanism**: Posting limit orders inside the underround allows earning the spread while still capturing the complete-set arbitrage. Lower execution risk than taker.

---

### H10_tight_spread: Tight Spread Entry

**Category**: PM_ONLY

**Condition**: spread < median_spread * 0.8

**Action**: Enter position when spreads are unusually tight

**Mechanism**: Tight spreads indicate either high liquidity or stale quotes. If it's liquidity, execution is cheaper. If it's staleness, there may be information advantage.

**Backtest**: t-stat=1.92, PnL=$3.73, Trades=125

---

### H9_early_inventory: Early Inventory Build

**Category**: INVENTORY

**Condition**: tau > 600s AND underround exists

**Action**: Build matched inventory early, hold to expiry

**Mechanism**: Building inventory early when spreads are tighter allows accumulating position at better prices. Hold to expiry for guaranteed payoff.

**Backtest**: t-stat=7.94, PnL=$0.75, Trades=58

---

### H8_late_directional: Late Directional Taker

**Category**: TIMING

**Condition**: tau < 300s AND |delta_bps| > threshold

**Action**: Take directional position based on CL signal

**Mechanism**: In late window, CL price movement has high predictive value for final outcome. Taking directional position captures this information advantage.

**Backtest**: t-stat=3.96, PnL=$0.28, Trades=23

---

### H11_momentum_follow: CL Momentum Following

**Category**: CL_PM_LEADLAG

**Condition**: |CL_momentum_10s| > threshold

**Action**: Trade in direction of CL momentum

**Mechanism**: CL price momentum indicates direction of underlying asset movement. PM prices may lag CL, creating directional opportunity.

---

## Validation Summary

| Strategy | Walk-Forward | Time Shift | Permutation | Bootstrap P(pos) |
|----------|--------------|------------|-------------|------------------|
| H6_underround_harvest | PASS | PASS | PASS | 100% |
| H9_early_inventory | PASS | PASS | PASS | 100% |
| H8_late_directional | PASS | PASS | PASS | 100% |

---

## Recommendations

### Strategies Ready for Paper Trading

1. **H9_early_inventory**: Highest t-stat (7.94), PM-only, passed walk-forward
2. **H6_underround_harvest**: Pure arbitrage mechanism, 100% P(positive)

### Strategies Requiring Further Validation

1. **H8_late_directional**: High t-stat but failed placebo tests - may have look-ahead bias or be capturing spurious patterns

### Next Steps

1. **Increase sample size**: Collect 100+ markets for more robust inference
2. **Multi-asset testing**: Validate strategies on BTC, SOL, XRP
3. **Execution modeling**: Add realistic slippage and capacity constraints
4. **Paper trading**: Run strategies in simulation before live deployment
5. **Cross-wallet validation**: Verify patterns appear in additional profitable wallets

---

## Disclaimers

**This analysis has significant limitations:**

1. Small sample size (47 markets) limits statistical confidence
2. In-sample parameter optimization may overfit
3. Execution assumptions are optimistic (no slippage, immediate fills)
4. Wallet profitability is assumed, not verified
5. Market regime changes may invalidate patterns

**Treat all results as hypotheses requiring further validation, not actionable trading signals.**

