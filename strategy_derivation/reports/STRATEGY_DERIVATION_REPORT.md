# Strategy Derivation Report

**Date:** January 6, 2026  
**Author:** PhD Quantitative Researcher  
**Data Period:** December 23, 2025 - January 6, 2026

---

## Important Disclaimers

**This analysis has several limitations that may affect the validity of conclusions:**

1. **Small sample size:** Only 36 ETH markets were used for backtesting. This is insufficient for high-confidence conclusions.
2. **In-sample optimization:** Parameters were swept on the same data used for evaluation. No out-of-sample holdout was used.
3. **Data quality unknowns:** We cannot verify the accuracy of scraped wallet data or Chainlink/Polymarket price feeds.
4. **Survivorship bias:** The 6 wallets analyzed were selected because they appeared profitable - we don't know how many similar strategies failed.
5. **Strategy B placebo failure:** The best-performing strategy showed edge even with shifted CL data, which is a red flag for potential look-ahead bias.

**Treat all results as preliminary hypotheses requiring further validation, not actionable trading signals.**

---

## Executive Summary

This report documents the derivation and backtesting of three trading strategies for 15-minute up/down binary options on Polymarket. Strategies were reverse-engineered from the trading behavior of six wallets (assumed profitable), then implemented and tested on 36 ETH markets.

### Key Findings

| Strategy | Description | Best PnL | t-stat | Trades | Verdict |
|----------|-------------|----------|--------|--------|---------|
| Strategy A | Underround Harvester | $12.45 | 1.09 | 120 | **Not significant** (28% could be chance) |
| Strategy B | Late Directional Taker | $3.16 | 3.09 | 43 | **Suspicious** (placebo test failed) |
| Strategy C | Two-Sided Early Tilt Late | $11.09 | 1.02 | 128 | **Not significant** (31% could be chance) |

### Bottom Line

**None of these strategies should be traded live based on this analysis.**

- Strategies A and C have t-statistics below 1.5, meaning results could easily be random noise
- Strategy B has a high t-stat (3.09) but FAILED the placebo test - the edge persisted even when we shifted CL data by 30 seconds, which is a red flag for potential look-ahead bias or overfitting
- All results are from in-sample optimization on only 36 markets - likely to degrade out-of-sample

**This analysis provides hypotheses to investigate further, not actionable trading signals.**

---

## 1. Data Overview

### Wallet Activity Data

- **Total rows:** 536,639 across 6 traders
- **15m updown rows:** 480,596 (89.6%)
- **Within-window trades:** 474,764 (after filtering to 0 ≤ t < 900)

**Traders analyzed:**
| Handle | Trades | Primary Style |
|--------|--------|---------------|
| Account88888 | 190,274 | Two-sided inventory builder |
| PurpleThunderBicycleMountain | 152,946 | Two-sided with late tilt |
| Lkjhgfdna | 96,847 | Two-sided with active scalping |
| vidarx | 32,910 | Pure underround harvester |
| tsaiTop | 1,734 | Late directional taker |
| FLO782 | 37 | Insufficient data |

**Asset distribution:**
- BTC: 274,554 trades (57.8%)
- ETH: 132,454 trades (27.9%)
- SOL: 34,764 trades (7.3%)
- XRP: 32,992 trades (6.9%)

### Canonical Dataset (Backtest Data)

- **Markets:** 36 ETH markets with ≥90% coverage
- **Rows:** 32,400 (36 markets × 900 seconds)
- **Volume markets:** 12 new markets (16:30-19:15 on 2026-01-06)
- **Note:** Volume/size columns exist but are unpopulated

---

## 2. Methodology & Statistical Framework

### 2.1 How We Tested Strategies

**Step 1: Signal Generation**
- For each market (900 seconds of data), the strategy scans each second looking for entry conditions
- When conditions are met, a "signal" is generated with entry time, exit time, and direction (buy_up or buy_down)
- Signals have a cooldown period (30s) to avoid overlapping trades

**Step 2: Trade Execution Simulation**
- Entry price = best ask price at signal time (we assume taking liquidity)
- Exit price = best bid price at exit time (we assume taking liquidity on exit)
- PnL = exit_price - entry_price (for buy trades)
- Latency is modeled by shifting the execution time forward from signal time

**Step 3: Metric Computation**
- Trades are grouped by market
- Per-market PnL is computed (sum of all trade PnLs in that market)
- Statistics are computed across markets (not across trades) to avoid pseudo-replication

### 2.2 Why We Use t-statistics (Not Just PnL)

**The Problem with Raw PnL:**
- A strategy showing $12 profit could be:
  - (a) A real edge captured consistently, OR
  - (b) Random luck from a few big wins

**t-statistic Definition:**
```
t = mean_pnl_per_market / (std_pnl_per_market / sqrt(n_markets))
```

This measures: "How many standard errors is the mean away from zero?"

**Interpretation Guidelines:**
| t-stat | p-value (approx) | Interpretation |
|--------|------------------|----------------|
| < 1.0 | > 0.32 | No evidence of edge |
| 1.0 - 1.5 | 0.15 - 0.32 | Weak evidence, likely noise |
| 1.5 - 2.0 | 0.05 - 0.15 | Suggestive but not significant |
| 2.0 - 3.0 | 0.003 - 0.05 | Statistically significant |
| > 3.0 | < 0.003 | Strong evidence |

**Why Strategy A is "Weak Edge" (t=1.09):**
- Mean PnL/market = $0.35
- Std PnL/market = $1.90
- With 36 markets: SE = 1.90/√36 = $0.32
- t = 0.35 / 0.32 = 1.09
- This means: there's a ~28% chance we'd see this result even if the true edge is zero
- Not enough evidence to conclude the strategy works

**Why Strategy B is "Strong Edge" (t=3.09):**
- Despite lower absolute PnL ($3.16 vs $12.45), the consistency is much higher
- The probability of seeing t=3.09 by chance is < 0.3%
- **However:** The placebo test failure casts doubt on this result (see Section 4.3)

### 2.3 What the Placebo Test Does

**Purpose:** Detect if our edge comes from "peeking into the future"

**Method:**
1. Shift all Chainlink (CL) price data forward by 30 seconds
2. This means: at time t, we now see what CL price was at t-30
3. Re-run the strategy with this "stale" CL data
4. If edge disappears → strategy was using real-time CL information (good)
5. If edge persists → strategy might have look-ahead bias (bad)

**Why Strategy B's Placebo Failure is Concerning:**
- Original t-stat: 3.09
- Placebo t-stat: 2.87 (still high!)
- The edge should have disappeared if it came from CL lead-lag
- Possible explanations:
  1. Strategy actually captures PM orderbook patterns, not CL-PM lead-lag
  2. There's a bug in how we shift the CL data
  3. The delta_bps signal is autocorrelated (past delta predicts future delta)
  4. We have unintentional look-ahead bias somewhere in the pipeline

### 2.4 Known Limitations of Our Testing

**Execution Assumptions (Optimistic):**
- We assume we can always fill at the best bid/ask
- We don't model queue position or partial fills
- We don't model market impact (our trades moving prices)
- We assume 1 unit position size regardless of available liquidity

**Data Quality Issues:**
- CL prices come from scraping, may have gaps or errors
- PM orderbook is snapshot-based (1-second), not tick-by-tick
- We don't have actual trade timestamps from the wallets we analyzed (only "within this second")

**Statistical Issues:**
- 36 markets is a small sample for robust inference
- Parameter sweep was done in-sample (no train/test split)
- We tested many parameter combinations, increasing false positive risk
- Markets are from a short time window (Dec 23 - Jan 6), may not generalize

---

## 3. Hypothesis Testing Results

**Note:** These tests analyze the behavior of wallets we *assumed* were profitable. We cannot verify their actual PnL from the activity export (REDEEM rows show size=0). The "profitable" label comes from external sources.

### H1: Late-Window Concentration

**Question:** Do profitable traders concentrate activity near expiry?

| Handle | Last 60s | Last 120s | Last 300s | Median τ |
|--------|----------|-----------|-----------|----------|
| tsaiTop | **16.9%** | **39.2%** | **63.2%** | 159s |
| Purple | 5.1% | 11.4% | 41.7% | 361s |
| vidarx | 0.0% | 0.9% | 24.5% | 515s |
| Account88888 | 2.1% | 6.2% | 26.0% | 535s |
| Lkjhgfdna | 1.1% | 4.0% | 19.1% | 563s |

**Insight:** tsaiTop is extremely late-focused (63% of trades in last 5 minutes), while others are more evenly distributed.

### H2: Two-Sided vs One-Sided Behavior

**Question:** Do traders build balanced positions or take directional bets?

| Handle | % Both Sides | % UP Only | % DOWN Only |
|--------|--------------|-----------|-------------|
| Account88888 | **99.6%** | 0.4% | 0.0% |
| vidarx | **98.9%** | 1.1% | 0.0% |
| Lkjhgfdna | **98.0%** | 1.1% | 0.9% |
| Purple | **95.8%** | 1.6% | 2.6% |
| tsaiTop | 3.8% | **48.1%** | **48.1%** |

**Insight:** Most profitable traders trade both sides (straddle-style), except tsaiTop who is almost purely directional.

### H3: Underround Harvesting

**Question:** Do traders capture underround (sum of asks < 1)?

| Handle | Paired Seconds | Median Edge | % Positive |
|--------|----------------|-------------|------------|
| vidarx | 951 | **$0.030** | **83.0%** |
| Purple | 1,037 | **$0.025** | 64.8% |
| Account88888 | 6,091 | $0.000 | 49.9% |
| Lkjhgfdna | 463 | -$0.031 | 13.4% |

**Insight:** vidarx and Purple successfully harvest underround with consistent positive edge. Account88888 breaks even, Lkjhgfdna loses on this strategy.

### H4: Short-Horizon Scalping

**Question:** Do traders scalp with short holding periods?

| Handle | Matched Lots | Median Hold | PnL |
|--------|--------------|-------------|-----|
| tsaiTop | 37,386 | **190s** | **+$1,640** |
| Lkjhgfdna | 1,914,277 | 114s | -$58,021 |
| Others | 0 | N/A | N/A |

**Insight:** Only tsaiTop and Lkjhgfdna actively scalp. tsaiTop is profitable with ~3 minute holds; Lkjhgfdna loses significantly.

### H5: Conversion Mechanics

Most traders use pure TRADE actions with minimal SPLIT/MERGE/CONVERSION usage within the 15-minute windows.

---

## 4. Extracted Strategy Parameters

### Strategy A: Underround Harvester (vidarx-inspired)

```
Trigger: sum_asks < 1 - epsilon
Action: Buy both sides (complete set)
Exit: Hold to expiry
```

| Parameter | Suggested | Sweep Range |
|-----------|-----------|-------------|
| epsilon | 0.025 | [0.01, 0.02, 0.03, 0.04] |
| min_tau | 60s | [30, 60, 120] |
| max_tau | 840s | [720, 840] |

### Strategy B: Late Directional Taker (tsaiTop-inspired)

```
Trigger: tau < tau_max AND |delta| > threshold
Action: Buy high-probability side based on CL momentum
Exit: Hold for N seconds or to expiry
```

| Parameter | Suggested | Sweep Range |
|-----------|-----------|-------------|
| tau_max | 300s | [120, 180, 300, 420] |
| delta_threshold_bps | 10 | [5, 10, 15, 20] |
| hold_seconds | 190s | [120, 180, 240] |
| momentum_window | 10s | [5, 10, 20, 30] |

### Strategy C: Two-Sided Early Tilt Late (Purple/Account-inspired)

```
Phase 1 (tau > 300s): Build matched inventory when underround
Phase 2 (tau < 180s): Add net directional exposure based on CL
Exit: Hold to expiry
```

| Parameter | Suggested | Sweep Range |
|-----------|-----------|-------------|
| inventory_phase_end | 300s | [180, 300, 420] |
| tilt_phase_start | 180s | [120, 180] |
| inventory_epsilon | 0.02 | [0.01, 0.02, 0.03] |
| tilt_delta_threshold_bps | 15 | [10, 15, 20] |

---

## 5. Backtest Results

### 5.1 Parameter Sweep Results

**Warning: In-Sample Results**

These results are from sweeping parameters on the SAME data we use for evaluation. This is a major methodological weakness:
- We tested 24 parameter combinations for Strategy A
- We tested 48 parameter combinations for Strategy B  
- We tested 54 parameter combinations for Strategy C
- With so many tests, we expect some to look good by chance alone

**Multiple Testing Problem:**
- If we test 100 random strategies, ~5 will show t>2.0 by pure chance (5% false positive rate)
- We should apply a Bonferroni correction or use out-of-sample testing
- The "best" parameters reported below are cherry-picked from in-sample results

### Parameter Sweep (36 ETH Markets)

#### Strategy A: Underround Harvester

| epsilon | min_tau | max_tau | Trades | PnL | t-stat | Hit Rate |
|---------|---------|---------|--------|-----|--------|----------|
| 0.01 | 30 | 720 | 120 | **$12.45** | 1.09 | 62.5% |
| 0.01 | 60 | 720 | 119 | $12.05 | 1.07 | 62.2% |
| 0.04 | 30 | 840 | 21 | $5.38 | 1.15 | 76.2% |

**Best:** epsilon=0.01, min_tau=30, max_tau=720 ($12.45, t=1.09)

#### Strategy B: Late Directional Taker

| tau_max | delta_bps | hold_s | Trades | PnL | t-stat | Hit Rate |
|---------|-----------|--------|--------|-----|--------|----------|
| 420 | 10 | 120 | 43 | **$3.16** | **3.09** | 62.8% |
| 420 | 5 | 120 | 67 | $2.08 | 1.43 | 53.7% |
| 300 | 10 | 120 | 35 | $2.32 | 2.64 | 60.0% |

**Best:** tau_max=420, delta=10bps, hold=120s ($3.16, t=3.09)

#### Strategy C: Two-Sided Tilt

| inv_end | tilt_start | inv_eps | tilt_delta | Trades | PnL | t-stat |
|---------|------------|---------|------------|--------|-----|--------|
| 180 | 180 | 0.01 | 20 | 128 | **$11.09** | 1.02 |
| 180 | 120 | 0.01 | 20 | 147 | $8.65 | 0.82 |

**Best:** inv_phase_end=180, tilt_start=180, epsilon=0.01, delta=20bps ($11.09, t=1.02)

### Latency Sensitivity

| Latency | Strategy A PnL | Strategy B PnL | Strategy C PnL |
|---------|----------------|----------------|----------------|
| 0.0s | $12.45 (t=1.09) | $3.16 (t=3.09) | $11.09 (t=1.02) |
| 0.5s | $12.45 (t=1.09) | $3.16 (t=3.09) | $11.09 (t=1.02) |
| 1.0s | $11.05 (t=0.97) | $3.10 (t=2.98) | $6.50 (t=0.60) |
| 2.0s | $11.05 (t=0.97) | $3.03 (t=2.86) | $5.75 (t=0.54) |
| 5.0s | $9.45 (t=0.81) | $3.15 (t=2.94) | $5.95 (t=0.55) |

**Insight:** Strategy B is remarkably robust to latency, maintaining t-stat >2.8 even at 5s latency. Strategies A and C degrade more significantly.

### Placebo Tests (CL shifted +30s)

| Strategy | Original t-stat | Placebo t-stat | Status |
|----------|-----------------|----------------|--------|
| Strategy A | 1.09 | 1.28 | PASS |
| Strategy B | 3.09 | 2.87 | **FAIL** (edge persists) |
| Strategy C | 1.02 | 1.32 | PASS |

**Interpretation:** 
- Strategies A and C show no edge based on future CL information (good - edge is real-time)
- Strategy B's edge persists even with shifted CL data, which is concerning - suggests either:
  1. Edge comes from PM orderbook dynamics, not CL lead-lag
  2. Possible data leakage or look-ahead bias
  3. Edge is structural and persists across CL windows

### Volume Markets Subset (12 new markets)

| Strategy | Markets | Trades | PnL | t-stat |
|----------|---------|--------|-----|--------|
| Strategy A | 12 | 49 | $1.82 | 0.23 |
| Strategy B | 12 | 0 | $0.00 | 0.00 |
| Strategy C | 12 | 45 | -$0.13 | -0.02 |

**Note:** Strategy B generated no trades on the 12 new markets, suggesting the required conditions (late window + sufficient delta) were not met in this subset.

---

## 6. Detailed Limitations & Potential Biases

### 6.1 Why "Weak Edge" for Strategies A and C

**Statistical Significance Thresholds:**

A t-statistic measures how many standard errors the mean is from zero:
- t < 1.5: Could easily be random chance (~15%+ probability)
- t = 2.0: Standard "significant" threshold (~5% false positive rate)
- t = 3.0: Strong evidence (~0.3% false positive rate)

**Strategy A (t=1.09):**
- Mean PnL per market: $0.35
- Standard deviation: $1.90
- With 36 markets, standard error = $1.90/√36 = $0.32
- t = $0.35 / $0.32 = 1.09
- **Interpretation:** There's a ~28% chance of seeing this result if the true edge is ZERO
- The positive PnL could easily be luck

**Strategy C (t=1.02):**
- Despite higher absolute PnL ($11.09), the variance across markets is high
- **Interpretation:** ~31% chance this is random noise

### 6.2 Why Strategy B's Results Are Questionable

Despite the impressive t-stat of 3.09, several red flags exist:

**Red Flag 1: Placebo Test Failure**
- When we shifted CL data by 30 seconds, edge should have disappeared
- Instead: t went from 3.09 → 2.87 (barely changed)
- This suggests the strategy may not actually be using CL lead-lag

**Red Flag 2: Extreme Latency Robustness**
- Edge persists even at 5 seconds latency
- Lead-lag edges typically decay within 1-2 seconds
- Either we found something different, or there's a bug

**Red Flag 3: Zero Trades on Volume Subset**
- Strategy generated 0 trades on 12 newer markets
- Could indicate overfitting to the original 24 markets

### 6.3 In-Sample Optimization Problem

**What We Did (Wrong):**
1. Tested 24 + 48 + 54 = 126 parameter combinations
2. Selected the "best" from each strategy
3. Reported those results as the strategy performance

**Why This Is Problematic:**
- With 126 tests at 5% significance, we expect ~6 false positives by chance
- The "best" parameters are cherry-picked from noisy data
- Performance will likely degrade out-of-sample

**What We Should Have Done:**
1. Split markets into train (70%) and test (30%) sets
2. Find best parameters on train set only
3. Report performance on test set
4. Use walk-forward validation for robustness

### 6.4 Data Quality Concerns

**Wallet Data Issues:**
- We cannot verify wallets are actually profitable (REDEEM rows have size=0)
- Wallet selection was based on external claims, not verified PnL
- Trade timestamps are second-resolution (not millisecond)

**Market Data Issues:**
- CL prices from scraping may have gaps or errors
- PM orderbook is snapshot-based, not tick-by-tick
- We don't know actual execution prices traders received

**Sample Size Issues:**
- Only 36 markets is too small for robust conclusions
- All markets are ETH from a 2-week period
- May not generalize to other assets or time periods

### 6.5 Execution Model Limitations

**What We Assume:**
- Always fill at best bid/ask
- No queue position or partial fills
- No market impact
- Fixed 1-unit position size

**Reality:**
- Large orders move prices
- May not get filled at desired price
- Queue position matters for passive orders
- Size constraints from available liquidity

---

## 7. Conclusions & Recommendations

### Honest Assessment

**None of the strategies are ready for live trading based on this analysis.**

| Strategy | Verdict | Key Issue |
|----------|---------|-----------|
| Strategy A | Not Recommended | t=1.09 is not statistically significant |
| Strategy B | Needs Investigation | Placebo test failure is a red flag |
| Strategy C | Not Recommended | t=1.02 is not statistically significant |

### What We Actually Learned

1. **The profitable wallets DO have distinctive behaviors:**
   - tsaiTop trades late and directionally (63% in last 5 min, 96% one-sided)
   - vidarx/Purple harvest underround with positive edge in their actual trades
   - But translating observed behavior into backtestable strategies didn't work well

2. **Our backtest framework has limitations:**
   - In-sample optimization inflates results
   - 36 markets is insufficient sample size
   - Placebo test revealed potential issues with Strategy B

3. **The lead-lag hypothesis needs more validation:**
   - Strategy B's placebo failure suggests edge may not come from CL-PM lead-lag
   - Or there's a bug in our testing methodology

### Required Next Steps (Before Any Live Trading)

1. **Fix the Placebo Test Issue:**
   - Debug why Strategy B edge persists with shifted CL data
   - Test multiple shift windows (15s, 30s, 60s, 120s)
   - Verify delta_bps calculation doesn't have look-ahead bias

2. **Implement Proper Train/Test Split:**
   - Split markets 70/30 chronologically
   - Optimize parameters on train set only
   - Report test set performance as final results

3. **Increase Sample Size:**
   - Collect 100+ markets for meaningful statistics
   - Test on multiple assets (BTC, SOL, XRP)
   - Test across different time periods

4. **Paper Trade Before Live:**
   - Run strategies in paper trading mode
   - Compare predicted vs actual fills
   - Measure real latency and slippage

---

## Appendix: Output Files

| File | Description |
|------|-------------|
| `wallet_data_normalized.parquet` | Enriched wallet trades with t, tau |
| `hypothesis_results.json` | H1-H5 test results per wallet |
| `strategy_params.json` | Extracted parameter priors |
| `parameter_sweep_results.csv` | All parameter combinations |
| `latency_sensitivity_results.csv` | Latency sweep results |
| `placebo_test_results.csv` | Placebo test results |
| `volume_subset_results.csv` | 12 volume markets subset |

---

*Report generated by strategy derivation pipeline v1.0*

