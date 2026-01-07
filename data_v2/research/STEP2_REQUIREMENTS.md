# Step 2: Latency-Aware Taker Backtest - Requirements Document

**Created**: 2026-01-06
**Purpose**: Document all requirements, assumptions, and methodology for the ETH lead-lag backtest

---

## 1. Project Context

### Goal
Determine if a tradeable lead-lag edge exists between Chainlink and Polymarket on ETH markets:
- "If I trade as a taker on PM based on Chainlink moves, how much edge exists after spread/latency, and what parameter region is robust?"

### Available Data
- **24 ETH markets** with >90% combined coverage
- **16 UP / 16 DOWN** outcome split (perfectly balanced)
- **~900 seconds** per market (15 minutes)
- Key columns: `t`, `tau`, `cl_mid`, `pm_up_best_bid`, `pm_up_best_ask`, `pm_down_best_bid`, `pm_down_best_ask`, `K`, `delta_bps`, `cl_ffill`, `pm_ffill`

### Key Assumptions
- **No trading fees** (Polymarket has no fees)
- **Taker strategy only** (maker analysis deferred)
- **Top-of-book only** (depth needed later for capacity)
- **Focus on ETH** (other assets later)

---

## 2. Execution Model (Critical)

### 2.1 Split/Redeem (Conversion) Routing

**Why this matters**: Successful traders use complete-set conversion to get better execution. If we don't model this, we overestimate costs and may miss the main edge.

**Polymarket mechanics**:
- Complete set = 1 UP token + 1 DOWN token = $1.00
- You can "split" $1 into 1 UP + 1 DOWN
- You can "redeem" 1 UP + 1 DOWN into $1

**Optimal execution prices**:

```
# Buy UP (want to acquire UP tokens)
effective_buy_up = min(
    pm_up_best_ask,           # Direct: buy UP from orderbook
    1 - pm_down_best_bid      # Conversion: sell DOWN, keep UP from split
)

# Sell UP (want to dispose UP tokens)
effective_sell_up = max(
    pm_up_best_bid,           # Direct: sell UP to orderbook
    1 - pm_down_best_ask      # Conversion: buy DOWN, redeem pair for $1
)

# Buy DOWN
effective_buy_down = min(
    pm_down_best_ask,
    1 - pm_up_best_bid
)

# Sell DOWN
effective_sell_down = max(
    pm_down_best_bid,
    1 - pm_up_best_ask
)
```

**Implementation requirement**: All trade simulations MUST use these effective prices, not raw bid/ask.

### 2.2 Latency Model

**Two distinct latency components**:

1. **Signal latency** (`signal_latency_s`): Time from CL price update to when we observe it
   - Depends on data source (API vs UI scraping vs direct node)
   - Our current data uses UI timestamps (~65s behind real-time for CL)

2. **Execution latency** (`exec_latency_s`): Time from decision to filled trade
   - Network latency + exchange processing
   - Typically 0.1-2 seconds for API trading

**Critical distinction**:
- **Event time**: When CL actually updated (the `timestamp` in our data)
- **Observable time**: When we could realistically see it

**Two backtest modes** (MUST run both):

| Mode | Assumption | Implication |
|------|------------|-------------|
| A: Optimistic | We observe CL at event-time (near-zero signal latency) | Best-case edge |
| B: Realistic | We observe CL with delay (e.g., 1-5s signal latency) | Tradeable edge |

**Requirement**: If edge exists only under Mode A but we can't achieve Mode A in practice, the edge is NOT real.

---

## 3. Event Study Design

### 3.1 Event Types to Detect

**Type 1: CL Jump Events**
- Condition: `|Δcl_mid| >= threshold_bps` in one second
- Parameters to sweep: threshold_bps ∈ [5, 10, 15, 20, 30, 50]
- Output: t0, direction ('up'/'down'), magnitude_bps

**Type 2: Strike-Crossing Events** (NEW - economically critical)
- Condition: `sign(cl_mid[t] - K) != sign(cl_mid[t-1] - K)`
- This is when probability should snap from ~50% to ~0% or ~100%
- Most important near expiry (low tau)

**Type 3: Near-Strike Regime**
- Condition: `|delta_bps| <= d` where d ∈ [5, 10, 20] bps
- PM probability is most sensitive in this regime
- Combined with tau buckets for regime analysis

### 3.2 Response Measurement

For each event at t0, compute PM response over lag ∈ [0, 30] seconds:

```python
pm_up_mid_change[lag] = pm_up_mid[t0 + lag] - pm_up_mid[t0 - 1]
pm_down_mid_change[lag] = pm_down_mid[t0 + lag] - pm_down_mid[t0 - 1]
```

### 3.3 Regime Splits

Analyze responses separately by:

1. **Time-to-expiry (tau)**:
   - Early: tau ∈ [600, 900] (10-15 min left)
   - Mid: tau ∈ [300, 600] (5-10 min left)
   - Late: tau ∈ [0, 300] (0-5 min left)

2. **Distance to strike (delta_bps)**:
   - Near-strike: |delta_bps| <= 20
   - Mid-range: |delta_bps| ∈ [20, 100]
   - Far: |delta_bps| > 100

3. **Event magnitude**:
   - Small: 5-10 bps
   - Medium: 10-20 bps
   - Large: >20 bps

### 3.4 Key Visualizations

1. **Average response curve**: E[ΔPM | CL event] vs lag seconds
2. **Response heatmap**: magnitude by (tau bucket, event size)
3. **Histogram of lag-to-first-response**: When does PM start moving?
4. **Strike-cross response**: PM probability snap after CL crosses K

---

## 4. Fair Value Baseline Model

### 4.1 Purpose

A pure "trade after CL jump" strategy can profit for two different reasons:
1. **Latency capture**: PM lags CL (the hypothesis we're testing)
2. **Directional momentum**: Price has short-term momentum (confounding factor)

To separate these, we need a baseline "fair value" model.

### 4.2 Simple Fair Value Model

```python
# Probability that final price > K
p_hat = f(delta_bps, tau, recent_vol)
```

**Simplest implementation** (empirical logistic):
```python
from sklearn.linear_model import LogisticRegression

# Features: delta_bps, tau, delta_bps * tau, recent_cl_vol
# Target: Y (outcome)
# Fit on training markets, predict on test markets
```

**Alternative**: Use Black-Scholes-style formula with implied vol fitted to data.

### 4.3 Mispricing Signal

```python
mispricing = pm_up_mid - p_hat

# Only trade when mispricing exceeds execution cost
trade_signal = (mispricing > cost_threshold) or (mispricing < -cost_threshold)
```

### 4.4 Documentation Requirement

The fair value model MUST be:
- Documented with formula/code
- Fitted on training data only
- Evaluated on holdout data
- Easy to modify/replace in future

---

## 5. Strategy Framework

### 5.1 Strategy A: Latency Capture After CL Jump

**Entry rule**:
1. At time t, CL moved by >= `threshold_bps` in direction `dir`
2. Signal observable at t + `signal_latency`
3. Trade executed at t + `signal_latency` + `exec_latency`
4. Trade in direction `dir` (buy UP if CL up, buy DOWN if CL down)

**Exit rule** (simple first):
- Hold for `hold_seconds` then exit
- (Later: target profit, stop loss, reversion)

### 5.2 Strategy B: Strike-Cross Capture

**Entry rule**:
1. CL crosses strike K at time t
2. Trade in direction of cross (buy UP if CL > K, buy DOWN if CL < K)
3. More aggressive sizing when tau is low (near expiry)

**Exit rule**:
- Hold until expiry (or fixed hold time)

### 5.3 Strategy C: Fair Value Deviation

**Entry rule**:
1. `|pm_up_mid - p_hat| > mispricing_threshold`
2. Trade to capture reversion to fair value

### 5.4 Parameter Space

| Parameter | Range | Notes |
|-----------|-------|-------|
| threshold_bps | [5, 10, 15, 20, 30, 50] | CL move to trigger |
| hold_seconds | [5, 10, 15, 30, 60, 120] | Position duration |
| signal_latency | [0, 1, 2, 5, 10] | Time to observe CL |
| exec_latency | [0, 1, 2, 5] | Time to fill trade |
| tau_min | [0, 60, 120, 300] | Min time-to-expiry |
| tau_max | [300, 600, 900] | Max time-to-expiry |

---

## 6. Robustness & Statistics

### 6.1 Train/Test Split

**Requirement**: Chronological split only (no random)

```python
# Markets sorted by start time
train_markets = markets[:int(0.7 * len(markets))]  # First 70%
test_markets = markets[int(0.7 * len(markets)):]   # Last 30%
```

**Walk-forward validation** (preferred):
- Train on markets 1-10, test on 11-12
- Train on markets 1-12, test on 13-14
- etc.

### 6.2 Clustering

**Each 15-minute market is ONE dependent observation**.

Do NOT treat individual seconds as independent samples. Report:
- Mean PnL per market (not per trade)
- Std dev of PnL across markets
- t-stat with N = number of markets

### 6.3 Placebo Tests (Critical)

**Placebo 1: Shift CL series**
```python
# Shift CL data forward by 10s, 30s
# If edge persists, it's NOT latency capture
df['cl_mid_shifted'] = df.groupby('market_id')['cl_mid'].shift(10)
# Run same backtest - should show NO edge
```

**Placebo 2: Randomize event times**
```python
# Randomly permute CL within each market
# Should destroy any true signal
df['cl_mid_random'] = df.groupby('market_id')['cl_mid'].transform(np.random.permutation)
```

**Requirement**: Both placebos MUST show no edge for results to be valid.

### 6.4 Key Metrics

**Primary metrics** (per market, not per trade):
- PnL per market
- Hit rate per market
- Worst market PnL
- Best market PnL
- Median market PnL

**Latency cliff** (most important robustness check):
- Plot: Total PnL vs execution latency
- Find: At what latency does edge disappear?

**Distribution analysis**:
- PnL histogram across markets
- Fat tail analysis (are profits concentrated in few markets?)

**Do NOT use**:
- Sharpe at 1-second granularity (meaningless)
- Per-trade metrics without market clustering

---

## 7. Implementation Order

**Execute in this order** (not parallel):

### Phase 1: Foundation
1. **Execution model** with Split/Redeem routing
2. **Data filtering** (ETH, >90% coverage, observed data only)

### Phase 2: Discovery
3. **Event study** with all event types (jump, strike-cross, near-strike)
4. **Latency cliff measurement** (edge vs exec latency)

### Phase 3: Strategy
5. **Trading rule implementation** (Strategies A, B, C)
6. **Parameter sweep** on training data only

### Phase 4: Validation
7. **Fair value model** + mispricing analysis
8. **Placebo tests** (shift CL, randomize)
9. **Out-of-sample evaluation** on test markets

---

## 8. Output Requirements

### 8.1 Files to Generate

```
data_v2/backtest_results/
    execution_model_test.json     # Verify execution model works
    event_study_results.json      # Event study findings
    latency_cliff.json            # Edge vs latency
    parameter_sweep.csv           # Full sweep results
    trades.csv                    # All individual trades
    backtest_summary.json         # Final metrics
    placebo_results.json          # Placebo test results
    
    plots/
        response_curve.html       # PM response to CL events
        response_heatmap.html     # Response by (tau, event_size)
        latency_cliff.html        # PnL vs latency
        parameter_heatmap.html    # PnL vs (threshold, hold)
        pnl_distribution.html     # Per-market PnL
        equity_curve.html         # Cumulative PnL
        placebo_comparison.html   # Real vs placebo
```

### 8.2 Key Questions to Answer

1. **Does PM respond to CL moves?** → Event study
2. **How fast does PM respond?** → Lag-to-first-response
3. **At what latency does edge disappear?** → Latency cliff
4. **What parameters are robust?** → Parameter sweep with OOS test
5. **Is it latency or momentum?** → Fair value model + placebo tests
6. **How much can we make per market?** → PnL distribution

---

## 9. Future Extensions (Not in Step 2)

- **Order book depth**: Model slippage for larger sizes
- **Maker strategy**: Queue position modeling
- **Multi-asset**: Extend to BTC, SOL once more data collected
- **Live execution**: Real-time signal generation
- **Capacity analysis**: How much capital can strategy deploy?

---

## 10. Success Criteria

Step 2 is **successful** if we can confidently answer:

| Question | Success | Failure |
|----------|---------|---------|
| Does edge exist at 0 latency? | Clear positive PnL | No signal |
| Does edge survive realistic latency? | Positive PnL at 2-5s | Edge disappears |
| Are results robust to placebos? | Placebos show no edge | Placebos show similar edge |
| Is edge consistent across markets? | Most markets positive | Concentrated in 1-2 markets |

**If all four pass**: Proceed to collect more data and refine strategy
**If any fail**: Understand why before proceeding

