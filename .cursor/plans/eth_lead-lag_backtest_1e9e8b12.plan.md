---
name: ETH Lead-Lag Backtest
overview: Build a latency-aware taker backtest to determine if a lead-lag edge exists between Chainlink and Polymarket on ETH, using the 24 high-coverage markets already collected.
todos:
  - id: create-docs
    content: Create STEP2_REQUIREMENTS.md with all requirements from AI assistant
    status: pending
  - id: execution-model
    content: Implement execution_model.py with taker fill simulation
    status: pending
  - id: event-study
    content: Implement event_study.py with CL event detection and PM response analysis
    status: pending
  - id: strategy
    content: Implement strategy.py with latency capture strategy class
    status: pending
  - id: backtest-engine
    content: Implement backtest_engine.py with parameter sweeps
    status: pending
    dependencies:
      - execution-model
      - strategy
  - id: visualizations
    content: Implement visualizations.py with all required plots
    status: pending
    dependencies:
      - backtest-engine
  - id: runner
    content: Implement run_backtest.py CLI with full pipeline
    status: pending
    dependencies:
      - event-study
      - backtest-engine
      - visualizations
  - id: run-analysis
    content: Run full analysis and generate results report
    status: pending
    dependencies:
      - runner
---

# Step 2: ETH Latency-Aware Taker Backtest

## Context and Requirements

### What We Have

- **24 ETH markets** with >90% combined coverage (out of 29 total ETH markets)
- **16 UP / 16 DOWN** outcome split across all 32 markets (perfectly balanced)
- **Canonical dataset**: `data_v2/research/canonical_dataset_all_assets.parquet` with 28,800 rows
- **Key columns**: `t`, `tau`, `cl_mid`, `pm_up_best_bid`, `pm_up_best_ask`, `pm_down_best_bid`, `pm_down_best_ask`, `cl_ffill`, `pm_ffill`, `delta_bps`

### Key Assumptions (from user)

- **No fees** (Polymarket has no trading fees)
- **Taker strategy only** (maker comparison deferred)
- **Top-of-book sufficient** for initial strategy discovery (depth needed later for capacity)
- **Focus on ETH** (other assets later)

---

## Implementation Plan

### Step 2.1: Execution + Cost Model

**Goal**: Create a function that simulates trade fills given market state and latency assumptions.**File to create**: [`scripts/backtest/execution_model.py`](scripts/backtest/execution_model.py)**Core function**:

```python
def simulate_taker_fill(
    side: str,              # 'buy_up', 'sell_up', 'buy_down', 'sell_down'
    entry_t: int,           # second to enter
    exit_t: int,            # second to exit
    signal_latency: int,    # seconds delay to observe CL move
    exec_latency: int,      # seconds delay from decision to fill
    df_market: pd.DataFrame # market data
) -> dict:
    """Returns entry_price, exit_price, pnl, filled (bool)"""
```

**Entry/Exit logic**:

- Buy UP: pay `pm_up_best_ask` at `entry_t + exec_latency`
- Sell UP: receive `pm_up_best_bid` at `exit_t + exec_latency`
- Buy DOWN: pay `pm_down_best_ask`
- Sell DOWN: receive `pm_down_best_bid`

**Edge cases to handle**:

- If `cl_ffill=1` or `pm_ffill=1` at trade time, mark as "missing data" (optional: skip trade)
- If requested timestamp exceeds market (t > 899), cannot trade

---

### Step 2.2: PM Response Function (Event Study)

**Goal**: Quantify how PM prices respond to CL moves before building trading rules.**File to create**: [`scripts/backtest/event_study.py`](scripts/backtest/event_study.py)**Event detection**:

```python
def detect_cl_events(df_market: pd.DataFrame, threshold_bps: float) -> pd.DataFrame:
    """
    Detect CL price moves >= threshold_bps.
    Returns DataFrame with columns: t0, direction ('up'/'down'), magnitude_bps
    """
```

**Response measurement**:

```python
def compute_pm_response(
    df_market: pd.DataFrame, 
    events: pd.DataFrame,
    max_lag: int = 30  # seconds after event
) -> pd.DataFrame:
    """
    For each event, compute PM price change over 0...max_lag seconds.
    Returns: event_id, lag, pm_up_mid_change, pm_down_mid_change
    """
```

**Regime splits** (for later analysis):

- Time-to-expiry buckets: `tau` in [0-300], [300-600], [600-900]
- Distance to strike: `delta_bps` buckets
- Event magnitude: threshold_bps buckets

**Key visualizations**:

1. Average response curve: E[delta PM | CL jump] vs lag seconds
2. Histogram of lag-to-first-response
3. Heatmap: response magnitude by (tau, event_size)

---

### Step 2.3: Strategy Definition

**Goal**: Define a simple "latency capture after CL jump" strategy.**File to create**: [`scripts/backtest/strategy.py`](scripts/backtest/strategy.py)**Strategy template**:

```python
@dataclass
class StrategyParams:
    threshold_bps: float      # Min CL move to trigger (e.g., 5, 10, 20 bps)
    hold_seconds: int         # How long to hold position
    signal_latency: int       # Assumed latency to observe CL (0, 1, 2s)
    exec_latency: int         # Assumed latency to execute (0, 1, 2, 5s)
    tau_min: int = 0          # Min time-to-expiry to trade
    tau_max: int = 900        # Max time-to-expiry to trade

def generate_trades(
    df_market: pd.DataFrame,
    params: StrategyParams
) -> List[Trade]:
    """
    Generate trade signals for one market.
    Returns list of Trade objects with: entry_t, exit_t, side, entry_reason
    """
```

**Entry logic**:

1. At time t, if CL moved by >= `threshold_bps` in direction `dir`
2. And `tau` is within [tau_min, tau_max]
3. Signal to enter PM in direction `dir`

**Exit logic** (simple first):

- Exit after `hold_seconds`
- (Later: add target profit, stop loss, reversion signal)

---

### Step 2.4: Backtest Engine + Robustness

**Goal**: Run strategy across markets and produce metrics + visualizations.**File to create**: [`scripts/backtest/backtest_engine.py`](scripts/backtest/backtest_engine.py)**Core function**:

```python
def run_backtest(
    df: pd.DataFrame,           # Full canonical dataset
    params: StrategyParams,
    markets: List[str] = None   # Filter to specific markets
) -> BacktestResult:
    """Run strategy across all markets, return aggregated results."""
```

**BacktestResult class**:

```python
@dataclass
class BacktestResult:
    trades: pd.DataFrame          # All trades with PnL
    total_pnl: float
    n_trades: int
    win_rate: float
    avg_pnl_per_trade: float
    sharpe: float                 # Careful: micro horizon
    max_drawdown: float
    pnl_by_market: pd.DataFrame
```

**Parameter sweeps to implement**:

```python
def sweep_parameters(
    df: pd.DataFrame,
    threshold_range: List[float],    # e.g., [5, 10, 15, 20, 30] bps
    hold_range: List[int],           # e.g., [5, 10, 15, 30, 60] seconds
    latency_range: List[int]         # e.g., [0, 1, 2, 5, 10] seconds
) -> pd.DataFrame:
    """Returns DataFrame with params and corresponding metrics."""
```

**Key visualizations** (file: [`scripts/backtest/visualizations.py`](scripts/backtest/visualizations.py)):

1. Equity curve (cumulative PnL)
2. Parameter heatmap: PnL vs (threshold, hold_time)
3. Latency cliff plot: PnL vs execution latency
4. Per-market PnL distribution (box plot)
5. Trade PnL histogram

**Robustness checks**:

- Out-of-sample split: first 70% markets for param selection, last 30% for test
- Performance by tau bucket
- Performance vs latency sensitivity

---

### Step 2.5: Main Runner + Reports

**File to create**: [`scripts/backtest/run_backtest.py`](scripts/backtest/run_backtest.py)**CLI interface**:

```bash
python scripts/backtest/run_backtest.py \
    --asset ETH \
    --min-coverage 90 \
    --sweep                    # Run parameter sweep
    --output-dir data_v2/backtest_results
```

**Output files**:

- `backtest_results.json` - All metrics
- `parameter_sweep.csv` - Sweep results
- `trades.csv` - Individual trades
- `plots/` - All visualizations

---

## File Structure

```javascript
scripts/backtest/
    __init__.py
    execution_model.py      # Step 2.1
    event_study.py          # Step 2.2
    strategy.py             # Step 2.3
    backtest_engine.py      # Step 2.4
    visualizations.py       # Step 2.4
    run_backtest.py         # Step 2.5 (CLI)
```

---

## Documentation File

**File to create**: [`data_v2/research/STEP2_REQUIREMENTS.md`](data_v2/research/STEP2_REQUIREMENTS.md)This will capture all requirements from the AI assistant's recommendations for future reference.---

## Success Criteria

Step 2 is complete when we can answer:

1. **Does PM respond to CL moves?** (Event study shows lag and magnitude)
2. **Is there edge after latency?** (Backtest shows positive PnL for realistic latency assumptions)
3. **What parameters are robust?** (Sweep shows stable parameter regions)
4. **What's the latency cliff?** (At what latency does edge disappear?)

## Data Filtering

For this backtest, filter to:

- Asset = ETH
- Markets with `both_coverage_pct >= 90%`
- Rows with `cl_ffill == 0 AND pm_ffill == 0` for lead-lag accuracy