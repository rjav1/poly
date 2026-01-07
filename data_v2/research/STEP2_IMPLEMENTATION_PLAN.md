# Step 2: Implementation Plan

**Created**: 2026-01-06
**Reference**: STEP2_REQUIREMENTS.md

---

## Overview

This plan executes the backtest in the **correct order** to avoid false positives:

1. Build execution model (with conversion routing)
2. Run event study (understand the mechanism)
3. Measure latency cliff (find realistic edge)
4. THEN build trading strategies
5. Add fair value model + placebo validation

---

## Phase 1: Foundation (Day 1)

### Task 1.1: Execution Model (`scripts/backtest/execution_model.py`)

**Purpose**: Simulate trade fills with optimal conversion routing

**Implementation**:

```python
from dataclasses import dataclass
from typing import Literal, Tuple

@dataclass
class ExecutionConfig:
    signal_latency_s: float = 0.0  # Time to observe CL update
    exec_latency_s: float = 0.0    # Time to fill trade
    
@dataclass
class FillResult:
    side: Literal['buy_up', 'sell_up', 'buy_down', 'sell_down']
    entry_price: float
    route: Literal['direct', 'conversion']
    timestamp: int  # Actual fill time
    signal_time: int  # When we saw the signal

def get_effective_prices(row) -> dict:
    """Calculate best execution prices including conversion."""
    up_ask = row['pm_up_best_ask']
    up_bid = row['pm_up_best_bid']
    down_ask = row['pm_down_best_ask']
    down_bid = row['pm_down_best_bid']
    
    return {
        'buy_up': min(up_ask, 1 - down_bid),
        'sell_up': max(up_bid, 1 - down_ask),
        'buy_down': min(down_ask, 1 - up_bid),
        'sell_down': max(down_bid, 1 - up_ask),
        'buy_up_route': 'direct' if up_ask <= 1 - down_bid else 'conversion',
        'sell_up_route': 'direct' if up_bid >= 1 - down_ask else 'conversion',
        'buy_down_route': 'direct' if down_ask <= 1 - up_bid else 'conversion',
        'sell_down_route': 'direct' if down_bid >= 1 - up_ask else 'conversion',
    }

def simulate_fill(
    market_df: pd.DataFrame,
    signal_time: int,
    side: str,
    config: ExecutionConfig
) -> FillResult:
    """Simulate a fill given signal time and latencies."""
    # Time when we observe the signal
    observe_time = signal_time + int(config.signal_latency_s)
    # Time when trade executes
    fill_time = observe_time + int(config.exec_latency_s)
    
    # Get market state at fill time
    fill_row = market_df[market_df['t'] == fill_time]
    if fill_row.empty:
        return None  # Can't fill - time doesn't exist
    
    prices = get_effective_prices(fill_row.iloc[0])
    
    return FillResult(
        side=side,
        entry_price=prices[side],
        route=prices[f'{side}_route'],
        timestamp=fill_time,
        signal_time=signal_time
    )
```

**Tests to write**:
- Verify conversion routing is chosen when optimal
- Verify latency is applied correctly
- Verify fills at market boundaries are handled

**Output**: `execution_model.py` with full test coverage

---

### Task 1.2: Data Loader (`scripts/backtest/data_loader.py`)

**Purpose**: Load and filter ETH markets with >90% coverage

**Implementation**:

```python
import pandas as pd
import json
from pathlib import Path

def load_eth_markets(min_coverage: float = 90.0) -> Tuple[pd.DataFrame, dict]:
    """Load ETH markets with minimum coverage."""
    research_dir = Path('data_v2/research')
    
    # Load canonical dataset
    df = pd.read_parquet(research_dir / 'canonical_dataset_all_assets.parquet')
    
    # Load market info
    with open(research_dir / 'market_info_all_assets.json') as f:
        market_info = json.load(f)
    
    # Filter to ETH only
    df = df[df['asset'] == 'ETH'].copy()
    
    # Filter by coverage
    valid_markets = []
    for mid, info in market_info.items():
        if info.get('asset') == 'ETH':
            coverage = info.get('combined_coverage', 0)
            if coverage >= min_coverage:
                valid_markets.append(mid)
    
    df = df[df['market_id'].isin(valid_markets)]
    
    # Sort by market start time for chronological split
    market_times = df.groupby('market_id')['timestamp'].min().sort_values()
    market_order = {m: i for i, m in enumerate(market_times.index)}
    df['market_order'] = df['market_id'].map(market_order)
    
    return df, {m: market_info[m] for m in valid_markets}

def get_train_test_split(df: pd.DataFrame, train_frac: float = 0.7):
    """Chronological train/test split."""
    markets = df.groupby('market_id')['market_order'].first().sort_values()
    n_train = int(len(markets) * train_frac)
    
    train_markets = markets.iloc[:n_train].index.tolist()
    test_markets = markets.iloc[n_train:].index.tolist()
    
    return (
        df[df['market_id'].isin(train_markets)],
        df[df['market_id'].isin(test_markets)]
    )
```

---

## Phase 2: Discovery (Day 2)

### Task 2.1: Event Detection (`scripts/backtest/event_detection.py`)

**Purpose**: Detect all event types

**Implementation**:

```python
from dataclasses import dataclass
from typing import List, Literal

@dataclass
class Event:
    market_id: str
    t: int
    event_type: Literal['cl_jump', 'strike_cross', 'near_strike']
    direction: Literal['up', 'down']
    magnitude_bps: float
    tau: int
    delta_bps: float

def detect_cl_jumps(market_df: pd.DataFrame, threshold_bps: float) -> List[Event]:
    """Detect CL price jumps exceeding threshold."""
    events = []
    market_id = market_df['market_id'].iloc[0]
    
    df = market_df.copy()
    df['cl_change_bps'] = df['cl_mid'].pct_change() * 10000
    
    for idx, row in df.iterrows():
        if abs(row['cl_change_bps']) >= threshold_bps:
            events.append(Event(
                market_id=market_id,
                t=row['t'],
                event_type='cl_jump',
                direction='up' if row['cl_change_bps'] > 0 else 'down',
                magnitude_bps=abs(row['cl_change_bps']),
                tau=row['tau'],
                delta_bps=row['delta_bps']
            ))
    
    return events

def detect_strike_crosses(market_df: pd.DataFrame) -> List[Event]:
    """Detect when CL crosses the strike price K."""
    events = []
    market_id = market_df['market_id'].iloc[0]
    K = market_df['K'].iloc[0]
    
    df = market_df.copy()
    df['above_strike'] = df['cl_mid'] > K
    df['strike_crossed'] = df['above_strike'] != df['above_strike'].shift(1)
    
    for idx, row in df[df['strike_crossed']].iterrows():
        if pd.isna(row['above_strike']):
            continue
        events.append(Event(
            market_id=market_id,
            t=row['t'],
            event_type='strike_cross',
            direction='up' if row['above_strike'] else 'down',
            magnitude_bps=abs(row['delta_bps']),  # Distance from strike
            tau=row['tau'],
            delta_bps=row['delta_bps']
        ))
    
    return events

def detect_near_strike_regime(
    market_df: pd.DataFrame, 
    threshold_bps: float = 20.0
) -> pd.DataFrame:
    """Flag seconds where CL is near strike."""
    df = market_df.copy()
    df['near_strike'] = abs(df['delta_bps']) <= threshold_bps
    return df
```

---

### Task 2.2: Event Study (`scripts/backtest/event_study.py`)

**Purpose**: Measure PM response to CL events

**Implementation**:

```python
import numpy as np
import pandas as pd
from typing import Dict, List

def compute_pm_response(
    market_df: pd.DataFrame,
    event_t: int,
    max_lag: int = 30
) -> Dict[int, float]:
    """Compute PM response at each lag after event."""
    responses = {}
    
    # Baseline: PM price just before event
    baseline_row = market_df[market_df['t'] == event_t - 1]
    if baseline_row.empty:
        return responses
    
    baseline_pm_up = baseline_row['pm_up_mid'].iloc[0]
    
    for lag in range(0, max_lag + 1):
        target_row = market_df[market_df['t'] == event_t + lag]
        if not target_row.empty:
            responses[lag] = target_row['pm_up_mid'].iloc[0] - baseline_pm_up
    
    return responses

def run_event_study(
    df: pd.DataFrame,
    events: List[Event],
    max_lag: int = 30
) -> pd.DataFrame:
    """Run event study across all events."""
    results = []
    
    for event in events:
        market_df = df[df['market_id'] == event.market_id]
        responses = compute_pm_response(market_df, event.t, max_lag)
        
        for lag, response in responses.items():
            # Normalize response direction
            directed_response = response if event.direction == 'up' else -response
            
            results.append({
                'market_id': event.market_id,
                'event_t': event.t,
                'event_type': event.event_type,
                'direction': event.direction,
                'magnitude_bps': event.magnitude_bps,
                'tau': event.tau,
                'delta_bps': event.delta_bps,
                'lag': lag,
                'pm_response': response,
                'pm_response_directed': directed_response,
                'tau_bucket': get_tau_bucket(event.tau),
                'magnitude_bucket': get_magnitude_bucket(event.magnitude_bps)
            })
    
    return pd.DataFrame(results)

def get_tau_bucket(tau: int) -> str:
    if tau > 600:
        return 'early (10-15min)'
    elif tau > 300:
        return 'mid (5-10min)'
    else:
        return 'late (0-5min)'

def get_magnitude_bucket(magnitude_bps: float) -> str:
    if magnitude_bps < 10:
        return 'small (<10bps)'
    elif magnitude_bps < 20:
        return 'medium (10-20bps)'
    else:
        return 'large (>20bps)'
```

**Key outputs**:
- Average response curve by event type
- Response heatmap by (tau_bucket, magnitude_bucket)
- Histogram of lag-to-first-significant-response
- Strike-cross specific analysis

---

### Task 2.3: Latency Cliff Analysis (`scripts/backtest/latency_cliff.py`)

**Purpose**: Determine at what latency edge disappears

**Implementation**:

```python
def compute_latency_cliff(
    df: pd.DataFrame,
    events: List[Event],
    latencies: List[int] = [0, 1, 2, 3, 5, 10, 15, 20, 30]
) -> pd.DataFrame:
    """Compute potential PnL at each latency level."""
    results = []
    
    for latency in latencies:
        total_pnl = 0
        n_trades = 0
        
        for event in events:
            market_df = df[df['market_id'] == event.market_id]
            
            # Entry time = event time + latency
            entry_t = event.t + latency
            entry_row = market_df[market_df['t'] == entry_t]
            if entry_row.empty:
                continue
            
            # Get entry price (with conversion)
            prices = get_effective_prices(entry_row.iloc[0])
            
            # Assume we hold for fixed period (e.g., 15s) or to expiry
            exit_t = min(entry_t + 15, market_df['t'].max())
            exit_row = market_df[market_df['t'] == exit_t]
            if exit_row.empty:
                continue
            
            exit_prices = get_effective_prices(exit_row.iloc[0])
            
            # Calculate PnL
            if event.direction == 'up':
                entry = prices['buy_up']
                exit = exit_prices['sell_up']
            else:
                entry = prices['buy_down']
                exit = exit_prices['sell_down']
            
            pnl = exit - entry
            total_pnl += pnl
            n_trades += 1
        
        results.append({
            'latency': latency,
            'total_pnl': total_pnl,
            'n_trades': n_trades,
            'avg_pnl': total_pnl / n_trades if n_trades > 0 else 0
        })
    
    return pd.DataFrame(results)
```

**Key output**: Plot showing PnL vs latency - the "cliff" where edge disappears

---

## Phase 3: Strategy (Day 3)

### Task 3.1: Strategy Classes (`scripts/backtest/strategies.py`)

**Purpose**: Define tradeable strategy rules

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

@dataclass
class Trade:
    market_id: str
    entry_t: int
    exit_t: int
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    entry_route: str
    exit_route: str

class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, market_df: pd.DataFrame) -> List[dict]:
        """Generate entry signals for a market."""
        pass

class LatencyCaptureStrategy(Strategy):
    """Trade after CL jump, betting PM will follow."""
    
    def __init__(
        self,
        threshold_bps: float = 10.0,
        hold_seconds: int = 15,
        tau_min: int = 0,
        tau_max: int = 900
    ):
        self.threshold_bps = threshold_bps
        self.hold_seconds = hold_seconds
        self.tau_min = tau_min
        self.tau_max = tau_max
    
    def generate_signals(self, market_df: pd.DataFrame) -> List[dict]:
        signals = []
        events = detect_cl_jumps(market_df, self.threshold_bps)
        
        for event in events:
            if self.tau_min <= event.tau <= self.tau_max:
                signals.append({
                    'entry_t': event.t,
                    'exit_t': min(event.t + self.hold_seconds, 899),
                    'side': 'buy_up' if event.direction == 'up' else 'buy_down',
                    'reason': f'cl_jump_{event.magnitude_bps:.1f}bps'
                })
        
        return signals

class StrikeCrossStrategy(Strategy):
    """Trade when CL crosses strike, especially near expiry."""
    
    def __init__(self, tau_max: int = 300):
        self.tau_max = tau_max  # Only trade in last 5 minutes
    
    def generate_signals(self, market_df: pd.DataFrame) -> List[dict]:
        signals = []
        events = detect_strike_crosses(market_df)
        
        for event in events:
            if event.tau <= self.tau_max:
                signals.append({
                    'entry_t': event.t,
                    'exit_t': 899,  # Hold to expiry
                    'side': 'buy_up' if event.direction == 'up' else 'buy_down',
                    'reason': f'strike_cross_tau{event.tau}'
                })
        
        return signals
```

---

### Task 3.2: Backtest Engine (`scripts/backtest/backtest_engine.py`)

**Purpose**: Execute strategies and compute metrics

```python
from typing import List, Dict
import pandas as pd
import numpy as np

def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    config: ExecutionConfig
) -> Dict:
    """Run backtest across all markets."""
    all_trades = []
    market_pnls = {}
    
    for market_id in df['market_id'].unique():
        market_df = df[df['market_id'] == market_id].copy()
        signals = strategy.generate_signals(market_df)
        
        market_trades = []
        for signal in signals:
            # Apply latency
            actual_entry_t = signal['entry_t'] + int(config.signal_latency_s + config.exec_latency_s)
            actual_exit_t = signal['exit_t'] + int(config.exec_latency_s)
            
            entry_row = market_df[market_df['t'] == actual_entry_t]
            exit_row = market_df[market_df['t'] == actual_exit_t]
            
            if entry_row.empty or exit_row.empty:
                continue
            
            entry_prices = get_effective_prices(entry_row.iloc[0])
            exit_prices = get_effective_prices(exit_row.iloc[0])
            
            # Map side to entry/exit
            if signal['side'] == 'buy_up':
                entry_price = entry_prices['buy_up']
                exit_price = exit_prices['sell_up']
                entry_route = entry_prices['buy_up_route']
                exit_route = exit_prices['sell_up_route']
            elif signal['side'] == 'buy_down':
                entry_price = entry_prices['buy_down']
                exit_price = exit_prices['sell_down']
                entry_route = entry_prices['buy_down_route']
                exit_route = exit_prices['sell_down_route']
            
            pnl = exit_price - entry_price
            
            trade = Trade(
                market_id=market_id,
                entry_t=actual_entry_t,
                exit_t=actual_exit_t,
                side=signal['side'],
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                entry_route=entry_route,
                exit_route=exit_route
            )
            market_trades.append(trade)
            all_trades.append(trade)
        
        market_pnls[market_id] = sum(t.pnl for t in market_trades)
    
    return {
        'trades': all_trades,
        'market_pnls': market_pnls,
        'metrics': compute_metrics(all_trades, market_pnls)
    }

def compute_metrics(trades: List[Trade], market_pnls: Dict) -> Dict:
    """Compute per-market clustered metrics."""
    pnls = list(market_pnls.values())
    
    return {
        'n_markets': len(pnls),
        'n_trades': len(trades),
        'total_pnl': sum(pnls),
        'mean_pnl_per_market': np.mean(pnls),
        'std_pnl_per_market': np.std(pnls),
        't_stat': np.mean(pnls) / (np.std(pnls) / np.sqrt(len(pnls))) if len(pnls) > 1 else 0,
        'hit_rate': sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0,
        'worst_market': min(pnls) if pnls else 0,
        'best_market': max(pnls) if pnls else 0,
        'median_market': np.median(pnls) if pnls else 0,
    }
```

---

### Task 3.3: Parameter Sweep (`scripts/backtest/parameter_sweep.py`)

**Purpose**: Find robust parameter regions

```python
from itertools import product
import pandas as pd

def run_parameter_sweep(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> pd.DataFrame:
    """Sweep parameters, train on train_df, evaluate on test_df."""
    
    param_grid = {
        'threshold_bps': [5, 10, 15, 20, 30],
        'hold_seconds': [5, 10, 15, 30, 60],
        'signal_latency': [0, 1, 2, 5, 10],
        'exec_latency': [0, 1, 2, 5],
        'tau_min': [0, 300],
        'tau_max': [300, 600, 900],
    }
    
    results = []
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        
        strategy = LatencyCaptureStrategy(
            threshold_bps=param_dict['threshold_bps'],
            hold_seconds=param_dict['hold_seconds'],
            tau_min=param_dict['tau_min'],
            tau_max=param_dict['tau_max']
        )
        
        config = ExecutionConfig(
            signal_latency_s=param_dict['signal_latency'],
            exec_latency_s=param_dict['exec_latency']
        )
        
        # Run on train
        train_result = run_backtest(train_df, strategy, config)
        # Run on test
        test_result = run_backtest(test_df, strategy, config)
        
        results.append({
            **param_dict,
            'train_pnl': train_result['metrics']['total_pnl'],
            'train_hit_rate': train_result['metrics']['hit_rate'],
            'test_pnl': test_result['metrics']['total_pnl'],
            'test_hit_rate': test_result['metrics']['hit_rate'],
            'train_n_trades': train_result['metrics']['n_trades'],
            'test_n_trades': test_result['metrics']['n_trades'],
        })
    
    return pd.DataFrame(results)
```

---

## Phase 4: Validation (Day 4)

### Task 4.1: Fair Value Model (`scripts/backtest/fair_value.py`)

**Purpose**: Separate latency capture from momentum

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

class FairValueModel:
    """Empirical fair value model for PM probabilities."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
    
    def fit(self, df: pd.DataFrame):
        """Fit on training markets."""
        # Get one row per market at end (with outcome)
        final_rows = df.groupby('market_id').last().reset_index()
        
        # Features
        X = self._build_features(df)
        # Target: outcome Y
        y = final_rows['Y'].values
        
        self.model = LogisticRegression()
        self.model.fit(X, y)
    
    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build feature matrix from market data."""
        # For each row, we need: delta_bps, tau, delta_bps*tau, recent_vol
        features = df[['delta_bps', 'tau']].copy()
        features['delta_tau'] = features['delta_bps'] * features['tau']
        
        # Recent CL volatility (rolling std of returns)
        features['cl_vol'] = df.groupby('market_id')['cl_mid'].transform(
            lambda x: x.pct_change().rolling(30, min_periods=1).std()
        )
        
        return features.values
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict fair probability for each row."""
        X = self._build_features(df)
        return self.model.predict_proba(X)[:, 1]  # P(Y=1)

def compute_mispricing(df: pd.DataFrame, model: FairValueModel) -> pd.DataFrame:
    """Add mispricing column to dataframe."""
    df = df.copy()
    df['p_hat'] = model.predict(df)
    df['mispricing'] = df['pm_up_mid'] - df['p_hat']
    return df
```

---

### Task 4.2: Placebo Tests (`scripts/backtest/placebo_tests.py`)

**Purpose**: Validate that edge is from latency, not spurious

```python
def run_placebo_shift(
    df: pd.DataFrame,
    strategy: Strategy,
    config: ExecutionConfig,
    shift_seconds: int = 10
) -> Dict:
    """Shift CL forward - should destroy edge."""
    df_shifted = df.copy()
    df_shifted['cl_mid'] = df_shifted.groupby('market_id')['cl_mid'].shift(-shift_seconds)
    df_shifted['delta_bps'] = (df_shifted['cl_mid'] - df_shifted['K']) / df_shifted['K'] * 10000
    df_shifted = df_shifted.dropna()
    
    return run_backtest(df_shifted, strategy, config)

def run_placebo_random(
    df: pd.DataFrame,
    strategy: Strategy,
    config: ExecutionConfig,
    seed: int = 42
) -> Dict:
    """Randomize CL within market - should destroy edge."""
    np.random.seed(seed)
    df_random = df.copy()
    df_random['cl_mid'] = df_random.groupby('market_id')['cl_mid'].transform(
        lambda x: np.random.permutation(x.values)
    )
    df_random['delta_bps'] = (df_random['cl_mid'] - df_random['K']) / df_random['K'] * 10000
    
    return run_backtest(df_random, strategy, config)

def validate_placebo_results(real_result: Dict, placebo_results: List[Dict]) -> bool:
    """Check that placebos show no edge."""
    real_pnl = real_result['metrics']['total_pnl']
    
    for placebo in placebo_results:
        if placebo['metrics']['total_pnl'] >= real_pnl * 0.5:
            return False  # Placebo has similar edge - suspicious
    
    return True
```

---

## Phase 5: Visualization & Reporting (Day 5)

### Task 5.1: Visualizations (`scripts/backtest/visualizations.py`)

**Required plots**:

1. **Response curve**: E[ΔPM | CL event] vs lag
2. **Response heatmap**: (tau_bucket, magnitude_bucket) → response
3. **Latency cliff**: PnL vs execution latency
4. **Parameter heatmap**: PnL vs (threshold, hold_time)
5. **Per-market PnL distribution**: Histogram
6. **Equity curve**: Cumulative PnL over markets
7. **Placebo comparison**: Real vs shifted vs random

**Implementation**: Use Plotly for interactive HTML charts

---

### Task 5.2: Results Report (`scripts/backtest/generate_report.py`)

**Output**: `backtest_results/BACKTEST_REPORT.md`

Sections:
1. Executive Summary
2. Data Overview (N markets, coverage, time range)
3. Event Study Results (response curves, key findings)
4. Latency Cliff Analysis
5. Strategy Performance (best parameters, robustness)
6. Fair Value Model Analysis
7. Placebo Test Results
8. Conclusions & Recommendations

---

## File Structure

```
scripts/backtest/
    __init__.py
    execution_model.py      # Fill simulation with conversion
    data_loader.py          # Load and filter ETH data
    event_detection.py      # Detect CL events
    event_study.py          # Measure PM response
    latency_cliff.py        # Edge vs latency analysis
    strategies.py           # Strategy classes
    backtest_engine.py      # Trade simulation
    parameter_sweep.py      # Parameter optimization
    fair_value.py           # Baseline model
    placebo_tests.py        # Validation tests
    visualizations.py       # All plots
    generate_report.py      # Final report

data_v2/backtest_results/
    event_study/
        response_curves.json
        response_heatmap.json
    latency_cliff.json
    parameter_sweep.csv
    trades.csv
    placebo_results.json
    metrics.json
    plots/
        *.html
    BACKTEST_REPORT.md
```

---

## Execution Timeline

| Day | Phase | Tasks | Output |
|-----|-------|-------|--------|
| 1 | Foundation | 1.1, 1.2 | execution_model.py, data_loader.py |
| 2 | Discovery | 2.1, 2.2, 2.3 | event_detection.py, event_study.py, latency_cliff.py |
| 3 | Strategy | 3.1, 3.2, 3.3 | strategies.py, backtest_engine.py, parameter_sweep.py |
| 4 | Validation | 4.1, 4.2 | fair_value.py, placebo_tests.py |
| 5 | Reporting | 5.1, 5.2 | visualizations.py, generate_report.py, BACKTEST_REPORT.md |

---

## Success Criteria Checklist

- [ ] Execution model handles Split/Redeem conversion
- [ ] Event study shows PM response curve
- [ ] Latency cliff identified (where edge disappears)
- [ ] Best parameters found on train data
- [ ] Parameters validated on test data
- [ ] Placebo tests pass (no edge with shifted/random CL)
- [ ] Fair value model separates latency from momentum
- [ ] All metrics are per-market clustered
- [ ] Results documented in report

