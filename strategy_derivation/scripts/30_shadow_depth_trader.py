#!/usr/bin/env python3
"""
Phase 5: Shadow Trader with Depth Logging

Updates shadow trader to log depth-derived executability:
- Best 6 ask levels (price, size) for UP and DOWN
- q_max executable at signal time
- q_max after 1s/2s/5s (track how capacity evolves)
- Time-to-disappear of edge (how long underround persists)
- VWAP cost at q=1, q=5, q=10

Outputs:
- shadow_depth_log.parquet - All signals with full depth snapshots
- shadow_depth_report.md - Dashboard with executability metrics
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import defaultdict

# Import execution math
import sys
sys.path.insert(0, str(Path(__file__).parent))
from importlib.util import spec_from_file_location, module_from_spec

exec_math_path = Path(__file__).parent / "28_execution_math.py"
spec = spec_from_file_location("execution_math", exec_math_path)
execution_math = module_from_spec(spec)
spec.loader.exec_module(execution_math)

BookLadder = execution_math.BookLadder
extract_ladder_from_row = execution_math.extract_ladder_from_row
set_edge_from_books = execution_math.set_edge_from_books
max_executable_size = execution_math.max_executable_size
compute_pnl_at_size = execution_math.compute_pnl_at_size

# Configuration
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
MARKETS_DIR = BASE_DIR.parent / "data_v2" / "markets_6levels" / "ETH"
MARKET_INFO_PATH = BASE_DIR.parent / "data_v2" / "research_6levels" / "market_info_all_assets.json"

# Strategy parameters
EPSILON = 0.005
TAU_MIN = 600
TAU_MAX = 900
SLIPPAGE_BUFFER = 0.005
COOLDOWN_SECONDS = 30
MAX_SIGNALS = 100


@dataclass
class DepthSnapshot:
    """Full 6-level depth snapshot."""
    up_ask_prices: List[float]
    up_ask_sizes: List[float]
    down_ask_prices: List[float]
    down_ask_sizes: List[float]


@dataclass
class DepthSignalLog:
    """Complete signal log with depth data."""
    signal_id: int
    market_id: str
    t: int
    tau: int
    
    # L1 underround
    underround: float
    
    # Full depth snapshot
    depth_snapshot: DepthSnapshot
    
    # Executability metrics at signal time
    q_max_at_signal: float
    edge_at_q1: float
    edge_at_q5: float
    edge_at_q10: float
    
    # VWAP costs at different sizes
    vwap_cost_q1: float
    vwap_cost_q5: float
    vwap_cost_q10: float
    
    # Capacity evolution (how q_max changes over time)
    q_max_after_0_5s: float
    q_max_after_1s: float
    q_max_after_2s: float
    q_max_after_5s: float
    
    # Persistence (seconds until edge disappears)
    edge_persisted_0_5s: bool
    edge_persisted_1s: bool
    edge_persisted_2s: bool
    edge_persisted_5s: bool
    edge_disappeared_at: Optional[int]  # seconds after signal
    
    # Edge(q=1) at follow-up times
    edge_q1_after_0_5s: float
    edge_q1_after_1s: float
    edge_q1_after_2s: float
    edge_q1_after_5s: float


def load_market_info() -> Dict:
    """Load market info."""
    with open(MARKET_INFO_PATH, 'r') as f:
        market_info = json.load(f)
    
    if isinstance(market_info, list):
        return {m['market_id']: m for m in market_info}
    return market_info


def load_market_data(market_dir: Path) -> pd.DataFrame:
    """Load polymarket CSV for a single market."""
    pm_path = market_dir / "polymarket.csv"
    if not pm_path.exists():
        return None
    return pd.read_csv(pm_path)


def get_depth_snapshot(row: dict) -> DepthSnapshot:
    """Extract full depth snapshot from row."""
    up_asks = extract_ladder_from_row(row, 'up', 'ask')
    down_asks = extract_ladder_from_row(row, 'down', 'ask')
    
    return DepthSnapshot(
        up_ask_prices=up_asks.prices,
        up_ask_sizes=up_asks.sizes,
        down_ask_prices=down_asks.prices,
        down_ask_sizes=down_asks.sizes
    )


def compute_edge_at_size(row: dict, q: float, slippage_buffer: float) -> Tuple[float, float]:
    """Compute edge and VWAP cost at specific size."""
    up_asks = extract_ladder_from_row(row, 'up', 'ask')
    down_asks = extract_ladder_from_row(row, 'down', 'ask')
    
    pnl = compute_pnl_at_size(up_asks, down_asks, q, slippage_buffer)
    
    return pnl['edge_per_set'], pnl['set_cost']


def compute_q_max_at_row(row: dict, epsilon: float, slippage_buffer: float) -> float:
    """Compute max executable size at a given row."""
    up_asks = extract_ladder_from_row(row, 'up', 'ask')
    down_asks = extract_ladder_from_row(row, 'down', 'ask')
    
    if not up_asks.is_valid() or not down_asks.is_valid():
        return 0.0
    
    q_max, _ = max_executable_size(up_asks, down_asks, epsilon, slippage_buffer)
    return q_max


def _nearest_row_by_time(df: pd.DataFrame, base_ms: int, dt_seconds: float) -> Optional[dict]:
    """Return nearest row to base_ms + dt_seconds*1000 using timestamp_ms if present; else use t."""
    target_ms = base_ms + int(dt_seconds * 1000)
    if 'timestamp_ms' in df.columns:
        # Choose row with minimal absolute delta
        idx = (df['timestamp_ms'] - target_ms).abs().idxmin()
        if pd.isna(idx):
            return None
        row = df.loc[idx].to_dict()
        row['_delta_ms'] = int(abs(int(row.get('timestamp_ms', target_ms)) - target_ms))
        return row
    else:
        # Fall back to whole-second 't' alignment
        base_t = None
        if 'timestamp_ms' in df.columns:
            # Shouldn't happen here, but keep for safety
            base_t = int(round(base_ms / 1000.0))
        # If we cannot infer, approximate by integer seconds
        approx_t = int(round(dt_seconds))
        # We cannot know exact base t here; caller should supply rows by 't' relative
        return None


def track_capacity_evolution(
    df: pd.DataFrame,
    signal_row: dict,
    epsilon: float,
    slippage_buffer: float
) -> Dict:
    """Track how capacity and edge evolve after signal at 0.5s/1s/2s/5s using nearest timestamp_ms if available."""
    evolution = {
        'q_max_after_0_5s': 0.0,
        'q_max_after_1s': 0.0,
        'q_max_after_2s': 0.0,
        'q_max_after_5s': 0.0,
        'edge_q1_after_0_5s': 0.0,
        'edge_q1_after_1s': 0.0,
        'edge_q1_after_2s': 0.0,
        'edge_q1_after_5s': 0.0,
        'edge_persisted_0_5s': False,
        'edge_persisted_1s': False,
        'edge_persisted_2s': False,
        'edge_persisted_5s': False,
        'edge_disappeared_at': None
    }

    # Establish base time
    if 'timestamp_ms' in signal_row:
        base_ms = int(signal_row['timestamp_ms'])
    else:
        # If no ms timestamps, attempt to construct from seconds 't' by mapping to df rows with same 't'
        # and using that row's timestamp_ms if present.
        if 't' in signal_row and 'timestamp_ms' in df.columns:
            candidates = df[df['t'] == signal_row['t']]
            if not candidates.empty:
                base_ms = int(candidates.iloc[0]['timestamp_ms'])
            else:
                base_ms = None
        else:
            base_ms = None

    # Helper to compute q_max and edge at q=1 for a chosen future row
    def compute_metrics_at_row(future_row: Optional[dict]) -> Tuple[float, float]:
        if not future_row:
            return 0.0, 0.0
        q_max_val = compute_q_max_at_row(future_row, epsilon, slippage_buffer)
        edge_q1_val, _ = compute_edge_at_size(future_row, 1.0, slippage_buffer)
        return q_max_val, edge_q1_val

    checks = [(0.5, '0_5s'), (1.0, '1s'), (2.0, '2s'), (5.0, '5s')]
    last_had_edge = True

    for seconds, label in checks:
        future_row = None
        if base_ms is not None and 'timestamp_ms' in df.columns:
            future_row = _nearest_row_by_time(df, base_ms, seconds)
        else:
            # Fallback: use integer t alignment if available
            if 't' in signal_row and 't' in df.columns:
                target_t = int(signal_row['t'] + round(seconds))
                match = df[df['t'] == target_t]
                if not match.empty:
                    future_row = match.iloc[0].to_dict()

        q_max_val, edge_q1_val = compute_metrics_at_row(future_row)
        evolution[f'q_max_after_{label}'] = q_max_val
        evolution[f'edge_q1_after_{label}'] = edge_q1_val
        edge_exists = q_max_val >= 1.0
        evolution[f'edge_persisted_{label}'] = edge_exists

        if last_had_edge and not edge_exists and evolution['edge_disappeared_at'] is None:
            # Record first disappearance time as integer seconds where applicable
            evolution['edge_disappeared_at'] = int(seconds) if seconds >= 1.0 else 0

        last_had_edge = edge_exists

    return evolution


def run_shadow_trader_for_market(
    market_dir: Path,
    params: Dict
) -> List[DepthSignalLog]:
    """Run shadow trader for a single market."""
    df = load_market_data(market_dir)
    if df is None:
        return []
    
    market_id = market_dir.name
    
    # Add t column if missing
    if 't' not in df.columns:
        df['t'] = range(len(df))
    
    signals = []
    last_signal_t = -params['cooldown']
    signal_id = 0
    
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        t = int(row_dict['t'])
        tau = params['market_duration'] - t
        
        # Check tau window
        if tau < params['tau_min'] or tau > params['tau_max']:
            continue
        
        # Check cooldown
        if t - last_signal_t < params['cooldown']:
            continue
        
        # Extract book ladders
        up_asks = extract_ladder_from_row(row_dict, 'up', 'ask')
        down_asks = extract_ladder_from_row(row_dict, 'down', 'ask')
        
        if not up_asks.is_valid() or not down_asks.is_valid():
            continue
        
        # Compute L1 underround
        l1_cost = up_asks.best_price + down_asks.best_price
        underround = 1.0 - l1_cost
        
        if underround < params['epsilon']:
            continue
        
        # Compute max executable size
        q_max, _ = max_executable_size(
            up_asks, down_asks,
            epsilon=params['epsilon'],
            slippage_buffer=params['slippage_buffer']
        )
        
        if q_max < 1.0:
            continue
        
        signal_id += 1
        last_signal_t = t
        
        # Get depth snapshot
        depth_snapshot = get_depth_snapshot(row_dict)
        
        # Compute edge at different sizes
        edge_q1, cost_q1 = compute_edge_at_size(row_dict, 1.0, params['slippage_buffer'])
        edge_q5, cost_q5 = compute_edge_at_size(row_dict, 5.0, params['slippage_buffer'])
        edge_q10, cost_q10 = compute_edge_at_size(row_dict, 10.0, params['slippage_buffer'])
        
        # Track capacity evolution
        evolution = track_capacity_evolution(
            df, row_dict,
            params['epsilon'],
            params['slippage_buffer']
        )
        
        signal = DepthSignalLog(
            signal_id=signal_id,
            market_id=market_id,
            t=t,
            tau=tau,
            underround=underround,
            depth_snapshot=depth_snapshot,
            q_max_at_signal=q_max,
            edge_at_q1=edge_q1,
            edge_at_q5=edge_q5,
            edge_at_q10=edge_q10,
            vwap_cost_q1=cost_q1,
            vwap_cost_q5=cost_q5,
            vwap_cost_q10=cost_q10,
            q_max_after_0_5s=evolution['q_max_after_0_5s'],
            q_max_after_1s=evolution['q_max_after_1s'],
            q_max_after_2s=evolution['q_max_after_2s'],
            q_max_after_5s=evolution['q_max_after_5s'],
            edge_persisted_0_5s=evolution['edge_persisted_0_5s'],
            edge_persisted_1s=evolution['edge_persisted_1s'],
            edge_persisted_2s=evolution['edge_persisted_2s'],
            edge_persisted_5s=evolution['edge_persisted_5s'],
            edge_disappeared_at=evolution['edge_disappeared_at'],
            edge_q1_after_0_5s=evolution['edge_q1_after_0_5s'],
            edge_q1_after_1s=evolution['edge_q1_after_1s'],
            edge_q1_after_2s=evolution['edge_q1_after_2s'],
            edge_q1_after_5s=evolution['edge_q1_after_5s']
        )
        signals.append(signal)
    
    return signals


def analyze_shadow_results(signals: List[DepthSignalLog]) -> Dict:
    """Analyze shadow trader results."""
    if not signals:
        return {'error': 'No signals'}
    
    n_signals = len(signals)
    
    # Executability metrics
    exec_at_q1 = sum(1 for s in signals if s.q_max_at_signal >= 1) / n_signals * 100
    exec_at_q5 = sum(1 for s in signals if s.q_max_at_signal >= 5) / n_signals * 100
    exec_at_q10 = sum(1 for s in signals if s.q_max_at_signal >= 10) / n_signals * 100
    
    # Capacity metrics
    q_max_values = [s.q_max_at_signal for s in signals]
    q_max_median = float(np.median(q_max_values))
    q_max_mean = float(np.mean(q_max_values))
    
    # Edge metrics
    edge_q1_values = [s.edge_at_q1 for s in signals]
    edge_q1_median = float(np.median(edge_q1_values))
    edge_q1_mean = float(np.mean(edge_q1_values))
    
    # Persistence metrics
    persisted_0_5s = sum(1 for s in signals if s.edge_persisted_0_5s) / n_signals * 100
    persisted_1s = sum(1 for s in signals if s.edge_persisted_1s) / n_signals * 100
    persisted_2s = sum(1 for s in signals if s.edge_persisted_2s) / n_signals * 100
    persisted_5s = sum(1 for s in signals if s.edge_persisted_5s) / n_signals * 100
    
    # Capacity decay
    q_max_at_signal = np.mean([s.q_max_at_signal for s in signals])
    q_max_after_0_5s = np.mean([s.q_max_after_0_5s for s in signals if s.q_max_after_0_5s > 0])
    q_max_after_1s = np.mean([s.q_max_after_1s for s in signals if s.q_max_after_1s > 0])
    q_max_after_2s = np.mean([s.q_max_after_2s for s in signals if s.q_max_after_2s > 0])
    q_max_after_5s = np.mean([s.q_max_after_5s for s in signals if s.q_max_after_5s > 0])
    
    return {
        'n_signals': n_signals,
        'executability': {
            'pct_executable_q1': exec_at_q1,
            'pct_executable_q5': exec_at_q5,
            'pct_executable_q10': exec_at_q10,
        },
        'capacity': {
            'q_max_median': q_max_median,
            'q_max_mean': q_max_mean,
        },
        'edge': {
            'edge_q1_median': edge_q1_median,
            'edge_q1_mean': edge_q1_mean,
        },
        'persistence': {
            'pct_persisted_0_5s': persisted_0_5s,
            'pct_persisted_1s': persisted_1s,
            'pct_persisted_2s': persisted_2s,
            'pct_persisted_5s': persisted_5s,
        },
        'capacity_decay': {
            'q_max_at_signal': float(q_max_at_signal),
            'q_max_after_0_5s': float(q_max_after_0_5s) if not np.isnan(q_max_after_0_5s) else 0,
            'q_max_after_1s': float(q_max_after_1s) if not np.isnan(q_max_after_1s) else 0,
            'q_max_after_2s': float(q_max_after_2s) if not np.isnan(q_max_after_2s) else 0,
            'q_max_after_5s': float(q_max_after_5s) if not np.isnan(q_max_after_5s) else 0,
        }
    }


def generate_shadow_report(signals: List[DepthSignalLog], analysis: Dict) -> str:
    """Generate shadow trader report."""
    report = []
    report.append("# Shadow Trader with Depth Logging - Report\n\n")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report.append(f"**Total Signals Observed**: {analysis['n_signals']}\n\n")
    
    report.append("---\n\n")
    
    # Executability Dashboard
    report.append("## Executability Dashboard\n\n")
    report.append("| Size | % Executable |\n")
    report.append("|------|-------------|\n")
    report.append(f"| q >= 1 | {analysis['executability']['pct_executable_q1']:.1f}% |\n")
    report.append(f"| q >= 5 | {analysis['executability']['pct_executable_q5']:.1f}% |\n")
    report.append(f"| q >= 10 | {analysis['executability']['pct_executable_q10']:.1f}% |\n\n")
    
    # Capacity metrics
    report.append("## Capacity Metrics\n\n")
    report.append(f"- **Median q_max**: {analysis['capacity']['q_max_median']:.1f} contracts\n")
    report.append(f"- **Mean q_max**: {analysis['capacity']['q_max_mean']:.1f} contracts\n\n")
    
    # Edge metrics
    report.append("## Edge Metrics\n\n")
    report.append(f"- **Median edge at q=1**: ${analysis['edge']['edge_q1_median']:.4f}\n")
    report.append(f"- **Mean edge at q=1**: ${analysis['edge']['edge_q1_mean']:.4f}\n\n")
    
    # Persistence
    report.append("---\n\n")
    report.append("## Edge Persistence\n\n")
    report.append("How long does the arbitrage opportunity persist after detection?\n\n")
    report.append("| Time After Signal | % Still Executable |\n")
    report.append("|-------------------|-------------------|\n")
    report.append(f"| +0.5 second | {analysis['persistence']['pct_persisted_0_5s']:.1f}% |\n")
    report.append(f"| +1 second | {analysis['persistence']['pct_persisted_1s']:.1f}% |\n")
    report.append(f"| +2 seconds | {analysis['persistence']['pct_persisted_2s']:.1f}% |\n")
    report.append(f"| +5 seconds | {analysis['persistence']['pct_persisted_5s']:.1f}% |\n\n")
    
    # Capacity decay
    report.append("## Capacity Over Time\n\n")
    report.append("How does available capacity change after signal?\n\n")
    report.append("| Time | Avg q_max |\n")
    report.append("|------|-----------|\n")
    report.append(f"| At signal | {analysis['capacity_decay']['q_max_at_signal']:.1f} |\n")
    report.append(f"| +0.5s | {analysis['capacity_decay']['q_max_after_0_5s']:.1f} |\n")
    report.append(f"| +1s | {analysis['capacity_decay']['q_max_after_1s']:.1f} |\n")
    report.append(f"| +2s | {analysis['capacity_decay']['q_max_after_2s']:.1f} |\n")
    report.append(f"| +5s | {analysis['capacity_decay']['q_max_after_5s']:.1f} |\n\n")
    
    # Go/No-Go
    report.append("---\n\n")
    report.append("## Go/No-Go Decision\n\n")
    
    criteria = []
    
    # Check executability
    if analysis['executability']['pct_executable_q1'] >= 80:
        criteria.append("[PASS] >80% executable at q=1")
        report.append(f"- **Executability at q=1**: {analysis['executability']['pct_executable_q1']:.1f}% [PASS]\n")
    else:
        criteria.append("[FAIL] <80% executable at q=1")
        report.append(f"- **Executability at q=1**: {analysis['executability']['pct_executable_q1']:.1f}% [FAIL]\n")
    
    # Check edge
    if analysis['edge']['edge_q1_median'] >= 0.005:
        criteria.append("[PASS] Median edge >= $0.005")
        report.append(f"- **Median edge**: ${analysis['edge']['edge_q1_median']:.4f} [PASS]\n")
    else:
        criteria.append("[FAIL] Median edge < $0.005")
        report.append(f"- **Median edge**: ${analysis['edge']['edge_q1_median']:.4f} [FAIL]\n")
    
    # Check persistence
    if analysis['persistence']['pct_persisted_1s'] >= 50:
        criteria.append("[PASS] >50% persist at +1s")
        report.append(f"- **Persistence at +1s**: {analysis['persistence']['pct_persisted_1s']:.1f}% [PASS]\n")
    else:
        criteria.append("[WARN] <50% persist at +1s")
        report.append(f"- **Persistence at +1s**: {analysis['persistence']['pct_persisted_1s']:.1f}% [WARN]\n")
    
    report.append("\n### Verdict\n\n")
    
    if all("[PASS]" in c for c in criteria):
        report.append("**GO FOR SMALL CAPITAL DEPLOYMENT**\n\n")
        report.append("All criteria met. Strategy is ready for paper trading with real orders.\n\n")
        report.append("Recommended:\n")
        report.append("- Start with q=1 per signal\n")
        report.append("- Use taker execution for reliability\n")
        report.append("- Track actual fill rates vs simulated\n")
    elif "[FAIL]" in str(criteria):
        report.append("**NO-GO - Criteria not met**\n\n")
        for c in criteria:
            if "[FAIL]" in c:
                report.append(f"- {c}\n")
    else:
        report.append("**PROCEED WITH CAUTION**\n\n")
        report.append("Core criteria met but some warnings:\n")
        for c in criteria:
            if "[WARN]" in c:
                report.append(f"- {c}\n")
    
    return ''.join(report)


def main():
    print("=" * 70)
    print("Shadow Trader with Depth Logging")
    print("=" * 70)
    
    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Parameters
    params = {
        'epsilon': EPSILON,
        'tau_min': TAU_MIN,
        'tau_max': TAU_MAX,
        'slippage_buffer': SLIPPAGE_BUFFER,
        'cooldown': COOLDOWN_SECONDS,
        'market_duration': 900,
    }
    
    print(f"\nParameters:")
    print(f"  Epsilon: {params['epsilon']}")
    print(f"  Tau window: [{params['tau_min']}, {params['tau_max']}]")
    print(f"  Slippage buffer: {params['slippage_buffer']}")
    print(f"  Max signals: {MAX_SIGNALS}")
    
    # Find markets
    if not MARKETS_DIR.exists():
        print(f"ERROR: Markets directory not found: {MARKETS_DIR}")
        return
    
    market_dirs = [d for d in MARKETS_DIR.iterdir() if d.is_dir()]
    print(f"\n  Found {len(market_dirs)} market directories")
    
    # Run shadow trader
    print("\nRunning shadow trader...")
    all_signals = []
    
    for i, market_dir in enumerate(market_dirs):
        if len(all_signals) >= MAX_SIGNALS:
            break
        
        signals = run_shadow_trader_for_market(market_dir, params)
        all_signals.extend(signals)
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(market_dirs)}] Processed, {len(all_signals)} signals so far")
    
    # Limit to MAX_SIGNALS
    all_signals = all_signals[:MAX_SIGNALS]
    print(f"\n  Total signals: {len(all_signals)}")
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = analyze_shadow_results(all_signals)
    
    # Generate report
    print("Generating report...")
    report = generate_shadow_report(all_signals, analysis)
    
    # Save outputs
    print("\n" + "=" * 70)
    print("Saving outputs...")
    
    # Convert signals to DataFrame for parquet
    signals_data = []
    for s in all_signals:
        d = asdict(s)
        # Flatten depth snapshot
        d['up_ask_prices'] = str(d['depth_snapshot']['up_ask_prices'])
        d['up_ask_sizes'] = str(d['depth_snapshot']['up_ask_sizes'])
        d['down_ask_prices'] = str(d['depth_snapshot']['down_ask_prices'])
        d['down_ask_sizes'] = str(d['depth_snapshot']['down_ask_sizes'])
        del d['depth_snapshot']
        signals_data.append(d)
    
    df_signals = pd.DataFrame(signals_data)
    
    # Save parquet
    parquet_path = RESULTS_DIR / "shadow_depth_log.parquet"
    df_signals.to_parquet(parquet_path, index=False)
    print(f"  Parquet saved to: {parquet_path}")
    
    # Save JSON analysis
    json_path = RESULTS_DIR / "shadow_depth_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"  Analysis saved to: {json_path}")
    
    # Save report
    report_path = REPORTS_DIR / "shadow_depth_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SHADOW TRADER SUMMARY")
    print("=" * 70)
    print(f"\n  Signals observed: {analysis['n_signals']}")
    print(f"\n  Executability:")
    print(f"    q >= 1: {analysis['executability']['pct_executable_q1']:.1f}%")
    print(f"    q >= 5: {analysis['executability']['pct_executable_q5']:.1f}%")
    print(f"    q >= 10: {analysis['executability']['pct_executable_q10']:.1f}%")
    print(f"\n  Capacity:")
    print(f"    Median q_max: {analysis['capacity']['q_max_median']:.1f}")
    print(f"    Mean q_max: {analysis['capacity']['q_max_mean']:.1f}")
    print(f"\n  Edge:")
    print(f"    Median at q=1: ${analysis['edge']['edge_q1_median']:.4f}")
    print(f"\n  Persistence:")
    print(f"    +1s: {analysis['persistence']['pct_persisted_1s']:.1f}%")
    print(f"    +2s: {analysis['persistence']['pct_persisted_2s']:.1f}%")
    print(f"    +5s: {analysis['persistence']['pct_persisted_5s']:.1f}%")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

