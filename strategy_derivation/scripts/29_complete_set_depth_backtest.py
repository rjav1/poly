#!/usr/bin/env python3
"""
Phase 4: Depth-Aware Complete-Set Arb Backtest

Re-runs the complete-set arbitrage strategy using 6-level orderbook depth:
- Replace L1 capacity check with max_executable_size()
- Use VWAP-based fill prices instead of best_ask
- Size signals at q = min(q_max, q_cap_limit)
- Track partial fills

Parameters:
- epsilon = 0.005 (from previous best region)
- tau_window = [600, 900] (from previous best region)
- slippage_buffer = 0.005 (conservative, 0.5c per leg)
- q_cap_limit = 10 (max contracts per signal)

Outputs:
- complete_set_depth_backtest_results.json
- complete_set_depth_backtest_report.md
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

# Import execution math
from importlib import import_module
import sys
sys.path.insert(0, str(Path(__file__).parent))
from importlib.util import spec_from_file_location, module_from_spec

# Load execution math module
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
EPSILON = 0.005  # Minimum underround (0.5%)
TAU_MIN = 600    # 10 minutes before expiry
TAU_MAX = 900    # 15 minutes (start of market)
SLIPPAGE_BUFFER = 0.005  # 0.5% total slippage
Q_CAP_LIMIT = 10  # Max contracts per signal
COOLDOWN_SECONDS = 30  # Cooldown between signals


@dataclass
class DepthSignal:
    """Signal with depth-aware execution metrics."""
    signal_id: int
    market_id: str
    t: int
    tau: int
    
    # Book snapshot (L1 only for logging)
    up_best_ask: float
    up_best_ask_size: float
    down_best_ask: float
    down_best_ask_size: float
    
    # Depth metrics
    up_total_ask_size: float
    down_total_ask_size: float
    
    # Execution metrics
    q_max_executable: float
    q_requested: float
    q_filled: float
    
    vwap_up: float
    vwap_down: float
    set_cost: float
    edge_per_set: float
    
    # PnL
    total_pnl: float
    executable: bool


def load_market_info() -> Dict:
    """Load market info with outcomes."""
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


def compute_tau(row: dict, market_duration: int = 900) -> int:
    """Compute tau (seconds to expiry) from row data."""
    # If we have a timestamp column, use it
    # Otherwise estimate from row position
    # For now, assume we have a 't' column or can derive it
    
    # Check for existing t column
    if 't' in row:
        t = int(row['t'])
        return market_duration - t
    
    # If no t, return None and we'll skip
    return None


def run_backtest_for_market(
    market_dir: Path,
    market_info: Dict,
    params: Dict
) -> List[DepthSignal]:
    """Run backtest for a single market."""
    df = load_market_data(market_dir)
    if df is None:
        return []
    
    market_id = market_dir.name
    info = market_info.get(market_id, {})
    
    # Add t column if missing (estimate from row index)
    if 't' not in df.columns:
        df['t'] = range(len(df))
    
    signals = []
    last_signal_t = -params['cooldown']
    signal_id = 0
    
    for idx, row in df.iterrows():
        t = int(row['t'])
        tau = params['market_duration'] - t
        
        # Check tau window
        if tau < params['tau_min'] or tau > params['tau_max']:
            continue
        
        # Check cooldown
        if t - last_signal_t < params['cooldown']:
            continue
        
        # Extract book ladders
        up_asks = extract_ladder_from_row(row, 'up', 'ask')
        down_asks = extract_ladder_from_row(row, 'down', 'ask')
        
        if not up_asks.is_valid() or not down_asks.is_valid():
            continue
        
        # Compute L1 underround first (quick check)
        l1_cost = up_asks.best_price + down_asks.best_price
        l1_underround = 1.0 - l1_cost
        
        if l1_underround < params['epsilon']:
            continue
        
        # Compute max executable size
        q_max, edge_at_max = max_executable_size(
            up_asks, down_asks,
            epsilon=params['epsilon'],
            slippage_buffer=params['slippage_buffer']
        )
        
        if q_max < 1.0:
            continue
        
        # Determine actual quantity to trade
        q_requested = min(q_max, params['q_cap_limit'])
        
        # Compute PnL at requested size
        pnl_metrics = compute_pnl_at_size(
            up_asks, down_asks,
            q_requested,
            params['slippage_buffer']
        )
        
        if not pnl_metrics['executable']:
            continue
        
        signal_id += 1
        last_signal_t = t
        
        signal = DepthSignal(
            signal_id=signal_id,
            market_id=market_id,
            t=t,
            tau=tau,
            up_best_ask=up_asks.best_price,
            up_best_ask_size=up_asks.sizes[0],
            down_best_ask=down_asks.best_price,
            down_best_ask_size=down_asks.sizes[0],
            up_total_ask_size=up_asks.total_size,
            down_total_ask_size=down_asks.total_size,
            q_max_executable=q_max,
            q_requested=q_requested,
            q_filled=pnl_metrics['q_filled'],
            vwap_up=pnl_metrics['vwap_up'],
            vwap_down=pnl_metrics['vwap_down'],
            set_cost=pnl_metrics['set_cost'],
            edge_per_set=pnl_metrics['edge_per_set'],
            total_pnl=pnl_metrics['total_pnl'],
            executable=pnl_metrics['executable']
        )
        signals.append(signal)
    
    return signals


def analyze_results(signals: List[DepthSignal]) -> Dict:
    """Analyze backtest results."""
    if not signals:
        return {'error': 'No signals generated'}
    
    # Convert to dataframe for analysis
    df = pd.DataFrame([asdict(s) for s in signals])
    
    # Basic stats
    n_signals = len(signals)
    n_markets = df['market_id'].nunique()
    
    # Capacity analysis
    q_max_values = df['q_max_executable']
    q_filled_values = df['q_filled']
    
    capacity_stats = {
        'q_max_p10': float(np.percentile(q_max_values, 10)),
        'q_max_p50': float(np.percentile(q_max_values, 50)),
        'q_max_p90': float(np.percentile(q_max_values, 90)),
        'q_max_mean': float(q_max_values.mean()),
        'q_filled_mean': float(q_filled_values.mean()),
    }
    
    # PnL analysis
    total_pnl = df['total_pnl'].sum()
    avg_pnl_per_signal = df['total_pnl'].mean()
    avg_edge_per_set = df['edge_per_set'].mean()
    
    pnl_stats = {
        'total_pnl': float(total_pnl),
        'avg_pnl_per_signal': float(avg_pnl_per_signal),
        'avg_edge_per_set': float(avg_edge_per_set),
        'min_pnl': float(df['total_pnl'].min()),
        'max_pnl': float(df['total_pnl'].max()),
    }
    
    # Executability at different sizes (q=1,5,10,20)
    exec_at_q = {}
    for q in [1, 5, 10, 20]:
        exec_count = (df['q_max_executable'] >= q).sum()
        exec_pct = exec_count / n_signals * 100
        exec_at_q[f'executable_at_q{q}'] = {
            'count': int(exec_count),
            'pct': float(exec_pct)
        }
    
    # Compute expected PnL and edge at different sizes using VWAP recomputation
    expected_pnl_by_size = {}
    avg_edge_by_size = {}
    # Recompute per-signal VWAP at sizes by revisiting market data rows
    # Build quick index: market_id -> list of (t)
    signals_index = {}
    for _, r in df.iterrows():
        signals_index.setdefault(r['market_id'], []).append(int(r['t']))
    from importlib.util import spec_from_file_location, module_from_spec
    exec_math_path = Path(__file__).parent / "28_execution_math.py"
    spec2 = spec_from_file_location("execution_math_aggr", exec_math_path)
    execution_math_aggr = module_from_spec(spec2)
    spec2.loader.exec_module(execution_math_aggr)
    extract_ladder_from_row_fn = execution_math_aggr.extract_ladder_from_row
    compute_pnl_at_size_fn = execution_math_aggr.compute_pnl_at_size

    # Gather per-q metrics
    per_q_edges = {1: [], 5: [], 10: [], 20: []}
    per_q_pnls = {1: [], 5: [], 10: [], 20: []}

    for market_id, ts in signals_index.items():
        pm_path = (MARKETS_DIR / market_id / "polymarket.csv")
        if not pm_path.exists():
            continue
        mdf = pd.read_csv(pm_path)
        if 't' not in mdf.columns:
            mdf['t'] = range(len(mdf))
        mdf_indexed = mdf.set_index('t')
        for t_val in ts:
            if t_val not in mdf_indexed.index:
                continue
            row = mdf_indexed.loc[t_val].to_dict()
            up_asks = extract_ladder_from_row_fn(row, 'up', 'ask')
            down_asks = extract_ladder_from_row_fn(row, 'down', 'ask')
            for q in [1, 5, 10, 20]:
                metrics = compute_pnl_at_size_fn(up_asks, down_asks, q, SLIPPAGE_BUFFER)
                if metrics['executable']:
                    per_q_edges[q].append(metrics['edge_per_set'])
                    per_q_pnls[q].append(metrics['edge_per_set'] * q)

    for q in [1, 5, 10, 20]:
        avg_edge_by_size[f'avg_edge_at_q{q}'] = float(np.mean(per_q_edges[q])) if per_q_edges[q] else 0.0
        expected_pnl_by_size[f'expected_pnl_at_q{q}'] = float(np.mean(per_q_pnls[q])) if per_q_pnls[q] else 0.0
    
    # VWAP analysis
    vwap_stats = {
        'avg_vwap_up': float(df['vwap_up'].mean()),
        'avg_vwap_down': float(df['vwap_down'].mean()),
        'avg_set_cost': float(df['set_cost'].mean()),
        'l1_set_cost_avg': float((df['up_best_ask'] + df['down_best_ask']).mean()),
        'slippage_from_l1': float(df['set_cost'].mean() - (df['up_best_ask'] + df['down_best_ask']).mean()),
    }
    
    return {
        'n_signals': n_signals,
        'n_markets': n_markets,
        'capacity': capacity_stats,
        'pnl': pnl_stats,
        'executability': exec_at_q,
        'expected_pnl_by_size': expected_pnl_by_size,
        'avg_edge_by_size': avg_edge_by_size,
        'vwap': vwap_stats,
    }


def compare_to_l1(signals: List[DepthSignal]) -> Dict:
    """Compare depth-aware results to L1-only baseline."""
    if not signals:
        return {}
    
    df = pd.DataFrame([asdict(s) for s in signals])
    
    # L1 capacity would be min(up_best_ask_size, down_best_ask_size)
    df['l1_capacity'] = df[['up_best_ask_size', 'down_best_ask_size']].min(axis=1)
    df['l1_executable'] = df['l1_capacity'] >= 1.0
    
    l1_executable_count = df['l1_executable'].sum()
    depth_executable_count = (df['q_max_executable'] >= 1).sum()
    
    return {
        'l1_executable_at_q1': int(l1_executable_count),
        'depth_executable_at_q1': int(depth_executable_count),
        'l1_executable_pct': float(l1_executable_count / len(df) * 100),
        'depth_executable_pct': float(depth_executable_count / len(df) * 100),
        'improvement_factor': float(depth_executable_count / l1_executable_count) if l1_executable_count > 0 else float('inf'),
        'avg_l1_capacity': float(df['l1_capacity'].mean()),
        'avg_depth_capacity': float(df['q_max_executable'].mean()),
    }


def generate_report(
    signals: List[DepthSignal],
    analysis: Dict,
    comparison: Dict,
    params: Dict
) -> str:
    """Generate backtest report."""
    report = []
    report.append("# Depth-Aware Complete-Set Arb Backtest Report\n\n")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    report.append("## Parameters\n\n")
    report.append(f"- **Epsilon (min underround)**: {params['epsilon']*100:.1f}%\n")
    report.append(f"- **Tau window**: [{params['tau_min']}, {params['tau_max']}] seconds\n")
    report.append(f"- **Slippage buffer**: {params['slippage_buffer']*100:.1f}%\n")
    report.append(f"- **Max position size**: {params['q_cap_limit']} contracts\n")
    report.append(f"- **Cooldown**: {params['cooldown']} seconds\n\n")
    
    report.append("---\n\n")
    
    report.append("## Summary\n\n")
    report.append(f"- **Total Signals**: {analysis['n_signals']}\n")
    report.append(f"- **Markets with Signals**: {analysis['n_markets']}\n")
    report.append(f"- **Total PnL**: ${analysis['pnl']['total_pnl']:.2f}\n")
    report.append(f"- **Avg PnL per Signal**: ${analysis['pnl']['avg_pnl_per_signal']:.4f}\n")
    report.append(f"- **Avg Edge per Set**: ${analysis['pnl']['avg_edge_per_set']:.4f}\n\n")
    
    report.append("---\n\n")
    
    report.append("## Capacity Analysis\n\n")
    report.append("Maximum executable size (contracts) at each signal:\n\n")
    report.append("| Percentile | q_max |\n")
    report.append("|------------|-------|\n")
    report.append(f"| p10 | {analysis['capacity']['q_max_p10']:.1f} |\n")
    report.append(f"| p50 (median) | {analysis['capacity']['q_max_p50']:.1f} |\n")
    report.append(f"| p90 | {analysis['capacity']['q_max_p90']:.1f} |\n")
    report.append(f"| Mean | {analysis['capacity']['q_max_mean']:.1f} |\n\n")
    
    report.append("---\n\n")
    
    report.append("## Executability by Size\n\n")
    report.append("| Target Size | Executable Signals | % |\n")
    report.append("|-------------|-------------------|---|\n")
    for q in [1, 5, 10, 20]:
        key = f'executable_at_q{q}'
        if key in analysis['executability']:
            data = analysis['executability'][key]
            report.append(f"| q={q} | {data['count']} | {data['pct']:.1f}% |\n")
    report.append("\n")
    
    report.append("---\n\n")
    
    report.append("## Expected PnL and Edge by Size\n\n")
    report.append("| Size | Avg Edge/Set | Expected PnL/Signal |\n")
    report.append("|------|---------------|--------------------|\n")
    for q in [1, 5, 10, 20]:
        key_p = f'expected_pnl_at_q{q}'
        key_e = f'avg_edge_at_q{q}'
        pnl = analysis['expected_pnl_by_size'].get(key_p, 0.0)
        edge = analysis.get('avg_edge_by_size', {}).get(key_e, 0.0)
        report.append(f"| q={q} | ${edge:.4f} | ${pnl:.4f} |\n")
    report.append("\n")
    
    report.append("---\n\n")
    
    report.append("## VWAP Analysis\n\n")
    report.append(f"- **Avg VWAP UP**: ${analysis['vwap']['avg_vwap_up']:.4f}\n")
    report.append(f"- **Avg VWAP DOWN**: ${analysis['vwap']['avg_vwap_down']:.4f}\n")
    report.append(f"- **Avg Set Cost (VWAP)**: ${analysis['vwap']['avg_set_cost']:.4f}\n")
    report.append(f"- **Avg Set Cost (L1 only)**: ${analysis['vwap']['l1_set_cost_avg']:.4f}\n")
    report.append(f"- **Slippage from L1**: ${analysis['vwap']['slippage_from_l1']:.4f}\n\n")
    
    report.append("---\n\n")
    
    report.append("## Comparison to L1-Only Baseline\n\n")
    if comparison:
        report.append("| Metric | L1-Only | Depth-Aware |\n")
        report.append("|--------|---------|-------------|\n")
        report.append(f"| Executable at q>=1 | {comparison['l1_executable_at_q1']} ({comparison['l1_executable_pct']:.1f}%) | {comparison['depth_executable_at_q1']} ({comparison['depth_executable_pct']:.1f}%) |\n")
        report.append(f"| Avg Capacity | {comparison['avg_l1_capacity']:.1f} | {comparison['avg_depth_capacity']:.1f} |\n")
        report.append(f"| Improvement Factor | - | {comparison['improvement_factor']:.2f}x |\n\n")
        
        if comparison['depth_executable_pct'] > 50:
            report.append("**[PASS] >50% of signals executable at q>=1 with depth-aware sizing**\n\n")
        else:
            report.append("**[WARN] <50% of signals executable - capacity still constrained**\n\n")
    
    report.append("---\n\n")
    
    report.append("## Go/No-Go Assessment\n\n")
    
    criteria = []
    
    # Check executability
    exec_pct = analysis['executability'].get('executable_at_q1', {}).get('pct', 0)
    if exec_pct >= 50:
        criteria.append("[PASS] Executability >= 50%")
        report.append(f"- **Executability**: {exec_pct:.1f}% >= 50% [PASS]\n")
    else:
        criteria.append("[FAIL] Executability < 50%")
        report.append(f"- **Executability**: {exec_pct:.1f}% < 50% [FAIL]\n")
    
    # Check edge
    avg_edge = analysis['pnl']['avg_edge_per_set']
    if avg_edge >= 0.01:
        criteria.append("[PASS] Avg edge >= $0.01")
        report.append(f"- **Avg Edge**: ${avg_edge:.4f} >= $0.01 [PASS]\n")
    else:
        criteria.append("[FAIL] Avg edge < $0.01")
        report.append(f"- **Avg Edge**: ${avg_edge:.4f} < $0.01 [FAIL]\n")
    
    # Check capacity improvement
    if comparison and comparison.get('improvement_factor', 0) >= 1.5:
        criteria.append("[PASS] Depth improves capacity >= 1.5x")
        report.append(f"- **Capacity Improvement**: {comparison['improvement_factor']:.2f}x >= 1.5x [PASS]\n")
    elif comparison:
        criteria.append("[INFO] Depth improvement < 1.5x")
        report.append(f"- **Capacity Improvement**: {comparison['improvement_factor']:.2f}x < 1.5x [INFO]\n")
    
    report.append("\n### Verdict\n\n")
    
    if all("[PASS]" in c or "[INFO]" in c for c in criteria):
        report.append("**GO - Strategy is executable with depth-aware sizing**\n\n")
        report.append("Recommended next steps:\n")
        report.append("1. Run shadow trader with depth logging\n")
        report.append("2. Validate fill rates in live conditions\n")
        report.append("3. Consider small capital deployment\n")
    else:
        report.append("**CAUTION - Some criteria not met**\n\n")
        for c in criteria:
            if "[FAIL]" in c:
                report.append(f"- {c}\n")
    
    return ''.join(report)


def main():
    print("=" * 70)
    print("Depth-Aware Complete-Set Arb Backtest")
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
        'q_cap_limit': Q_CAP_LIMIT,
        'cooldown': COOLDOWN_SECONDS,
        'market_duration': 900,
    }
    
    print(f"\nParameters:")
    print(f"  Epsilon: {params['epsilon']}")
    print(f"  Tau window: [{params['tau_min']}, {params['tau_max']}]")
    print(f"  Slippage buffer: {params['slippage_buffer']}")
    print(f"  Max position: {params['q_cap_limit']}")
    
    # Load market info
    print("\nLoading market info...")
    market_info = load_market_info()
    print(f"  Loaded info for {len(market_info)} markets")
    
    # Find markets
    if not MARKETS_DIR.exists():
        print(f"ERROR: Markets directory not found: {MARKETS_DIR}")
        return
    
    market_dirs = [d for d in MARKETS_DIR.iterdir() if d.is_dir()]
    print(f"  Found {len(market_dirs)} market directories")
    
    # Run backtest for each market
    print("\nRunning backtest...")
    all_signals = []
    
    for i, market_dir in enumerate(market_dirs):
        signals = run_backtest_for_market(market_dir, market_info, params)
        all_signals.extend(signals)
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(market_dirs)}] Processed, {len(all_signals)} signals so far")
    
    print(f"\n  Total signals: {len(all_signals)}")
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = analyze_results(all_signals)
    comparison = compare_to_l1(all_signals)
    
    # Generate report
    print("Generating report...")
    report = generate_report(all_signals, analysis, comparison, params)
    
    # Save outputs
    print("\n" + "=" * 70)
    print("Saving outputs...")
    
    # Save JSON results
    results = {
        'params': params,
        'analysis': analysis,
        'comparison': comparison,
        'n_signals': len(all_signals),
        'generated': datetime.now().isoformat(),
    }
    
    results_path = RESULTS_DIR / "complete_set_depth_backtest_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {results_path}")
    
    # Save detailed signals
    signals_path = RESULTS_DIR / "complete_set_depth_signals.json"
    with open(signals_path, 'w') as f:
        json.dump([asdict(s) for s in all_signals], f, indent=2)
    print(f"  Signals saved to: {signals_path}")
    
    # Save report
    report_path = REPORTS_DIR / "complete_set_depth_backtest_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)
    print(f"\n  Signals: {len(all_signals)}")
    print(f"  Markets: {analysis.get('n_markets', 0)}")
    
    if 'pnl' in analysis:
        print(f"\n  Total PnL: ${analysis['pnl']['total_pnl']:.2f}")
        print(f"  Avg PnL/Signal: ${analysis['pnl']['avg_pnl_per_signal']:.4f}")
        print(f"  Avg Edge/Set: ${analysis['pnl']['avg_edge_per_set']:.4f}")
    
    if 'capacity' in analysis:
        print(f"\n  Capacity p50: {analysis['capacity']['q_max_p50']:.1f} contracts")
        print(f"  Capacity p90: {analysis['capacity']['q_max_p90']:.1f} contracts")
    
    if comparison:
        print(f"\n  L1 executable: {comparison['l1_executable_pct']:.1f}%")
        print(f"  Depth executable: {comparison['depth_executable_pct']:.1f}%")
        print(f"  Improvement: {comparison['improvement_factor']:.2f}x")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

