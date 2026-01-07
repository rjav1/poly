#!/usr/bin/env python3
"""
Sensitivity sweep for depth-aware complete-set arb:
- Sweep epsilon in {0.004, 0.005, 0.006}
- Sweep slippage_buffer in {0.003, 0.005, 0.007}

Outputs:
- strategy_derivation/results/depth_sensitivity.json
- strategy_derivation/reports/depth_sensitivity.md
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from importlib.util import spec_from_file_location, module_from_spec

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
MARKETS_DIR = BASE_DIR.parent / "data_v2" / "markets_6levels" / "ETH"

# Import execution math
exec_math_path = Path(__file__).parent / "28_execution_math.py"
spec = spec_from_file_location("execution_math_sens", exec_math_path)
execution_math_sens = module_from_spec(spec)
spec.loader.exec_module(execution_math_sens)

extract_ladder_from_row = execution_math_sens.extract_ladder_from_row
max_executable_size = execution_math_sens.max_executable_size
compute_pnl_at_size = execution_math_sens.compute_pnl_at_size

EPSILONS = [0.004, 0.005, 0.006]
BUFFERS = [0.003, 0.005, 0.007]
SIZES = [1, 10]


def gather_signal_rows() -> pd.DataFrame:
    """Gather candidate rows across all markets (tau in [600,900]) with simple L1 underround screen."""
    rows = []
    for market_dir in MARKETS_DIR.iterdir():
        if not market_dir.is_dir():
            continue
        pm_path = market_dir / "polymarket.csv"
        if not pm_path.exists():
            continue
        df = pd.read_csv(pm_path)
        if 't' not in df.columns:
            df['t'] = range(len(df))
        # tau = 900 - t
        df['tau'] = 900 - df['t']
        df = df[(df['tau'] >= 600) & (df['tau'] <= 900)]
        for _, row in df.iterrows():
            r = row.to_dict()
            up_best = r.get('up_best_ask', np.nan)
            down_best = r.get('down_best_ask', np.nan)
            if not pd.isna(up_best) and not pd.isna(down_best):
                rows.append({
                    'market_id': market_dir.name,
                    't': int(r['t']),
                    'row': r
                })
    return pd.DataFrame(rows)


def evaluate_combo(rows_df: pd.DataFrame, epsilon: float, buffer: float) -> Dict:
    """Evaluate a single (epsilon, buffer) combo."""
    exec_stats = {q: {'count': 0, 'exec': 0, 'edges': [], 'pnls': []} for q in SIZES}
    for _, r in rows_df.iterrows():
        row = r['row']
        up_asks = extract_ladder_from_row(row, 'up', 'ask')
        down_asks = extract_ladder_from_row(row, 'down', 'ask')
        l1_underround = 1.0 - (up_asks.best_price + down_asks.best_price)
        if l1_underround < epsilon:
            continue
        # Exec per size
        for q in SIZES:
            exec_stats[q]['count'] += 1
            m = compute_pnl_at_size(up_asks, down_asks, q, buffer)
            if m['executable']:
                exec_stats[q]['exec'] += 1
                exec_stats[q]['edges'].append(m['edge_per_set'])
                exec_stats[q]['pnls'].append(m['edge_per_set'] * q)
    # Aggregate
    out = {}
    for q in SIZES:
        cnt = exec_stats[q]['count'] or 1
        exec_pct = exec_stats[q]['exec'] / cnt * 100.0
        avg_edge = float(np.mean(exec_stats[q]['edges'])) if exec_stats[q]['edges'] else 0.0
        exp_pnl = float(np.mean(exec_stats[q]['pnls'])) if exec_stats[q]['pnls'] else 0.0
        out[f'q{q}'] = {'exec_pct': exec_pct, 'avg_edge': avg_edge, 'expected_pnl': exp_pnl}
    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Collecting rows...")
    rows_df = gather_signal_rows()
    print(f"Rows collected for sweep: {len(rows_df)}")

    results = {}
    for eps in EPSILONS:
        for buf in BUFFERS:
            key = f'eps_{eps:.3f}_buf_{buf:.3f}'
            results[key] = evaluate_combo(rows_df, eps, buf)
            print(f"Evaluated {key}")

    # Save JSON
    json_path = RESULTS_DIR / "depth_sensitivity.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")

    # Save report
    md = []
    md.append("# Depth Sensitivity Report\n\n")
    md.append("| Epsilon | Buffer | q | Exec % | Avg Edge | Exp PnL |\n")
    md.append("|---------|--------|---|--------:|---------:|--------:|\n")
    for eps in EPSILONS:
        for buf in BUFFERS:
            key = f'eps_{eps:.3f}_buf_{buf:.3f}'
            res = results[key]
            for q in SIZES:
                qk = f'q{q}'
                md.append(f"| {eps:.3f} | {buf:.3f} | {q} | {res[qk]['exec_pct']:.1f}% | ${res[qk]['avg_edge']:.4f} | ${res[qk]['expected_pnl']:.4f} |\n")
    report_path = REPORTS_DIR / "depth_sensitivity.md"
    with open(report_path, 'w') as f:
        f.write(''.join(md))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()


