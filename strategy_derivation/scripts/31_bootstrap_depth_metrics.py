#!/usr/bin/env python3
"""
Bootstrap depth-aware metrics by market (block bootstrap).

Computes confidence intervals for:
- avg edge per set at q = {1,5,10,20}
- expected PnL per signal at q = {1,5,10,20}
- executable fraction at q = {1,5,10,20}

Resamples markets with replacement; for each resample, includes all signals from selected markets.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Import execution math
import sys
sys.path.insert(0, str(Path(__file__).parent))
from importlib.util import spec_from_file_location, module_from_spec

exec_math_path = Path(__file__).parent / "28_execution_math.py"
spec = spec_from_file_location("execution_math_boot", exec_math_path)
execution_math_boot = module_from_spec(spec)
spec.loader.exec_module(execution_math_boot)

extract_ladder_from_row = execution_math_boot.extract_ladder_from_row
compute_pnl_at_size = execution_math_boot.compute_pnl_at_size

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
MARKETS_DIR = BASE_DIR.parent / "data_v2" / "markets_6levels" / "ETH"

N_BOOT = 1000
SIZES = [1, 5, 10, 20]
SLIPPAGE_BUFFER = 0.005


def load_signals() -> pd.DataFrame:
    """Load signal list from previous depth backtest outputs."""
    path = RESULTS_DIR / "complete_set_depth_signals.json"
    df = pd.read_json(path)
    return df[['market_id', 't']]


def compute_metrics_for_signals(signal_df: pd.DataFrame) -> Dict:
    """Compute metrics at sizes for a given set of signals."""
    metrics = {q: {'edges': [], 'pnls': [], 'exec_flags': []} for q in SIZES}

    for market_id, df_group in signal_df.groupby('market_id'):
        pm_path = (MARKETS_DIR / str(market_id) / 'polymarket.csv')
        if not pm_path.exists():
            continue
        mdf = pd.read_csv(pm_path)
        if 't' not in mdf.columns:
            mdf['t'] = range(len(mdf))
        mdf_index = mdf.set_index('t')

        for t_val in df_group['t']:
            if t_val not in mdf_index.index:
                continue
            row = mdf_index.loc[t_val].to_dict()
            up_asks = extract_ladder_from_row(row, 'up', 'ask')
            down_asks = extract_ladder_from_row(row, 'down', 'ask')
            for q in SIZES:
                m = compute_pnl_at_size(up_asks, down_asks, q, SLIPPAGE_BUFFER)
                metrics[q]['exec_flags'].append(1 if m['executable'] else 0)
                if m['executable']:
                    metrics[q]['edges'].append(m['edge_per_set'])
                    metrics[q]['pnls'].append(m['edge_per_set'] * q)

    # Aggregate
    out = {}
    for q in SIZES:
        exec_rate = np.mean(metrics[q]['exec_flags']) if metrics[q]['exec_flags'] else 0.0
        avg_edge = np.mean(metrics[q]['edges']) if metrics[q]['edges'] else 0.0
        exp_pnl = np.mean(metrics[q]['pnls']) if metrics[q]['pnls'] else 0.0
        out[q] = {'exec_rate': float(exec_rate), 'avg_edge': float(avg_edge), 'expected_pnl': float(exp_pnl)}
    return out


def bootstrap() -> Dict:
    """Run block bootstrap by market."""
    sigs = load_signals()
    markets = sigs['market_id'].unique().tolist()
    results = {q: {'exec_rate': [], 'avg_edge': [], 'expected_pnl': []} for q in SIZES}

    rng = np.random.default_rng(42)

    for _ in range(N_BOOT):
        # Resample markets with replacement
        sampled = rng.choice(markets, size=len(markets), replace=True)
        boot_df = pd.concat([sigs[sigs['market_id'] == m] for m in sampled], ignore_index=True)
        metrics = compute_metrics_for_signals(boot_df)
        for q in SIZES:
            results[q]['exec_rate'].append(metrics[q]['exec_rate'])
            results[q]['avg_edge'].append(metrics[q]['avg_edge'])
            results[q]['expected_pnl'].append(metrics[q]['expected_pnl'])

    # Summarize percentiles
    summary = {}
    for q in SIZES:
        summary[q] = {
            'exec_rate': {
                'p2_5': float(np.percentile(results[q]['exec_rate'], 2.5)),
                'p50': float(np.percentile(results[q]['exec_rate'], 50)),
                'p97_5': float(np.percentile(results[q]['exec_rate'], 97.5)),
            },
            'avg_edge': {
                'p2_5': float(np.percentile(results[q]['avg_edge'], 2.5)),
                'p50': float(np.percentile(results[q]['avg_edge'], 50)),
                'p97_5': float(np.percentile(results[q]['avg_edge'], 97.5)),
            },
            'expected_pnl': {
                'p2_5': float(np.percentile(results[q]['expected_pnl'], 2.5)),
                'p50': float(np.percentile(results[q]['expected_pnl'], 50)),
                'p97_5': float(np.percentile(results[q]['expected_pnl'], 97.5)),
            },
        }
    return summary


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Running bootstrap...")
    summary = bootstrap()
    out_path = RESULTS_DIR / 'depth_bootstrap.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()

{
  "cells": [],
  "metadata": {
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 2
}