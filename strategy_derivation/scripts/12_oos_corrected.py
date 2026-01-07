#!/usr/bin/env python3
"""
CORRECTED Out-of-Sample Validation

The original OOS test was invalid because the test markets had 100% null delta_bps.
This script:
1. Filters to only markets with valid delta_bps
2. Re-runs train/test split on valid data only
3. Reports proper OOS statistics

Frozen params from audit:
- tau_max = 420
- delta_threshold_bps = 10
- hold_seconds = 120
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest.data_loader import load_eth_markets, add_derived_columns
from scripts.backtest.backtest_engine import run_backtest, ExecutionConfig
from scripts.backtest.strategies import LateDirectionalTakerStrategy

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "results"

# FROZEN PARAMS from audit
TAU_MAX = 420
DELTA_THRESHOLD_BPS = 10
HOLD_SECONDS = 120


def main():
    print("=" * 70)
    print("CORRECTED OUT-OF-SAMPLE VALIDATION")
    print("=" * 70)
    print(f"\nFrozen params: tau_max={TAU_MAX}, delta={DELTA_THRESHOLD_BPS}bps, hold={HOLD_SECONDS}s")
    
    # Load data
    print("\nLoading data...")
    df, _ = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    # Filter to markets with valid delta_bps
    print("\n--- Filtering to valid markets ---")
    valid_markets = []
    for mid in df['market_id'].unique():
        mdf = df[df['market_id'] == mid]
        null_pct = mdf['delta_bps'].isna().mean() * 100
        if null_pct < 10:  # Less than 10% null
            valid_markets.append(mid)
        else:
            print(f"  Excluding {mid}: {null_pct:.0f}% null delta_bps")
    
    print(f"\nValid markets: {len(valid_markets)} / {df['market_id'].nunique()}")
    
    df_valid = df[df['market_id'].isin(valid_markets)].copy()
    
    # Sort by market_start and split
    market_starts = df_valid.groupby('market_id')['market_start'].first().sort_values()
    markets_sorted = market_starts.index.tolist()
    
    # 70/30 split
    n_train = int(len(markets_sorted) * 0.70)
    train_markets = set(markets_sorted[:n_train])
    test_markets = set(markets_sorted[n_train:])
    
    train_df = df_valid[df_valid['market_id'].isin(train_markets)].copy()
    test_df = df_valid[df_valid['market_id'].isin(test_markets)].copy()
    
    print(f"\nCorrected split:")
    print(f"  Train: {len(train_markets)} markets ({len(train_df):,} rows)")
    print(f"  Test:  {len(test_markets)} markets ({len(test_df):,} rows)")
    
    print("\nTrain markets:")
    for mid in sorted(train_markets):
        print(f"  {mid}")
    print("\nTest markets:")
    for mid in sorted(test_markets):
        print(f"  {mid}")
    
    # Verify both sets have delta_bps data
    print("\n--- Verifying data quality ---")
    train_late = train_df[train_df['tau'] <= TAU_MAX]
    test_late = test_df[test_df['tau'] <= TAU_MAX]
    
    train_opps = train_late[train_late['delta_bps'].abs() >= DELTA_THRESHOLD_BPS]
    test_opps = test_late[test_late['delta_bps'].abs() >= DELTA_THRESHOLD_BPS]
    
    print(f"Train opportunities: {len(train_opps):,} rows")
    print(f"Test opportunities:  {len(test_opps):,} rows")
    
    if len(test_opps) == 0:
        print("\nWARNING: Still no test opportunities. Lowering threshold to find any...")
        for thresh in [5, 2, 0]:
            test_opps_low = test_late[test_late['delta_bps'].abs() >= thresh]
            if len(test_opps_low) > 0:
                print(f"  At threshold {thresh}bps: {len(test_opps_low)} opportunities")
    
    # Run strategy
    print("\n" + "=" * 60)
    print("RUNNING CORRECTED OOS TEST")
    print("=" * 60)
    
    strategy = LateDirectionalTakerStrategy(
        tau_max=TAU_MAX,
        delta_threshold_bps=DELTA_THRESHOLD_BPS,
        hold_seconds=HOLD_SECONDS,
    )
    config = ExecutionConfig()
    
    # Count signals
    train_signals = []
    for mid in train_markets:
        m_df = train_df[train_df['market_id'] == mid]
        signals = strategy.generate_signals(m_df)
        train_signals.extend(signals)
    
    test_signals = []
    for mid in test_markets:
        m_df = test_df[test_df['market_id'] == mid]
        signals = strategy.generate_signals(m_df)
        test_signals.extend(signals)
    
    print(f"\nSignals generated:")
    print(f"  Train: {len(train_signals)} signals")
    print(f"  Test:  {len(test_signals)} signals")
    
    # Run backtests
    train_result = run_backtest(train_df, strategy, config)
    test_result = run_backtest(test_df, strategy, config)
    
    print(f"\nBacktest results:")
    print(f"  Train: n_trades={train_result['metrics']['n_trades']}, "
          f"PnL=${train_result['metrics']['total_pnl']:.2f}, "
          f"t={train_result['metrics']['t_stat']:.2f}")
    print(f"  Test:  n_trades={test_result['metrics']['n_trades']}, "
          f"PnL=${test_result['metrics']['total_pnl']:.2f}, "
          f"t={test_result['metrics']['t_stat']:.2f}")
    
    # Compute degradation
    train_t = train_result['metrics']['t_stat']
    test_t = test_result['metrics']['t_stat']
    
    if train_t > 0:
        degradation = (train_t - test_t) / train_t * 100
    else:
        degradation = 0
    
    print(f"\nDegradation: {degradation:.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("CORRECTED OOS SUMMARY")
    print("=" * 70)
    
    results = {
        'valid_markets': len(valid_markets),
        'excluded_markets': df['market_id'].nunique() - len(valid_markets),
        'train_markets': len(train_markets),
        'test_markets': len(test_markets),
        'train_signals': len(train_signals),
        'test_signals': len(test_signals),
        'train_trades': train_result['metrics']['n_trades'],
        'test_trades': test_result['metrics']['n_trades'],
        'train_pnl': train_result['metrics']['total_pnl'],
        'test_pnl': test_result['metrics']['total_pnl'],
        'train_t_stat': train_t,
        'test_t_stat': test_t,
        'degradation_pct': degradation,
        'frozen_params': {
            'tau_max': TAU_MAX,
            'delta_threshold_bps': DELTA_THRESHOLD_BPS,
            'hold_seconds': HOLD_SECONDS,
        },
    }
    
    if len(test_signals) == 0:
        results['conclusion'] = 'INCONCLUSIVE - no test signals even with valid data'
        print(f"\nCONCLUSION: INCONCLUSIVE")
        print("Even with valid data, test set has no signals.")
        print("This suggests the test period had fundamentally different market conditions.")
    elif degradation > 50:
        results['conclusion'] = 'WEAK - high OOS degradation'
        print(f"\nCONCLUSION: WEAK")
        print(f"Strategy degrades {degradation:.0f}% out-of-sample.")
    elif test_t < 1.5:
        results['conclusion'] = 'WEAK - test t-stat below 1.5'
        print(f"\nCONCLUSION: WEAK")
        print(f"Test t-stat ({test_t:.2f}) is below significance threshold.")
    else:
        results['conclusion'] = 'PROMISING - maintains edge OOS'
        print(f"\nCONCLUSION: PROMISING")
        print(f"Strategy maintains significant edge (t={test_t:.2f}) out-of-sample.")
    
    # Save results
    with open(OUTPUT_DIR / 'oos_corrected_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/oos_corrected_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()

