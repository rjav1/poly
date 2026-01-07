#!/usr/bin/env python3
"""
OOS Test Audit

Verify the OOS test was conducted correctly and diagnose why test PnL = 0.

Key questions:
1. Were there ANY signals/trades in the test set?
2. What's the distribution of |delta_bps| in train vs test?
3. Were there opportunities in the test set at all?
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
    print("OOS TEST AUDIT")
    print("=" * 70)
    print(f"\nFrozen params: tau_max={TAU_MAX}, delta={DELTA_THRESHOLD_BPS}bps, hold={HOLD_SECONDS}s")
    
    # Load data
    print("\nLoading data...")
    df, _ = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    # Split exactly as OOS validation does
    market_starts = df.groupby('market_id')['market_start'].first().sort_values()
    markets_sorted = market_starts.index.tolist()
    n_train = int(len(markets_sorted) * 0.67)
    
    train_markets = set(markets_sorted[:n_train])
    test_markets = set(markets_sorted[n_train:])
    
    train_df = df[df['market_id'].isin(train_markets)].copy()
    test_df = df[df['market_id'].isin(test_markets)].copy()
    
    print(f"\nTrain markets: {len(train_markets)}")
    print(f"Test markets: {len(test_markets)}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    
    # List the market IDs
    print("\nTrain market IDs:")
    for mid in sorted(train_markets):
        print(f"  {mid}")
    print("\nTest market IDs:")
    for mid in sorted(test_markets):
        print(f"  {mid}")
    
    # Check delta_bps distribution in late window
    print("\n" + "=" * 60)
    print("ANALYSIS 1: delta_bps in late window (tau <= 420)")
    print("=" * 60)
    
    train_late = train_df[train_df['tau'] <= TAU_MAX]
    test_late = test_df[test_df['tau'] <= TAU_MAX]
    
    print(f"\nTrain late-window rows: {len(train_late):,}")
    print(f"Test late-window rows: {len(test_late):,}")
    
    print("\n--- Train |delta_bps| distribution ---")
    print(train_late['delta_bps'].abs().describe())
    
    print("\n--- Test |delta_bps| distribution ---")
    print(test_late['delta_bps'].abs().describe())
    
    # Check how many opportunities (|delta_bps| >= 10)
    train_opps = train_late[train_late['delta_bps'].abs() >= DELTA_THRESHOLD_BPS]
    test_opps = test_late[test_late['delta_bps'].abs() >= DELTA_THRESHOLD_BPS]
    
    print(f"\nOpportunities (|delta| >= {DELTA_THRESHOLD_BPS}bps in tau<={TAU_MAX}):")
    print(f"  Train: {len(train_opps):,} rows ({len(train_opps)/len(train_late)*100:.1f}% of late window)")
    print(f"  Test:  {len(test_opps):,} rows ({len(test_opps)/len(test_late)*100:.1f}% of late window)")
    
    # Per-market opportunity count
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Per-market opportunity count")
    print("=" * 60)
    
    print("\nTrain markets:")
    for mid in sorted(train_markets):
        m_late = train_df[(train_df['market_id'] == mid) & (train_df['tau'] <= TAU_MAX)]
        m_opps = m_late[m_late['delta_bps'].abs() >= DELTA_THRESHOLD_BPS]
        print(f"  {mid}: {len(m_opps)} opportunities")
    
    print("\nTest markets:")
    for mid in sorted(test_markets):
        m_late = test_df[(test_df['market_id'] == mid) & (test_df['tau'] <= TAU_MAX)]
        m_opps = m_late[m_late['delta_bps'].abs() >= DELTA_THRESHOLD_BPS]
        print(f"  {mid}: {len(m_opps)} opportunities")
    
    # Run strategy and count signals/trades
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Strategy Signals & Trades")
    print("=" * 60)
    
    strategy = LateDirectionalTakerStrategy(
        tau_max=TAU_MAX,
        delta_threshold_bps=DELTA_THRESHOLD_BPS,
        hold_seconds=HOLD_SECONDS,
    )
    config = ExecutionConfig()
    
    print(f"\nStrategy: {strategy.name}")
    
    # Generate signals manually to count them
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
    
    # Run full backtest
    train_result = run_backtest(train_df, strategy, config)
    test_result = run_backtest(test_df, strategy, config)
    
    print(f"\nBacktest results:")
    print(f"  Train: n_trades={train_result['metrics']['n_trades']}, "
          f"n_markets={train_result['metrics']['n_markets']}, "
          f"PnL=${train_result['metrics']['total_pnl']:.2f}, "
          f"t={train_result['metrics']['t_stat']:.2f}")
    print(f"  Test:  n_trades={test_result['metrics']['n_trades']}, "
          f"n_markets={test_result['metrics']['n_markets']}, "
          f"PnL=${test_result['metrics']['total_pnl']:.2f}, "
          f"t={test_result['metrics']['t_stat']:.2f}")
    
    # DIAGNOSIS
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    if len(test_signals) == 0:
        print("\n*** CRITICAL: TEST SET HAS ZERO SIGNALS ***")
        print("The OOS failure is INCONCLUSIVE - there were no trading opportunities.")
        print("This could mean:")
        print("  1. Test markets had lower volatility (CL stayed near strike)")
        print("  2. Time-of-day/regime difference between train and test periods")
        print("  3. The 12 test markets are fundamentally different from train")
    elif test_result['metrics']['n_trades'] == 0:
        print("\n*** CRITICAL: SIGNALS GENERATED BUT NO TRADES EXECUTED ***")
        print("Check execution model for issues with signal->trade conversion.")
    else:
        print(f"\nTest had {test_result['metrics']['n_trades']} trades but PnL = ${test_result['metrics']['total_pnl']:.2f}")
        print("The OOS failure is REAL - strategy lost/broke even on test data.")
    
    # Save audit results
    audit_results = {
        'train_markets': len(train_markets),
        'test_markets': len(test_markets),
        'train_late_rows': len(train_late),
        'test_late_rows': len(test_late),
        'train_opportunities': len(train_opps),
        'test_opportunities': len(test_opps),
        'train_signals': len(train_signals),
        'test_signals': len(test_signals),
        'train_trades': train_result['metrics']['n_trades'],
        'test_trades': test_result['metrics']['n_trades'],
        'train_pnl': train_result['metrics']['total_pnl'],
        'test_pnl': test_result['metrics']['total_pnl'],
        'train_t_stat': train_result['metrics']['t_stat'],
        'test_t_stat': test_result['metrics']['t_stat'],
        'diagnosis': 'INCONCLUSIVE - zero signals' if len(test_signals) == 0 else 
                     'INCONCLUSIVE - signals but no trades' if test_result['metrics']['n_trades'] == 0 else
                     'REAL FAILURE',
    }
    
    with open(OUTPUT_DIR / 'oos_audit_results.json', 'w') as f:
        json.dump(audit_results, f, indent=2)
    
    print(f"\nAudit results saved to results/oos_audit_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()

