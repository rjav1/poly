#!/usr/bin/env python3
"""
Final OOS Validation with ALL Fixed Markets

Uses all 43 ETH markets from the rebuilt canonical dataset.
"""

import sys
import pandas as pd
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest.data_loader import add_derived_columns
from scripts.backtest.backtest_engine import run_backtest, ExecutionConfig
from scripts.backtest.strategies import LateDirectionalTakerStrategy

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "results"

# FROZEN PARAMS
TAU_MAX = 420
DELTA_THRESHOLD_BPS = 10
HOLD_SECONDS = 120


def main():
    print("=" * 70)
    print("FINAL OOS VALIDATION (ALL FIXED MARKETS)")
    print("=" * 70)
    print(f"\nFrozen params: tau_max={TAU_MAX}, delta={DELTA_THRESHOLD_BPS}bps, hold={HOLD_SECONDS}s")
    
    # Load ALL ETH markets from rebuilt dataset
    df = pd.read_parquet('data_v2/research/canonical_dataset_all_assets.parquet')
    df_eth = df[df['asset'] == 'ETH'].copy()
    df_eth = add_derived_columns(df_eth)
    
    print(f"\nAll ETH markets: {df_eth['market_id'].nunique()}")
    print(f"Total rows: {len(df_eth):,}")
    
    # Filter to markets with valid delta_bps (<10% null)
    valid_markets = []
    for mid in df_eth['market_id'].unique():
        mdf = df_eth[df_eth['market_id'] == mid]
        null_pct = mdf['delta_bps'].isna().mean()
        if null_pct < 0.1:
            valid_markets.append(mid)
    
    print(f"Valid markets (delta_bps <10% null): {len(valid_markets)}")
    
    df_valid = df_eth[df_eth['market_id'].isin(valid_markets)].copy()
    
    # 70/30 split by chronological order
    market_starts = df_valid.groupby('market_id')['market_start'].first().sort_values()
    markets_sorted = market_starts.index.tolist()
    
    n_train = int(len(markets_sorted) * 0.70)
    train_markets = set(markets_sorted[:n_train])
    test_markets = set(markets_sorted[n_train:])
    
    train_df = df_valid[df_valid['market_id'].isin(train_markets)].copy()
    test_df = df_valid[df_valid['market_id'].isin(test_markets)].copy()
    
    print(f"\nSplit (70/30):")
    print(f"  Train: {len(train_markets)} markets ({len(train_df):,} rows)")
    print(f"  Test:  {len(test_markets)} markets ({len(test_df):,} rows)")
    
    # Check opportunities
    train_late = train_df[train_df['tau'] <= TAU_MAX]
    test_late = test_df[test_df['tau'] <= TAU_MAX]
    
    train_opps = train_late[train_late['delta_bps'].abs() >= DELTA_THRESHOLD_BPS]
    test_opps = test_late[test_late['delta_bps'].abs() >= DELTA_THRESHOLD_BPS]
    
    print(f"\nOpportunities (|delta|>={DELTA_THRESHOLD_BPS}bps in tau<={TAU_MAX}):")
    print(f"  Train: {len(train_opps):,} rows")
    print(f"  Test:  {len(test_opps):,} rows")
    
    # Run strategy
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
    print(f"  Train: {len(train_signals)}")
    print(f"  Test:  {len(test_signals)}")
    
    # Run backtests
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    
    train_result = run_backtest(train_df, strategy, config)
    test_result = run_backtest(test_df, strategy, config)
    
    train_t = train_result['metrics']['t_stat']
    test_t = test_result['metrics']['t_stat']
    train_pnl = train_result['metrics']['total_pnl']
    test_pnl = test_result['metrics']['total_pnl']
    train_trades = train_result['metrics']['n_trades']
    test_trades = test_result['metrics']['n_trades']
    
    print(f"\nTrain: {train_trades} trades, PnL=${train_pnl:.2f}, t={train_t:.2f}")
    print(f"Test:  {test_trades} trades, PnL=${test_pnl:.2f}, t={test_t:.2f}")
    
    degradation = (train_t - test_t) / train_t * 100 if train_t > 0 else 0
    print(f"\nDegradation: {degradation:.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    results = {
        'total_eth_markets': df_eth['market_id'].nunique(),
        'valid_markets': len(valid_markets),
        'train_markets': len(train_markets),
        'test_markets': len(test_markets),
        'train_signals': len(train_signals),
        'test_signals': len(test_signals),
        'train_trades': train_trades,
        'test_trades': test_trades,
        'train_pnl': train_pnl,
        'test_pnl': test_pnl,
        'train_t_stat': train_t,
        'test_t_stat': test_t,
        'degradation_pct': degradation,
        'frozen_params': {
            'tau_max': TAU_MAX,
            'delta_threshold_bps': DELTA_THRESHOLD_BPS,
            'hold_seconds': HOLD_SECONDS,
        },
    }
    
    if test_t >= 2.0:
        results['verdict'] = 'STRONG - significant OOS edge'
        print(f"\nVERDICT: STRONG")
        print(f"Test t-stat ({test_t:.2f}) >= 2.0 threshold")
        print("Strategy maintains significant edge out-of-sample")
    elif test_t >= 1.5:
        results['verdict'] = 'MODERATE - marginal OOS edge'
        print(f"\nVERDICT: MODERATE")
        print(f"Test t-stat ({test_t:.2f}) is marginally significant")
    else:
        results['verdict'] = 'WEAK - no significant OOS edge'
        print(f"\nVERDICT: WEAK")
        print(f"Test t-stat ({test_t:.2f}) < 1.5 threshold")
        print("Strategy does NOT maintain significant edge out-of-sample")
    
    # Save results
    with open(OUTPUT_DIR / 'final_oos_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/final_oos_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()

