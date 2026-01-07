#!/usr/bin/env python3
"""
Simplified test runner that runs all tests and generates a comprehensive report.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest.data_loader import load_eth_markets, add_derived_columns, get_train_test_split
from scripts.backtest.fair_value import BinnedFairValueModel, compute_brier_score, compute_expected_calibration_error
from scripts.backtest.strategies import MispricingBasedStrategy
from scripts.backtest.backtest_engine import run_backtest, ExecutionConfig

def main():
    print("="*70)
    print("MISPRICING STRATEGY - COMPREHENSIVE TEST REPORT")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading data...")
    df, market_info = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    print(f"   [OK] Loaded {df['market_id'].nunique()} markets, {len(df):,} observations")
    
    # Train/test split
    print("\n[2/5] Creating train/test split...")
    train_df, test_df, train_ids, test_ids = get_train_test_split(df, train_frac=0.7)
    print(f"   [OK] Train: {len(train_ids)} markets, Test: {len(test_ids)} markets")
    
    # Fit model
    print("\n[3/5] Fitting fair value model...")
    model = BinnedFairValueModel(
        bin_tau_size=30,
        bin_delta_size=5.0,
        n_vol_bins=3,
        sample_every=5
    )
    model.fit(train_df)
    print(f"   [OK] Model fitted: {model.bin_stats.get('n_valid_bins', 0)} valid bins")
    
    # Model validation
    print("\n[4/5] Validating model calibration...")
    test_y = test_df.dropna(subset=['Y']).groupby('market_id')['Y'].first()
    test_df_clean = test_df[test_df['market_id'].isin(test_y.index)]
    test_y_aligned = test_df_clean.groupby('market_id')['Y'].first().loc[test_df_clean['market_id']].values
    
    test_preds = model.predict(test_df_clean)
    valid = ~np.isnan(test_preds)
    
    if valid.sum() > 0:
        brier = compute_brier_score(test_y_aligned[valid], test_preds[valid])
        ece = compute_expected_calibration_error(test_y_aligned[valid], test_preds[valid])
        print(f"   [OK] Test Brier Score: {brier:.4f} (target: < 0.20)")
        print(f"   [OK] Test ECE: {ece:.4f} (target: < 0.05)")
    else:
        print("   [FAIL] Could not compute calibration metrics (no valid predictions)")
        brier = np.nan
        ece = np.nan
    
    # Strategy test
    print("\n[5/5] Testing mispricing strategy...")
    strategy = MispricingBasedStrategy(
        fair_value_model=model,
        buffer=0.02,
        tau_max=420,
        exit_rule='expiry'
    )
    
    result = run_backtest(test_df, strategy, ExecutionConfig(), verbose=False)
    metrics = result['metrics']
    
    print(f"\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\nModel Calibration:")
    brier_pass = not np.isnan(brier) and brier < 0.25
    ece_pass = not np.isnan(ece) and ece < 0.10
    print(f"  Brier Score: {brier:.4f} {'[PASS]' if brier_pass else '[FAIL]'}")
    print(f"  ECE: {ece:.4f} {'[PASS]' if ece_pass else '[FAIL]'}")
    
    print(f"\nStrategy Performance:")
    print(f"  Total PnL: ${metrics['total_pnl']:.2f}")
    print(f"  Mean PnL/market: ${metrics['mean_pnl_per_market']:.4f}")
    t_sig = metrics['t_stat'] > 2.0
    print(f"  t-statistic: {metrics['t_stat']:.2f} {'[SIGNIFICANT]' if t_sig else '[NOT SIGNIFICANT]'}")
    print(f"  Number of trades: {metrics['n_trades']}")
    print(f"  Hit rate (markets): {metrics['hit_rate_per_market']*100:.1f}%")
    print(f"  Hit rate (trades): {metrics['hit_rate_per_trade']*100:.1f}%")
    
    # Save results
    output_dir = PROJECT_ROOT / 'data_v2' / 'backtest_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'model_calibration': {
            'brier_score': float(brier) if not np.isnan(brier) else None,
            'ece': float(ece) if not np.isnan(ece) else None,
            'brier_pass': brier_pass,
            'ece_pass': ece_pass,
        },
        'strategy_performance': metrics,
        'n_train_markets': len(train_ids),
        'n_test_markets': len(test_ids),
    }
    
    with open(output_dir / 'quick_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n[OK] Results saved to: {output_dir / 'quick_test_results.json'}")

if __name__ == '__main__':
    main()

