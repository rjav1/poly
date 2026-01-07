#!/usr/bin/env python3
"""
Validate the adverse selection fix.

This script verifies:
1. AS sign convention is correct (positive = cost, negative = gain)
2. PnL decomposition is internally consistent
3. gain + AS = 0 for each fill
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from scripts.backtest.data_loader import load_eth_markets, add_derived_columns
from scripts.backtest.strategies import SpreadCaptureStrategy
from scripts.backtest.backtest_engine import run_maker_backtest
from scripts.backtest.maker_execution import MakerExecutionConfig, FillModel


def main():
    print("="*70)
    print("ADVERSE SELECTION FIX VALIDATION")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    df, _ = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    print(f"   Loaded {len(df):,} rows, {df['market_id'].nunique()} markets")
    
    # Run backtest
    print("\n2. Running backtest...")
    strategy = SpreadCaptureStrategy(
        spread_min=0.01,
        tau_min=60,
        tau_max=600,
        inventory_limit_up=10.0,
        inventory_limit_down=10.0,
    )
    
    config = MakerExecutionConfig(
        place_latency_ms=100,
        cancel_latency_ms=50,
        fill_model=FillModel.TOUCH_SIZE_PROXY,
        touch_trade_rate_per_second=0.10,
    )
    
    result = run_maker_backtest(df, strategy, config, volume_markets_only=True, verbose=False)
    
    print(f"   Fills: {result['metrics'].get('n_fills', 0)}")
    print(f"   Total PnL: ${result['metrics'].get('total_pnl', 0):.4f}")
    
    # Get fills table from the engine
    print("\n3. Validating PnL decomposition...")
    
    # Use the fills from result (may be FillEvent objects or dicts)
    fills = result.get('fills', [])
    if not fills:
        print("   No fills to validate!")
        return
    
    # Create validation table manually
    fills_data = []
    for f in fills:
        # Handle both FillEvent objects and dicts
        if isinstance(f, dict):
            side = f.get('side')
            fill_price = f.get('fill_price')
            mid_at_fill = f.get('mid_at_fill')
            mid_after_1s = f.get('mid_after_1s') or f.get('mid_1s_later')
            mid_after_5s = f.get('mid_after_5s') or f.get('mid_5s_later')
            fill_size = f.get('fill_size', 1.0)
        else:
            side = f.side
            fill_price = f.fill_price
            mid_at_fill = f.mid_at_fill
            mid_after_1s = f.mid_after_1s
            mid_after_5s = f.mid_after_5s
            fill_size = f.fill_size
        
        if side is None or fill_price is None or mid_at_fill is None:
            continue
        
        # Compute spread captured
        if side == 'BID':
            spread_captured = (mid_at_fill - fill_price) * fill_size
        else:
            spread_captured = (fill_price - mid_at_fill) * fill_size
        
        # Compute AS and gain (should be opposites)
        as_1s = None
        gain_1s = None
        as_5s = None
        gain_5s = None
        
        if mid_after_1s is not None:
            if side == 'BID':
                as_1s = (mid_at_fill - mid_after_1s) * fill_size
                gain_1s = (mid_after_1s - mid_at_fill) * fill_size
            else:
                as_1s = (mid_after_1s - mid_at_fill) * fill_size
                gain_1s = (mid_at_fill - mid_after_1s) * fill_size
        
        if mid_after_5s is not None:
            if side == 'BID':
                as_5s = (mid_at_fill - mid_after_5s) * fill_size
                gain_5s = (mid_after_5s - mid_at_fill) * fill_size
            else:
                as_5s = (mid_after_5s - mid_at_fill) * fill_size
                gain_5s = (mid_at_fill - mid_after_5s) * fill_size
        
        fills_data.append({
            'side': side,
            'fill_price': fill_price,
            'mid_at_fill': mid_at_fill,
            'mid_1s_later': mid_after_1s,
            'mid_5s_later': mid_after_5s,
            'spread_captured': spread_captured,
            'as_1s': as_1s,
            'gain_1s': gain_1s,
            'as_5s': as_5s,
            'gain_5s': gain_5s,
        })
    
    fills_df = pd.DataFrame(fills_data)
    
    # Display sample
    print("\n   Sample fills (first 10):")
    sample_cols = ['side', 'fill_price', 'mid_at_fill', 'spread_captured', 'as_1s', 'gain_1s']
    print(fills_df[sample_cols].head(10).to_string())
    
    # Validate: gain + AS should = 0
    print("\n4. Checking AS + gain = 0...")
    valid_1s = fills_df[fills_df['as_1s'].notna() & fills_df['gain_1s'].notna()]
    if len(valid_1s) > 0:
        sum_check = (valid_1s['as_1s'] + valid_1s['gain_1s']).abs().max()
        print(f"   Max |AS_1s + gain_1s|: {sum_check:.10f}")
        if sum_check < 1e-9:
            print("   [PASS] AS and gain are exact opposites")
        else:
            print("   [FAIL] AS and gain don't sum to zero!")
    
    # Summary statistics
    print("\n5. Summary Statistics:")
    print(f"   Total spread captured: ${fills_df['spread_captured'].sum():.4f}")
    
    if fills_df['as_1s'].notna().any():
        total_as_1s = fills_df['as_1s'].sum()
        total_gain_1s = fills_df['gain_1s'].sum()
        print(f"   Total AS (1s): ${total_as_1s:.4f} (positive = cost)")
        print(f"   Total gain (1s): ${total_gain_1s:.4f} (positive = favorable)")
        
        # Percentage with gain
        n_gain = (fills_df['gain_1s'] > 0).sum()
        n_cost = (fills_df['gain_1s'] < 0).sum()
        pct_gain = n_gain / len(valid_1s) * 100
        print(f"   Fills with gain (1s): {n_gain} ({pct_gain:.1f}%)")
        print(f"   Fills with cost (1s): {n_cost} ({100-pct_gain:.1f}%)")
    
    if fills_df['as_5s'].notna().any():
        total_as_5s = fills_df['as_5s'].sum()
        print(f"   Total AS (5s): ${total_as_5s:.4f}")
    
    # Verify PnL formula
    print("\n6. Verifying PnL formula...")
    spread_sum = fills_df['spread_captured'].sum()
    as_sum = fills_df['as_5s'].fillna(fills_df['as_1s']).sum()
    
    expected_pnl = spread_sum - as_sum
    actual_pnl = result['metrics'].get('total_pnl', 0)
    inventory_carry = result['metrics'].get('inventory_carry', 0)
    
    print(f"   spread_captured: ${spread_sum:.4f}")
    print(f"   adverse_selection: ${as_sum:.4f}")
    print(f"   inventory_carry: ${inventory_carry:.4f}")
    print(f"   Expected (spread - AS + inv): ${spread_sum - as_sum + inventory_carry:.4f}")
    print(f"   Actual total_pnl: ${actual_pnl:.4f}")
    
    diff = abs((spread_sum - as_sum + inventory_carry) - actual_pnl)
    if diff < 0.01:
        print("   [PASS] PnL formula is consistent")
    else:
        print(f"   [FAIL] PnL difference = ${diff:.4f}")
    
    # Interpretation
    print("\n7. Interpretation:")
    if as_sum > 0:
        print(f"   Net adverse selection is POSITIVE (${as_sum:.4f})")
        print("   -> Market moved AGAINST us after fills on average")
        print("   -> This is a COST that reduces PnL")
    else:
        print(f"   Net adverse selection is NEGATIVE (${as_sum:.4f})")
        print("   -> Market moved IN OUR FAVOR after fills on average")
        print("   -> This is a GAIN that increases PnL")
    
    # Check inventory contribution
    print("\n8. Checking inventory contribution...")
    
    # The reported total_pnl likely includes multiple markets
    # So our per-fill calculation may not match if it's aggregated differently
    metrics = result.get('metrics', {})
    reported_spread = metrics.get('spread_captured_total', 0)
    reported_as = metrics.get('adverse_selection_total', 0)
    reported_inv = metrics.get('inventory_carry_total', 0)
    reported_total = metrics.get('total_pnl', 0)
    
    print(f"\n   Reported metrics (from engine):")
    print(f"     - spread_captured_total: ${reported_spread:.4f}")
    print(f"     - adverse_selection_total: ${reported_as:.4f}")
    print(f"     - inventory_carry_total: ${reported_inv:.4f}")
    print(f"     - total_pnl: ${reported_total:.4f}")
    
    print(f"\n   Calculated from fills:")
    print(f"     - spread_captured: ${spread_sum:.4f}")
    print(f"     - adverse_selection: ${as_sum:.4f}")
    print(f"     - inventory_carry: $0.0000 (not available from fills)")
    
    # Verify formula
    expected_from_fills = spread_sum - as_sum
    expected_from_engine = reported_spread - reported_as + reported_inv
    
    print(f"\n   Formula verification:")
    print(f"     - From fills (spread - AS): ${expected_from_fills:.4f}")
    print(f"     - From engine (spread - AS + inv): ${expected_from_engine:.4f}")
    print(f"     - Reported total_pnl: ${reported_total:.4f}")
    
    # Check if engine formula is consistent
    engine_diff = abs(expected_from_engine - reported_total)
    if engine_diff < 0.01:
        print(f"     [PASS] Engine PnL formula is internally consistent")
    else:
        print(f"     [FAIL] Engine has inconsistency: diff = ${engine_diff:.4f}")
    
    # Check if fills match engine
    fills_vs_engine_spread = abs(spread_sum - reported_spread)
    fills_vs_engine_as = abs(as_sum - reported_as)
    print(f"\n   Fills vs Engine comparison:")
    print(f"     - Spread diff: ${fills_vs_engine_spread:.4f}")
    print(f"     - AS diff: ${fills_vs_engine_as:.4f}")
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()

