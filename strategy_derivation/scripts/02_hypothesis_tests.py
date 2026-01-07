#!/usr/bin/env python3
"""
Phase 2: Hypothesis Testing (H1-H5)

Tests behavioral signatures of profitable traders to extract strategy parameters.

Input: wallet_data_normalized.parquet
Output: hypothesis_results.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "results"
MARKET_DURATION_SECONDS = 900


def load_normalized_data() -> pd.DataFrame:
    """Load normalized wallet data."""
    path = DATA_DIR / "wallet_data_normalized.parquet"
    print(f"Loading: {path}")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows")
    return df


# =============================================================================
# H1: Late-Window Concentration
# =============================================================================

def test_h1_late_window_concentration(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    H1: Test trade intensity vs tau (time to expiry).
    
    Compute share of trades in last 60s, 120s, 300s per wallet.
    """
    print("\n" + "=" * 60)
    print("H1: Late-Window Concentration")
    print("=" * 60)
    
    # Filter to TRADE type only
    trades = df[df['type'] == 'TRADE'].copy()
    
    results = {}
    
    for handle in trades['_handle'].unique():
        handle_df = trades[trades['_handle'] == handle]
        n_total = len(handle_df)
        
        if n_total == 0:
            continue
        
        # Count trades in different tau windows
        n_last_60 = (handle_df['tau'] <= 60).sum()
        n_last_120 = (handle_df['tau'] <= 120).sum()
        n_last_300 = (handle_df['tau'] <= 300).sum()
        n_last_600 = (handle_df['tau'] <= 600).sum()
        
        results[handle] = {
            'n_trades': n_total,
            'share_last_60s': n_last_60 / n_total,
            'share_last_120s': n_last_120 / n_total,
            'share_last_300s': n_last_300 / n_total,
            'share_last_600s': n_last_600 / n_total,
            'mean_tau': handle_df['tau'].mean(),
            'median_tau': handle_df['tau'].median(),
        }
        
        print(f"\n{handle}:")
        print(f"  Total trades: {n_total:,}")
        print(f"  Last 60s: {results[handle]['share_last_60s']*100:.1f}%")
        print(f"  Last 120s: {results[handle]['share_last_120s']*100:.1f}%")
        print(f"  Last 300s (5min): {results[handle]['share_last_300s']*100:.1f}%")
        print(f"  Median tau: {results[handle]['median_tau']:.0f}s")
    
    return results


# =============================================================================
# H2: Two-Sided vs One-Sided Behavior
# =============================================================================

def test_h2_two_sided_behavior(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    H2: For each (wallet, conditionId), check if both UP and DOWN outcomes traded.
    
    Compute % of markets where wallet trades both sides.
    """
    print("\n" + "=" * 60)
    print("H2: Two-Sided vs One-Sided Behavior")
    print("=" * 60)
    
    # Filter to TRADE type only
    trades = df[df['type'] == 'TRADE'].copy()
    
    results = {}
    
    for handle in trades['_handle'].unique():
        handle_df = trades[trades['_handle'] == handle]
        
        # Group by conditionId (market)
        markets_traded = handle_df.groupby('conditionId')['outcome'].apply(set).reset_index()
        
        n_markets = len(markets_traded)
        n_both = (markets_traded['outcome'].apply(lambda x: 'Up' in x and 'Down' in x)).sum()
        n_up_only = (markets_traded['outcome'].apply(lambda x: 'Up' in x and 'Down' not in x)).sum()
        n_down_only = (markets_traded['outcome'].apply(lambda x: 'Down' in x and 'Up' not in x)).sum()
        
        results[handle] = {
            'n_markets': n_markets,
            'n_both_outcomes': n_both,
            'n_up_only': n_up_only,
            'n_down_only': n_down_only,
            'pct_both_outcomes': n_both / n_markets if n_markets > 0 else 0,
            'pct_up_only': n_up_only / n_markets if n_markets > 0 else 0,
            'pct_down_only': n_down_only / n_markets if n_markets > 0 else 0,
        }
        
        print(f"\n{handle}:")
        print(f"  Markets traded: {n_markets:,}")
        print(f"  Both outcomes: {n_both:,} ({results[handle]['pct_both_outcomes']*100:.1f}%)")
        print(f"  UP only: {n_up_only:,} ({results[handle]['pct_up_only']*100:.1f}%)")
        print(f"  DOWN only: {n_down_only:,} ({results[handle]['pct_down_only']*100:.1f}%)")
    
    return results


# =============================================================================
# H3: Underround Harvesting (Paired Per-Second)
# =============================================================================

def test_h3_underround_harvesting(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    H3: For each (wallet, conditionId, second), compute paired complete-set edge.
    
    edge = 1 - (vwap_up + vwap_down) for matched pairs in the same second.
    """
    print("\n" + "=" * 60)
    print("H3: Underround Harvesting (Paired Per-Second)")
    print("=" * 60)
    
    # Filter to BUY trades only (underround harvesting = buying both sides)
    trades = df[(df['type'] == 'TRADE') & (df['side'] == 'BUY')].copy()
    
    # Floor timestamp to second for grouping
    trades['t_second'] = trades['t'].astype(int)
    
    results = {}
    
    for handle in trades['_handle'].unique():
        handle_df = trades[trades['_handle'] == handle]
        
        # Group by (conditionId, t_second)
        edges = []
        matched_volumes = []
        
        for (cond_id, t_sec), group in handle_df.groupby(['conditionId', 't_second']):
            up_trades = group[group['outcome'] == 'Up']
            down_trades = group[group['outcome'] == 'Down']
            
            if len(up_trades) == 0 or len(down_trades) == 0:
                continue
            
            # Compute VWAP for each side
            up_qty = up_trades['size'].sum()
            up_vwap = (up_trades['price'] * up_trades['size']).sum() / up_qty if up_qty > 0 else 0
            
            down_qty = down_trades['size'].sum()
            down_vwap = (down_trades['price'] * down_trades['size']).sum() / down_qty if down_qty > 0 else 0
            
            # Matched quantity
            matched_qty = min(up_qty, down_qty)
            
            # Set cost and edge
            set_cost = up_vwap + down_vwap
            edge = 1 - set_cost
            
            if matched_qty > 0:
                edges.append(edge)
                matched_volumes.append(matched_qty)
        
        if len(edges) == 0:
            results[handle] = {
                'n_paired_seconds': 0,
                'median_edge': None,
                'mean_edge': None,
                'pct_positive_edge': None,
                'total_matched_volume': 0,
            }
            print(f"\n{handle}: No paired seconds found")
            continue
        
        edges = np.array(edges)
        matched_volumes = np.array(matched_volumes)
        
        results[handle] = {
            'n_paired_seconds': len(edges),
            'median_edge': float(np.median(edges)),
            'mean_edge': float(np.mean(edges)),
            'pct_positive_edge': float((edges > 0).sum() / len(edges)),
            'p25_edge': float(np.percentile(edges, 25)),
            'p75_edge': float(np.percentile(edges, 75)),
            'total_matched_volume': float(matched_volumes.sum()),
            'mean_matched_volume': float(matched_volumes.mean()),
        }
        
        print(f"\n{handle}:")
        print(f"  Paired seconds: {len(edges):,}")
        print(f"  Median edge: ${results[handle]['median_edge']:.4f}")
        print(f"  Mean edge: ${results[handle]['mean_edge']:.4f}")
        print(f"  % positive edge: {results[handle]['pct_positive_edge']*100:.1f}%")
        print(f"  Total matched volume: {results[handle]['total_matched_volume']:.0f}")
    
    return results


# =============================================================================
# H4: Short-Horizon Scalping (FIFO Matching)
# =============================================================================

def test_h4_short_horizon_scalping(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    H4: For wallets with SELLs, FIFO match BUY->SELL within (conditionId, outcome).
    
    Compute holding times and matched PnL.
    """
    print("\n" + "=" * 60)
    print("H4: Short-Horizon Scalping (FIFO Matching)")
    print("=" * 60)
    
    # Filter to TRADE type only
    trades = df[df['type'] == 'TRADE'].copy()
    
    results = {}
    
    for handle in trades['_handle'].unique():
        handle_df = trades[trades['_handle'] == handle]
        
        # Check if this wallet has sells
        n_sells = (handle_df['side'] == 'SELL').sum()
        n_buys = (handle_df['side'] == 'BUY').sum()
        
        if n_sells == 0:
            results[handle] = {
                'n_buys': n_buys,
                'n_sells': 0,
                'n_matched_lots': 0,
                'median_hold_seconds': None,
                'mean_hold_seconds': None,
                'matched_pnl': None,
            }
            print(f"\n{handle}: No SELLs (BUYs only: {n_buys:,})")
            continue
        
        # FIFO matching per (conditionId, outcome)
        hold_times = []
        pnls = []
        matched_lots = 0
        
        for (cond_id, outcome), group in handle_df.groupby(['conditionId', 'outcome']):
            buys = group[group['side'] == 'BUY'].sort_values('timestamp').copy()
            sells = group[group['side'] == 'SELL'].sort_values('timestamp').copy()
            
            buy_queue = []  # (timestamp, price, remaining_qty)
            
            # Process buys into queue
            for _, buy in buys.iterrows():
                buy_queue.append({
                    'ts': buy['timestamp'],
                    't': buy['t'],
                    'price': buy['price'],
                    'qty': buy['size']
                })
            
            # Match sells against buy queue (FIFO)
            for _, sell in sells.iterrows():
                sell_qty = sell['size']
                sell_price = sell['price']
                sell_ts = sell['timestamp']
                sell_t = sell['t']
                
                while sell_qty > 0 and buy_queue:
                    buy = buy_queue[0]
                    
                    # Skip if buy is after sell (can't match)
                    if buy['ts'] > sell_ts:
                        break
                    
                    match_qty = min(buy['qty'], sell_qty)
                    
                    # Record holding time and PnL
                    hold_time = sell_t - buy['t']  # in seconds
                    pnl = (sell_price - buy['price']) * match_qty
                    
                    if hold_time >= 0:  # Valid match
                        hold_times.append(hold_time)
                        pnls.append(pnl)
                        matched_lots += match_qty
                    
                    # Update quantities
                    buy['qty'] -= match_qty
                    sell_qty -= match_qty
                    
                    # Remove exhausted buy
                    if buy['qty'] <= 0:
                        buy_queue.pop(0)
        
        if len(hold_times) == 0:
            results[handle] = {
                'n_buys': n_buys,
                'n_sells': n_sells,
                'n_matched_lots': 0,
                'median_hold_seconds': None,
                'mean_hold_seconds': None,
                'matched_pnl': None,
            }
            print(f"\n{handle}: No valid matches (BUYs: {n_buys:,}, SELLs: {n_sells:,})")
            continue
        
        hold_times = np.array(hold_times)
        pnls = np.array(pnls)
        
        results[handle] = {
            'n_buys': n_buys,
            'n_sells': n_sells,
            'n_matched_lots': int(matched_lots),
            'median_hold_seconds': float(np.median(hold_times)),
            'mean_hold_seconds': float(np.mean(hold_times)),
            'p25_hold_seconds': float(np.percentile(hold_times, 25)),
            'p75_hold_seconds': float(np.percentile(hold_times, 75)),
            'matched_pnl': float(pnls.sum()),
            'mean_pnl_per_lot': float(pnls.mean()),
        }
        
        print(f"\n{handle}:")
        print(f"  BUYs: {n_buys:,}, SELLs: {n_sells:,}")
        print(f"  Matched lots: {matched_lots:,.0f}")
        print(f"  Median hold: {results[handle]['median_hold_seconds']:.0f}s")
        print(f"  Mean hold: {results[handle]['mean_hold_seconds']:.0f}s")
        print(f"  Matched PnL: ${results[handle]['matched_pnl']:.2f}")
    
    return results


# =============================================================================
# H5: Conversion Mechanics Usage
# =============================================================================

def test_h5_conversion_mechanics(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    H5: Count SPLIT, MERGE, CONVERSION by wallet.
    """
    print("\n" + "=" * 60)
    print("H5: Conversion Mechanics Usage")
    print("=" * 60)
    
    results = {}
    
    for handle in df['_handle'].unique():
        handle_df = df[df['_handle'] == handle]
        
        type_counts = handle_df['type'].value_counts().to_dict()
        
        results[handle] = {
            'TRADE': type_counts.get('TRADE', 0),
            'SPLIT': type_counts.get('SPLIT', 0),
            'MERGE': type_counts.get('MERGE', 0),
            'CONVERSION': type_counts.get('CONVERSION', 0),
            'REDEEM': type_counts.get('REDEEM', 0),
        }
        
        print(f"\n{handle}:")
        for action, count in results[handle].items():
            if count > 0:
                print(f"  {action}: {count:,}")
    
    # Also load from summary file for comparison
    summary_path = DATA_DIR / "profitable_traders_wallet_data" / "polymarket_multi_summary_by_user.csv"
    if summary_path.exists():
        print("\nFrom summary file (all markets, not just 15m):")
        summary_df = pd.read_csv(summary_path)
        print(summary_df.to_string(index=False))
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Phase 2: Hypothesis Testing (H1-H5)")
    print("=" * 60)
    
    # Load data
    df = load_normalized_data()
    
    # Run all hypothesis tests
    results = {
        'H1_late_window_concentration': test_h1_late_window_concentration(df),
        'H2_two_sided_behavior': test_h2_two_sided_behavior(df),
        'H3_underround_harvesting': test_h3_underround_harvesting(df),
        'H4_short_horizon_scalping': test_h4_short_horizon_scalping(df),
        'H5_conversion_mechanics': test_h5_conversion_mechanics(df),
    }
    
    # Save results
    output_path = OUTPUT_DIR / "hypothesis_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nResults saved to: {output_path}")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    
    print("\n{:<35} {:>12} {:>12} {:>12} {:>12}".format(
        "Wallet", "Late 300s%", "Both Sides%", "Med Edge $", "Med Hold s"
    ))
    print("-" * 85)
    
    for handle in df['_handle'].unique():
        h1 = results['H1_late_window_concentration'].get(handle, {})
        h2 = results['H2_two_sided_behavior'].get(handle, {})
        h3 = results['H3_underround_harvesting'].get(handle, {})
        h4 = results['H4_short_horizon_scalping'].get(handle, {})
        
        late_300 = h1.get('share_last_300s', 0) * 100
        both_pct = h2.get('pct_both_outcomes', 0) * 100
        med_edge = h3.get('median_edge', None)
        med_hold = h4.get('median_hold_seconds', None)
        
        med_edge_str = f"${med_edge:.4f}" if med_edge is not None else "N/A"
        med_hold_str = f"{med_hold:.0f}" if med_hold is not None else "N/A"
        
        print("{:<35} {:>12.1f} {:>12.1f} {:>12} {:>12}".format(
            handle, late_300, both_pct, med_edge_str, med_hold_str
        ))
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()

