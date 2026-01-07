#!/usr/bin/env python3
"""
Fix Broken Market Data

The broken markets (20260106_1630 onwards) have column shift issues:
- mid column contains 'ETH' (string)
- bid column contains what should be mid
- ask column contains what should be bid
- asset column contains the time

This script:
1. Identifies all broken markets
2. Fixes the chainlink.csv column mapping
3. Updates summary.json with correct strike price
4. Creates a list of fixed markets for rebuilding canonical dataset
"""

import pandas as pd
import json
from pathlib import Path
import shutil

MARKETS_DIR = Path('data_v2/markets/ETH')
OUTPUT_DIR = Path('strategy_derivation/results')


def is_broken_market(market_dir: Path) -> bool:
    """Check if a market has the column shift issue."""
    cl_path = market_dir / 'chainlink.csv'
    if not cl_path.exists():
        return False
    
    cl = pd.read_csv(cl_path, nrows=1)
    
    # Check if mid is a string (should be numeric)
    if 'mid' in cl.columns:
        mid_val = cl['mid'].iloc[0]
        if isinstance(mid_val, str) or pd.isna(mid_val):
            return True
        try:
            float(mid_val)
            return False
        except (ValueError, TypeError):
            return True
    return False


def fix_chainlink_csv(market_dir: Path) -> float:
    """
    Fix the column shift in chainlink.csv.
    
    Returns the strike price (K) from the first row.
    """
    cl_path = market_dir / 'chainlink.csv'
    cl = pd.read_csv(cl_path)
    
    # Backup original
    backup_path = market_dir / 'chainlink_backup.csv'
    if not backup_path.exists():
        shutil.copy(cl_path, backup_path)
    
    # Fix columns:
    # - bid (numeric) -> mid
    # - ask (numeric) -> bid  
    # - Compute new ask from mid and bid
    cl_fixed = pd.DataFrame()
    cl_fixed['timestamp'] = cl['timestamp']
    cl_fixed['collected_at'] = cl['collected_at']
    cl_fixed['seconds'] = cl['seconds']
    cl_fixed['mid'] = cl['bid']  # bid contains the mid price
    cl_fixed['bid'] = cl['ask']  # ask contains the bid price
    # Compute ask as: mid + (mid - bid) = 2*mid - bid
    cl_fixed['ask'] = 2 * cl_fixed['mid'] - cl_fixed['bid']
    cl_fixed['asset'] = 'ETH'
    
    # Save fixed file
    cl_fixed.to_csv(cl_path, index=False)
    
    # Return K (strike price) from first row
    K = cl_fixed['mid'].iloc[0]
    return K


def update_summary_json(market_dir: Path, K: float) -> None:
    """Update summary.json with correct strike price."""
    summary_path = market_dir / 'summary.json'
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    summary['strike_price'] = round(K)
    summary['strike_price_exact'] = K
    summary['fixed'] = True
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    print("=" * 70)
    print("FIX BROKEN MARKET DATA")
    print("=" * 70)
    
    # Find all broken markets
    broken_markets = []
    for market_dir in sorted(MARKETS_DIR.iterdir()):
        if not market_dir.is_dir():
            continue
        if is_broken_market(market_dir):
            broken_markets.append(market_dir)
    
    print(f"\nFound {len(broken_markets)} broken markets:")
    for m in broken_markets:
        print(f"  {m.name}")
    
    if len(broken_markets) == 0:
        print("\nNo broken markets found!")
        return
    
    # Fix each market
    print("\n" + "=" * 60)
    print("FIXING MARKETS")
    print("=" * 60)
    
    fixed_markets = []
    
    for market_dir in broken_markets:
        print(f"\nFixing {market_dir.name}...")
        
        try:
            # Fix chainlink.csv
            K = fix_chainlink_csv(market_dir)
            print(f"  K (strike price) = {K:.2f}")
            
            # Update summary.json
            update_summary_json(market_dir, K)
            print(f"  Updated summary.json")
            
            fixed_markets.append({
                'market_id': market_dir.name,
                'K': K,
                'status': 'fixed'
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            fixed_markets.append({
                'market_id': market_dir.name,
                'error': str(e),
                'status': 'failed'
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    n_fixed = sum(1 for m in fixed_markets if m['status'] == 'fixed')
    n_failed = sum(1 for m in fixed_markets if m['status'] == 'failed')
    
    print(f"\nFixed: {n_fixed}")
    print(f"Failed: {n_failed}")
    
    # Save results
    with open(OUTPUT_DIR / 'fix_broken_markets_results.json', 'w') as f:
        json.dump(fixed_markets, f, indent=2)
    
    print(f"\nResults saved to results/fix_broken_markets_results.json")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Rebuild the canonical dataset:
   python scripts/build_research_dataset_v2.py

2. Re-run the OOS validation:
   python strategy_derivation/scripts/12_oos_corrected.py

This should now include all 36+ markets with valid delta_bps data.
""")


if __name__ == "__main__":
    main()

