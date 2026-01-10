#!/usr/bin/env python3
"""
Delete markets with less than a specified coverage threshold and report statistics.

This script:
1. Reads all markets from data_v2/markets_6levels/{ASSET}/
2. Checks coverage from summary.json
3. Deletes markets with < threshold% both_coverage
4. Reports exact coverage numbers from the last collection
5. Counts remaining markets with 6 levels of data

Usage:
    python scripts/delete_low_coverage_markets.py [--threshold THRESHOLD] [--asset ASSET] [--yes]
"""

import json
import pandas as pd
import shutil
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import STORAGE, SUPPORTED_ASSETS

def has_6_levels(market_dir: Path) -> bool:
    """Check if a market has all 6 levels of order book data."""
    pm_path = market_dir / 'polymarket.csv'
    
    if not pm_path.exists():
        return False
    
    try:
        # Read just the header to check columns
        df = pd.read_csv(pm_path, nrows=0)
        columns = set(df.columns)
        
        # Expected columns for 6 levels
        expected_levels = [1, 2, 3, 4, 5, 6]
        
        # Check each level
        for level in expected_levels:
            if level == 1:
                # Level 1 uses "best" prefix: up_best_bid, up_best_ask, etc.
                level_cols = ['up_best_bid', 'up_best_ask', 'up_best_bid_size', 'up_best_ask_size',
                             'down_best_bid', 'down_best_ask', 'down_best_bid_size', 'down_best_ask_size']
            else:
                level_cols = [f'up_bid_{level}', f'up_ask_{level}', 
                             f'up_bid_{level}_size', f'up_ask_{level}_size',
                             f'down_bid_{level}', f'down_ask_{level}',
                             f'down_bid_{level}_size', f'down_ask_{level}_size']
            
            if not all(col in columns for col in level_cols):
                return False
        
        return True
    except Exception as e:
        print(f"  Error checking {market_dir.name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Delete markets with low coverage",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=80.0,
        help='Coverage threshold (default: 80.0)'
    )
    parser.add_argument(
        '--asset',
        type=str,
        default='ETH',
        choices=SUPPORTED_ASSETS,
        help='Asset to process (default: ETH)'
    )
    parser.add_argument(
        '--yes',
        action='store_true',
        help='Skip confirmation prompt'
    )
    args = parser.parse_args()
    
    threshold = args.threshold
    asset = args.asset.upper()
    markets_dir = Path(STORAGE.markets_dir) / asset

    print('=' * 100)
    print(f'DELETE MARKETS WITH < {threshold}% COVERAGE ({asset})')
    print('=' * 100)
    print()
    
    if not markets_dir.exists():
        print(f"ERROR: {markets_dir} does not exist!")
        return 1
    
    # Collect all market data
    all_markets = []
    market_dirs = sorted([d for d in markets_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(market_dirs)} {asset} markets to process\n")
    
    for market_dir in market_dirs:
        summary_path = market_dir / 'summary.json'
        
        if not summary_path.exists():
            print(f"  [WARN] {market_dir.name}: Missing summary.json, skipping")
            continue
        
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            
            cl_cov = summary.get('cl_coverage', 0)
            pm_cov = summary.get('pm_coverage', 0)
            both_cov = summary.get('both_coverage', 0)
            cl_points = summary.get('cl_records', 0)
            pm_points = summary.get('pm_records', 0)
            market_start = summary.get('market_start', '')
            
            # Check if has 6 levels
            has_6 = has_6_levels(market_dir)
            
            all_markets.append({
                'market_dir': market_dir,
                'market_id': market_dir.name,
                'cl_coverage': cl_cov,
                'pm_coverage': pm_cov,
                'both_coverage': both_cov,
                'cl_points': cl_points,
                'pm_points': pm_points,
                'market_start': market_start,
                'has_6_levels': has_6
            })
        except Exception as e:
            print(f"  [ERROR] {market_dir.name}: Error reading summary.json: {e}")
            continue

    print(f"Successfully loaded {len(all_markets)} markets\n")
    
    # Sort by market_start (most recent first)
    all_markets.sort(key=lambda x: x['market_start'], reverse=True)
    
    # Identify markets to delete (< threshold% coverage)
    markets_to_delete = [m for m in all_markets if m['both_coverage'] < threshold]
    
    print("=" * 100)
    print(f"MARKETS TO DELETE (< {threshold}% COVERAGE)")
    print("=" * 100)
    print(f"Found {len(markets_to_delete)} markets with < {threshold}% coverage:\n")

    if markets_to_delete:
        print(f"{'Market ID':<35} {'CL %':>7} {'PM %':>7} {'Both %':>8} {'6 Levels':>10}")
        print("-" * 100)
        for m in sorted(markets_to_delete, key=lambda x: x['both_coverage']):
            print(f"{m['market_id']:<35} {m['cl_coverage']:>6.1f}% {m['pm_coverage']:>6.1f}% "
                  f"{m['both_coverage']:>7.1f}% {'YES' if m['has_6_levels'] else 'NO':>10}")
        print()
        
        # Confirm deletion
        if not args.yes:
            response = input(f"\nDelete {len(markets_to_delete)} markets? [y/N]: ").strip().lower()
            if response != 'y':
                print("Cancelled.")
                return 0
        
        # Delete markets
        print(f"\nDeleting {len(markets_to_delete)} markets...")
        deleted_count = 0
        for m in markets_to_delete:
            try:
                shutil.rmtree(m['market_dir'])
                print(f"  [DELETED] {m['market_id']} (coverage: {m['both_coverage']:.1f}%)")
                deleted_count += 1
            except Exception as e:
                print(f"  [ERROR] Failed to delete {m['market_id']}: {e}")
        
        print(f"\nDeleted {deleted_count} markets")
    else:
        print(f"No markets to delete - all have >= {threshold}% coverage!")

    # Remaining markets
    remaining_markets = [m for m in all_markets if m['both_coverage'] >= threshold]
    
    print("\n" + "=" * 100)
    print(f"REMAINING MARKETS (>= {threshold}% COVERAGE)")
    print("=" * 100)
    print(f"Total remaining: {len(remaining_markets)} markets\n")

    # Sort by both_coverage (descending)
    remaining_markets.sort(key=lambda x: x['both_coverage'], reverse=True)

    # Print coverage report
    print("EXACT COVERAGE NUMBERS FROM LAST COLLECTION:")
    print("-" * 100)
    print(f"{'Market ID':<35} {'CL %':>7} {'PM %':>7} {'Both %':>8} {'CL Pts':>8} {'PM Pts':>8} {'6 Levels':>10}")
    print("-" * 100)

    for m in remaining_markets:
        print(f"{m['market_id']:<35} {m['cl_coverage']:>6.2f}% {m['pm_coverage']:>6.2f}% "
              f"{m['both_coverage']:>7.2f}% {m['cl_points']:>8} {m['pm_points']:>8} "
              f"{'YES' if m['has_6_levels'] else 'NO':>10}")

    print()

    # Count markets with 6 levels
    markets_with_6_levels = [m for m in remaining_markets if m['has_6_levels']]

    print("=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    print(f"Total {asset} markets processed: {len(all_markets)}")
    print(f"Markets deleted (< {threshold}% coverage): {len(markets_to_delete)}")
    print(f"Markets remaining (>= {threshold}% coverage): {len(remaining_markets)}")
    print(f"\n{asset} markets with 6 levels data: {len(markets_with_6_levels)}")
    print()

    if remaining_markets:
        avg_cl = sum(m['cl_coverage'] for m in remaining_markets) / len(remaining_markets)
        avg_pm = sum(m['pm_coverage'] for m in remaining_markets) / len(remaining_markets)
        avg_both = sum(m['both_coverage'] for m in remaining_markets) / len(remaining_markets)
        
        print("Average Coverage (remaining markets):")
        print(f"  CL: {avg_cl:.2f}%")
        print(f"  PM: {avg_pm:.2f}%")
        print(f"  Both: {avg_both:.2f}%")
        print()

    # Coverage distribution
    coverage_ranges = {
        '>= 95%': sum(1 for m in remaining_markets if m['both_coverage'] >= 95),
        '90-95%': sum(1 for m in remaining_markets if 90 <= m['both_coverage'] < 95),
        '85-90%': sum(1 for m in remaining_markets if 85 <= m['both_coverage'] < 90),
        '80-85%': sum(1 for m in remaining_markets if 80 <= m['both_coverage'] < 85),
    }

    print("Coverage Distribution (remaining markets):")
    for range_name, count in coverage_ranges.items():
        print(f"  {range_name}: {count} markets")
    
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
