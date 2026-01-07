#!/usr/bin/env python3
"""Check coverage and verify 6 levels for all processed markets."""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

markets_dir = Path('data_v2/markets')

# Expected columns for 6 levels
expected_levels = [1, 2, 3, 4, 5, 6]
level_columns = {
    'up': ['up_bid', 'up_ask', 'up_bid_size', 'up_ask_size'],
    'down': ['down_bid', 'down_ask', 'down_bid_size', 'down_ask_size']
}

# Build expected column names for all 6 levels
expected_cols = []
for level in expected_levels:
    if level == 1:
        # Level 1 uses base names (up_bid, up_ask, etc.)
        expected_cols.extend(['up_bid', 'up_ask', 'up_bid_size', 'up_ask_size'])
        expected_cols.extend(['down_bid', 'down_ask', 'down_bid_size', 'down_ask_size'])
    else:
        # Levels 2-6 use suffixes (up_bid_2, up_ask_2, etc.)
        expected_cols.extend([f'up_bid_{level}', f'up_ask_{level}', f'up_bid_{level}_size', f'up_ask_{level}_size'])
        expected_cols.extend([f'down_bid_{level}', f'down_ask_{level}', f'down_bid_{level}_size', f'down_ask_{level}_size'])

print('=' * 100)
print('COMPREHENSIVE MARKET COVERAGE AND LEVEL VERIFICATION')
print('=' * 100)
print()

# Collect data for all markets
all_markets_data = []
issues = []

for asset_dir in sorted(markets_dir.iterdir()):
    if not asset_dir.is_dir():
        continue
    
    asset = asset_dir.name
    print(f"\n{'='*100}")
    print(f"ASSET: {asset}")
    print(f"{'='*100}")
    
    asset_markets = sorted([d for d in asset_dir.iterdir() if d.is_dir()])
    print(f"Found {len(asset_markets)} markets\n")
    
    # Print header
    print(f"{'Market':<30} {'CL %':>7} {'PM %':>7} {'Both %':>8} {'CL Pts':>8} {'PM Pts':>8} {'Levels':>8} {'Status':>10}")
    print("-" * 100)
    
    asset_cl_total = 0
    asset_pm_total = 0
    asset_both_total = 0
    asset_markets_count = 0
    
    for market_dir in asset_markets:
        summary_path = market_dir / 'summary.json'
        pm_path = market_dir / 'polymarket.csv'
        
        if not summary_path.exists():
            print(f"{market_dir.name:<30} {'NO SUMMARY':>50}")
            issues.append(f"{asset}/{market_dir.name}: Missing summary.json")
            continue
        
        # Load summary
        with open(summary_path) as f:
            summary = json.load(f)
        
        cl_cov = summary.get('cl_coverage', 0)
        pm_cov = summary.get('pm_coverage', 0)
        both_cov = summary.get('both_coverage', 0)
        cl_points = summary.get('cl_records', 0)
        pm_points = summary.get('pm_records', 0)
        
        # Check levels
        levels_present = 0
        levels_status = "OK"
        
        if pm_path.exists():
            try:
                # Read just the header to check columns
                df = pd.read_csv(pm_path, nrows=0)
                columns = set(df.columns)
                
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
                    
                    if all(col in columns for col in level_cols):
                        levels_present += 1
                    else:
                        missing = [col for col in level_cols if col not in columns]
                        if level == 1:
                            issues.append(f"{asset}/{market_dir.name}: Missing Level 1 columns: {missing}")
                        else:
                            issues.append(f"{asset}/{market_dir.name}: Missing Level {level} columns: {missing}")
                
                if levels_present < 6:
                    levels_status = f"MISSING {6-levels_present}"
                elif levels_present == 6:
                    levels_status = "OK"
                else:
                    levels_status = "EXTRA"
                    
            except Exception as e:
                levels_status = f"ERROR: {str(e)[:20]}"
                issues.append(f"{asset}/{market_dir.name}: Error reading PM CSV: {str(e)}")
        else:
            levels_status = "NO FILE"
            issues.append(f"{asset}/{market_dir.name}: Missing polymarket.csv")
        
        # Status indicator
        if both_cov >= 80 and levels_present == 6:
            status = "[OK] GOOD"
        elif both_cov >= 50 and levels_present == 6:
            status = "[OK]"
        elif levels_present < 6:
            status = "[X] LEVELS"
        elif both_cov < 50:
            status = "[X] LOW"
        else:
            status = "[?] CHECK"
        
        print(f"{market_dir.name:<30} {cl_cov:>6.1f}% {pm_cov:>6.1f}% {both_cov:>7.1f}% "
              f"{cl_points:>8} {pm_points:>8} {levels_present:>7}/6 {levels_status:>10} {status:>10}")
        
        all_markets_data.append({
            'asset': asset,
            'market': market_dir.name,
            'cl_coverage': cl_cov,
            'pm_coverage': pm_cov,
            'both_coverage': both_cov,
            'cl_points': cl_points,
            'pm_points': pm_points,
            'levels_present': levels_present
        })
        
        asset_cl_total += cl_cov
        asset_pm_total += pm_cov
        asset_both_total += both_cov
        asset_markets_count += 1
    
    # Print asset summary
    if asset_markets_count > 0:
        print("-" * 100)
        print(f"{'AVERAGE':<30} {asset_cl_total/asset_markets_count:>6.1f}% "
              f"{asset_pm_total/asset_markets_count:>6.1f}% "
              f"{asset_both_total/asset_markets_count:>7.1f}%")
        print()

# Overall summary
print("\n" + "=" * 100)
print("OVERALL SUMMARY")
print("=" * 100)

if all_markets_data:
    total_markets = len(all_markets_data)
    markets_with_6_levels = sum(1 for m in all_markets_data if m['levels_present'] == 6)
    markets_high_coverage = sum(1 for m in all_markets_data if m['both_coverage'] >= 80)
    markets_good = sum(1 for m in all_markets_data if m['both_coverage'] >= 80 and m['levels_present'] == 6)
    
    avg_cl = sum(m['cl_coverage'] for m in all_markets_data) / total_markets
    avg_pm = sum(m['pm_coverage'] for m in all_markets_data) / total_markets
    avg_both = sum(m['both_coverage'] for m in all_markets_data) / total_markets
    
    print(f"\nTotal Markets: {total_markets}")
    print(f"Markets with 6 levels: {markets_with_6_levels} ({markets_with_6_levels/total_markets*100:.1f}%)")
    print(f"Markets with >=80% both coverage: {markets_high_coverage} ({markets_high_coverage/total_markets*100:.1f}%)")
    print(f"Markets with 6 levels AND >=80% coverage: {markets_good} ({markets_good/total_markets*100:.1f}%)")
    print(f"\nAverage Coverage:")
    print(f"  CL: {avg_cl:.1f}%")
    print(f"  PM: {avg_pm:.1f}%")
    print(f"  Both: {avg_both:.1f}%")
    
    # Group by asset
    print(f"\nBy Asset:")
    asset_stats = defaultdict(lambda: {'count': 0, 'levels_6': 0, 'high_cov': 0, 'good': 0})
    for m in all_markets_data:
        asset_stats[m['asset']]['count'] += 1
        if m['levels_present'] == 6:
            asset_stats[m['asset']]['levels_6'] += 1
        if m['both_coverage'] >= 80:
            asset_stats[m['asset']]['high_cov'] += 1
        if m['both_coverage'] >= 80 and m['levels_present'] == 6:
            asset_stats[m['asset']]['good'] += 1
    
    print(f"{'Asset':<10} {'Markets':>10} {'6 Levels':>12} {'>=80% Cov':>12} {'Both':>12}")
    print("-" * 60)
    for asset in sorted(asset_stats.keys()):
        stats = asset_stats[asset]
        print(f"{asset:<10} {stats['count']:>10} {stats['levels_6']:>12} "
              f"{stats['high_cov']:>12} {stats['good']:>12}")

# Report issues
if issues:
    print(f"\n{'='*100}")
    print(f"ISSUES FOUND: {len(issues)}")
    print("=" * 100)
    for issue in issues[:20]:  # Show first 20 issues
        print(f"  - {issue}")
    if len(issues) > 20:
        print(f"  ... and {len(issues) - 20} more issues")
else:
    print(f"\n{'='*100}")
    print("[OK] NO ISSUES FOUND - All markets have proper structure!")
    print("=" * 100)

print()

