#!/usr/bin/env python3
"""List exact coverage percentages for all markets in the 6-level folder."""

import json
from pathlib import Path
from collections import defaultdict

markets_dir = Path('data_v2/markets_6levels')

print('=' * 100)
print('6-LEVEL MARKETS COVERAGE REPORT')
print('=' * 100)
print()

if not markets_dir.exists():
    print(f"ERROR: {markets_dir} does not exist!")
    exit(1)

# Collect all market data
all_markets = []
total_markets = 0

for asset_dir in sorted(markets_dir.iterdir()):
    if not asset_dir.is_dir():
        continue
    
    asset = asset_dir.name
    asset_markets = sorted([d for d in asset_dir.iterdir() if d.is_dir()])
    
    for market_dir in asset_markets:
        summary_path = market_dir / 'summary.json'
        
        if not summary_path.exists():
            continue
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        cl_cov = summary.get('cl_coverage', 0)
        pm_cov = summary.get('pm_coverage', 0)
        both_cov = summary.get('both_coverage', 0)
        cl_points = summary.get('cl_records', 0)
        pm_points = summary.get('pm_records', 0)
        
        all_markets.append({
            'asset': asset,
            'market_id': market_dir.name,
            'cl_coverage': cl_cov,
            'pm_coverage': pm_cov,
            'both_coverage': both_cov,
            'cl_points': cl_points,
            'pm_points': pm_points
        })
        total_markets += 1

# Sort by both_coverage (descending)
all_markets.sort(key=lambda x: x['both_coverage'], reverse=True)

# Print detailed table
print(f"{'Asset':<6} {'Market ID':<30} {'CL %':>7} {'PM %':>7} {'Both %':>8} {'CL Pts':>8} {'PM Pts':>8} {'Status':>10}")
print("-" * 100)

markets_above_80 = 0
markets_above_90 = 0

for m in all_markets:
    status = ""
    if m['both_coverage'] >= 90:
        status = "[EXCELLENT]"
        markets_above_90 += 1
        markets_above_80 += 1
    elif m['both_coverage'] >= 80:
        status = "[GOOD]"
        markets_above_80 += 1
    elif m['both_coverage'] >= 50:
        status = "[OK]"
    else:
        status = "[LOW]"
    
    print(f"{m['asset']:<6} {m['market_id']:<30} {m['cl_coverage']:>6.1f}% {m['pm_coverage']:>6.1f}% "
          f"{m['both_coverage']:>7.1f}% {m['cl_points']:>8} {m['pm_points']:>8} {status:>10}")

print("-" * 100)

# Summary statistics
if all_markets:
    avg_cl = sum(m['cl_coverage'] for m in all_markets) / len(all_markets)
    avg_pm = sum(m['pm_coverage'] for m in all_markets) / len(all_markets)
    avg_both = sum(m['both_coverage'] for m in all_markets) / len(all_markets)
    
    print(f"\n{'SUMMARY':<30} {avg_cl:>6.1f}% {avg_pm:>6.1f}% {avg_both:>7.1f}%")
    print()

print("=" * 100)
print("COVERAGE STATISTICS")
print("=" * 100)
print(f"Total markets with 6 levels: {total_markets}")
print(f"Markets with >=80% both coverage: {markets_above_80} ({markets_above_80/total_markets*100:.1f}%)")
print(f"Markets with >=90% both coverage: {markets_above_90} ({markets_above_90/total_markets*100:.1f}%)")
print()

# Group by asset
print("By Asset:")
asset_stats = defaultdict(lambda: {'total': 0, 'above_80': 0, 'above_90': 0})
for m in all_markets:
    asset_stats[m['asset']]['total'] += 1
    if m['both_coverage'] >= 80:
        asset_stats[m['asset']]['above_80'] += 1
    if m['both_coverage'] >= 90:
        asset_stats[m['asset']]['above_90'] += 1

print(f"{'Asset':<10} {'Total':>8} {'>=80%':>8} {'>=90%':>8}")
print("-" * 40)
for asset in sorted(asset_stats.keys()):
    stats = asset_stats[asset]
    print(f"{asset:<10} {stats['total']:>8} {stats['above_80']:>8} {stats['above_90']:>8}")

print()

# Save to JSON for programmatic access
output_json = markets_dir / 'coverage_report.json'
coverage_data = {
    'total_markets': total_markets,
    'markets_above_80': markets_above_80,
    'markets_above_90': markets_above_90,
    'markets': all_markets
}

with open(output_json, 'w') as f:
    json.dump(coverage_data, f, indent=2)

print(f"Coverage data saved to: {output_json}")
print()

# List markets above 80% for easy reference
print("=" * 100)
print("MARKETS WITH >=80% BOTH COVERAGE (Recommended for Analysis)")
print("=" * 100)
print()

above_80_markets = [m for m in all_markets if m['both_coverage'] >= 80]
print(f"Found {len(above_80_markets)} markets with >=80% coverage:\n")

for m in above_80_markets:
    print(f"  {m['asset']}/{m['market_id']}: {m['both_coverage']:.1f}% both coverage")

print()


