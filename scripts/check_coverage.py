#!/usr/bin/env python3
"""Quick script to check coverage percentages."""
import json
from pathlib import Path

print('=' * 80)
print('COVERAGE ANALYSIS - Markets with >90% Combined (Intersection) Coverage')
print('=' * 80)
print()

markets_dir = Path('data_v2/markets')
high_coverage_markets = []

for asset_dir in markets_dir.iterdir():
    if not asset_dir.is_dir():
        continue
    
    asset = asset_dir.name
    for market_dir in asset_dir.iterdir():
        if not market_dir.is_dir():
            continue
        
        summary_path = market_dir / 'summary.json'
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            
            both_cov = summary.get('both_coverage', 0)
            if both_cov >= 90.0:
                high_coverage_markets.append({
                    'asset': asset,
                    'market': summary.get('market_id', market_dir.name),
                    'both_coverage': both_cov,
                    'cl_coverage': summary.get('cl_coverage', 0),
                    'pm_coverage': summary.get('pm_coverage', 0)
                })

# Sort by coverage (highest first)
high_coverage_markets.sort(key=lambda x: x['both_coverage'], reverse=True)

print(f'Total markets with >=90% combined coverage: {len(high_coverage_markets)}')
print()

if high_coverage_markets:
    print('Top markets:')
    for m in high_coverage_markets[:10]:  # Show top 10
        print(f'  {m["asset"]:4s} {m["market"]:20s} - Both: {m["both_coverage"]:5.1f}% (CL: {m["cl_coverage"]:5.1f}%, PM: {m["pm_coverage"]:5.1f}%)')
    
    if len(high_coverage_markets) > 10:
        print(f'  ... and {len(high_coverage_markets) - 10} more')
    
    print()
    print(f'[OK] YES - At least one market has >90% combined coverage!')
    print(f'  Best: {high_coverage_markets[0]["asset"]} {high_coverage_markets[0]["market"]} with {high_coverage_markets[0]["both_coverage"]:.1f}%')
else:
    print('[WARN] NO - No markets have >=90% combined coverage')
