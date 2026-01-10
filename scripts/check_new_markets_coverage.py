#!/usr/bin/env python3
"""Check coverage for newly processed markets."""

import json
from pathlib import Path
from datetime import datetime

markets_dir = Path('data_v2/markets_6levels')

if not markets_dir.exists():
    print(f"ERROR: {markets_dir} does not exist!")
    exit(1)

# Find all summary.json files and sort by modification time
all_summaries = []
for asset_dir in markets_dir.iterdir():
    if not asset_dir.is_dir():
        continue
    
    asset = asset_dir.name
    for market_dir in asset_dir.iterdir():
        if not market_dir.is_dir():
            continue
        
        summary_path = market_dir / 'summary.json'
        if summary_path.exists():
            mtime = summary_path.stat().st_mtime
            all_summaries.append((mtime, asset, market_dir, summary_path))

# Sort by modification time (newest first)
all_summaries.sort(key=lambda x: x[0], reverse=True)

# Get markets modified in the last 2 hours, or at least the 20 most recent
current_time = datetime.now().timestamp()
two_hours_ago = current_time - 7200
recent_summaries = [s for s in all_summaries if s[0] > two_hours_ago]

# If no markets in last 2 hours, take the 20 most recent
if not recent_summaries:
    recent_summaries = all_summaries[:20]

print('=' * 100)
print('NEWLY PROCESSED MARKETS - COVERAGE REPORT')
print('=' * 100)
print()

if not recent_summaries:
    print("No recently processed markets found.")
    exit(0)

# Load and display coverage for each market
markets_data = []
total_cl_cov = 0
total_pm_cov = 0
total_both_cov = 0

for mtime, asset, market_dir, summary_path in recent_summaries:
    try:
        with open(summary_path) as f:
            summary = json.load(f)
        
        cl_cov = summary.get('cl_coverage', 0)
        pm_cov = summary.get('pm_coverage', 0)
        both_cov = summary.get('both_coverage', 0)
        cl_points = summary.get('cl_records', 0)
        pm_points = summary.get('pm_records', 0)
        market_id = summary.get('market_id', market_dir.name)
        
        markets_data.append({
            'asset': asset,
            'market_id': market_id,
            'cl_coverage': cl_cov,
            'pm_coverage': pm_cov,
            'both_coverage': both_cov,
            'cl_points': cl_points,
            'pm_points': pm_points,
            'mtime': mtime
        })
        
        total_cl_cov += cl_cov
        total_pm_cov += pm_cov
        total_both_cov += both_cov
    except Exception as e:
        print(f"Error reading {summary_path}: {e}")
        continue

# Sort by both_coverage (descending)
markets_data.sort(key=lambda x: x['both_coverage'], reverse=True)

# Print detailed table
print(f"{'Asset':<6} {'Market ID':<35} {'CL %':>7} {'PM %':>7} {'Both %':>8} {'CL Pts':>8} {'PM Pts':>8}")
print("-" * 100)

for m in markets_data:
    print(f"{m['asset']:<6} {m['market_id']:<35} {m['cl_coverage']:>7.1f}% {m['pm_coverage']:>7.1f}% {m['both_coverage']:>7.1f}% {m['cl_points']:>8} {m['pm_points']:>8}")

print("-" * 100)

if markets_data:
    n = len(markets_data)
    avg_cl = total_cl_cov / n
    avg_pm = total_pm_cov / n
    avg_both = total_both_cov / n
    
    print(f"{'COMBINED AVERAGE':<42} {avg_cl:>7.1f}% {avg_pm:>7.1f}% {avg_both:>7.1f}%")
    print()
    print(f"Total newly processed markets: {n}")
    print(f"Average CL coverage: {avg_cl:.2f}%")
    print(f"Average PM coverage: {avg_pm:.2f}%")
    print(f"Average Both (Combined) coverage: {avg_both:.2f}%")
    print()
    print(f"Total CL coverage sum: {total_cl_cov:.2f}%")
    print(f"Total PM coverage sum: {total_pm_cov:.2f}%")
    print(f"Total Both coverage sum: {total_both_cov:.2f}%")

