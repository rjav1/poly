#!/usr/bin/env python3
"""Check coverage for newly processed markets (Level 2 data)."""

import json
from pathlib import Path

markets_dir = Path('data_v2/markets/ETH')

# Find all markets from 12:00 onwards (new markets with Level 2 data)
all_markets = sorted([d for d in markets_dir.iterdir() if d.is_dir()])
new_markets = [m for m in all_markets if any(m.name.startswith(f'20260106_{h:02d}') for h in range(12, 21))]

print(f"Found {len(new_markets)} new markets (12:00-20:00)")
print("\nCoverage for new markets (Level 2 data):")
print("=" * 80)
print(f"{'Market':<20} {'CL %':>8} {'PM %':>8} {'Both %':>8} {'CL Points':>12} {'PM Points':>12}")
print("-" * 80)

total_cl_cov = 0
total_pm_cov = 0
total_both_cov = 0

for m in new_markets:
    summary_path = m / 'summary.json'
    if summary_path.exists():
        with open(summary_path) as f:
            s = json.load(f)
        cl_cov = s.get('cl_coverage', 0)
        pm_cov = s.get('pm_coverage', 0)
        both_cov = s.get('both_coverage', 0)
        cl_points = s.get('cl_records', 0)
        pm_points = s.get('pm_records', 0)
        
        total_cl_cov += cl_cov
        total_pm_cov += pm_cov
        total_both_cov += both_cov
        
        print(f"{m.name:<20} {cl_cov:>7.1f}% {pm_cov:>7.1f}% {both_cov:>7.1f}% {cl_points:>12} {pm_points:>12}")

print("-" * 80)
if new_markets:
    avg_cl = total_cl_cov / len(new_markets)
    avg_pm = total_pm_cov / len(new_markets)
    avg_both = total_both_cov / len(new_markets)
    print(f"{'AVERAGE':<20} {avg_cl:>7.1f}% {avg_pm:>7.1f}% {avg_both:>7.1f}%")
    print(f"\nTotal new markets: {len(new_markets)}")
    print(f"Average CL coverage: {avg_cl:.1f}%")
    print(f"Average PM coverage: {avg_pm:.1f}%")
    print(f"Average Both coverage: {avg_both:.1f}%")

