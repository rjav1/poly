#!/usr/bin/env python3
"""Check which markets are in the built dataset."""

import json
from pathlib import Path

info_path = Path('data_v2/research/market_info_all_assets.json')
if not info_path.exists():
    print("Dataset not built yet. Run 'python scripts/cli_v2.py build' first.")
    exit(1)

with open(info_path) as f:
    info = json.load(f)

# info is a list, not a dict
if isinstance(info, list):
    all_markets = info
else:
    all_markets = info.get('markets', [])

eth_markets = [m for m in all_markets if m.get('asset') == 'ETH']
print(f"Total ETH markets in dataset: {len(eth_markets)}")

# New markets (12:00-20:00)
new_markets = [m for m in eth_markets if any(m['market_id'].startswith(f'20260106_{h:02d}') for h in range(12, 21))]

print(f"\nNew markets (12:00-20:00) in dataset: {len(new_markets)}")
print("\nNew markets included:")
for m in sorted(new_markets, key=lambda x: x['market_id']):
    both_cov = m.get('both_coverage_pct', m.get('both_coverage', 0))
    print(f"  {m['market_id']}: {both_cov:.1f}% coverage")

# Check which new markets were excluded
all_new_market_dirs = sorted([d.name for d in Path('data_v2/markets/ETH').iterdir() 
                              if d.is_dir() and any(d.name.startswith(f'20260106_{h:02d}') for h in range(12, 21))])
included_names = {m['market_id'] for m in new_markets}
excluded = [name for name in all_new_market_dirs if name not in included_names]
if excluded:
    print(f"\nExcluded new markets (low coverage): {len(excluded)}")
    for name in excluded:
        summary_path = Path(f'data_v2/markets/ETH/{name}/summary.json')
        if summary_path.exists():
            with open(summary_path) as f:
                s = json.load(f)
            print(f"  {name}: {s.get('both_coverage', 0):.1f}% coverage")

