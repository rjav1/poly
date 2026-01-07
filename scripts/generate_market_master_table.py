#!/usr/bin/env python3
"""Generate market master table in CSV format for AI assistant."""
import json
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import STORAGE

# Load market info
market_info_path = Path(STORAGE.research_dir) / "market_info_all_assets.json"
with open(market_info_path) as f:
    market_infos = json.load(f)

# Create market master table
master_table = []
for info in market_infos:
    master_table.append({
        'market_id': info['market_id'],
        'asset': info['asset'],
        'market_start': info['market_start'],
        'market_end': info['market_end'],
        'strike_K': info['K'],
        'price_to_beat_from_folder': info.get('price_to_beat_from_folder', ''),
        'settlement': info['settlement'],
        'outcome_Y': info['Y'],
        'outcome_label': 'UP' if info['Y'] == 1 else 'DOWN',
        'k_source': info.get('k_source', 'unknown'),
        'k_offset_seconds': info.get('k_offset_seconds', 0),
        'has_exact_k': info.get('has_exact_k', False),
        'settlement_offset_seconds': info.get('settlement_offset_seconds', 0),
        'has_exact_settlement': info.get('has_exact_settlement', False),
        'cl_coverage_pct': round(info.get('cl_coverage_pct', 0), 2),
        'pm_coverage_pct': round(info.get('pm_coverage_pct', 0), 2),
        'both_coverage_pct': round(info.get('both_coverage_pct', 0), 2),
        'either_coverage_pct': round(info.get('either_coverage_pct', 0), 2),
    })

df = pd.DataFrame(master_table)
df = df.sort_values(['asset', 'market_start'])

# Save to CSV
output_path = Path(STORAGE.research_dir) / "market_master_table.csv"
df.to_csv(output_path, index=False)

print(f"Market master table saved to: {output_path}")
print(f"\nTotal markets: {len(df)}")
print(f"\nBy asset:")
print(df['asset'].value_counts().to_string())
print(f"\nFirst 5 rows:")
print(df.head().to_string())

