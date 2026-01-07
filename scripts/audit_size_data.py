#!/usr/bin/env python3
"""
Audit size data quality in ETH volume markets.
Part of spread capture strategy implementation.
"""

import pandas as pd
import json
from pathlib import Path

# Load canonical dataset
df = pd.read_parquet('data_v2/research/canonical_dataset_all_assets.parquet')
df_eth = df[df['asset'] == 'ETH'].copy()

# Volume market IDs (12 markets from 16:30-19:15 on Jan 6)
volume_market_prefixes = [
    '20260106_1630', '20260106_1645', '20260106_1700', '20260106_1715',
    '20260106_1730', '20260106_1745', '20260106_1800', '20260106_1815',
    '20260106_1830', '20260106_1845', '20260106_1900', '20260106_1915'
]

# Find matching markets
all_market_ids = df_eth['market_id'].unique()
volume_markets = [m for m in all_market_ids if any(m.startswith(p) for p in volume_market_prefixes)]
df_vol = df_eth[df_eth['market_id'].isin(volume_markets)].copy()

print("=" * 60)
print("SIZE DATA AUDIT - ETH Volume Markets")
print("=" * 60)

print(f"\nTotal ETH markets: {len(all_market_ids)}")
print(f"Volume markets found: {len(volume_markets)}")
print(f"Volume market IDs: {volume_markets}")
print(f"Volume market rows: {len(df_vol)}")

# All columns
print(f"\nAll columns in dataset:")
print([c for c in df_vol.columns])

# Level-2 columns
level2_cols = [c for c in df_vol.columns if 'bid_2' in c or 'ask_2' in c]
print(f"\nLevel-2 columns: {level2_cols}")

# Size column coverage
size_cols = ['pm_up_best_bid_size', 'pm_up_best_ask_size', 'pm_down_best_bid_size', 'pm_down_best_ask_size']
print("\n" + "-" * 60)
print("SIZE DATA COVERAGE")
print("-" * 60)

print("\nIn volume markets:")
for col in size_cols:
    if col in df_vol.columns:
        non_nan = df_vol[col].notna().sum()
        total = len(df_vol)
        pct = non_nan / total * 100
        print(f"  {col}: {non_nan}/{total} ({pct:.1f}%)")
        if non_nan > 0:
            mean_val = df_vol[col].mean()
            med_val = df_vol[col].median()
            max_val = df_vol[col].max()
            print(f"    Mean: {mean_val:.2f}, Median: {med_val:.2f}, Max: {max_val:.2f}")

print("\nIn ALL ETH markets:")
for col in size_cols:
    if col in df_eth.columns:
        non_nan = df_eth[col].notna().sum()
        total = len(df_eth)
        pct = non_nan / total * 100
        print(f"  {col}: {non_nan}/{total} ({pct:.1f}%)")

# Spread statistics
print("\n" + "-" * 60)
print("SPREAD STATISTICS (Volume Markets)")
print("-" * 60)

df_vol['up_spread'] = df_vol['pm_up_best_ask'] - df_vol['pm_up_best_bid']
df_vol['down_spread'] = df_vol['pm_down_best_ask'] - df_vol['pm_down_best_bid']

print(f"\nUP spread:")
print(f"  Mean: {df_vol['up_spread'].mean():.4f}")
print(f"  Median: {df_vol['up_spread'].median():.4f}")
print(f"  Min: {df_vol['up_spread'].min():.4f}")
print(f"  Max: {df_vol['up_spread'].max():.4f}")

print(f"\nDOWN spread:")
print(f"  Mean: {df_vol['down_spread'].mean():.4f}")
print(f"  Median: {df_vol['down_spread'].median():.4f}")
print(f"  Min: {df_vol['down_spread'].min():.4f}")
print(f"  Max: {df_vol['down_spread'].max():.4f}")

# Spread >= thresholds
print(f"\nSpread thresholds:")
print(f"  UP spread >= 0.01: {(df_vol['up_spread'] >= 0.01).mean()*100:.1f}%")
print(f"  UP spread >= 0.02: {(df_vol['up_spread'] >= 0.02).mean()*100:.1f}%")
print(f"  UP spread >= 0.03: {(df_vol['up_spread'] >= 0.03).mean()*100:.1f}%")
print(f"  DOWN spread >= 0.01: {(df_vol['down_spread'] >= 0.01).mean()*100:.1f}%")
print(f"  DOWN spread >= 0.02: {(df_vol['down_spread'] >= 0.02).mean()*100:.1f}%")
print(f"  DOWN spread >= 0.03: {(df_vol['down_spread'] >= 0.03).mean()*100:.1f}%")

# Per-market breakdown
print("\n" + "-" * 60)
print("PER-MARKET SIZE COVERAGE")
print("-" * 60)

for mid in sorted(volume_markets):
    mdf = df_vol[df_vol['market_id'] == mid]
    up_bid_cov = mdf['pm_up_best_bid_size'].notna().mean() * 100
    up_ask_cov = mdf['pm_up_best_ask_size'].notna().mean() * 100
    down_bid_cov = mdf['pm_down_best_bid_size'].notna().mean() * 100
    down_ask_cov = mdf['pm_down_best_ask_size'].notna().mean() * 100
    print(f"  {mid}:")
    print(f"    UP bid: {up_bid_cov:.1f}%, UP ask: {up_ask_cov:.1f}%")
    print(f"    DOWN bid: {down_bid_cov:.1f}%, DOWN ask: {down_ask_cov:.1f}%")

# Save audit results
audit_results = {
    "total_eth_markets": len(all_market_ids),
    "volume_markets_count": len(volume_markets),
    "volume_markets": volume_markets,
    "volume_market_rows": len(df_vol),
    "size_coverage": {
        col: {
            "non_nan": int(df_vol[col].notna().sum()),
            "total": len(df_vol),
            "pct": float(df_vol[col].notna().mean() * 100),
            "mean": float(df_vol[col].mean()) if df_vol[col].notna().any() else None,
            "median": float(df_vol[col].median()) if df_vol[col].notna().any() else None,
            "max": float(df_vol[col].max()) if df_vol[col].notna().any() else None
        }
        for col in size_cols if col in df_vol.columns
    },
    "spread_stats": {
        "up_spread": {
            "mean": float(df_vol['up_spread'].mean()),
            "median": float(df_vol['up_spread'].median()),
            "min": float(df_vol['up_spread'].min()),
            "max": float(df_vol['up_spread'].max()),
            "pct_gte_1c": float((df_vol['up_spread'] >= 0.01).mean() * 100),
            "pct_gte_2c": float((df_vol['up_spread'] >= 0.02).mean() * 100),
        },
        "down_spread": {
            "mean": float(df_vol['down_spread'].mean()),
            "median": float(df_vol['down_spread'].median()),
            "min": float(df_vol['down_spread'].min()),
            "max": float(df_vol['down_spread'].max()),
            "pct_gte_1c": float((df_vol['down_spread'] >= 0.01).mean() * 100),
            "pct_gte_2c": float((df_vol['down_spread'] >= 0.02).mean() * 100),
        }
    }
}

output_path = Path("data_v2/research/size_data_audit.json")
with open(output_path, 'w') as f:
    json.dump(audit_results, f, indent=2)
print(f"\n\nAudit results saved to: {output_path}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"- 12 volume markets have SIZE DATA available")
print(f"- UP bid size coverage: {audit_results['size_coverage']['pm_up_best_bid_size']['pct']:.1f}%")
print(f"- UP ask size coverage: {audit_results['size_coverage']['pm_up_best_ask_size']['pct']:.1f}%")
print(f"- DOWN bid size coverage: {audit_results['size_coverage']['pm_down_best_bid_size']['pct']:.1f}%")
print(f"- DOWN ask size coverage: {audit_results['size_coverage']['pm_down_best_ask_size']['pct']:.1f}%")
print(f"- Mean UP spread: {audit_results['spread_stats']['up_spread']['mean']:.4f}")
print(f"- Mean DOWN spread: {audit_results['spread_stats']['down_spread']['mean']:.4f}")
print(f"- Spreads >= 2c: UP {audit_results['spread_stats']['up_spread']['pct_gte_2c']:.1f}%, DOWN {audit_results['spread_stats']['down_spread']['pct_gte_2c']:.1f}%")
print("=" * 60)

