#!/usr/bin/env python3
"""Copy markets with 6 levels of order book data to a separate folder."""

import json
import pandas as pd
import shutil
from pathlib import Path

markets_dir = Path('data_v2/markets')
output_dir = Path('data_v2/markets_6levels')

print('=' * 100)
print('COPYING MARKETS WITH 6 LEVELS OF ORDER BOOK DATA')
print('=' * 100)
print()

# Expected columns for 6 levels
expected_levels = [1, 2, 3, 4, 5, 6]

def has_6_levels(market_dir: Path) -> bool:
    """Check if a market has all 6 levels of order book data."""
    pm_path = market_dir / 'polymarket.csv'
    
    if not pm_path.exists():
        return False
    
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
            
            if not all(col in columns for col in level_cols):
                return False
        
        return True
    except Exception as e:
        print(f"  Error checking {market_dir.name}: {e}")
        return False

# Collect markets with 6 levels
markets_to_copy = []
total_markets = 0

for asset_dir in sorted(markets_dir.iterdir()):
    if not asset_dir.is_dir():
        continue
    
    asset = asset_dir.name
    asset_markets = sorted([d for d in asset_dir.iterdir() if d.is_dir()])
    
    print(f"Checking {asset}: {len(asset_markets)} markets...")
    
    for market_dir in asset_markets:
        total_markets += 1
        if has_6_levels(market_dir):
            markets_to_copy.append((asset, market_dir))
            print(f"  [OK] {market_dir.name} - has 6 levels")

print()
print(f"Found {len(markets_to_copy)} markets with 6 levels out of {total_markets} total markets")
print()

# Create output directory structure
output_dir.mkdir(parents=True, exist_ok=True)

# Copy markets
copied_count = 0
skipped_count = 0

print("Copying markets...")
print("-" * 100)

for asset, market_dir in markets_to_copy:
    # Create asset directory in output
    asset_output_dir = output_dir / asset
    asset_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Destination path
    dest_market_dir = asset_output_dir / market_dir.name
    
    # Skip if already exists
    if dest_market_dir.exists():
        print(f"  [SKIP] {asset}/{market_dir.name} - already exists")
        skipped_count += 1
        continue
    
    # Copy the entire market folder
    try:
        shutil.copytree(market_dir, dest_market_dir)
        print(f"  [COPY] {asset}/{market_dir.name}")
        copied_count += 1
    except Exception as e:
        print(f"  [ERROR] {asset}/{market_dir.name}: {e}")
        skipped_count += 1

print("-" * 100)
print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"Total markets checked: {total_markets}")
print(f"Markets with 6 levels: {len(markets_to_copy)}")
print(f"Markets copied: {copied_count}")
print(f"Markets skipped (already exist): {skipped_count}")
print(f"\nOutput location: {output_dir.absolute()}")
print()

# Show breakdown by asset
if markets_to_copy:
    asset_counts = {}
    for asset, _ in markets_to_copy:
        asset_counts[asset] = asset_counts.get(asset, 0) + 1
    
    print("Markets with 6 levels by asset:")
    for asset in sorted(asset_counts.keys()):
        print(f"  {asset}: {asset_counts[asset]} markets")

print()


