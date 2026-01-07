#!/usr/bin/env python3
"""Verify that processed markets have 6 levels of order book depth."""

import pandas as pd
from pathlib import Path

markets_dir = Path('data_v2/markets/ETH')
markets = sorted([d for d in markets_dir.iterdir() if d.is_dir()])

if not markets:
    print("No markets found")
    exit(1)

latest = markets[-1]
pm_file = latest / 'polymarket.csv'

if not pm_file.exists():
    print(f"No polymarket.csv found in {latest.name}")
    exit(1)

df = pd.read_csv(pm_file, nrows=1)
print(f"Market: {latest.name}")
print(f"Total columns: {len(df.columns)}")

# Check for Level 6 columns
level_6_cols = ['up_bid_6', 'up_ask_6', 'up_bid_6_size', 'up_ask_6_size',
                'down_bid_6', 'down_ask_6', 'down_bid_6_size', 'down_ask_6_size']
has_level_6 = all(c in df.columns for c in level_6_cols)

print(f"Has Level 6 columns: {has_level_6}")

if has_level_6:
    print("\nSample Level 6 data (UP token):")
    level_cols = ['up_bid', 'up_bid_2', 'up_bid_3', 'up_bid_4', 'up_bid_5', 'up_bid_6']
    available_cols = [c for c in level_cols if c in df.columns]
    if available_cols:
        print(df[available_cols].iloc[0].to_dict())
    
    print("\nSample Level 6 data (DOWN token):")
    level_cols = ['down_bid', 'down_bid_2', 'down_bid_3', 'down_bid_4', 'down_bid_5', 'down_bid_6']
    available_cols = [c for c in level_cols if c in df.columns]
    if available_cols:
        print(df[available_cols].iloc[0].to_dict())
else:
    print("ERROR: Level 6 columns missing!")
    missing = [c for c in level_6_cols if c not in df.columns]
    print(f"Missing columns: {missing}")

