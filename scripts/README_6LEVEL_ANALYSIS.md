# Using 6-Level Markets for Analysis

This guide explains how to use the 6-level order book markets for analysis with proper coverage filtering.

## Overview

- **6-Level Markets Folder**: `data_v2/markets_6levels/`
- **Coverage Report**: `data_v2/markets_6levels/coverage_report.json`
- **Total Markets**: 41 ETH markets with all 6 levels of order book depth
- **Markets with >=80% coverage**: 35 markets (recommended for analysis)
- **Markets with >=90% coverage**: 25 markets (high quality)

## Quick Start

### 1. View Coverage Report

```bash
python scripts/list_6level_markets_coverage.py
```

This shows:
- Exact coverage % for each market (CL, PM, Both)
- Which markets have >=80% coverage (recommended)
- Which markets have >=90% coverage (high quality)

### 2. Build Dataset with Coverage Filtering

**Option A: Use the convenience script (recommended)**

```bash
# Build with default 80% coverage threshold
python scripts/build_6level_dataset.py

# Build with 90% coverage threshold (higher quality)
python scripts/build_6level_dataset.py --min-coverage 90

# Custom output directory
python scripts/build_6level_dataset.py --output-dir data_v2/research_6levels_90pct
```

**Option B: Use build script directly**

```bash
python scripts/build_research_dataset_v2.py \
    --markets-dir data_v2/markets_6levels \
    --output-dir data_v2/research_6levels \
    --min-coverage 80
```

### 3. Filter Markets in Your Analysis Code

When loading data programmatically, filter by coverage:

```python
import json
import pandas as pd
from pathlib import Path

# Load coverage report
coverage_path = Path('data_v2/markets_6levels/coverage_report.json')
with open(coverage_path) as f:
    coverage_data = json.load(f)

# Get markets with >=80% coverage
min_coverage = 80.0
valid_markets = [
    m['market_id'] 
    for m in coverage_data['markets'] 
    if m['both_coverage'] >= min_coverage
]

print(f"Found {len(valid_markets)} markets with >=80% coverage")

# Filter your dataset
df = pd.read_parquet('data_v2/research_6levels/canonical_dataset_all_assets.parquet')
df_filtered = df[df['market_id'].isin(valid_markets)].copy()

print(f"Filtered dataset: {len(df_filtered):,} rows from {df_filtered['market_id'].nunique()} markets")
```

## Coverage Statistics

From the latest report:

- **Total 6-level markets**: 41
- **Markets with >=80% both coverage**: 35 (85.4%)
- **Markets with >=90% both coverage**: 25 (61.0%)
- **Average coverage**: 80.7% both, 88.7% CL, 83.5% PM

## Market Quality Tiers

### Tier 1: Excellent (>=90% coverage)
25 markets - Highest quality, recommended for all analysis

### Tier 2: Good (80-90% coverage)
10 markets - Good quality, suitable for most analysis

### Tier 3: Acceptable (50-80% coverage)
1 market - May have gaps, use with caution

### Tier 4: Low (<50% coverage)
5 markets - Not recommended for analysis

## File Locations

- **6-Level Markets**: `data_v2/markets_6levels/ETH/{MARKET_ID}/`
- **Coverage Report**: `data_v2/markets_6levels/coverage_report.json`
- **Built Dataset (80% threshold)**: `data_v2/research_6levels/`
- **Original Markets**: `data_v2/markets/` (unchanged)

## Notes

- The `both_coverage` metric represents the intersection of CL and PM data (both sources have data for the same timestamp)
- Markets are automatically filtered during dataset building based on `--min-coverage` threshold
- You can always filter further in your analysis code if needed
- The coverage report is updated whenever you run `list_6level_markets_coverage.py`


