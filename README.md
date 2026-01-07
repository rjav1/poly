# Polymarket BTC 15-min Data Collection Pipeline

Production-ready data collection system for Polymarket BTC 15-minute "Up/Down" prediction markets and Chainlink oracle prices.

## Overview

This pipeline collects synchronized data from:
- **Chainlink**: BTC/USD price data from Data Streams (via GraphQL API)
- **Polymarket**: Market data including midpoints, orderbooks, and best bid/ask

The goal is to build a clean, time-synced dataset for predictive modeling of short-term crypto price prediction markets.

## Features

- ✅ **Automatic Market Discovery** - Finds the currently active 15-min market
- ✅ **Smart Market Switching** - Automatically switches to new markets when current one closes
- ✅ **Time-Synchronized Collection** - Both data sources collected with precise timestamps
- ✅ **Production Logging** - Daily log files with rotation
- ✅ **Professional Visualizations** - Matplotlib dashboards and charts
- ✅ **Settlement Analysis** - Test boundary rules for market resolution
- ✅ **Correlation Analysis** - Price change vs probability change correlation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test the Pipeline

```bash
python scripts/test_pipeline.py
```

This verifies both Chainlink and Polymarket APIs are working.

### 3. Start Data Collection

**Smart Mode (Recommended)** - Automatically switches markets when they close:

```bash
# Collect for 15 minutes (will span multiple markets if needed)
python scripts/collect.py --duration 900 --smart

# Collect indefinitely with auto-switching
python scripts/collect.py --smart
```

**Standard Mode** - Single market only:

```bash
# Automatically finds and uses the current active market
python scripts/collect.py --duration 900

# Specify token IDs manually
python scripts/collect.py --token-up "TOKEN_ID" --token-down "TOKEN_ID" --duration 900
```

### 4. Generate Visualizations

```bash
# Create matplotlib charts (saved to data/plots/)
python scripts/visualize.py

# Or show interactively
python scripts/visualize.py --show
```

### 5. Run Analysis

```bash
# Diagnostic report (correlation, lag analysis)
python scripts/diagnostic_report.py

# Settlement rule analysis  
python scripts/settlement_analysis.py

# ASCII plots (quick terminal view)
python scripts/plot_data.py
```

## Data Storage

Data is saved to `data/raw/`:
- `data/raw/chainlink/prices_YYYY-MM-DD.parquet` - Chainlink price data
- `data/raw/polymarket/market_data_YYYY-MM-DD.parquet` - Polymarket market data

Processed data:
- `data/processed/merged_timeseries.csv` - Combined time-synced dataset
- `data/processed/diagnostic_report.json` - Analysis results

Visualizations:
- `data/plots/dashboard_*.png` - Full dashboard
- `data/plots/price_probability_*.png` - Price vs probability overlay
- `data/plots/oracle_lag_*.png` - Lag distribution
- `data/plots/correlation_*.png` - Price change vs prob change scatter

## Project Structure

```
poly/
├── config/
│   └── settings.py              # Central configuration
├── src/
│   ├── analysis/
│   │   ├── settlement.py        # Settlement rule analyzer
│   │   └── diagnostics.py       # Correlation & lag analysis
│   ├── chainlink/
│   │   └── collector.py         # Chainlink GraphQL API collector
│   ├── polymarket/
│   │   └── collector.py         # Polymarket CLOB API collector
│   ├── utils/
│   │   └── logging.py           # Logging utilities
│   ├── collector.py             # Standard data orchestrator
│   └── smart_collector.py       # Smart collector with market switching
├── scripts/
│   ├── test_pipeline.py         # Test all components
│   ├── discover_market.py       # Find market token IDs  
│   ├── collect.py               # Main collection script
│   ├── visualize.py             # Matplotlib visualizations
│   ├── analyze_data.py          # Data quality analysis
│   ├── diagnostic_report.py     # Correlation & lag report
│   ├── settlement_analysis.py   # Settlement rule testing
│   └── plot_data.py             # ASCII visualization
├── data/
│   ├── raw/                     # Raw parquet files
│   ├── processed/               # Merged CSV & reports
│   └── plots/                   # Generated charts
├── logs/                        # Daily log files
└── requirements.txt
```

## Smart Collection Mode

The `--smart` flag enables intelligent market switching:

1. **Auto-Discovery**: Finds the currently active BTC 15-min market
2. **End Detection**: Monitors for market closure (via end time or consecutive API errors)
3. **Seamless Switching**: Automatically switches to the next market
4. **Continuous Chainlink**: Oracle data collection continues uninterrupted
5. **Per-Market Stats**: Tracks statistics separately for each market collected

Example: If you set `--duration 1200` (20 minutes) at 7:10 PM:
- Collects from 7:10-7:15 PM market
- Switches to 7:15-7:30 PM market automatically
- Continues until total duration reached

## Data Schema

### Chainlink Data

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | Price report timestamp (UTC) |
| price | float | BTC/USD price |
| bid | float | Bid price |
| ask | float | Ask price |
| mid | float | Mid price |
| collected_at | datetime | When data was collected |

### Polymarket Data

| Column | Type | Description |
|--------|------|-------------|
| collected_at | datetime | Collection timestamp (UTC) |
| up_mid | float | UP outcome midpoint (0-1) |
| up_best_bid | float | Best bid for UP |
| up_best_ask | float | Best ask for UP |
| down_mid | float | DOWN outcome midpoint (0-1) |
| down_best_bid | float | Best bid for DOWN |
| down_best_ask | float | Best ask for DOWN |

### Merged Time-Series

| Column | Type | Description |
|--------|------|-------------|
| time | datetime | Observation timestamp |
| cl_timestamp | datetime | Chainlink report timestamp |
| cl_price | float | BTC/USD price |
| pm_up_mid | float | UP probability |
| pm_down_mid | float | DOWN probability |
| oracle_age | float | Seconds since last Chainlink update |
| cl_price_change | float | Price change from previous |
| pm_up_change | float | Probability change from previous |

## API Sources

### Chainlink
- **Endpoint**: `https://data.chain.link/api/query-timescale`
- **Query**: `LIVE_STREAM_REPORTS_QUERY`
- **Feed ID**: BTC/USD Data Stream
- **Note**: Prices are returned as large integers (divide by 10^18)

### Polymarket
- **CLOB API**: `https://clob.polymarket.com`
  - `/midpoint?token_id=` - Get current midpoint
  - `/book?token_id=` - Get full orderbook
- **Gamma API**: `https://gamma-api.polymarket.com`
  - `/markets/slug/{slug}` - Get market details & token IDs

## Market Resolution

BTC 15-min markets resolve based on Chainlink BTC/USD:
- **UP**: End price ≥ Start price
- **DOWN**: End price < Start price

Resolution source: https://data.chain.link/streams/btc-usd

## Logs

Logs are written to:
- Console (INFO level)
- `logs/` directory (daily files)

## Analysis Results

### Typical Metrics

| Metric | Typical Value | Description |
|--------|--------------|-------------|
| Correlation (lag 0) | 0.90-0.95 | Price change vs prob change |
| Oracle Age (mean) | 0.3-0.5s | Average staleness |
| Oracle Age (P95) | 5-8s | 95th percentile staleness |
| Collection Rate | ~1/second | Data points per second |

### Interpreting Results

- **High correlation (>0.9)**: Market responds quickly to price changes
- **Low oracle_age (<1s mean)**: Fresh oracle data available
- **Negative correlation at lag N**: Market may lead/lag oracle by N observations

## Step 1 Deliverables Status

1. ✅ **Chainlink price collection** - Working via GraphQL API
2. ✅ **Polymarket market data collection** - Working via CLOB/Gamma API
3. ✅ **Time-synchronized data storage** - Parquet files with timestamps
4. ✅ **Settlement rule reconstruction** - `settlement_analysis.py` tests all boundary rules
5. ✅ **Boundary rule verification** - Compare predicted vs actual outcomes
6. ✅ **Diagnostic plots** - Matplotlib charts + ASCII + CSV export
7. ✅ **Merged time-series** - `merged_timeseries.csv` with oracle_age
8. ✅ **Quality metrics** - Correlation analysis, oracle_age stats
9. ✅ **Smart market switching** - Auto-switch when markets close

## License

MIT
