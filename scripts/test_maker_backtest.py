#!/usr/bin/env python3
"""
Test script for maker backtest engine.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backtest.data_loader import load_eth_markets, add_derived_columns
from scripts.backtest.strategies import SpreadCaptureStrategy
from scripts.backtest.backtest_engine import run_maker_backtest
from scripts.backtest.maker_execution import MakerExecutionConfig, FillModel

print('Loading data...')
df, market_info = load_eth_markets(min_coverage=90.0)
df = add_derived_columns(df)
n_markets = df['market_id'].nunique()
print(f'Loaded {len(df)} rows, {n_markets} markets')

# Create strategy
strategy = SpreadCaptureStrategy(
    spread_min=0.02,
    tau_min=120,
    tau_max=600,
    inventory_limit_up=10.0,
    inventory_limit_down=10.0,
    tau_flatten=60,
    quote_size=1.0,
    two_sided=True,
)

# Create config
config = MakerExecutionConfig(
    place_latency_ms=100,
    cancel_latency_ms=50,
    fill_model=FillModel.TOUCH_SIZE_PROXY,
    touch_trade_rate_per_second=0.1,
)

print(f'\nRunning backtest: {strategy.name}')
print(f'Config: {config.describe()}')

result = run_maker_backtest(df, strategy, config, verbose=True, volume_markets_only=True)

print('\n[OK] Maker backtest completed')

