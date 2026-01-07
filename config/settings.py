"""
Central configuration for the data collection pipeline.

All settings are defined here for easy management.
Supports multi-asset collection: BTC, ETH, SOL, XRP.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, List


# =============================================================================
# ASSET CONFIGURATION
# =============================================================================

@dataclass
class AssetConfig:
    """Configuration for a single asset (BTC, ETH, SOL, XRP)."""
    symbol: str                    # "BTC", "ETH", "SOL", "XRP"
    name: str                      # Full name for display
    chainlink_url: str             # Chainlink data stream URL
    polymarket_slug_prefix: str    # e.g., "btc-updown-15m"
    price_precision: int = 2       # Decimal places for price display
    

# Define all supported assets
ASSETS: Dict[str, AssetConfig] = {
    "BTC": AssetConfig(
        symbol="BTC",
        name="Bitcoin",
        chainlink_url="https://data.chain.link/streams/btc-usd-cexprice-streams",
        polymarket_slug_prefix="btc-updown-15m",
        price_precision=2,
    ),
    "ETH": AssetConfig(
        symbol="ETH",
        name="Ethereum",
        chainlink_url="https://data.chain.link/streams/eth-usd-cexprice-streams",
        polymarket_slug_prefix="eth-updown-15m",
        price_precision=2,
    ),
    "SOL": AssetConfig(
        symbol="SOL",
        name="Solana",
        chainlink_url="https://data.chain.link/streams/sol-usd-cexprice-streams",
        polymarket_slug_prefix="sol-updown-15m",
        price_precision=2,
    ),
    "XRP": AssetConfig(
        symbol="XRP",
        name="XRP",
        chainlink_url="https://data.chain.link/streams/xrp-usd-cexprice-streams",
        polymarket_slug_prefix="xrp-updown-15m",
        price_precision=4,  # XRP has lower price, need more precision
    ),
}

# List of all supported asset symbols
SUPPORTED_ASSETS: List[str] = list(ASSETS.keys())


def get_asset_config(symbol: str) -> AssetConfig:
    """
    Get configuration for a specific asset.
    
    Args:
        symbol: Asset symbol (BTC, ETH, SOL, XRP)
        
    Returns:
        AssetConfig for the asset
        
    Raises:
        ValueError: If asset symbol is not supported
    """
    symbol = symbol.upper()
    if symbol not in ASSETS:
        raise ValueError(f"Unsupported asset: {symbol}. Supported: {SUPPORTED_ASSETS}")
    return ASSETS[symbol]


# =============================================================================
# CHAINLINK CONFIGURATION
# =============================================================================

@dataclass
class ChainlinkConfig:
    """Chainlink API configuration."""
    base_url: str = "https://data.chain.link/api"
    feed_id: str = "0x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b8"
    price_divisor: float = 1e18  # Prices are in wei-like format
    request_timeout: int = 10
    
    # Continuous collection settings
    collection_interval: float = 1.0  # Collect every second (1 Hz)
    max_extraction_retries: int = 3   # Retries per extraction attempt
    page_load_timeout: int = 30       # Browser page load timeout


# =============================================================================
# POLYMARKET CONFIGURATION
# =============================================================================

@dataclass
class PolymarketConfig:
    """Polymarket API configuration."""
    base_url: str = "https://clob.polymarket.com"
    gamma_api_url: str = "https://gamma-api.polymarket.com"
    request_timeout: int = 10
    orderbook_depth: int = 100  # Number of levels to capture
    
    # Market transition settings
    preload_seconds: int = 60   # Start loading next market 60s before current ends
    market_duration: int = 900  # 15 minutes = 900 seconds


# =============================================================================
# STORAGE CONFIGURATION
# =============================================================================

@dataclass
class StorageConfig:
    """Data storage configuration."""
    base_dir: str = "data"
    # V2 organized structure
    v2_base_dir: str = "data_v2"
    raw_dir: str = "data_v2/raw"  # Organized by asset
    processed_dir: str = "data_v2/processed"
    research_dir: str = "data_v2/research_6levels"  # 6-level research dataset (standard)
    live_dir: str = "data_v2/live"
    markets_dir: str = "data_v2/markets_6levels"  # Organized by asset/market (6-level data standard)
    ground_truth_dir: str = "data_v2/ground_truth"
    # Legacy paths (for backwards compatibility)
    legacy_raw_dir: str = "data/raw"
    legacy_processed_dir: str = "data/processed"
    legacy_research_dir: str = "data/research"
    legacy_live_dir: str = "data/live"
    legacy_markets_dir: str = "data/markets"
    format: str = "parquet"  # Options: "parquet", "csv"


# =============================================================================
# COLLECTION CONFIGURATION
# =============================================================================

@dataclass
class CollectionConfig:
    """Data collection configuration."""
    # Collection intervals (in seconds)
    chainlink_interval: float = 1.0  # Collect Chainlink data every second
    polymarket_interval: float = 1.0  # Collect Polymarket data every second
    
    # How long to run collection (in seconds, None = until stopped)
    collection_duration: Optional[int] = None
    
    # Market settings
    market_duration: int = 900  # 15 minutes = 900 seconds
    
    # Error handling
    max_consecutive_errors: int = 10  # Stop after this many consecutive errors
    
    # Logging
    log_interval: int = 10  # Log stats every N seconds


# =============================================================================
# MARKET CONFIGURATION (Legacy - for backwards compatibility)
# =============================================================================

@dataclass
class MarketConfig:
    """Configuration for the specific market being tracked (legacy)."""
    url: str = "https://polymarket.com/event/btc-updown-15m-1767482100"
    name: str = "Bitcoin Up or Down"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    token_id_up: Optional[str] = None
    token_id_down: Optional[str] = None
    market_slug: str = "btc-updown-15m-1767482100"


# =============================================================================
# GLOBAL CONFIGURATION INSTANCES
# =============================================================================

CHAINLINK = ChainlinkConfig()
POLYMARKET = PolymarketConfig()
MARKET = MarketConfig()
STORAGE = StorageConfig()
COLLECTION = CollectionConfig()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_chainlink_url(asset: str) -> str:
    """Get Chainlink URL for an asset."""
    return get_asset_config(asset).chainlink_url


def get_polymarket_slug_prefix(asset: str) -> str:
    """Get Polymarket slug prefix for an asset."""
    return get_asset_config(asset).polymarket_slug_prefix


def generate_market_slug(asset: str, timestamp: int) -> str:
    """
    Generate a market slug for a given asset and timestamp.
    
    Args:
        asset: Asset symbol (BTC, ETH, SOL, XRP)
        timestamp: Unix timestamp for market start
        
    Returns:
        Market slug (e.g., "btc-updown-15m-1767557700")
    """
    prefix = get_polymarket_slug_prefix(asset)
    return f"{prefix}-{timestamp}"
