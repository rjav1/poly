"""Tests for configuration module."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    ASSETS, SUPPORTED_ASSETS, get_asset_config, 
    get_chainlink_url, get_polymarket_slug_prefix,
    generate_market_slug, AssetConfig,
    CHAINLINK, POLYMARKET, STORAGE
)


class TestAssetConfig:
    """Tests for asset configuration."""
    
    def test_supported_assets(self):
        """Verify all expected assets are supported."""
        expected = ["BTC", "ETH", "SOL", "XRP"]
        assert SUPPORTED_ASSETS == expected
    
    def test_assets_dict_has_all_supported(self):
        """Verify ASSETS dict has all supported assets."""
        for asset in SUPPORTED_ASSETS:
            assert asset in ASSETS
            assert isinstance(ASSETS[asset], AssetConfig)
    
    def test_get_asset_config_valid(self):
        """Test getting config for valid assets."""
        for asset in SUPPORTED_ASSETS:
            config = get_asset_config(asset)
            assert config.symbol == asset
            assert config.chainlink_url.startswith("https://")
            assert config.polymarket_slug_prefix != ""
    
    def test_get_asset_config_invalid(self):
        """Test getting config for invalid asset raises error."""
        with pytest.raises(ValueError) as excinfo:
            get_asset_config("INVALID")
        assert "Unsupported asset" in str(excinfo.value)
    
    def test_get_asset_config_case_insensitive(self):
        """Test asset lookup is case insensitive."""
        config_upper = get_asset_config("BTC")
        config_lower = get_asset_config("btc")
        assert config_upper.symbol == config_lower.symbol
    
    def test_btc_config(self):
        """Test BTC specific configuration."""
        config = get_asset_config("BTC")
        assert config.symbol == "BTC"
        assert config.name == "Bitcoin"
        assert "btc" in config.chainlink_url.lower()
        assert config.polymarket_slug_prefix == "btc-updown-15m"
    
    def test_eth_config(self):
        """Test ETH specific configuration."""
        config = get_asset_config("ETH")
        assert config.symbol == "ETH"
        assert config.name == "Ethereum"
        assert "eth" in config.chainlink_url.lower()
        assert config.polymarket_slug_prefix == "eth-updown-15m"
    
    def test_sol_config(self):
        """Test SOL specific configuration."""
        config = get_asset_config("SOL")
        assert config.symbol == "SOL"
        assert config.name == "Solana"
        assert "sol" in config.chainlink_url.lower()
        assert config.polymarket_slug_prefix == "sol-updown-15m"
    
    def test_xrp_config(self):
        """Test XRP specific configuration."""
        config = get_asset_config("XRP")
        assert config.symbol == "XRP"
        assert "xrp" in config.chainlink_url.lower()
        assert config.polymarket_slug_prefix == "xrp-updown-15m"


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_get_chainlink_url(self):
        """Test getting Chainlink URL for assets."""
        for asset in SUPPORTED_ASSETS:
            url = get_chainlink_url(asset)
            assert url.startswith("https://data.chain.link/streams/")
            assert asset.lower() in url.lower()
    
    def test_get_polymarket_slug_prefix(self):
        """Test getting Polymarket slug prefix."""
        for asset in SUPPORTED_ASSETS:
            prefix = get_polymarket_slug_prefix(asset)
            assert asset.lower() in prefix.lower()
            assert "updown-15m" in prefix
    
    def test_generate_market_slug(self):
        """Test generating market slugs."""
        timestamp = 1767557700
        
        slug = generate_market_slug("BTC", timestamp)
        assert slug == "btc-updown-15m-1767557700"
        
        slug = generate_market_slug("ETH", timestamp)
        assert slug == "eth-updown-15m-1767557700"


class TestGlobalConfigs:
    """Tests for global configuration objects."""
    
    def test_chainlink_config(self):
        """Test Chainlink configuration."""
        assert CHAINLINK.base_url.startswith("https://")
        assert CHAINLINK.request_timeout > 0
        assert CHAINLINK.collection_interval > 0
    
    def test_polymarket_config(self):
        """Test Polymarket configuration."""
        assert POLYMARKET.base_url.startswith("https://")
        assert POLYMARKET.gamma_api_url.startswith("https://")
        assert POLYMARKET.orderbook_depth > 0
        assert POLYMARKET.preload_seconds > 0
        assert POLYMARKET.market_duration == 900
    
    def test_storage_config(self):
        """Test storage configuration."""
        assert STORAGE.base_dir == "data"
        assert STORAGE.raw_dir.startswith("data/")
        assert STORAGE.research_dir.startswith("data/")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

