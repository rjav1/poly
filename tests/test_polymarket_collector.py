"""Tests for Polymarket collector module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.polymarket.collector import PolymarketCollector


class TestPolymarketCollector:
    """Tests for PolymarketCollector class."""
    
    @pytest.fixture
    def collector(self):
        """Create a collector instance."""
        return PolymarketCollector(log_level=40)  # ERROR level to reduce noise
    
    def test_initialization(self, collector):
        """Test collector initializes correctly."""
        assert collector.base_url == "https://clob.polymarket.com"
        assert collector.timeout > 0
        assert collector.orderbook_depth > 0
    
    @patch.object(PolymarketCollector, 'get_midpoint')
    def test_get_midpoint_success(self, mock_get, collector):
        """Test successful midpoint retrieval."""
        mock_get.return_value = 0.55
        result = collector.get_midpoint("test_token_id")
        assert result == 0.55
    
    @patch.object(PolymarketCollector, 'get_midpoint')
    def test_get_midpoint_failure(self, mock_get, collector):
        """Test midpoint retrieval failure."""
        mock_get.return_value = None
        result = collector.get_midpoint("invalid_token")
        assert result is None
    
    @patch.object(PolymarketCollector, 'get_orderbook')
    def test_get_orderbook_success(self, mock_get, collector):
        """Test successful orderbook retrieval."""
        mock_book = {
            "bids": [{"price": "0.50", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}]
        }
        mock_get.return_value = mock_book
        result = collector.get_orderbook("test_token_id")
        assert result is not None
        assert "bids" in result
        assert "asks" in result
    
    @patch.object(PolymarketCollector, 'get_midpoint')
    @patch.object(PolymarketCollector, 'get_orderbook')
    def test_get_market_data_success(self, mock_book, mock_mid, collector):
        """Test successful market data retrieval."""
        mock_mid.return_value = 0.50
        mock_book.return_value = {
            "bids": [{"price": "0.48", "size": "100"}],
            "asks": [{"price": "0.52", "size": "100"}]
        }
        
        result = collector.get_market_data("up_token", "down_token")
        
        assert result is not None
        assert "collected_at" in result
        assert "up_mid" in result
        assert "down_mid" in result
        assert "up_best_bid" in result
        assert "up_best_ask" in result
    
    @patch.object(PolymarketCollector, 'get_midpoint')
    @patch.object(PolymarketCollector, 'get_orderbook')
    def test_get_market_data_failure(self, mock_book, mock_mid, collector):
        """Test market data retrieval when both midpoints fail."""
        mock_mid.return_value = None
        mock_book.return_value = None
        
        result = collector.get_market_data("up_token", "down_token")
        
        # Should return None when both midpoints fail
        assert result is None


class TestFindActiveMarket:
    """Tests for find_active_market functionality."""
    
    @pytest.fixture
    def collector(self):
        return PolymarketCollector(log_level=40)
    
    @patch.object(PolymarketCollector, 'get_market_by_slug')
    def test_find_active_market_btc(self, mock_get, collector):
        """Test finding active BTC market."""
        # Mock a valid active market response
        now = datetime.now(timezone.utc)
        end_time = now + timedelta(minutes=10)
        
        mock_get.return_value = {
            "market_slug": "btc-updown-15m-12345",
            "question": "Bitcoin Up or Down?",
            "token_id_up": "up_token",
            "token_id_down": "down_token",
            "end_date": end_time.isoformat(),
            "active": True,
            "closed": False
        }
        
        result = collector.find_active_market("BTC")
        
        assert result is not None
        assert result["asset"] == "BTC"
        assert "token_id_up" in result
        assert "token_id_down" in result
    
    @patch.object(PolymarketCollector, 'get_market_by_slug')
    def test_find_active_market_not_found(self, mock_get, collector):
        """Test when no active market is found."""
        mock_get.return_value = None
        
        result = collector.find_active_market("BTC")
        
        assert result is None
    
    def test_find_active_market_invalid_asset(self, collector):
        """Test with invalid asset returns None."""
        result = collector.find_active_market("INVALID")
        
        assert result is None


class TestFindNextMarket:
    """Tests for find_next_market functionality."""
    
    @pytest.fixture
    def collector(self):
        return PolymarketCollector(log_level=40)
    
    @patch.object(PolymarketCollector, 'get_market_by_slug')
    def test_find_next_market(self, mock_get, collector):
        """Test finding next market."""
        now = datetime.now(timezone.utc)
        end_time = now + timedelta(minutes=25)  # Next market
        
        mock_get.return_value = {
            "market_slug": "btc-updown-15m-12345",
            "question": "Bitcoin Up or Down?",
            "token_id_up": "up_token",
            "token_id_down": "down_token",
            "end_date": end_time.isoformat(),
            "active": True,
            "closed": False
        }
        
        result = collector.find_next_market("BTC")
        
        assert result is not None
        assert result["asset"] == "BTC"


class TestGetMarketTimes:
    """Tests for get_market_times functionality."""
    
    @pytest.fixture
    def collector(self):
        return PolymarketCollector(log_level=40)
    
    def test_get_market_times_valid(self, collector):
        """Test extracting times from valid market."""
        end_time = datetime(2026, 1, 4, 12, 30, 0, tzinfo=timezone.utc)
        market = {"end_date": end_time.isoformat()}
        
        start, end = collector.get_market_times(market)
        
        assert end == end_time
        assert start == end_time - timedelta(minutes=15)
    
    def test_get_market_times_missing_end(self, collector):
        """Test with missing end_date."""
        market = {}
        
        start, end = collector.get_market_times(market)
        
        assert start is None
        assert end is None
    
    def test_get_market_times_invalid_format(self, collector):
        """Test with invalid date format."""
        market = {"end_date": "invalid_date"}
        
        start, end = collector.get_market_times(market)
        
        assert start is None
        assert end is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

