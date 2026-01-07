"""Tests for Ground Truth module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import tempfile
import json
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ground_truth import (
    MarketGroundTruth, GroundTruthBuilder, GroundTruthRepository
)


class TestMarketGroundTruth:
    """Tests for MarketGroundTruth dataclass."""
    
    def test_basic_creation(self):
        """Test basic creation of ground truth."""
        gt = MarketGroundTruth(
            market_id="btc-updown-15m-12345",
            asset="BTC",
            start_timestamp=datetime(2026, 1, 4, 12, 0, tzinfo=timezone.utc),
            end_timestamp=datetime(2026, 1, 4, 12, 15, tzinfo=timezone.utc),
            strike_K=91234.56,
            settlement_price=91500.00,
            resolved_outcome="Up",
            computed_outcome="Up",
            outcome_match=True
        )
        
        assert gt.market_id == "btc-updown-15m-12345"
        assert gt.asset == "BTC"
        assert gt.strike_K == 91234.56
        assert gt.outcome_match is True
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        start = datetime(2026, 1, 4, 12, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 4, 12, 15, tzinfo=timezone.utc)
        
        gt = MarketGroundTruth(
            market_id="btc-updown-15m-12345",
            asset="BTC",
            start_timestamp=start,
            end_timestamp=end,
            strike_K=91234.56,
            settlement_price=91500.00,
            resolved_outcome="Up",
            computed_outcome="Up",
            outcome_match=True
        )
        
        d = gt.to_dict()
        
        assert isinstance(d, dict)
        assert d["market_id"] == "btc-updown-15m-12345"
        # Timestamps should be ISO strings
        assert isinstance(d["start_timestamp"], str)
        assert isinstance(d["end_timestamp"], str)


class TestGroundTruthBuilder:
    """Tests for GroundTruthBuilder class."""
    
    @pytest.fixture
    def builder(self):
        return GroundTruthBuilder(log_level=40)
    
    def test_extract_asset_from_slug(self, builder):
        """Test extracting asset from slug."""
        assert builder.extract_asset_from_slug("btc-updown-15m-12345") == "BTC"
        assert builder.extract_asset_from_slug("eth-updown-15m-12345") == "ETH"
        assert builder.extract_asset_from_slug("sol-updown-15m-12345") == "SOL"
        assert builder.extract_asset_from_slug("xrp-updown-15m-12345") == "XRP"
        assert builder.extract_asset_from_slug("unknown-market") == "UNKNOWN"
    
    def test_compute_outcome(self, builder):
        """Test outcome computation."""
        # Up case: settlement >= strike
        assert builder.compute_outcome(91000, 91500) == "Up"
        assert builder.compute_outcome(91000, 91000) == "Up"  # Equal is Up
        
        # Down case: settlement < strike
        assert builder.compute_outcome(91000, 90500) == "Down"
    
    def test_compute_strike_from_chainlink(self, builder):
        """Test strike computation from Chainlink data."""
        start_time = datetime(2026, 1, 4, 12, 0, 0, tzinfo=timezone.utc)
        
        # Create sample CL data
        df_cl = pd.DataFrame({
            'collected_at': [
                start_time - timedelta(seconds=5),
                start_time + timedelta(seconds=1),
                start_time + timedelta(seconds=2)
            ],
            'mid': [91000.0, 91100.0, 91200.0]
        })
        
        K, offset, has_exact = builder.compute_strike_from_chainlink(df_cl, start_time)
        
        assert K == 91100.0  # First price at/after start
        assert offset <= 2
        assert has_exact is True
    
    def test_compute_strike_empty_data(self, builder):
        """Test strike computation with empty data."""
        df_cl = pd.DataFrame(columns=['collected_at', 'mid'])
        start_time = datetime(2026, 1, 4, 12, 0, tzinfo=timezone.utc)
        
        K, offset, has_exact = builder.compute_strike_from_chainlink(df_cl, start_time)
        
        assert K is None
        assert has_exact is False
    
    def test_compute_settlement_from_chainlink(self, builder):
        """Test settlement computation from Chainlink data."""
        end_time = datetime(2026, 1, 4, 12, 15, 0, tzinfo=timezone.utc)
        
        df_cl = pd.DataFrame({
            'collected_at': [
                end_time - timedelta(seconds=2),
                end_time + timedelta(seconds=1),
                end_time + timedelta(seconds=5)
            ],
            'mid': [91000.0, 91500.0, 91600.0]
        })
        
        settlement, offset, has_exact = builder.compute_settlement_from_chainlink(df_cl, end_time)
        
        assert settlement == 91500.0  # First price at/after end
        assert offset <= 2
        assert has_exact is True


class TestGroundTruthRepository:
    """Tests for GroundTruthRepository class."""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def repo(self, temp_dir):
        return GroundTruthRepository(storage_dir=temp_dir)
    
    def test_save_market(self, repo):
        """Test saving a single market."""
        gt = MarketGroundTruth(
            market_id="btc-updown-15m-12345",
            asset="BTC",
            start_timestamp=datetime(2026, 1, 4, 12, 0, tzinfo=timezone.utc),
            end_timestamp=datetime(2026, 1, 4, 12, 15, tzinfo=timezone.utc),
            strike_K=91234.56,
            settlement_price=91500.00,
            resolved_outcome="Up",
            computed_outcome="Up",
            outcome_match=True
        )
        
        repo.save_market(gt)
        
        # Verify saved
        data = repo.load_all()
        assert "btc-updown-15m-12345" in data
    
    def test_save_many(self, repo):
        """Test saving multiple markets."""
        ground_truths = [
            MarketGroundTruth(
                market_id=f"btc-updown-15m-{i}",
                asset="BTC",
                start_timestamp=datetime(2026, 1, 4, 12, 0, tzinfo=timezone.utc),
                end_timestamp=datetime(2026, 1, 4, 12, 15, tzinfo=timezone.utc),
                strike_K=91000 + i,
                settlement_price=91000 + i + 100,
                resolved_outcome="Up",
                computed_outcome="Up",
                outcome_match=True
            )
            for i in range(5)
        ]
        
        repo.save_many(ground_truths)
        
        data = repo.load_all()
        assert len(data) == 5
    
    def test_load_market(self, repo):
        """Test loading a specific market."""
        # First save
        gt = MarketGroundTruth(
            market_id="test-market-123",
            asset="BTC",
            start_timestamp=datetime(2026, 1, 4, 12, 0, tzinfo=timezone.utc),
            end_timestamp=datetime(2026, 1, 4, 12, 15, tzinfo=timezone.utc),
            strike_K=91000,
            settlement_price=91500,
            resolved_outcome="Up",
            computed_outcome="Up",
            outcome_match=True
        )
        repo.save_market(gt)
        
        # Then load
        loaded = repo.load_market("test-market-123")
        
        assert loaded is not None
        assert loaded["market_id"] == "test-market-123"
        assert loaded["strike_K"] == 91000
    
    def test_load_nonexistent_market(self, repo):
        """Test loading a market that doesn't exist."""
        loaded = repo.load_market("nonexistent-market")
        assert loaded is None
    
    def test_get_markets_by_asset(self, repo):
        """Test getting markets by asset."""
        # Save markets for different assets with unique IDs
        markets = [
            ("BTC", 12345),
            ("ETH", 12345),
            ("BTC", 12346)  # Second BTC market with different ID
        ]
        for asset, timestamp in markets:
            gt = MarketGroundTruth(
                market_id=f"{asset.lower()}-updown-15m-{timestamp}",
                asset=asset,
                start_timestamp=datetime(2026, 1, 4, 12, 0, tzinfo=timezone.utc),
                end_timestamp=datetime(2026, 1, 4, 12, 15, tzinfo=timezone.utc),
                strike_K=91000,
                settlement_price=91500,
                resolved_outcome="Up",
                computed_outcome="Up",
                outcome_match=True
            )
            repo.save_market(gt)
        
        btc_markets = repo.get_markets_by_asset("BTC")
        eth_markets = repo.get_markets_by_asset("ETH")
        
        assert len(btc_markets) == 2
        assert len(eth_markets) == 1
    
    def test_get_outcome_stats(self, repo):
        """Test getting outcome statistics."""
        # Save some markets with different outcomes
        for i, outcome in enumerate(["Up", "Down", "Up"]):
            gt = MarketGroundTruth(
                market_id=f"btc-updown-15m-{i}",
                asset="BTC",
                start_timestamp=datetime(2026, 1, 4, 12, 0, tzinfo=timezone.utc),
                end_timestamp=datetime(2026, 1, 4, 12, 15, tzinfo=timezone.utc),
                strike_K=91000,
                settlement_price=91500 if outcome == "Up" else 90500,
                resolved_outcome=outcome,
                computed_outcome=outcome,
                outcome_match=True
            )
            repo.save_market(gt)
        
        stats = repo.get_outcome_stats()
        
        assert stats["total_markets"] == 3
        assert stats["with_resolved_outcome"] == 3
        assert stats["outcome_matches"] == 3
        assert stats["match_rate"] == 1.0
    
    def test_export_to_csv(self, repo, temp_dir):
        """Test exporting to CSV."""
        gt = MarketGroundTruth(
            market_id="btc-updown-15m-12345",
            asset="BTC",
            start_timestamp=datetime(2026, 1, 4, 12, 0, tzinfo=timezone.utc),
            end_timestamp=datetime(2026, 1, 4, 12, 15, tzinfo=timezone.utc),
            strike_K=91000,
            settlement_price=91500,
            resolved_outcome="Up",
            computed_outcome="Up",
            outcome_match=True
        )
        repo.save_market(gt)
        
        output_path = repo.export_to_csv()
        
        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert len(df) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

