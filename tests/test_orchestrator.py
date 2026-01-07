"""Tests for Market Orchestrator module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import MarketOrchestrator, MarketState


class TestMarketState:
    """Tests for MarketState dataclass."""
    
    def test_default_state(self):
        """Test default state initialization."""
        state = MarketState(asset="BTC")
        
        assert state.asset == "BTC"
        assert state.market_id is None
        assert state.token_id_up is None
        assert state.token_id_down is None
        assert state.is_active is False
        assert state.collected_count == 0
        assert state.error_count == 0
    
    def test_state_with_values(self):
        """Test state with custom values."""
        now = datetime.now(timezone.utc)
        
        state = MarketState(
            asset="ETH",
            market_id="eth-updown-15m-12345",
            token_id_up="up_token",
            token_id_down="down_token",
            start_time=now,
            end_time=now + timedelta(minutes=15),
            is_active=True,
            collected_count=100
        )
        
        assert state.asset == "ETH"
        assert state.market_id == "eth-updown-15m-12345"
        assert state.is_active is True
        assert state.collected_count == 100


class TestMarketOrchestrator:
    """Tests for MarketOrchestrator class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for output."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        # Cleanup handled manually due to Windows file locking issues
        import shutil
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except:
            pass
    
    @pytest.fixture
    def orchestrator(self, temp_dir):
        """Create an orchestrator instance."""
        orch = MarketOrchestrator(
            assets=["BTC"],
            output_dir=temp_dir,
            preload_seconds=60,
            log_level=40  # ERROR level
        )
        yield orch
        # Close file handles before cleanup
        for fh in orch.file_handles.values():
            try:
                fh.close()
            except:
                pass
    
    def test_initialization(self, orchestrator):
        """Test orchestrator initializes correctly."""
        assert "BTC" in orchestrator.assets
        assert orchestrator.preload_seconds == 60
        assert orchestrator.running is False
    
    def test_market_states_created(self, orchestrator):
        """Test market states are created for each asset."""
        assert "BTC" in orchestrator.market_states
        state = orchestrator.market_states["BTC"]
        assert isinstance(state, MarketState)
        assert state.asset == "BTC"
    
    def test_output_files_created(self, orchestrator, temp_dir):
        """Test output files are initialized."""
        assert "BTC" in orchestrator.output_files
        filepath = orchestrator.output_files["BTC"]
        assert str(temp_dir) in str(filepath)
        assert "polymarket_btc_continuous.csv" in str(filepath)
    
    def test_get_state(self, orchestrator):
        """Test getting state for an asset."""
        state = orchestrator.get_state("BTC")
        assert state is not None
        assert state.asset == "BTC"
        
        state = orchestrator.get_state("INVALID")
        assert state is None
    
    def test_get_all_states(self, orchestrator):
        """Test getting all states."""
        states = orchestrator.get_all_states()
        assert "BTC" in states
        assert isinstance(states["BTC"], MarketState)


class TestMarketTransitions:
    """Tests for market transition logic."""
    
    @pytest.fixture
    def temp_dir(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        import shutil
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except:
            pass
    
    @pytest.fixture
    def orchestrator(self, temp_dir):
        orch = MarketOrchestrator(
            assets=["BTC"],
            output_dir=temp_dir,
            preload_seconds=60,
            log_level=40
        )
        yield orch
        for fh in orch.file_handles.values():
            try:
                fh.close()
            except:
                pass
    
    @patch.object(MarketOrchestrator, '_discover_market')
    def test_discover_market_called(self, mock_discover, orchestrator):
        """Test that discover market is called during transition check."""
        orchestrator.market_states["BTC"].is_active = False
        orchestrator._check_market_transitions()
        
        mock_discover.assert_called_once_with("BTC")
    
    def test_switch_to_next_market(self, orchestrator):
        """Test switching to next market."""
        state = orchestrator.market_states["BTC"]
        
        # Set up next market
        state.next_market_id = "btc-updown-15m-99999"
        state.next_token_id_up = "new_up"
        state.next_token_id_down = "new_down"
        state.next_start_time = datetime.now(timezone.utc)
        state.next_end_time = datetime.now(timezone.utc) + timedelta(minutes=15)
        state.next_preloaded = True
        state.collected_count = 100
        
        orchestrator._switch_to_next_market("BTC")
        
        # Verify switch happened
        assert state.market_id == "btc-updown-15m-99999"
        assert state.token_id_up == "new_up"
        assert state.token_id_down == "new_down"
        assert state.collected_count == 0  # Reset
        assert state.next_preloaded is False


class TestDataWriting:
    """Tests for data writing functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        import shutil
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except:
            pass
    
    @pytest.fixture
    def orchestrator(self, temp_dir):
        orch = MarketOrchestrator(
            assets=["BTC"],
            output_dir=temp_dir,
            log_level=40
        )
        yield orch
        for fh in orch.file_handles.values():
            try:
                fh.close()
            except:
                pass
    
    def test_write_data(self, orchestrator, temp_dir):
        """Test writing data to CSV."""
        data = {
            "collected_at": datetime.now(timezone.utc),
            "up_mid": 0.55,
            "up_best_bid": 0.53,
            "up_best_ask": 0.57,
            "down_mid": 0.45,
            "down_best_bid": 0.43,
            "down_best_ask": 0.47
        }
        
        orchestrator._write_data("BTC", data, "test-market-123")
        
        # Flush and check file
        orchestrator.file_handles["BTC"].flush()
        
        filepath = Path(temp_dir) / "polymarket_btc_continuous.csv"
        assert filepath.exists()
        
        # Read and verify content
        import pandas as pd
        df = pd.read_csv(filepath)
        assert len(df) == 1
        assert df.iloc[0]["asset"] == "BTC"
        assert df.iloc[0]["market_id"] == "test-market-123"
    
    def test_write_missing_data(self, orchestrator, temp_dir):
        """Test writing missing data placeholder."""
        last_data = {
            "up_mid": 0.55,
            "up_best_bid": 0.53,
            "up_best_ask": 0.57,
            "down_mid": 0.45,
            "down_best_bid": 0.43,
            "down_best_ask": 0.47
        }
        
        orchestrator._write_missing_data("BTC", "test-market-123", last_data)
        
        # Flush and check file
        orchestrator.file_handles["BTC"].flush()
        
        filepath = Path(temp_dir) / "polymarket_btc_continuous.csv"
        
        import pandas as pd
        df = pd.read_csv(filepath)
        assert len(df) == 1
        assert df.iloc[0]["is_observed"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

