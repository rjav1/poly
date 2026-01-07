"""Tests for Dataset Validation module."""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timezone
import tempfile
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.validate_dataset import (
    ValidationResult,
    check_coverage_sanity,
    check_timestamp_gaps,
    check_noarb_bounds,
    check_strike_consistency,
    check_ffill_reasonableness,
    check_data_completeness
)


class TestValidationResult:
    """Tests for ValidationResult class."""
    
    def test_pass_result(self):
        """Test passing validation result."""
        result = ValidationResult(
            name="Test Check",
            passed=True,
            message="All good"
        )
        
        assert result.name == "Test Check"
        assert result.passed is True
        assert "[PASS]" in str(result)
    
    def test_fail_result(self):
        """Test failing validation result."""
        result = ValidationResult(
            name="Test Check",
            passed=False,
            message="Something wrong",
            details={"error": "details here"}
        )
        
        assert result.passed is False
        assert "[FAIL]" in str(result)
        assert result.details["error"] == "details here"


class TestCoverageSanity:
    """Tests for coverage sanity check."""
    
    def test_valid_coverage(self):
        """Test with valid coverage values."""
        market_infos = [
            {"market_id": "m1", "cl_coverage_pct": 80, "pm_coverage_pct": 90, "both_coverage_pct": 75},
            {"market_id": "m2", "cl_coverage_pct": 85, "pm_coverage_pct": 85, "both_coverage_pct": 80},
        ]
        
        result = check_coverage_sanity(market_infos)
        
        assert result.passed is True
    
    def test_invalid_coverage(self):
        """Test with invalid coverage (both > min of cl, pm)."""
        market_infos = [
            {"market_id": "m1", "cl_coverage_pct": 70, "pm_coverage_pct": 80, "both_coverage_pct": 85},  # Invalid!
        ]
        
        result = check_coverage_sanity(market_infos)
        
        assert result.passed is False


class TestTimestampGaps:
    """Tests for timestamp gap check."""
    
    def test_continuous_data(self):
        """Test with continuous data (no gaps)."""
        # Create data with continuous t values
        data = {
            'market_id': ['m1'] * 10,
            't': list(range(10))
        }
        df = pd.DataFrame(data)
        
        result = check_timestamp_gaps(df, max_gap_seconds=2)
        
        assert result.passed is True
    
    def test_data_with_gaps(self):
        """Test with data containing gaps."""
        # Create data with a large gap
        data = {
            'market_id': ['m1'] * 5,
            't': [0, 1, 2, 10, 11]  # Gap of 8 seconds
        }
        df = pd.DataFrame(data)
        
        result = check_timestamp_gaps(df, max_gap_seconds=2)
        
        assert result.passed is False


class TestNoArbBounds:
    """Tests for no-arb bounds check."""
    
    def test_valid_bounds(self):
        """Test with values within bounds."""
        df = pd.DataFrame({
            'sum_bids': [0.98, 0.99, 1.0, 1.01, 1.02],
            'sum_asks': [0.98, 0.99, 1.0, 1.01, 1.02]
        })
        
        result = check_noarb_bounds(df)
        
        assert result.passed == True
    
    def test_values_out_of_bounds(self):
        """Test with values outside bounds."""
        # More than 5% outside [0.90, 1.10]
        df = pd.DataFrame({
            'sum_bids': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],  # Many outside
            'sum_asks': [0.98, 0.99, 1.0, 1.01, 1.02, 1.0, 1.0, 1.0, 1.0, 1.0]
        })
        
        result = check_noarb_bounds(df)
        
        # Should fail because >5% of sum_bids are outside bounds
        assert result.passed == False


class TestStrikeConsistency:
    """Tests for strike consistency check."""
    
    def test_consistent_strikes(self):
        """Test with consistent strikes."""
        market_infos = [
            {"market_id": "m1", "price_to_beat_from_folder": 91000, "K": 91010},
            {"market_id": "m2", "price_to_beat_from_folder": 91500, "K": 91520},
        ]
        
        result = check_strike_consistency(market_infos, tolerance=50)
        
        assert result.passed is True
    
    def test_inconsistent_strikes(self):
        """Test with inconsistent strikes."""
        market_infos = [
            {"market_id": "m1", "price_to_beat_from_folder": 91000, "K": 92000},  # Off by 1000
        ]
        
        result = check_strike_consistency(market_infos, tolerance=50)
        
        assert result.passed is False
    
    def test_missing_values(self):
        """Test with missing values."""
        market_infos = [
            {"market_id": "m1", "price_to_beat_from_folder": None, "K": 91000},
        ]
        
        result = check_strike_consistency(market_infos)
        
        # Should be skipped (no comparisons possible)
        assert result.passed is True


class TestFFillReasonableness:
    """Tests for forward-fill reasonableness check."""
    
    def test_reasonable_ffill(self):
        """Test with reasonable forward-fill percentages."""
        df = pd.DataFrame({
            'cl_ffill': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # 20% ffill
            'pm_ffill': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # 20% ffill
        })
        
        result = check_ffill_reasonableness(df)
        
        assert result.passed == True
    
    def test_excessive_ffill(self):
        """Test with excessive forward-fill percentages."""
        df = pd.DataFrame({
            'cl_ffill': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 90% ffill
            'pm_ffill': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0% ffill
        })
        
        result = check_ffill_reasonableness(df)
        
        assert result.passed == False  # CL ffill is > 80%


class TestDataCompleteness:
    """Tests for data completeness check."""
    
    def test_complete_data(self):
        """Test with complete data."""
        market_infos = [
            {"market_id": f"m{i}", "cl_coverage_pct": 80, "pm_coverage_pct": 85}
            for i in range(15)
        ]
        
        result = check_data_completeness(market_infos)
        
        assert result.passed is True
    
    def test_incomplete_data(self):
        """Test with incomplete data."""
        market_infos = [
            {"market_id": f"m{i}", "cl_coverage_pct": 30, "pm_coverage_pct": 40}  # Poor coverage
            for i in range(5)
        ]
        
        result = check_data_completeness(market_infos)
        
        # Should fail because fewer than 10 markets with >50% coverage
        assert result.passed is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

