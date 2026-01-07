"""
Ground Truth Market Metadata and Label System.

This module provides:
1. Authoritative market master table matching Polymarket's exact resolution rules
2. Correct K (strike) and settlement computation
3. Outcome label validation against Polymarket resolved outcomes
4. Research-grade dataset labeling

Critical for avoiding "fake alpha" from timestamp or labeling errors.
"""

import json
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import csv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logger
from src.polymarket.collector import PolymarketCollector
from config.settings import STORAGE, SUPPORTED_ASSETS, get_asset_config


@dataclass
class MarketGroundTruth:
    """Ground truth metadata for a single market."""
    
    # Market identification
    market_id: str              # e.g., "btc-updown-15m-1767557700"
    asset: str                  # e.g., "BTC"
    
    # Timestamps (exact, from Polymarket API)
    start_timestamp: datetime   # Market start time
    end_timestamp: datetime     # Market end time
    
    # Strike and settlement (the critical numbers)
    strike_K: Optional[float]           # "Price to beat" - CL price at market start
    settlement_price: Optional[float]   # CL price at market end
    
    # Outcomes
    resolved_outcome: Optional[str]     # From Polymarket ("Up" or "Down")
    computed_outcome: Optional[str]     # Our computation based on CL data
    outcome_match: Optional[bool]       # resolved == computed
    
    # Polymarket token IDs
    token_id_up: Optional[str] = None
    token_id_down: Optional[str] = None
    
    # Additional metadata
    question: Optional[str] = None
    resolution_source: Optional[str] = None
    polymarket_url: Optional[str] = None
    
    # Data quality flags
    has_cl_at_start: bool = False       # Did we capture CL at exact start?
    has_cl_at_end: bool = False         # Did we capture CL at exact end?
    cl_start_offset_seconds: int = 0    # How many seconds off from exact start
    cl_end_offset_seconds: int = 0      # How many seconds off from exact end
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with ISO timestamps."""
        d = asdict(self)
        d['start_timestamp'] = self.start_timestamp.isoformat() if self.start_timestamp else None
        d['end_timestamp'] = self.end_timestamp.isoformat() if self.end_timestamp else None
        return d


class GroundTruthBuilder:
    """
    Builds and validates ground truth market metadata.
    """
    
    def __init__(self, log_level: int = 20):
        """Initialize ground truth builder."""
        self.logger = setup_logger("ground_truth", level=log_level)
        self.pm_collector = PolymarketCollector(log_level=log_level)
    
    def fetch_market_metadata(self, market_slug: str) -> Optional[Dict]:
        """
        Fetch market metadata from Polymarket API.
        
        Args:
            market_slug: Market slug (e.g., "btc-updown-15m-1767557700")
            
        Returns:
            Market metadata dictionary or None
        """
        return self.pm_collector.get_market_by_slug(market_slug)
    
    def extract_asset_from_slug(self, market_slug: str) -> str:
        """Extract asset symbol from market slug."""
        slug_lower = market_slug.lower()
        for asset in SUPPORTED_ASSETS:
            if asset.lower() in slug_lower:
                return asset
        return "UNKNOWN"
    
    def parse_market_times(self, market_data: Dict) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Parse start and end times from market data.
        
        Args:
            market_data: Market metadata from Polymarket API
            
        Returns:
            Tuple of (start_time, end_time)
        """
        end_date_str = market_data.get("end_date")
        if not end_date_str:
            return None, None
        
        try:
            end_time = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
            start_time = end_time - timedelta(minutes=15)
            return start_time, end_time
        except (ValueError, AttributeError) as e:
            self.logger.error(f"Error parsing market times: {e}")
            return None, None
    
    def get_resolved_outcome(self, market_data: Dict) -> Optional[str]:
        """
        Get the resolved outcome from Polymarket.
        
        For closed markets, Polymarket provides the outcome.
        
        Returns:
            "Up" or "Down" or None if not resolved
        """
        # Check if market is closed
        if not market_data.get("closed"):
            return None
        
        # Try to get outcome from market data
        # The outcome might be in different places depending on API version
        outcome = market_data.get("outcome")
        if outcome:
            return outcome.capitalize()
        
        # Try to get from resolution
        resolution = market_data.get("resolution")
        if resolution:
            return resolution.capitalize()
        
        return None
    
    def compute_strike_from_chainlink(
        self,
        df_cl: pd.DataFrame,
        start_time: datetime,
        tolerance_seconds: int = 5
    ) -> Tuple[Optional[float], int, bool]:
        """
        Compute strike price K from Chainlink data.
        
        The strike is the first Chainlink price at or after market start.
        
        Args:
            df_cl: Chainlink data DataFrame with 'collected_at' and 'mid' columns
            start_time: Market start timestamp
            tolerance_seconds: How many seconds after start to look
            
        Returns:
            Tuple of (strike_K, offset_seconds, has_exact)
        """
        if df_cl.empty or 'mid' not in df_cl.columns:
            return None, 0, False
        
        df = df_cl.copy()
        if 'collected_at' in df.columns:
            df['collected_at'] = pd.to_datetime(df['collected_at'], utc=True)
        
        # Find first CL price at or after start
        df_after_start = df[df['collected_at'] >= start_time].sort_values('collected_at')
        
        if df_after_start.empty:
            # Fall back to last price before start
            df_before_start = df[df['collected_at'] < start_time].sort_values('collected_at', ascending=False)
            if df_before_start.empty:
                return None, 0, False
            
            first_row = df_before_start.iloc[0]
            offset = (start_time - first_row['collected_at']).total_seconds()
            return first_row['mid'], int(offset), False
        
        first_row = df_after_start.iloc[0]
        offset = (first_row['collected_at'] - start_time).total_seconds()
        has_exact = offset <= tolerance_seconds
        
        return first_row['mid'], int(offset), has_exact
    
    def compute_settlement_from_chainlink(
        self,
        df_cl: pd.DataFrame,
        end_time: datetime,
        tolerance_seconds: int = 5
    ) -> Tuple[Optional[float], int, bool]:
        """
        Compute settlement price from Chainlink data.
        
        The settlement is the first Chainlink price at or after market end.
        
        Args:
            df_cl: Chainlink data DataFrame
            end_time: Market end timestamp
            tolerance_seconds: How many seconds after end to look
            
        Returns:
            Tuple of (settlement_price, offset_seconds, has_exact)
        """
        if df_cl.empty or 'mid' not in df_cl.columns:
            return None, 0, False
        
        df = df_cl.copy()
        if 'collected_at' in df.columns:
            df['collected_at'] = pd.to_datetime(df['collected_at'], utc=True)
        
        # Find first CL price at or after end
        df_after_end = df[df['collected_at'] >= end_time].sort_values('collected_at')
        
        if df_after_end.empty:
            # Fall back to last price before end
            df_before_end = df[df['collected_at'] < end_time].sort_values('collected_at', ascending=False)
            if df_before_end.empty:
                return None, 0, False
            
            first_row = df_before_end.iloc[0]
            offset = (end_time - first_row['collected_at']).total_seconds()
            return first_row['mid'], int(offset), False
        
        first_row = df_after_end.iloc[0]
        offset = (first_row['collected_at'] - end_time).total_seconds()
        has_exact = offset <= tolerance_seconds
        
        return first_row['mid'], int(offset), has_exact
    
    def compute_outcome(self, strike_K: float, settlement_price: float) -> str:
        """
        Compute outcome based on strike and settlement.
        
        According to Polymarket rules:
        - "Up" if settlement >= strike (price went up or stayed same)
        - "Down" if settlement < strike (price went down)
        
        Args:
            strike_K: Strike price at market start
            settlement_price: Price at market end
            
        Returns:
            "Up" or "Down"
        """
        return "Up" if settlement_price >= strike_K else "Down"
    
    def build_ground_truth(
        self,
        market_slug: str,
        df_cl: Optional[pd.DataFrame] = None
    ) -> Optional[MarketGroundTruth]:
        """
        Build complete ground truth for a market.
        
        Args:
            market_slug: Market slug
            df_cl: Optional Chainlink data for this market period
            
        Returns:
            MarketGroundTruth object or None if failed
        """
        # Fetch market metadata from Polymarket
        market_data = self.fetch_market_metadata(market_slug)
        if not market_data:
            self.logger.error(f"Could not fetch metadata for {market_slug}")
            return None
        
        # Parse times
        start_time, end_time = self.parse_market_times(market_data)
        if not start_time or not end_time:
            self.logger.error(f"Could not parse times for {market_slug}")
            return None
        
        # Extract asset
        asset = self.extract_asset_from_slug(market_slug)
        
        # Get resolved outcome
        resolved_outcome = self.get_resolved_outcome(market_data)
        
        # Compute strike and settlement from Chainlink (if data provided)
        strike_K = None
        settlement_price = None
        computed_outcome = None
        has_cl_at_start = False
        has_cl_at_end = False
        cl_start_offset = 0
        cl_end_offset = 0
        
        if df_cl is not None and not df_cl.empty:
            strike_K, cl_start_offset, has_cl_at_start = self.compute_strike_from_chainlink(
                df_cl, start_time
            )
            settlement_price, cl_end_offset, has_cl_at_end = self.compute_settlement_from_chainlink(
                df_cl, end_time
            )
            
            if strike_K is not None and settlement_price is not None:
                computed_outcome = self.compute_outcome(strike_K, settlement_price)
        
        # Check outcome match
        outcome_match = None
        if resolved_outcome and computed_outcome:
            outcome_match = resolved_outcome.lower() == computed_outcome.lower()
            if not outcome_match:
                self.logger.warning(
                    f"Outcome mismatch for {market_slug}: "
                    f"resolved={resolved_outcome}, computed={computed_outcome}"
                )
        
        return MarketGroundTruth(
            market_id=market_slug,
            asset=asset,
            start_timestamp=start_time,
            end_timestamp=end_time,
            strike_K=strike_K,
            settlement_price=settlement_price,
            resolved_outcome=resolved_outcome,
            computed_outcome=computed_outcome,
            outcome_match=outcome_match,
            token_id_up=market_data.get("token_id_up"),
            token_id_down=market_data.get("token_id_down"),
            question=market_data.get("question"),
            resolution_source=market_data.get("resolution_source"),
            polymarket_url=f"https://polymarket.com/event/{market_slug}",
            has_cl_at_start=has_cl_at_start,
            has_cl_at_end=has_cl_at_end,
            cl_start_offset_seconds=cl_start_offset,
            cl_end_offset_seconds=cl_end_offset
        )


class GroundTruthRepository:
    """
    Repository for storing and loading ground truth data.
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize repository."""
        self.storage_dir = Path(storage_dir or STORAGE.ground_truth_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("ground_truth_repo")
        
        self.master_file = self.storage_dir / "market_master.json"
    
    def save_market(self, gt: MarketGroundTruth):
        """Save a single market's ground truth."""
        data = self.load_all()
        data[gt.market_id] = gt.to_dict()
        self._save_data(data)
        self.logger.info(f"Saved ground truth for {gt.market_id}")
    
    def save_many(self, ground_truths: List[MarketGroundTruth]):
        """Save multiple markets' ground truth."""
        data = self.load_all()
        for gt in ground_truths:
            data[gt.market_id] = gt.to_dict()
        self._save_data(data)
        self.logger.info(f"Saved ground truth for {len(ground_truths)} markets")
    
    def load_all(self) -> Dict[str, Dict]:
        """Load all ground truth data."""
        if not self.master_file.exists():
            return {}
        
        with open(self.master_file, 'r') as f:
            return json.load(f)
    
    def load_market(self, market_id: str) -> Optional[Dict]:
        """Load ground truth for a specific market."""
        data = self.load_all()
        return data.get(market_id)
    
    def _save_data(self, data: Dict):
        """Save data to file."""
        with open(self.master_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_markets_by_asset(self, asset: str) -> List[Dict]:
        """Get all markets for a specific asset."""
        data = self.load_all()
        return [m for m in data.values() if m.get('asset', '').upper() == asset.upper()]
    
    def get_outcome_stats(self) -> Dict:
        """Get statistics about outcomes."""
        data = self.load_all()
        
        total = len(data)
        with_resolved = sum(1 for m in data.values() if m.get('resolved_outcome'))
        with_computed = sum(1 for m in data.values() if m.get('computed_outcome'))
        matches = sum(1 for m in data.values() if m.get('outcome_match') is True)
        mismatches = sum(1 for m in data.values() if m.get('outcome_match') is False)
        
        return {
            'total_markets': total,
            'with_resolved_outcome': with_resolved,
            'with_computed_outcome': with_computed,
            'outcome_matches': matches,
            'outcome_mismatches': mismatches,
            'match_rate': matches / (matches + mismatches) if (matches + mismatches) > 0 else None
        }
    
    def export_to_csv(self, output_path: Optional[str] = None) -> Path:
        """Export ground truth to CSV."""
        output_path = Path(output_path or (self.storage_dir / "market_master.csv"))
        
        data = self.load_all()
        if not data:
            self.logger.warning("No ground truth data to export")
            return output_path
        
        df = pd.DataFrame(list(data.values()))
        df.to_csv(output_path, index=False)
        self.logger.info(f"Exported {len(df)} markets to {output_path}")
        return output_path


def build_ground_truth_from_collected_data(markets_dir: Optional[str] = None) -> List[MarketGroundTruth]:
    """
    Build ground truth for all collected markets.
    
    Args:
        markets_dir: Directory containing market folders
        
    Returns:
        List of MarketGroundTruth objects
    """
    markets_dir = Path(markets_dir or STORAGE.markets_dir)
    logger = setup_logger("build_ground_truth")
    
    if not markets_dir.exists():
        logger.warning(f"Markets directory not found: {markets_dir}")
        return []
    
    builder = GroundTruthBuilder()
    ground_truths = []
    
    for market_folder in sorted(markets_dir.iterdir()):
        if not market_folder.is_dir():
            continue
        
        folder_name = market_folder.name
        logger.info(f"Processing market: {folder_name}")
        
        # Extract market slug from folder name
        # Format: YYYYMMDD_HHMM or YYYYMMDD_HHMM_price
        parts = folder_name.split('_')
        if len(parts) < 2:
            continue
        
        # Try to find market slug from the timestamp
        try:
            # Parse date/time from folder name
            date_str = parts[0]  # YYYYMMDD
            time_str = parts[1]  # HHMM
            dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
            dt = dt.replace(tzinfo=timezone.utc)
            timestamp = int(dt.timestamp())
            
            # Determine asset from folder contents
            asset = "BTC"  # Default
            for f in market_folder.glob("chainlink_*.csv"):
                # Could parse asset from filename
                break
            
            config = get_asset_config(asset)
            market_slug = f"{config.polymarket_slug_prefix}-{timestamp}"
            
        except Exception as e:
            logger.warning(f"Could not parse folder {folder_name}: {e}")
            continue
        
        # Load Chainlink data for this market
        df_cl = None
        cl_files = list(market_folder.glob("chainlink_*.csv"))
        if cl_files:
            try:
                df_cl = pd.read_csv(cl_files[0])
                if 'collected_at' in df_cl.columns:
                    df_cl['collected_at'] = pd.to_datetime(df_cl['collected_at'], format='ISO8601')
            except Exception as e:
                logger.warning(f"Could not load Chainlink data: {e}")
        
        # Build ground truth
        gt = builder.build_ground_truth(market_slug, df_cl)
        if gt:
            ground_truths.append(gt)
            logger.info(
                f"Built ground truth for {market_slug}: "
                f"K={gt.strike_K}, settlement={gt.settlement_price}, "
                f"computed={gt.computed_outcome}, match={gt.outcome_match}"
            )
    
    return ground_truths


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

def main():
    """Build ground truth from collected data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build ground truth market metadata")
    parser.add_argument(
        "--markets-dir",
        type=str,
        default=None,
        help="Directory containing market folders"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for ground truth files"
    )
    
    args = parser.parse_args()
    
    # Build ground truth
    ground_truths = build_ground_truth_from_collected_data(args.markets_dir)
    
    if not ground_truths:
        print("No ground truth data built")
        return
    
    # Save to repository
    repo = GroundTruthRepository(args.output_dir)
    repo.save_many(ground_truths)
    
    # Export to CSV
    repo.export_to_csv()
    
    # Print stats
    stats = repo.get_outcome_stats()
    print("\nGround Truth Statistics:")
    print(f"  Total markets: {stats['total_markets']}")
    print(f"  With resolved outcome: {stats['with_resolved_outcome']}")
    print(f"  With computed outcome: {stats['with_computed_outcome']}")
    print(f"  Outcome matches: {stats['outcome_matches']}")
    print(f"  Outcome mismatches: {stats['outcome_mismatches']}")
    if stats['match_rate'] is not None:
        print(f"  Match rate: {stats['match_rate']:.1%}")


if __name__ == "__main__":
    main()

