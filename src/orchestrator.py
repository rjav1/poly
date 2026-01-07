"""
Market Orchestrator - Handles seamless market transitions for multi-asset collection.

Key features:
1. Pre-loads next market tokens before current market ends
2. Zero-gap transitions between 15-minute markets
3. Parallel collection for multiple assets
4. Automatic market discovery and switching

Usage:
    orchestrator = MarketOrchestrator(assets=["BTC", "ETH", "SOL", "XRP"])
    await orchestrator.start()  # Runs indefinitely
"""

import asyncio
import time
import csv
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Callable
from pathlib import Path
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.polymarket.collector import PolymarketCollector
from src.utils.logging import setup_logger
from config.settings import POLYMARKET, STORAGE, SUPPORTED_ASSETS, get_asset_config


@dataclass
class MarketState:
    """State for a single market."""
    asset: str
    market_id: Optional[str] = None
    token_id_up: Optional[str] = None
    token_id_down: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    question: Optional[str] = None
    is_active: bool = False
    collected_count: int = 0
    error_count: int = 0
    last_data: Optional[Dict] = None
    
    # Next market (for pre-loading)
    next_market_id: Optional[str] = None
    next_token_id_up: Optional[str] = None
    next_token_id_down: Optional[str] = None
    next_start_time: Optional[datetime] = None
    next_end_time: Optional[datetime] = None
    next_preloaded: bool = False


class MarketOrchestrator:
    """
    Orchestrates data collection across multiple assets with seamless market transitions.
    """
    
    def __init__(
        self,
        assets: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        preload_seconds: int = 60,
        log_level: int = 20,
    ):
        """
        Initialize orchestrator.
        
        Args:
            assets: List of asset symbols to collect (default: all supported)
            output_dir: Directory for output files
            preload_seconds: How many seconds before market end to preload next market
            log_level: Logging level
        """
        self.assets = [a.upper() for a in (assets or SUPPORTED_ASSETS)]
        base_output_dir = Path(output_dir or STORAGE.raw_dir)
        # Organize by asset: data_v2/raw/polymarket/{asset}/
        self.output_dir = base_output_dir / "polymarket"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preload_seconds = preload_seconds
        
        self.logger = setup_logger("orchestrator", level=log_level)
        self.logger.info(f"Initializing orchestrator for assets: {self.assets}")
        
        # Polymarket API client
        self.pm_collector = PolymarketCollector(log_level=log_level)
        
        # Market states (one per asset)
        self.market_states: Dict[str, MarketState] = {
            asset: MarketState(asset=asset) for asset in self.assets
        }
        
        # Collection state
        self.running = False
        
        # Output files and writers
        self.output_files: Dict[str, Path] = {}
        self.csv_writers: Dict[str, csv.DictWriter] = {}
        self.file_handles: Dict[str, any] = {}
        
        # Deduplication: track written seconds and pending data per asset
        self._last_written_second: Dict[str, Optional[datetime]] = {}
        self._written_seconds: Dict[str, set] = {}  # asset -> set of seconds already written (ISO string keys)
        self._pending_data: Dict[str, Optional[Dict]] = {}  # asset -> most recent data for current second
        
        self._init_output_files()
    
    def _init_output_files(self):
        """Initialize CSV output files for each asset in asset subdirectories.
        
        HEADER VALIDATION:
        If an existing file has different headers, we backup the old file and start fresh.
        This prevents column misalignment bugs when the format changes (e.g., adding order book depth).
        """
        # TIMESTAMP SEMANTICS (CRITICAL FOR LATENCY ANALYSIS):
        # - source_timestamp: When the orderbook data was generated (API time)
        # - received_timestamp: When we actually received/observed the data
        # For PM data, these are typically very close (< 1s) unless API is slow.
        #
        # ORDER BOOK DEPTH: Levels 1-6 for both UP and DOWN tokens
        # Level 1 = best bid/ask, Level 2-6 = subsequent levels
        fieldnames = [
            "source_timestamp",      # Orderbook time (was 'timestamp')
            "received_timestamp",    # When we saw it (was 'collected_at')
            "timestamp_ms",          # Original ms precision
            "asset", "market_id",
            # UP token - Level 1
            "up_mid", "up_bid", "up_bid_size", "up_ask", "up_ask_size",
            # UP token - Levels 2-6
            "up_bid_2", "up_bid_2_size", "up_ask_2", "up_ask_2_size",
            "up_bid_3", "up_bid_3_size", "up_ask_3", "up_ask_3_size",
            "up_bid_4", "up_bid_4_size", "up_ask_4", "up_ask_4_size",
            "up_bid_5", "up_bid_5_size", "up_ask_5", "up_ask_5_size",
            "up_bid_6", "up_bid_6_size", "up_ask_6", "up_ask_6_size",
            # DOWN token - Level 1
            "down_mid", "down_bid", "down_bid_size", "down_ask", "down_ask_size",
            # DOWN token - Levels 2-6
            "down_bid_2", "down_bid_2_size", "down_ask_2", "down_ask_2_size",
            "down_bid_3", "down_bid_3_size", "down_ask_3", "down_ask_3_size",
            "down_bid_4", "down_bid_4_size", "down_ask_4", "down_ask_4_size",
            "down_bid_5", "down_bid_5_size", "down_ask_5", "down_ask_5_size",
            "down_bid_6", "down_bid_6_size", "down_ask_6", "down_ask_6_size",
            "is_observed"
        ]
        
        for asset in self.assets:
            # Create asset subdirectory
            asset_dir = self.output_dir / asset.upper()
            asset_dir.mkdir(parents=True, exist_ok=True)
            
            # File: polymarket_BTC_continuous.csv (more readable)
            filepath = asset_dir / f"polymarket_{asset.upper()}_continuous.csv"
            self.output_files[asset] = filepath
            
            # Check if file exists and validate headers match expected format
            write_header = True
            if filepath.exists():
                try:
                    with open(filepath, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        existing_headers = next(reader, None)
                        
                    if existing_headers:
                        if existing_headers == fieldnames:
                            # Headers match - can safely append
                            write_header = False
                            self.logger.info(f"Existing file for {asset} has matching headers - appending")
                        else:
                            # Headers DON'T match - backup old file and start fresh
                            from datetime import datetime as dt
                            backup_name = f"polymarket_{asset.upper()}_continuous_backup_{dt.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            backup_path = asset_dir / backup_name
                            import shutil
                            shutil.move(str(filepath), str(backup_path))
                            self.logger.warning(
                                f"HEADER MISMATCH for {asset}! "
                                f"Expected: {fieldnames}, "
                                f"Found: {existing_headers}. "
                                f"Old file backed up to: {backup_name}"
                            )
                            write_header = True
                except Exception as e:
                    self.logger.warning(f"Error reading existing file for {asset}: {e}. Starting fresh.")
                    write_header = True
            
            fh = open(filepath, 'a', newline='', encoding='utf-8')
            self.file_handles[asset] = fh
            
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            self.csv_writers[asset] = writer
            
            self.logger.info(f"Output file for {asset}: {filepath}")
    
    def _write_data(self, asset: str, data: Dict, market_id: str):
        """Write data row to CSV file, deduplicating by second (keep most recent)."""
        writer = self.csv_writers.get(asset)
        if not writer:
            return
        
        # Use the API timestamp (from /book endpoint) as the data timestamp
        api_timestamp = data.get('timestamp')
        if isinstance(api_timestamp, datetime):
            ts = api_timestamp
        elif api_timestamp:
            ts = pd.to_datetime(api_timestamp)
        else:
            # Fallback to collected_at if no API timestamp
            collected_at = data.get('collected_at')
            if isinstance(collected_at, datetime):
                ts = collected_at
            else:
                ts = pd.to_datetime(collected_at) if collected_at else datetime.now(timezone.utc)
        
        # Ensure timezone-aware for comparison
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts_second = ts.replace(microsecond=0)  # Floor to second
        ts_second_str = ts_second.isoformat()
        
        # Initialize written_seconds set for this asset if needed
        if asset not in self._written_seconds:
            self._written_seconds[asset] = set()
        
        # If we've already written this second, skip (deduplication)
        if ts_second_str in self._written_seconds[asset]:
            # Update pending in case we want to overwrite, but don't write yet
            self._pending_data[asset] = {**data, 'market_id': market_id}
            return
        
        last_second_str = self._last_written_second[asset].isoformat() if self._last_written_second.get(asset) else None
        
        if last_second_str is None:
            # First data point - just store as pending
            self._pending_data[asset] = {**data, 'market_id': market_id}
            self._last_written_second[asset] = ts_second
            return
        
        if ts_second_str == last_second_str:
            # Same second - update pending data (keep most recent)
            self._pending_data[asset] = {**data, 'market_id': market_id}
            return  # Don't write yet, wait for next second
        
        # New second detected - write the pending data from previous second
        if asset in self._pending_data and self._pending_data[asset] is not None:
            pending = self._pending_data[asset]
            pending_ts = pending.get('timestamp')
            if isinstance(pending_ts, datetime):
                pass
            elif pending_ts:
                pending_ts = pd.to_datetime(pending_ts)
            else:
                pending_ts = pd.to_datetime(pending.get('collected_at'))
            
            if pending_ts.tzinfo is None:
                pending_ts = pending_ts.replace(tzinfo=timezone.utc)
            pending_second_str = pending_ts.replace(microsecond=0).isoformat()
            
            # Only write if we haven't written this second before
            if pending_second_str not in self._written_seconds[asset]:
                timestamp_str = pending_ts.isoformat() if isinstance(pending_ts, datetime) else str(pending_ts)
                received_ts = pending.get('received_timestamp', pending.get('collected_at', datetime.now(timezone.utc)))
                received_str = received_ts.isoformat() if isinstance(received_ts, datetime) else str(received_ts)
                row = {
                    'source_timestamp': timestamp_str,
                    'received_timestamp': received_str,
                    'timestamp_ms': pending.get('timestamp_ms', ''),
                    'asset': asset,
                    'market_id': pending.get('market_id', market_id),
                    # UP token - Level 1 (use new field names, fallback to old)
                    'up_mid': pending.get('up_mid'),
                    'up_bid': pending.get('up_bid', pending.get('up_best_bid')),
                    'up_bid_size': pending.get('up_bid_size', pending.get('up_best_bid_size')),
                    'up_ask': pending.get('up_ask', pending.get('up_best_ask')),
                    'up_ask_size': pending.get('up_ask_size', pending.get('up_best_ask_size')),
                    # UP token - Levels 2-6
                    'up_bid_2': pending.get('up_bid_2'),
                    'up_bid_2_size': pending.get('up_bid_2_size'),
                    'up_ask_2': pending.get('up_ask_2'),
                    'up_ask_2_size': pending.get('up_ask_2_size'),
                    'up_bid_3': pending.get('up_bid_3'),
                    'up_bid_3_size': pending.get('up_bid_3_size'),
                    'up_ask_3': pending.get('up_ask_3'),
                    'up_ask_3_size': pending.get('up_ask_3_size'),
                    'up_bid_4': pending.get('up_bid_4'),
                    'up_bid_4_size': pending.get('up_bid_4_size'),
                    'up_ask_4': pending.get('up_ask_4'),
                    'up_ask_4_size': pending.get('up_ask_4_size'),
                    'up_bid_5': pending.get('up_bid_5'),
                    'up_bid_5_size': pending.get('up_bid_5_size'),
                    'up_ask_5': pending.get('up_ask_5'),
                    'up_ask_5_size': pending.get('up_ask_5_size'),
                    'up_bid_6': pending.get('up_bid_6'),
                    'up_bid_6_size': pending.get('up_bid_6_size'),
                    'up_ask_6': pending.get('up_ask_6'),
                    'up_ask_6_size': pending.get('up_ask_6_size'),
                    # DOWN token - Level 1
                    'down_mid': pending.get('down_mid'),
                    'down_bid': pending.get('down_bid', pending.get('down_best_bid')),
                    'down_bid_size': pending.get('down_bid_size', pending.get('down_best_bid_size')),
                    'down_ask': pending.get('down_ask', pending.get('down_best_ask')),
                    'down_ask_size': pending.get('down_ask_size', pending.get('down_best_ask_size')),
                    # DOWN token - Levels 2-6
                    'down_bid_2': pending.get('down_bid_2'),
                    'down_bid_2_size': pending.get('down_bid_2_size'),
                    'down_ask_2': pending.get('down_ask_2'),
                    'down_ask_2_size': pending.get('down_ask_2_size'),
                    'down_bid_3': pending.get('down_bid_3'),
                    'down_bid_3_size': pending.get('down_bid_3_size'),
                    'down_ask_3': pending.get('down_ask_3'),
                    'down_ask_3_size': pending.get('down_ask_3_size'),
                    'down_bid_4': pending.get('down_bid_4'),
                    'down_bid_4_size': pending.get('down_bid_4_size'),
                    'down_ask_4': pending.get('down_ask_4'),
                    'down_ask_4_size': pending.get('down_ask_4_size'),
                    'down_bid_5': pending.get('down_bid_5'),
                    'down_bid_5_size': pending.get('down_bid_5_size'),
                    'down_ask_5': pending.get('down_ask_5'),
                    'down_ask_5_size': pending.get('down_ask_5_size'),
                    'down_bid_6': pending.get('down_bid_6'),
                    'down_bid_6_size': pending.get('down_bid_6_size'),
                    'down_ask_6': pending.get('down_ask_6'),
                    'down_ask_6_size': pending.get('down_ask_6_size'),
                    'is_observed': 1,
                }
                writer.writerow(row)
                self.file_handles[asset].flush()
                self._written_seconds[asset].add(pending_second_str)
        
        # Store current data as pending for this new second
        self._pending_data[asset] = {**data, 'market_id': market_id}
        self._last_written_second[asset] = ts_second
    
    def _write_missing_data(self, asset: str, market_id: str, last_data: Optional[Dict] = None):
        """Write a row indicating missing data (forward-fill placeholder)."""
        writer = self.csv_writers.get(asset)
        if not writer:
            return
        
        now = datetime.now(timezone.utc)
        row = {
            'source_timestamp': now.isoformat(),  # For ffill, source = received
            'received_timestamp': now.isoformat(),
            'timestamp_ms': '',  # No API timestamp for missing data
            'asset': asset,
            'market_id': market_id,
            # UP token - Level 1
            'up_mid': last_data.get('up_mid') if last_data else None,
            'up_bid': last_data.get('up_bid', last_data.get('up_best_bid')) if last_data else None,
            'up_bid_size': last_data.get('up_bid_size', last_data.get('up_best_bid_size')) if last_data else None,
            'up_ask': last_data.get('up_ask', last_data.get('up_best_ask')) if last_data else None,
            'up_ask_size': last_data.get('up_ask_size', last_data.get('up_best_ask_size')) if last_data else None,
            # UP token - Levels 2-6
            'up_bid_2': last_data.get('up_bid_2') if last_data else None,
            'up_bid_2_size': last_data.get('up_bid_2_size') if last_data else None,
            'up_ask_2': last_data.get('up_ask_2') if last_data else None,
            'up_ask_2_size': last_data.get('up_ask_2_size') if last_data else None,
            'up_bid_3': last_data.get('up_bid_3') if last_data else None,
            'up_bid_3_size': last_data.get('up_bid_3_size') if last_data else None,
            'up_ask_3': last_data.get('up_ask_3') if last_data else None,
            'up_ask_3_size': last_data.get('up_ask_3_size') if last_data else None,
            'up_bid_4': last_data.get('up_bid_4') if last_data else None,
            'up_bid_4_size': last_data.get('up_bid_4_size') if last_data else None,
            'up_ask_4': last_data.get('up_ask_4') if last_data else None,
            'up_ask_4_size': last_data.get('up_ask_4_size') if last_data else None,
            'up_bid_5': last_data.get('up_bid_5') if last_data else None,
            'up_bid_5_size': last_data.get('up_bid_5_size') if last_data else None,
            'up_ask_5': last_data.get('up_ask_5') if last_data else None,
            'up_ask_5_size': last_data.get('up_ask_5_size') if last_data else None,
            'up_bid_6': last_data.get('up_bid_6') if last_data else None,
            'up_bid_6_size': last_data.get('up_bid_6_size') if last_data else None,
            'up_ask_6': last_data.get('up_ask_6') if last_data else None,
            'up_ask_6_size': last_data.get('up_ask_6_size') if last_data else None,
            # DOWN token - Level 1
            'down_mid': last_data.get('down_mid') if last_data else None,
            'down_bid': last_data.get('down_bid', last_data.get('down_best_bid')) if last_data else None,
            'down_bid_size': last_data.get('down_bid_size', last_data.get('down_best_bid_size')) if last_data else None,
            'down_ask': last_data.get('down_ask', last_data.get('down_best_ask')) if last_data else None,
            'down_ask_size': last_data.get('down_ask_size', last_data.get('down_best_ask_size')) if last_data else None,
            # DOWN token - Levels 2-6
            'down_bid_2': last_data.get('down_bid_2') if last_data else None,
            'down_bid_2_size': last_data.get('down_bid_2_size') if last_data else None,
            'down_ask_2': last_data.get('down_ask_2') if last_data else None,
            'down_ask_2_size': last_data.get('down_ask_2_size') if last_data else None,
            'down_bid_3': last_data.get('down_bid_3') if last_data else None,
            'down_bid_3_size': last_data.get('down_bid_3_size') if last_data else None,
            'down_ask_3': last_data.get('down_ask_3') if last_data else None,
            'down_ask_3_size': last_data.get('down_ask_3_size') if last_data else None,
            'down_bid_4': last_data.get('down_bid_4') if last_data else None,
            'down_bid_4_size': last_data.get('down_bid_4_size') if last_data else None,
            'down_ask_4': last_data.get('down_ask_4') if last_data else None,
            'down_ask_4_size': last_data.get('down_ask_4_size') if last_data else None,
            'down_bid_5': last_data.get('down_bid_5') if last_data else None,
            'down_bid_5_size': last_data.get('down_bid_5_size') if last_data else None,
            'down_ask_5': last_data.get('down_ask_5') if last_data else None,
            'down_ask_5_size': last_data.get('down_ask_5_size') if last_data else None,
            'down_bid_6': last_data.get('down_bid_6') if last_data else None,
            'down_bid_6_size': last_data.get('down_bid_6_size') if last_data else None,
            'down_ask_6': last_data.get('down_ask_6') if last_data else None,
            'down_ask_6_size': last_data.get('down_ask_6_size') if last_data else None,
            'is_observed': 0,
        }
        writer.writerow(row)
        self.file_handles[asset].flush()
    
    def _discover_market(self, asset: str) -> bool:
        """
        Discover and set the current active market for an asset.
        
        Returns:
            True if market found and set, False otherwise
        """
        market = self.pm_collector.find_active_market(asset)
        
        if not market:
            self.logger.warning(f"No active market found for {asset}")
            self.market_states[asset].is_active = False
            return False
        
        state = self.market_states[asset]
        state.market_id = market.get("market_slug")
        state.token_id_up = market.get("token_id_up")
        state.token_id_down = market.get("token_id_down")
        state.question = market.get("question")
        
        start_time, end_time = self.pm_collector.get_market_times(market)
        state.start_time = start_time
        state.end_time = end_time
        state.is_active = True
        state.collected_count = 0
        state.error_count = 0
        
        self.logger.info(
            f"Discovered {asset} market: {state.market_id} "
            f"(ends at {end_time.isoformat() if end_time else 'unknown'})"
        )
        return True
    
    def _preload_next_market(self, asset: str) -> bool:
        """
        Preload the next market for seamless transition.
        
        Returns:
            True if next market found and preloaded, False otherwise
        """
        market = self.pm_collector.find_next_market(asset)
        
        if not market:
            self.logger.warning(f"Could not preload next market for {asset}")
            return False
        
        state = self.market_states[asset]
        state.next_market_id = market.get("market_slug")
        state.next_token_id_up = market.get("token_id_up")
        state.next_token_id_down = market.get("token_id_down")
        
        start_time, end_time = self.pm_collector.get_market_times(market)
        state.next_start_time = start_time
        state.next_end_time = end_time
        state.next_preloaded = True
        
        self.logger.info(
            f"Preloaded next {asset} market: {state.next_market_id} "
            f"(starts at {start_time.isoformat() if start_time else 'unknown'})"
        )
        return True
    
    def _switch_to_next_market(self, asset: str):
        """Switch from current market to preloaded next market."""
        state = self.market_states[asset]
        
        if not state.next_preloaded:
            self.logger.warning(f"No preloaded market for {asset}, discovering fresh")
            self._discover_market(asset)
            return
        
        # Move next market to current
        state.market_id = state.next_market_id
        state.token_id_up = state.next_token_id_up
        state.token_id_down = state.next_token_id_down
        state.start_time = state.next_start_time
        state.end_time = state.next_end_time
        state.collected_count = 0
        state.error_count = 0
        state.is_active = True
        
        # Clear next market
        state.next_market_id = None
        state.next_token_id_up = None
        state.next_token_id_down = None
        state.next_start_time = None
        state.next_end_time = None
        state.next_preloaded = False
        
        self.logger.info(f"Switched {asset} to new market: {state.market_id}")
    
    def _collect_asset(self, asset: str) -> Optional[Dict]:
        """
        Collect data for a single asset.
        
        Returns:
            Collected data dictionary or None if failed
        """
        state = self.market_states[asset]
        
        if not state.is_active or not state.token_id_up or not state.token_id_down:
            self.logger.debug(f"Asset {asset} not ready for collection")
            return None
        
        try:
            data = self.pm_collector.get_market_data(
                state.token_id_up,
                state.token_id_down
            )
            
            if data:
                state.collected_count += 1
                state.last_data = data
                self._write_data(asset, data, state.market_id)
                return data
            else:
                state.error_count += 1
                self._write_missing_data(asset, state.market_id, state.last_data)
                return None
                
        except Exception as e:
            self.logger.error(f"Error collecting {asset}: {e}")
            state.error_count += 1
            self._write_missing_data(asset, state.market_id, state.last_data)
            return None
    
    def _check_market_transitions(self):
        """Check if any markets need transitioning or preloading."""
        now = datetime.now(timezone.utc)
        
        for asset in self.assets:
            state = self.market_states[asset]
            
            if not state.is_active:
                # Try to discover market
                self._discover_market(asset)
                continue
            
            if not state.end_time:
                continue
            
            time_to_end = (state.end_time - now).total_seconds()
            
            # Check if it's time to preload next market
            if not state.next_preloaded and time_to_end <= self.preload_seconds:
                self.logger.info(f"{asset} market ending in {time_to_end:.0f}s, preloading next")
                self._preload_next_market(asset)
            
            # Check if current market has ended
            if time_to_end <= 0:
                self.logger.info(f"{asset} market ended, switching to next")
                self._switch_to_next_market(asset)
    
    async def start(
        self,
        interval: float = 1.0,
        duration: Optional[float] = None,
        callback: Optional[Callable[[Dict[str, Optional[Dict]]], None]] = None,
        log_interval: int = 10
    ):
        """
        Start collection for all assets.
        
        Args:
            interval: Collection interval in seconds (default: 1.0)
            duration: How long to run in seconds (None = indefinite)
            callback: Optional callback after each collection round
            log_interval: How often to log stats in seconds
        """
        self.logger.info(f"Starting orchestrator at {interval}s intervals")
        if duration:
            self.logger.info(f"Will run for {duration} seconds")
        else:
            self.logger.info("Running indefinitely (Ctrl+C to stop)")
        
        # Initial market discovery for all assets
        for asset in self.assets:
            self._discover_market(asset)
        
        self.running = True
        start_time = time.time()
        last_log_time = start_time
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Check duration limit
                if duration and (loop_start - start_time) >= duration:
                    self.logger.info("Duration limit reached, stopping")
                    break
                
                # Check for market transitions
                self._check_market_transitions()
                
                # Collect from all assets
                results = {}
                for asset in self.assets:
                    results[asset] = self._collect_asset(asset)
                
                # Call callback if provided
                if callback:
                    callback(results)
                
                # Periodic logging
                if loop_start - last_log_time >= log_interval:
                    self._log_stats()
                    last_log_time = loop_start
                
                # Sleep until next interval
                elapsed = time.time() - loop_start
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
        except asyncio.CancelledError:
            self.logger.info("Collection cancelled")
        except Exception as e:
            self.logger.error(f"Collection error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            self._log_stats()
    
    def _log_stats(self):
        """Log collection statistics."""
        for asset in self.assets:
            state = self.market_states[asset]
            self.logger.info(
                f"{asset}: market={state.market_id or 'none'}, "
                f"collected={state.collected_count}, errors={state.error_count}, "
                f"active={state.is_active}"
            )
    
    async def stop(self):
        """Stop collection and cleanup."""
        self.logger.info("Stopping orchestrator...")
        self.running = False
        
        # Flush any pending data before closing
        for asset in self.assets:
            if asset in self._pending_data and self._pending_data[asset] is not None:
                pending = self._pending_data[asset]
                writer = self.csv_writers.get(asset)
                if writer:
                    pending_ts = pending.get('timestamp')
                    if isinstance(pending_ts, datetime):
                        pass
                    elif pending_ts:
                        pending_ts = pd.to_datetime(pending_ts)
                    else:
                        pending_ts = pd.to_datetime(pending.get('collected_at', datetime.now(timezone.utc)))
                    
                    if pending_ts.tzinfo is None:
                        pending_ts = pending_ts.replace(tzinfo=timezone.utc)
                    pending_second_str = pending_ts.replace(microsecond=0).isoformat()
                    
                    # Write pending data if not already written
                    if pending_second_str not in self._written_seconds.get(asset, set()):
                        timestamp_str = pending_ts.isoformat() if isinstance(pending_ts, datetime) else str(pending_ts)
                        received_ts = pending.get('received_timestamp', pending.get('collected_at', datetime.now(timezone.utc)))
                        received_str = received_ts.isoformat() if isinstance(received_ts, datetime) else str(received_ts)
                        row = {
                            'source_timestamp': timestamp_str,
                            'received_timestamp': received_str,
                            'timestamp_ms': pending.get('timestamp_ms', ''),
                            'asset': asset,
                            'market_id': pending.get('market_id', ''),
                            'up_mid': pending.get('up_mid'),
                            'up_bid': pending.get('up_bid', pending.get('up_best_bid')),
                            'up_bid_size': pending.get('up_bid_size', pending.get('up_best_bid_size')),
                            'up_ask': pending.get('up_ask', pending.get('up_best_ask')),
                            'up_ask_size': pending.get('up_ask_size', pending.get('up_best_ask_size')),
                            'up_bid_2': pending.get('up_bid_2'),
                            'up_bid_2_size': pending.get('up_bid_2_size'),
                            'up_ask_2': pending.get('up_ask_2'),
                            'up_ask_2_size': pending.get('up_ask_2_size'),
                            'up_bid_3': pending.get('up_bid_3'),
                            'up_bid_3_size': pending.get('up_bid_3_size'),
                            'up_ask_3': pending.get('up_ask_3'),
                            'up_ask_3_size': pending.get('up_ask_3_size'),
                            'up_bid_4': pending.get('up_bid_4'),
                            'up_bid_4_size': pending.get('up_bid_4_size'),
                            'up_ask_4': pending.get('up_ask_4'),
                            'up_ask_4_size': pending.get('up_ask_4_size'),
                            'up_bid_5': pending.get('up_bid_5'),
                            'up_bid_5_size': pending.get('up_bid_5_size'),
                            'up_ask_5': pending.get('up_ask_5'),
                            'up_ask_5_size': pending.get('up_ask_5_size'),
                            'up_bid_6': pending.get('up_bid_6'),
                            'up_bid_6_size': pending.get('up_bid_6_size'),
                            'up_ask_6': pending.get('up_ask_6'),
                            'up_ask_6_size': pending.get('up_ask_6_size'),
                            'down_mid': pending.get('down_mid'),
                            'down_bid': pending.get('down_bid', pending.get('down_best_bid')),
                            'down_bid_size': pending.get('down_bid_size', pending.get('down_best_bid_size')),
                            'down_ask': pending.get('down_ask', pending.get('down_best_ask')),
                            'down_ask_size': pending.get('down_ask_size', pending.get('down_best_ask_size')),
                            'down_bid_2': pending.get('down_bid_2'),
                            'down_bid_2_size': pending.get('down_bid_2_size'),
                            'down_ask_2': pending.get('down_ask_2'),
                            'down_ask_2_size': pending.get('down_ask_2_size'),
                            'down_bid_3': pending.get('down_bid_3'),
                            'down_bid_3_size': pending.get('down_bid_3_size'),
                            'down_ask_3': pending.get('down_ask_3'),
                            'down_ask_3_size': pending.get('down_ask_3_size'),
                            'down_bid_4': pending.get('down_bid_4'),
                            'down_bid_4_size': pending.get('down_bid_4_size'),
                            'down_ask_4': pending.get('down_ask_4'),
                            'down_ask_4_size': pending.get('down_ask_4_size'),
                            'down_bid_5': pending.get('down_bid_5'),
                            'down_bid_5_size': pending.get('down_bid_5_size'),
                            'down_ask_5': pending.get('down_ask_5'),
                            'down_ask_5_size': pending.get('down_ask_5_size'),
                            'down_bid_6': pending.get('down_bid_6'),
                            'down_bid_6_size': pending.get('down_bid_6_size'),
                            'down_ask_6': pending.get('down_ask_6'),
                            'down_ask_6_size': pending.get('down_ask_6_size'),
                            'down_ask_2': pending.get('down_ask_2'),
                            'down_ask_2_size': pending.get('down_ask_2_size'),
                            'is_observed': 1,
                        }
                        writer.writerow(row)
                        self.file_handles[asset].flush()
                        if asset not in self._written_seconds:
                            self._written_seconds[asset] = set()
                        self._written_seconds[asset].add(pending_second_str)
        
        # Close file handles
        for fh in self.file_handles.values():
            try:
                fh.flush()  # Ensure all data is written
                fh.close()
            except:
                pass
        
        self.logger.info("Orchestrator stopped")
    
    def get_state(self, asset: str) -> Optional[MarketState]:
        """Get current state for an asset."""
        return self.market_states.get(asset.upper())
    
    def get_all_states(self) -> Dict[str, MarketState]:
        """Get states for all assets."""
        return self.market_states.copy()


class MarketOrchestratorSync:
    """Synchronous wrapper for MarketOrchestrator."""
    
    def __init__(
        self,
        assets: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        preload_seconds: int = 60,
        log_level: int = 20,
    ):
        self.orchestrator = MarketOrchestrator(
            assets=assets,
            output_dir=output_dir,
            preload_seconds=preload_seconds,
            log_level=log_level
        )
        self._loop = None
    
    def _get_loop(self):
        """Get or create event loop."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop
    
    def start(
        self,
        interval: float = 1.0,
        duration: Optional[float] = None,
        callback: Optional[Callable] = None,
        log_interval: int = 10
    ):
        """Start collection (blocking)."""
        loop = self._get_loop()
        loop.run_until_complete(
            self.orchestrator.start(interval, duration, callback, log_interval)
        )
    
    def stop(self):
        """Stop collection."""
        if self._loop:
            loop = self._get_loop()
            loop.run_until_complete(self.orchestrator.stop())
    
    @property
    def running(self):
        return self.orchestrator.running
    
    def get_state(self, asset: str) -> Optional[MarketState]:
        return self.orchestrator.get_state(asset)
    
    def get_all_states(self) -> Dict[str, MarketState]:
        return self.orchestrator.get_all_states()


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

async def main():
    """Run orchestrator for all assets."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Polymarket market orchestrator")
    parser.add_argument(
        "--assets",
        type=str,
        default="BTC,ETH,SOL,XRP",
        help="Comma-separated list of assets"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Collection duration in seconds (default: indefinite)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Collection interval in seconds"
    )
    parser.add_argument(
        "--preload",
        type=int,
        default=60,
        help="Seconds before market end to preload next market"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    assets = [a.strip().upper() for a in args.assets.split(",")]
    
    orchestrator = MarketOrchestrator(
        assets=assets,
        output_dir=args.output_dir,
        preload_seconds=args.preload,
    )
    
    try:
        await orchestrator.start(
            interval=args.interval,
            duration=args.duration,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())

