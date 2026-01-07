"""
Fast Market Orchestrator with wall-clock anchored timing.

Key optimizations:
1. Wall-clock anchored timing - targets exact second boundaries
2. Uses FastPolymarketCollector for parallel API calls
3. Minimal hot-path overhead - no debug logging during collection
4. Pre-flight market discovery to avoid delays during collection
"""

import asyncio
import csv
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logger
from src.fast_pm_collector import FastPolymarketCollector
from config.settings import STORAGE, SUPPORTED_ASSETS, get_polymarket_slug_prefix


@dataclass
class FastMarketState:
    """Lightweight market state."""
    asset: str
    market_id: Optional[str] = None
    token_id_up: Optional[str] = None
    token_id_down: Optional[str] = None
    end_time: Optional[datetime] = None
    is_active: bool = False
    collected_count: int = 0
    error_count: int = 0
    last_data: Optional[Dict] = None
    
    # Next market (for seamless transitions)
    next_market_id: Optional[str] = None
    next_token_id_up: Optional[str] = None
    next_token_id_down: Optional[str] = None
    next_end_time: Optional[datetime] = None
    next_preloaded: bool = False


class FastOrchestrator:
    """
    High-performance market data orchestrator.
    
    Key features:
    - Wall-clock anchored timing (no drift)
    - Parallel API calls via aiohttp
    - Seamless market transitions
    - Minimal overhead during collection
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
            assets: List of asset symbols (default: all supported)
            output_dir: Output directory (default: data_v2/raw)
            preload_seconds: Seconds before market end to preload next
            log_level: Logging level
        """
        self.assets = [a.upper() for a in (assets or SUPPORTED_ASSETS)]
        base_output_dir = Path(output_dir or STORAGE.raw_dir)
        self.output_dir = base_output_dir / "polymarket"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preload_seconds = preload_seconds
        
        self.logger = setup_logger("fast_orchestrator", level=log_level)
        self.logger.info(f"FastOrchestrator initialized for: {self.assets}")
        
        # Fast PM collector (uses aiohttp)
        self.pm_collector = FastPolymarketCollector(timeout=5.0)
        
        # Market states
        self.market_states: Dict[str, FastMarketState] = {
            asset: FastMarketState(asset=asset) for asset in self.assets
        }
        
        # Collection state
        self.running = False
        
        # Output files
        self.output_files: Dict[str, Path] = {}
        self.csv_writers: Dict[str, csv.DictWriter] = {}
        self.file_handles: Dict[str, any] = {}
        
        # Deduplication
        self._written_seconds: Dict[str, set] = {asset: set() for asset in self.assets}
        self._pending_data: Dict[str, Optional[Dict]] = {asset: None for asset in self.assets}
        self._last_written_second: Dict[str, Optional[datetime]] = {asset: None for asset in self.assets}
    
    def _init_output_files(self):
        """Initialize CSV output files."""
        fieldnames = [
            "source_timestamp", "received_timestamp", "timestamp_ms",
            "asset", "market_id",
            # UP token - 6 levels
            "up_mid", "up_bid", "up_bid_size", "up_ask", "up_ask_size",
            "up_bid_2", "up_bid_2_size", "up_ask_2", "up_ask_2_size",
            "up_bid_3", "up_bid_3_size", "up_ask_3", "up_ask_3_size",
            "up_bid_4", "up_bid_4_size", "up_ask_4", "up_ask_4_size",
            "up_bid_5", "up_bid_5_size", "up_ask_5", "up_ask_5_size",
            "up_bid_6", "up_bid_6_size", "up_ask_6", "up_ask_6_size",
            # DOWN token - 6 levels
            "down_mid", "down_bid", "down_bid_size", "down_ask", "down_ask_size",
            "down_bid_2", "down_bid_2_size", "down_ask_2", "down_ask_2_size",
            "down_bid_3", "down_bid_3_size", "down_ask_3", "down_ask_3_size",
            "down_bid_4", "down_bid_4_size", "down_ask_4", "down_ask_4_size",
            "down_bid_5", "down_bid_5_size", "down_ask_5", "down_ask_5_size",
            "down_bid_6", "down_bid_6_size", "down_ask_6", "down_ask_6_size",
            "is_observed"
        ]
        
        for asset in self.assets:
            asset_dir = self.output_dir / asset.upper()
            asset_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = asset_dir / f"polymarket_{asset.upper()}_continuous.csv"
            self.output_files[asset] = filepath
            
            # Check if existing file has matching headers
            write_header = True
            if filepath.exists():
                try:
                    with open(filepath, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        existing_headers = next(reader, None)
                    
                    if existing_headers and existing_headers == fieldnames:
                        write_header = False
                        self.logger.info(f"Appending to existing file for {asset}")
                    else:
                        # Backup and start fresh
                        backup_name = f"polymarket_{asset}_continuous_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        backup_path = asset_dir / backup_name
                        import shutil
                        shutil.move(str(filepath), str(backup_path))
                        self.logger.warning(f"Header mismatch for {asset}, backed up to {backup_name}")
                except Exception as e:
                    self.logger.warning(f"Error reading existing file for {asset}: {e}")
            
            fh = open(filepath, 'a', newline='', encoding='utf-8')
            self.file_handles[asset] = fh
            
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            self.csv_writers[asset] = writer
            
            self.logger.info(f"Output: {filepath}")
    
    def _discover_market(self, asset: str) -> bool:
        """Discover current active market for an asset."""
        import requests
        
        try:
            slug_prefix = get_polymarket_slug_prefix(asset)
            url = f"https://gamma-api.polymarket.com/events?slug={slug_prefix}"
            
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return False
            
            events = response.json()
            if not events:
                return False
            
            event = events[0]
            markets = event.get('markets', [])
            
            now = datetime.now(timezone.utc)
            
            for market in markets:
                end_str = market.get('endDate') or market.get('end_date_iso')
                if not end_str:
                    continue
                
                end_time = pd.to_datetime(end_str)
                if end_time.tzinfo is None:
                    end_time = end_time.replace(tzinfo=timezone.utc)
                
                # Find active market (not ended yet)
                if end_time > now:
                    state = self.market_states[asset]
                    state.market_id = market.get('conditionId') or market.get('id')
                    state.end_time = end_time
                    
                    # Get token IDs
                    tokens = market.get('tokens', [])
                    for token in tokens:
                        outcome = token.get('outcome', '').upper()
                        token_id = token.get('token_id')
                        if outcome == 'YES' or outcome == 'UP':
                            state.token_id_up = token_id
                        elif outcome == 'NO' or outcome == 'DOWN':
                            state.token_id_down = token_id
                    
                    if state.token_id_up and state.token_id_down:
                        state.is_active = True
                        self.logger.info(f"{asset}: Found market {state.market_id}, ends {end_time}")
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error discovering market for {asset}: {e}")
            return False
    
    def _preload_next_market(self, asset: str):
        """Preload next market for seamless transition."""
        import requests
        
        try:
            slug_prefix = get_polymarket_slug_prefix(asset)
            url = f"https://gamma-api.polymarket.com/events?slug={slug_prefix}"
            
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return
            
            events = response.json()
            if not events:
                return
            
            event = events[0]
            markets = event.get('markets', [])
            
            state = self.market_states[asset]
            current_end = state.end_time
            
            for market in markets:
                end_str = market.get('endDate') or market.get('end_date_iso')
                if not end_str:
                    continue
                
                end_time = pd.to_datetime(end_str)
                if end_time.tzinfo is None:
                    end_time = end_time.replace(tzinfo=timezone.utc)
                
                # Find market that ends after current one
                if current_end and end_time > current_end:
                    state.next_market_id = market.get('conditionId') or market.get('id')
                    state.next_end_time = end_time
                    
                    tokens = market.get('tokens', [])
                    for token in tokens:
                        outcome = token.get('outcome', '').upper()
                        token_id = token.get('token_id')
                        if outcome == 'YES' or outcome == 'UP':
                            state.next_token_id_up = token_id
                        elif outcome == 'NO' or outcome == 'DOWN':
                            state.next_token_id_down = token_id
                    
                    if state.next_token_id_up and state.next_token_id_down:
                        state.next_preloaded = True
                        self.logger.info(f"{asset}: Preloaded next market {state.next_market_id}")
                        return
                        
        except Exception as e:
            self.logger.debug(f"Error preloading next market for {asset}: {e}")
    
    def _switch_to_next_market(self, asset: str):
        """Switch to preloaded next market."""
        state = self.market_states[asset]
        
        if state.next_preloaded and state.next_token_id_up and state.next_token_id_down:
            state.market_id = state.next_market_id
            state.token_id_up = state.next_token_id_up
            state.token_id_down = state.next_token_id_down
            state.end_time = state.next_end_time
            state.is_active = True
            state.collected_count = 0
            state.error_count = 0
            
            # Clear next market
            state.next_market_id = None
            state.next_token_id_up = None
            state.next_token_id_down = None
            state.next_end_time = None
            state.next_preloaded = False
            
            self.logger.info(f"{asset}: Switched to market {state.market_id}")
        else:
            # Try to discover new market
            state.is_active = False
            self._discover_market(asset)
    
    async def _collect_asset(self, asset: str) -> Optional[Dict]:
        """Collect data for a single asset using fast parallel API."""
        state = self.market_states[asset]
        
        if not state.is_active or not state.token_id_up or not state.token_id_down:
            return None
        
        try:
            # PARALLEL fetch using aiohttp
            data = await self.pm_collector.get_market_data_parallel(
                state.token_id_up,
                state.token_id_down,
                n_levels=6
            )
            
            if data:
                state.collected_count += 1
                state.error_count = 0
                state.last_data = data
                self._write_data(asset, data, state.market_id)
                return data
            else:
                state.error_count += 1
                return None
                
        except Exception as e:
            state.error_count += 1
            return None
    
    def _write_data(self, asset: str, data: Dict, market_id: str):
        """Write data to CSV with deduplication."""
        writer = self.csv_writers.get(asset)
        if not writer:
            return
        
        # Get timestamp and floor to second
        ts = data.get('timestamp') or data.get('collected_at')
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts_second = ts.replace(microsecond=0)
        ts_second_str = ts_second.isoformat()
        
        # Check if we've already written this second
        if ts_second_str in self._written_seconds[asset]:
            return
        
        # Write previous pending data if we've moved to a new second
        if self._pending_data[asset] is not None:
            pending = self._pending_data[asset]
            pending_ts = pending.get('_ts_second_str')
            if pending_ts and pending_ts != ts_second_str and pending_ts not in self._written_seconds[asset]:
                self._write_row(asset, pending, pending.get('_market_id'))
                self._written_seconds[asset].add(pending_ts)
        
        # Store as pending
        data['_ts_second_str'] = ts_second_str
        data['_market_id'] = market_id
        self._pending_data[asset] = data
    
    def _write_row(self, asset: str, data: Dict, market_id: str):
        """Actually write a row to CSV."""
        writer = self.csv_writers.get(asset)
        if not writer:
            return
        
        ts = data.get('timestamp') or data.get('collected_at')
        rcv_ts = data.get('received_timestamp') or data.get('collected_at')
        
        row = {
            'source_timestamp': ts.isoformat() if isinstance(ts, datetime) else str(ts),
            'received_timestamp': rcv_ts.isoformat() if isinstance(rcv_ts, datetime) else str(rcv_ts),
            'timestamp_ms': data.get('timestamp_ms', ''),
            'asset': asset,
            'market_id': market_id,
            # UP - Level 1
            'up_mid': data.get('up_mid'),
            'up_bid': data.get('up_bid'),
            'up_bid_size': data.get('up_bid_size'),
            'up_ask': data.get('up_ask'),
            'up_ask_size': data.get('up_ask_size'),
            # UP - Levels 2-6
            'up_bid_2': data.get('up_bid_2'),
            'up_bid_2_size': data.get('up_bid_2_size'),
            'up_ask_2': data.get('up_ask_2'),
            'up_ask_2_size': data.get('up_ask_2_size'),
            'up_bid_3': data.get('up_bid_3'),
            'up_bid_3_size': data.get('up_bid_3_size'),
            'up_ask_3': data.get('up_ask_3'),
            'up_ask_3_size': data.get('up_ask_3_size'),
            'up_bid_4': data.get('up_bid_4'),
            'up_bid_4_size': data.get('up_bid_4_size'),
            'up_ask_4': data.get('up_ask_4'),
            'up_ask_4_size': data.get('up_ask_4_size'),
            'up_bid_5': data.get('up_bid_5'),
            'up_bid_5_size': data.get('up_bid_5_size'),
            'up_ask_5': data.get('up_ask_5'),
            'up_ask_5_size': data.get('up_ask_5_size'),
            'up_bid_6': data.get('up_bid_6'),
            'up_bid_6_size': data.get('up_bid_6_size'),
            'up_ask_6': data.get('up_ask_6'),
            'up_ask_6_size': data.get('up_ask_6_size'),
            # DOWN - Level 1
            'down_mid': data.get('down_mid'),
            'down_bid': data.get('down_bid'),
            'down_bid_size': data.get('down_bid_size'),
            'down_ask': data.get('down_ask'),
            'down_ask_size': data.get('down_ask_size'),
            # DOWN - Levels 2-6
            'down_bid_2': data.get('down_bid_2'),
            'down_bid_2_size': data.get('down_bid_2_size'),
            'down_ask_2': data.get('down_ask_2'),
            'down_ask_2_size': data.get('down_ask_2_size'),
            'down_bid_3': data.get('down_bid_3'),
            'down_bid_3_size': data.get('down_bid_3_size'),
            'down_ask_3': data.get('down_ask_3'),
            'down_ask_3_size': data.get('down_ask_3_size'),
            'down_bid_4': data.get('down_bid_4'),
            'down_bid_4_size': data.get('down_bid_4_size'),
            'down_ask_4': data.get('down_ask_4'),
            'down_ask_4_size': data.get('down_ask_4_size'),
            'down_bid_5': data.get('down_bid_5'),
            'down_bid_5_size': data.get('down_bid_5_size'),
            'down_ask_5': data.get('down_ask_5'),
            'down_ask_5_size': data.get('down_ask_5_size'),
            'down_bid_6': data.get('down_bid_6'),
            'down_bid_6_size': data.get('down_bid_6_size'),
            'down_ask_6': data.get('down_ask_6'),
            'down_ask_6_size': data.get('down_ask_6_size'),
            'is_observed': 1,
        }
        
        writer.writerow(row)
        self.file_handles[asset].flush()
    
    def _check_market_transitions(self):
        """Check for market transitions and preloading."""
        now = datetime.now(timezone.utc)
        
        for asset in self.assets:
            state = self.market_states[asset]
            
            if not state.is_active:
                self._discover_market(asset)
                continue
            
            if not state.end_time:
                continue
            
            time_to_end = (state.end_time - now).total_seconds()
            
            # Preload next market
            if not state.next_preloaded and time_to_end <= self.preload_seconds:
                self._preload_next_market(asset)
            
            # Switch to next market
            if time_to_end <= 0:
                self._switch_to_next_market(asset)
    
    async def start(
        self,
        interval: float = 1.0,
        duration: Optional[float] = None,
        callback: Optional[Callable[[Dict[str, Optional[Dict]]], None]] = None,
        log_interval: int = 60
    ):
        """
        Start collection with WALL-CLOCK ANCHORED timing.
        
        This ensures we always target exact second boundaries,
        preventing drift over time.
        """
        self.logger.info(f"Starting FastOrchestrator at {interval}s intervals")
        if duration:
            self.logger.info(f"Duration: {duration}s")
        
        # Initialize files
        self._init_output_files()
        
        # Initial market discovery
        for asset in self.assets:
            self._discover_market(asset)
        
        self.running = True
        start_time = time.time()
        last_log_time = start_time
        
        # WALL-CLOCK ANCHORING: Start at the next whole second
        now = time.time()
        next_target = (int(now) + 1)  # Next whole second
        
        try:
            while self.running:
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    self.logger.info("Duration reached")
                    break
                
                # WALL-CLOCK ANCHORED SLEEP
                # Sleep until the next target second boundary
                sleep_time = next_target - time.time()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
                loop_start = time.time()
                
                # Check market transitions (infrequent, minimal overhead)
                self._check_market_transitions()
                
                # Collect from all assets in PARALLEL
                tasks = [self._collect_asset(asset) for asset in self.assets]
                results_list = await asyncio.gather(*tasks, return_exceptions=True)
                
                results = {}
                for i, asset in enumerate(self.assets):
                    r = results_list[i]
                    if isinstance(r, Exception):
                        results[asset] = None
                    else:
                        results[asset] = r
                
                # Callback
                if callback:
                    callback(results)
                
                # Periodic logging
                if loop_start - last_log_time >= log_interval:
                    self._log_stats()
                    last_log_time = loop_start
                
                # Set next target (WALL-CLOCK ANCHORED)
                next_target += interval
                
                # If we've fallen behind, catch up to current time
                now = time.time()
                if next_target < now:
                    skipped = int((now - next_target) / interval)
                    if skipped > 0:
                        self.logger.warning(f"Skipped {skipped} iterations (catching up)")
                    next_target = now + interval
                
        except asyncio.CancelledError:
            self.logger.info("Collection cancelled")
        except Exception as e:
            self.logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            await self.stop()
    
    def _log_stats(self):
        """Log collection statistics."""
        for asset in self.assets:
            state = self.market_states[asset]
            self.logger.info(
                f"{asset}: collected={state.collected_count}, errors={state.error_count}, "
                f"market={state.market_id or 'none'}"
            )
    
    async def stop(self):
        """Stop and cleanup."""
        self.logger.info("Stopping FastOrchestrator...")
        self.running = False
        
        # Flush pending data
        for asset in self.assets:
            if self._pending_data[asset]:
                pending = self._pending_data[asset]
                ts_str = pending.get('_ts_second_str')
                if ts_str and ts_str not in self._written_seconds[asset]:
                    self._write_row(asset, pending, pending.get('_market_id'))
        
        # Close files
        for fh in self.file_handles.values():
            try:
                fh.close()
            except:
                pass
        
        # Close aiohttp session
        await self.pm_collector.close()
        
        self.logger.info("FastOrchestrator stopped")

