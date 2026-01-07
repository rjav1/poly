"""
Matching Collector - Collects data and tracks matched timestamps between CL and PM.

Duration means "collect until we have N matched data points", not N seconds elapsed.
Matched = same second timestamp in both CL and PM data.
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Set, Optional, Callable, List
from collections import defaultdict
import pandas as pd

from src.utils.logging import setup_logger


class TimestampMatcher:
    """Tracks matched timestamps between Chainlink and Polymarket data."""
    
    def __init__(self):
        self.cl_timestamps: Dict[str, Set[str]] = defaultdict(set)  # asset -> set of second timestamps
        self.pm_timestamps: Dict[str, Set[str]] = defaultdict(set)  # asset -> set of second timestamps
        self.matched_timestamps: Dict[str, Set[str]] = defaultdict(set)  # asset -> set of matched second timestamps
        self.latest_cl_ts: Dict[str, Optional[datetime]] = defaultdict(lambda: None)  # asset -> latest CL timestamp
        self.latest_pm_ts: Dict[str, Optional[datetime]] = defaultdict(lambda: None)  # asset -> latest PM timestamp
        self.lock = asyncio.Lock()
    
    def _floor_to_second(self, ts) -> str:
        """Floor datetime to second and return as string key."""
        if isinstance(ts, str):
            try:
                ts = pd.to_datetime(ts)
            except:
                return ts[:19]  # Fallback to string slicing
        return ts.strftime('%Y-%m-%d %H:%M:%S')
    
    async def add_cl_timestamp(self, asset: str, timestamp):
        """Add a Chainlink timestamp and check for match."""
        async with self.lock:
            second_key = self._floor_to_second(timestamp)
            self.cl_timestamps[asset].add(second_key)
            
            # Check if PM already has this second
            if second_key in self.pm_timestamps[asset]:
                self.matched_timestamps[asset].add(second_key)
    
    async def add_pm_timestamp(self, asset: str, timestamp):
        """Add a Polymarket timestamp and check for match."""
        async with self.lock:
            second_key = self._floor_to_second(timestamp)
            self.pm_timestamps[asset].add(second_key)
            
            # Check if CL already has this second
            if second_key in self.cl_timestamps[asset]:
                self.matched_timestamps[asset].add(second_key)
    
    def add_cl_timestamp_sync(self, asset: str, timestamp):
        """Add a Chainlink timestamp synchronously."""
        # Convert to datetime if string
        if isinstance(timestamp, str):
            try:
                ts_dt = pd.to_datetime(timestamp)
            except:
                ts_dt = None
        else:
            ts_dt = timestamp
        
        second_key = self._floor_to_second(timestamp)
        self.cl_timestamps[asset].add(second_key)
        
        # Track latest timestamp
        if ts_dt and (self.latest_cl_ts[asset] is None or ts_dt > self.latest_cl_ts[asset]):
            self.latest_cl_ts[asset] = ts_dt
        
        if second_key in self.pm_timestamps[asset]:
            self.matched_timestamps[asset].add(second_key)
    
    def add_pm_timestamp_sync(self, asset: str, timestamp):
        """Add a Polymarket timestamp synchronously."""
        # Convert to datetime if string
        if isinstance(timestamp, str):
            try:
                ts_dt = pd.to_datetime(timestamp)
            except:
                ts_dt = None
        else:
            ts_dt = timestamp
        
        second_key = self._floor_to_second(timestamp)
        self.pm_timestamps[asset].add(second_key)
        
        # Track latest timestamp
        if ts_dt and (self.latest_pm_ts[asset] is None or ts_dt > self.latest_pm_ts[asset]):
            self.latest_pm_ts[asset] = ts_dt
        
        if second_key in self.cl_timestamps[asset]:
            self.matched_timestamps[asset].add(second_key)
    
    def get_matched_count(self, asset: str) -> int:
        """Get count of matched timestamps for an asset."""
        return len(self.matched_timestamps[asset])
    
    def get_total_matched_count(self) -> int:
        """Get total matched timestamps across all assets."""
        return sum(len(ts) for ts in self.matched_timestamps.values())
    
    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics for all assets."""
        stats = {}
        for asset in set(list(self.cl_timestamps.keys()) + list(self.pm_timestamps.keys())):
            stats[asset] = {
                'cl_unique_seconds': len(self.cl_timestamps[asset]),
                'pm_unique_seconds': len(self.pm_timestamps[asset]),
                'matched_seconds': len(self.matched_timestamps[asset])
            }
        return stats
    
    def get_latest_timestamps(self, asset: str) -> Dict[str, Optional[datetime]]:
        """Get latest CL and PM timestamps for an asset."""
        return {
            'cl_latest': self.latest_cl_ts[asset],
            'pm_latest': self.latest_pm_ts[asset]
        }


class MatchingCollector:
    """
    Collector that stops when target matched points is reached.
    
    This wraps the CL and PM collectors and tracks timestamp matching.
    Duration parameter means "target number of matched data points", not wall-clock time.
    """
    
    def __init__(
        self,
        assets: List[str],
        output_dir: str = "data_v2/raw",
        preload_seconds: int = 60,
        log_level: int = 20,
        sequential_cl_collection: bool = False,
    ):
        self.assets = assets
        self.output_dir = output_dir
        self.preload_seconds = preload_seconds
        self.logger = setup_logger("matching_collector", level=log_level)
        self.sequential_cl_collection = sequential_cl_collection
        
        self.matcher = TimestampMatcher()
        self.running = False
        self.target_matched_points = 0
        
        # Collectors will be initialized in start()
        self.cl_collector = None
        self.pm_orchestrator = None
        
        # Stats tracking
        self.stats = defaultdict(lambda: {
            'chainlink': {'points': 0, 'errors': 0, 'last_price': None},
            'polymarket': {'points': 0, 'errors': 0, 'last_market': None},
            'matched': 0
        })
    
    def _cl_callback(self, results: Dict):
        """Callback for Chainlink data - updates matcher."""
        if not results:
            return
        
        for asset, data in results.items():
            if data and isinstance(data, dict):
                self.stats[asset]['chainlink']['points'] += 1
                if 'mid' in data:
                    self.stats[asset]['chainlink']['last_price'] = data.get('mid')
                
                # Extract timestamp - use source_timestamp (new name) or timestamp (legacy)
                timestamp = data.get('source_timestamp') or data.get('timestamp')
                if timestamp:
                    self.matcher.add_cl_timestamp_sync(asset, timestamp)
                    self.stats[asset]['matched'] = self.matcher.get_matched_count(asset)
    
    def _pm_callback(self, results: Dict):
        """Callback for Polymarket data - updates matcher."""
        if not results:
            return
        
        for asset, data in results.items():
            if data and isinstance(data, dict):
                self.stats[asset]['polymarket']['points'] += 1
                market_id = data.get('market_id')
                if market_id:
                    self.stats[asset]['polymarket']['last_market'] = market_id
                
                # Extract timestamp - use source_timestamp (new name) or timestamp (legacy)
                # PM provides API timestamp from /book endpoint - this is the actual data timestamp
                timestamp = data.get('source_timestamp') or data.get('timestamp')
                if timestamp:
                    self.matcher.add_pm_timestamp_sync(asset, timestamp)
                    self.stats[asset]['matched'] = self.matcher.get_matched_count(asset)
    
    def _check_target_reached(self) -> bool:
        """Check if target matched points has been reached for all assets."""
        if self.target_matched_points <= 0:
            return False  # No target, run indefinitely
        
        # Check each asset
        for asset in self.assets:
            if self.matcher.get_matched_count(asset) < self.target_matched_points:
                return False
        
        return True
    
    def get_progress(self) -> Dict[str, Dict]:
        """Get progress for each asset."""
        progress = {}
        for asset in self.assets:
            matched = self.matcher.get_matched_count(asset)
            latest_ts = self.matcher.get_latest_timestamps(asset)
            
            # Calculate delay between CL and PM (PM - CL in seconds)
            cl_delay_sec = None
            if latest_ts['cl_latest'] and latest_ts['pm_latest']:
                delay = (latest_ts['pm_latest'] - latest_ts['cl_latest']).total_seconds()
                cl_delay_sec = max(0, delay)  # Only positive delays (CL behind PM)
            
            progress[asset] = {
                'matched': matched,
                'target': self.target_matched_points,
                'percentage': (matched / self.target_matched_points * 100) if self.target_matched_points > 0 else 0,
                'cl_points': self.stats[asset]['chainlink']['points'],
                'pm_points': self.stats[asset]['polymarket']['points'],
                'cl_price': self.stats[asset]['chainlink']['last_price'],
                'pm_market': self.stats[asset]['polymarket']['last_market'],
                'cl_latest_ts': latest_ts['cl_latest'],
                'pm_latest_ts': latest_ts['pm_latest'],
                'cl_delay_seconds': cl_delay_sec
            }
        return progress
    
    async def start(
        self,
        target_matched_points: int = 60,
        interval: float = 1.0,
        callback: Optional[Callable] = None,
        max_wall_time: Optional[float] = None
    ):
        """
        Start collection until target matched points is reached.
        
        Args:
            target_matched_points: Number of matched timestamps to collect for each asset
            interval: Collection interval in seconds
            callback: Optional callback called with progress updates
            max_wall_time: Maximum wall-clock time in seconds (safety limit)
        """
        from src.chainlink.continuous_collector import ContinuousChainlinkCollector
        from src.orchestrator import MarketOrchestrator
        
        self.target_matched_points = target_matched_points
        self.running = True
        start_time = time.time()
        
        self.logger.info(f"Starting matching collection for assets: {self.assets}")
        self.logger.info(f"Target: {target_matched_points} matched points per asset")
        self.logger.info(f"Note: Due to ~2 min CL delay, this will take at least {target_matched_points + 130}s")
        
        # Initialize collectors
        self.cl_collector = ContinuousChainlinkCollector(
            assets=self.assets,
            output_dir=self.output_dir,
            log_level=20,
            sequential_collection=self.sequential_cl_collection
        )
        
        self.pm_orchestrator = MarketOrchestrator(
            assets=self.assets,
            output_dir=self.output_dir,
            preload_seconds=self.preload_seconds,
            log_level=20
        )
        
        # Monitor task that checks for target completion
        async def monitor_and_stop():
            last_log_time = time.time()
            while self.running:
                await asyncio.sleep(1.0)
                
                elapsed = time.time() - start_time
                
                # Safety limit on wall-clock time
                if max_wall_time and elapsed >= max_wall_time:
                    self.logger.warning(f"Max wall time ({max_wall_time}s) reached, stopping")
                    self.running = False
                    self.cl_collector.running = False
                    self.pm_orchestrator.running = False
                    break
                
                # Check target
                if self._check_target_reached():
                    self.logger.info("Target matched points reached!")
                    self.running = False
                    self.cl_collector.running = False
                    self.pm_orchestrator.running = False
                    break
                
                # Periodic logging
                if time.time() - last_log_time >= 10:
                    progress = self.get_progress()
                    for asset, p in progress.items():
                        self.logger.info(
                            f"{asset}: matched={p['matched']}/{p['target']} ({p['percentage']:.1f}%), "
                            f"CL={p['cl_points']}, PM={p['pm_points']}"
                        )
                    last_log_time = time.time()
                
                # Call user callback
                if callback:
                    callback(self.get_progress())
        
        try:
            # Run all collectors in parallel
            # Each collector has its own start() method with callbacks
            await asyncio.gather(
                self.cl_collector.start(
                    interval=interval,
                    duration=None,  # No time limit, we control stopping via self.running
                    callback=self._cl_callback,
                    log_interval=60
                ),
                self.pm_orchestrator.start(
                    interval=interval,
                    duration=None,  # No time limit
                    callback=self._pm_callback,
                    log_interval=60
                ),
                monitor_and_stop(),
                return_exceptions=True
            )
            
        except asyncio.CancelledError:
            self.logger.info("Collection cancelled")
        except Exception as e:
            self.logger.error(f"Collection error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
    
    async def stop(self):
        """Stop collection and cleanup."""
        self.logger.info("Stopping matching collector...")
        self.running = False
        
        # Signal collectors to stop
        if self.cl_collector:
            self.cl_collector.running = False
        if self.pm_orchestrator:
            self.pm_orchestrator.running = False
        
        # Wait a moment for loops to exit
        await asyncio.sleep(0.5)
        
        # Stop Chainlink collector (cleanup browser)
        if self.cl_collector:
            try:
                await self.cl_collector.stop()
            except Exception as e:
                self.logger.debug(f"CL stop error: {e}")
        
        self.logger.info("Matching collector stopped")
    
    def get_final_stats(self) -> Dict:
        """Get final collection statistics."""
        return {
            'matcher_stats': self.matcher.get_stats(),
            'asset_stats': dict(self.stats),
            'progress': self.get_progress()
        }
