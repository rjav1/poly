"""
Fast Collector - Optimized for 1:1 second:point collection ratio.

Key optimizations:
1. Wall-clock anchored timing (no drift)
2. Parallel API calls via aiohttp  
3. Minimal hot-path overhead
4. Both CL and PM run independently with their own timing

Usage:
    collector = FastCollector(assets=["ETH"], target_matched=900)
    await collector.start()
"""

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Callable, Dict
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logger


@dataclass
class FastCollectionStats:
    """Collection stats."""
    asset: str
    cl_points: int = 0
    pm_points: int = 0
    elapsed_seconds: float = 0
    target_seconds: float = 0
    
    @property
    def progress_pct(self) -> float:
        if self.target_seconds <= 0:
            return 0
        return min(100, (self.elapsed_seconds / self.target_seconds) * 100)
    
    @property
    def cl_rate(self) -> float:
        """CL points per second."""
        if self.elapsed_seconds <= 0:
            return 0
        return self.cl_points / self.elapsed_seconds
    
    @property
    def pm_rate(self) -> float:
        """PM points per second."""
        if self.elapsed_seconds <= 0:
            return 0
        return self.pm_points / self.elapsed_seconds


class FastCollector:
    """
    High-performance collector with wall-clock anchored timing.
    
    Achieves ~1.0 points/second collection rate by:
    1. Using wall-clock anchored timing (targets exact second boundaries)
    2. Using async aiohttp for PM API calls (non-blocking)
    3. Running CL and PM independently in parallel
    4. Minimal overhead in the collection loop
    """
    
    BUFFER_MULTIPLIER = 1.2
    
    def __init__(
        self,
        assets: List[str],
        target_matched: int = 900,
        output_dir: str = "data_v2/raw",
        log_level: int = 20,
        sequential_cl: bool = False,
    ):
        """
        Initialize fast collector.
        
        Args:
            assets: List of asset symbols
            target_matched: Target matched points (will collect 1.2x seconds)
            output_dir: Output directory
            log_level: Logging level
            sequential_cl: If True, collect CL assets sequentially
        """
        self.assets = [a.upper() for a in assets]
        self.target_matched = target_matched
        self.target_seconds = int(target_matched * self.BUFFER_MULTIPLIER)
        self.output_dir = output_dir
        self.log_level = log_level
        self.sequential_cl = sequential_cl
        
        self.logger = setup_logger("fast_collector", level=log_level)
        self.running = False
        
        self.stats: Dict[str, FastCollectionStats] = {
            asset: FastCollectionStats(asset=asset, target_seconds=self.target_seconds)
            for asset in self.assets
        }
        
        self.cl_collector = None
        self.pm_orchestrator = None
        
        self.logger.info(f"FastCollector initialized for {self.assets}")
        self.logger.info(f"Target: {target_matched} points → collecting for {self.target_seconds}s")
    
    def _cl_callback(self, results: Dict):
        """Count CL points."""
        for asset, data in results.items():
            if data and isinstance(data, dict):
                self.stats[asset].cl_points += 1
    
    def _pm_callback(self, results: Dict):
        """Count PM points."""
        for asset, data in results.items():
            if data and isinstance(data, dict):
                self.stats[asset].pm_points += 1
    
    def get_stats(self) -> Dict[str, FastCollectionStats]:
        """Get current stats."""
        return self.stats.copy()
    
    async def start(
        self,
        interval: float = 1.0,
        callback: Optional[Callable[[Dict[str, FastCollectionStats]], None]] = None,
    ):
        """
        Start collection with wall-clock anchored timing.
        """
        # Use fast orchestrator for PM
        from src.fast_orchestrator import FastOrchestrator
        from src.chainlink.continuous_collector import ContinuousChainlinkCollector
        
        self.running = True
        start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info("STARTING FAST COLLECTION (Wall-Clock Anchored)")
        self.logger.info(f"Assets: {self.assets}")
        self.logger.info(f"Duration: {self.target_seconds}s")
        self.logger.info("=" * 60)
        
        # Initialize collectors
        self.cl_collector = ContinuousChainlinkCollector(
            assets=self.assets,
            output_dir=self.output_dir,
            log_level=self.log_level,
            sequential_collection=self.sequential_cl
        )
        
        # Use FAST orchestrator (aiohttp + wall-clock timing)
        self.pm_orchestrator = FastOrchestrator(
            assets=self.assets,
            output_dir=self.output_dir,
            preload_seconds=60,
            log_level=self.log_level
        )
        
        async def run_collector(coro, name: str):
            """Wrapper for error handling."""
            try:
                await coro
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.logger.error(f"{name} error: {e}")
        
        async def progress_monitor():
            """Monitor progress."""
            last_log = time.time()
            while self.running:
                await asyncio.sleep(1.0)
                
                elapsed = time.time() - start_time
                
                for asset in self.assets:
                    self.stats[asset].elapsed_seconds = elapsed
                
                if elapsed >= self.target_seconds:
                    self.logger.info(f"Target duration reached ({self.target_seconds}s)")
                    self.running = False
                    break
                
                # Log every 30s
                if time.time() - last_log >= 30:
                    for asset in self.assets:
                        s = self.stats[asset]
                        self.logger.info(
                            f"{asset}: CL={s.cl_points} ({s.cl_rate:.2f}/s), "
                            f"PM={s.pm_points} ({s.pm_rate:.2f}/s), "
                            f"{elapsed:.0f}/{self.target_seconds}s"
                        )
                    last_log = time.time()
                
                if callback:
                    callback(self.get_stats())
        
        try:
            await asyncio.gather(
                run_collector(
                    self.cl_collector.start(
                        interval=interval,
                        duration=self.target_seconds,
                        callback=self._cl_callback,
                        log_interval=60
                    ),
                    "CL"
                ),
                run_collector(
                    self.pm_orchestrator.start(
                        interval=interval,
                        duration=self.target_seconds,
                        callback=self._pm_callback,
                        log_interval=60
                    ),
                    "PM"
                ),
                progress_monitor(),
                return_exceptions=True
            )
            
        except asyncio.CancelledError:
            self.logger.info("Collection cancelled")
        except Exception as e:
            self.logger.error(f"Error: {e}")
        finally:
            self.running = False
            await self._cleanup()
            self._log_final_stats()
    
    async def _cleanup(self):
        """Cleanup collectors."""
        if self.cl_collector:
            try:
                await self.cl_collector.stop()
            except:
                pass
        
        if self.pm_orchestrator:
            try:
                await self.pm_orchestrator.stop()
            except:
                pass
    
    def _log_final_stats(self):
        """Log final stats."""
        self.logger.info("=" * 60)
        self.logger.info("COLLECTION COMPLETE")
        self.logger.info("=" * 60)
        
        for asset in self.assets:
            s = self.stats[asset]
            self.logger.info(
                f"{asset}: CL={s.cl_points} ({s.cl_rate:.3f}/s), "
                f"PM={s.pm_points} ({s.pm_rate:.3f}/s), "
                f"elapsed={s.elapsed_seconds:.0f}s"
            )
        
        self.logger.info("")
        self.logger.info("Next: python scripts/process_raw_data.py")
        self.logger.info("=" * 60)
    
    async def stop(self):
        """Stop collection."""
        self.logger.info("Stopping fast collector...")
        self.running = False
        await self._cleanup()

