"""
Raw Collector - Lightweight collection with NO real-time matching.

Design Philosophy:
- Collection should be as lightweight as possible for accuracy
- CL and PM run independently, writing to separate files
- Matching happens AFTER collection is complete (post-processing)
- Target duration uses 1.2x buffer since matched count is unknown until post-processing

Usage:
    collector = RawCollector(assets=["ETH"], target_matched=900)
    await collector.start()  # Runs for target_matched * 1.2 seconds
    # After collection, run process_raw_data.py to match and create datasets
"""

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Callable, Dict
from dataclasses import dataclass

from src.utils.logging import setup_logger


@dataclass
class CollectionStats:
    """Simple stats container."""
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


class RawCollector:
    """
    Lightweight collector that runs CL and PM independently.
    
    NO real-time matching - just collect and dump to files.
    Matching is done post-collection by process_raw_data.py.
    """
    
    # Buffer multiplier: collect 1.2x target seconds to ensure enough matched points
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
        Initialize raw collector.
        
        Args:
            assets: List of asset symbols to collect
            target_matched: Target number of matched points (will collect 1.2x this in seconds)
            output_dir: Directory for output files
            log_level: Logging level
            sequential_cl: If True, collect CL assets sequentially (more reliable)
        """
        self.assets = [a.upper() for a in assets]
        self.target_matched = target_matched
        self.target_seconds = int(target_matched * self.BUFFER_MULTIPLIER)
        self.output_dir = output_dir
        self.log_level = log_level
        self.sequential_cl = sequential_cl
        
        self.logger = setup_logger("raw_collector", level=log_level)
        self.running = False
        
        # Stats (lightweight - no locks needed since each collector updates its own)
        self.stats: Dict[str, CollectionStats] = {
            asset: CollectionStats(asset=asset, target_seconds=self.target_seconds)
            for asset in self.assets
        }
        
        # Collectors (initialized in start())
        self.cl_collector = None
        self.pm_orchestrator = None
        
        self.logger.info(f"RawCollector initialized for {self.assets}")
        self.logger.info(f"Target: {target_matched} matched points → collecting for {self.target_seconds}s (1.2x buffer)")
    
    def _cl_callback(self, results: Dict):
        """Minimal callback - just count points."""
        for asset, data in results.items():
            if data and isinstance(data, dict):
                self.stats[asset].cl_points += 1
    
    def _pm_callback(self, results: Dict):
        """Minimal callback - just count points."""
        for asset, data in results.items():
            if data and isinstance(data, dict):
                self.stats[asset].pm_points += 1
    
    def get_stats(self) -> Dict[str, CollectionStats]:
        """Get current collection stats."""
        return self.stats.copy()
    
    async def start(
        self,
        interval: float = 1.0,
        callback: Optional[Callable[[Dict[str, CollectionStats]], None]] = None,
    ):
        """
        Start raw collection for target_seconds duration.
        
        Args:
            interval: Collection interval in seconds
            callback: Optional callback for progress updates
        """
        from src.chainlink.continuous_collector import ContinuousChainlinkCollector
        from src.orchestrator import MarketOrchestrator
        
        self.running = True
        start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info("STARTING LIGHTWEIGHT RAW COLLECTION")
        self.logger.info(f"Assets: {self.assets}")
        self.logger.info(f"Duration: {self.target_seconds}s ({self.target_matched} target × 1.2 buffer)")
        self.logger.info(f"Interval: {interval}s")
        self.logger.info("Matching will be done POST-COLLECTION by process_raw_data.py")
        self.logger.info("=" * 60)
        
        # Initialize collectors with minimal callbacks
        self.cl_collector = ContinuousChainlinkCollector(
            assets=self.assets,
            output_dir=self.output_dir,
            log_level=self.log_level,
            sequential_collection=self.sequential_cl
        )
        
        self.pm_orchestrator = MarketOrchestrator(
            assets=self.assets,
            output_dir=self.output_dir,
            preload_seconds=60,
            log_level=self.log_level
        )
        
        async def run_with_duration(coro, name: str):
            """Wrapper to ensure each collector runs for exactly target_seconds."""
            try:
                await coro
            except asyncio.CancelledError:
                self.logger.debug(f"{name} cancelled")
            except Exception as e:
                self.logger.error(f"{name} error: {e}")
        
        async def progress_monitor():
            """Monitor progress and call user callback."""
            last_log = time.time()
            while self.running:
                await asyncio.sleep(1.0)
                
                elapsed = time.time() - start_time
                
                # Update elapsed in stats
                for asset in self.assets:
                    self.stats[asset].elapsed_seconds = elapsed
                
                # Check if we've reached target duration
                if elapsed >= self.target_seconds:
                    self.logger.info(f"Target duration reached ({self.target_seconds}s)")
                    self.running = False
                    break
                
                # Progress logging every 10s
                if time.time() - last_log >= 10:
                    for asset in self.assets:
                        s = self.stats[asset]
                        self.logger.info(
                            f"{asset}: CL={s.cl_points}, PM={s.pm_points}, "
                            f"elapsed={elapsed:.0f}/{self.target_seconds}s ({s.progress_pct:.1f}%)"
                        )
                    last_log = time.time()
                
                # User callback
                if callback:
                    callback(self.get_stats())
        
        try:
            # Run both collectors in parallel with duration limit
            # They will stop when self.running becomes False
            await asyncio.gather(
                run_with_duration(
                    self.cl_collector.start(
                        interval=interval,
                        duration=self.target_seconds,  # Hard duration limit
                        callback=self._cl_callback,
                        log_interval=60
                    ),
                    "CL"
                ),
                run_with_duration(
                    self.pm_orchestrator.start(
                        interval=interval,
                        duration=self.target_seconds,  # Hard duration limit
                        callback=self._pm_callback,
                        log_interval=60
                    ),
                    "PM"
                ),
                progress_monitor(),
                return_exceptions=True
            )
            
        except asyncio.CancelledError:
            self.logger.info("Collection cancelled by user")
        except Exception as e:
            self.logger.error(f"Collection error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            await self._cleanup()
            self._log_final_stats()
    
    async def _cleanup(self):
        """Clean up collectors."""
        if self.cl_collector:
            try:
                await self.cl_collector.stop()
            except Exception as e:
                self.logger.debug(f"CL cleanup error: {e}")
        
        if self.pm_orchestrator:
            try:
                await self.pm_orchestrator.stop()
            except Exception as e:
                self.logger.debug(f"PM cleanup error: {e}")
    
    def _log_final_stats(self):
        """Log final collection statistics."""
        self.logger.info("=" * 60)
        self.logger.info("COLLECTION COMPLETE")
        self.logger.info("=" * 60)
        for asset in self.assets:
            s = self.stats[asset]
            self.logger.info(
                f"{asset}: CL={s.cl_points} points, PM={s.pm_points} points, "
                f"elapsed={s.elapsed_seconds:.0f}s"
            )
        self.logger.info("")
        self.logger.info("NEXT STEP: Run 'python scripts/process_raw_data.py' to match and create datasets")
        self.logger.info("=" * 60)
    
    async def stop(self):
        """Stop collection gracefully."""
        self.logger.info("Stopping raw collector...")
        self.running = False
        await self._cleanup()


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

async def main():
    """Run raw collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Lightweight raw data collector")
    parser.add_argument(
        "--assets",
        type=str,
        default="ETH",
        help="Comma-separated list of assets (default: ETH)"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=900,
        help="Target matched points (will collect 1.2x this in seconds)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Collection interval in seconds"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_v2/raw",
        help="Output directory"
    )
    parser.add_argument(
        "--sequential-cl",
        action="store_true",
        help="Collect Chainlink assets sequentially (more reliable)"
    )
    
    args = parser.parse_args()
    
    assets = [a.strip().upper() for a in args.assets.split(",")]
    
    collector = RawCollector(
        assets=assets,
        target_matched=args.target,
        output_dir=args.output_dir,
        sequential_cl=args.sequential_cl,
    )
    
    try:
        await collector.start(interval=args.interval)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        await collector.stop()
    
    print("\nCollection complete. Now run:")
    print("  python scripts/process_raw_data.py")


if __name__ == "__main__":
    asyncio.run(main())

