#!/usr/bin/env python3
"""
V3 Data Collector - RTDS-based high-quality data collection.

This is the new standard for data collection, using:
1. Polymarket RTDS websocket for real-time Chainlink prices (no 1-min delay!)
2. Polymarket API for orderbook data (6 levels of depth)
3. Perfect timestamp alignment (both collected at same moment)
4. Near 100% coverage (streaming, not polling with delays)

KEY ADVANTAGES over V2:
- Real-time Chainlink prices via websocket (matches what Polymarket uses for resolution)
- No 60-second oracle delay like the old on-chain method
- No coverage gaps from timing mismatches
- More robust (websocket stream vs HTTP polling)
- Less resource intensive

DESIGN:
- RTDSChainlinkCollector runs continuously, updating latest_cl on each message
- PolymarketCollector polls on each tick
- Both are combined into aligned snapshots
- Data is saved in the same format as V2 for compatibility with build scripts
"""

import asyncio
import csv
import json
import time
import signal
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import STORAGE, SUPPORTED_ASSETS


@dataclass
class AssetStats:
    """Statistics for a single asset."""
    asset: str
    cl_points: int = 0
    pm_points: int = 0
    matched_points: int = 0
    elapsed_seconds: float = 0.0
    current_market: Optional[str] = None
    cl_price: Optional[float] = None
    pm_up_mid: Optional[float] = None
    status: str = "initializing"


@dataclass
class V3CollectorConfig:
    """Configuration for V3 collector."""
    # Collection settings
    poll_interval: float = 1.0  # How often to collect PM and save data
    
    # RTDS settings
    rtds_endpoint: str = "wss://ws-live-data.polymarket.com"
    
    # Output settings
    output_dir: str = "data_v2/raw"
    
    # Market settings
    market_duration_seconds: int = 900  # 15 minutes


class V3Collector:
    """
    V3 Data Collector using RTDS for Chainlink prices.
    
    This collector:
    1. Connects to Polymarket RTDS websocket for real-time Chainlink prices
    2. Polls Polymarket API for orderbook data
    3. Saves aligned snapshots to CSV
    4. Automatically handles market transitions
    """
    
    def __init__(
        self,
        assets: List[str],
        output_dir: Optional[str] = None,
        config: Optional[V3CollectorConfig] = None,
        log_level: int = 20,
    ):
        """
        Initialize V3 collector.
        
        Args:
            assets: List of assets to collect (ETH, BTC, SOL, XRP)
            output_dir: Output directory for raw data
            config: Collector configuration
            log_level: Logging level (20=INFO, 30=WARNING)
        """
        self.assets = [a.upper() for a in assets]
        self.config = config or V3CollectorConfig()
        self.output_dir = Path(output_dir or self.config.output_dir)
        self.log_level = log_level
        
        # Validate assets
        for asset in self.assets:
            if asset not in SUPPORTED_ASSETS:
                raise ValueError(f"Unsupported asset: {asset}. Supported: {SUPPORTED_ASSETS}")
        
        # Create output directories
        self.cl_dir = self.output_dir / "chainlink"
        self.pm_dir = self.output_dir / "polymarket"
        self.cl_dir.mkdir(parents=True, exist_ok=True)
        self.pm_dir.mkdir(parents=True, exist_ok=True)
        
        # Import collectors (lazy import to avoid import errors if websockets not installed)
        from paper_trading.chainlink_rtds import RTDSChainlinkCollector
        from src.polymarket.collector import PolymarketCollector
        
        # Initialize RTDS collectors (one per asset)
        self.rtds_collectors: Dict[str, RTDSChainlinkCollector] = {}
        for asset in self.assets:
            self.rtds_collectors[asset] = RTDSChainlinkCollector(
                asset=asset,
                endpoint=self.config.rtds_endpoint
            )
        
        # Initialize Polymarket collector (shared)
        self.pm_collector = PolymarketCollector(log_level=40)  # ERROR only
        
        # Market state per asset
        self.markets: Dict[str, Optional[Dict]] = {a: None for a in self.assets}
        self.market_start_times: Dict[str, Optional[datetime]] = {a: None for a in self.assets}
        self.market_end_times: Dict[str, Optional[datetime]] = {a: None for a in self.assets}
        
        # Latest data per asset
        self.latest_cl: Dict[str, Optional[Dict]] = {a: None for a in self.assets}
        self.latest_pm: Dict[str, Optional[Dict]] = {a: None for a in self.assets}
        
        # Stats
        self.stats: Dict[str, AssetStats] = {
            a: AssetStats(asset=a) for a in self.assets
        }
        
        # CSV writers
        self._cl_files: Dict[str, any] = {}
        self._pm_files: Dict[str, any] = {}
        self._cl_writers: Dict[str, csv.DictWriter] = {}
        self._pm_writers: Dict[str, csv.DictWriter] = {}
        
        # Control
        self.running = False
        self._rtds_tasks: Dict[str, asyncio.Task] = {}
        self._start_time: Optional[float] = None
    
    def _log(self, level: int, msg: str):
        """Simple logging."""
        if level >= self.log_level:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [V3Collector] {msg}")
    
    def _init_csv_files(self):
        """Initialize CSV files for each asset."""
        # Chainlink columns (compatible with V2)
        cl_columns = [
            "source_timestamp",  # RTDS timestamp (when Chainlink updated)
            "received_timestamp",  # When we received via websocket
            "asset",
            "mid",
            "staleness_seconds",
            "is_observed",
        ]
        
        # Polymarket columns - 6 levels of depth (compatible with V2)
        pm_columns = [
            "source_timestamp",  # When PM says data was valid
            "received_timestamp",  # When we received it
            "timestamp_ms",
            "asset",
            "market_id",
            "up_mid",
            "up_bid", "up_bid_size", "up_ask", "up_ask_size",
            "up_bid_2", "up_bid_2_size", "up_ask_2", "up_ask_2_size",
            "up_bid_3", "up_bid_3_size", "up_ask_3", "up_ask_3_size",
            "up_bid_4", "up_bid_4_size", "up_ask_4", "up_ask_4_size",
            "up_bid_5", "up_bid_5_size", "up_ask_5", "up_ask_5_size",
            "up_bid_6", "up_bid_6_size", "up_ask_6", "up_ask_6_size",
            "down_mid",
            "down_bid", "down_bid_size", "down_ask", "down_ask_size",
            "down_bid_2", "down_bid_2_size", "down_ask_2", "down_ask_2_size",
            "down_bid_3", "down_bid_3_size", "down_ask_3", "down_ask_3_size",
            "down_bid_4", "down_bid_4_size", "down_ask_4", "down_ask_4_size",
            "down_bid_5", "down_bid_5_size", "down_ask_5", "down_ask_5_size",
            "down_bid_6", "down_bid_6_size", "down_ask_6", "down_ask_6_size",
            "is_observed",
        ]
        
        for asset in self.assets:
            # Create asset directories
            asset_cl_dir = self.cl_dir / asset
            asset_pm_dir = self.pm_dir / asset
            asset_cl_dir.mkdir(parents=True, exist_ok=True)
            asset_pm_dir.mkdir(parents=True, exist_ok=True)
            
            # Open files
            cl_path = asset_cl_dir / f"chainlink_{asset}_continuous.csv"
            pm_path = asset_pm_dir / f"polymarket_{asset}_continuous.csv"
            
            # Check if files exist (append mode)
            cl_exists = cl_path.exists() and cl_path.stat().st_size > 0
            pm_exists = pm_path.exists() and pm_path.stat().st_size > 0
            
            self._cl_files[asset] = open(cl_path, 'a', newline='', encoding='utf-8')
            self._pm_files[asset] = open(pm_path, 'a', newline='', encoding='utf-8')
            
            self._cl_writers[asset] = csv.DictWriter(self._cl_files[asset], fieldnames=cl_columns)
            self._pm_writers[asset] = csv.DictWriter(self._pm_files[asset], fieldnames=pm_columns)
            
            # Write headers if new files
            if not cl_exists:
                self._cl_writers[asset].writeheader()
                self._cl_files[asset].flush()
            if not pm_exists:
                self._pm_writers[asset].writeheader()
                self._pm_files[asset].flush()
    
    def _close_csv_files(self):
        """Close all CSV files."""
        for f in self._cl_files.values():
            try:
                f.close()
            except:
                pass
        for f in self._pm_files.values():
            try:
                f.close()
            except:
                pass
    
    async def _find_market(self, asset: str) -> Optional[Dict]:
        """Find active market for an asset."""
        try:
            market = await asyncio.to_thread(
                self.pm_collector.find_active_market,
                asset
            )
            if market:
                self.markets[asset] = market
                
                # Extract times from slug
                slug = market.get("market_slug", "")
                try:
                    parts = slug.split("-")
                    if len(parts) >= 4:
                        start_unix = int(parts[-1])
                        start_time = datetime.fromtimestamp(start_unix, tz=timezone.utc)
                        end_time = start_time + timedelta(seconds=self.config.market_duration_seconds)
                        self.market_start_times[asset] = start_time
                        self.market_end_times[asset] = end_time
                except:
                    pass
                
                self._log(20, f"{asset}: Found market {slug}")
                self.stats[asset].current_market = slug
                return market
            else:
                self._log(30, f"{asset}: No active market found")
                return None
        except Exception as e:
            self._log(40, f"{asset}: Error finding market: {e}")
            return None
    
    def _on_rtds_update(self, asset: str):
        """Create callback for RTDS updates for a specific asset."""
        def callback(price_data):
            self.latest_cl[asset] = {
                "price": price_data.price,
                "chainlink_timestamp": price_data.chainlink_timestamp,
                "local_timestamp": price_data.local_timestamp,
                "staleness_seconds": price_data.staleness_seconds,
            }
            self.stats[asset].cl_price = price_data.price
        return callback
    
    async def _start_rtds_collectors(self):
        """Start RTDS collectors for all assets."""
        for asset in self.assets:
            collector = self.rtds_collectors[asset]
            callback = self._on_rtds_update(asset)
            
            # Start collector
            await collector.start(callback=callback)
            
            # Start receive loop in background
            async def receive_loop(c=collector, cb=callback):
                await c._receive_loop(callback=cb)
            
            self._rtds_tasks[asset] = asyncio.create_task(receive_loop())
            self._log(20, f"{asset}: RTDS collector started")
        
        # Wait for first data from all collectors
        self._log(20, "Waiting for RTDS data...")
        for _ in range(50):  # Wait up to 5 seconds
            if all(self.latest_cl[a] is not None for a in self.assets):
                break
            await asyncio.sleep(0.1)
    
    async def _stop_rtds_collectors(self):
        """Stop all RTDS collectors."""
        for asset, collector in self.rtds_collectors.items():
            await collector.stop()
            if asset in self._rtds_tasks:
                self._rtds_tasks[asset].cancel()
                try:
                    await self._rtds_tasks[asset]
                except asyncio.CancelledError:
                    pass
    
    async def _collect_polymarket(self, asset: str) -> Optional[Dict]:
        """Collect Polymarket data for an asset."""
        market = self.markets.get(asset)
        if not market:
            return None
        
        token_up = market.get("token_id_up")
        token_down = market.get("token_id_down")
        
        if not token_up or not token_down:
            return None
        
        try:
            local_ts_before = datetime.now(timezone.utc)
            
            raw_data = await asyncio.to_thread(
                self.pm_collector.get_market_data,
                token_up,
                token_down
            )
            
            local_ts_after = datetime.now(timezone.utc)
            local_ts = local_ts_before + (local_ts_after - local_ts_before) / 2
            
            if raw_data:
                raw_data["local_timestamp"] = local_ts
                raw_data["market_slug"] = market.get("market_slug")
                self.latest_pm[asset] = raw_data
                return raw_data
            return None
        except Exception as e:
            self._log(40, f"{asset}: PM collection error: {e}")
            return None
    
    def _write_chainlink(self, asset: str):
        """Write Chainlink data to CSV."""
        cl_data = self.latest_cl.get(asset)
        if not cl_data:
            return
        
        row = {
            "source_timestamp": cl_data["chainlink_timestamp"].isoformat(),
            "received_timestamp": cl_data["local_timestamp"].isoformat(),
            "asset": asset,
            "mid": cl_data["price"],
            "staleness_seconds": cl_data["staleness_seconds"],
            "is_observed": 1,
        }
        
        self._cl_writers[asset].writerow(row)
        self._cl_files[asset].flush()
        self.stats[asset].cl_points += 1
    
    def _write_polymarket(self, asset: str):
        """Write Polymarket data to CSV."""
        pm_data = self.latest_pm.get(asset)
        if not pm_data:
            return
        
        market = self.markets.get(asset)
        local_ts = pm_data.get("local_timestamp", datetime.now(timezone.utc))
        
        row = {
            "source_timestamp": pm_data.get("source_timestamp", local_ts.isoformat()),
            "received_timestamp": local_ts.isoformat(),
            "timestamp_ms": int(local_ts.timestamp() * 1000),
            "asset": asset,
            "market_id": market.get("market_slug", "") if market else "",
            "up_mid": pm_data.get("up_mid"),
            "up_bid": pm_data.get("up_bid"),
            "up_bid_size": pm_data.get("up_bid_size"),
            "up_ask": pm_data.get("up_ask"),
            "up_ask_size": pm_data.get("up_ask_size"),
            "down_mid": pm_data.get("down_mid"),
            "down_bid": pm_data.get("down_bid"),
            "down_bid_size": pm_data.get("down_bid_size"),
            "down_ask": pm_data.get("down_ask"),
            "down_ask_size": pm_data.get("down_ask_size"),
            "is_observed": 1,
        }
        
        # Add levels 2-6
        for level in range(2, 7):
            for side in ["up", "down"]:
                for field in ["bid", "ask"]:
                    key = f"{side}_{field}_{level}"
                    row[key] = pm_data.get(key)
                    row[f"{key}_size"] = pm_data.get(f"{key}_size")
        
        self._pm_writers[asset].writerow(row)
        self._pm_files[asset].flush()
        self.stats[asset].pm_points += 1
    
    async def _collection_loop(self, duration: Optional[float] = None):
        """Main collection loop."""
        start_time = time.time()
        self._start_time = start_time
        last_market_check = 0
        
        while self.running:
            loop_start = time.time()
            
            # Check duration
            if duration and (loop_start - start_time) >= duration:
                self._log(20, "Collection duration reached")
                break
            
            # Check markets every 10 seconds
            if loop_start - last_market_check >= 10:
                for asset in self.assets:
                    if not self.markets.get(asset):
                        await self._find_market(asset)
                    else:
                        # Check if market ended
                        end_time = self.market_end_times.get(asset)
                        if end_time and datetime.now(timezone.utc) > end_time:
                            self._log(20, f"{asset}: Market ended, finding next...")
                            self.markets[asset] = None
                            await self._find_market(asset)
                
                last_market_check = loop_start
            
            # Collect PM data for all assets in parallel
            pm_tasks = [self._collect_polymarket(a) for a in self.assets]
            await asyncio.gather(*pm_tasks, return_exceptions=True)
            
            # Write data
            for asset in self.assets:
                # Write CL data (RTDS updates continuously)
                if self.latest_cl.get(asset):
                    self._write_chainlink(asset)
                
                # Write PM data
                if self.latest_pm.get(asset):
                    self._write_polymarket(asset)
                
                # Update stats
                self.stats[asset].elapsed_seconds = loop_start - start_time
                
                # Count matched points (where we have both CL and PM)
                if self.latest_cl.get(asset) and self.latest_pm.get(asset):
                    self.stats[asset].matched_points += 1
                    self.stats[asset].status = "collecting"
                elif self.latest_cl.get(asset):
                    self.stats[asset].status = "waiting for PM"
                elif self.latest_pm.get(asset):
                    self.stats[asset].status = "waiting for CL"
                else:
                    self.stats[asset].status = "no data"
            
            # Sleep to maintain interval
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.config.poll_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    
    async def start(self, duration: Optional[float] = None, target_points: Optional[int] = None):
        """
        Start data collection.
        
        Args:
            duration: How long to collect in seconds (None = until stopped)
            target_points: Target matched points per asset (alternative to duration)
        """
        if self.running:
            return
        
        self.running = True
        self._log(20, f"Starting V3 collector for {self.assets}")
        
        # Calculate duration from target_points if provided
        if target_points and not duration:
            # With RTDS, we get ~1 point per second, so duration ≈ target_points
            duration = target_points * 1.1  # 10% buffer
            self._log(20, f"Target: {target_points} points → {duration:.0f}s duration")
        
        try:
            # Initialize CSV files
            self._init_csv_files()
            
            # Start RTDS collectors
            await self._start_rtds_collectors()
            
            # Find initial markets
            for asset in self.assets:
                await self._find_market(asset)
            
            # Run collection loop
            await self._collection_loop(duration=duration)
            
        except asyncio.CancelledError:
            self._log(20, "Collection cancelled")
        except Exception as e:
            self._log(40, f"Collection error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop data collection."""
        self._log(20, "Stopping V3 collector...")
        self.running = False
        
        # Stop RTDS collectors
        await self._stop_rtds_collectors()
        
        # Close CSV files
        self._close_csv_files()
        
        self._log(20, "V3 collector stopped")
    
    def get_stats(self) -> Dict[str, AssetStats]:
        """Get current statistics."""
        return self.stats
    
    def get_summary(self) -> Dict:
        """Get collection summary."""
        total_cl = sum(s.cl_points for s in self.stats.values())
        total_pm = sum(s.pm_points for s in self.stats.values())
        total_matched = sum(s.matched_points for s in self.stats.values())
        
        return {
            "assets": self.assets,
            "total_cl_points": total_cl,
            "total_pm_points": total_pm,
            "total_matched_points": total_matched,
            "per_asset": {
                asset: {
                    "cl_points": s.cl_points,
                    "pm_points": s.pm_points,
                    "matched_points": s.matched_points,
                    "elapsed_seconds": s.elapsed_seconds,
                    "current_market": s.current_market,
                }
                for asset, s in self.stats.items()
            }
        }


# =============================================================================
# TESTING
# =============================================================================

async def test_v3_collector():
    """Test V3 collector."""
    print("=" * 70)
    print("V3 COLLECTOR TEST")
    print("=" * 70)
    print()
    print("Testing RTDS-based data collection for 30 seconds...")
    print()
    
    collector = V3Collector(
        assets=["ETH"],
        log_level=20
    )
    
    # Run for 30 seconds
    await collector.start(duration=30)
    
    # Print summary
    summary = collector.get_summary()
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"CL points: {summary['total_cl_points']}")
    print(f"PM points: {summary['total_pm_points']}")
    print(f"Matched points: {summary['total_matched_points']}")
    print()
    print("Per asset:")
    for asset, stats in summary['per_asset'].items():
        print(f"  {asset}: CL={stats['cl_points']}, PM={stats['pm_points']}, Matched={stats['matched_points']}")
    print()
    print("✓ Test complete!")


if __name__ == "__main__":
    asyncio.run(test_v3_collector())

