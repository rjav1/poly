"""
High-frequency data collector with 1-minute lag compensation.

This collector:
1. Uses Chainlink frontend scraping for ~0.5s resolution data
2. Accounts for 1-minute lag in Chainlink data
3. Collects Polymarket at ~0.5s intervals
4. Properly aligns data for lag analysis

Key concept: If we want to analyze Polymarket data from 7:15-7:30,
we need to collect Chainlink data from 7:16-7:31 (1 minute later)
to get the same underlying price data that Polymarket saw.
"""

import time
import signal
import asyncio
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chainlink.frontend_collector import ChainlinkFrontendCollectorSync
from src.polymarket.collector import PolymarketCollector, PolymarketStorage
from src.chainlink.collector import ChainlinkStorage
from src.utils.logging import setup_logger
from config.settings import CHAINLINK, POLYMARKET, STORAGE


class HighFreqCollector:
    """
    High-frequency collector with lag compensation.
    
    Collection strategy:
    - Polymarket: Collect at 1.0s intervals (1 Hz)
    - Chainlink: Collect at 1.0s intervals (1 Hz) but start 1 minute later
    - This ensures we capture the same underlying price data
    """
    
    def __init__(
        self,
        token_id_up: Optional[str] = None,
        token_id_down: Optional[str] = None,
        market_name: Optional[str] = None,
        chainlink_interval: float = 1.0,
        polymarket_interval: float = 1.0,
        chainlink_lag_minutes: int = 1,
        log_level: int = 20,
    ):
        """
        Initialize high-frequency collector.
        
        Args:
            token_id_up: Polymarket UP token ID (auto-discovered if None)
            token_id_down: Polymarket DOWN token ID (auto-discovered if None)
            market_name: Market name (auto-discovered if None)
            chainlink_interval: Chainlink collection interval (seconds, default 0.5)
            polymarket_interval: Polymarket collection interval (seconds, default 0.5)
            chainlink_lag_minutes: Lag in Chainlink frontend data (default 1 minute)
            log_level: Logging level
        """
        self.chainlink_interval = chainlink_interval
        self.polymarket_interval = polymarket_interval
        self.chainlink_lag_seconds = chainlink_lag_minutes * 60
        
        self.logger = setup_logger("high_freq_collector", level=log_level)
        
        # Initialize collectors
        self.chainlink = ChainlinkFrontendCollectorSync(log_level=log_level)
        self.polymarket = PolymarketCollector(log_level=log_level)
        
        # Initialize storage
        self.chainlink_storage = ChainlinkStorage()
        self.polymarket_storage = PolymarketStorage()
        
        # Market state
        self.token_id_up = token_id_up
        self.token_id_down = token_id_down
        self.market_name = market_name
        self.market_end_time = None
        
        # Auto-discover market if not provided
        if not self.token_id_up or not self.token_id_down:
            self._find_active_market()
        
        # Collection state
        self.running = False
        self.stats = {
            "chainlink_collections": 0,
            "polymarket_collections": 0,
            "chainlink_errors": 0,
            "polymarket_errors": 0,
            "start_time": None,
            "end_time": None,
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _find_active_market(self):
        """Auto-discover active BTC 15-min market."""
        try:
            market_info = self.polymarket.find_active_btc_15min_market()
            if market_info:
                self.token_id_up = market_info.get("token_id_up")
                self.token_id_down = market_info.get("token_id_down")
                self.market_name = market_info.get("question") or market_info.get("market_slug") or "Unknown"
                
                # Parse market end time
                end_date_str = market_info.get("end_date")
                if end_date_str:
                    try:
                        self.market_end_time = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                    except:
                        self.market_end_time = None
                else:
                    self.market_end_time = None
                
                self.logger.info(f"Found active market: {self.market_name}")
                if self.market_end_time:
                    self.logger.info(f"Market ends at: {self.market_end_time.isoformat()}")
                return True
            else:
                self.logger.warning("No active market found")
                return False
        except Exception as e:
            self.logger.error(f"Error finding active market: {e}")
            return False
    
    def _switch_to_new_market(self):
        """Switch to a new market when current one ends."""
        self.logger.info("=" * 50)
        self.logger.info("MARKET ENDED - Switching to new market...")
        self.logger.info("=" * 50)
        
        # Clear live data for fresh start
        self._clear_live_csv()
        
        # Wait a bit for new market to be available
        max_retries = 30  # Wait up to 30 seconds
        for i in range(max_retries):
            if self._find_active_market():
                # Verify it's a NEW market (end time should be in future)
                if self.market_end_time and self.market_end_time > datetime.now(timezone.utc):
                    self.logger.info(f"Switched to new market: {self.market_name}")
                    return True
            time.sleep(1)
            if i % 5 == 0:
                self.logger.info(f"Waiting for new market... ({i}s)")
        
        self.logger.error("Failed to find new market after 30 seconds")
        return False
    
    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self.logger.info("Interrupt received, stopping collection...")
        self.running = False
    
    def _clear_live_csv(self):
        """Clear live CSV files when switching markets."""
        live_dir = Path("data/live")
        pm_file = live_dir / "polymarket_live.csv"
        cl_file = live_dir / "chainlink_live.csv"
        if pm_file.exists():
            pm_file.unlink()
        if cl_file.exists():
            cl_file.unlink()
        self.logger.info("Cleared live data for new market")
    
    def _save_live_csv(self, pm_data: Optional[Dict], cl_data: Optional[Dict]):
        """
        Save data to live CSV files for real-time dashboard updates.
        CSV is simpler and has no read/write conflicts like parquet.
        """
        live_dir = Path("data/live")
        live_dir.mkdir(parents=True, exist_ok=True)
        
        if pm_data:
            pm_file = live_dir / "polymarket_live.csv"
            row = {
                "collected_at": pm_data["collected_at"].isoformat(),
                "market_name": self.market_name or "Unknown",
                "up_mid": pm_data.get("up_mid") if pd.notna(pm_data.get("up_mid")) else 0,
                "up_best_bid": pm_data.get("up_best_bid") if pd.notna(pm_data.get("up_best_bid")) else 0,
                "up_best_ask": pm_data.get("up_best_ask") if pd.notna(pm_data.get("up_best_ask")) else 0,
                "down_mid": pm_data.get("down_mid") if pd.notna(pm_data.get("down_mid")) else 0,
                "down_best_bid": pm_data.get("down_best_bid") if pd.notna(pm_data.get("down_best_bid")) else 0,
                "down_best_ask": pm_data.get("down_best_ask") if pd.notna(pm_data.get("down_best_ask")) else 0,
            }
            df = pd.DataFrame([row])
            # Append to CSV (create header only if file doesn't exist)
            df.to_csv(pm_file, mode='a', header=not pm_file.exists(), index=False)
        
        if cl_data:
            cl_file = live_dir / "chainlink_live.csv"
            row = {
                "collected_at": cl_data["collected_at"].isoformat() if isinstance(cl_data.get("collected_at"), datetime) else str(cl_data.get("collected_at", "")),
                "mid": cl_data.get("mid", 0),
                "bid": cl_data.get("bid", 0),
                "ask": cl_data.get("ask", 0),
            }
            df = pd.DataFrame([row])
            # Append to CSV (create header only if file doesn't exist)
            df.to_csv(cl_file, mode='a', header=not cl_file.exists(), index=False)
    
    def collect_chainlink(self, save: bool = False) -> Optional[Dict]:
        """Collect single Chainlink data point."""
        try:
            data = self.chainlink.get_latest_price()
            if data:
                self.stats["chainlink_collections"] += 1
                
                # Only save if explicitly requested (we'll save with synced timestamp later)
                if save:
                    df = pd.DataFrame([data])
                    self.chainlink_storage.save(df)
                
                return data
            else:
                self.stats["chainlink_errors"] += 1
                return None
                
        except Exception as e:
            self.logger.error(f"Chainlink collection error: {e}")
            self.stats["chainlink_errors"] += 1
            return None
    
    def collect_polymarket(self, save: bool = False) -> Optional[Dict]:
        """Collect single Polymarket data point."""
        if not self.token_id_up or not self.token_id_down:
            return None
        
        try:
            data = self.polymarket.get_market_data(self.token_id_up, self.token_id_down)
            if data:
                self.stats["polymarket_collections"] += 1
                
                # Only save if explicitly requested (we'll save with synced timestamp later)
                if save:
                    self.polymarket_storage.save_market_data(data)
                
                return data
            else:
                self.stats["polymarket_errors"] += 1
                return None
                
        except Exception as e:
            self.logger.error(f"Polymarket collection error: {e}")
            self.stats["polymarket_errors"] += 1
            return None
    
    def start(
        self,
        duration_seconds: float,
        polymarket_start_time: Optional[datetime] = None,
        live_save: bool = False,
    ):
        """
        Start high-frequency collection.
        
        Args:
            duration_seconds: Total collection duration (seconds)
            polymarket_start_time: When to start Polymarket collection (default: now)
                                  Chainlink will start 1 minute later
            live_save: If True, save data incrementally for live dashboard viewing
        """
        if polymarket_start_time is None:
            polymarket_start_time = datetime.now(timezone.utc)
        
        chainlink_start_time = polymarket_start_time + timedelta(seconds=self.chainlink_lag_seconds)
        
        # Start Chainlink page loading earlier to account for load time
        # Load page 10 seconds before actual collection starts
        chainlink_page_load_time = chainlink_start_time - timedelta(seconds=10)
        
        self.logger.info("=" * 80)
        self.logger.info("HIGH-FREQUENCY DATA COLLECTION")
        self.logger.info("=" * 80)
        self.logger.info(f"Polymarket start: {polymarket_start_time.isoformat()}")
        self.logger.info(f"Chainlink page load: {chainlink_page_load_time.isoformat()} (10s early)")
        self.logger.info(f"Chainlink start:  {chainlink_start_time.isoformat()} (1 min lag)")
        self.logger.info(f"Duration: {duration_seconds}s")
        self.logger.info(f"Intervals: CL={self.chainlink_interval}s, PM={self.polymarket_interval}s")
        self.logger.info("=" * 80)
        
        self.running = True
        self.stats["start_time"] = datetime.now(timezone.utc)
        
        # Calculate end times
        polymarket_end_time = polymarket_start_time + timedelta(seconds=duration_seconds)
        chainlink_end_time = chainlink_start_time + timedelta(seconds=duration_seconds)
        
        # Wait until Polymarket start time
        now = datetime.now(timezone.utc)
        if polymarket_start_time > now:
            wait_seconds = (polymarket_start_time - now).total_seconds()
            self.logger.info(f"Waiting {wait_seconds:.1f}s until Polymarket start time...")
            time.sleep(wait_seconds)
        
        # Initialize collection timers
        # Use precise second-based timing to ensure exactly 1 point per second
        now_ts = time.time()
        polymarket_start_second = int(polymarket_start_time.timestamp())
        # Set to start at the next whole second after start time
        if now_ts < polymarket_start_second:
            polymarket_next_second = polymarket_start_second
        else:
            polymarket_next_second = int(now_ts) + 1
        chainlink_next_second = 0  # Will be set when Chainlink starts
        
        # Pre-load Chainlink page in background (don't block Polymarket collection)
        # Start a thread or just try to load it early if we have time
        now = datetime.now(timezone.utc)
        if chainlink_page_load_time <= now:
            # Time to load page now
            self.logger.info("Pre-loading Chainlink page...")
            try:
                self.chainlink.get_latest_price()
                self.logger.info("Chainlink page pre-loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to pre-load Chainlink page: {e}")
        
        # Start collection earlier to ensure we get enough data
        # Collect extra data, then keep only the exact amount needed
        buffer_seconds = 30  # Start 30 seconds early to ensure we get enough
        actual_polymarket_start = polymarket_start_time - timedelta(seconds=buffer_seconds)
        actual_chainlink_start = chainlink_start_time - timedelta(seconds=buffer_seconds)
        
        # Wait until actual start time (with buffer)
        now = datetime.now(timezone.utc)
        if actual_polymarket_start > now:
            wait_seconds = (actual_polymarket_start - now).total_seconds()
            self.logger.info(f"Waiting {wait_seconds:.1f}s until collection start (with buffer)...")
            time.sleep(wait_seconds)
        
        self.logger.info("Starting collection (with buffer to ensure enough data)...")
        
        # Store all collected data, then filter to exact amount needed
        polymarket_collected = []
        chainlink_collected = []
        
        # Initialize timers - collect at 1 second intervals
        polymarket_last_collection = time.time()
        chainlink_last_collection = 0
        
        try:
            last_market_check = time.time()
            
            while self.running:
                now_utc = datetime.now(timezone.utc)
                now_ts = time.time()
                
                # Check for market switch every 5 seconds
                if now_ts - last_market_check >= 5.0:
                    last_market_check = now_ts
                    # Check if current market has ended
                    if self.market_end_time and now_utc >= self.market_end_time:
                        self.logger.info(f"Market ended at {self.market_end_time.isoformat()}")
                        if self._switch_to_new_market():
                            # Update end times for new collection window
                            # Reset polymarket collected data for new market
                            polymarket_collected = []
                            chainlink_collected = []
                            self.logger.info("Continuing collection with new market...")
                        else:
                            self.logger.error("Failed to switch to new market, stopping")
                            break
                
                # Check if we've exceeded the extended duration (with buffer)
                polymarket_done = now_utc >= polymarket_end_time + timedelta(seconds=buffer_seconds)
                chainlink_done = now_utc >= chainlink_end_time + timedelta(seconds=buffer_seconds) if now_utc >= actual_chainlink_start else False
                
                if polymarket_done and chainlink_done:
                    self.logger.info("Collection complete (with buffer)")
                    break
                
                # Collect Polymarket at 1-second intervals (starting early with buffer)
                if now_utc >= actual_polymarket_start and now_ts - polymarket_last_collection >= 1.0:
                    pm_data = self.collect_polymarket(save=False)
                    if pm_data:
                        # Keep original timestamp (not rounded) for precise matching
                        polymarket_collected.append(pm_data)
                        # Enhanced terminal logging with full data
                        pm_count = len(polymarket_collected)
                        ts = pm_data['collected_at'].strftime('%H:%M:%S')
                        up_mid = pm_data.get('up_mid', 0) or 0
                        down_mid = pm_data.get('down_mid', 0) or 0
                        up_ask = pm_data.get('up_best_ask', 0) or 0
                        down_bid = pm_data.get('down_best_bid', 0) or 0
                        print(f"[PM #{pm_count:3d}] {ts} | UP: mid={up_mid:.3f} ask={up_ask:.3f} | DOWN: mid={down_mid:.3f} bid={down_bid:.3f}")
                        # Save incrementally for live dashboard
                        if live_save:
                            self.polymarket_storage.save_market_data(pm_data)
                            self._save_live_csv(pm_data, None)
                    polymarket_last_collection = now_ts
                
                # Collect Chainlink at 1-second intervals (starting early with buffer)
                if now_utc >= actual_chainlink_start:
                    if chainlink_last_collection == 0:
                        chainlink_last_collection = now_ts
                        self.logger.info("Starting Chainlink collection (1 min lag compensated)")
                    
                    if now_ts - chainlink_last_collection >= 1.0:
                        cl_data = self.collect_chainlink(save=False)
                        if cl_data:
                            # Keep original timestamp (not rounded) for precise matching
                            chainlink_collected.append(cl_data)
                            # Enhanced terminal logging with full data
                            cl_count = len(chainlink_collected)
                            ts = cl_data['collected_at'].strftime('%H:%M:%S') if isinstance(cl_data.get('collected_at'), datetime) else '--:--:--'
                            mid = cl_data.get('mid', 0) or 0
                            bid = cl_data.get('bid', 0) or 0
                            ask = cl_data.get('ask', 0) or 0
                            print(f"[CL #{cl_count:3d}] {ts} | price=${mid:.2f} bid=${bid:.2f} ask=${ask:.2f}")
                            # Save incrementally for live dashboard
                            if live_save:
                                df = pd.DataFrame([cl_data])
                                self.chainlink_storage.save(df)
                                self._save_live_csv(None, cl_data)
                        chainlink_last_collection = now_ts
                
                # Small sleep to avoid busy waiting
                time.sleep(0.01)
            
            # Filter to exact amount needed (keep only data within the requested time window)
            self.logger.info(f"Collected {len(polymarket_collected)} Polymarket points, filtering to exact window...")
            self.logger.info(f"Collected {len(chainlink_collected)} Chainlink points, filtering to exact window...")
            
            # Filter Polymarket data to exact window
            polymarket_filtered = [
                d for d in polymarket_collected
                if polymarket_start_time <= d['collected_at'] < polymarket_end_time
            ]
            
            # Filter Chainlink data to exact window
            chainlink_filtered = [
                d for d in chainlink_collected
                if chainlink_start_time <= d['collected_at'] < chainlink_end_time
            ]
            
            self.logger.info(f"Filtered to {len(polymarket_filtered)} Polymarket points in exact window")
            self.logger.info(f"Filtered to {len(chainlink_filtered)} Chainlink points in exact window")
            
            # Save filtered data with original timestamps (only if not already saved live)
            if not live_save:
                for pm_data in polymarket_filtered:
                    self.polymarket_storage.save_market_data(pm_data)
                
                for cl_data in chainlink_filtered:
                    df = pd.DataFrame([cl_data])
                    self.chainlink_storage.save(df)
            
            self.stats["polymarket_collections"] = len(polymarket_filtered)
            self.stats["chainlink_collections"] = len(chainlink_filtered)
        
        except KeyboardInterrupt:
            self.logger.info("Collection interrupted by user")
        finally:
            self.running = False
            self.stats["end_time"] = datetime.now(timezone.utc)
            self._print_final_stats()
            
            # Close browser
            self.chainlink.close()
    
    def _print_final_stats(self):
        """Print final collection statistics."""
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        
        self.logger.info("=" * 80)
        self.logger.info("COLLECTION COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Duration: {duration:.1f}s")
        self.logger.info(f"Chainlink collections: {self.stats['chainlink_collections']}")
        self.logger.info(f"Chainlink errors: {self.stats['chainlink_errors']}")
        self.logger.info(f"Polymarket collections: {self.stats['polymarket_collections']}")
        self.logger.info(f"Polymarket errors: {self.stats['polymarket_errors']}")
        self.logger.info("=" * 80)

