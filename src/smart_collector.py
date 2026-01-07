"""
Smart Data Collection Orchestrator with Intelligent Market Switching.

Features:
- Automatically detects when market closes
- Switches to next active market seamlessly
- Continues collecting for remaining duration
- Tracks separate statistics per market
"""

import time
import signal
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chainlink.collector import ChainlinkCollector, ChainlinkStorage
from src.polymarket.collector import PolymarketCollector, PolymarketStorage
from src.utils.logging import setup_logger
from config.settings import CHAINLINK, POLYMARKET, MARKET, STORAGE, COLLECTION


class SmartCollector:
    """
    Intelligent data collector that handles market transitions.
    
    Automatically:
    - Detects market closure (via API errors or end time)
    - Switches to the next active market
    - Continues collecting for the remaining duration
    - Maintains Chainlink collection throughout (oracle never stops)
    """
    
    def __init__(
        self,
        chainlink_interval: float = 1.0,
        polymarket_interval: float = 1.0,
        log_level: int = 20,
        max_consecutive_errors: int = 5,
    ):
        """
        Initialize the smart collector.
        
        Args:
            chainlink_interval: Seconds between Chainlink collections
            polymarket_interval: Seconds between Polymarket collections
            log_level: Logging level
            max_consecutive_errors: Errors before assuming market closed
        """
        self.chainlink_interval = chainlink_interval
        self.polymarket_interval = polymarket_interval
        self.max_consecutive_errors = max_consecutive_errors
        
        # Initialize logger
        self.logger = setup_logger("smart_collector", level=log_level)
        
        # Initialize collectors
        self.chainlink = ChainlinkCollector(log_level=log_level)
        self.polymarket = PolymarketCollector(log_level=log_level)
        
        # Initialize storage
        self.chainlink_storage = ChainlinkStorage()
        self.polymarket_storage = PolymarketStorage()
        
        # Current market state
        self.current_market = None
        self.token_id_up = None
        self.token_id_down = None
        self.market_end_time = None
        
        # Collection state
        self.running = False
        self.consecutive_errors = 0
        
        # Statistics per market
        self.market_stats = {}
        self.total_stats = {
            "chainlink_collections": 0,
            "chainlink_errors": 0,
            "polymarket_collections": 0,
            "polymarket_errors": 0,
            "markets_collected": 0,
            "start_time": None,
        }
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self.logger.info("Interrupt received, stopping collection...")
        self.running = False
    
    def _find_and_set_active_market(self) -> bool:
        """
        Find and configure the currently active market.
        
        Returns:
            True if active market found, False otherwise
        """
        market = self.polymarket.find_active_btc_15min_market()
        
        if not market:
            self.logger.warning("No active BTC 15-min market found")
            return False
        
        # Parse market end time
        try:
            end_date_str = market.get("end_date", "")
            self.market_end_time = datetime.fromisoformat(
                end_date_str.replace('Z', '+00:00')
            )
        except (ValueError, AttributeError):
            # Estimate from slug timestamp + 15 minutes
            try:
                slug = market.get("slug", "")
                ts = int(slug.split("-")[-1])
                self.market_end_time = datetime.fromtimestamp(ts + 900, tz=timezone.utc)
            except:
                self.market_end_time = None
        
        self.current_market = market
        self.token_id_up = market.get("token_id_up")
        self.token_id_down = market.get("token_id_down")
        self.consecutive_errors = 0
        
        # Initialize stats for this market
        market_id = market.get("slug", "unknown")
        if market_id not in self.market_stats:
            self.market_stats[market_id] = {
                "question": market.get("question"),
                "start_time": datetime.now(timezone.utc),
                "end_time": self.market_end_time,
                "collections": 0,
                "errors": 0,
                "last_up_mid": None,
                "last_down_mid": None,
            }
        
        self.logger.info("=" * 60)
        self.logger.info(f"Active market: {market.get('question')}")
        self.logger.info(f"  End time: {self.market_end_time}")
        self.logger.info(f"  Token UP: {self.token_id_up[:40]}...")
        self.logger.info(f"  Token DOWN: {self.token_id_down[:40]}...")
        self.logger.info("=" * 60)
        
        return True
    
    def _check_market_expired(self) -> bool:
        """Check if current market has expired."""
        if not self.market_end_time:
            return False
        
        now = datetime.now(timezone.utc)
        
        # Consider market expired 30 seconds after end time
        # (to allow for settlement delay)
        return now > self.market_end_time + timedelta(seconds=30)
    
    def _should_switch_market(self) -> bool:
        """Determine if we should switch to a new market."""
        # Check if end time passed
        if self._check_market_expired():
            self.logger.info("Market end time reached")
            return True
        
        # Check consecutive errors (market likely closed)
        if self.consecutive_errors >= self.max_consecutive_errors:
            self.logger.info(f"Market appears closed ({self.consecutive_errors} consecutive errors)")
            return True
        
        return False
    
    def collect_chainlink(self) -> Optional[Dict]:
        """Collect single Chainlink data point."""
        try:
            data = self.chainlink.get_latest_price()
            if data:
                self.total_stats["chainlink_collections"] += 1
                
                # Save to storage
                df = pd.DataFrame([data])
                self.chainlink_storage.save(df)
                
                return data
            else:
                self.total_stats["chainlink_errors"] += 1
                return None
                
        except Exception as e:
            self.logger.error(f"Chainlink collection error: {e}")
            self.total_stats["chainlink_errors"] += 1
            return None
    
    def collect_polymarket(self) -> Optional[Dict]:
        """Collect single Polymarket data point."""
        if not self.token_id_up or not self.token_id_down:
            return None
        
        try:
            data = self.polymarket.get_market_data(self.token_id_up, self.token_id_down)
            
            if data and data.get("up_mid") is not None:
                self.total_stats["polymarket_collections"] += 1
                self.consecutive_errors = 0  # Reset error counter
                
                # Update market-specific stats
                if self.current_market:
                    market_id = self.current_market.get("slug", "unknown")
                    if market_id in self.market_stats:
                        self.market_stats[market_id]["collections"] += 1
                        self.market_stats[market_id]["last_up_mid"] = data.get("up_mid")
                        self.market_stats[market_id]["last_down_mid"] = data.get("down_mid")
                
                # Save to storage
                self.polymarket_storage.save_market_data(data)
                
                return data
            else:
                self.consecutive_errors += 1
                self.total_stats["polymarket_errors"] += 1
                return None
                
        except Exception as e:
            self.consecutive_errors += 1
            self.total_stats["polymarket_errors"] += 1
            return None
    
    def start(self, duration_seconds: Optional[int] = None):
        """
        Start smart data collection with automatic market switching.
        
        Args:
            duration_seconds: Total duration to collect (across all markets)
        """
        self.running = True
        self.total_stats["start_time"] = datetime.now(timezone.utc)
        
        self.logger.info("=" * 60)
        self.logger.info("SMART DATA COLLECTION")
        self.logger.info("=" * 60)
        self.logger.info(f"Chainlink interval: {self.chainlink_interval}s")
        self.logger.info(f"Polymarket interval: {self.polymarket_interval}s")
        if duration_seconds:
            self.logger.info(f"Total duration: {duration_seconds}s")
        else:
            self.logger.info("Duration: Until interrupted (Ctrl+C)")
        self.logger.info("=" * 60)
        
        # Find initial market
        if not self._find_and_set_active_market():
            self.logger.error("Could not find active market to start")
            return
        
        self.total_stats["markets_collected"] = 1
        
        end_time = None
        if duration_seconds:
            end_time = time.time() + duration_seconds
        
        last_chainlink = 0
        last_polymarket = 0
        last_status = 0
        last_market_check = 0
        
        try:
            while self.running:
                current_time = time.time()
                
                # Check if total duration exceeded
                if end_time and current_time >= end_time:
                    self.logger.info("Collection duration reached")
                    break
                
                # Check if we should switch markets (every 5 seconds)
                if current_time - last_market_check >= 5:
                    if self._should_switch_market():
                        self.logger.info("Switching to next market...")
                        
                        # Brief pause to let market settle
                        time.sleep(2)
                        
                        if self._find_and_set_active_market():
                            self.total_stats["markets_collected"] += 1
                        else:
                            self.logger.warning("No new market found, continuing with Chainlink only")
                            # Reset tokens so we don't keep hitting closed market
                            self.token_id_up = None
                            self.token_id_down = None
                    
                    last_market_check = current_time
                
                # Collect Chainlink data (always continues)
                if current_time - last_chainlink >= self.chainlink_interval:
                    self.collect_chainlink()
                    last_chainlink = current_time
                
                # Collect Polymarket data (only if we have valid tokens)
                if current_time - last_polymarket >= self.polymarket_interval:
                    if self.token_id_up and self.token_id_down:
                        self.collect_polymarket()
                    last_polymarket = current_time
                
                # Print status every 10 seconds
                if current_time - last_status >= 10:
                    self._print_status()
                    last_status = current_time
                
                time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Collection error: {e}")
        
        finally:
            self._print_final_stats()
    
    def _print_status(self):
        """Print current collection status."""
        # Get latest data
        cl_count = self.total_stats["chainlink_collections"]
        pm_count = self.total_stats["polymarket_collections"]
        
        market_name = "None"
        up_mid = None
        down_mid = None
        
        if self.current_market:
            market_id = self.current_market.get("slug", "unknown")
            market_name = market_id
            if market_id in self.market_stats:
                up_mid = self.market_stats[market_id].get("last_up_mid")
                down_mid = self.market_stats[market_id].get("last_down_mid")
        
        status = f"Market: {market_name}"
        status += f" | UP: {up_mid:.3f}" if up_mid else " | UP: --"
        status += f" | DOWN: {down_mid:.3f}" if down_mid else " | DOWN: --"
        status += f" | CL={cl_count}, PM={pm_count}"
        
        self.logger.info(status)
    
    def _print_final_stats(self):
        """Print final collection statistics."""
        duration = datetime.now(timezone.utc) - self.total_stats["start_time"]
        
        self.logger.info("=" * 60)
        self.logger.info("COLLECTION COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Total duration: {duration}")
        self.logger.info(f"Markets collected: {self.total_stats['markets_collected']}")
        self.logger.info(f"Chainlink collections: {self.total_stats['chainlink_collections']}")
        self.logger.info(f"Chainlink errors: {self.total_stats['chainlink_errors']}")
        self.logger.info(f"Polymarket collections: {self.total_stats['polymarket_collections']}")
        self.logger.info(f"Polymarket errors: {self.total_stats['polymarket_errors']}")
        self.logger.info("=" * 60)
        
        # Per-market breakdown
        if self.market_stats:
            self.logger.info("Per-market breakdown:")
            for market_id, stats in self.market_stats.items():
                self.logger.info(f"  {market_id}:")
                self.logger.info(f"    Collections: {stats['collections']}")
                self.logger.info(f"    Final UP: {stats.get('last_up_mid', 'N/A')}")
                self.logger.info(f"    Final DOWN: {stats.get('last_down_mid', 'N/A')}")
        
        self.logger.info("=" * 60)


def main():
    """CLI entry point for smart collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart data collection with market switching")
    parser.add_argument("--duration", type=int, help="Total collection duration in seconds")
    parser.add_argument("--interval", type=float, default=1.0, help="Collection interval")
    args = parser.parse_args()
    
    print("=" * 70)
    print("SMART DATA COLLECTION")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)
    
    collector = SmartCollector(
        chainlink_interval=args.interval,
        polymarket_interval=args.interval,
    )
    
    collector.start(duration_seconds=args.duration)
    
    print("\nData saved to:")
    print("  Chainlink: data/raw/chainlink/")
    print("  Polymarket: data/raw/polymarket/")


if __name__ == "__main__":
    main()

