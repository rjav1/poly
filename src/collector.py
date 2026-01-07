"""
Unified Data Collection Orchestrator.

Coordinates data collection from Chainlink and Polymarket,
ensures time synchronization, and handles storage.
"""

import time
import signal
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict
from pathlib import Path
import pandas as pd
import threading

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chainlink.collector import ChainlinkCollector, ChainlinkStorage
from src.polymarket.collector import PolymarketCollector, PolymarketStorage
from src.utils.logging import setup_logger
from config.settings import CHAINLINK, POLYMARKET, MARKET, STORAGE, COLLECTION


class DataCollector:
    """
    Unified data collector for prediction market research.
    
    Coordinates:
    - Chainlink price data collection (BTC/USD)
    - Polymarket market data collection (orderbooks, midpoints)
    - Time synchronization between sources
    - Persistent storage
    
    Usage:
        collector = DataCollector(token_id_up="...", token_id_down="...")
        collector.start()  # Runs until interrupted or market ends
    """
    
    def __init__(
        self,
        token_id_up: str,
        token_id_down: str,
        market_name: str = "BTC 15-min Up/Down",
        chainlink_interval: float = 1.0,
        polymarket_interval: float = 1.0,
        log_level: int = 20,
    ):
        """
        Initialize the data collector.
        
        Args:
            token_id_up: Polymarket token ID for UP outcome
            token_id_down: Polymarket token ID for DOWN outcome
            market_name: Human-readable market name
            chainlink_interval: Seconds between Chainlink collections
            polymarket_interval: Seconds between Polymarket collections
            log_level: Logging level (default INFO=20)
        """
        self.token_id_up = token_id_up
        self.token_id_down = token_id_down
        self.market_name = market_name
        self.chainlink_interval = chainlink_interval
        self.polymarket_interval = polymarket_interval
        
        # Initialize logger
        self.logger = setup_logger("collector", level=log_level)
        self.logger.info(f"Initializing DataCollector for market: {market_name}")
        
        # Initialize collectors
        self.chainlink = ChainlinkCollector(log_level=log_level)
        self.polymarket = PolymarketCollector(log_level=log_level)
        
        # Initialize storage
        self.chainlink_storage = ChainlinkStorage()
        self.polymarket_storage = PolymarketStorage()
        
        # Collection state
        self.running = False
        self.stats = {
            "chainlink_collections": 0,
            "chainlink_errors": 0,
            "polymarket_collections": 0,
            "polymarket_errors": 0,
            "start_time": None,
            "last_chainlink_price": None,
            "last_polymarket_up_mid": None,
            "last_polymarket_down_mid": None,
        }
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self.logger.info("Interrupt received, stopping collection...")
        self.running = False
    
    def collect_chainlink(self) -> Optional[Dict]:
        """Collect single Chainlink data point."""
        try:
            data = self.chainlink.get_latest_price()
            if data:
                self.stats["chainlink_collections"] += 1
                self.stats["last_chainlink_price"] = data.get("price")
                
                # Save to storage
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
    
    def collect_polymarket(self) -> Optional[Dict]:
        """Collect single Polymarket data point."""
        try:
            data = self.polymarket.get_market_data(self.token_id_up, self.token_id_down)
            if data:
                self.stats["polymarket_collections"] += 1
                self.stats["last_polymarket_up_mid"] = data.get("up_mid")
                self.stats["last_polymarket_down_mid"] = data.get("down_mid")
                
                # Save to storage
                self.polymarket_storage.save_market_data(data)
                
                return data
            else:
                self.stats["polymarket_errors"] += 1
                return None
                
        except Exception as e:
            self.logger.error(f"Polymarket collection error: {e}")
            self.stats["polymarket_errors"] += 1
            return None
    
    def collect_once(self) -> Dict:
        """
        Collect one synchronized data point from both sources.
        
        Returns:
            Dictionary with all collected data
        """
        timestamp = datetime.now(timezone.utc)
        
        chainlink_data = self.collect_chainlink()
        polymarket_data = self.collect_polymarket()
        
        return {
            "timestamp": timestamp,
            "chainlink": chainlink_data,
            "polymarket": polymarket_data,
        }
    
    def start(self, duration_seconds: Optional[int] = None):
        """
        Start continuous data collection.
        
        Args:
            duration_seconds: How long to run (None = until interrupted)
        """
        self.running = True
        self.stats["start_time"] = datetime.now(timezone.utc)
        
        self.logger.info("=" * 60)
        self.logger.info("Starting data collection")
        self.logger.info(f"  Market: {self.market_name}")
        self.logger.info(f"  Chainlink interval: {self.chainlink_interval}s")
        self.logger.info(f"  Polymarket interval: {self.polymarket_interval}s")
        if duration_seconds:
            self.logger.info(f"  Duration: {duration_seconds}s")
        else:
            self.logger.info("  Duration: Until interrupted (Ctrl+C)")
        self.logger.info("=" * 60)
        
        end_time = None
        if duration_seconds:
            end_time = time.time() + duration_seconds
        
        last_chainlink = 0
        last_polymarket = 0
        last_status = 0
        
        try:
            while self.running:
                current_time = time.time()
                
                # Check if duration exceeded
                if end_time and current_time >= end_time:
                    self.logger.info("Collection duration reached")
                    break
                
                # Collect Chainlink data
                if current_time - last_chainlink >= self.chainlink_interval:
                    self.collect_chainlink()
                    last_chainlink = current_time
                
                # Collect Polymarket data
                if current_time - last_polymarket >= self.polymarket_interval:
                    self.collect_polymarket()
                    last_polymarket = current_time
                
                # Print status every 10 seconds
                if current_time - last_status >= 10:
                    self._print_status()
                    last_status = current_time
                
                # Small sleep to prevent busy-waiting
                time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Collection error: {e}")
        
        finally:
            self._print_final_stats()
    
    def _print_status(self):
        """Print current collection status."""
        cl_price = self.stats.get("last_chainlink_price")
        up_mid = self.stats.get("last_polymarket_up_mid")
        down_mid = self.stats.get("last_polymarket_down_mid")
        
        status = f"CL: ${cl_price:,.2f}" if cl_price else "CL: --"
        status += f" | UP: {up_mid:.3f}" if up_mid else " | UP: --"
        status += f" | DOWN: {down_mid:.3f}" if down_mid else " | DOWN: --"
        status += f" | Collected: CL={self.stats['chainlink_collections']}, PM={self.stats['polymarket_collections']}"
        
        self.logger.info(status)
    
    def _print_final_stats(self):
        """Print final collection statistics."""
        duration = datetime.now(timezone.utc) - self.stats["start_time"]
        
        self.logger.info("=" * 60)
        self.logger.info("Collection Complete")
        self.logger.info("=" * 60)
        self.logger.info(f"  Duration: {duration}")
        self.logger.info(f"  Chainlink collections: {self.stats['chainlink_collections']}")
        self.logger.info(f"  Chainlink errors: {self.stats['chainlink_errors']}")
        self.logger.info(f"  Polymarket collections: {self.stats['polymarket_collections']}")
        self.logger.info(f"  Polymarket errors: {self.stats['polymarket_errors']}")
        self.logger.info("=" * 60)
    
    def get_collected_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get all collected data from storage.
        
        Returns:
            Dictionary with 'chainlink' and 'polymarket' DataFrames
        """
        return {
            "chainlink": self.chainlink_storage.load(),
            "polymarket": self.polymarket_storage.load_market_data(),
        }


def discover_market_tokens(market_slug: str) -> Optional[Dict[str, str]]:
    """
    Discover token IDs for a market by searching the CLOB API.
    
    Args:
        market_slug: Market slug from URL (e.g., "btc-updown-15m-1767482100")
        
    Returns:
        Dictionary with token_id_up and token_id_down, or None if not found
    """
    logger = setup_logger("market_discovery")
    logger.info(f"Searching for market: {market_slug}")
    
    collector = PolymarketCollector()
    markets = collector.search_markets("btc")
    
    if len(markets) == 0:
        logger.warning("No BTC markets found")
        return None
    
    # Search for matching slug
    matching = markets[markets["market_slug"] == market_slug]
    
    if len(matching) > 0:
        market = matching.iloc[0]
        result = {
            "token_id_up": market["token_id_up"],
            "token_id_down": market["token_id_down"],
            "question": market["question"],
        }
        logger.info(f"Found market: {result['question']}")
        logger.info(f"  UP token: {result['token_id_up'][:30] if result['token_id_up'] else 'None'}...")
        logger.info(f"  DOWN token: {result['token_id_down'][:30] if result['token_id_down'] else 'None'}...")
        return result
    
    # If not found by slug, show available BTC markets
    logger.warning(f"Market '{market_slug}' not found")
    logger.info("Available BTC markets:")
    for _, m in markets.head(10).iterrows():
        logger.info(f"  {m['market_slug']}: {m['question']}")
    
    return None

