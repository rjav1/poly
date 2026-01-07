"""
Production-ready Polymarket CLOB collector.

Uses the official CLOB API endpoints.
API Documentation: https://docs.polymarket.com/quickstart/overview

Supports multi-asset collection: BTC, ETH, SOL, XRP.
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Callable
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logging import setup_logger
from config.settings import POLYMARKET, STORAGE, ASSETS, get_asset_config, SUPPORTED_ASSETS


class PolymarketCollector:
    """
    Collects market data from Polymarket's CLOB API.
    
    API Endpoints:
    - /markets: List all markets
    - /midpoint?token_id={id}: Get current midpoint price
    - /book?token_id={id}: Get full orderbook
    
    Reference: https://docs.polymarket.com/quickstart/overview
    """
    
    def __init__(self, log_level: int = 20):
        """
        Initialize Polymarket collector.
        
        Args:
            log_level: Logging level (default INFO=20)
        """
        self.base_url = POLYMARKET.base_url
        self.timeout = POLYMARKET.request_timeout
        self.orderbook_depth = POLYMARKET.orderbook_depth
        
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        
        self.logger = setup_logger("polymarket", level=log_level)
        self.logger.info(f"PolymarketCollector initialized with base_url: {self.base_url}")
    
    def get_midpoint(self, token_id: str) -> Optional[float]:
        """
        Get current midpoint price for a token.
        
        Args:
            token_id: Token ID (UP or DOWN)
            
        Returns:
            Midpoint price as float (0.0 to 1.0) or None if failed
        """
        url = f"{self.base_url}/midpoint"
        params = {"token_id": token_id}
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            mid = float(data.get("mid", 0.0))
            self.logger.debug(f"Midpoint for {token_id[:20]}...: {mid:.4f}")
            return mid
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error getting midpoint for {token_id[:20]}...: {e}")
            return None
        except (ValueError, KeyError) as e:
            self.logger.error(f"Error parsing midpoint response: {e}")
            return None
    
    def get_orderbook(self, token_id: str, depth: Optional[int] = None) -> Optional[Dict]:
        """
        Get full orderbook for a token.
        
        Args:
            token_id: Token ID (UP or DOWN)
            depth: Number of levels to return (None = all)
            
        Returns:
            Dictionary with bids, asks, and metadata or None if failed
        """
        url = f"{self.base_url}/book"
        params = {"token_id": token_id}
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            # Apply depth limit if specified
            depth = depth or self.orderbook_depth
            if depth and "bids" in data:
                data["bids"] = data["bids"][:depth]
            if depth and "asks" in data:
                data["asks"] = data["asks"][:depth]
            
            self.logger.debug(f"Orderbook for {token_id[:20]}...: {len(data.get('bids', []))} bids, {len(data.get('asks', []))} asks")
            return data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error getting orderbook for {token_id[:20]}...: {e}")
            return None
    
    def get_market_data(self, token_id_up: str, token_id_down: str) -> Optional[Dict]:
        """
        Get complete market data for both UP and DOWN tokens.
        
        Uses /book endpoint which provides actual data timestamps from the API.
        
        Args:
            token_id_up: Token ID for UP outcome
            token_id_down: Token ID for DOWN outcome
            
        Returns:
            Dictionary with all market data including actual API timestamp
        """
        collected_at = datetime.now(timezone.utc)
        
        # Get orderbooks (these include timestamps from the API)
        book_up = self.get_orderbook(token_id_up)
        book_down = self.get_orderbook(token_id_down)
        
        if book_up is None and book_down is None:
            self.logger.warning("Failed to get any orderbook data")
            return None
        
        # Extract the API timestamp from the book response
        # The /book endpoint returns a 'timestamp' field in milliseconds
        api_timestamp = None
        api_timestamp_ms = None
        
        if book_up and 'timestamp' in book_up:
            api_timestamp_ms = book_up['timestamp']
        elif book_down and 'timestamp' in book_down:
            api_timestamp_ms = book_down['timestamp']
        
        if api_timestamp_ms:
            # Convert string to int if needed, then to datetime
            if isinstance(api_timestamp_ms, str):
                api_timestamp_ms = int(api_timestamp_ms)
            api_timestamp = datetime.fromtimestamp(api_timestamp_ms / 1000, tz=timezone.utc)
        else:
            # Fallback to collection time if no API timestamp
            api_timestamp = collected_at
            self.logger.debug("No API timestamp found, using collection time")
        
        # Extract top 6 levels of order book depth
        # Best bid = HIGHEST bid price (what buyers will pay)
        # Best ask = LOWEST ask price (what sellers will accept)
        def extract_depth(book: Optional[Dict], n_levels: int = 6) -> Dict:
            """Extract top N levels of bid/ask with sizes."""
            result = {
                "mid": None,
            }
            
            # Initialize all levels to None
            for i in range(1, n_levels + 1):
                suffix = "" if i == 1 else f"_{i}"
                result[f"bid{suffix}"] = None
                result[f"bid{suffix}_size"] = None
                result[f"ask{suffix}"] = None
                result[f"ask{suffix}_size"] = None
            
            if not book:
                return result
            
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            
            # Sort bids descending (highest first = best for sellers)
            if bids:
                sorted_bids = sorted(bids, key=lambda x: float(x["price"]), reverse=True)
                for i, bid in enumerate(sorted_bids[:n_levels]):
                    suffix = "" if i == 0 else f"_{i+1}"
                    result[f"bid{suffix}"] = float(bid["price"])
                    result[f"bid{suffix}_size"] = float(bid["size"])
            
            # Sort asks ascending (lowest first = best for buyers)
            if asks:
                sorted_asks = sorted(asks, key=lambda x: float(x["price"]))
                for i, ask in enumerate(sorted_asks[:n_levels]):
                    suffix = "" if i == 0 else f"_{i+1}"
                    result[f"ask{suffix}"] = float(ask["price"])
                    result[f"ask{suffix}_size"] = float(ask["size"])
            
            # Calculate midpoint from best bid/ask
            best_bid = result.get("bid")
            best_ask = result.get("ask")
            if best_bid is not None and best_ask is not None:
                result["mid"] = (best_bid + best_ask) / 2
            elif best_bid is not None:
                result["mid"] = best_bid
            elif best_ask is not None:
                result["mid"] = best_ask
            
            return result
        
        up_data = extract_depth(book_up, n_levels=6)
        down_data = extract_depth(book_down, n_levels=6)
        
        return {
            # TIMESTAMP SEMANTICS:
            # - source_timestamp (was 'timestamp'): When PM says the data was valid
            # - received_timestamp (was 'collected_at'): When we actually saw it
            # For PM, these are typically within ~1s of each other.
            "source_timestamp": api_timestamp,
            "received_timestamp": collected_at,
            "timestamp": api_timestamp,  # DEPRECATED: Use source_timestamp
            "timestamp_ms": api_timestamp_ms,
            "collected_at": collected_at,  # DEPRECATED: Use received_timestamp
            
            # UP token - Level 1 (best)
            "up_mid": up_data["mid"],
            "up_bid": up_data["bid"],
            "up_bid_size": up_data["bid_size"],
            "up_ask": up_data["ask"],
            "up_ask_size": up_data["ask_size"],
            
            # UP token - Levels 2-6
            "up_bid_2": up_data["bid_2"],
            "up_bid_2_size": up_data["bid_2_size"],
            "up_ask_2": up_data["ask_2"],
            "up_ask_2_size": up_data["ask_2_size"],
            "up_bid_3": up_data["bid_3"],
            "up_bid_3_size": up_data["bid_3_size"],
            "up_ask_3": up_data["ask_3"],
            "up_ask_3_size": up_data["ask_3_size"],
            "up_bid_4": up_data["bid_4"],
            "up_bid_4_size": up_data["bid_4_size"],
            "up_ask_4": up_data["ask_4"],
            "up_ask_4_size": up_data["ask_4_size"],
            "up_bid_5": up_data["bid_5"],
            "up_bid_5_size": up_data["bid_5_size"],
            "up_ask_5": up_data["ask_5"],
            "up_ask_5_size": up_data["ask_5_size"],
            "up_bid_6": up_data["bid_6"],
            "up_bid_6_size": up_data["bid_6_size"],
            "up_ask_6": up_data["ask_6"],
            "up_ask_6_size": up_data["ask_6_size"],
            
            # DOWN token - Level 1 (best)
            "down_mid": down_data["mid"],
            "down_bid": down_data["bid"],
            "down_bid_size": down_data["bid_size"],
            "down_ask": down_data["ask"],
            "down_ask_size": down_data["ask_size"],
            
            # DOWN token - Levels 2-6
            "down_bid_2": down_data["bid_2"],
            "down_bid_2_size": down_data["bid_2_size"],
            "down_ask_2": down_data["ask_2"],
            "down_ask_2_size": down_data["ask_2_size"],
            "down_bid_3": down_data["bid_3"],
            "down_bid_3_size": down_data["bid_3_size"],
            "down_ask_3": down_data["ask_3"],
            "down_ask_3_size": down_data["ask_3_size"],
            "down_bid_4": down_data["bid_4"],
            "down_bid_4_size": down_data["bid_4_size"],
            "down_ask_4": down_data["ask_4"],
            "down_ask_4_size": down_data["ask_4_size"],
            "down_bid_5": down_data["bid_5"],
            "down_bid_5_size": down_data["bid_5_size"],
            "down_ask_5": down_data["ask_5"],
            "down_ask_5_size": down_data["ask_5_size"],
            "down_bid_6": down_data["bid_6"],
            "down_bid_6_size": down_data["bid_6_size"],
            "down_ask_6": down_data["ask_6"],
            "down_ask_6_size": down_data["ask_6_size"],
            
            # DEPRECATED field names (for backward compatibility)
            "up_best_bid": up_data["bid"],
            "up_best_bid_size": up_data["bid_size"],
            "up_best_ask": up_data["ask"],
            "up_best_ask_size": up_data["ask_size"],
            "down_best_bid": down_data["bid"],
            "down_best_bid_size": down_data["bid_size"],
            "down_best_ask": down_data["ask"],
            "down_best_ask_size": down_data["ask_size"],
            
            # Raw book data (for debugging/extended analysis)
            "book_up": book_up,
            "book_down": book_down,
        }
    
    def search_markets(self, query: str = "btc", limit: int = 100) -> pd.DataFrame:
        """
        Search for markets containing a query string.
        
        Args:
            query: Search query (e.g., "btc", "bitcoin")
            limit: Maximum number of results
            
        Returns:
            DataFrame with matching markets
        """
        url = f"{self.base_url}/markets"
        params = {"limit": limit}
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            markets = data.get("data", [])
            
            # Filter for query
            matching = []
            for market in markets:
                question = market.get("question", "").lower()
                description = market.get("description", "").lower()
                
                if query.lower() in question or query.lower() in description:
                    tokens = market.get("tokens", [])
                    
                    # Extract UP/DOWN token IDs
                    token_up = None
                    token_down = None
                    for token in tokens:
                        outcome = token.get("outcome", "").lower()
                        if "up" in outcome:
                            token_up = token.get("token_id")
                        elif "down" in outcome:
                            token_down = token.get("token_id")
                    
                    matching.append({
                        "market_slug": market.get("market_slug"),
                        "question": market.get("question"),
                        "condition_id": market.get("condition_id"),
                        "token_id_up": token_up,
                        "token_id_down": token_down,
                        "end_date": market.get("end_date_iso"),
                        "active": market.get("active"),
                        "closed": market.get("closed"),
                    })
            
            self.logger.info(f"Found {len(matching)} markets matching '{query}'")
            return pd.DataFrame(matching)
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error searching markets: {e}")
            return pd.DataFrame()
    
    def get_market_by_slug(self, slug: str) -> Optional[Dict]:
        """
        Get market details by slug using gamma API.
        
        This is the preferred method for 15-min BTC markets as they
        don't appear in the standard CLOB /markets endpoint.
        
        Args:
            slug: Market slug (e.g., "btc-updown-15m-1767482100")
            
        Returns:
            Dictionary with market details including token IDs
        """
        url = f"https://gamma-api.polymarket.com/markets/slug/{slug}"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse outcomes and token IDs
            outcomes = json.loads(data.get("outcomes", "[]"))
            clob_token_ids = json.loads(data.get("clobTokenIds", "[]"))
            
            # Map outcomes to token IDs (order matches)
            token_up = None
            token_down = None
            for i, outcome in enumerate(outcomes):
                if outcome.lower() == "up" and i < len(clob_token_ids):
                    token_up = clob_token_ids[i]
                elif outcome.lower() == "down" and i < len(clob_token_ids):
                    token_down = clob_token_ids[i]
            
            result = {
                "id": data.get("id"),
                "slug": data.get("slug"),
                "question": data.get("question"),
                "description": data.get("description"),
                "end_date": data.get("endDate"),
                "resolution_source": data.get("resolutionSource"),
                "token_id_up": token_up,
                "token_id_down": token_down,
                "outcomes": outcomes,
                "active": data.get("active"),
                "closed": data.get("closed"),
                "condition_id": data.get("conditionId"),
            }
            
            self.logger.info(f"Found market: {result['question']}")
            return result
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error getting market {slug}: {e}")
            return None
    
    def find_active_btc_15min_market(self) -> Optional[Dict]:
        """
        Find the currently active BTC 15-minute Up/Down market.
        
        Markets are named btc-updown-15m-{timestamp} where timestamp is Unix seconds.
        Markets run for 15 minutes, so we calculate the current market based on time.
        
        Returns:
            Dictionary with market details or None if not found
        """
        now = datetime.now(timezone.utc)
        now_timestamp = int(now.timestamp())
        
        # BTC 15-min markets start at 15-minute intervals
        # Round down to nearest 15-minute mark
        market_start_timestamp = (now_timestamp // 900) * 900
        
        # Try current market first
        slugs_to_try = [
            f"btc-updown-15m-{market_start_timestamp}",
            f"btc-updown-15m-{market_start_timestamp - 900}",  # Previous market (might still be active)
            f"btc-updown-15m-{market_start_timestamp + 900}",  # Next market (might be created early)
        ]
        
        self.logger.info(f"Searching for active BTC 15-min market (current time: {now.isoformat()})")
        
        for slug in slugs_to_try:
            market = self.get_market_by_slug(slug)
            
            if market:
                # Check if market is active and not closed
                end_date_str = market.get("end_date")
                if end_date_str:
                    try:
                        end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                        is_future = end_date > now
                        
                        if market.get("active") and not market.get("closed") and is_future:
                            self.logger.info(f"Found active market: {market['question']} (ends at {end_date_str})")
                            return market
                        else:
                            self.logger.debug(f"Market {slug} is not active (active={market.get('active')}, closed={market.get('closed')}, future={is_future})")
                    except (ValueError, AttributeError) as e:
                        self.logger.warning(f"Could not parse end_date for {slug}: {e}")
                        # Still return it if it has token IDs
                        if market.get("token_id_up") and market.get("token_id_down"):
                            return market
        
        self.logger.warning("No active BTC 15-min market found")
        return None
    
    def find_active_market(self, asset: str) -> Optional[Dict]:
        """
        Find the currently active 15-minute Up/Down market for any asset.
        
        Supports: BTC, ETH, SOL, XRP
        
        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)
            
        Returns:
            Dictionary with market details or None if not found
        """
        asset = asset.upper()
        if asset not in SUPPORTED_ASSETS:
            self.logger.error(f"Unsupported asset: {asset}. Supported: {SUPPORTED_ASSETS}")
            return None
        
        config = get_asset_config(asset)
        slug_prefix = config.polymarket_slug_prefix
        
        now = datetime.now(timezone.utc)
        now_timestamp = int(now.timestamp())
        
        # 15-min markets start at 15-minute intervals
        # Round down to nearest 15-minute mark
        market_start_timestamp = (now_timestamp // 900) * 900
        
        # Try current market first, then previous and next
        slugs_to_try = [
            f"{slug_prefix}-{market_start_timestamp}",
            f"{slug_prefix}-{market_start_timestamp - 900}",  # Previous market
            f"{slug_prefix}-{market_start_timestamp + 900}",  # Next market
        ]
        
        self.logger.info(f"Searching for active {asset} 15-min market (current time: {now.isoformat()})")
        
        for slug in slugs_to_try:
            market = self.get_market_by_slug(slug)
            
            if market:
                # Add asset info to market
                market["asset"] = asset
                market["market_slug"] = slug
                
                # Check if market is active and not closed
                end_date_str = market.get("end_date")
                if end_date_str:
                    try:
                        end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                        is_future = end_date > now
                        
                        if market.get("active") and not market.get("closed") and is_future:
                            self.logger.info(f"Found active {asset} market: {market['question']} (ends at {end_date_str})")
                            return market
                        else:
                            self.logger.debug(
                                f"Market {slug} is not active (active={market.get('active')}, "
                                f"closed={market.get('closed')}, future={is_future})"
                            )
                    except (ValueError, AttributeError) as e:
                        self.logger.warning(f"Could not parse end_date for {slug}: {e}")
                        # Still return it if it has token IDs
                        if market.get("token_id_up") and market.get("token_id_down"):
                            return market
        
        self.logger.warning(f"No active {asset} 15-min market found")
        return None
    
    def find_next_market(self, asset: str) -> Optional[Dict]:
        """
        Find the next upcoming 15-minute market for an asset.
        
        This is used for pre-loading the next market before the current one ends.
        
        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)
            
        Returns:
            Dictionary with market details or None if not found
        """
        asset = asset.upper()
        if asset not in SUPPORTED_ASSETS:
            return None
        
        config = get_asset_config(asset)
        slug_prefix = config.polymarket_slug_prefix
        
        now = datetime.now(timezone.utc)
        now_timestamp = int(now.timestamp())
        
        # Calculate next market start
        current_market_start = (now_timestamp // 900) * 900
        next_market_start = current_market_start + 900
        
        slug = f"{slug_prefix}-{next_market_start}"
        self.logger.info(f"Looking for next {asset} market: {slug}")
        
        market = self.get_market_by_slug(slug)
        if market:
            market["asset"] = asset
            market["market_slug"] = slug
            self.logger.info(f"Found next {asset} market: {market.get('question')}")
        
        return market
    
    def get_market_times(self, market: Dict) -> tuple:
        """
        Extract start and end times from a market.
        
        Args:
            market: Market dictionary from find_active_market
            
        Returns:
            Tuple of (start_time, end_time) as datetime objects
        """
        end_date_str = market.get("end_date")
        if not end_date_str:
            return None, None
        
        try:
            end_time = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
            start_time = end_time - timedelta(minutes=15)
            return start_time, end_time
        except (ValueError, AttributeError):
            return None, None


class PolymarketStorage:
    """Handles persistent storage of Polymarket data."""
    
    def __init__(self, storage_dir: Optional[str] = None, format: str = "parquet"):
        """
        Initialize storage handler.
        
        Args:
            storage_dir: Directory to store data (defaults to config)
            format: Storage format ("parquet" or "sqlite")
        """
        self.storage_dir = Path(storage_dir or f"{STORAGE.raw_dir}/polymarket")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
        
        self.logger = setup_logger("polymarket_storage")
        self.logger.info(f"PolymarketStorage initialized at {self.storage_dir}")
        
        if format == "sqlite":
            import sqlite3
            self.db_path = self.storage_dir / "polymarket.db"
            self.conn = sqlite3.connect(self.db_path)
            self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database schema."""
        if self.format != "sqlite":
            return
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                collected_at TEXT PRIMARY KEY,
                up_mid REAL,
                up_best_bid REAL,
                up_best_bid_size REAL,
                up_best_ask REAL,
                up_best_ask_size REAL,
                down_mid REAL,
                down_best_bid REAL,
                down_best_bid_size REAL,
                down_best_ask REAL,
                down_best_ask_size REAL
            )
        """)
        self.conn.commit()
    
    def save_market_data(self, data: Dict):
        """
        Save market data snapshot.
        
        Args:
            data: Market data dictionary from get_market_data()
        """
        if not data:
            return
        
        # Create DataFrame row
        row = {
            "collected_at": data["collected_at"],
            "up_mid": data.get("up_mid"),
            "up_best_bid": data.get("up_best_bid"),
            "up_best_bid_size": data.get("up_best_bid_size"),
            "up_best_ask": data.get("up_best_ask"),
            "up_best_ask_size": data.get("up_best_ask_size"),
            "down_mid": data.get("down_mid"),
            "down_best_bid": data.get("down_best_bid"),
            "down_best_bid_size": data.get("down_best_bid_size"),
            "down_best_ask": data.get("down_best_ask"),
            "down_best_ask_size": data.get("down_best_ask_size"),
        }
        
        df = pd.DataFrame([row])
        
        if self.format == "parquet":
            date_str = data["collected_at"].strftime("%Y-%m-%d")
            file_path = self.storage_dir / f"market_data_{date_str}.parquet"
            
            if file_path.exists():
                existing = pd.read_parquet(file_path)
                # Filter out empty/all-NA rows and columns before concat to avoid FutureWarning
                existing = existing.dropna(how='all').dropna(axis=1, how='all')
                df_clean = df.dropna(how='all').dropna(axis=1, how='all')
                if len(existing) > 0 and len(df_clean) > 0:
                    combined = pd.concat([existing, df_clean], ignore_index=True)
                elif len(df_clean) > 0:
                    combined = df_clean
                else:
                    combined = existing
                combined.to_parquet(file_path, index=False)
            else:
                df.to_parquet(file_path, index=False)
                self.logger.info(f"Created {file_path}")
        
        elif self.format == "sqlite":
            df_copy = df.copy()
            df_copy["collected_at"] = df_copy["collected_at"].astype(str)
            df_copy.to_sql("market_data", self.conn, if_exists="append", index=False)
            self.conn.commit()
    
    def save_orderbook_snapshot(self, token_id: str, book: Dict, timestamp: datetime):
        """Save full orderbook snapshot (for detailed analysis)."""
        if not book:
            return
        
        date_str = timestamp.strftime("%Y-%m-%d")
        file_path = self.storage_dir / f"orderbook_{token_id[:20]}_{date_str}.json"
        
        # Append to JSON lines file
        snapshot = {
            "timestamp": timestamp.isoformat(),
            "token_id": token_id,
            "book": book,
        }
        
        with open(file_path, "a") as f:
            f.write(json.dumps(snapshot) + "\n")
    
    def load_market_data(self, start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load market data from storage.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            DataFrame with market data
        """
        if self.format == "parquet":
            files = sorted(self.storage_dir.glob("market_data_*.parquet"))
            if not files:
                return pd.DataFrame()
            
            dfs = [pd.read_parquet(f) for f in files]
            # Filter out empty dataframes and rows to avoid FutureWarning
            cleaned_dfs = []
            for d in dfs:
                if len(d) > 0:
                    # Drop rows and columns that are all NA
                    d = d.dropna(how='all').dropna(axis=1, how='all')
                    if len(d) > 0:
                        cleaned_dfs.append(d)
            if not cleaned_dfs:
                return pd.DataFrame()
            df = pd.concat(cleaned_dfs, ignore_index=True)
            
        elif self.format == "sqlite":
            df = pd.read_sql_query("SELECT * FROM market_data", self.conn)
        
        else:
            return pd.DataFrame()
        
        if "collected_at" in df.columns:
            df["collected_at"] = pd.to_datetime(df["collected_at"])
            
            if start_time:
                df = df[df["collected_at"] >= start_time]
            if end_time:
                df = df[df["collected_at"] <= end_time]
        
        return df.sort_values("collected_at") if "collected_at" in df.columns else df
