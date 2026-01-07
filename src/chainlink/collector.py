"""
Production-ready Chainlink BTC/USD Data Streams collector.

Uses the GraphQL API endpoints discovered during exploration.
Provides live price data with bid/ask/mid.
"""

import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
import json
from urllib.parse import urlencode
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logging import setup_logger, get_logger
from config.settings import CHAINLINK, STORAGE


class ChainlinkCollector:
    """
    Collects Chainlink BTC/USD price data via GraphQL API.
    
    API Endpoints:
    - LIVE_STREAM_REPORTS_QUERY: Real-time price reports
    - HISTORICAL_1D_QUERY: Historical 15-minute aggregated data
    
    Data Format:
    - Prices are returned as large integers (divide by 10^18 for USD)
    - Timestamps are ISO format with timezone
    """
    
    def __init__(self, feed_id: Optional[str] = None, log_level: int = 20):
        """
        Initialize Chainlink collector.
        
        Args:
            feed_id: Chainlink feed ID for BTC/USD (uses default if not provided)
            log_level: Logging level (default INFO=20)
        """
        self.feed_id = feed_id or CHAINLINK.feed_id
        self.base_url = CHAINLINK.base_url
        self.timeout = CHAINLINK.request_timeout
        self.price_divisor = CHAINLINK.price_divisor
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        })
        
        self.logger = setup_logger("chainlink", level=log_level)
        self.logger.info(f"ChainlinkCollector initialized with feed_id: {self.feed_id[:20]}...")
    
    def _parse_price(self, price_str: str) -> Optional[float]:
        """
        Parse price string (large integer) to float USD.
        
        Args:
            price_str: Price as string integer
            
        Returns:
            Price in USD or None if parsing fails
        """
        try:
            price_int = int(price_str)
            return price_int / self.price_divisor
        except (ValueError, TypeError):
            self.logger.warning(f"Failed to parse price: {price_str}")
            return None
    
    def get_live_reports(self, limit: int = 100) -> pd.DataFrame:
        """
        Get latest live stream reports.
        
        Args:
            limit: Maximum number of reports to return
            
        Returns:
            DataFrame with columns: timestamp, price, bid, ask, mid
        """
        variables = {"feedId": self.feed_id}
        params = {
            "query": "LIVE_STREAM_REPORTS_QUERY",
            "variables": json.dumps(variables)
        }
        
        url = f"{self.base_url}/query-timescale?{urlencode(params)}"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            nodes = data.get("data", {}).get("liveStreamReports", {}).get("nodes", [])
            
            records = []
            for node in nodes[:limit]:
                timestamp_str = node.get("validFromTimestamp")
                timestamp = pd.to_datetime(timestamp_str) if timestamp_str else None
                
                price = self._parse_price(node.get("price", "0"))
                bid = self._parse_price(node.get("bid", "0"))
                ask = self._parse_price(node.get("ask", "0"))
                
                records.append({
                    "timestamp": timestamp,
                    "price": price,
                    "bid": bid,
                    "ask": ask,
                    "mid": price,  # mid = price in this API
                    "collected_at": datetime.now(timezone.utc),
                })
            
            df = pd.DataFrame(records)
            self.logger.debug(f"Collected {len(df)} live reports")
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching live reports: {e}")
            return pd.DataFrame()
    
    def get_latest_price(self) -> Optional[Dict]:
        """
        Get the single latest price report.
        
        Returns:
            Dictionary with price data or None if failed
        """
        df = self.get_live_reports(limit=1)
        if len(df) > 0:
            row = df.iloc[0]
            return {
                "timestamp": row["timestamp"],
                "price": row["price"],
                "bid": row["bid"],
                "ask": row["ask"],
                "mid": row["mid"],
                "collected_at": row["collected_at"],
            }
        return None
    
    def get_historical_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Get historical aggregated data for a time range.
        
        Note: This returns 15-minute aggregated data, not tick-level.
        
        Args:
            start_time: Start time (UTC)
            end_time: End time (UTC)
            
        Returns:
            DataFrame with columns: timestamp, open, mid, bid, ask
        """
        variables = {
            "feedId": self.feed_id,
            "startTime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "endTime": end_time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        params = {
            "query": "HISTORICAL_1D_QUERY",
            "variables": json.dumps(variables)
        }
        
        url = f"{self.base_url}/query-timescale?{urlencode(params)}"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            nodes = data.get("data", {}).get("mercuryHistory15MinMarkers", {}).get("nodes", [])
            
            records = []
            for node in nodes:
                timestamp_str = node.get("timeBucket")
                timestamp = pd.to_datetime(timestamp_str) if timestamp_str else None
                
                records.append({
                    "timestamp": timestamp,
                    "open": self._parse_price(node.get("open", "0")),
                    "mid": self._parse_price(node.get("mid", "0")),
                    "bid": self._parse_price(node.get("bid", "0")),
                    "ask": self._parse_price(node.get("ask", "0")),
                })
            
            df = pd.DataFrame(records)
            self.logger.info(f"Collected {len(df)} historical records from {start_time} to {end_time}")
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()


class ChainlinkStorage:
    """Handles persistent storage of Chainlink price data."""
    
    def __init__(self, storage_dir: Optional[str] = None, format: str = "parquet"):
        """
        Initialize storage handler.
        
        Args:
            storage_dir: Directory to store data (defaults to config)
            format: Storage format ("parquet" or "sqlite")
        """
        self.storage_dir = Path(storage_dir or f"{STORAGE.raw_dir}/chainlink")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
        
        self.logger = setup_logger("chainlink_storage")
        self.logger.info(f"ChainlinkStorage initialized at {self.storage_dir}")
        
        if format == "sqlite":
            import sqlite3
            self.db_path = self.storage_dir / "chainlink_prices.db"
            self.conn = sqlite3.connect(self.db_path)
            self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database schema."""
        if self.format != "sqlite":
            return
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                timestamp TEXT,
                price REAL,
                bid REAL,
                ask REAL,
                mid REAL,
                collected_at TEXT,
                PRIMARY KEY (timestamp)
            )
        """)
        self.conn.commit()
    
    def save(self, df: pd.DataFrame):
        """
        Save DataFrame to storage.
        
        Args:
            df: DataFrame with price data
        """
        if len(df) == 0:
            return
        
        if self.format == "parquet":
            # Save as daily partitioned parquet files
            if "timestamp" in df.columns:
                df = df.copy()
                df["date"] = pd.to_datetime(df["timestamp"]).dt.date
                
                for date, group in df.groupby("date"):
                    file_path = self.storage_dir / f"prices_{date}.parquet"
                    
                    if file_path.exists():
                        existing = pd.read_parquet(file_path)
                        # Filter out empty rows before concat to avoid FutureWarning
                        existing = existing.dropna(how='all')
                        group_clean = group.dropna(how='all')
                        combined = pd.concat([existing, group_clean], ignore_index=True).drop_duplicates(subset=["timestamp"])
                        combined = combined.sort_values("timestamp")
                        combined.to_parquet(file_path, index=False)
                        self.logger.debug(f"Appended {len(group)} records to {file_path}")
                    else:
                        group.to_parquet(file_path, index=False)
                        self.logger.info(f"Created {file_path} with {len(group)} records")
        
        elif self.format == "sqlite":
            # Convert timestamps to strings for SQLite
            df_copy = df.copy()
            for col in ["timestamp", "collected_at"]:
                if col in df_copy.columns:
                    df_copy[col] = df_copy[col].astype(str)
            
            df_copy.to_sql("prices", self.conn, if_exists="append", index=False)
            self.conn.commit()
            self.logger.debug(f"Saved {len(df)} records to SQLite")
    
    def load(self, start_time: Optional[datetime] = None, 
             end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load data from storage.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            DataFrame with price data
        """
        if self.format == "parquet":
            files = sorted(self.storage_dir.glob("prices_*.parquet"))
            if not files:
                return pd.DataFrame()
            
            dfs = [pd.read_parquet(f) for f in files]
            # Filter out empty dataframes before concat to avoid FutureWarning
            dfs = [d.dropna(how='all') for d in dfs if len(d) > 0]
            if not dfs:
                return pd.DataFrame()
            df = pd.concat(dfs, ignore_index=True)
            
        elif self.format == "sqlite":
            query = "SELECT * FROM prices"
            df = pd.read_sql_query(query, self.conn)
        
        else:
            return pd.DataFrame()
        
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            if start_time:
                df = df[df["timestamp"] >= start_time]
            if end_time:
                df = df[df["timestamp"] <= end_time]
        
        return df.sort_values("timestamp") if "timestamp" in df.columns else df
