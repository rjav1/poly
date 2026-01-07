"""
Fast async Polymarket collector using aiohttp for parallel requests.

Key optimizations:
1. Uses aiohttp instead of requests (non-blocking)
2. Fetches UP and DOWN orderbooks in parallel
3. Minimal overhead - no debug logging in hot path
4. Connection pooling for reduced latency
"""

import asyncio
import aiohttp
from datetime import datetime, timezone
from typing import Optional, Dict, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import POLYMARKET


class FastPolymarketCollector:
    """
    High-performance async Polymarket collector.
    
    Uses aiohttp with connection pooling for minimal latency.
    Fetches UP and DOWN orderbooks in parallel.
    """
    
    def __init__(self, timeout: float = 5.0):
        """
        Initialize collector.
        
        Args:
            timeout: Request timeout in seconds (default: 5.0)
        """
        self.base_url = POLYMARKET.base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None
    
    async def _ensure_session(self):
        """Ensure aiohttp session is initialized."""
        if self.session is None or self.session.closed:
            # Connection pooling: keep connections alive, limit per host
            self._connector = aiohttp.TCPConnector(
                limit=20,  # Max connections
                limit_per_host=10,  # Max per host
                keepalive_timeout=30,  # Keep connections alive
                enable_cleanup_closed=True,
            )
            self.session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=self.timeout,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            )
    
    async def close(self):
        """Close the session."""
        if self.session and not self.session.closed:
            await self.session.close()
        if self._connector:
            await self._connector.close()
    
    async def get_orderbook(self, token_id: str) -> Optional[Dict]:
        """
        Get orderbook for a single token (async).
        
        Args:
            token_id: Token ID
            
        Returns:
            Orderbook dict or None if failed
        """
        await self._ensure_session()
        
        url = f"{self.base_url}/book"
        params = {"token_id": token_id}
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception:
            return None
    
    async def get_market_data_parallel(
        self, 
        token_id_up: str, 
        token_id_down: str,
        n_levels: int = 6
    ) -> Optional[Dict]:
        """
        Get market data for both tokens in PARALLEL.
        
        This is the key optimization - both API calls happen simultaneously.
        
        Args:
            token_id_up: UP token ID
            token_id_down: DOWN token ID
            n_levels: Number of order book levels to extract
            
        Returns:
            Combined market data dict or None if both failed
        """
        await self._ensure_session()
        
        # PARALLEL fetch - both requests at once
        book_up, book_down = await asyncio.gather(
            self.get_orderbook(token_id_up),
            self.get_orderbook(token_id_down),
            return_exceptions=True
        )
        
        # Handle exceptions from gather
        if isinstance(book_up, Exception):
            book_up = None
        if isinstance(book_down, Exception):
            book_down = None
        
        if book_up is None and book_down is None:
            return None
        
        collected_at = datetime.now(timezone.utc)
        
        # Extract API timestamp
        api_timestamp = None
        api_timestamp_ms = None
        
        if book_up and 'timestamp' in book_up:
            api_timestamp_ms = book_up['timestamp']
        elif book_down and 'timestamp' in book_down:
            api_timestamp_ms = book_down['timestamp']
        
        if api_timestamp_ms:
            if isinstance(api_timestamp_ms, str):
                api_timestamp_ms = int(api_timestamp_ms)
            api_timestamp = datetime.fromtimestamp(api_timestamp_ms / 1000, tz=timezone.utc)
        else:
            api_timestamp = collected_at
        
        # Build result with all levels
        result = {
            'timestamp': api_timestamp,
            'timestamp_ms': api_timestamp_ms,
            'collected_at': collected_at,
            'received_timestamp': collected_at,
        }
        
        # Extract UP token data
        up_data = self._extract_depth(book_up, n_levels)
        for key, val in up_data.items():
            result[f'up_{key}'] = val
        
        # Extract DOWN token data
        down_data = self._extract_depth(book_down, n_levels)
        for key, val in down_data.items():
            result[f'down_{key}'] = val
        
        return result
    
    def _extract_depth(self, book: Optional[Dict], n_levels: int = 6) -> Dict:
        """Extract top N levels of bid/ask with sizes."""
        result = {"mid": None}
        
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
        
        # Sort bids descending (highest first)
        if bids:
            sorted_bids = sorted(bids, key=lambda x: float(x["price"]), reverse=True)
            for i, bid in enumerate(sorted_bids[:n_levels]):
                suffix = "" if i == 0 else f"_{i+1}"
                result[f"bid{suffix}"] = float(bid["price"])
                result[f"bid{suffix}_size"] = float(bid["size"])
        
        # Sort asks ascending (lowest first)
        if asks:
            sorted_asks = sorted(asks, key=lambda x: float(x["price"]))
            for i, ask in enumerate(sorted_asks[:n_levels]):
                suffix = "" if i == 0 else f"_{i+1}"
                result[f"ask{suffix}"] = float(ask["price"])
                result[f"ask{suffix}_size"] = float(ask["size"])
        
        # Calculate mid from best bid/ask
        best_bid = result.get("bid")
        best_ask = result.get("ask")
        if best_bid is not None and best_ask is not None:
            result["mid"] = (best_bid + best_ask) / 2
        elif best_bid is not None:
            result["mid"] = best_bid
        elif best_ask is not None:
            result["mid"] = best_ask
        
        return result


async def benchmark():
    """Benchmark the collector speed."""
    import time
    
    # Test token IDs (ETH market example)
    token_up = "21742633143463906290569050155826241533067272736897614950488156847949938836455"
    token_down = "48331043336612883890938759509493159234755048973500640148014422747788308965732"
    
    collector = FastPolymarketCollector()
    
    try:
        # Warmup
        await collector.get_market_data_parallel(token_up, token_down)
        
        # Benchmark
        n_iters = 20
        start = time.time()
        
        for _ in range(n_iters):
            data = await collector.get_market_data_parallel(token_up, token_down)
        
        elapsed = time.time() - start
        avg_ms = (elapsed / n_iters) * 1000
        
        print(f"Average latency: {avg_ms:.1f}ms per request ({n_iters} iterations)")
        print(f"Max sustainable rate: {1000/avg_ms:.1f} requests/second")
        
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(benchmark())

