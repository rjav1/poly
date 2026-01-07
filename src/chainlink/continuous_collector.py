"""
Continuous multi-asset Chainlink data collector.

This collector:
1. Runs continuously, never stopping across market boundaries
2. Collects from multiple assets (BTC, ETH, SOL, XRP) simultaneously
3. Uses one browser instance with multiple pages for efficiency
4. Stores data with explicit is_observed flags
5. Maintains microsecond timestamp precision

Usage:
    collector = ContinuousChainlinkCollector(assets=["BTC", "ETH", "SOL", "XRP"])
    await collector.start()  # Runs indefinitely until stopped
    await collector.stop()
"""

import asyncio
import time
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, Dict, List, Callable
from pathlib import Path
import csv
import threading
from dateutil import parser

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logging import setup_logger
from config.settings import ASSETS, CHAINLINK, STORAGE, get_asset_config, SUPPORTED_ASSETS


class ContinuousChainlinkCollector:
    """
    Continuous multi-asset Chainlink data collector.
    
    Runs continuously collecting price data from all configured assets.
    Uses Playwright to scrape the Chainlink data streams frontend.
    """
    
    def __init__(
        self,
        assets: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        log_level: int = 20,
        sequential_collection: bool = False,
    ):
        """
        Initialize continuous collector.
        
        Args:
            assets: List of asset symbols to collect (default: all supported)
            output_dir: Directory for output files (default: data/raw)
            log_level: Logging level
            sequential_collection: If True, collect one asset at a time (more reliable, less resource-intensive)
        """
        self.assets = assets or SUPPORTED_ASSETS
        base_output_dir = Path(output_dir or STORAGE.raw_dir)
        # Organize by asset: data_v2/raw/chainlink/{asset}/
        self.output_dir = base_output_dir / "chainlink"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger("chainlink_continuous", level=log_level)
        self.sequential_collection = sequential_collection
        self.logger.info(f"Initializing continuous collector for assets: {self.assets}")
        if sequential_collection:
            self.logger.info("Using sequential collection mode (one asset at a time)")
        
        # Browser state
        self.browser = None
        self.pages: Dict[str, any] = {}  # asset -> page
        self._playwright_factory = None  # The async_playwright function
        self._playwright = None  # The running Playwright instance
        self._playwright_installed = False
        
        # Collection state
        self.running = False
        self.stats = {asset: {"collected": 0, "errors": 0, "last_price": None} for asset in self.assets}
        
        # Output files (one per asset, in asset subdirectory)
        self.output_files: Dict[str, Path] = {}
        self.csv_writers: Dict[str, csv.DictWriter] = {}
        self.file_handles: Dict[str, any] = {}
        
        # Track timestamps to detect "stuck" UI extraction
        self._last_ui_timestamp: Dict[str, datetime] = {}
        self._last_assigned_timestamp: Dict[str, datetime] = {}
        
        # Deduplication: track written seconds and pending data per asset
        self._last_written_second: Dict[str, Optional[datetime]] = {}
        self._written_seconds: Dict[str, set] = {}  # asset -> set of seconds already written (ISO string keys)
        self._pending_data: Dict[str, Optional[Dict]] = {}  # asset -> most recent data for current second
        
        # Initialize output files
        self._init_output_files()
    
    def _init_output_files(self):
        """Initialize CSV output files for each asset in asset subdirectories.
        
        TIMESTAMP SEMANTICS (CRITICAL FOR LATENCY ANALYSIS):
        - source_timestamp: What the UI claims the data time is (may be delayed 60-90s)
        - received_timestamp: Wall-clock time when we actually observed the data
        
        For latency-edge analysis, use received_timestamp to understand when
        information was actually available for trading.
        
        HEADER VALIDATION:
        If an existing file has different headers, we backup the old file and start fresh.
        This prevents column misalignment bugs when the format changes.
        """
        fieldnames = [
            "source_timestamp",      # What UI says (was 'timestamp')
            "received_timestamp",    # When we actually saw it (was 'collected_at')
            "data_timestamp_raw",    # Raw string for debugging
            "asset", "mid", "bid", "ask", 
            "source", "is_observed"
        ]
        
        for asset in self.assets:
            # Create asset subdirectory
            asset_dir = self.output_dir / asset.upper()
            asset_dir.mkdir(parents=True, exist_ok=True)
            
            # File: chainlink_BTC_continuous.csv (more readable)
            filepath = asset_dir / f"chainlink_{asset.upper()}_continuous.csv"
            self.output_files[asset] = filepath
            
            # Check if file exists and validate headers match expected format
            write_header = True
            if filepath.exists():
                try:
                    with open(filepath, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        existing_headers = next(reader, None)
                        
                    if existing_headers:
                        if existing_headers == fieldnames:
                            # Headers match - can safely append
                            write_header = False
                            self.logger.info(f"Existing file for {asset} has matching headers - appending")
                        else:
                            # Headers DON'T match - backup old file and start fresh
                            from datetime import datetime as dt
                            backup_name = f"chainlink_{asset.upper()}_continuous_backup_{dt.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            backup_path = asset_dir / backup_name
                            import shutil
                            shutil.move(str(filepath), str(backup_path))
                            self.logger.warning(
                                f"HEADER MISMATCH for {asset}! "
                                f"Expected: {fieldnames}, "
                                f"Found: {existing_headers}. "
                                f"Old file backed up to: {backup_name}"
                            )
                            write_header = True
                except Exception as e:
                    self.logger.warning(f"Error reading existing file for {asset}: {e}. Starting fresh.")
                    write_header = True
            
            # Open file in append mode
            fh = open(filepath, 'a', newline='', encoding='utf-8')
            self.file_handles[asset] = fh
            
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                fh.flush()  # Ensure headers written to disk immediately
            self.csv_writers[asset] = writer
            
            self.logger.info(f"Output file for {asset}: {filepath}")
    
    async def _ensure_playwright(self):
        """Ensure Playwright is installed and ready."""
        if self._playwright_installed and self.browser:
            return
        
        try:
            from playwright.async_api import async_playwright
            self._playwright_factory = async_playwright
            self._playwright_installed = True
        except ImportError:
            raise ImportError(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )
        
        if not self.browser:
            self._playwright = await self._playwright_factory().start()
            self.browser = await self._playwright.chromium.launch(headless=True)
            self.logger.info("Playwright browser launched")
    
    async def _load_page(self, asset: str, retry_count: int = 0) -> bool:
        """
        Load the Chainlink page for a specific asset.
        
        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)
            retry_count: Number of retries attempted
            
        Returns:
            True if page loaded successfully
        """
        config = get_asset_config(asset)
        url = config.chainlink_url
        max_retries = 3
        
        try:
            # Close existing page if any
            if asset in self.pages:
                try:
                    await self.pages[asset].close()
                except:
                    pass
                del self.pages[asset]
            
            page = await self.browser.new_page()
            await page.set_viewport_size({"width": 1920, "height": 1080})
            
            self.logger.info(f"Loading {asset} page: {url}")
            await page.goto(url, timeout=CHAINLINK.page_load_timeout * 1000, wait_until="domcontentloaded")
            
            # Wait for chart to render (shorter wait)
            await asyncio.sleep(2)
            
            # Verify page has price data
            has_price = await page.evaluate("""
                () => {
                    const bodyText = document.body.textContent || '';
                    return bodyText.includes('Mid-price') || bodyText.includes('Mid price') || bodyText.includes('$');
                }
            """)
            
            if not has_price:
                self.logger.warning(f"Page loaded but no price data found for {asset}")
                # Retry if we haven't exceeded max retries
                if retry_count < max_retries:
                    await page.close()
                    await asyncio.sleep(1)
                    return await self._load_page(asset, retry_count + 1)
            
            self.pages[asset] = page
            self.logger.info(f"Successfully loaded page for {asset}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load page for {asset}: {e}")
            if retry_count < max_retries:
                await asyncio.sleep(2)
                return await self._load_page(asset, retry_count + 1)
            return False
    
    async def _extract_price(self, asset: str) -> Optional[Dict]:
        """
        Extract price data from a page.
        
        Args:
            asset: Asset symbol
            
        Returns:
            Dictionary with price data or None if extraction failed
        """
        page = self.pages.get(asset)
        if not page:
            return None
        
        try:
            # Method 1: DOM extraction - look for specific text patterns INCLUDING the data timestamp
            chart_data = await page.evaluate("""
                () => {
                    // Look for Mid-price, Bid, Ask values in the DOM
                    const bodyText = document.body.textContent || '';
                    
                    // CRITICAL: Extract the ACTUAL DATA TIMESTAMP from the UI
                    // The chart shows time like "22:02:44" which is the ACTUAL data time
                    // This is delayed ~60-90 seconds from real-time
                    let dataTimestamp = null;
                    
                    // First try to find full timestamps (from tooltip if visible)
                    const fullTimestampPatterns = [
                        // "Jan 4, 2026 4:58:35 PM" format
                        /(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\s+\\d{1,2},?\\s+\\d{4}\\s+\\d{1,2}:\\d{2}:\\d{2}\\s*[AP]M/gi
                    ];
                    
                    for (const pattern of fullTimestampPatterns) {
                        const matches = bodyText.match(pattern);
                        if (matches && matches.length > 0) {
                            dataTimestamp = matches[0];
                            break;
                        }
                    }
                    
                    // If no full timestamp, extract just the time (HH:MM:SS) from chart
                    if (!dataTimestamp) {
                        const timeMatch = bodyText.match(/(\\d{1,2}):(\\d{2}):(\\d{2})/);
                        if (timeMatch) {
                            dataTimestamp = timeMatch[0];  // Just "HH:MM:SS"
                        }
                    }
                    
                    // Try to find mid price
                    let midMatch = bodyText.match(/Mid-price[\\s\\S]*?\\$([\\d,]+\\.?\\d*)/i);
                    if (!midMatch) {
                        midMatch = bodyText.match(/Mid price[\\s\\S]*?\\$([\\d,]+\\.?\\d*)/i);
                    }
                    
                    // Try to find bid
                    let bidMatch = bodyText.match(/Bid[\\s\\S]*?\\$([\\d,]+\\.?\\d*)/i);
                    
                    // Try to find ask
                    let askMatch = bodyText.match(/Ask[\\s\\S]*?\\$([\\d,]+\\.?\\d*)/i);
                    
                    // Parse values
                    const parsePrice = (str) => {
                        if (!str) return null;
                        return parseFloat(str.replace(/,/g, ''));
                    };
                    
                    const mid = midMatch ? parsePrice(midMatch[1]) : null;
                    const bid = bidMatch ? parsePrice(bidMatch[1]) : null;
                    const ask = askMatch ? parsePrice(askMatch[1]) : null;
                    
                    return {
                        mid: mid,
                        bid: bid,
                        ask: ask,
                        dataTimestamp: dataTimestamp,
                        found: mid !== null && mid > 0
                    };
                }
            """)
            
            if chart_data and chart_data.get('found') and chart_data.get('mid'):
                # Parse the actual data timestamp from the UI
                # The chart shows a time like "22:02:44" which is the ACTUAL data time (delayed ~60-90s)
                data_timestamp = None
                raw_ts = chart_data.get('dataTimestamp')  # Full timestamp if found
                
                if raw_ts:
                    try:
                        # Parse "Jan 4, 2026 4:58:35 PM" or "HH:MM:SS" format
                        if len(raw_ts) <= 10:  # Just time "HH:MM:SS"
                            h, m, s = map(int, raw_ts.split(':'))
                            now = datetime.now(timezone.utc)
                            data_timestamp = now.replace(hour=h, minute=m, second=s, microsecond=0)
                            # Handle day wrap (if data time > current time, it's from yesterday)
                            if data_timestamp > now:
                                from datetime import timedelta
                                data_timestamp -= timedelta(days=1)
                        else:
                            # Parse full timestamp
                            from dateutil import parser as dateutil_parser
                            data_timestamp = dateutil_parser.parse(raw_ts)
                            if data_timestamp.tzinfo is None:
                                data_timestamp = data_timestamp.replace(tzinfo=timezone.utc)
                    except Exception as e:
                        self.logger.debug(f"Could not parse data timestamp '{raw_ts}': {e}")
                        data_timestamp = None
                
                collected_at = datetime.now(timezone.utc)
                current_price = chart_data['mid']
                
                # Timestamp handling strategy:
                # The UI timestamp extraction often gets "stuck" (same value repeated).
                # We detect this and auto-increment to ensure unique timestamps.
                from datetime import timedelta
                CL_DELAY_SECONDS = 65  # Chainlink data is typically ~65 seconds behind
                
                last_ui_ts = self._last_ui_timestamp.get(asset)
                last_assigned_ts = self._last_assigned_timestamp.get(asset)
                
                if data_timestamp:
                    if last_ui_ts and data_timestamp == last_ui_ts and last_assigned_ts:
                        # UI timestamp is stuck - increment from last assigned
                        timestamp = last_assigned_ts + timedelta(seconds=1)
                    else:
                        # New UI timestamp - use it
                        timestamp = data_timestamp
                    # Track the UI timestamp we saw
                    self._last_ui_timestamp[asset] = data_timestamp
                else:
                    # No UI timestamp - increment from last or estimate
                    if last_assigned_ts:
                        timestamp = last_assigned_ts + timedelta(seconds=1)
                    else:
                        timestamp = collected_at - timedelta(seconds=CL_DELAY_SECONDS)
                
                # Floor to second for consistent matching with PM
                timestamp = timestamp.replace(microsecond=0)
                
                # Track the assigned timestamp
                self._last_assigned_timestamp[asset] = timestamp
                
                return {
                    'source_timestamp': timestamp,  # What UI claims (may be delayed)
                    'received_timestamp': collected_at,  # When we actually saw it
                    'data_timestamp_raw': raw_ts,  # Keep raw for debugging
                    'asset': asset,
                    'mid': current_price,
                    'bid': chart_data.get('bid'),
                    'ask': chart_data.get('ask'),
                    'source': 'dom',
                    'is_observed': 1,
                }
            
            # Method 2: Look for any visible price number
            price_data = await page.evaluate("""
                () => {
                    // Find elements that might contain price
                    const priceElements = document.querySelectorAll('[class*="price"], [class*="value"], h1, h2, h3');
                    
                    for (let el of priceElements) {
                        const text = el.textContent || '';
                        const match = text.match(/\\$([\\d,]+\\.\\d{2,})/);
                        if (match) {
                            const price = parseFloat(match[1].replace(/,/g, ''));
                            if (price > 10) {  // Filter out tiny values
                                return { price: price, found: true };
                            }
                        }
                    }
                    
                    return { found: false };
                }
            """)
            
            if price_data and price_data.get('found'):
                return {
                    'timestamp': datetime.now(timezone.utc),
                    'asset': asset,
                    'mid': price_data['price'],
                    'bid': None,
                    'ask': None,
                    'source': 'element',
                    'is_observed': 1,
                    'collected_at': datetime.now(timezone.utc).isoformat()
                }
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error extracting price for {asset}: {e}")
            return None
    
    def _write_data(self, asset: str, data: Dict):
        """Write data row to CSV file, deduplicating by second (keep most recent).
        
        Deduplication is based on source_timestamp (when the data was valid),
        not received_timestamp (when we observed it).
        """
        writer = self.csv_writers.get(asset)
        if not writer:
            return
        
        # Get source_timestamp and floor to second for deduplication
        ts = data['source_timestamp']
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
        # Ensure timezone-aware for comparison
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts_second = ts.replace(microsecond=0)  # Floor to second
        
        last_second = self._last_written_second.get(asset)
        
        # Compare as ISO strings to avoid timezone comparison issues
        ts_second_str = ts_second.isoformat()
        
        # Initialize written_seconds set for this asset if needed
        if asset not in self._written_seconds:
            self._written_seconds[asset] = set()
        
        # If we've already written this second, skip (deduplication)
        if ts_second_str in self._written_seconds[asset]:
            # Update pending in case we want to overwrite, but don't write yet
            # (This handles the case where CL UI repeats old timestamps)
            self._pending_data[asset] = data
            return
        
        last_second_str = self._last_written_second[asset].isoformat() if self._last_written_second.get(asset) else None
        
        if last_second_str is None:
            # First data point - just store as pending
            self._pending_data[asset] = data
            self._last_written_second[asset] = ts_second
            return
        
        if ts_second_str == last_second_str:
            # Same second - update pending data (keep most recent)
            self._pending_data[asset] = data
            return  # Don't write yet, wait for next second
        
        # New second detected - write the pending data from previous second
        if asset in self._pending_data and self._pending_data[asset] is not None:
            pending = self._pending_data[asset]
            pending_ts = pending['source_timestamp']
            if isinstance(pending_ts, str):
                pending_ts = pd.to_datetime(pending_ts)
            if pending_ts.tzinfo is None:
                pending_ts = pending_ts.replace(tzinfo=timezone.utc)
            pending_second_str = pending_ts.replace(microsecond=0).isoformat()
            
            # Only write if we haven't written this second before
            if pending_second_str not in self._written_seconds[asset]:
                # Get source and received timestamps
                src_ts = pending['source_timestamp']
                rcv_ts = pending['received_timestamp']
                row = {
                    'source_timestamp': src_ts.isoformat() if isinstance(src_ts, datetime) else src_ts,
                    'received_timestamp': rcv_ts.isoformat() if isinstance(rcv_ts, datetime) else rcv_ts,
                    'data_timestamp_raw': pending.get('data_timestamp_raw', ''),
                    'asset': pending['asset'],
                    'mid': pending.get('mid'),
                    'bid': pending.get('bid'),
                    'ask': pending.get('ask'),
                    'source': pending.get('source', 'unknown'),
                    'is_observed': pending.get('is_observed', 1),
                }
                writer.writerow(row)
                self.file_handles[asset].flush()
                self._written_seconds[asset].add(pending_second_str)
        
        # Store current data as pending for this new second
        self._pending_data[asset] = data
        self._last_written_second[asset] = ts_second
    
    def _write_missing_data(self, asset: str, last_price: Optional[float] = None):
        """Write a row indicating missing data (forward-fill placeholder)."""
        now = datetime.now(timezone.utc)
        row = {
            'source_timestamp': now.isoformat(),  # For ffill, source = received
            'received_timestamp': now.isoformat(),
            'data_timestamp_raw': '',
            'asset': asset,
            'mid': last_price,
            'bid': None,
            'ask': None,
            'source': 'ffill',
            'is_observed': 0,  # Not observed this second
        }
        writer = self.csv_writers.get(asset)
        if writer:
            writer.writerow(row)
            self.file_handles[asset].flush()
    
    async def _collect_all_assets(self, sequential: bool = False) -> Dict[str, Optional[Dict]]:
        """
        Collect price data from all assets.
        
        Args:
            sequential: If True, collect one asset at a time (slower but more reliable)
        """
        results = {}
        
        if sequential:
            # Collect one asset at a time (more reliable, less resource-intensive)
            for asset in self.assets:
                try:
                    data = await self._extract_price(asset)
                    results[asset] = data
                    
                    if data:
                        self.stats[asset]["collected"] += 1
                        self.stats[asset]["last_price"] = data.get('mid')
                        self.stats[asset]["errors"] = 0  # Reset error count on success
                        self._write_data(asset, data)
                    else:
                        self.stats[asset]["errors"] += 1
                        # Write forward-fill placeholder
                        self._write_missing_data(asset, self.stats[asset]["last_price"])
                        
                except Exception as e:
                    self.logger.debug(f"Error collecting {asset}: {e}")
                    self.stats[asset]["errors"] += 1
                    self._write_missing_data(asset, self.stats[asset]["last_price"])
                    results[asset] = None
        else:
            # Collect all assets in parallel (faster but more resource-intensive)
            async def collect_one(asset: str) -> tuple:
                try:
                    data = await self._extract_price(asset)
                    if data:
                        self.stats[asset]["collected"] += 1
                        self.stats[asset]["last_price"] = data.get('mid')
                        self.stats[asset]["errors"] = 0  # Reset error count on success
                        self._write_data(asset, data)
                    else:
                        self.stats[asset]["errors"] += 1
                        self._write_missing_data(asset, self.stats[asset]["last_price"])
                    return (asset, data)
                except Exception as e:
                    self.logger.debug(f"Error collecting {asset}: {e}")
                    self.stats[asset]["errors"] += 1
                    self._write_missing_data(asset, self.stats[asset]["last_price"])
                    return (asset, None)
            
            # Collect all in parallel
            tasks = [collect_one(asset) for asset in self.assets]
            collected = await asyncio.gather(*tasks, return_exceptions=True)
            
            for item in collected:
                if isinstance(item, Exception):
                    continue
                asset, data = item
                results[asset] = data
        
        return results
    
    async def _reload_page(self, asset: str):
        """Reload a page that may have become stale."""
        try:
            if asset in self.pages:
                await self.pages[asset].close()
                del self.pages[asset]
            
            await self._load_page(asset)
        except Exception as e:
            self.logger.error(f"Failed to reload page for {asset}: {e}")
    
    async def start(
        self,
        interval: float = 1.0,
        duration: Optional[float] = None,
        callback: Optional[Callable[[Dict[str, Optional[Dict]]], None]] = None,
        log_interval: int = 10
    ):
        """
        Start continuous collection.
        
        Args:
            interval: Collection interval in seconds (default: 1.0 for 1 Hz)
            duration: How long to run in seconds (None = run indefinitely)
            callback: Optional callback function called after each collection round
            log_interval: How often to log stats in seconds
        """
        self.logger.info(f"Starting continuous collection at {interval}s intervals")
        if duration:
            self.logger.info(f"Will run for {duration} seconds")
        else:
            self.logger.info("Running indefinitely (Ctrl+C to stop)")
        
        # Initialize browser and pages
        self.logger.info("Initializing Playwright browser...")
        try:
            await self._ensure_playwright()
            self.logger.info("Playwright browser initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Playwright: {e}")
            raise
        
        # Load all asset pages with delay between each to avoid overwhelming browser
        pages_loaded = 0
        for i, asset in enumerate(self.assets):
            if i > 0:
                await asyncio.sleep(1)  # Small delay between page loads
            success = await self._load_page(asset)
            if success:
                pages_loaded += 1
            else:
                self.logger.warning(f"Failed to load page for {asset}, will retry during collection")
        
        self.running = True
        start_time = time.time()
        last_log_time = start_time
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Check duration limit
                if duration and (loop_start - start_time) >= duration:
                    self.logger.info("Duration limit reached, stopping")
                    break
                
                # Collect from all assets
                results = await self._collect_all_assets(sequential=self.sequential_collection)
                
                # Call callback if provided
                if callback:
                    callback(results)
                
                # Periodic logging
                if loop_start - last_log_time >= log_interval:
                    self._log_stats()
                    last_log_time = loop_start
                
                # Check for pages that need reloading (lower threshold for faster recovery)
                for asset in self.assets:
                    # Reload if page missing or too many errors (5 consecutive)
                    if asset not in self.pages or self.stats[asset]["errors"] >= 5:
                        self.logger.warning(f"Reloading page for {asset} (errors={self.stats[asset]['errors']})")
                        await self._reload_page(asset)
                        self.stats[asset]["errors"] = 0
                
                # Calculate elapsed time and adjust sleep
                elapsed = time.time() - loop_start
                sleep_time = max(0, interval - elapsed)
                
                # If we're falling behind (elapsed > interval), log a warning
                if elapsed > interval * 1.5:  # More than 50% over
                    self.logger.warning(f"Collection falling behind: {elapsed:.2f}s elapsed (target: {interval}s)")
                
                # Sleep until next interval (but don't sleep if we're already way behind)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                elif elapsed > interval * 2:  # If we're more than 2x behind, skip sleep entirely
                    self.logger.warning("Skipping sleep due to significant delay")
                    
        except asyncio.CancelledError:
            self.logger.info("Collection cancelled")
        except Exception as e:
            self.logger.error(f"Collection error: {e}")
        finally:
            self.running = False
            self._log_stats()
    
    def _log_stats(self):
        """Log collection statistics."""
        for asset in self.assets:
            stats = self.stats[asset]
            last_price = stats['last_price'] if stats['last_price'] is not None else 0.0
            self.logger.info(
                f"{asset}: collected={stats['collected']}, errors={stats['errors']}, "
                f"last_price=${last_price:.2f}"
            )
    
    async def stop(self):
        """Stop collection and cleanup."""
        self.logger.info("Stopping collection...")
        self.running = False
        
        # Write any pending data before closing
        for asset, pending_data in self._pending_data.items():
            if pending_data:
                writer = self.csv_writers.get(asset)
                if writer:
                    src_ts = pending_data['source_timestamp']
                    rcv_ts = pending_data['received_timestamp']
                    row = {
                        'source_timestamp': src_ts.isoformat() if isinstance(src_ts, datetime) else src_ts,
                        'received_timestamp': rcv_ts.isoformat() if isinstance(rcv_ts, datetime) else rcv_ts,
                        'data_timestamp_raw': pending_data.get('data_timestamp_raw', ''),
                        'asset': pending_data['asset'],
                        'mid': pending_data.get('mid'),
                        'bid': pending_data.get('bid'),
                        'ask': pending_data.get('ask'),
                        'source': pending_data.get('source', 'unknown'),
                        'is_observed': pending_data.get('is_observed', 1),
                    }
                    writer.writerow(row)
        
        # Close file handles first
        for fh in self.file_handles.values():
            try:
                fh.flush()  # Ensure all data is written
                fh.close()
            except Exception as e:
                self.logger.debug(f"Error closing file handle: {e}")
        self.file_handles.clear()
        
        # Close browser pages (with timeout to prevent hanging)
        for asset, page in list(self.pages.items()):
            try:
                await asyncio.wait_for(page.close(), timeout=2.0)
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout closing page for {asset}")
            except Exception as e:
                self.logger.debug(f"Error closing page for {asset}: {e}")
        self.pages.clear()
        
        # Close browser (with timeout)
        if self.browser:
            try:
                await asyncio.wait_for(self.browser.close(), timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("Timeout closing browser")
            except Exception as e:
                self.logger.debug(f"Error closing browser: {e}")
            finally:
                self.browser = None
        
        # Stop Playwright (with timeout)
        if self._playwright:
            try:
                await asyncio.wait_for(self._playwright.stop(), timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("Timeout stopping Playwright")
            except Exception as e:
                self.logger.debug(f"Error stopping Playwright: {e}")
            finally:
                self._playwright = None
        
        # Small delay to ensure cleanup
        await asyncio.sleep(0.5)
        
        self.logger.info("Collection stopped and cleanup complete")


class ContinuousChainlinkCollectorSync:
    """Synchronous wrapper for ContinuousChainlinkCollector."""
    
    def __init__(
        self,
        assets: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        log_level: int = 20,
    ):
        self.collector = ContinuousChainlinkCollector(assets, output_dir, log_level)
        self._loop = None
        self._thread = None
    
    def _get_loop(self):
        """Get or create event loop."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop
    
    def start(
        self,
        interval: float = 1.0,
        duration: Optional[float] = None,
        callback: Optional[Callable] = None,
        log_interval: int = 10
    ):
        """Start collection (blocking)."""
        loop = self._get_loop()
        loop.run_until_complete(
            self.collector.start(interval, duration, callback, log_interval)
        )
    
    def start_background(
        self,
        interval: float = 1.0,
        duration: Optional[float] = None,
        callback: Optional[Callable] = None,
        log_interval: int = 10
    ):
        """Start collection in a background thread."""
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                self.collector.start(interval, duration, callback, log_interval)
            )
        
        self._thread = threading.Thread(target=run_in_thread, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop collection."""
        if self._loop:
            loop = self._get_loop()
            loop.run_until_complete(self.collector.stop())
    
    @property
    def running(self):
        return self.collector.running
    
    @property
    def stats(self):
        return self.collector.stats


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

async def main():
    """Run continuous collection for all assets."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous Chainlink data collector")
    parser.add_argument(
        "--assets", 
        type=str, 
        default="BTC,ETH,SOL,XRP",
        help="Comma-separated list of assets to collect"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Collection duration in seconds (default: indefinite)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Collection interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for CSV files"
    )
    
    args = parser.parse_args()
    
    assets = [a.strip().upper() for a in args.assets.split(",")]
    
    collector = ContinuousChainlinkCollector(
        assets=assets,
        output_dir=args.output_dir,
    )
    
    try:
        await collector.start(
            interval=args.interval,
            duration=args.duration,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        await collector.stop()


if __name__ == "__main__":
    asyncio.run(main())

