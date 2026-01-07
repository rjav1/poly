"""
High-frequency Chainlink data collector using frontend scraping.

This collector uses Playwright to scrape the Chainlink website chart,
which provides ~0.5 second resolution data (vs ~12 seconds from API).

Note: The frontend data is ~1 minute behind real-time, so collection
timing must be adjusted accordingly.
"""

import time
import asyncio
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
from pathlib import Path
import json
from dateutil import parser

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logging import setup_logger
from config.settings import CHAINLINK, STORAGE


class ChainlinkFrontendCollector:
    """
    Collects Chainlink BTC/USD price data by scraping the frontend chart.
    
    Uses Playwright to:
    1. Load the Chainlink data stream page
    2. Extract chart data via JavaScript
    3. Collect at ~0.5 second intervals (2 Hz)
    
    Note: Frontend data is ~1 minute behind real-time.
    """
    
    def __init__(self, feed_id: Optional[str] = None, log_level: int = 20):
        """
        Initialize frontend collector.
        
        Args:
            feed_id: Chainlink feed ID (for URL construction)
            log_level: Logging level
        """
        self.feed_id = feed_id or CHAINLINK.feed_id
        self.chart_url = f"https://data.chain.link/streams/btc-usd-cexprice-streams"
        self.price_divisor = CHAINLINK.price_divisor
        
        self.logger = setup_logger("chainlink_frontend", level=log_level)
        self.logger.info(f"ChainlinkFrontendCollector initialized")
        
        self.browser = None
        self.page = None
        self._playwright_installed = False
        
    async def _ensure_playwright(self):
        """Ensure Playwright is installed and browser is ready."""
        if self._playwright_installed and self.browser:
            return
        
        try:
            from playwright.async_api import async_playwright
            self._playwright = async_playwright
            self._playwright_installed = True
        except ImportError:
            raise ImportError(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )
        
        if not self.browser:
            playwright = await self._playwright().start()
            self.browser = await playwright.chromium.launch(headless=True)
            self.page = await self.browser.new_page()
            
            # Set viewport
            await self.page.set_viewport_size({"width": 1920, "height": 1080})
            
            self.logger.info("Playwright browser launched")
    
    async def _load_chart_page(self) -> bool:
        """Load the Chainlink chart page and wait for it to be ready."""
        try:
            self.logger.info(f"Loading chart page: {self.chart_url}")
            await self.page.goto(self.chart_url, wait_until="networkidle", timeout=30000)
            
            # Wait for page to be fully interactive
            await self.page.wait_for_load_state("domcontentloaded")
            await asyncio.sleep(2)  # Give chart time to render
            
            # Wait for price data to appear in DOM
            # Look for "Mid-price" text which indicates the page is ready
            max_wait = 10
            price_found = False
            for i in range(max_wait):
                try:
                    price_text = await self.page.evaluate("""
                        () => {
                            const bodyText = document.body.textContent || '';
                            return bodyText.includes('Mid-price') || bodyText.includes('Mid price');
                        }
                    """)
                    if price_text:
                        price_found = True
                        break
                    await asyncio.sleep(0.5)
                except:
                    await asyncio.sleep(0.5)
            
            if not price_found:
                self.logger.warning("Price text not found in DOM, but continuing...")
            
            # Try to find chart element (SVG, canvas, or chart container)
            chart_selectors = [
                'svg',
                'canvas',
                '[class*="chart"]',
                '[class*="graph"]',
                '[data-testid*="chart"]'
            ]
            
            chart_found = False
            for selector in chart_selectors:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        chart_found = True
                        self.logger.info(f"Found chart element: {selector}")
                        break
                except:
                    continue
            
            if not chart_found:
                self.logger.warning("Chart element not found, but continuing...")
            
            self.logger.info("Chart page loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load chart page: {e}")
            return False
    
    async def _extract_chart_data(self) -> Optional[Dict]:
        """
        Extract price data from the chart using JavaScript.
        
        This method searches the entire page body text for price patterns.
        This is the most reliable method as it doesn't depend on specific DOM structure.
        """
        try:
            chart_data = await self.page.evaluate("""
                () => {
                    // Get all text from the page body
                    const bodyText = document.body.textContent || document.body.innerText || '';
                    
                    let price = null;
                    let bid = null;
                    let ask = null;
                    
                    // Method 1: Look for "Mid-price" followed by price
                    // Format examples: "Mid-price$91,363.17" or "Mid-price: $91,363.17"
                    const midMatch = bodyText.match(/Mid-price[\\s:]*\\$?([\\d,]+(?:\\.[\\d]+)?)/i);
                    if (midMatch) {
                        price = parseFloat(midMatch[1].replace(/,/g, ''));
                    }
                    
                    // Extract Bid price
                    const bidMatch = bodyText.match(/Bid[\\s:]*price[\\s:]*\\$?([\\d,]+(?:\\.[\\d]+)?)/i);
                    if (bidMatch) {
                        bid = parseFloat(bidMatch[1].replace(/,/g, ''));
                    }
                    
                    // Extract Ask price
                    const askMatch = bodyText.match(/Ask[\\s:]*price[\\s:]*\\$?([\\d,]+(?:\\.[\\d]+)?)/i);
                    if (askMatch) {
                        ask = parseFloat(askMatch[1].replace(/,/g, ''));
                    }
                    
                    // Method 2: If we didn't find mid-price, look for any large BTC price
                    // BTC prices are typically 5-6 digits (90,000 - 100,000 range)
                    if (!price) {
                        // Find all price-like numbers in the text
                        const priceMatches = bodyText.match(/\\$([\\d,]{5,}(?:\\.[\\d]+)?)/g);
                        if (priceMatches && priceMatches.length > 0) {
                            // Try each match - look for reasonable BTC price
                            for (let match of priceMatches) {
                                const numStr = match.replace(/[$,]/g, '');
                                const num = parseFloat(numStr);
                                // BTC price range: $10,000 - $1,000,000
                                if (num >= 10000 && num <= 1000000) {
                                    price = num;
                                    break;
                                }
                            }
                        }
                    }
                    
                    return {
                        price: price,
                        mid: price,
                        bid: bid,
                        ask: ask,
                        timestamp: new Date().toISOString(),
                        found: price !== null && price > 0
                    };
                }
            """)
            
            if chart_data and chart_data.get('found') and chart_data.get('price'):
                return {
                    'timestamp': pd.to_datetime(chart_data['timestamp']),
                    'price': chart_data['price'],
                    'mid': chart_data.get('mid') or chart_data['price'],
                    'bid': chart_data.get('bid'),
                    'ask': chart_data.get('ask'),
                    'collected_at': datetime.now(timezone.utc),
                    'source': 'frontend_dom'
                }
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error extracting chart data: {e}")
            return None
    
    async def _setup_request_interception(self):
        """Set up request interception to capture API responses."""
        if not hasattr(self, '_latest_api_data'):
            self._latest_api_data = []
        
        async def handle_response(response):
            """Capture API responses containing price data."""
            url = response.url
            
            # Check if this is the Chainlink API endpoint
            if 'query-timescale' in url or 'api/query' in url:
                try:
                    data = await response.json()
                    # Store the latest response for later use
                    self._latest_api_data.append({
                        'timestamp': datetime.now(timezone.utc),
                        'data': data
                    })
                    # Keep only last 10 responses
                    if len(self._latest_api_data) > 10:
                        self._latest_api_data.pop(0)
                except Exception as e:
                    self.logger.debug(f"Error parsing API response: {e}")
        
        self.page.on("response", handle_response)
    
    async def _extract_from_api_responses(self) -> Optional[Dict]:
        """Extract price from intercepted API responses."""
        if not hasattr(self, '_latest_api_data') or not self._latest_api_data:
            return None
        
        # Get the most recent API response
        latest = self._latest_api_data[-1]
        data = latest['data']
        
        try:
            # Parse GraphQL response structure
            # LIVE_STREAM_REPORTS_QUERY response format
            nodes = data.get("data", {}).get("liveStreamReports", {}).get("nodes", [])
            if nodes:
                node = nodes[0]  # Most recent report
                price_str = node.get("price", "0")
                bid_str = node.get("bid", "0")
                ask_str = node.get("ask", "0")
                
                price = int(price_str) / self.price_divisor if price_str else None
                bid = int(bid_str) / self.price_divisor if bid_str else None
                ask = int(ask_str) / self.price_divisor if ask_str else None
                
                timestamp_str = node.get("validFromTimestamp")
                timestamp = pd.to_datetime(timestamp_str) if timestamp_str else None
                
                return {
                    'timestamp': timestamp,
                    'price': price,
                    'bid': bid,
                    'ask': ask,
                    'mid': price,
                    'collected_at': datetime.now(timezone.utc),
                    'source': 'api_intercept'
                }
        except Exception as e:
            self.logger.debug(f"Error parsing API response: {e}")
        
        return None
    
    async def _hover_and_extract_tooltip(self) -> Optional[Dict]:
        """
        Hover over the chart and extract tooltip data.
        
        The tooltip shows: timestamp, bid, mid-price, ask
        """
        try:
            # Find chart element to hover over
            # Try multiple selectors - the chart might be in different places
            chart_selectors = [
                'svg[width]',  # SVG with width attribute (actual chart)
                'canvas',
                '[class*="chart"]',
                '[class*="graph"]',
                '[class*="Chart"]',
                'svg:not([width="32"]):not([width="24"])'  # Exclude small icon SVGs
            ]
            chart_element = None
            
            for selector in chart_selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    # Find the largest SVG/canvas (likely the chart)
                    for element in elements:
                        box = await element.bounding_box()
                        if box and box['width'] > 100 and box['height'] > 100:  # Must be reasonably sized
                            chart_element = element
                            break
                    if chart_element:
                        break
                except:
                    continue
            
            if not chart_element:
                self.logger.warning("Chart element not found for hovering - trying to find by content")
                # Fallback: look for section with price data
                section = await self.page.query_selector('section')
                if section:
                    box = await section.bounding_box()
                    if box and box['width'] > 100:
                        chart_element = section
                        self.logger.info("Using section element as chart")
            
            if not chart_element:
                self.logger.warning("Chart element not found for hovering")
                return None
            
            # Scroll chart into view first
            await chart_element.scroll_into_view_if_needed()
            await asyncio.sleep(0.3)
            
            # Get chart bounding box
            box = await chart_element.bounding_box()
            if not box:
                return None
            
            # Hover near the right edge of the chart (most recent data)
            # Use ~80% across the chart width
            hover_x = box['x'] + box['width'] * 0.8
            hover_y = box['y'] + box['height'] * 0.5  # Middle of chart
            
            await self.page.mouse.move(hover_x, hover_y)
            await asyncio.sleep(0.8)  # Wait longer for tooltip to appear and update
            
            # Take a screenshot for debugging (optional, can remove later)
            # await self.page.screenshot(path="debug_tooltip.png")
            
            # Extract tooltip data - try multiple methods
            tooltip_data = await self.page.evaluate("""
                () => {
                    // Method 1: Look for tooltip with specific text patterns
                    const tooltipSelectors = [
                        '[class*="tooltip"]',
                        '[class*="Tooltip"]',
                        '[role="tooltip"]',
                        '[data-testid*="tooltip"]',
                        '[class*="popover"]',
                        '[class*="overlay"]',
                        '[class*="hover"]',
                        'div[style*="position"][style*="absolute"]'  // Common tooltip pattern
                    ];
                    
                    let tooltip = null;
                    let tooltipText = '';
                    
                    // Try selectors first
                    for (const selector of tooltipSelectors) {
                        const elements = document.querySelectorAll(selector);
                        for (let el of elements) {
                            const text = el.textContent || el.innerText || '';
                            // Check if it looks like our tooltip (has price and timestamp)
                            if (text.includes('$') && (text.includes('Bid') || text.includes('Mid') || text.includes('Ask'))) {
                                tooltip = el;
                                tooltipText = text;
                                break;
                            }
                        }
                        if (tooltip) break;
                    }
                    
                    // Method 2: Find all elements with high z-index and check content
                    if (!tooltip) {
                        const allElements = document.querySelectorAll('*');
                        for (let el of allElements) {
                            const style = window.getComputedStyle(el);
                            const zIndex = parseInt(style.zIndex) || 0;
                            const text = el.textContent || el.innerText || '';
                            
                            // High z-index + contains price + contains date/time pattern
                            if (zIndex > 100 && text.includes('$') && 
                                (text.match(/Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec/i) || 
                                 text.includes('Bid') || text.includes('Mid') || text.includes('Ask'))) {
                                tooltip = el;
                                tooltipText = text;
                                break;
                            }
                        }
                    }
                    
                    // Method 3: Look for elements containing the specific format from screenshot
                    if (!tooltip) {
                        const allDivs = document.querySelectorAll('div, section, span');
                        for (let el of allDivs) {
                            const text = el.textContent || el.innerText || '';
                            // Look for pattern: "Jan 3, 2026 7:57:07 PM" + "Bid:" + "Mid-price:" + "Ask:"
                            // OR just "Bid:" + "Mid-price:" + "Ask:" (tooltip might not have timestamp in same element)
                            if ((text.includes('Mid-price') || text.includes('Mid price')) && 
                                (text.includes('Bid') || text.includes('Bid:')) && 
                                (text.includes('Ask') || text.includes('Ask:'))) {
                                const style = window.getComputedStyle(el);
                                // Check if it's visible and positioned (likely a tooltip)
                                if (style.display !== 'none' && style.visibility !== 'hidden') {
                                    tooltip = el;
                                    tooltipText = text;
                                    break;
                                }
                            }
                        }
                    }
                    
                    // Method 4: Look for any element that appeared recently (tooltip might be dynamically created)
                    // Check for elements with pointer-events: none (common for tooltips)
                    if (!tooltip) {
                        const allElements = document.querySelectorAll('*');
                        for (let el of allElements) {
                            const style = window.getComputedStyle(el);
                            const text = el.textContent || el.innerText || '';
                            if (text.includes('$') && (text.includes('Bid') || text.includes('Mid') || text.includes('Ask'))) {
                                // Check if it looks like a tooltip (high z-index, absolute/fixed position)
                                if ((style.position === 'absolute' || style.position === 'fixed') && 
                                    parseInt(style.zIndex) > 100) {
                                    tooltip = el;
                                    tooltipText = text;
                                    break;
                                }
                            }
                        }
                    }
                    
                    if (!tooltip || !tooltipText) {
                        return {
                            error: 'tooltip_not_found',
                            debug: {
                                bodyText: document.body.textContent.substring(0, 500),
                                highZIndex: Array.from(document.querySelectorAll('*'))
                                    .filter(el => {
                                        const z = parseInt(window.getComputedStyle(el).zIndex) || 0;
                                        return z > 100;
                                    })
                                    .map(el => ({
                                        tag: el.tagName,
                                        zIndex: window.getComputedStyle(el).zIndex,
                                        text: (el.textContent || '').substring(0, 100)
                                    }))
                                    .slice(0, 5)
                            }
                        };
                    }
                    
                    // Parse timestamp - format: "Jan 3, 2026 7:57:07 PM"
                    let timestampMatch = tooltipText.match(/(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\s+\\d+,\\s+\\d{4}\\s+\\d{1,2}:\\d{2}:\\d{2}\\s+(AM|PM)/i);
                    
                    // Also try format without seconds: "Jan 3, 2026 7:57 PM"
                    if (!timestampMatch) {
                        timestampMatch = tooltipText.match(/(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\s+\\d+,\\s+\\d{4}\\s+\\d{1,2}:\\d{2}\\s+(AM|PM)/i);
                    }
                    
                    // Parse prices - look for "Bid: $XX,XXX.XX", "Mid-price: $XX,XXX.XX", "Ask: $XX,XXX.XX"
                    const bidMatch = tooltipText.match(/Bid:\\s*\\$?([\\d,]+(?:\\.[\\d]+)?)/i);
                    const midMatch = tooltipText.match(/Mid-price:\\s*\\$?([\\d,]+(?:\\.[\\d]+)?)/i);
                    const askMatch = tooltipText.match(/Ask:\\s*\\$?([\\d,]+(?:\\.[\\d]+)?)/i);
                    
                    return {
                        timestamp: timestampMatch ? timestampMatch[0] : null,
                        bid: bidMatch ? parseFloat(bidMatch[1].replace(/,/g, '')) : null,
                        mid: midMatch ? parseFloat(midMatch[1].replace(/,/g, '')) : null,
                        ask: askMatch ? parseFloat(askMatch[1].replace(/,/g, '')) : null,
                        rawText: tooltipText.substring(0, 300)  // First 300 chars for debugging
                    };
                }
            """)
            
            # Check for errors
            if tooltip_data and 'error' in tooltip_data:
                self.logger.warning(f"Tooltip extraction error: {tooltip_data.get('error')}")
                if 'debug' in tooltip_data:
                    debug_info = tooltip_data['debug']
                    self.logger.debug(f"Debug: bodyText length={len(debug_info.get('bodyText', ''))}")
                    self.logger.debug(f"Debug: highZIndex elements={len(debug_info.get('highZIndex', []))}")
                return None
            
            if not tooltip_data or not tooltip_data.get('mid'):
                if tooltip_data:
                    if tooltip_data.get('rawText'):
                        self.logger.warning(f"Found tooltip but couldn't parse mid price. Raw text: {tooltip_data['rawText'][:200]}")
                    else:
                        self.logger.warning("Tooltip data found but no rawText or mid price")
                else:
                    self.logger.warning("No tooltip data returned from page.evaluate")
                return None
            
            # Parse timestamp
            timestamp = None
            if tooltip_data.get('timestamp'):
                try:
                    # Parse format: "Jan 3, 2026 7:57:07 PM"
                    timestamp = parser.parse(tooltip_data['timestamp'])
                    # Assume UTC if no timezone specified
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                except Exception as e:
                    self.logger.debug(f"Failed to parse timestamp '{tooltip_data.get('timestamp')}': {e}")
                    timestamp = datetime.now(timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)
            
            return {
                'timestamp': pd.to_datetime(timestamp),
                'price': tooltip_data['mid'],
                'bid': tooltip_data['bid'],
                'ask': tooltip_data['ask'],
                'mid': tooltip_data['mid'],
                'collected_at': datetime.now(timezone.utc),
                'source': 'tooltip_hover'
            }
            
        except Exception as e:
            self.logger.debug(f"Error extracting tooltip: {e}")
            return None
    
    async def get_latest_price(self) -> Optional[Dict]:
        """
        Get the latest price from the frontend.
        
        Returns:
            Dictionary with price data or None if failed
        """
        await self._ensure_playwright()
        
        # Ensure page is loaded
        if not self.page:
            if not await self._load_chart_page():
                return None
        else:
            # Check if page is still valid and has content
            try:
                ready_state = await self.page.evaluate("() => document.readyState")
                if ready_state != "complete":
                    # Wait a bit for page to finish loading
                    await asyncio.sleep(0.5)
                
                # Check if price data is available
                has_price = await self.page.evaluate("""
                    () => {
                        const bodyText = document.body.textContent || '';
                        return bodyText.includes('Mid-price') || bodyText.includes('Mid price');
                    }
                """)
                
                if not has_price:
                    # Page might have lost content, reload
                    self.logger.debug("Price data not found, reloading page...")
                    if not await self._load_chart_page():
                        return None
            except Exception as e:
                # Page is closed or invalid, reload
                self.logger.debug(f"Page invalid ({e}), reloading...")
                if not await self._load_chart_page():
                    return None
        
        # Method 1: DOM extraction (fastest, most reliable)
        price_data = await self._extract_chart_data()
        if price_data:
            return price_data
        
        # Method 2: Hover and extract tooltip (more detailed, includes bid/ask)
        # Only try if DOM extraction failed
        price_data = await self._hover_and_extract_tooltip()
        if price_data:
            return price_data
        
        # Method 3: Try API interception (fallback)
        price_data = await self._extract_from_api_responses()
        if price_data:
            return price_data
        
        self.logger.warning("Failed to collect price data - all methods failed")
        return None
    
    async def collect_continuous(
        self, 
        duration_seconds: float,
        interval: float = 0.5,
        callback: Optional[callable] = None
    ) -> List[Dict]:
        """
        Collect data continuously at specified interval using hover method.
        
        The mouse stays hovered over the chart, and the tooltip updates automatically.
        
        Args:
            duration_seconds: How long to collect (seconds)
            interval: Collection interval (seconds, default 0.5 for 2 Hz)
            callback: Optional callback function(data) called after each collection
            
        Returns:
            List of collected data dictionaries
        """
        await self._ensure_playwright()
        
        if not await self._load_chart_page():
            self.logger.error("Failed to load chart page")
            return []
        
        # Find chart and position mouse for hovering
        chart_selectors = ['svg', 'canvas', '[class*="chart"]', '[class*="graph"]']
        chart_element = None
        
        for selector in chart_selectors:
            try:
                element = await self.page.query_selector(selector)
                if element:
                    chart_element = element
                    break
            except:
                continue
        
        if not chart_element:
            self.logger.error("Chart element not found")
            return []
        
        # Scroll chart into view first
        await chart_element.scroll_into_view_if_needed()
        await asyncio.sleep(0.5)
        
        # Get chart bounding box
        box = await chart_element.bounding_box()
        if not box:
            self.logger.error("Could not get chart bounding box")
            return []
        
        # Position mouse near right edge (most recent data)
        # Use viewport coordinates
        hover_x = box['x'] + box['width'] * 0.8
        hover_y = box['y'] + box['height'] * 0.5
        
        self.logger.info(f"Chart box: x={box['x']:.0f}, y={box['y']:.0f}, w={box['width']:.0f}, h={box['height']:.0f}")
        self.logger.info(f"Hover position: x={hover_x:.0f}, y={hover_y:.0f}")
        
        # Move mouse to chart
        await self.page.mouse.move(hover_x, hover_y)
        await asyncio.sleep(1.0)  # Wait longer for tooltip to appear and stabilize
        
        collected_data = []
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        self.logger.info(f"Starting continuous collection for {duration_seconds}s at {interval}s intervals")
        
        while time.time() < end_time:
            loop_start = time.time()
            
            # Use get_latest_price() which is the most reliable method
            # It uses DOM extraction first, then falls back to other methods
            data = await self.get_latest_price()
            
            if data:
                collected_data.append(data)
                if callback:
                    callback(data)
                self.logger.debug(
                    f"Collected: price=${data.get('mid', 0):.2f}, "
                    f"source={data.get('source', 'unknown')}"
                )
            else:
                self.logger.warning("Failed to collect price data")
            
            # Sleep until next interval
            elapsed = time.time() - loop_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.logger.info(f"Collection complete: {len(collected_data)} data points")
        return collected_data
    
    async def close(self):
        """Close browser and cleanup."""
        if self.browser:
            await self.browser.close()
            self.browser = None
            self.page = None
            self.logger.info("Browser closed")


# Synchronous wrapper for easier integration
class ChainlinkFrontendCollectorSync:
    """Synchronous wrapper around async frontend collector."""
    
    def __init__(self, feed_id: Optional[str] = None, log_level: int = 20):
        self.collector = ChainlinkFrontendCollector(feed_id, log_level)
        self._loop = None
    
    def _get_loop(self):
        """Get or create event loop."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop
    
    def get_latest_price(self) -> Optional[Dict]:
        """Get latest price (synchronous)."""
        loop = self._get_loop()
        return loop.run_until_complete(self.collector.get_latest_price())
    
    def collect_continuous(
        self,
        duration_seconds: float,
        interval: float = 0.5,
        callback: Optional[callable] = None
    ) -> List[Dict]:
        """Collect continuously (synchronous)."""
        loop = self._get_loop()
        return loop.run_until_complete(
            self.collector.collect_continuous(duration_seconds, interval, callback)
        )
    
    def close(self):
        """Close browser."""
        if self._loop:
            loop = self._get_loop()
            loop.run_until_complete(self.collector.close())
            self._loop.close()
            self._loop = None

