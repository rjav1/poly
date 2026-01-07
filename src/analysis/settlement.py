"""
Settlement Rule Reconstruction for Polymarket BTC 15-min Markets.

This module helps identify:
1. Which exact Chainlink price is used at start/end boundaries
2. The precise settlement rule (last report before boundary? first after?)
3. Verification against resolved markets

Settlement Rules (from Polymarket):
- UP: End price >= Start price
- DOWN: End price < Start price
- Resolution source: https://data.chain.link/streams/btc-usd
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logging import setup_logger


class SettlementAnalyzer:
    """
    Analyzes Chainlink price data to reconstruct settlement boundaries.
    
    Key questions to answer:
    1. Which price report is used at market start?
    2. Which price report is used at market end?
    3. What boundary rule reproduces actual resolutions?
    """
    
    BOUNDARY_RULES = {
        "last_before": "Last report strictly before boundary",
        "first_after": "First report at or after boundary",
        "last_at_or_before": "Last report at or before boundary",
        "closest": "Report closest to boundary",
    }
    
    def __init__(self):
        self.logger = setup_logger("settlement")
    
    def find_boundary_price(
        self,
        df: pd.DataFrame,
        boundary_time: datetime,
        rule: str = "last_before"
    ) -> Optional[Dict]:
        """
        Find the price at a boundary using a specific rule.
        
        Args:
            df: DataFrame with 'timestamp' and 'price' columns
            boundary_time: The boundary time (market start or end)
            rule: Which price to use ("last_before", "first_after", etc.)
            
        Returns:
            Dictionary with price and timestamp, or None if not found
        """
        if len(df) == 0:
            return None
        
        df = df.sort_values("timestamp").copy()
        
        # Ensure boundary_time is timezone-aware
        if boundary_time.tzinfo is None:
            boundary_time = boundary_time.replace(tzinfo=timezone.utc)
        
        if rule == "last_before":
            # Last report strictly before boundary
            before = df[df["timestamp"] < boundary_time]
            if len(before) > 0:
                row = before.iloc[-1]
                return {"price": row["price"], "timestamp": row["timestamp"], "rule": rule}
                
        elif rule == "first_after":
            # First report at or after boundary
            after = df[df["timestamp"] >= boundary_time]
            if len(after) > 0:
                row = after.iloc[0]
                return {"price": row["price"], "timestamp": row["timestamp"], "rule": rule}
                
        elif rule == "last_at_or_before":
            # Last report at or before boundary
            at_or_before = df[df["timestamp"] <= boundary_time]
            if len(at_or_before) > 0:
                row = at_or_before.iloc[-1]
                return {"price": row["price"], "timestamp": row["timestamp"], "rule": rule}
                
        elif rule == "closest":
            # Report closest to boundary (by absolute time difference)
            df["time_diff"] = abs((df["timestamp"] - boundary_time).dt.total_seconds())
            idx = df["time_diff"].idxmin()
            row = df.loc[idx]
            return {"price": row["price"], "timestamp": row["timestamp"], "rule": rule}
        
        return None
    
    def compute_settlement(
        self,
        df: pd.DataFrame,
        start_time: datetime,
        end_time: datetime,
        start_rule: str = "last_before",
        end_rule: str = "last_before"
    ) -> Dict:
        """
        Compute the settlement outcome for a market.
        
        Args:
            df: DataFrame with Chainlink price data
            start_time: Market start time
            end_time: Market end time
            start_rule: Rule for finding start price
            end_rule: Rule for finding end price
            
        Returns:
            Dictionary with S0, ST, outcome, and metadata
        """
        start_price_info = self.find_boundary_price(df, start_time, start_rule)
        end_price_info = self.find_boundary_price(df, end_time, end_rule)
        
        if not start_price_info or not end_price_info:
            return {
                "error": "Could not find boundary prices",
                "start_price_info": start_price_info,
                "end_price_info": end_price_info,
            }
        
        S0 = start_price_info["price"]
        ST = end_price_info["price"]
        
        outcome = "UP" if ST >= S0 else "DOWN"
        
        return {
            "S0": S0,
            "ST": ST,
            "outcome": outcome,
            "start_time": start_time,
            "end_time": end_time,
            "start_rule": start_rule,
            "end_rule": end_rule,
            "start_timestamp": start_price_info["timestamp"],
            "end_timestamp": end_price_info["timestamp"],
            "start_lag_seconds": (start_time - start_price_info["timestamp"]).total_seconds(),
            "end_lag_seconds": (end_time - end_price_info["timestamp"]).total_seconds(),
            "price_change": ST - S0,
            "price_change_pct": (ST - S0) / S0 * 100 if S0 > 0 else 0,
        }
    
    def test_boundary_rules(
        self,
        df: pd.DataFrame,
        start_time: datetime,
        end_time: datetime,
        actual_outcome: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Test all combinations of boundary rules and compare to actual outcome.
        
        Args:
            df: DataFrame with Chainlink price data
            start_time: Market start time
            end_time: Market end time
            actual_outcome: Actual resolution ("UP" or "DOWN"), if known
            
        Returns:
            DataFrame with results for each rule combination
        """
        results = []
        
        for start_rule in self.BOUNDARY_RULES.keys():
            for end_rule in self.BOUNDARY_RULES.keys():
                settlement = self.compute_settlement(
                    df, start_time, end_time, start_rule, end_rule
                )
                
                if "error" not in settlement:
                    match = None
                    if actual_outcome:
                        match = settlement["outcome"] == actual_outcome.upper()
                    
                    results.append({
                        "start_rule": start_rule,
                        "end_rule": end_rule,
                        "S0": settlement["S0"],
                        "ST": settlement["ST"],
                        "predicted_outcome": settlement["outcome"],
                        "actual_outcome": actual_outcome,
                        "match": match,
                        "price_change": settlement["price_change"],
                        "start_lag_sec": settlement["start_lag_seconds"],
                        "end_lag_sec": settlement["end_lag_seconds"],
                    })
        
        return pd.DataFrame(results)
    
    def analyze_market(
        self,
        df: pd.DataFrame,
        market_slug: str,
        actual_outcome: Optional[str] = None
    ) -> Dict:
        """
        Analyze a market given its slug and Chainlink data.
        
        Args:
            df: DataFrame with Chainlink price data
            market_slug: Market slug (e.g., "btc-updown-15m-1767483000")
            actual_outcome: Actual resolution if known
            
        Returns:
            Analysis results
        """
        # Extract timestamp from slug
        try:
            timestamp = int(market_slug.split("-")[-1])
            start_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            end_time = start_time + timedelta(minutes=15)
        except (ValueError, IndexError) as e:
            return {"error": f"Could not parse market slug: {e}"}
        
        self.logger.info(f"Analyzing market: {market_slug}")
        self.logger.info(f"  Start: {start_time.isoformat()}")
        self.logger.info(f"  End: {end_time.isoformat()}")
        
        # Test all boundary rules
        rules_df = self.test_boundary_rules(df, start_time, end_time, actual_outcome)
        
        # Find best rule (if actual outcome known)
        best_rule = None
        if actual_outcome and len(rules_df) > 0:
            matching = rules_df[rules_df["match"] == True]
            if len(matching) > 0:
                best_rule = matching.iloc[0]
                self.logger.info(f"  Best rule: start={best_rule['start_rule']}, end={best_rule['end_rule']}")
        
        return {
            "market_slug": market_slug,
            "start_time": start_time,
            "end_time": end_time,
            "actual_outcome": actual_outcome,
            "rules_tested": rules_df,
            "best_rule": best_rule,
            "data_range": {
                "min": df["timestamp"].min() if len(df) > 0 else None,
                "max": df["timestamp"].max() if len(df) > 0 else None,
                "count": len(df),
            }
        }


def parse_market_times(market_slug: str) -> Tuple[datetime, datetime]:
    """
    Parse market start and end times from slug.
    
    Args:
        market_slug: e.g., "btc-updown-15m-1767483000"
        
    Returns:
        Tuple of (start_time, end_time) in UTC
    """
    timestamp = int(market_slug.split("-")[-1])
    start_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    end_time = start_time + timedelta(minutes=15)
    return start_time, end_time

