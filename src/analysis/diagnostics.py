"""
Diagnostic tools for analyzing Chainlink vs Polymarket data.

Provides:
1. Time-series merging and synchronization
2. Lag analysis (oracle_age, market_age)
3. Correlation analysis
4. Visualization-ready data preparation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logging import setup_logger


class DiagnosticsAnalyzer:
    """
    Analyzes the relationship between Chainlink and Polymarket data.
    
    Key analyses:
    1. Time synchronization and lag
    2. Price change vs probability change correlation
    3. Oracle update patterns
    """
    
    def __init__(self):
        self.logger = setup_logger("diagnostics")
    
    def merge_datasets(
        self,
        df_chainlink: pd.DataFrame,
        df_polymarket: pd.DataFrame,
        method: str = "synchronized",
        max_time_diff_seconds: float = 0.5
    ) -> pd.DataFrame:
        """
        Merge Chainlink and Polymarket data into a single time-synced dataset.
        
        Only keeps observations where both were collected at the same timestamp
        (within max_time_diff_seconds) to ensure valid analysis.
        
        Args:
            df_chainlink: Chainlink data with 'timestamp' and 'collected_at' columns
            df_polymarket: Polymarket data with 'collected_at' column
            method: Merge method ("synchronized", "nearest", "asof_backward")
            max_time_diff_seconds: Maximum time difference for synchronized merge (default 0.5s)
            
        Returns:
            Merged DataFrame with all columns
        """
        if len(df_chainlink) == 0 or len(df_polymarket) == 0:
            self.logger.warning("Cannot merge empty datasets")
            return pd.DataFrame()
        
        # Prepare Chainlink data
        df_cl = df_chainlink.copy()
        df_cl = df_cl.sort_values("collected_at")
        df_cl["timestamp"] = pd.to_datetime(df_cl["timestamp"])
        df_cl["collected_at"] = pd.to_datetime(df_cl["collected_at"])
        
        # Rename Chainlink columns to avoid conflicts
        df_cl = df_cl.rename(columns={
            "price": "cl_price",
            "bid": "cl_bid",
            "ask": "cl_ask",
            "mid": "cl_mid",
        })
        
        # Prepare Polymarket data
        df_pm = df_polymarket.copy()
        df_pm = df_pm.sort_values("collected_at")
        df_pm["collected_at"] = pd.to_datetime(df_pm["collected_at"])
        
        # Rename Polymarket columns
        df_pm = df_pm.rename(columns={
            "up_mid": "pm_up_mid",
            "down_mid": "pm_down_mid",
            "up_best_bid": "pm_up_bid",
            "up_best_ask": "pm_up_ask",
            "down_best_bid": "pm_down_bid",
            "down_best_ask": "pm_down_ask",
        })
        
        if method == "synchronized":
            # Only merge observations collected at the same time (within tolerance)
            # This ensures valid analysis - both data points represent the same moment
            merged_rows = []
            
            # Round to nearest second for matching
            df_cl["collected_at_rounded"] = df_cl["collected_at"].dt.round("1s")
            df_pm["collected_at_rounded"] = df_pm["collected_at"].dt.round("1s")
            
            # Create lookup for Chainlink data by rounded collected_at
            cl_lookup = {}
            for _, cl_row in df_cl.iterrows():
                rounded_time = cl_row["collected_at_rounded"]
                if rounded_time not in cl_lookup:
                    cl_lookup[rounded_time] = []
                cl_lookup[rounded_time].append(cl_row)
            
            # Match Polymarket observations to Chainlink
            for _, pm_row in df_pm.iterrows():
                pm_time = pm_row["collected_at"]
                pm_time_rounded = pm_row["collected_at_rounded"]
                
                # Find matching Chainlink observation(s) at same rounded time
                if pm_time_rounded in cl_lookup:
                    # Use the closest one by actual time difference
                    best_cl = None
                    best_diff = float('inf')
                    
                    for cl_row in cl_lookup[pm_time_rounded]:
                        time_diff = abs((pm_time - cl_row["collected_at"]).total_seconds())
                        if time_diff < best_diff and time_diff <= max_time_diff_seconds:
                            best_diff = time_diff
                            best_cl = cl_row
                    
                    if best_cl is not None:
                        merged_row = {
                            "time": pm_time,  # Use Polymarket time as reference
                            "cl_timestamp": best_cl["timestamp"],  # Chainlink publish time
                            "cl_collected_at": best_cl["collected_at"],  # When we collected Chainlink
                            "cl_price": best_cl["cl_price"],
                            "cl_bid": best_cl.get("cl_bid"),
                            "cl_ask": best_cl.get("cl_ask"),
                            "pm_up_mid": pm_row.get("pm_up_mid"),
                            "pm_down_mid": pm_row.get("pm_down_mid"),
                            "pm_up_bid": pm_row.get("pm_up_bid"),
                            "pm_up_ask": pm_row.get("pm_up_ask"),
                            "pm_down_bid": pm_row.get("pm_down_bid"),
                            "pm_down_ask": pm_row.get("pm_down_ask"),
                            "oracle_age": (pm_time - best_cl["timestamp"]).total_seconds(),
                            "collection_time_diff": best_diff,  # How close the collection times were
                        }
                        merged_rows.append(merged_row)
            
            merged = pd.DataFrame(merged_rows)
            self.logger.info(f"Synchronized merge: {len(merged)} matched pairs from {len(df_cl)} Chainlink + {len(df_pm)} Polymarket")
            
        elif method == "nearest":
            # For each Polymarket observation, find the LAST Chainlink observation
            # (oracle_age = how stale is the price data, should always be >= 0)
            merged_rows = []
            
            for _, pm_row in df_pm.iterrows():
                pm_time = pm_row["collected_at"]
                
                # Find last Chainlink update BEFORE or AT this time
                cl_before = df_cl[df_cl["timestamp"] <= pm_time]
                
                if len(cl_before) > 0:
                    cl_row = cl_before.iloc[-1]  # Last one before/at this time
                    
                    merged_row = {
                        "time": pm_time,
                        "cl_timestamp": cl_row["timestamp"],
                        "cl_collected_at": cl_row["collected_at"],
                        "cl_price": cl_row["cl_price"],
                        "cl_bid": cl_row.get("cl_bid"),
                        "cl_ask": cl_row.get("cl_ask"),
                        "pm_up_mid": pm_row.get("pm_up_mid"),
                        "pm_down_mid": pm_row.get("pm_down_mid"),
                        "pm_up_bid": pm_row.get("pm_up_bid"),
                        "pm_up_ask": pm_row.get("pm_up_ask"),
                        "oracle_age": (pm_time - cl_row["timestamp"]).total_seconds(),
                    }
                    merged_rows.append(merged_row)
            
            merged = pd.DataFrame(merged_rows)
            
        elif method == "asof_backward":
            # Use pandas merge_asof (backward looking)
            df_cl_indexed = df_cl.set_index("collected_at")
            df_pm_indexed = df_pm.set_index("collected_at")
            
            merged = pd.merge_asof(
                df_pm_indexed.reset_index(),
                df_cl_indexed.reset_index(),
                left_on="collected_at",
                right_on="collected_at",
                direction="backward",
                tolerance=pd.Timedelta(seconds=max_time_diff_seconds)
            )
            merged["time"] = merged["collected_at"]
            merged["oracle_age"] = (merged["collected_at"] - merged["cl_timestamp"]).dt.total_seconds()
            
        else:
            # Default: simple concat and sort
            merged = pd.concat([df_cl, df_pm], ignore_index=True)
            merged = merged.sort_values("collected_at")
        
        return merged
    
    def compute_oracle_age(
        self,
        df_chainlink: pd.DataFrame,
        df_polymarket: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute oracle_age for each Polymarket observation.
        
        oracle_age = seconds since last Chainlink price update
        
        Args:
            df_chainlink: Chainlink data with 'timestamp' column
            df_polymarket: Polymarket data with 'collected_at' column
            
        Returns:
            Polymarket DataFrame with 'oracle_age' column added
        """
        df_cl = df_chainlink.sort_values("timestamp").copy()
        df_pm = df_polymarket.sort_values("collected_at").copy()
        
        oracle_ages = []
        last_cl_prices = []
        
        for _, row in df_pm.iterrows():
            pm_time = row["collected_at"]
            
            # Find last Chainlink update before this time
            cl_before = df_cl[df_cl["timestamp"] <= pm_time]
            
            if len(cl_before) > 0:
                last_cl = cl_before.iloc[-1]
                age = (pm_time - last_cl["timestamp"]).total_seconds()
                oracle_ages.append(age)
                last_cl_prices.append(last_cl["price"])
            else:
                oracle_ages.append(None)
                last_cl_prices.append(None)
        
        df_pm = df_pm.copy()
        df_pm["oracle_age"] = oracle_ages
        df_pm["last_cl_price"] = last_cl_prices
        
        return df_pm
    
    def compute_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute price and probability changes between observations.
        
        Args:
            df: Merged DataFrame with cl_price and pm_up_mid columns
            
        Returns:
            DataFrame with change columns added
        """
        df = df.copy()
        
        if "cl_price" in df.columns:
            df["cl_price_change"] = df["cl_price"].diff()
            df["cl_price_change_pct"] = df["cl_price"].pct_change() * 100
        
        if "pm_up_mid" in df.columns:
            df["pm_up_change"] = df["pm_up_mid"].diff()
            df["pm_up_change_pct"] = df["pm_up_mid"].pct_change() * 100
        
        if "pm_down_mid" in df.columns:
            df["pm_down_change"] = df["pm_down_mid"].diff()
        
        return df
    
    def compute_lag_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Compute statistics about oracle lag.
        
        Args:
            df: DataFrame with 'oracle_age' column
            
        Returns:
            Dictionary with lag statistics
        """
        if "oracle_age" not in df.columns or df["oracle_age"].isna().all():
            return {"error": "No oracle_age data"}
        
        ages = df["oracle_age"].dropna()
        
        return {
            "mean_oracle_age": ages.mean(),
            "median_oracle_age": ages.median(),
            "min_oracle_age": ages.min(),
            "max_oracle_age": ages.max(),
            "std_oracle_age": ages.std(),
            "p95_oracle_age": ages.quantile(0.95),
            "p99_oracle_age": ages.quantile(0.99),
            "count": len(ages),
        }
    
    def compute_correlation(self, df: pd.DataFrame) -> Dict:
        """
        Compute correlation between price changes and probability changes.
        
        Args:
            df: DataFrame with change columns
            
        Returns:
            Dictionary with correlation statistics
        """
        results = {}
        
        if "cl_price_change" in df.columns and "pm_up_change" in df.columns:
            # Remove NaN values
            valid = df[["cl_price_change", "pm_up_change"]].dropna()
            
            if len(valid) > 2:
                corr = valid["cl_price_change"].corr(valid["pm_up_change"])
                results["price_vs_up_mid_corr"] = corr
                results["n_observations"] = len(valid)
                
                # Lagged correlations
                for lag in [1, 2, 3, 5]:
                    if len(valid) > lag + 2:
                        lagged_corr = valid["cl_price_change"].corr(
                            valid["pm_up_change"].shift(-lag)
                        )
                        results[f"price_vs_up_mid_corr_lag{lag}"] = lagged_corr
        
        return results
    
    def generate_summary_report(
        self,
        df_chainlink: pd.DataFrame,
        df_polymarket: pd.DataFrame,
        market_slug: Optional[str] = None
    ) -> Dict:
        """
        Generate a comprehensive diagnostic report.
        
        Args:
            df_chainlink: Chainlink price data
            df_polymarket: Polymarket market data
            market_slug: Optional market identifier
            
        Returns:
            Dictionary with all diagnostic metrics
        """
        report = {
            "market_slug": market_slug,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Data coverage
        report["data_coverage"] = {
            "chainlink_records": len(df_chainlink),
            "polymarket_records": len(df_polymarket),
            "chainlink_time_range": {
                "start": df_chainlink["timestamp"].min().isoformat() if len(df_chainlink) > 0 else None,
                "end": df_chainlink["timestamp"].max().isoformat() if len(df_chainlink) > 0 else None,
            },
            "polymarket_time_range": {
                "start": df_polymarket["collected_at"].min().isoformat() if len(df_polymarket) > 0 else None,
                "end": df_polymarket["collected_at"].max().isoformat() if len(df_polymarket) > 0 else None,
            },
        }
        
        # Merge and compute metrics
        merged = self.merge_datasets(df_chainlink, df_polymarket)
        
        if len(merged) > 0:
            merged = self.compute_changes(merged)
            
            report["oracle_lag"] = self.compute_lag_statistics(merged)
            report["correlation"] = self.compute_correlation(merged)
            
            # Price statistics
            if "cl_price" in merged.columns:
                report["chainlink_price_stats"] = {
                    "mean": merged["cl_price"].mean(),
                    "std": merged["cl_price"].std(),
                    "min": merged["cl_price"].min(),
                    "max": merged["cl_price"].max(),
                }
            
            # Probability statistics
            if "pm_up_mid" in merged.columns:
                report["polymarket_up_mid_stats"] = {
                    "mean": merged["pm_up_mid"].mean(),
                    "std": merged["pm_up_mid"].std(),
                    "min": merged["pm_up_mid"].min(),
                    "max": merged["pm_up_mid"].max(),
                }
        
        return report
    
    def split_by_markets(
        self,
        merged_df: pd.DataFrame,
        market_duration_minutes: int = 15
    ) -> Dict[str, pd.DataFrame]:
        """
        Split merged data by 15-minute market boundaries.
        
        Args:
            merged_df: Merged DataFrame with 'time' column
            market_duration_minutes: Duration of each market in minutes (default 15)
            
        Returns:
            Dictionary mapping market_id to DataFrame
        """
        if len(merged_df) == 0 or "time" not in merged_df.columns:
            return {}
        
        df = merged_df.copy()
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")
        
        # Round to market boundaries (every 15 minutes)
        df["market_boundary"] = df["time"].dt.floor(f"{market_duration_minutes}min")
        
        markets = {}
        for boundary, group in df.groupby("market_boundary"):
            market_id = boundary.strftime("%Y%m%d_%H%M")
            markets[market_id] = group.drop(columns=["market_boundary"])
        
        self.logger.info(f"Split into {len(markets)} markets")
        return markets
    
    def analyze_per_market(
        self,
        df_chainlink: pd.DataFrame,
        df_polymarket: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Analyze each 15-minute market separately.
        
        Args:
            df_chainlink: Chainlink price data
            df_polymarket: Polymarket market data
            
        Returns:
            Dictionary mapping market_id to analysis results
        """
        # Merge with synchronization
        merged = self.merge_datasets(df_chainlink, df_polymarket, method="synchronized")
        
        if len(merged) == 0:
            self.logger.warning("No synchronized data found")
            return {}
        
        # Compute changes
        merged = self.compute_changes(merged)
        
        # Split by markets
        markets = self.split_by_markets(merged)
        
        # Analyze each market
        results = {}
        for market_id, market_df in markets.items():
            if len(market_df) < 2:
                continue  # Need at least 2 observations for correlation
            
            results[market_id] = {
                "n_observations": len(market_df),
                "start_time": market_df["time"].min().isoformat(),
                "end_time": market_df["time"].max().isoformat(),
                "oracle_lag": self.compute_lag_statistics(market_df),
                "correlation": self.compute_correlation(market_df),
                "price_stats": {
                    "start_price": market_df["cl_price"].iloc[0] if len(market_df) > 0 else None,
                    "end_price": market_df["cl_price"].iloc[-1] if len(market_df) > 0 else None,
                    "min": market_df["cl_price"].min(),
                    "max": market_df["cl_price"].max(),
                } if "cl_price" in market_df.columns else {},
                "prob_stats": {
                    "start_up": market_df["pm_up_mid"].iloc[0] if len(market_df) > 0 else None,
                    "end_up": market_df["pm_up_mid"].iloc[-1] if len(market_df) > 0 else None,
                    "min": market_df["pm_up_mid"].min(),
                    "max": market_df["pm_up_mid"].max(),
                } if "pm_up_mid" in market_df.columns else {},
            }
        
        return results


def create_merged_timeseries(
    df_chainlink: pd.DataFrame,
    df_polymarket: pd.DataFrame,
    synchronized: bool = True
) -> pd.DataFrame:
    """
    Convenience function to create a merged time-series dataset.
    
    Args:
        df_chainlink: Chainlink price data
        df_polymarket: Polymarket market data
        synchronized: If True, only keep observations collected at same timestamp
        
    Returns:
        Merged DataFrame suitable for analysis
    """
    analyzer = DiagnosticsAnalyzer()
    method = "synchronized" if synchronized else "nearest"
    merged = analyzer.merge_datasets(df_chainlink, df_polymarket, method=method)
    merged = analyzer.compute_changes(merged)
    return merged

