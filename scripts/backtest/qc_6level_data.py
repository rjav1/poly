#!/usr/bin/env python3
"""
Quality Control for 6-Level Orderbook Data

This script validates the integrity of 6-level orderbook data before backtesting.

Checks performed:
1. Book integrity: Bids strictly decreasing, asks strictly increasing
2. Crossed markets: Flag when bid_1 >= ask_1
3. Missingness: % seconds where any level is missing
4. Level churn: How often prices jump between levels
5. Size validation: Non-negative, no NaN spikes

Output: QC table per market with all metrics
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import json

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class MarketQCResult:
    """QC results for a single market."""
    market_id: str
    asset: str
    n_rows: int
    
    # Book integrity
    bid_not_decreasing_up: int  # Count of rows where bids not decreasing
    bid_not_decreasing_down: int
    ask_not_increasing_up: int  # Count of rows where asks not increasing
    ask_not_increasing_down: int
    
    # Crossed markets
    crossed_up: int  # Count where bid_1 >= ask_1
    crossed_down: int
    crossed_rate_up: float
    crossed_rate_down: float
    
    # Missingness per level
    missing_l1_up: float  # % rows missing level 1
    missing_l2_up: float
    missing_l3_up: float
    missing_l4_up: float
    missing_l5_up: float
    missing_l6_up: float
    missing_l1_down: float
    missing_l2_down: float
    missing_l3_down: float
    missing_l4_down: float
    missing_l5_down: float
    missing_l6_down: float
    
    # Spreads
    median_spread_l1_up: float  # Median spread at level 1
    median_spread_l1_down: float
    
    # Depth
    median_top6_depth_up: float  # Median total depth across 6 levels
    median_top6_depth_down: float
    
    # Level churn (how often L1 price changes)
    l1_price_churn_up: float  # % of ticks with L1 price change
    l1_price_churn_down: float
    
    # Size validation
    negative_sizes_up: int
    negative_sizes_down: int
    
    # Overall quality score
    quality_score: float  # 0-100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'market_id': self.market_id,
            'asset': self.asset,
            'n_rows': self.n_rows,
            'bid_not_decreasing_up': self.bid_not_decreasing_up,
            'bid_not_decreasing_down': self.bid_not_decreasing_down,
            'ask_not_increasing_up': self.ask_not_increasing_up,
            'ask_not_increasing_down': self.ask_not_increasing_down,
            'crossed_up': self.crossed_up,
            'crossed_down': self.crossed_down,
            'crossed_rate_up': self.crossed_rate_up,
            'crossed_rate_down': self.crossed_rate_down,
            'missing_l1_up': self.missing_l1_up,
            'missing_l2_up': self.missing_l2_up,
            'missing_l3_up': self.missing_l3_up,
            'missing_l4_up': self.missing_l4_up,
            'missing_l5_up': self.missing_l5_up,
            'missing_l6_up': self.missing_l6_up,
            'missing_l1_down': self.missing_l1_down,
            'missing_l2_down': self.missing_l2_down,
            'missing_l3_down': self.missing_l3_down,
            'missing_l4_down': self.missing_l4_down,
            'missing_l5_down': self.missing_l5_down,
            'missing_l6_down': self.missing_l6_down,
            'median_spread_l1_up': self.median_spread_l1_up,
            'median_spread_l1_down': self.median_spread_l1_down,
            'median_top6_depth_up': self.median_top6_depth_up,
            'median_top6_depth_down': self.median_top6_depth_down,
            'l1_price_churn_up': self.l1_price_churn_up,
            'l1_price_churn_down': self.l1_price_churn_down,
            'negative_sizes_up': self.negative_sizes_up,
            'negative_sizes_down': self.negative_sizes_down,
            'quality_score': self.quality_score,
        }


def get_bid_columns(token: str) -> Tuple[List[str], List[str]]:
    """Get bid price and size column names for a token."""
    prefix = token.lower()
    price_cols = [
        f'{prefix}_best_bid',
        f'{prefix}_bid_2',
        f'{prefix}_bid_3',
        f'{prefix}_bid_4',
        f'{prefix}_bid_5',
        f'{prefix}_bid_6',
    ]
    size_cols = [
        f'{prefix}_best_bid_size',
        f'{prefix}_bid_2_size',
        f'{prefix}_bid_3_size',
        f'{prefix}_bid_4_size',
        f'{prefix}_bid_5_size',
        f'{prefix}_bid_6_size',
    ]
    return price_cols, size_cols


def get_ask_columns(token: str) -> Tuple[List[str], List[str]]:
    """Get ask price and size column names for a token."""
    prefix = token.lower()
    price_cols = [
        f'{prefix}_best_ask',
        f'{prefix}_ask_2',
        f'{prefix}_ask_3',
        f'{prefix}_ask_4',
        f'{prefix}_ask_5',
        f'{prefix}_ask_6',
    ]
    size_cols = [
        f'{prefix}_best_ask_size',
        f'{prefix}_ask_2_size',
        f'{prefix}_ask_3_size',
        f'{prefix}_ask_4_size',
        f'{prefix}_ask_5_size',
        f'{prefix}_ask_6_size',
    ]
    return price_cols, size_cols


def check_bids_decreasing(df: pd.DataFrame, token: str) -> int:
    """Check that bid prices are strictly decreasing (L1 > L2 > ... > L6)."""
    price_cols, _ = get_bid_columns(token)
    
    # Only check rows where multiple levels exist
    violations = 0
    for i in range(len(price_cols) - 1):
        col1 = price_cols[i]
        col2 = price_cols[i + 1]
        if col1 in df.columns and col2 in df.columns:
            # Violation: L1 <= L2 (should be L1 > L2 for bids)
            mask = df[col1].notna() & df[col2].notna()
            violations += (df.loc[mask, col1] <= df.loc[mask, col2]).sum()
    
    return violations


def check_asks_increasing(df: pd.DataFrame, token: str) -> int:
    """Check that ask prices are strictly increasing (L1 < L2 < ... < L6)."""
    price_cols, _ = get_ask_columns(token)
    
    violations = 0
    for i in range(len(price_cols) - 1):
        col1 = price_cols[i]
        col2 = price_cols[i + 1]
        if col1 in df.columns and col2 in df.columns:
            # Violation: L1 >= L2 (should be L1 < L2 for asks)
            mask = df[col1].notna() & df[col2].notna()
            violations += (df.loc[mask, col1] >= df.loc[mask, col2]).sum()
    
    return violations


def check_crossed_markets(df: pd.DataFrame, token: str) -> int:
    """Check for crossed markets (bid_1 >= ask_1)."""
    prefix = token.lower()
    bid_col = f'{prefix}_best_bid'
    ask_col = f'{prefix}_best_ask'
    
    if bid_col not in df.columns or ask_col not in df.columns:
        return 0
    
    mask = df[bid_col].notna() & df[ask_col].notna()
    return (df.loc[mask, bid_col] >= df.loc[mask, ask_col]).sum()


def compute_missingness(df: pd.DataFrame, token: str) -> List[float]:
    """Compute missingness rate for each level (1-6)."""
    bid_price_cols, bid_size_cols = get_bid_columns(token)
    ask_price_cols, ask_size_cols = get_ask_columns(token)
    
    n_rows = len(df)
    missing_rates = []
    
    for i in range(6):
        # A level is missing if both price columns are NaN
        bid_col = bid_price_cols[i]
        ask_col = ask_price_cols[i]
        
        bid_missing = df[bid_col].isna() if bid_col in df.columns else pd.Series([True] * n_rows)
        ask_missing = df[ask_col].isna() if ask_col in df.columns else pd.Series([True] * n_rows)
        
        # Level is missing if both bid and ask are missing
        level_missing = bid_missing & ask_missing
        missing_rate = level_missing.sum() / n_rows * 100 if n_rows > 0 else 100
        missing_rates.append(missing_rate)
    
    return missing_rates


def compute_median_spread(df: pd.DataFrame, token: str) -> float:
    """Compute median L1 spread."""
    prefix = token.lower()
    bid_col = f'{prefix}_best_bid'
    ask_col = f'{prefix}_best_ask'
    
    if bid_col not in df.columns or ask_col not in df.columns:
        return np.nan
    
    spread = df[ask_col] - df[bid_col]
    return spread.median()


def compute_median_depth(df: pd.DataFrame, token: str) -> float:
    """Compute median total depth across all 6 levels."""
    _, bid_size_cols = get_bid_columns(token)
    _, ask_size_cols = get_ask_columns(token)
    
    total_depth = pd.Series(0.0, index=df.index)
    
    for col in bid_size_cols + ask_size_cols:
        if col in df.columns:
            total_depth += df[col].fillna(0)
    
    return total_depth.median()


def compute_price_churn(df: pd.DataFrame, token: str) -> float:
    """Compute L1 price churn rate (% of ticks with price change)."""
    prefix = token.lower()
    bid_col = f'{prefix}_best_bid'
    ask_col = f'{prefix}_best_ask'
    
    if bid_col not in df.columns or ask_col not in df.columns:
        return 0.0
    
    # Count ticks where bid or ask price changed
    bid_changed = df[bid_col].diff().fillna(0) != 0
    ask_changed = df[ask_col].diff().fillna(0) != 0
    
    churn_count = (bid_changed | ask_changed).sum()
    return churn_count / len(df) * 100 if len(df) > 0 else 0.0


def count_negative_sizes(df: pd.DataFrame, token: str) -> int:
    """Count rows with negative sizes."""
    _, bid_size_cols = get_bid_columns(token)
    _, ask_size_cols = get_ask_columns(token)
    
    count = 0
    for col in bid_size_cols + ask_size_cols:
        if col in df.columns:
            count += (df[col] < 0).sum()
    
    return count


def compute_quality_score(result: MarketQCResult) -> float:
    """Compute overall quality score (0-100)."""
    score = 100.0
    
    # Deduct for book integrity issues
    integrity_issues = (
        result.bid_not_decreasing_up + result.bid_not_decreasing_down +
        result.ask_not_increasing_up + result.ask_not_increasing_down
    )
    if integrity_issues > 0:
        score -= min(20, integrity_issues / result.n_rows * 100 * 20)
    
    # Deduct for crossed markets
    crossed_rate = max(result.crossed_rate_up, result.crossed_rate_down)
    score -= min(20, crossed_rate * 20)
    
    # Deduct for L1 missingness
    l1_missing = max(result.missing_l1_up, result.missing_l1_down)
    score -= min(20, l1_missing * 0.5)
    
    # Deduct for negative sizes
    neg_sizes = result.negative_sizes_up + result.negative_sizes_down
    if neg_sizes > 0:
        score -= min(10, neg_sizes / result.n_rows * 100 * 10)
    
    return max(0, score)


def qc_market(market_folder: Path, asset: str) -> Optional[MarketQCResult]:
    """Run QC on a single market."""
    pm_file = market_folder / 'polymarket.csv'
    if not pm_file.exists():
        return None
    
    try:
        df = pd.read_csv(pm_file)
    except Exception as e:
        print(f"  Error reading {pm_file}: {e}")
        return None
    
    if len(df) == 0:
        return None
    
    market_id = market_folder.name
    n_rows = len(df)
    
    # Book integrity
    bid_not_dec_up = check_bids_decreasing(df, 'up')
    bid_not_dec_down = check_bids_decreasing(df, 'down')
    ask_not_inc_up = check_asks_increasing(df, 'up')
    ask_not_inc_down = check_asks_increasing(df, 'down')
    
    # Crossed markets
    crossed_up = check_crossed_markets(df, 'up')
    crossed_down = check_crossed_markets(df, 'down')
    crossed_rate_up = crossed_up / n_rows * 100 if n_rows > 0 else 0
    crossed_rate_down = crossed_down / n_rows * 100 if n_rows > 0 else 0
    
    # Missingness
    missing_up = compute_missingness(df, 'up')
    missing_down = compute_missingness(df, 'down')
    
    # Spreads
    spread_up = compute_median_spread(df, 'up')
    spread_down = compute_median_spread(df, 'down')
    
    # Depth
    depth_up = compute_median_depth(df, 'up')
    depth_down = compute_median_depth(df, 'down')
    
    # Churn
    churn_up = compute_price_churn(df, 'up')
    churn_down = compute_price_churn(df, 'down')
    
    # Negative sizes
    neg_up = count_negative_sizes(df, 'up')
    neg_down = count_negative_sizes(df, 'down')
    
    result = MarketQCResult(
        market_id=market_id,
        asset=asset,
        n_rows=n_rows,
        bid_not_decreasing_up=bid_not_dec_up,
        bid_not_decreasing_down=bid_not_dec_down,
        ask_not_increasing_up=ask_not_inc_up,
        ask_not_increasing_down=ask_not_inc_down,
        crossed_up=crossed_up,
        crossed_down=crossed_down,
        crossed_rate_up=crossed_rate_up,
        crossed_rate_down=crossed_rate_down,
        missing_l1_up=missing_up[0],
        missing_l2_up=missing_up[1],
        missing_l3_up=missing_up[2],
        missing_l4_up=missing_up[3],
        missing_l5_up=missing_up[4],
        missing_l6_up=missing_up[5],
        missing_l1_down=missing_down[0],
        missing_l2_down=missing_down[1],
        missing_l3_down=missing_down[2],
        missing_l4_down=missing_down[3],
        missing_l5_down=missing_down[4],
        missing_l6_down=missing_down[5],
        median_spread_l1_up=spread_up,
        median_spread_l1_down=spread_down,
        median_top6_depth_up=depth_up,
        median_top6_depth_down=depth_down,
        l1_price_churn_up=churn_up,
        l1_price_churn_down=churn_down,
        negative_sizes_up=neg_up,
        negative_sizes_down=neg_down,
        quality_score=0,  # Computed below
    )
    
    result.quality_score = compute_quality_score(result)
    
    return result


def run_qc_all_markets(
    markets_dir: Path = None,
    verbose: bool = True,
) -> List[MarketQCResult]:
    """Run QC on all 6-level markets."""
    if markets_dir is None:
        markets_dir = project_root / 'data_v2' / 'markets_6levels'
    
    results = []
    
    if not markets_dir.exists():
        print(f"Markets directory not found: {markets_dir}")
        return results
    
    # Find all asset directories
    for asset_dir in sorted(markets_dir.iterdir()):
        if not asset_dir.is_dir():
            continue
        
        asset = asset_dir.name
        if verbose:
            print(f"\nProcessing {asset}...")
        
        # Find all market folders
        for market_folder in sorted(asset_dir.iterdir()):
            if not market_folder.is_dir():
                continue
            
            result = qc_market(market_folder, asset)
            if result:
                results.append(result)
                if verbose:
                    quality = result.quality_score
                    issues = []
                    if result.crossed_rate_up > 1 or result.crossed_rate_down > 1:
                        issues.append("CROSSED")
                    if result.missing_l1_up > 10 or result.missing_l1_down > 10:
                        issues.append("MISSING")
                    if result.bid_not_decreasing_up > 0 or result.bid_not_decreasing_down > 0:
                        issues.append("BID_ORDER")
                    if result.ask_not_increasing_up > 0 or result.ask_not_increasing_down > 0:
                        issues.append("ASK_ORDER")
                    
                    status = "OK" if quality >= 90 else "WARN" if quality >= 70 else "FAIL"
                    issue_str = f" [{', '.join(issues)}]" if issues else ""
                    print(f"  {result.market_id}: {quality:.1f}% {status}{issue_str}")
    
    return results


def print_qc_summary(results: List[MarketQCResult]):
    """Print summary of QC results."""
    if not results:
        print("No results to summarize.")
        return
    
    print("\n" + "="*70)
    print("6-LEVEL DATA QC SUMMARY")
    print("="*70)
    
    print(f"\nTotal markets analyzed: {len(results)}")
    
    # Quality distribution
    high_quality = sum(1 for r in results if r.quality_score >= 90)
    medium_quality = sum(1 for r in results if 70 <= r.quality_score < 90)
    low_quality = sum(1 for r in results if r.quality_score < 70)
    
    print(f"\nQuality distribution:")
    print(f"  High (>=90): {high_quality} markets")
    print(f"  Medium (70-90): {medium_quality} markets")
    print(f"  Low (<70): {low_quality} markets")
    
    # Aggregate metrics
    df = pd.DataFrame([r.to_dict() for r in results])
    
    print(f"\n--- BOOK INTEGRITY ---")
    total_bid_issues = df['bid_not_decreasing_up'].sum() + df['bid_not_decreasing_down'].sum()
    total_ask_issues = df['ask_not_increasing_up'].sum() + df['ask_not_increasing_down'].sum()
    print(f"  Bid ordering violations: {total_bid_issues}")
    print(f"  Ask ordering violations: {total_ask_issues}")
    
    print(f"\n--- CROSSED MARKETS ---")
    print(f"  UP crossed rate (mean): {df['crossed_rate_up'].mean():.2f}%")
    print(f"  DOWN crossed rate (mean): {df['crossed_rate_down'].mean():.2f}%")
    
    print(f"\n--- MISSINGNESS (mean % per level) ---")
    print(f"  L1 UP: {df['missing_l1_up'].mean():.1f}%, DOWN: {df['missing_l1_down'].mean():.1f}%")
    print(f"  L2 UP: {df['missing_l2_up'].mean():.1f}%, DOWN: {df['missing_l2_down'].mean():.1f}%")
    print(f"  L3 UP: {df['missing_l3_up'].mean():.1f}%, DOWN: {df['missing_l3_down'].mean():.1f}%")
    print(f"  L4 UP: {df['missing_l4_up'].mean():.1f}%, DOWN: {df['missing_l4_down'].mean():.1f}%")
    print(f"  L5 UP: {df['missing_l5_up'].mean():.1f}%, DOWN: {df['missing_l5_down'].mean():.1f}%")
    print(f"  L6 UP: {df['missing_l6_up'].mean():.1f}%, DOWN: {df['missing_l6_down'].mean():.1f}%")
    
    print(f"\n--- SPREADS (median) ---")
    print(f"  L1 UP: ${df['median_spread_l1_up'].median():.4f}")
    print(f"  L1 DOWN: ${df['median_spread_l1_down'].median():.4f}")
    
    print(f"\n--- DEPTH (median top-6 total) ---")
    print(f"  UP: {df['median_top6_depth_up'].median():.0f} units")
    print(f"  DOWN: {df['median_top6_depth_down'].median():.0f} units")
    
    print(f"\n--- L1 PRICE CHURN (% ticks with change) ---")
    print(f"  UP: {df['l1_price_churn_up'].mean():.1f}%")
    print(f"  DOWN: {df['l1_price_churn_down'].mean():.1f}%")
    
    # Data quality assessment
    print("\n" + "-"*70)
    print("DATA QUALITY ASSESSMENT")
    print("-"*70)
    
    if high_quality >= len(results) * 0.8:
        print("\n[PASS] Data quality is GOOD - suitable for L2 backtesting")
    elif medium_quality + high_quality >= len(results) * 0.8:
        print("\n[WARN] Data quality is FAIR - proceed with caution")
    else:
        print("\n[FAIL] Data quality is POOR - review data collection")
    
    print("="*70)


def save_qc_results(results: List[MarketQCResult], output_dir: Path = None):
    """Save QC results to file."""
    if output_dir is None:
        output_dir = project_root / 'data_v2' / 'backtest_results' / 'maker_analysis'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_path = output_dir / 'qc_6level_results.json'
    with open(json_path, 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2, default=str)
    
    # Save as CSV
    csv_path = output_dir / 'qc_6level_results.csv'
    df = pd.DataFrame([r.to_dict() for r in results])
    df.to_csv(csv_path, index=False)
    
    print(f"\nResults saved to:")
    print(f"  {json_path}")
    print(f"  {csv_path}")


def main():
    """Run QC on all 6-level markets."""
    print("="*70)
    print("6-LEVEL ORDERBOOK DATA QC")
    print("="*70)
    
    results = run_qc_all_markets(verbose=True)
    
    print_qc_summary(results)
    save_qc_results(results)


if __name__ == '__main__':
    main()

