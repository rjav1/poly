#!/usr/bin/env python3
"""
L6 Order Book Structure Quality Control

Validates 6-level order book data for:
- Monotonicity: bids descending, asks ascending
- No crossed books: bid_p1 < ask_p1
- Size validity: sizes >= 0, price present iff size > 0
- Depth continuity: if level k is empty, higher levels should be empty
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class BookQCResult:
    """QC result for a single market."""
    market_id: str
    n_rows: int
    
    # Monotonicity violations
    up_bid_monotonicity_violations: int
    up_ask_monotonicity_violations: int
    down_bid_monotonicity_violations: int
    down_ask_monotonicity_violations: int
    
    # Crossed book violations
    up_crossed_violations: int
    down_crossed_violations: int
    
    # Size violations (negative sizes)
    size_negative_violations: int
    
    # Price/size consistency violations
    price_without_size_violations: int
    size_without_price_violations: int
    
    # Depth continuity violations
    depth_continuity_violations: int
    
    # Missing data per level
    up_bid_missing_by_level: Dict[int, float]  # level -> % missing
    up_ask_missing_by_level: Dict[int, float]
    down_bid_missing_by_level: Dict[int, float]
    down_ask_missing_by_level: Dict[int, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'market_id': self.market_id,
            'n_rows': self.n_rows,
            'up_bid_monotonicity_violations': self.up_bid_monotonicity_violations,
            'up_ask_monotonicity_violations': self.up_ask_monotonicity_violations,
            'down_bid_monotonicity_violations': self.down_bid_monotonicity_violations,
            'down_ask_monotonicity_violations': self.down_ask_monotonicity_violations,
            'up_crossed_violations': self.up_crossed_violations,
            'down_crossed_violations': self.down_crossed_violations,
            'size_negative_violations': self.size_negative_violations,
            'price_without_size_violations': self.price_without_size_violations,
            'size_without_price_violations': self.size_without_price_violations,
            'depth_continuity_violations': self.depth_continuity_violations,
            'up_bid_missing_by_level': self.up_bid_missing_by_level,
            'up_ask_missing_by_level': self.up_ask_missing_by_level,
            'down_bid_missing_by_level': self.down_bid_missing_by_level,
            'down_ask_missing_by_level': self.down_ask_missing_by_level,
        }
    
    @property
    def total_violations(self) -> int:
        """Total number of violations."""
        return (
            self.up_bid_monotonicity_violations +
            self.up_ask_monotonicity_violations +
            self.down_bid_monotonicity_violations +
            self.down_ask_monotonicity_violations +
            self.up_crossed_violations +
            self.down_crossed_violations +
            self.size_negative_violations +
            self.price_without_size_violations +
            self.size_without_price_violations +
            self.depth_continuity_violations
        )
    
    @property
    def violation_rate(self) -> float:
        """Violation rate as percentage of rows."""
        if self.n_rows == 0:
            return 0.0
        return self.total_violations / self.n_rows * 100


def get_l6_column_names() -> Dict[str, List[str]]:
    """
    Get column names for L1-L6 data.
    
    Returns dict with keys: 'up_bid', 'up_ask', 'down_bid', 'down_ask'
    Each value is a list of column names for levels 1-6.
    """
    # Level 1 uses 'best_' prefix, levels 2-6 use numeric suffix
    return {
        'up_bid': ['up_best_bid', 'up_bid_2', 'up_bid_3', 'up_bid_4', 'up_bid_5', 'up_bid_6'],
        'up_ask': ['up_best_ask', 'up_ask_2', 'up_ask_3', 'up_ask_4', 'up_ask_5', 'up_ask_6'],
        'down_bid': ['down_best_bid', 'down_bid_2', 'down_bid_3', 'down_bid_4', 'down_bid_5', 'down_bid_6'],
        'down_ask': ['down_best_ask', 'down_ask_2', 'down_ask_3', 'down_ask_4', 'down_ask_5', 'down_ask_6'],
        'up_bid_size': ['up_best_bid_size', 'up_bid_2_size', 'up_bid_3_size', 'up_bid_4_size', 'up_bid_5_size', 'up_bid_6_size'],
        'up_ask_size': ['up_best_ask_size', 'up_ask_2_size', 'up_ask_3_size', 'up_ask_4_size', 'up_ask_5_size', 'up_ask_6_size'],
        'down_bid_size': ['down_best_bid_size', 'down_bid_2_size', 'down_bid_3_size', 'down_bid_4_size', 'down_bid_5_size', 'down_bid_6_size'],
        'down_ask_size': ['down_best_ask_size', 'down_ask_2_size', 'down_ask_3_size', 'down_ask_4_size', 'down_ask_5_size', 'down_ask_6_size'],
    }


def check_monotonicity_bids(df: pd.DataFrame, bid_cols: List[str]) -> int:
    """
    Check bid monotonicity: bid_p1 >= bid_p2 >= ... >= bid_p6
    
    Returns number of rows with violations.
    """
    violations = 0
    
    # Get available columns
    available = [c for c in bid_cols if c in df.columns]
    if len(available) < 2:
        return 0
    
    # Check each pair of adjacent levels
    for i in range(len(available) - 1):
        col1, col2 = available[i], available[i + 1]
        # Violation if col1 < col2 (lower level price < higher level price)
        # Only count where both are non-NaN
        mask = df[col1].notna() & df[col2].notna()
        violations += (df.loc[mask, col1] < df.loc[mask, col2]).sum()
    
    return int(violations)


def check_monotonicity_asks(df: pd.DataFrame, ask_cols: List[str]) -> int:
    """
    Check ask monotonicity: ask_p1 <= ask_p2 <= ... <= ask_p6
    
    Returns number of rows with violations.
    """
    violations = 0
    
    # Get available columns
    available = [c for c in ask_cols if c in df.columns]
    if len(available) < 2:
        return 0
    
    # Check each pair of adjacent levels
    for i in range(len(available) - 1):
        col1, col2 = available[i], available[i + 1]
        # Violation if col1 > col2 (lower level price > higher level price)
        mask = df[col1].notna() & df[col2].notna()
        violations += (df.loc[mask, col1] > df.loc[mask, col2]).sum()
    
    return int(violations)


def check_crossed_book(df: pd.DataFrame, bid_col: str, ask_col: str, epsilon: float = 0.001) -> int:
    """
    Check for crossed book: bid_p1 should be < ask_p1.
    
    Allow tiny epsilon for rounding issues.
    
    Returns number of rows with crossed book.
    """
    if bid_col not in df.columns or ask_col not in df.columns:
        return 0
    
    mask = df[bid_col].notna() & df[ask_col].notna()
    # Crossed if bid >= ask - epsilon
    crossed = df.loc[mask, bid_col] >= df.loc[mask, ask_col] - epsilon
    return int(crossed.sum())


def check_size_validity(df: pd.DataFrame, size_cols: List[str]) -> int:
    """
    Check for negative sizes.
    
    Returns number of negative size values.
    """
    violations = 0
    for col in size_cols:
        if col in df.columns:
            violations += (df[col] < 0).sum()
    return int(violations)


def check_price_size_consistency(df: pd.DataFrame, price_cols: List[str], size_cols: List[str]) -> Tuple[int, int]:
    """
    Check price/size consistency:
    - Price present implies size > 0
    - Size > 0 implies price present
    
    Returns (price_without_size, size_without_price) violation counts.
    """
    price_without_size = 0
    size_without_price = 0
    
    for price_col, size_col in zip(price_cols, size_cols):
        if price_col not in df.columns or size_col not in df.columns:
            continue
        
        # Price present but size missing/zero
        price_present = df[price_col].notna()
        size_present = df[size_col].notna() & (df[size_col] > 0)
        
        price_without_size += (price_present & ~size_present).sum()
        size_without_price += (~price_present & size_present).sum()
    
    return int(price_without_size), int(size_without_price)


def check_depth_continuity(df: pd.DataFrame, price_cols: List[str]) -> int:
    """
    Check depth continuity: if level k is empty, higher levels should be empty.
    
    Returns number of rows with violations.
    """
    violations = 0
    
    # Get available columns
    available = [c for c in price_cols if c in df.columns]
    if len(available) < 2:
        return 0
    
    for i in range(len(available) - 1):
        col_lower, col_higher = available[i], available[i + 1]
        # Violation if lower level is empty but higher level has data
        lower_empty = df[col_lower].isna()
        higher_present = df[col_higher].notna()
        violations += (lower_empty & higher_present).sum()
    
    return int(violations)


def compute_missing_by_level(df: pd.DataFrame, cols: List[str]) -> Dict[int, float]:
    """
    Compute % missing for each level.
    
    Returns dict mapping level (1-6) to % missing.
    """
    result = {}
    for i, col in enumerate(cols):
        level = i + 1
        if col in df.columns:
            missing_pct = df[col].isna().sum() / len(df) * 100
        else:
            missing_pct = 100.0
        result[level] = round(missing_pct, 2)
    return result


def validate_market_book_structure(df: pd.DataFrame, market_id: str) -> BookQCResult:
    """
    Validate book structure for a single market.
    
    Args:
        df: DataFrame with L6 order book data
        market_id: Market identifier
        
    Returns:
        BookQCResult with validation results
    """
    cols = get_l6_column_names()
    
    # Monotonicity checks
    up_bid_mono = check_monotonicity_bids(df, cols['up_bid'])
    up_ask_mono = check_monotonicity_asks(df, cols['up_ask'])
    down_bid_mono = check_monotonicity_bids(df, cols['down_bid'])
    down_ask_mono = check_monotonicity_asks(df, cols['down_ask'])
    
    # Crossed book checks
    up_crossed = check_crossed_book(df, cols['up_bid'][0], cols['up_ask'][0])
    down_crossed = check_crossed_book(df, cols['down_bid'][0], cols['down_ask'][0])
    
    # Size validity
    all_size_cols = cols['up_bid_size'] + cols['up_ask_size'] + cols['down_bid_size'] + cols['down_ask_size']
    size_negative = check_size_validity(df, all_size_cols)
    
    # Price/size consistency
    up_bid_price_wo_size, up_bid_size_wo_price = check_price_size_consistency(
        df, cols['up_bid'], cols['up_bid_size']
    )
    up_ask_price_wo_size, up_ask_size_wo_price = check_price_size_consistency(
        df, cols['up_ask'], cols['up_ask_size']
    )
    down_bid_price_wo_size, down_bid_size_wo_price = check_price_size_consistency(
        df, cols['down_bid'], cols['down_bid_size']
    )
    down_ask_price_wo_size, down_ask_size_wo_price = check_price_size_consistency(
        df, cols['down_ask'], cols['down_ask_size']
    )
    
    price_without_size = up_bid_price_wo_size + up_ask_price_wo_size + down_bid_price_wo_size + down_ask_price_wo_size
    size_without_price = up_bid_size_wo_price + up_ask_size_wo_price + down_bid_size_wo_price + down_ask_size_wo_price
    
    # Depth continuity
    up_bid_cont = check_depth_continuity(df, cols['up_bid'])
    up_ask_cont = check_depth_continuity(df, cols['up_ask'])
    down_bid_cont = check_depth_continuity(df, cols['down_bid'])
    down_ask_cont = check_depth_continuity(df, cols['down_ask'])
    depth_continuity = up_bid_cont + up_ask_cont + down_bid_cont + down_ask_cont
    
    # Missing data by level
    up_bid_missing = compute_missing_by_level(df, cols['up_bid'])
    up_ask_missing = compute_missing_by_level(df, cols['up_ask'])
    down_bid_missing = compute_missing_by_level(df, cols['down_bid'])
    down_ask_missing = compute_missing_by_level(df, cols['down_ask'])
    
    return BookQCResult(
        market_id=market_id,
        n_rows=len(df),
        up_bid_monotonicity_violations=up_bid_mono,
        up_ask_monotonicity_violations=up_ask_mono,
        down_bid_monotonicity_violations=down_bid_mono,
        down_ask_monotonicity_violations=down_ask_mono,
        up_crossed_violations=up_crossed,
        down_crossed_violations=down_crossed,
        size_negative_violations=size_negative,
        price_without_size_violations=price_without_size,
        size_without_price_violations=size_without_price,
        depth_continuity_violations=depth_continuity,
        up_bid_missing_by_level=up_bid_missing,
        up_ask_missing_by_level=up_ask_missing,
        down_bid_missing_by_level=down_bid_missing,
        down_ask_missing_by_level=down_ask_missing,
    )


def run_l6_qc_on_raw_data(
    markets_dir: Path,
    asset: str = 'ETH',
    verbose: bool = True
) -> Tuple[List[BookQCResult], Dict[str, Any]]:
    """
    Run L6 book structure QC on raw market data.
    
    Args:
        markets_dir: Path to markets_6levels directory
        asset: Asset to process (e.g., 'ETH')
        verbose: Print progress
        
    Returns:
        (list of BookQCResult, aggregate stats dict)
    """
    asset_dir = markets_dir / asset
    if not asset_dir.exists():
        raise FileNotFoundError(f"Asset directory not found: {asset_dir}")
    
    # Find all market folders
    market_folders = sorted([d for d in asset_dir.iterdir() if d.is_dir()])
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"L6 BOOK STRUCTURE QC - {asset}")
        print(f"{'='*70}")
        print(f"Found {len(market_folders)} markets")
    
    results = []
    
    for folder in market_folders:
        pm_file = folder / 'polymarket.csv'
        if not pm_file.exists():
            if verbose:
                print(f"  [SKIP] {folder.name}: No polymarket.csv")
            continue
        
        try:
            df = pd.read_csv(pm_file)
            market_id = folder.name
            result = validate_market_book_structure(df, market_id)
            results.append(result)
            
            if verbose:
                status = "[OK]" if result.total_violations == 0 else f"[WARN: {result.total_violations} violations]"
                print(f"  {status} {market_id}: {result.n_rows} rows")
        except Exception as e:
            if verbose:
                print(f"  [ERROR] {folder.name}: {e}")
    
    # Compute aggregate stats
    if results:
        total_rows = sum(r.n_rows for r in results)
        total_violations = sum(r.total_violations for r in results)
        
        agg_stats = {
            'n_markets': len(results),
            'total_rows': total_rows,
            'total_violations': total_violations,
            'violation_rate_pct': round(total_violations / total_rows * 100, 4) if total_rows > 0 else 0,
            'markets_with_violations': sum(1 for r in results if r.total_violations > 0),
            'monotonicity_violations': {
                'up_bid': sum(r.up_bid_monotonicity_violations for r in results),
                'up_ask': sum(r.up_ask_monotonicity_violations for r in results),
                'down_bid': sum(r.down_bid_monotonicity_violations for r in results),
                'down_ask': sum(r.down_ask_monotonicity_violations for r in results),
            },
            'crossed_book_violations': {
                'up': sum(r.up_crossed_violations for r in results),
                'down': sum(r.down_crossed_violations for r in results),
            },
            'size_negative_violations': sum(r.size_negative_violations for r in results),
            'price_size_consistency': {
                'price_without_size': sum(r.price_without_size_violations for r in results),
                'size_without_price': sum(r.size_without_price_violations for r in results),
            },
            'depth_continuity_violations': sum(r.depth_continuity_violations for r in results),
            'avg_missing_by_level': {
                'up_bid': {level: round(np.mean([r.up_bid_missing_by_level.get(level, 100) for r in results]), 2) for level in range(1, 7)},
                'up_ask': {level: round(np.mean([r.up_ask_missing_by_level.get(level, 100) for r in results]), 2) for level in range(1, 7)},
                'down_bid': {level: round(np.mean([r.down_bid_missing_by_level.get(level, 100) for r in results]), 2) for level in range(1, 7)},
                'down_ask': {level: round(np.mean([r.down_ask_missing_by_level.get(level, 100) for r in results]), 2) for level in range(1, 7)},
            },
        }
    else:
        agg_stats = {'n_markets': 0, 'error': 'No markets processed'}
    
    if verbose:
        print(f"\n{'='*70}")
        print("AGGREGATE RESULTS")
        print(f"{'='*70}")
        print(f"Markets processed: {agg_stats.get('n_markets', 0)}")
        print(f"Total rows: {agg_stats.get('total_rows', 0)}")
        print(f"Total violations: {agg_stats.get('total_violations', 0)}")
        print(f"Violation rate: {agg_stats.get('violation_rate_pct', 0):.4f}%")
        print(f"Markets with violations: {agg_stats.get('markets_with_violations', 0)}")
        
        print("\nViolation breakdown:")
        mono = agg_stats.get('monotonicity_violations', {})
        print(f"  Monotonicity: up_bid={mono.get('up_bid', 0)}, up_ask={mono.get('up_ask', 0)}, "
              f"down_bid={mono.get('down_bid', 0)}, down_ask={mono.get('down_ask', 0)}")
        crossed = agg_stats.get('crossed_book_violations', {})
        print(f"  Crossed book: up={crossed.get('up', 0)}, down={crossed.get('down', 0)}")
        print(f"  Negative sizes: {agg_stats.get('size_negative_violations', 0)}")
        ps = agg_stats.get('price_size_consistency', {})
        print(f"  Price/size consistency: price_wo_size={ps.get('price_without_size', 0)}, "
              f"size_wo_price={ps.get('size_without_price', 0)}")
        print(f"  Depth continuity: {agg_stats.get('depth_continuity_violations', 0)}")
        
        print("\nAverage missing % by level (UP bids):")
        up_bid_missing = agg_stats.get('avg_missing_by_level', {}).get('up_bid', {})
        for level in range(1, 7):
            print(f"  L{level}: {up_bid_missing.get(level, 100):.1f}%")
    
    return results, agg_stats


def validate_l6_book_structure(
    df: pd.DataFrame,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Validate L6 book structure on a canonical dataset.
    
    This version works on a merged canonical dataset (not raw market CSVs).
    
    Args:
        df: Canonical dataset with L6 columns
        verbose: Print progress
        
    Returns:
        Validation results dict
    """
    results = []
    
    for market_id in df['market_id'].unique():
        market_df = df[df['market_id'] == market_id]
        result = validate_market_book_structure(market_df, market_id)
        results.append(result)
        
        if verbose:
            status = "[OK]" if result.total_violations == 0 else f"[WARN: {result.total_violations}]"
            print(f"  {status} {market_id}")
    
    # Compute aggregate stats
    if results:
        total_rows = sum(r.n_rows for r in results)
        total_violations = sum(r.total_violations for r in results)
        
        return {
            'n_markets': len(results),
            'total_rows': total_rows,
            'total_violations': total_violations,
            'violation_rate_pct': round(total_violations / total_rows * 100, 4) if total_rows > 0 else 0,
            'passed': total_violations == 0,
            'per_market': [r.to_dict() for r in results],
        }
    
    return {'n_markets': 0, 'passed': False, 'error': 'No markets'}


def main():
    """Run L6 QC on raw market data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="L6 Book Structure QC")
    parser.add_argument('--markets-dir', type=str, default='data_v2/markets_6levels',
                       help='Path to markets_6levels directory')
    parser.add_argument('--asset', type=str, default='ETH',
                       help='Asset to process')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    args = parser.parse_args()
    
    markets_dir = Path(args.markets_dir)
    results, agg_stats = run_l6_qc_on_raw_data(markets_dir, args.asset)
    
    if args.output:
        output_path = Path(args.output)
        output_data = {
            'aggregate': agg_stats,
            'per_market': [r.to_dict() for r in results],
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    # Return exit code based on violations
    if agg_stats.get('total_violations', 0) == 0:
        print("\n[PASS] All book structure checks passed!")
        return 0
    else:
        print(f"\n[WARN] {agg_stats.get('total_violations', 0)} violations found")
        return 1


if __name__ == '__main__':
    exit(main())

