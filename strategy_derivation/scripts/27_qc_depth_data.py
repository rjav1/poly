#!/usr/bin/env python3
"""
Phase 1: Enhanced QC for 6-Level Orderbook Data

Verifies data quality for depth-aware execution:
1. Monotonic ladders: bids decrease, asks increase as level increases
2. Positive sizes: no negatives, flag suspicious zeros
3. Crossed books: best_bid <= best_ask always
4. Missing levels: count % of snapshots with <6 levels
5. Size consistency: flag size=0 but price exists (or vice versa)

Outputs:
- qc_depth_report.md
- qc_depth_stats.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
from collections import defaultdict

# Configuration
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
MARKETS_DIR = BASE_DIR.parent / "data_v2" / "markets_6levels" / "ETH"

# Column definitions for 6-level data
UP_ASK_PRICES = ['up_best_ask', 'up_ask_2', 'up_ask_3', 'up_ask_4', 'up_ask_5', 'up_ask_6']
UP_ASK_SIZES = ['up_best_ask_size', 'up_ask_2_size', 'up_ask_3_size', 'up_ask_4_size', 'up_ask_5_size', 'up_ask_6_size']
UP_BID_PRICES = ['up_best_bid', 'up_bid_2', 'up_bid_3', 'up_bid_4', 'up_bid_5', 'up_bid_6']
UP_BID_SIZES = ['up_best_bid_size', 'up_bid_2_size', 'up_bid_3_size', 'up_bid_4_size', 'up_bid_5_size', 'up_bid_6_size']

DOWN_ASK_PRICES = ['down_best_ask', 'down_ask_2', 'down_ask_3', 'down_ask_4', 'down_ask_5', 'down_ask_6']
DOWN_ASK_SIZES = ['down_best_ask_size', 'down_ask_2_size', 'down_ask_3_size', 'down_ask_4_size', 'down_ask_5_size', 'down_ask_6_size']
DOWN_BID_PRICES = ['down_best_bid', 'down_bid_2', 'down_bid_3', 'down_bid_4', 'down_bid_5', 'down_bid_6']
DOWN_BID_SIZES = ['down_best_bid_size', 'down_bid_2_size', 'down_bid_3_size', 'down_bid_4_size', 'down_bid_5_size', 'down_bid_6_size']


def load_market_data(market_dir: Path) -> pd.DataFrame:
    """Load polymarket CSV for a single market."""
    pm_path = market_dir / "polymarket.csv"
    if not pm_path.exists():
        return None
    return pd.read_csv(pm_path)


def check_monotonic_ladder(df: pd.DataFrame, price_cols: List[str], direction: str) -> Dict:
    """
    Check if price ladder is monotonic.
    
    Args:
        df: DataFrame with price columns
        price_cols: List of price column names (level 1-6)
        direction: 'increasing' for asks, 'decreasing' for bids
    
    Returns:
        Dict with violation stats
    """
    violations = 0
    total_checks = 0
    violation_examples = []
    
    for idx, row in df.iterrows():
        prices = [row.get(col, np.nan) for col in price_cols]
        
        # Only check non-NaN consecutive pairs
        for i in range(len(prices) - 1):
            p1, p2 = prices[i], prices[i + 1]
            if pd.isna(p1) or pd.isna(p2):
                continue
            
            total_checks += 1
            
            if direction == 'increasing':
                # Asks should increase (or stay same)
                if p2 < p1:
                    violations += 1
                    if len(violation_examples) < 3:
                        violation_examples.append({
                            'row': int(idx),
                            'level': i + 1,
                            'price1': float(p1),
                            'price2': float(p2)
                        })
            else:
                # Bids should decrease (or stay same)
                if p2 > p1:
                    violations += 1
                    if len(violation_examples) < 3:
                        violation_examples.append({
                            'row': int(idx),
                            'level': i + 1,
                            'price1': float(p1),
                            'price2': float(p2)
                        })
    
    return {
        'violations': violations,
        'total_checks': total_checks,
        'violation_pct': violations / total_checks * 100 if total_checks > 0 else 0,
        'examples': violation_examples
    }


def check_positive_sizes(df: pd.DataFrame, size_cols: List[str]) -> Dict:
    """Check for negative or suspicious zero sizes."""
    negative_count = 0
    zero_count = 0
    total_values = 0
    negative_examples = []
    
    for col in size_cols:
        if col not in df.columns:
            continue
        
        values = df[col].dropna()
        total_values += len(values)
        
        negatives = values[values < 0]
        zeros = values[values == 0]
        
        negative_count += len(negatives)
        zero_count += len(zeros)
        
        if len(negatives) > 0 and len(negative_examples) < 3:
            for idx in negatives.index[:3]:
                negative_examples.append({
                    'row': int(idx),
                    'column': col,
                    'value': float(negatives.loc[idx])
                })
    
    return {
        'negative_count': negative_count,
        'zero_count': zero_count,
        'total_values': total_values,
        'negative_pct': negative_count / total_values * 100 if total_values > 0 else 0,
        'zero_pct': zero_count / total_values * 100 if total_values > 0 else 0,
        'negative_examples': negative_examples
    }


def check_crossed_books(df: pd.DataFrame) -> Dict:
    """Check for crossed books (best_bid > best_ask)."""
    crossed_up = 0
    crossed_down = 0
    total_up = 0
    total_down = 0
    examples = []
    
    for idx, row in df.iterrows():
        # UP token
        up_bid = row.get('up_best_bid', np.nan)
        up_ask = row.get('up_best_ask', np.nan)
        if not pd.isna(up_bid) and not pd.isna(up_ask):
            total_up += 1
            if up_bid > up_ask:
                crossed_up += 1
                if len(examples) < 3:
                    examples.append({
                        'row': int(idx),
                        'token': 'UP',
                        'best_bid': float(up_bid),
                        'best_ask': float(up_ask)
                    })
        
        # DOWN token
        down_bid = row.get('down_best_bid', np.nan)
        down_ask = row.get('down_best_ask', np.nan)
        if not pd.isna(down_bid) and not pd.isna(down_ask):
            total_down += 1
            if down_bid > down_ask:
                crossed_down += 1
                if len(examples) < 3:
                    examples.append({
                        'row': int(idx),
                        'token': 'DOWN',
                        'best_bid': float(down_bid),
                        'best_ask': float(down_ask)
                    })
    
    return {
        'crossed_up': crossed_up,
        'crossed_down': crossed_down,
        'total_up': total_up,
        'total_down': total_down,
        'crossed_up_pct': crossed_up / total_up * 100 if total_up > 0 else 0,
        'crossed_down_pct': crossed_down / total_down * 100 if total_down > 0 else 0,
        'examples': examples
    }


def check_missing_levels(df: pd.DataFrame, price_cols: List[str], size_cols: List[str]) -> Dict:
    """Count snapshots with missing levels."""
    level_counts = defaultdict(int)
    total_rows = len(df)
    missing_patterns = defaultdict(int)
    
    for idx, row in df.iterrows():
        levels_present = 0
        pattern = []
        
        for i, (price_col, size_col) in enumerate(zip(price_cols, size_cols)):
            price = row.get(price_col, np.nan)
            size = row.get(size_col, np.nan)
            
            if not pd.isna(price) and not pd.isna(size) and size > 0:
                levels_present += 1
                pattern.append(str(i + 1))
            else:
                pattern.append('_')
        
        level_counts[levels_present] += 1
        pattern_str = ''.join(pattern)
        missing_patterns[pattern_str] += 1
    
    # Calculate % with each level count
    level_distribution = {
        f'levels_{k}': {
            'count': v,
            'pct': v / total_rows * 100
        } for k, v in sorted(level_counts.items())
    }
    
    # Top missing patterns
    top_patterns = sorted(missing_patterns.items(), key=lambda x: -x[1])[:5]
    
    return {
        'total_rows': total_rows,
        'level_distribution': level_distribution,
        'full_depth_pct': level_counts[6] / total_rows * 100 if total_rows > 0 else 0,
        'at_least_1_level_pct': (total_rows - level_counts[0]) / total_rows * 100 if total_rows > 0 else 0,
        'top_patterns': [{'pattern': p, 'count': c} for p, c in top_patterns]
    }


def check_size_consistency(df: pd.DataFrame, price_cols: List[str], size_cols: List[str]) -> Dict:
    """Check for inconsistencies: price exists but size=0/NaN, or vice versa."""
    price_no_size = 0
    size_no_price = 0
    total_checks = 0
    examples = []
    
    for idx, row in df.iterrows():
        for price_col, size_col in zip(price_cols, size_cols):
            price = row.get(price_col, np.nan)
            size = row.get(size_col, np.nan)
            
            total_checks += 1
            
            # Price exists but no size
            if not pd.isna(price) and price > 0 and (pd.isna(size) or size == 0):
                price_no_size += 1
                if len(examples) < 3:
                    examples.append({
                        'row': int(idx),
                        'price_col': price_col,
                        'size_col': size_col,
                        'price': float(price),
                        'size': float(size) if not pd.isna(size) else None,
                        'issue': 'price_no_size'
                    })
            
            # Size exists but no price
            if not pd.isna(size) and size > 0 and (pd.isna(price) or price == 0):
                size_no_price += 1
                if len(examples) < 3:
                    examples.append({
                        'row': int(idx),
                        'price_col': price_col,
                        'size_col': size_col,
                        'price': float(price) if not pd.isna(price) else None,
                        'size': float(size),
                        'issue': 'size_no_price'
                    })
    
    return {
        'price_no_size': price_no_size,
        'size_no_price': size_no_price,
        'total_checks': total_checks,
        'price_no_size_pct': price_no_size / total_checks * 100 if total_checks > 0 else 0,
        'size_no_price_pct': size_no_price / total_checks * 100 if total_checks > 0 else 0,
        'examples': examples
    }


def run_qc_for_market(market_dir: Path) -> Dict:
    """Run all QC checks for a single market."""
    df = load_market_data(market_dir)
    if df is None:
        return None
    
    market_id = market_dir.name
    
    results = {
        'market_id': market_id,
        'n_rows': len(df),
        'checks': {}
    }
    
    # 1. Monotonic ladders
    results['checks']['monotonic'] = {
        'up_asks': check_monotonic_ladder(df, UP_ASK_PRICES, 'increasing'),
        'up_bids': check_monotonic_ladder(df, UP_BID_PRICES, 'decreasing'),
        'down_asks': check_monotonic_ladder(df, DOWN_ASK_PRICES, 'increasing'),
        'down_bids': check_monotonic_ladder(df, DOWN_BID_PRICES, 'decreasing')
    }
    
    # 2. Positive sizes
    all_size_cols = UP_ASK_SIZES + UP_BID_SIZES + DOWN_ASK_SIZES + DOWN_BID_SIZES
    results['checks']['positive_sizes'] = check_positive_sizes(df, all_size_cols)
    
    # 3. Crossed books
    results['checks']['crossed_books'] = check_crossed_books(df)
    
    # 4. Missing levels
    results['checks']['missing_levels'] = {
        'up_asks': check_missing_levels(df, UP_ASK_PRICES, UP_ASK_SIZES),
        'up_bids': check_missing_levels(df, UP_BID_PRICES, UP_BID_SIZES),
        'down_asks': check_missing_levels(df, DOWN_ASK_PRICES, DOWN_ASK_SIZES),
        'down_bids': check_missing_levels(df, DOWN_BID_PRICES, DOWN_BID_SIZES)
    }
    
    # 5. Size consistency
    results['checks']['size_consistency'] = {
        'up': check_size_consistency(df, UP_ASK_PRICES + UP_BID_PRICES, UP_ASK_SIZES + UP_BID_SIZES),
        'down': check_size_consistency(df, DOWN_ASK_PRICES + DOWN_BID_PRICES, DOWN_ASK_SIZES + DOWN_BID_SIZES)
    }
    
    return results


def aggregate_results(market_results: List[Dict]) -> Dict:
    """Aggregate QC results across all markets."""
    agg = {
        'n_markets': len(market_results),
        'total_rows': sum(m['n_rows'] for m in market_results),
        'monotonic': {
            'up_asks_violation_pct': np.mean([m['checks']['monotonic']['up_asks']['violation_pct'] for m in market_results]),
            'up_bids_violation_pct': np.mean([m['checks']['monotonic']['up_bids']['violation_pct'] for m in market_results]),
            'down_asks_violation_pct': np.mean([m['checks']['monotonic']['down_asks']['violation_pct'] for m in market_results]),
            'down_bids_violation_pct': np.mean([m['checks']['monotonic']['down_bids']['violation_pct'] for m in market_results]),
        },
        'positive_sizes': {
            'avg_negative_pct': np.mean([m['checks']['positive_sizes']['negative_pct'] for m in market_results]),
            'avg_zero_pct': np.mean([m['checks']['positive_sizes']['zero_pct'] for m in market_results]),
            'total_negatives': sum(m['checks']['positive_sizes']['negative_count'] for m in market_results),
        },
        'crossed_books': {
            'avg_crossed_up_pct': np.mean([m['checks']['crossed_books']['crossed_up_pct'] for m in market_results]),
            'avg_crossed_down_pct': np.mean([m['checks']['crossed_books']['crossed_down_pct'] for m in market_results]),
            'total_crossed_up': sum(m['checks']['crossed_books']['crossed_up'] for m in market_results),
            'total_crossed_down': sum(m['checks']['crossed_books']['crossed_down'] for m in market_results),
        },
        'missing_levels': {
            'up_asks_full_depth_pct': np.mean([m['checks']['missing_levels']['up_asks']['full_depth_pct'] for m in market_results]),
            'down_asks_full_depth_pct': np.mean([m['checks']['missing_levels']['down_asks']['full_depth_pct'] for m in market_results]),
        },
        'size_consistency': {
            'avg_price_no_size_pct': np.mean([m['checks']['size_consistency']['up']['price_no_size_pct'] for m in market_results]),
            'avg_size_no_price_pct': np.mean([m['checks']['size_consistency']['up']['size_no_price_pct'] for m in market_results]),
        }
    }
    
    return agg


def generate_report(market_results: List[Dict], agg: Dict) -> str:
    """Generate QC report."""
    report = []
    report.append("# 6-Level Orderbook Data QC Report\n")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Markets Analyzed**: {agg['n_markets']}\n")
    report.append(f"**Total Rows**: {agg['total_rows']:,}\n\n")
    
    report.append("---\n\n")
    
    # Summary
    report.append("## Summary\n\n")
    
    # Determine pass/fail for each check
    monotonic_ok = all([
        agg['monotonic']['up_asks_violation_pct'] < 5,
        agg['monotonic']['up_bids_violation_pct'] < 5,
        agg['monotonic']['down_asks_violation_pct'] < 5,
        agg['monotonic']['down_bids_violation_pct'] < 5
    ])
    crossed_ok = agg['crossed_books']['avg_crossed_up_pct'] < 5 and agg['crossed_books']['avg_crossed_down_pct'] < 5
    sizes_ok = agg['positive_sizes']['total_negatives'] == 0
    missing_ok = agg['missing_levels']['up_asks_full_depth_pct'] > 10 or agg['missing_levels']['down_asks_full_depth_pct'] > 10
    
    report.append("| Check | Status | Details |\n")
    report.append("|-------|--------|--------|\n")
    report.append(f"| Monotonic Ladders | {'[PASS]' if monotonic_ok else '[FAIL]'} | Up asks: {agg['monotonic']['up_asks_violation_pct']:.2f}%, Down asks: {agg['monotonic']['down_asks_violation_pct']:.2f}% violations |\n")
    report.append(f"| Crossed Books | {'[PASS]' if crossed_ok else '[FAIL]'} | Up: {agg['crossed_books']['avg_crossed_up_pct']:.2f}%, Down: {agg['crossed_books']['avg_crossed_down_pct']:.2f}% crossed |\n")
    report.append(f"| Positive Sizes | {'[PASS]' if sizes_ok else '[FAIL]'} | {agg['positive_sizes']['total_negatives']} negative values found |\n")
    report.append(f"| Depth Coverage | {'[PASS]' if missing_ok else '[WARN]'} | UP full-depth: {agg['missing_levels']['up_asks_full_depth_pct']:.1f}%, DOWN full-depth: {agg['missing_levels']['down_asks_full_depth_pct']:.1f}% |\n\n")
    
    # Detailed sections
    report.append("---\n\n")
    report.append("## 1. Monotonic Ladder Check\n\n")
    report.append("Verifies that ask prices increase and bid prices decrease as we move deeper into the book.\n\n")
    report.append("| Side | Violation % |\n")
    report.append("|------|-------------|\n")
    report.append(f"| UP Asks | {agg['monotonic']['up_asks_violation_pct']:.3f}% |\n")
    report.append(f"| UP Bids | {agg['monotonic']['up_bids_violation_pct']:.3f}% |\n")
    report.append(f"| DOWN Asks | {agg['monotonic']['down_asks_violation_pct']:.3f}% |\n")
    report.append(f"| DOWN Bids | {agg['monotonic']['down_bids_violation_pct']:.3f}% |\n\n")
    
    report.append("---\n\n")
    report.append("## 2. Crossed Books Check\n\n")
    report.append("Verifies that best_bid <= best_ask (no arbitrage within the book).\n\n")
    report.append(f"- **UP token crossed**: {agg['crossed_books']['total_crossed_up']} rows ({agg['crossed_books']['avg_crossed_up_pct']:.2f}%)\n")
    report.append(f"- **DOWN token crossed**: {agg['crossed_books']['total_crossed_down']} rows ({agg['crossed_books']['avg_crossed_down_pct']:.2f}%)\n\n")
    
    report.append("---\n\n")
    report.append("## 3. Size Validity Check\n\n")
    report.append(f"- **Negative sizes found**: {agg['positive_sizes']['total_negatives']}\n")
    report.append(f"- **Zero sizes (avg)**: {agg['positive_sizes']['avg_zero_pct']:.2f}%\n\n")
    
    report.append("---\n\n")
    report.append("## 4. Depth Coverage\n\n")
    report.append("Percentage of snapshots with all 6 levels available (with non-zero size).\n\n")
    report.append(f"- **UP asks full depth**: {agg['missing_levels']['up_asks_full_depth_pct']:.1f}%\n")
    report.append(f"- **DOWN asks full depth**: {agg['missing_levels']['down_asks_full_depth_pct']:.1f}%\n\n")
    
    report.append("---\n\n")
    report.append("## 5. Size Consistency\n\n")
    report.append("Checks for prices without sizes or sizes without prices.\n\n")
    report.append(f"- **Price exists but no size**: {agg['size_consistency']['avg_price_no_size_pct']:.2f}%\n")
    report.append(f"- **Size exists but no price**: {agg['size_consistency']['avg_size_no_price_pct']:.2f}%\n\n")
    
    # Overall verdict
    report.append("---\n\n")
    report.append("## Overall Verdict\n\n")
    
    all_pass = monotonic_ok and crossed_ok and sizes_ok
    if all_pass:
        report.append("**[PASS] Data quality sufficient for depth-aware execution modeling.**\n\n")
        report.append("The 6-level orderbook data passes all critical checks and can be used for:\n")
        report.append("- VWAP calculations\n")
        report.append("- Executable size estimation\n")
        report.append("- Slippage modeling\n")
    else:
        report.append("**[WARN] Some data quality issues detected.**\n\n")
        if not monotonic_ok:
            report.append("- Monotonic ladder violations may affect VWAP accuracy\n")
        if not crossed_ok:
            report.append("- Crossed books indicate data quality issues or stale quotes\n")
        if not sizes_ok:
            report.append("- Negative sizes are invalid and should be investigated\n")
    
    return ''.join(report)


def main():
    print("=" * 70)
    print("6-Level Orderbook Data QC")
    print("=" * 70)
    
    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all markets
    if not MARKETS_DIR.exists():
        print(f"ERROR: Markets directory not found: {MARKETS_DIR}")
        return
    
    market_dirs = [d for d in MARKETS_DIR.iterdir() if d.is_dir()]
    print(f"\nFound {len(market_dirs)} markets to analyze")
    
    # Run QC for each market
    market_results = []
    for i, market_dir in enumerate(market_dirs):
        print(f"\n  [{i+1}/{len(market_dirs)}] Processing {market_dir.name}...")
        result = run_qc_for_market(market_dir)
        if result:
            market_results.append(result)
            print(f"    Rows: {result['n_rows']}")
    
    print(f"\n  Successfully processed {len(market_results)} markets")
    
    # Aggregate results
    print("\nAggregating results...")
    agg = aggregate_results(market_results)
    
    # Generate report
    print("Generating report...")
    report = generate_report(market_results, agg)
    
    # Save outputs
    print("\n" + "=" * 70)
    print("Saving outputs...")
    
    # Save JSON stats
    stats_path = RESULTS_DIR / "qc_depth_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"  Stats saved to: {stats_path}")
    
    # Save detailed results
    detailed_path = RESULTS_DIR / "qc_depth_detailed.json"
    with open(detailed_path, 'w') as f:
        # Convert to serializable format
        serializable = []
        for m in market_results:
            m_copy = {
                'market_id': m['market_id'],
                'n_rows': m['n_rows'],
                'monotonic_up_asks_violation_pct': m['checks']['monotonic']['up_asks']['violation_pct'],
                'monotonic_down_asks_violation_pct': m['checks']['monotonic']['down_asks']['violation_pct'],
                'crossed_up_pct': m['checks']['crossed_books']['crossed_up_pct'],
                'crossed_down_pct': m['checks']['crossed_books']['crossed_down_pct'],
                'up_asks_full_depth_pct': m['checks']['missing_levels']['up_asks']['full_depth_pct'],
                'down_asks_full_depth_pct': m['checks']['missing_levels']['down_asks']['full_depth_pct'],
            }
            serializable.append(m_copy)
        json.dump(serializable, f, indent=2)
    print(f"  Detailed results saved to: {detailed_path}")
    
    # Save report
    report_path = REPORTS_DIR / "qc_depth_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("QC SUMMARY")
    print("=" * 70)
    print(f"\n  Markets analyzed: {agg['n_markets']}")
    print(f"  Total rows: {agg['total_rows']:,}")
    print(f"\n  Monotonic violations:")
    print(f"    UP asks: {agg['monotonic']['up_asks_violation_pct']:.3f}%")
    print(f"    DOWN asks: {agg['monotonic']['down_asks_violation_pct']:.3f}%")
    print(f"\n  Crossed books:")
    print(f"    UP: {agg['crossed_books']['total_crossed_up']} ({agg['crossed_books']['avg_crossed_up_pct']:.2f}%)")
    print(f"    DOWN: {agg['crossed_books']['total_crossed_down']} ({agg['crossed_books']['avg_crossed_down_pct']:.2f}%)")
    print(f"\n  Full depth coverage:")
    print(f"    UP asks: {agg['missing_levels']['up_asks_full_depth_pct']:.1f}%")
    print(f"    DOWN asks: {agg['missing_levels']['down_asks_full_depth_pct']:.1f}%")
    print(f"\n  Negative sizes: {agg['positive_sizes']['total_negatives']}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

