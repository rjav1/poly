#!/usr/bin/env python3
"""
Optionality Check: Is the Strategy Just Long Optionality?

Even two-sided makers can end up with net exposure if fills are asymmetric.
This module checks if PnL is explained by directional bets rather than
spread capture.

Key checks:
1. Correlation between PnL and final market direction
2. Correlation between PnL and max net exposure
3. Correlation between PnL and spread captured
4. Inventory over time analysis
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class OptionalityResults:
    """Results from optionality check."""
    n_markets: int
    
    # Correlations
    corr_pnl_direction: float  # Correlation with final direction
    corr_pnl_exposure: float   # Correlation with max net exposure
    corr_pnl_spread: float     # Correlation with spread captured
    
    # P-values for correlations
    pval_pnl_direction: float
    pval_pnl_exposure: float
    pval_pnl_spread: float
    
    # PnL decomposition
    directional_pnl: float  # Component correlated with direction
    spread_pnl: float       # Residual (true spread capture)
    
    # Flags
    has_directional_bias: bool
    has_optionality: bool


def compute_direction(market_df: pd.DataFrame) -> int:
    """
    Compute final market direction.
    +1 if UP settled (price went up), -1 if DOWN settled (price went down)
    """
    # Check if we have settlement data
    if 'settled_up' in market_df.columns:
        last_row = market_df.iloc[-1]
        if last_row.get('settled_up', False):
            return 1
        elif last_row.get('settled_down', False):
            return -1
    
    # Otherwise, use CL price change
    cl_first = None
    cl_last = None
    
    if 'cl_mid' in market_df.columns:
        cl_valid = market_df['cl_mid'].dropna()
        if len(cl_valid) > 0:
            cl_first = cl_valid.iloc[0]
            cl_last = cl_valid.iloc[-1]
    
    if cl_first is not None and cl_last is not None and cl_first > 0:
        return 1 if cl_last > cl_first else -1
    
    return 0


def run_optionality_check(
    df: pd.DataFrame,
    strategy: Any = None,
    volume_markets_only: bool = True,
    verbose: bool = True,
) -> OptionalityResults:
    """
    Run optionality check on strategy results.
    
    Args:
        df: Market data
        strategy: Strategy to test (or default)
        volume_markets_only: Only use volume markets
        verbose: Print report
        
    Returns:
        OptionalityResults
    """
    from scripts.backtest.strategies import SpreadCaptureStrategy
    from scripts.backtest.backtest_engine import run_maker_backtest
    from scripts.backtest.maker_execution import MakerExecutionConfig, FillModel
    
    if strategy is None:
        strategy = SpreadCaptureStrategy(
            spread_min=0.01,
            tau_min=60,
            tau_max=600,
            inventory_limit_up=10.0,
            inventory_limit_down=10.0,
        )
    
    config = MakerExecutionConfig(
        place_latency_ms=100,
        cancel_latency_ms=50,
        fill_model=FillModel.TOUCH_SIZE_PROXY,
        touch_trade_rate_per_second=0.03,
    )
    
    # Run backtest
    result = run_maker_backtest(df, strategy, config,
                                volume_markets_only=volume_markets_only,
                                verbose=verbose)
    
    # Get per-market data
    market_results = result.get('market_results', {})
    
    # Collect data for correlation analysis
    analysis_data = []
    
    for market_id, mkt_result in market_results.items():
        market_df = df[df['market_id'] == market_id]
        
        # Get direction
        direction = compute_direction(market_df)
        
        # Get metrics
        pnl = mkt_result.get('pnl', 0)
        spread_captured = mkt_result.get('spread_captured', 0)
        
        # Compute max net exposure from fills (if available)
        fills = mkt_result.get('fills', [])
        max_exposure = 0
        cumulative_exposure = 0
        
        for fill in fills:
            if isinstance(fill, dict):
                side = fill.get('side', '')
                size = fill.get('fill_size', 1)
                token = fill.get('token', 'UP')
            else:
                side = fill.side
                size = fill.fill_size
                token = fill.token
            
            # Compute exposure change
            if side == 'BID':  # Bought
                cumulative_exposure += size if token == 'UP' else -size
            else:  # Sold
                cumulative_exposure -= size if token == 'UP' else size
            
            max_exposure = max(max_exposure, abs(cumulative_exposure))
        
        analysis_data.append({
            'market_id': market_id,
            'pnl': pnl,
            'direction': direction,
            'max_exposure': max_exposure,
            'spread_captured': spread_captured,
        })
    
    analysis_df = pd.DataFrame(analysis_data)
    n_markets = len(analysis_df)
    
    if n_markets < 3:
        return OptionalityResults(
            n_markets=n_markets,
            corr_pnl_direction=0, corr_pnl_exposure=0, corr_pnl_spread=0,
            pval_pnl_direction=1, pval_pnl_exposure=1, pval_pnl_spread=1,
            directional_pnl=0, spread_pnl=0,
            has_directional_bias=False, has_optionality=False,
        )
    
    # Compute correlations
    corr_dir, pval_dir = stats.pearsonr(analysis_df['pnl'], analysis_df['direction'])
    
    if analysis_df['max_exposure'].std() > 0:
        corr_exp, pval_exp = stats.pearsonr(analysis_df['pnl'], analysis_df['max_exposure'])
    else:
        corr_exp, pval_exp = 0, 1
    
    if analysis_df['spread_captured'].std() > 0:
        corr_spread, pval_spread = stats.pearsonr(analysis_df['pnl'], analysis_df['spread_captured'])
    else:
        corr_spread, pval_spread = 0, 1
    
    # Decompose PnL into directional and spread components
    # Simple regression: PnL = a * direction + b
    if analysis_df['direction'].std() > 0:
        slope, intercept, _, _, _ = stats.linregress(analysis_df['direction'], analysis_df['pnl'])
        directional_pnl = slope * analysis_df['direction'].mean() * n_markets
        spread_pnl = analysis_df['pnl'].sum() - directional_pnl
    else:
        directional_pnl = 0
        spread_pnl = analysis_df['pnl'].sum()
    
    # Determine flags
    has_directional_bias = abs(corr_dir) > 0.5 and pval_dir < 0.10
    has_optionality = abs(corr_exp) > 0.5 and pval_exp < 0.10
    
    results = OptionalityResults(
        n_markets=n_markets,
        corr_pnl_direction=corr_dir,
        corr_pnl_exposure=corr_exp,
        corr_pnl_spread=corr_spread,
        pval_pnl_direction=pval_dir,
        pval_pnl_exposure=pval_exp,
        pval_pnl_spread=pval_spread,
        directional_pnl=directional_pnl,
        spread_pnl=spread_pnl,
        has_directional_bias=has_directional_bias,
        has_optionality=has_optionality,
    )
    
    if verbose:
        print_optionality_report(results, analysis_df)
    
    return results


def print_optionality_report(results: OptionalityResults, analysis_df: pd.DataFrame = None):
    """Print optionality check report."""
    print("\n" + "="*70)
    print("OPTIONALITY CHECK REPORT")
    print("="*70)
    
    print("\n1. CORRELATION ANALYSIS")
    print("-"*50)
    print(f"  Markets analyzed: {results.n_markets}")
    
    print(f"\n  PnL vs Direction:")
    print(f"    Correlation: {results.corr_pnl_direction:.3f}")
    print(f"    P-value: {results.pval_pnl_direction:.4f}")
    
    print(f"\n  PnL vs Max Exposure:")
    print(f"    Correlation: {results.corr_pnl_exposure:.3f}")
    print(f"    P-value: {results.pval_pnl_exposure:.4f}")
    
    print(f"\n  PnL vs Spread Captured:")
    print(f"    Correlation: {results.corr_pnl_spread:.3f}")
    print(f"    P-value: {results.pval_pnl_spread:.4f}")
    
    print("\n2. PNL DECOMPOSITION")
    print("-"*50)
    total_pnl = results.directional_pnl + results.spread_pnl
    print(f"  Total PnL: ${total_pnl:.4f}")
    print(f"  Directional component: ${results.directional_pnl:.4f} ({results.directional_pnl/total_pnl*100:.1f}% if total != 0)")
    print(f"  Spread component: ${results.spread_pnl:.4f}")
    
    print("\n3. DIAGNOSTIC FLAGS")
    print("-"*50)
    
    if results.has_directional_bias:
        print("\n  [WARNING] Strategy has DIRECTIONAL BIAS")
        print("  -> PnL is significantly correlated with market direction")
        print("  -> This may indicate bets rather than pure spread capture")
    else:
        print("\n  [OK] No significant directional bias detected")
    
    if results.has_optionality:
        print("\n  [WARNING] Strategy shows OPTIONALITY pattern")
        print("  -> PnL is significantly correlated with max exposure")
        print("  -> Strategy may be accumulating directional risk")
    else:
        print("\n  [OK] No significant optionality pattern detected")
    
    print("\n4. INTERPRETATION")
    print("-"*50)
    
    if not results.has_directional_bias and not results.has_optionality:
        print("\n  [GOOD] PnL appears to be from SPREAD CAPTURE")
        print("  -> Low correlation with direction suggests true market making")
    elif results.has_directional_bias:
        print("\n  [CAUTION] PnL may be from DIRECTIONAL BETS")
        print("  -> Strategy profits when betting on direction correctly")
        print("  -> This is not pure spread capture")
    elif results.has_optionality:
        print("\n  [CAUTION] Strategy may be LONG OPTIONALITY")
        print("  -> Accumulating positions that benefit from large moves")
    
    if analysis_df is not None and len(analysis_df) > 0:
        print("\n5. PER-MARKET SUMMARY")
        print("-"*50)
        for _, row in analysis_df.iterrows():
            dir_str = "UP" if row['direction'] > 0 else "DOWN" if row['direction'] < 0 else "?"
            print(f"  {row['market_id'][:20]}: PnL=${row['pnl']:.4f}, dir={dir_str}, exp={row['max_exposure']:.1f}")
    
    print("\n" + "="*70)


def main():
    """Run optionality check."""
    from scripts.backtest.data_loader import load_eth_markets, add_derived_columns
    
    print("Loading data...")
    df, _ = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    print(f"Loaded {len(df):,} rows, {df['market_id'].nunique()} markets")
    
    # Run check
    results = run_optionality_check(df, volume_markets_only=True)


if __name__ == '__main__':
    main()

