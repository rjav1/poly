#!/usr/bin/env python3
"""
Fill Model Calibration for TOUCH_SIZE_PROXY

This module calibrates the fill model using internal market observables:
1. Analyze "size depletion events" - when size at best decreases without price change
2. Use these to estimate realistic fill rates
3. Provide upper/lower bounds using BOUNDS_ONLY model
4. Compare TOUCH_SIZE_PROXY results to bounds

Key insight: When best_bid_size decreases while best_bid price stays the same,
either trades happened or orders were cancelled. This gives us a proxy for
execution activity without needing trade tape.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class DepletionStats:
    """Statistics about size depletion events."""
    total_ticks: int
    depletion_events_bid: int  # Ticks where bid size decreased
    depletion_events_ask: int
    avg_depletion_size_bid: float
    avg_depletion_size_ask: float
    depletion_rate_bid: float  # Fraction of ticks with depletion
    depletion_rate_ask: float
    avg_touch_size_bid: float
    avg_touch_size_ask: float
    implied_fill_rate_per_second: float  # Calibrated rate


def compute_size_depletion(df: pd.DataFrame, token: str = 'UP') -> pd.DataFrame:
    """
    Compute size changes at best bid/ask for a token.
    
    Args:
        df: Market data (must have pm_{token}_best_bid, pm_{token}_best_bid_size, etc.)
        token: 'UP' or 'DOWN'
        
    Returns:
        DataFrame with depletion analysis per tick
    """
    token_lower = token.lower()
    bid_col = f'pm_{token_lower}_best_bid'
    bid_size_col = f'pm_{token_lower}_best_bid_size'
    ask_col = f'pm_{token_lower}_best_ask'
    ask_size_col = f'pm_{token_lower}_best_ask_size'
    
    result = df[['market_id', 't']].copy()
    
    # Get prices and sizes
    result['bid'] = df[bid_col]
    result['bid_size'] = df[bid_size_col]
    result['ask'] = df[ask_col]
    result['ask_size'] = df[ask_size_col]
    
    # Compute changes
    result['bid_prev'] = df.groupby('market_id')[bid_col].shift(1)
    result['bid_size_prev'] = df.groupby('market_id')[bid_size_col].shift(1)
    result['ask_prev'] = df.groupby('market_id')[ask_col].shift(1)
    result['ask_size_prev'] = df.groupby('market_id')[ask_size_col].shift(1)
    
    # Compute deltas
    result['delta_bid_size'] = result['bid_size'] - result['bid_size_prev']
    result['delta_ask_size'] = result['ask_size'] - result['ask_size_prev']
    
    # Check if price unchanged
    result['bid_price_unchanged'] = (result['bid'] == result['bid_prev']).astype(int)
    result['ask_price_unchanged'] = (result['ask'] == result['ask_prev']).astype(int)
    
    # Depletion event: size decreased AND price unchanged
    result['bid_depletion'] = (
        (result['delta_bid_size'] < 0) & 
        (result['bid_price_unchanged'] == 1)
    ).astype(int)
    
    result['ask_depletion'] = (
        (result['delta_ask_size'] < 0) & 
        (result['ask_price_unchanged'] == 1)
    ).astype(int)
    
    # Size of depletion (absolute value)
    result['bid_depletion_size'] = np.where(
        result['bid_depletion'] == 1,
        -result['delta_bid_size'],  # Make positive
        0.0
    )
    result['ask_depletion_size'] = np.where(
        result['ask_depletion'] == 1,
        -result['delta_ask_size'],
        0.0
    )
    
    return result


def analyze_depletion_stats(
    df: pd.DataFrame,
    volume_markets_only: bool = True
) -> Dict[str, DepletionStats]:
    """
    Analyze size depletion statistics for calibration.
    
    Args:
        df: Market data with size columns
        volume_markets_only: Only use markets with size data
        
    Returns:
        Dictionary with stats for UP and DOWN tokens
    """
    # Filter to volume markets if requested
    if volume_markets_only:
        volume_prefixes = [
            '20260106_1630', '20260106_1645', '20260106_1700', '20260106_1715',
            '20260106_1730', '20260106_1745', '20260106_1800', '20260106_1815',
            '20260106_1830', '20260106_1845', '20260106_1900', '20260106_1915'
        ]
        market_ids = df['market_id'].unique()
        valid_markets = [m for m in market_ids if any(m.startswith(p) for p in volume_prefixes)]
        df = df[df['market_id'].isin(valid_markets)]
    
    results = {}
    
    for token in ['UP', 'DOWN']:
        depl_df = compute_size_depletion(df, token)
        
        # Filter to rows with valid data
        valid = depl_df[depl_df['bid_size'].notna() & depl_df['bid_size_prev'].notna()]
        
        if len(valid) == 0:
            continue
        
        total_ticks = len(valid)
        depl_bid = valid['bid_depletion'].sum()
        depl_ask = valid['ask_depletion'].sum()
        
        avg_depl_bid = valid[valid['bid_depletion'] == 1]['bid_depletion_size'].mean() if depl_bid > 0 else 0
        avg_depl_ask = valid[valid['ask_depletion'] == 1]['ask_depletion_size'].mean() if depl_ask > 0 else 0
        
        rate_bid = depl_bid / total_ticks
        rate_ask = depl_ask / total_ticks
        
        avg_touch_bid = valid['bid_size'].mean()
        avg_touch_ask = valid['ask_size'].mean()
        
        # Implied fill rate: if depletion happens at rate R, and average touch size is S,
        # then the fraction of touch size that trades per second is approximately:
        # fill_rate = (depletion_rate * avg_depletion_size) / avg_touch_size
        if avg_touch_bid > 0 and avg_touch_ask > 0:
            implied_rate_bid = (rate_bid * avg_depl_bid) / avg_touch_bid
            implied_rate_ask = (rate_ask * avg_depl_ask) / avg_touch_ask
            implied_rate = (implied_rate_bid + implied_rate_ask) / 2
        else:
            implied_rate = 0.1  # Default
        
        results[token] = DepletionStats(
            total_ticks=total_ticks,
            depletion_events_bid=depl_bid,
            depletion_events_ask=depl_ask,
            avg_depletion_size_bid=avg_depl_bid,
            avg_depletion_size_ask=avg_depl_ask,
            depletion_rate_bid=rate_bid,
            depletion_rate_ask=rate_ask,
            avg_touch_size_bid=avg_touch_bid,
            avg_touch_size_ask=avg_touch_ask,
            implied_fill_rate_per_second=implied_rate,
        )
    
    return results


def run_bounds_comparison(
    df: pd.DataFrame,
    strategy: Any,
    volume_markets_only: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run strategy with different fill models to get PnL bounds.
    
    Args:
        df: Market data
        strategy: Strategy to test
        volume_markets_only: Only use volume markets
        verbose: Print progress
        
    Returns:
        Dictionary with results for each fill model
    """
    from scripts.backtest.maker_execution import MakerExecutionConfig, FillModel
    from scripts.backtest.backtest_engine import run_maker_backtest
    
    results = {}
    
    # Model 1: TOUCH_SIZE_PROXY with calibrated rate
    depl_stats = analyze_depletion_stats(df, volume_markets_only)
    if 'UP' in depl_stats:
        calibrated_rate = depl_stats['UP'].implied_fill_rate_per_second
    else:
        calibrated_rate = 0.1
    
    if verbose:
        print(f"Calibrated touch_trade_rate: {calibrated_rate:.4f}")
    
    config_proxy = MakerExecutionConfig(
        place_latency_ms=100,
        cancel_latency_ms=50,
        fill_model=FillModel.TOUCH_SIZE_PROXY,
        touch_trade_rate_per_second=calibrated_rate,
    )
    
    result_proxy = run_maker_backtest(
        df, strategy, config_proxy, 
        volume_markets_only=volume_markets_only, 
        verbose=verbose
    )
    
    results['TOUCH_SIZE_PROXY'] = {
        'config': config_proxy.describe(),
        'touch_trade_rate': calibrated_rate,
        'total_pnl': result_proxy['metrics']['total_pnl'],
        't_stat': result_proxy['metrics']['t_stat'],
        'n_fills': result_proxy['metrics']['n_fills'],
        'fill_rate': result_proxy['metrics']['fill_rate'],
        'spread_captured': result_proxy['metrics']['spread_captured_total'],
        'adverse_selection': result_proxy['metrics']['adverse_selection_total'],
    }
    
    # Model 2: BOUNDS_ONLY optimistic (fill at touch)
    config_optimistic = MakerExecutionConfig(
        place_latency_ms=100,
        cancel_latency_ms=50,
        fill_model=FillModel.BOUNDS_ONLY,
        bounds_fill_at_touch=True,
        bounds_fill_on_price_through=False,
    )
    
    if verbose:
        print("\nRunning BOUNDS_ONLY (optimistic)...")
    
    result_opt = run_maker_backtest(
        df, strategy, config_optimistic,
        volume_markets_only=volume_markets_only,
        verbose=False
    )
    
    results['BOUNDS_OPTIMISTIC'] = {
        'config': config_optimistic.describe(),
        'total_pnl': result_opt['metrics']['total_pnl'],
        't_stat': result_opt['metrics']['t_stat'],
        'n_fills': result_opt['metrics']['n_fills'],
        'fill_rate': result_opt['metrics']['fill_rate'],
        'spread_captured': result_opt['metrics']['spread_captured_total'],
        'adverse_selection': result_opt['metrics']['adverse_selection_total'],
    }
    
    # Model 3: BOUNDS_ONLY pessimistic (fill only on price through)
    config_pessimistic = MakerExecutionConfig(
        place_latency_ms=100,
        cancel_latency_ms=50,
        fill_model=FillModel.BOUNDS_ONLY,
        bounds_fill_at_touch=False,
        bounds_fill_on_price_through=True,
    )
    
    if verbose:
        print("Running BOUNDS_ONLY (pessimistic)...")
    
    result_pess = run_maker_backtest(
        df, strategy, config_pessimistic,
        volume_markets_only=volume_markets_only,
        verbose=False
    )
    
    results['BOUNDS_PESSIMISTIC'] = {
        'config': config_pessimistic.describe(),
        'total_pnl': result_pess['metrics']['total_pnl'],
        't_stat': result_pess['metrics']['t_stat'],
        'n_fills': result_pess['metrics']['n_fills'],
        'fill_rate': result_pess['metrics']['fill_rate'],
        'spread_captured': result_pess['metrics']['spread_captured_total'],
        'adverse_selection': result_pess['metrics']['adverse_selection_total'],
    }
    
    # Summary
    results['summary'] = {
        'pnl_range': (
            results['BOUNDS_PESSIMISTIC']['total_pnl'],
            results['BOUNDS_OPTIMISTIC']['total_pnl']
        ),
        'fill_rate_range': (
            results['BOUNDS_PESSIMISTIC']['fill_rate'],
            results['BOUNDS_OPTIMISTIC']['fill_rate']
        ),
        'proxy_within_bounds': (
            results['BOUNDS_PESSIMISTIC']['total_pnl'] <= 
            results['TOUCH_SIZE_PROXY']['total_pnl'] <= 
            results['BOUNDS_OPTIMISTIC']['total_pnl']
        ),
        'pnl_sign_robust': (
            np.sign(results['BOUNDS_PESSIMISTIC']['total_pnl']) == 
            np.sign(results['BOUNDS_OPTIMISTIC']['total_pnl'])
        ),
    }
    
    return results


def run_l2_queue_comparison(
    df: pd.DataFrame,
    strategy: Any,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run strategy with L2_QUEUE fill model in 3 modes: conservative, base, optimistic.
    
    This uses the 6-level orderbook data to track queue consumption more accurately
    than TOUCH_SIZE_PROXY.
    
    Args:
        df: Market data (must have 6-level columns)
        strategy: Strategy to test
        verbose: Print progress
        
    Returns:
        Dictionary with results for each L2 mode
    """
    from scripts.backtest.maker_execution import MakerExecutionConfig, FillModel
    from scripts.backtest.backtest_engine import run_maker_backtest
    
    results = {}
    
    # Check if 6-level data is available
    l2_cols = [c for c in df.columns if 'bid_2' in c or 'ask_2' in c]
    if not l2_cols:
        if verbose:
            print("[WARN] 6-level data not found. Skipping L2_QUEUE models.")
        return {'error': 'No 6-level data available'}
    
    # Model 1: L2_QUEUE Conservative
    # Only count consumption when price stays visible in top 6
    config_conservative = MakerExecutionConfig(
        place_latency_ms=100,
        cancel_latency_ms=50,
        fill_model=FillModel.L2_QUEUE,
        l2_conservative_mode=True,
        l2_allow_level_drift=False,
        l2_optimistic_disappear=False,
    )
    
    if verbose:
        print("Running L2_QUEUE (conservative)...")
    
    result_cons = run_maker_backtest(df, strategy, config_conservative, verbose=False)
    
    results['L2_CONSERVATIVE'] = {
        'config': config_conservative.describe(),
        'total_pnl': result_cons['metrics']['total_pnl'],
        't_stat': result_cons['metrics']['t_stat'],
        'n_fills': result_cons['metrics']['n_fills'],
        'fill_rate': result_cons['metrics']['fill_rate'],
        'spread_captured': result_cons['metrics']['spread_captured_total'],
        'adverse_selection': result_cons['metrics']['adverse_selection_total'],
    }
    
    # Model 2: L2_QUEUE Base
    # Allow tracking when price moves between levels
    config_base = MakerExecutionConfig(
        place_latency_ms=100,
        cancel_latency_ms=50,
        fill_model=FillModel.L2_QUEUE,
        l2_conservative_mode=False,
        l2_allow_level_drift=True,
        l2_optimistic_disappear=False,
    )
    
    if verbose:
        print("Running L2_QUEUE (base)...")
    
    result_base = run_maker_backtest(df, strategy, config_base, verbose=False)
    
    results['L2_BASE'] = {
        'config': config_base.describe(),
        'total_pnl': result_base['metrics']['total_pnl'],
        't_stat': result_base['metrics']['t_stat'],
        'n_fills': result_base['metrics']['n_fills'],
        'fill_rate': result_base['metrics']['fill_rate'],
        'spread_captured': result_base['metrics']['spread_captured_total'],
        'adverse_selection': result_base['metrics']['adverse_selection_total'],
    }
    
    # Model 3: L2_QUEUE Optimistic
    # Assume disappearance = full consumption
    config_optimistic = MakerExecutionConfig(
        place_latency_ms=100,
        cancel_latency_ms=50,
        fill_model=FillModel.L2_QUEUE,
        l2_conservative_mode=False,
        l2_allow_level_drift=True,
        l2_optimistic_disappear=True,
    )
    
    if verbose:
        print("Running L2_QUEUE (optimistic)...")
    
    result_opt = run_maker_backtest(df, strategy, config_optimistic, verbose=False)
    
    results['L2_OPTIMISTIC'] = {
        'config': config_optimistic.describe(),
        'total_pnl': result_opt['metrics']['total_pnl'],
        't_stat': result_opt['metrics']['t_stat'],
        'n_fills': result_opt['metrics']['n_fills'],
        'fill_rate': result_opt['metrics']['fill_rate'],
        'spread_captured': result_opt['metrics']['spread_captured_total'],
        'adverse_selection': result_opt['metrics']['adverse_selection_total'],
    }
    
    # Summary
    pnl_cons = results['L2_CONSERVATIVE']['total_pnl']
    pnl_base = results['L2_BASE']['total_pnl']
    pnl_opt = results['L2_OPTIMISTIC']['total_pnl']
    
    pnl_min = min(pnl_cons, pnl_base, pnl_opt)
    pnl_max = max(pnl_cons, pnl_base, pnl_opt)
    
    # Check if sign is robust (all same sign)
    signs = [np.sign(pnl_cons), np.sign(pnl_base), np.sign(pnl_opt)]
    sign_robust = len(set(signs)) == 1
    
    # Check if conservative and base agree on sign (primary robustness check)
    conservative_base_agree = np.sign(pnl_cons) == np.sign(pnl_base)
    
    results['summary'] = {
        'pnl_range': (pnl_min, pnl_max),
        'pnl_range_abs': pnl_max - pnl_min,
        'fill_rate_range': (
            results['L2_CONSERVATIVE']['fill_rate'],
            results['L2_OPTIMISTIC']['fill_rate'],
        ),
        'pnl_sign_robust': sign_robust,
        'conservative_base_agree': conservative_base_agree,
        'all_positive': pnl_min > 0,
        'all_negative': pnl_max < 0,
    }
    
    return results


def print_l2_comparison_report(results: Dict[str, Any]):
    """Print L2 queue model comparison report."""
    print("\n" + "="*70)
    print("L2 QUEUE MODEL COMPARISON REPORT")
    print("="*70)
    
    if 'error' in results:
        print(f"\n[ERROR] {results['error']}")
        return
    
    print("\n1. L2 FILL MODEL RESULTS")
    print("-"*50)
    
    for model in ['L2_CONSERVATIVE', 'L2_BASE', 'L2_OPTIMISTIC']:
        if model not in results:
            continue
        r = results[model]
        print(f"\n  {model}:")
        print(f"    Total PnL: ${r['total_pnl']:.4f}")
        print(f"    t-stat: {r['t_stat']:.2f}")
        print(f"    Fills: {r['n_fills']}")
        print(f"    Fill rate: {r['fill_rate']*100:.2f}%")
        print(f"    Spread captured: ${r['spread_captured']:.4f}")
        print(f"    Adverse selection: ${r['adverse_selection']:.4f}")
    
    summary = results.get('summary', {})
    
    print("\n2. ROBUSTNESS SUMMARY")
    print("-"*50)
    
    pnl_min, pnl_max = summary.get('pnl_range', (0, 0))
    fill_min, fill_max = summary.get('fill_rate_range', (0, 0))
    
    print(f"\n  PnL Range: ${pnl_min:.4f} to ${pnl_max:.4f}")
    print(f"  PnL Range (absolute): ${summary.get('pnl_range_abs', 0):.4f}")
    print(f"  Fill Rate Range: {fill_min*100:.2f}% to {fill_max*100:.2f}%")
    
    print(f"\n  Sign Robust (all 3 modes): {summary.get('pnl_sign_robust', False)}")
    print(f"  Conservative/Base Agree: {summary.get('conservative_base_agree', False)}")
    
    if summary.get('all_positive'):
        print("\n  [PASS] PnL is POSITIVE under all L2 fill models")
    elif summary.get('all_negative'):
        print("\n  [FAIL] PnL is NEGATIVE under all L2 fill models")
    else:
        print("\n  [WARN] PnL sign depends on fill model assumptions")
    
    # Key insight
    print("\n3. KEY INSIGHT")
    print("-"*50)
    
    if summary.get('conservative_base_agree'):
        if pnl_min > 0:
            print("\n  Strategy shows ROBUST positive edge.")
            print("  Conservative and base models agree on profitability.")
        else:
            print("\n  Strategy shows ROBUST negative performance.")
            print("  Conservative and base models agree on losses.")
    else:
        print("\n  Strategy edge is NOT robust to fill assumptions.")
        print("  RECOMMENDATION: Collect trade tape data for better validation.")
    
    # Comparison to L1-only models
    pnl_range_abs = summary.get('pnl_range_abs', 0)
    if pnl_range_abs < 50:
        print(f"\n  L2 PnL range (${pnl_range_abs:.2f}) is reasonable.")
    else:
        print(f"\n  L2 PnL range (${pnl_range_abs:.2f}) is still wide.")
        print("  L2 queue tracking helps but doesn't eliminate uncertainty.")


def run_full_fill_model_comparison(
    df: pd.DataFrame,
    strategy: Any,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run comprehensive fill model comparison including both L1 and L2 models.
    
    Args:
        df: Market data
        strategy: Strategy to test
        verbose: Print progress
        
    Returns:
        Combined results from all fill models
    """
    results = {
        'strategy': strategy.name,
    }
    
    # Run L1-based bounds comparison
    if verbose:
        print("\n--- L1 FILL MODELS (TOUCH_SIZE_PROXY, BOUNDS) ---")
    l1_results = run_bounds_comparison(df, strategy, volume_markets_only=True, verbose=verbose)
    results['l1_models'] = l1_results
    
    # Run L2-based comparison
    if verbose:
        print("\n--- L2 FILL MODELS (L2_QUEUE) ---")
    l2_results = run_l2_queue_comparison(df, strategy, verbose=verbose)
    results['l2_models'] = l2_results
    
    # Combined summary
    if verbose:
        print("\n--- COMBINED SUMMARY ---")
    
    l1_summary = l1_results.get('summary', {})
    l2_summary = l2_results.get('summary', {}) if 'error' not in l2_results else {}
    
    l1_pnl_range = l1_summary.get('pnl_range', (0, 0))
    l2_pnl_range = l2_summary.get('pnl_range', (0, 0))
    
    # Get overall min/max
    all_pnls = []
    for r in l1_results.values():
        if isinstance(r, dict) and 'total_pnl' in r:
            all_pnls.append(r['total_pnl'])
    if 'error' not in l2_results:
        for r in l2_results.values():
            if isinstance(r, dict) and 'total_pnl' in r:
                all_pnls.append(r['total_pnl'])
    
    if all_pnls:
        overall_min = min(all_pnls)
        overall_max = max(all_pnls)
        signs = [np.sign(p) for p in all_pnls]
        overall_sign_robust = len(set(signs)) == 1
    else:
        overall_min = overall_max = 0
        overall_sign_robust = False
    
    results['combined_summary'] = {
        'l1_pnl_range': l1_pnl_range,
        'l2_pnl_range': l2_pnl_range,
        'overall_pnl_range': (overall_min, overall_max),
        'overall_pnl_range_abs': overall_max - overall_min,
        'l1_sign_robust': l1_summary.get('pnl_sign_robust', False),
        'l2_sign_robust': l2_summary.get('pnl_sign_robust', False),
        'overall_sign_robust': overall_sign_robust,
        'l2_improved_bounds': (
            l2_pnl_range[1] - l2_pnl_range[0] < l1_pnl_range[1] - l1_pnl_range[0]
            if l2_pnl_range[0] != l2_pnl_range[1] else False
        ),
    }
    
    if verbose:
        print(f"  Overall PnL Range: ${overall_min:.4f} to ${overall_max:.4f}")
        print(f"  Overall Range (abs): ${overall_max - overall_min:.4f}")
        print(f"  L1 Range: ${l1_pnl_range[1] - l1_pnl_range[0]:.4f}")
        if l2_pnl_range:
            print(f"  L2 Range: ${l2_pnl_range[1] - l2_pnl_range[0]:.4f}")
            if results['combined_summary']['l2_improved_bounds']:
                print("  [GOOD] L2 model reduced PnL uncertainty")
        print(f"  Sign Robust: {overall_sign_robust}")
    
    return results


def print_calibration_report(
    depl_stats: Dict[str, DepletionStats],
    bounds_results: Dict[str, Any],
):
    """Print a calibration report."""
    print("\n" + "="*70)
    print("FILL MODEL CALIBRATION REPORT")
    print("="*70)
    
    print("\n1. SIZE DEPLETION ANALYSIS")
    print("-"*50)
    
    for token, stats in depl_stats.items():
        print(f"\n  {token} Token:")
        print(f"    Total ticks analyzed: {stats.total_ticks:,}")
        print(f"    Bid depletion events: {stats.depletion_events_bid:,} ({stats.depletion_rate_bid*100:.2f}%)")
        print(f"    Ask depletion events: {stats.depletion_events_ask:,} ({stats.depletion_rate_ask*100:.2f}%)")
        print(f"    Avg bid depletion size: {stats.avg_depletion_size_bid:.2f}")
        print(f"    Avg ask depletion size: {stats.avg_depletion_size_ask:.2f}")
        print(f"    Avg touch size (bid): {stats.avg_touch_size_bid:.2f}")
        print(f"    Avg touch size (ask): {stats.avg_touch_size_ask:.2f}")
        print(f"    Implied fill rate/sec: {stats.implied_fill_rate_per_second:.4f}")
    
    print("\n2. FILL MODEL COMPARISON")
    print("-"*50)
    
    for model, result in bounds_results.items():
        if model == 'summary':
            continue
        print(f"\n  {model}:")
        print(f"    Total PnL: ${result['total_pnl']:.4f}")
        print(f"    t-stat: {result['t_stat']:.2f}")
        print(f"    Fills: {result['n_fills']}")
        print(f"    Fill rate: {result['fill_rate']*100:.2f}%")
    
    print("\n3. SUMMARY")
    print("-"*50)
    
    summary = bounds_results['summary']
    pnl_lo, pnl_hi = summary['pnl_range']
    fill_lo, fill_hi = summary['fill_rate_range']
    
    print(f"\n  PnL Range: ${pnl_lo:.4f} to ${pnl_hi:.4f}")
    print(f"  Fill Rate Range: {fill_lo*100:.2f}% to {fill_hi*100:.2f}%")
    print(f"  TOUCH_SIZE_PROXY within bounds: {summary['proxy_within_bounds']}")
    print(f"  PnL sign robust: {summary['pnl_sign_robust']}")
    
    if summary['pnl_sign_robust']:
        if pnl_lo > 0:
            print("\n  [OK] PnL is POSITIVE under all fill models")
        elif pnl_hi < 0:
            print("\n  [WARNING] PnL is NEGATIVE under all fill models")
        else:
            print("\n  [WARNING] PnL sign depends on fill model assumptions")
    else:
        print("\n  [WARNING] PnL sign is NOT robust - varies by fill model!")
    
    # Recommended touch_trade_rate
    if 'UP' in depl_stats:
        rec_rate = depl_stats['UP'].implied_fill_rate_per_second
        print(f"\n4. RECOMMENDED PARAMETERS")
        print("-"*50)
        print(f"  touch_trade_rate_per_second: {rec_rate:.4f}")
        print(f"  (Based on observed size depletion patterns)")
        
        # Upper/lower bounds
        upper_rate = min(0.5, rec_rate * 2)  # All depletions are trades
        lower_rate = rec_rate * 0.5  # Only half of depletions are trades
        print(f"  Upper bound (all depletions are trades): {upper_rate:.4f}")
        print(f"  Lower bound (half are trades): {lower_rate:.4f}")


def main():
    """Run calibration analysis."""
    from scripts.backtest.data_loader import load_eth_markets, load_6level_markets, add_derived_columns
    from scripts.backtest.strategies import SpreadCaptureStrategy
    
    print("Loading data (6-level)...")
    try:
        df, _ = load_6level_markets(min_coverage=90.0)
    except Exception:
        print("6-level data not available, falling back to standard data...")
        df, _ = load_eth_markets(min_coverage=90.0)
    
    df = add_derived_columns(df)
    
    print(f"Loaded {len(df):,} rows, {df['market_id'].nunique()} markets")
    
    # Check for 6-level columns
    l2_cols = [c for c in df.columns if 'bid_2' in c or 'ask_2' in c]
    print(f"6-level columns present: {len(l2_cols)}")
    
    # Analyze depletion
    print("\nAnalyzing size depletion patterns...")
    depl_stats = analyze_depletion_stats(df, volume_markets_only=False)
    
    # Create strategy
    strategy = SpreadCaptureStrategy(
        spread_min=0.01,
        tau_min=60,
        tau_max=600,
        inventory_limit_up=10.0,
        inventory_limit_down=10.0,
    )
    
    # Run L1 bounds comparison
    print("\nRunning L1 fill model comparison...")
    bounds_results = run_bounds_comparison(df, strategy, volume_markets_only=False)
    
    # Print L1 report
    print_calibration_report(depl_stats, bounds_results)
    
    # Run L2 comparison if data available
    if l2_cols:
        print("\n\n" + "="*70)
        print("RUNNING L2 QUEUE MODEL COMPARISON")
        print("="*70)
        l2_results = run_l2_queue_comparison(df, strategy, verbose=True)
        print_l2_comparison_report(l2_results)
    else:
        print("\n[SKIP] L2 queue comparison - no 6-level data available")


if __name__ == '__main__':
    main()

