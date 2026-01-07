#!/usr/bin/env python3
"""
L6 Depth-Aware Validation Suite

This script runs the complete validation suite for the mispricing strategy
with L6 depth-aware execution at different order sizes:
- tiny: q=5 (fits in L1)  
- medium: q=25 (walks to L2-L3)
- large: q=100 (walks to L6, sometimes rejects)

The goal is to determine if the strategy edge survives at scale.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest.data_loader import add_derived_columns
from scripts.backtest.fair_value import BinnedFairValueModel
from scripts.backtest.strategies import MispricingBasedStrategy, Signal
from scripts.backtest.backtest_engine import run_backtest, ExecutionConfig
from scripts.backtest.execution_model import (
    get_effective_prices_with_depth,
    walk_the_book,
    DepthEffectivePrices,
)
from scripts.backtest.capacity_model import (
    run_capacity_analysis,
    compute_max_tradable_size,
    compute_capacity_curve,
    filter_signals_by_capacity,
    CapacityResult,
)

OUTPUT_DIR = PROJECT_ROOT / 'data_v2' / 'backtest_results' / 'l6_validation'

# Size tiers for validation
SIZE_TIERS = {
    'tiny': 5,      # Fits in L1
    'medium': 25,   # Walks L2-L3
    'large': 100,   # Walks L6+
}


def run_backtest_with_depth(
    df: pd.DataFrame,
    strategy: MispricingBasedStrategy,
    size: float,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run backtest with depth-aware execution at specified size.
    
    Args:
        df: DataFrame with L6 columns
        strategy: Strategy object
        size: Order size
        verbose: Print progress
        
    Returns:
        Backtest results dict with depth info
    """
    # Generate signals
    all_signals = []
    for market_id in df['market_id'].unique():
        market_df = df[df['market_id'] == market_id]
        signals = strategy.generate_signals(market_df)
        all_signals.extend(signals)
    
    if not all_signals:
        return {
            'metrics': {
                'n_trades': 0,
                'n_markets': 0,
                'total_pnl': 0,
                't_stat': 0,
                'hit_rate_per_trade': 0,
            },
            'trades': [],
            'size': size,
        }
    
    # Execute with depth-aware pricing
    trades = []
    market_pnls = {}
    
    for signal in all_signals:
        market_df = df[df['market_id'] == signal.market_id]
        
        # Get entry row
        entry_rows = market_df[market_df['t'] == signal.entry_t]
        if entry_rows.empty:
            continue
        entry_row = entry_rows.iloc[0]
        
        # Get exit row (at expiry or signal.exit_t)
        exit_t = min(signal.exit_t, market_df['t'].max())
        exit_rows = market_df[market_df['t'] == exit_t]
        if exit_rows.empty:
            exit_rows = market_df.iloc[-1:]
        exit_row = exit_rows.iloc[0]
        
        # Get depth-aware entry price
        entry_prices = get_effective_prices_with_depth(entry_row, size)
        exit_prices = get_effective_prices_with_depth(exit_row, size)
        
        if signal.side == 'buy_up':
            entry_price = entry_prices.buy_up_vwap
            exit_price = exit_prices.sell_up_vwap
            entry_complete = entry_prices.buy_up_complete
            entry_levels = entry_prices.buy_up_levels
        elif signal.side == 'buy_down':
            entry_price = entry_prices.buy_down_vwap
            exit_price = exit_prices.sell_down_vwap
            entry_complete = entry_prices.buy_down_complete
            entry_levels = entry_prices.buy_down_levels
        else:
            continue
        
        # Skip if can't fill
        if pd.isna(entry_price) or pd.isna(exit_price):
            continue
        
        # Compute PnL (per share)
        pnl_per_share = exit_price - entry_price
        pnl_total = pnl_per_share * size
        
        trade = {
            'market_id': signal.market_id,
            'entry_t': signal.entry_t,
            'exit_t': exit_t,
            'side': signal.side,
            'size': size,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_per_share': pnl_per_share,
            'pnl': pnl_total,
            'entry_complete': entry_complete,
            'entry_levels': entry_levels,
        }
        trades.append(trade)
        
        if signal.market_id not in market_pnls:
            market_pnls[signal.market_id] = 0
        market_pnls[signal.market_id] += pnl_total
    
    # Compute metrics
    if trades:
        pnls = [t['pnl'] for t in trades]
        market_pnl_list = list(market_pnls.values())
        
        n_trades = len(trades)
        n_markets = len(market_pnls)
        total_pnl = sum(pnls)
        mean_pnl = np.mean(market_pnl_list)
        std_pnl = np.std(market_pnl_list, ddof=1) if len(market_pnl_list) > 1 else 0
        t_stat = mean_pnl / (std_pnl / np.sqrt(n_markets)) if std_pnl > 0 else 0
        hit_rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        
        # Depth metrics
        avg_levels = np.mean([t['entry_levels'] for t in trades])
        pct_complete = sum(1 for t in trades if t['entry_complete']) / len(trades) * 100
    else:
        n_trades = n_markets = 0
        total_pnl = mean_pnl = std_pnl = t_stat = hit_rate = 0
        avg_levels = pct_complete = 0
    
    return {
        'metrics': {
            'n_trades': n_trades,
            'n_markets': n_markets,
            'total_pnl': total_pnl,
            'mean_pnl_per_market': mean_pnl,
            'std_pnl_per_market': std_pnl,
            't_stat': t_stat,
            'hit_rate_per_trade': hit_rate,
            'avg_entry_levels': avg_levels,
            'pct_complete_fills': pct_complete,
        },
        'trades': trades,
        'size': size,
    }


def run_size_sweep(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model: BinnedFairValueModel,
    strategy_params: Dict[str, Any],
    sizes: List[float] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run strategy at different order sizes.
    
    Args:
        train_df: Training data
        test_df: Test data
        model: Fitted fair value model
        strategy_params: Strategy parameters (buffer, tau_max, etc.)
        sizes: List of sizes to test
        verbose: Print progress
        
    Returns:
        Results dict with per-size metrics
    """
    if sizes is None:
        sizes = [1, 5, 10, 25, 50, 100, 200, 500]
    
    if verbose:
        print(f"\n{'='*70}")
        print("SIZE SWEEP")
        print(f"{'='*70}")
        print(f"Sizes: {sizes}")
    
    results = {}
    
    for size in sizes:
        strategy = MispricingBasedStrategy(
            fair_value_model=model,
            **strategy_params
        )
        
        result = run_backtest_with_depth(test_df, strategy, size)
        metrics = result['metrics']
        
        results[size] = {
            'n_trades': metrics['n_trades'],
            'total_pnl': metrics['total_pnl'],
            't_stat': metrics['t_stat'],
            'hit_rate': metrics['hit_rate_per_trade'],
            'avg_levels': metrics['avg_entry_levels'],
            'pct_complete': metrics['pct_complete_fills'],
        }
        
        if verbose:
            print(f"  Size {size:>4}: PnL=${metrics['total_pnl']:>8.2f}, "
                  f"t={metrics['t_stat']:>5.2f}, "
                  f"trades={metrics['n_trades']:>3}, "
                  f"levels={metrics['avg_entry_levels']:.1f}, "
                  f"complete={metrics['pct_complete_fills']:.0f}%")
    
    return results


def run_tiered_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model: BinnedFairValueModel,
    strategy_params: Dict[str, Any],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run validation at tiny/medium/large size tiers.
    
    Returns comparison of edge survival across tiers.
    """
    if verbose:
        print(f"\n{'='*70}")
        print("TIERED VALIDATION (Tiny / Medium / Large)")
        print(f"{'='*70}")
    
    results = {}
    
    for tier_name, size in SIZE_TIERS.items():
        strategy = MispricingBasedStrategy(
            fair_value_model=model,
            **strategy_params
        )
        
        result = run_backtest_with_depth(test_df, strategy, size)
        metrics = result['metrics']
        
        results[tier_name] = {
            'size': size,
            'n_trades': metrics['n_trades'],
            'total_pnl': metrics['total_pnl'],
            'mean_pnl_per_market': metrics['mean_pnl_per_market'],
            't_stat': metrics['t_stat'],
            'hit_rate': metrics['hit_rate_per_trade'],
            'avg_levels': metrics['avg_entry_levels'],
            'pct_complete': metrics['pct_complete_fills'],
            'edge_survives': metrics['t_stat'] > 2.0,
            'edge_marginal': metrics['t_stat'] > 1.5,
        }
        
        if verbose:
            m = results[tier_name]
            status = "[PASS]" if m['edge_survives'] else "[WARN]" if m['edge_marginal'] else "[FAIL]"
            print(f"\n{tier_name.upper()} (size={size}):")
            print(f"  PnL: ${m['total_pnl']:.2f}")
            print(f"  t-stat: {m['t_stat']:.2f} {status}")
            print(f"  Trades: {m['n_trades']}")
            print(f"  Avg levels: {m['avg_levels']:.1f}")
            print(f"  Complete fills: {m['pct_complete']:.0f}%")
    
    return results


def run_slippage_monte_carlo_with_depth(
    test_df: pd.DataFrame,
    model: BinnedFairValueModel,
    strategy_params: Dict[str, Any],
    size: float,
    n_simulations: int = 200,
    slippage_range: tuple = (0, 0.015),
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Monte Carlo simulation with random slippage at specified size.
    
    Slippage is added on top of depth-aware fill prices.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"SLIPPAGE MONTE CARLO (size={size})")
        print(f"{'='*70}")
        print(f"Simulations: {n_simulations}")
        print(f"Slippage range: {slippage_range}")
    
    # First run baseline (no extra slippage)
    strategy = MispricingBasedStrategy(
        fair_value_model=model,
        **strategy_params
    )
    baseline = run_backtest_with_depth(test_df, strategy, size)
    baseline_pnl = baseline['metrics']['total_pnl']
    baseline_t = baseline['metrics']['t_stat']
    
    if verbose:
        print(f"\nBaseline: PnL=${baseline_pnl:.2f}, t={baseline_t:.2f}")
    
    # Monte Carlo
    mc_pnls = []
    mc_tstats = []
    
    for sim in range(n_simulations):
        # Add random slippage to each trade
        trades = baseline['trades'].copy()
        market_pnls = {}
        
        for trade in trades:
            # Random slippage (uniform)
            slippage = np.random.uniform(*slippage_range)
            
            # Apply slippage (increases entry price for buys)
            if trade['side'] in ['buy_up', 'buy_down']:
                adjusted_entry = trade['entry_price'] + slippage
            else:
                adjusted_entry = trade['entry_price'] - slippage
            
            pnl = (trade['exit_price'] - adjusted_entry) * size
            
            if trade['market_id'] not in market_pnls:
                market_pnls[trade['market_id']] = 0
            market_pnls[trade['market_id']] += pnl
        
        # Compute t-stat for this simulation
        if market_pnls:
            pnl_list = list(market_pnls.values())
            total_pnl = sum(pnl_list)
            mean_pnl = np.mean(pnl_list)
            std_pnl = np.std(pnl_list, ddof=1) if len(pnl_list) > 1 else 0
            t_stat = mean_pnl / (std_pnl / np.sqrt(len(pnl_list))) if std_pnl > 0 else 0
            
            mc_pnls.append(total_pnl)
            mc_tstats.append(t_stat)
    
    # Compute stats
    results = {
        'size': size,
        'n_simulations': n_simulations,
        'slippage_range': slippage_range,
        'baseline_pnl': baseline_pnl,
        'baseline_t': baseline_t,
        'mc_pnl_mean': np.mean(mc_pnls) if mc_pnls else 0,
        'mc_pnl_median': np.median(mc_pnls) if mc_pnls else 0,
        'mc_pnl_p5': np.percentile(mc_pnls, 5) if mc_pnls else 0,
        'mc_t_mean': np.mean(mc_tstats) if mc_tstats else 0,
        'mc_t_median': np.median(mc_tstats) if mc_tstats else 0,
        'prob_pnl_positive': np.mean([p > 0 for p in mc_pnls]) * 100 if mc_pnls else 0,
        'prob_t_gt_2': np.mean([t > 2 for t in mc_tstats]) * 100 if mc_tstats else 0,
        'prob_t_gt_1_5': np.mean([t > 1.5 for t in mc_tstats]) * 100 if mc_tstats else 0,
    }
    
    if verbose:
        print(f"\nResults:")
        print(f"  P(PnL > 0): {results['prob_pnl_positive']:.1f}%")
        print(f"  P(t > 2.0): {results['prob_t_gt_2']:.1f}%")
        print(f"  P(t > 1.5): {results['prob_t_gt_1_5']:.1f}%")
        print(f"  Median PnL: ${results['mc_pnl_median']:.2f}")
        print(f"  Median t-stat: {results['mc_t_median']:.2f}")
    
    return results


def run_l6_validation_suite(
    data_path: str = 'data_v2/research_6levels/canonical_dataset_all_assets.parquet',
    output_dir: Path = OUTPUT_DIR,
    strategy_params: Dict[str, Any] = None,
    n_slippage_sims: int = 200,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete L6 validation suite.
    
    Includes:
    1. Capacity analysis
    2. Size sweep
    3. Tiered validation (tiny/medium/large)
    4. Slippage Monte Carlo at each tier
    """
    if strategy_params is None:
        strategy_params = {
            'buffer': 0.02,
            'tau_max': 420,
            'exit_rule': 'expiry',
        }
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("L6 DEPTH-AWARE VALIDATION SUITE")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Output: {output_dir}")
    
    # Load data
    print("\nLoading L6 data...")
    df = pd.read_parquet(data_path)
    df = add_derived_columns(df)
    
    # Check for L6 columns
    l6_cols = [c for c in df.columns if 'bid_2' in c or 'ask_2' in c]
    if not l6_cols:
        raise ValueError("No L6 columns found in dataset!")
    print(f"L6 columns found: {len(l6_cols)}")
    
    # Split chronologically
    markets = sorted(df['market_id'].unique())
    n_train = int(len(markets) * 0.7)
    train_markets = markets[:n_train]
    test_markets = markets[n_train:]
    train_df = df[df['market_id'].isin(train_markets)].copy()
    test_df = df[df['market_id'].isin(test_markets)].copy()
    
    print(f"Train: {len(train_markets)} markets ({len(train_df):,} obs)")
    print(f"Test: {len(test_markets)} markets ({len(test_df):,} obs)")
    
    # Train model
    print("\nTraining fair value model...")
    model = BinnedFairValueModel(sample_every=5)
    model.fit(train_df)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'data_path': str(data_path),
        'n_train_markets': len(train_markets),
        'n_test_markets': len(test_markets),
        'strategy_params': strategy_params,
    }
    
    # 1. Capacity Analysis
    print("\n" + "="*70)
    print("1. CAPACITY ANALYSIS")
    print("="*70)
    
    strategy = MispricingBasedStrategy(
        fair_value_model=model,
        **strategy_params
    )
    
    all_signals = []
    for market_id in test_df['market_id'].unique():
        market_df = test_df[test_df['market_id'] == market_id]
        signals = strategy.generate_signals(market_df)
        all_signals.extend(signals)
    
    print(f"Generated {len(all_signals)} signals")
    
    if all_signals:
        capacity_results = run_capacity_analysis(
            test_df, all_signals, model,
            buffer=strategy_params.get('buffer', 0.02),
            verbose=verbose
        )
        results['capacity_analysis'] = {
            'q_star_distribution': capacity_results['q_star_distribution'],
            'survival_rates': capacity_results['survival_rates'],
            'avg_pnl_by_size': capacity_results['avg_pnl_by_size'],
        }
    
    # 2. Size Sweep
    print("\n" + "="*70)
    print("2. SIZE SWEEP")
    print("="*70)
    
    size_sweep = run_size_sweep(
        train_df, test_df, model, strategy_params,
        sizes=[1, 5, 10, 25, 50, 100, 200, 500],
        verbose=verbose
    )
    results['size_sweep'] = size_sweep
    
    # 3. Tiered Validation
    print("\n" + "="*70)
    print("3. TIERED VALIDATION")
    print("="*70)
    
    tiered = run_tiered_validation(
        train_df, test_df, model, strategy_params,
        verbose=verbose
    )
    results['tiered_validation'] = tiered
    
    # 4. Slippage Monte Carlo at each tier
    print("\n" + "="*70)
    print("4. SLIPPAGE MONTE CARLO (by tier)")
    print("="*70)
    
    slippage_results = {}
    for tier_name, size in SIZE_TIERS.items():
        slippage_results[tier_name] = run_slippage_monte_carlo_with_depth(
            test_df, model, strategy_params,
            size=size,
            n_simulations=n_slippage_sims,
            verbose=verbose
        )
    results['slippage_monte_carlo'] = slippage_results
    
    # Save results
    results_path = output_dir / 'l6_validation_results.json'
    
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Generate report
    generate_l6_report(results, output_dir)
    
    return results


def generate_l6_report(results: Dict[str, Any], output_dir: Path):
    """Generate markdown report."""
    report_path = output_dir / 'L6_VALIDATION_REPORT.md'
    
    with open(report_path, 'w') as f:
        f.write("# L6 Depth-Aware Validation Report\n\n")
        f.write(f"Generated: {results.get('timestamp', 'N/A')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Train markets: {results.get('n_train_markets', 'N/A')}\n")
        f.write(f"- Test markets: {results.get('n_test_markets', 'N/A')}\n")
        f.write(f"- Strategy params: {results.get('strategy_params', {})}\n\n")
        
        # Capacity Analysis
        if 'capacity_analysis' in results:
            f.write("## Capacity Analysis\n\n")
            ca = results['capacity_analysis']
            
            f.write("### q* Distribution\n\n")
            qs = ca.get('q_star_distribution', {})
            f.write(f"- Mean: {qs.get('mean', 0):.1f}\n")
            f.write(f"- Median: {qs.get('median', 0):.1f}\n")
            f.write(f"- 25th-75th: [{qs.get('p25', 0):.1f}, {qs.get('p75', 0):.1f}]\n\n")
            
            f.write("### Survival Rates\n\n")
            sr = ca.get('survival_rates', {})
            f.write("| Min q* | Survival Rate |\n")
            f.write("|--------|---------------|\n")
            for min_q, rate in sorted(sr.items()):
                f.write(f"| {min_q} | {rate:.1f}% |\n")
            f.write("\n")
        
        # Tiered Validation
        if 'tiered_validation' in results:
            f.write("## Tiered Validation\n\n")
            f.write("| Tier | Size | PnL | t-stat | Status |\n")
            f.write("|------|------|-----|--------|--------|\n")
            
            for tier, data in results['tiered_validation'].items():
                status = "PASS" if data.get('edge_survives') else "WARN" if data.get('edge_marginal') else "FAIL"
                f.write(f"| {tier} | {data.get('size', 0)} | ${data.get('total_pnl', 0):.2f} | "
                       f"{data.get('t_stat', 0):.2f} | {status} |\n")
            f.write("\n")
        
        # Slippage Monte Carlo
        if 'slippage_monte_carlo' in results:
            f.write("## Slippage Monte Carlo\n\n")
            f.write("| Tier | P(PnL>0) | P(t>2.0) | P(t>1.5) | Median t |\n")
            f.write("|------|----------|----------|----------|----------|\n")
            
            for tier, data in results['slippage_monte_carlo'].items():
                f.write(f"| {tier} | {data.get('prob_pnl_positive', 0):.1f}% | "
                       f"{data.get('prob_t_gt_2', 0):.1f}% | "
                       f"{data.get('prob_t_gt_1_5', 0):.1f}% | "
                       f"{data.get('mc_t_median', 0):.2f} |\n")
            f.write("\n")
        
        # Verdict
        f.write("## Verdict\n\n")
        
        tiered = results.get('tiered_validation', {})
        tiny_survives = tiered.get('tiny', {}).get('edge_survives', False)
        medium_survives = tiered.get('medium', {}).get('edge_survives', False)
        large_survives = tiered.get('large', {}).get('edge_survives', False)
        
        if large_survives:
            f.write("**STRONG**: Edge survives at large size (q=100). Strategy is scalable.\n")
        elif medium_survives:
            f.write("**MODERATE**: Edge survives at medium size (q=25) but degrades at large. "
                   "Strategy has moderate capacity.\n")
        elif tiny_survives:
            f.write("**LIMITED**: Edge survives only at tiny size (q=5). Strategy has limited capacity.\n")
        else:
            f.write("**WEAK**: Edge doesn't survive even at tiny sizes. Review strategy.\n")
    
    print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='L6 Depth-Aware Validation')
    parser.add_argument('--data-path', type=str, 
                       default='data_v2/research_6levels/canonical_dataset_all_assets.parquet')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--buffer', type=float, default=0.02)
    parser.add_argument('--tau-max', type=int, default=420)
    parser.add_argument('--n-slippage-sims', type=int, default=200)
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer simulations)')
    
    args = parser.parse_args()
    
    strategy_params = {
        'buffer': args.buffer,
        'tau_max': args.tau_max,
        'exit_rule': 'expiry',
    }
    
    n_sims = 50 if args.quick else args.n_slippage_sims
    
    results = run_l6_validation_suite(
        data_path=args.data_path,
        output_dir=Path(args.output_dir),
        strategy_params=strategy_params,
        n_slippage_sims=n_sims,
    )


if __name__ == '__main__':
    main()

