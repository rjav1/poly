#!/usr/bin/env python3
"""
Phase 6.2: Execution Realism Stress Tests

Tests execution assumptions with:
- Random slippage Monte Carlo (1000 runs)
- Quote fade / adverse selection model
- Size/capacity stress (max trades per market)

These tests quantify how much slippage the strategy can tolerate
and whether the edge survives more realistic execution assumptions.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest.data_loader import load_eth_markets, add_derived_columns, get_train_test_split
from scripts.backtest.fair_value import BinnedFairValueModel
from scripts.backtest.strategies import MispricingBasedStrategy, Signal
from scripts.backtest.backtest_engine import run_backtest, ExecutionConfig
from scripts.backtest.execution_model import get_effective_prices


# ==============================================================================
# 6.2.1: RANDOM SLIPPAGE MONTE CARLO
# ==============================================================================

def run_slippage_monte_carlo(
    test_df: pd.DataFrame,
    fair_value_model,
    strategy_params: Dict,
    n_simulations: int = 1000,
    slippage_type: str = 'uniform',
    slippage_max: float = 0.015,  # 1.5 cents
    slippage_mean: float = 0.008,  # 0.8 cents
    slippage_std: float = 0.004,  # 0.4 cents
    random_seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation with random slippage per trade.
    
    Instead of fixed slippage, draw slippage per trade from a distribution.
    This tests how robust the strategy is to execution uncertainty.
    
    Args:
        test_df: Test data
        fair_value_model: Fitted fair value model
        strategy_params: Strategy parameters
        n_simulations: Number of Monte Carlo runs (default 1000)
        slippage_type: 'uniform' or 'normal'
        slippage_max: Max slippage for uniform (default 1.5 cents)
        slippage_mean: Mean slippage for normal (default 0.8 cents)
        slippage_std: Std of slippage for normal (default 0.4 cents)
        random_seed: Random seed for reproducibility
        verbose: Print progress
        
    Returns:
        Dict with:
        - pnl_distribution: Statistics on PnL across runs
        - t_stat_distribution: Statistics on t-stat across runs
        - prob_positive_pnl: P(PnL > 0)
        - prob_t_stat_gt_2: P(t-stat > 2.0)
        - percentiles: 5th, 50th, 95th percentiles
    """
    np.random.seed(random_seed)
    
    if verbose:
        print(f"\nRunning Slippage Monte Carlo ({n_simulations} simulations)...")
        print(f"  Slippage type: {slippage_type}")
        if slippage_type == 'uniform':
            print(f"  Slippage range: [0, {slippage_max:.4f}]")
        else:
            print(f"  Slippage mean: {slippage_mean:.4f}, std: {slippage_std:.4f}")
    
    # First run baseline without slippage to get trades
    strategy = MispricingBasedStrategy(
        fair_value_model=fair_value_model,
        **strategy_params
    )
    baseline_result = run_backtest(test_df, strategy, ExecutionConfig(), verbose=False)
    baseline_trades = baseline_result['trades']
    baseline_metrics = baseline_result['metrics']
    
    if len(baseline_trades) == 0:
        return {
            'error': 'No trades in baseline',
            'n_simulations': 0
        }
    
    if verbose:
        print(f"  Baseline: {len(baseline_trades)} trades, "
              f"PnL=${baseline_metrics['total_pnl']:.2f}, t={baseline_metrics['t_stat']:.2f}")
    
    # Store simulation results
    sim_pnls = []
    sim_tstats = []
    sim_mean_pnls = []
    
    # Get market-level PnLs for t-stat calculation
    market_ids = test_df['market_id'].unique()
    
    for sim in range(n_simulations):
        # Generate random slippage for each trade
        n_trades = len(baseline_trades)
        
        if slippage_type == 'uniform':
            slippages = np.random.uniform(0, slippage_max, size=n_trades)
        else:  # normal
            slippages = np.random.normal(slippage_mean, slippage_std, size=n_trades)
            slippages = np.maximum(slippages, 0)  # Clip at 0
        
        # Apply slippage to trades
        adjusted_pnls = []
        market_pnls = {m: 0.0 for m in market_ids}
        
        for i, trade in enumerate(baseline_trades):
            original_pnl = trade['pnl']
            slippage_cost = slippages[i] * 2  # Entry and exit slippage
            adjusted_pnl = original_pnl - slippage_cost
            adjusted_pnls.append(adjusted_pnl)
            market_pnls[trade['market_id']] += adjusted_pnl
        
        # Compute metrics
        total_pnl = sum(adjusted_pnls)
        mean_pnl_per_market = np.mean(list(market_pnls.values()))
        std_pnl_per_market = np.std(list(market_pnls.values()), ddof=1) if len(market_pnls) > 1 else 1.0
        n_markets = len([p for p in market_pnls.values() if p != 0]) or 1
        t_stat = mean_pnl_per_market / (std_pnl_per_market / np.sqrt(n_markets)) if std_pnl_per_market > 0 else 0
        
        sim_pnls.append(total_pnl)
        sim_tstats.append(t_stat)
        sim_mean_pnls.append(mean_pnl_per_market)
        
        if verbose and (sim + 1) % 200 == 0:
            print(f"  Completed {sim + 1}/{n_simulations} simulations")
    
    sim_pnls = np.array(sim_pnls)
    sim_tstats = np.array(sim_tstats)
    sim_mean_pnls = np.array(sim_mean_pnls)
    
    results = {
        'n_simulations': n_simulations,
        'slippage_type': slippage_type,
        'slippage_params': {
            'max': slippage_max if slippage_type == 'uniform' else None,
            'mean': slippage_mean if slippage_type == 'normal' else None,
            'std': slippage_std if slippage_type == 'normal' else None,
        },
        'baseline': {
            'n_trades': len(baseline_trades),
            'total_pnl': baseline_metrics['total_pnl'],
            't_stat': baseline_metrics['t_stat'],
        },
        'pnl_distribution': {
            'mean': float(sim_pnls.mean()),
            'std': float(sim_pnls.std()),
            'percentile_5': float(np.percentile(sim_pnls, 5)),
            'percentile_50': float(np.percentile(sim_pnls, 50)),
            'percentile_95': float(np.percentile(sim_pnls, 95)),
        },
        't_stat_distribution': {
            'mean': float(sim_tstats.mean()),
            'std': float(sim_tstats.std()),
            'percentile_5': float(np.percentile(sim_tstats, 5)),
            'percentile_50': float(np.percentile(sim_tstats, 50)),
            'percentile_95': float(np.percentile(sim_tstats, 95)),
        },
        'prob_positive_pnl': float((sim_pnls > 0).mean()),
        'prob_t_stat_gt_2': float((sim_tstats > 2.0).mean()),
        'prob_t_stat_gt_1_5': float((sim_tstats > 1.5).mean()),
    }
    
    if verbose:
        print(f"\n  Results:")
        print(f"    P(PnL > 0): {results['prob_positive_pnl']:.1%}")
        print(f"    P(t-stat > 2.0): {results['prob_t_stat_gt_2']:.1%}")
        print(f"    P(t-stat > 1.5): {results['prob_t_stat_gt_1_5']:.1%}")
        print(f"    Median PnL: ${results['pnl_distribution']['percentile_50']:.2f}")
        print(f"    Median t-stat: {results['t_stat_distribution']['percentile_50']:.2f}")
    
    return results


# ==============================================================================
# 6.2.2: QUOTE FADE / ADVERSE SELECTION MODEL
# ==============================================================================

def run_quote_fade_test(
    test_df: pd.DataFrame,
    fair_value_model,
    strategy_params: Dict,
    fade_delays: List[int] = None,
    fade_type: str = 'fixed',
    random_seed: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Test strategy with quote fade (adverse selection).
    
    When buying at time t, actually pay price at t + delta where delta
    simulates quotes moving against you during fast moments.
    
    Args:
        test_df: Test data
        fair_value_model: Fitted fair value model
        strategy_params: Strategy parameters
        fade_delays: List of fade delays in seconds to test
        fade_type: 'fixed' (single delay) or 'random' (sample from range)
        random_seed: Random seed
        verbose: Print progress
        
    Returns:
        DataFrame with results for each fade delay
    """
    fade_delays = fade_delays or [0, 1, 2, 3]
    
    if verbose:
        print(f"\nRunning Quote Fade Test ({len(fade_delays)} delays)...")
    
    np.random.seed(random_seed)
    
    results = []
    
    for delay in fade_delays:
        # Create modified test data with delayed prices
        df_faded = test_df.copy()
        
        if delay > 0:
            # Shift PM prices forward (so we're trading at future prices = adverse)
            price_cols = ['pm_up_best_bid', 'pm_up_best_ask', 
                         'pm_down_best_bid', 'pm_down_best_ask']
            
            for col in price_cols:
                if col in df_faded.columns:
                    # Shift prices forward by delay seconds (within each market)
                    df_faded[col] = df_faded.groupby('market_id')[col].shift(-delay)
            
            # Fill NaN at end with last valid
            for col in price_cols:
                if col in df_faded.columns:
                    df_faded[col] = df_faded.groupby('market_id')[col].ffill()
            
            # Drop rows that still have NaN
            df_faded = df_faded.dropna(subset=['pm_up_best_bid', 'pm_up_best_ask'])
        
        # Run strategy
        strategy = MispricingBasedStrategy(
            fair_value_model=fair_value_model,
            **strategy_params
        )
        
        result = run_backtest(df_faded, strategy, ExecutionConfig(), verbose=False)
        metrics = result['metrics']
        
        results.append({
            'fade_delay': delay,
            'n_trades': metrics['n_trades'],
            'total_pnl': metrics['total_pnl'],
            'mean_pnl_per_market': metrics['mean_pnl_per_market'],
            't_stat': metrics['t_stat'],
            'hit_rate': metrics['hit_rate_per_trade'],
        })
        
        if verbose:
            print(f"  Delay {delay}s: t={metrics['t_stat']:.2f}, "
                  f"PnL=${metrics['total_pnl']:.2f}, trades={metrics['n_trades']}")
    
    return pd.DataFrame(results)


# ==============================================================================
# 6.2.3: SIZE/CAPACITY STRESS
# ==============================================================================

def run_capacity_stress(
    test_df: pd.DataFrame,
    fair_value_model,
    strategy_params: Dict,
    max_trades_per_market: List[int] = None,
    selection_method: str = 'first',  # 'first', 'strongest', 'random'
    random_seed: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Test strategy with capacity limits.
    
    Limits trades per market to simulate limited capacity.
    Tests if edge persists with fewer, higher-quality trades.
    
    Args:
        test_df: Test data
        fair_value_model: Fitted fair value model
        strategy_params: Strategy parameters
        max_trades_per_market: List of max trades to test
        selection_method: How to select trades when limited
            - 'first': Keep first N trades chronologically
            - 'strongest': Keep N trades with largest mispricing
            - 'random': Randomly select N trades
        random_seed: Random seed
        verbose: Print progress
        
    Returns:
        DataFrame with results for each capacity limit
    """
    max_trades_list = max_trades_per_market or [1, 2, 3, 5, 10, None]
    
    if verbose:
        print(f"\nRunning Capacity Stress Test ({len(max_trades_list)} limits, method={selection_method})...")
    
    np.random.seed(random_seed)
    
    # First, run full strategy to get all signals
    strategy = MispricingBasedStrategy(
        fair_value_model=fair_value_model,
        **strategy_params
    )
    
    # Get signals for each market
    all_signals = {}
    for market_id in test_df['market_id'].unique():
        market_df = test_df[test_df['market_id'] == market_id]
        signals = strategy.generate_signals(market_df)
        all_signals[market_id] = signals
    
    results = []
    
    for max_trades in max_trades_list:
        # Filter signals based on capacity limit
        filtered_signals = []
        
        for market_id, signals in all_signals.items():
            if max_trades is None or len(signals) <= max_trades:
                filtered_signals.extend(signals)
            else:
                if selection_method == 'first':
                    # Keep first N trades
                    selected = sorted(signals, key=lambda s: s.entry_t)[:max_trades]
                elif selection_method == 'strongest':
                    # Keep N trades with largest mispricing
                    selected = sorted(signals, 
                                     key=lambda s: s.metadata.get('mispricing', 0), 
                                     reverse=True)[:max_trades]
                else:  # random
                    selected = list(np.random.choice(signals, size=max_trades, replace=False))
                filtered_signals.extend(selected)
        
        if len(filtered_signals) == 0:
            results.append({
                'max_trades': max_trades if max_trades else 'unlimited',
                'n_trades': 0,
                'total_pnl': 0,
                't_stat': 0,
            })
            continue
        
        # Execute filtered signals manually
        market_pnls = {}
        total_trades = 0
        
        for signal in filtered_signals:
            market_df = test_df[test_df['market_id'] == signal.market_id]
            max_t = int(market_df['t'].max()) if len(market_df) > 0 else -1
            
            # Get entry/exit rows
            entry_row = market_df[market_df['t'] == signal.entry_t]
            exit_row = market_df[market_df['t'] == signal.exit_t]
            
            # Handle exit_t that might be beyond available data (e.g., expiry at t=900 when max_t=899)
            if exit_row.empty:
                # If exit_rule is 'expiry', use last available row
                if signal.exit_t > max_t:
                    exit_row = market_df[market_df['t'] == max_t]
                    if exit_row.empty:
                        # Fall back to last row
                        exit_row = market_df.iloc[-1:]
                    if exit_row.empty:
                        continue
            
            if entry_row.empty or exit_row.empty:
                continue
            
            entry_row = entry_row.iloc[0]
            exit_row = exit_row.iloc[0]
            
            # Get effective prices (conversion-aware)
            entry_prices = get_effective_prices(entry_row)
            exit_prices = get_effective_prices(exit_row)
            
            # Get entry price
            if signal.side == 'buy_up':
                entry_price = entry_prices.buy_up
            elif signal.side == 'buy_down':
                entry_price = entry_prices.buy_down
            else:
                continue  # Skip unsupported sides
            
            if pd.isna(entry_price):
                continue
            
            # Check if expiry trade (use Y for settlement payout)
            max_t = int(market_df['t'].max())
            is_expiry = (signal.exit_t >= max_t - 1)
            
            if is_expiry:
                # Use settlement outcome (Y) for expiry trades
                Y = entry_row.get('Y', np.nan)
                if pd.isna(Y):
                    Y = market_df['Y'].iloc[0] if 'Y' in market_df.columns else np.nan
                
                if not pd.isna(Y):
                    if signal.side == 'buy_up':
                        exit_price = float(Y)  # $1 if Y=1, $0 if Y=0
                    elif signal.side == 'buy_down':
                        exit_price = 1.0 - float(Y)  # $1 if Y=0, $0 if Y=1
                else:
                    # Y not available, fall back to orderbook
                    if signal.side == 'buy_up':
                        exit_price = exit_prices.sell_up
                    elif signal.side == 'buy_down':
                        exit_price = exit_prices.sell_down
                    if pd.isna(exit_price):
                        continue
            else:
                # Early exit: use orderbook exit price
                if signal.side == 'buy_up':
                    exit_price = exit_prices.sell_up
                elif signal.side == 'buy_down':
                    exit_price = exit_prices.sell_down
                if pd.isna(exit_price):
                    continue
            
            # PnL = exit_price - entry_price for buy orders
            pnl = exit_price - entry_price
            
            if signal.market_id not in market_pnls:
                market_pnls[signal.market_id] = 0
            market_pnls[signal.market_id] += pnl
            total_trades += 1
        
        if len(market_pnls) == 0:
            results.append({
                'max_trades': max_trades if max_trades else 'unlimited',
                'n_trades': 0,
                'total_pnl': 0,
                't_stat': 0,
            })
            continue
        
        # Compute metrics
        total_pnl = sum(market_pnls.values())
        mean_pnl = np.mean(list(market_pnls.values()))
        std_pnl = np.std(list(market_pnls.values()), ddof=1) if len(market_pnls) > 1 else 1.0
        t_stat = mean_pnl / (std_pnl / np.sqrt(len(market_pnls))) if std_pnl > 0 else 0
        
        results.append({
            'max_trades': max_trades if max_trades else 'unlimited',
            'selection_method': selection_method,
            'n_trades': total_trades,
            'n_markets': len(market_pnls),
            'total_pnl': total_pnl,
            'mean_pnl_per_market': mean_pnl,
            't_stat': t_stat,
        })
        
        if verbose:
            print(f"  Max {max_trades if max_trades else 'unlimited'}: "
                  f"t={t_stat:.2f}, PnL=${total_pnl:.2f}, trades={total_trades}")
    
    return pd.DataFrame(results)


# ==============================================================================
# COMBINED EXECUTION STRESS SUITE
# ==============================================================================

def run_execution_stress_suite(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    strategy_params: Dict,
    output_dir: Optional[Path] = None,
    n_slippage_sims: int = 1000,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete execution stress test suite.
    
    Args:
        test_df: Test data
        train_df: Training data (for model fitting)
        strategy_params: Strategy parameters
        output_dir: Directory to save results
        n_slippage_sims: Number of slippage Monte Carlo simulations
        verbose: Print progress
        
    Returns:
        Dict with all execution stress test results
    """
    print("\n" + "="*70)
    print("PHASE 6.2: EXECUTION REALISM STRESS TESTS")
    print("="*70)
    
    # Fit model
    model = BinnedFairValueModel(sample_every=5)
    model.fit(train_df)
    
    results = {}
    
    # 6.2.1: Slippage Monte Carlo
    print("\n--- 6.2.1: Slippage Monte Carlo ---")
    
    # Uniform slippage
    uniform_results = run_slippage_monte_carlo(
        test_df, model, strategy_params,
        n_simulations=n_slippage_sims,
        slippage_type='uniform',
        slippage_max=0.015,
        verbose=verbose
    )
    results['slippage_uniform'] = uniform_results
    
    # Normal slippage
    normal_results = run_slippage_monte_carlo(
        test_df, model, strategy_params,
        n_simulations=n_slippage_sims,
        slippage_type='normal',
        slippage_mean=0.008,
        slippage_std=0.004,
        verbose=verbose
    )
    results['slippage_normal'] = normal_results
    
    # 6.2.2: Quote Fade
    print("\n--- 6.2.2: Quote Fade / Adverse Selection ---")
    fade_results = run_quote_fade_test(
        test_df, model, strategy_params,
        fade_delays=[0, 1, 2, 3, 5],
        verbose=verbose
    )
    results['quote_fade'] = fade_results.to_dict('records')
    
    # 6.2.3: Capacity Stress
    print("\n--- 6.2.3: Capacity Stress ---")
    
    # First selection method
    capacity_first = run_capacity_stress(
        test_df, model, strategy_params,
        max_trades_per_market=[1, 2, 3, 5, 10, None],
        selection_method='first',
        verbose=verbose
    )
    results['capacity_first'] = capacity_first.to_dict('records')
    
    # Strongest selection method
    capacity_strongest = run_capacity_stress(
        test_df, model, strategy_params,
        max_trades_per_market=[1, 2, 3, 5, 10, None],
        selection_method='strongest',
        verbose=verbose
    )
    results['capacity_strongest'] = capacity_strongest.to_dict('records')
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        with open(output_dir / 'execution_stress_results.json', 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)
        
        fade_results.to_csv(output_dir / 'quote_fade_results.csv', index=False)
        capacity_first.to_csv(output_dir / 'capacity_stress_first.csv', index=False)
        capacity_strongest.to_csv(output_dir / 'capacity_stress_strongest.csv', index=False)
        
        print(f"\nResults saved to {output_dir}")
    
    # Summary
    print("\n" + "="*70)
    print("EXECUTION STRESS TEST SUMMARY")
    print("="*70)
    
    print("\nSlippage Robustness:")
    print(f"  Uniform slippage [0, 1.5c]:")
    print(f"    P(t-stat > 2.0): {results['slippage_uniform']['prob_t_stat_gt_2']:.1%}")
    print(f"    Median t-stat: {results['slippage_uniform']['t_stat_distribution']['percentile_50']:.2f}")
    print(f"  Normal slippage (mean=0.8c, std=0.4c):")
    print(f"    P(t-stat > 2.0): {results['slippage_normal']['prob_t_stat_gt_2']:.1%}")
    print(f"    Median t-stat: {results['slippage_normal']['t_stat_distribution']['percentile_50']:.2f}")
    
    print("\nQuote Fade Survival:")
    for row in results['quote_fade']:
        print(f"  Delay {row['fade_delay']}s: t={row['t_stat']:.2f}")
    
    print("\nCapacity Stress (strongest selection):")
    for row in results['capacity_strongest']:
        max_t = row['max_trades']
        print(f"  Max {max_t} trades/market: t={row['t_stat']:.2f}, trades={row['n_trades']}")
    
    return results


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run execution stress tests')
    parser.add_argument('--n-sims', type=int, default=1000,
                       help='Number of slippage Monte Carlo simulations')
    parser.add_argument('--output-dir', type=str, 
                       default=str(PROJECT_ROOT / 'data_v2' / 'backtest_results' / 'stress_tests'))
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    df, _ = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    train_df, test_df, _, _ = get_train_test_split(df, train_frac=0.7)
    
    # Default strategy params
    strategy_params = {
        'buffer': 0.02,
        'tau_max': 420,
        'min_tau': 0,
        'cooldown': 30,
        'exit_rule': 'expiry',
    }
    
    # Run suite
    results = run_execution_stress_suite(
        test_df=test_df,
        train_df=train_df,
        strategy_params=strategy_params,
        output_dir=Path(args.output_dir),
        n_slippage_sims=args.n_sims
    )

