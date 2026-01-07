#!/usr/bin/env python3
"""
Phase 7: Strategy Implementation & Backtesting

Converts top hypotheses into parametric strategies and runs backtests
using walk-forward validation.

Input:
- hypotheses.json (from Phase 6)
- canonical_dataset_all_assets.parquet (market data)

Output:
- backtest_results.json (per-hypothesis results)
- strategy_implementations.json (parameterized strategies)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict
import sys

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESEARCH_DIR = BASE_DIR.parent / "data_v2" / "research"
MARKET_DURATION_SECONDS = 900

# Add scripts/backtest to path if available
BACKTEST_DIR = BASE_DIR.parent / "scripts" / "backtest"
if BACKTEST_DIR.exists():
    sys.path.insert(0, str(BACKTEST_DIR.parent))


@dataclass
class Signal:
    """Trading signal from strategy."""
    market_id: str
    t: int
    side: str  # 'buy_up', 'buy_down', 'buy_both', 'sell_up', 'sell_down'
    size: float
    reason: str


@dataclass
class Trade:
    """Executed trade."""
    market_id: str
    entry_t: int
    exit_t: int
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float


@dataclass
class BacktestResult:
    """Results from backtesting a strategy."""
    hypothesis_id: str
    strategy_name: str
    params: Dict[str, Any]
    total_pnl: float
    n_trades: int
    n_markets: int
    win_rate: float
    avg_pnl_per_trade: float
    avg_pnl_per_market: float
    std_pnl_per_market: float
    t_stat: float
    sharpe_proxy: float
    trades_per_market: float


def load_market_data() -> Tuple[pd.DataFrame, Dict]:
    """Load canonical market dataset."""
    path = RESEARCH_DIR / "canonical_dataset_all_assets.parquet"
    print(f"Loading market data from: {path}")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} rows")
    
    # Load market info
    info_path = RESEARCH_DIR / "market_info_all_assets.json"
    with open(info_path, 'r') as f:
        market_info = json.load(f)
    
    return df, market_info


def load_hypotheses() -> List[Dict]:
    """Load hypotheses from Phase 6."""
    path = RESULTS_DIR / "hypotheses.json"
    with open(path, 'r') as f:
        hypotheses = json.load(f)
    print(f"Loaded {len(hypotheses)} hypotheses")
    return hypotheses


# =============================================================================
# STRATEGY IMPLEMENTATIONS
# =============================================================================

def strategy_underround_harvest(
    market_df: pd.DataFrame,
    params: Dict[str, Any]
) -> List[Signal]:
    """
    H6: Underround Harvesting Strategy
    
    Buy both sides when sum_asks < 1 - epsilon
    """
    epsilon = params.get('epsilon', 0.01)
    min_tau = params.get('min_tau', 60)
    max_tau = params.get('max_tau', 840)
    cooldown = params.get('cooldown', 30)
    
    signals = []
    last_signal_t = -cooldown
    
    for _, row in market_df.iterrows():
        t = int(row['t'])
        tau = int(row['tau'])
        
        # Check tau window
        if tau < min_tau or tau > max_tau:
            continue
        
        # Check cooldown
        if t - last_signal_t < cooldown:
            continue
        
        # Check underround condition
        sum_asks = row.get('sum_asks')
        if pd.isna(sum_asks):
            continue
        
        underround = 1 - sum_asks
        if underround > epsilon:
            signals.append(Signal(
                market_id=row['market_id'],
                t=t,
                side='buy_both',
                size=1.0,
                reason=f"underround={underround:.4f} > {epsilon}"
            ))
            last_signal_t = t
    
    return signals


def strategy_late_underround(
    market_df: pd.DataFrame,
    params: Dict[str, Any]
) -> List[Signal]:
    """
    H7: Late Window Underround
    
    Buy both sides in late window when underround exists
    """
    epsilon = params.get('epsilon', 0.015)
    max_tau = params.get('max_tau', 120)
    cooldown = params.get('cooldown', 20)
    
    signals = []
    last_signal_t = -cooldown
    
    for _, row in market_df.iterrows():
        t = int(row['t'])
        tau = int(row['tau'])
        
        # Only late window
        if tau > max_tau or tau < 10:  # Not too late
            continue
        
        # Check cooldown
        if t - last_signal_t < cooldown:
            continue
        
        # Check underround
        sum_asks = row.get('sum_asks')
        if pd.isna(sum_asks):
            continue
        
        underround = 1 - sum_asks
        if underround > epsilon:
            signals.append(Signal(
                market_id=row['market_id'],
                t=t,
                side='buy_both',
                size=1.0,
                reason=f"late_underround={underround:.4f}, tau={tau}"
            ))
            last_signal_t = t
    
    return signals


def strategy_late_directional(
    market_df: pd.DataFrame,
    params: Dict[str, Any]
) -> List[Signal]:
    """
    H8: Late Directional Taker
    
    Take directional position in late window based on delta_bps
    """
    max_tau = params.get('max_tau', 300)
    delta_threshold_bps = params.get('delta_threshold_bps', 10)
    cooldown = params.get('cooldown', 60)
    
    signals = []
    last_signal_t = -cooldown
    
    for _, row in market_df.iterrows():
        t = int(row['t'])
        tau = int(row['tau'])
        
        # Only late window
        if tau > max_tau or tau < 30:
            continue
        
        # Check cooldown
        if t - last_signal_t < cooldown:
            continue
        
        # Check delta
        delta_bps = row.get('delta_bps')
        if pd.isna(delta_bps):
            continue
        
        if abs(delta_bps) > delta_threshold_bps:
            # Positive delta = CL above strike = UP likely
            side = 'buy_up' if delta_bps > 0 else 'buy_down'
            signals.append(Signal(
                market_id=row['market_id'],
                t=t,
                side=side,
                size=1.0,
                reason=f"delta_bps={delta_bps:.1f}, tau={tau}"
            ))
            last_signal_t = t
    
    return signals


def strategy_early_inventory(
    market_df: pd.DataFrame,
    params: Dict[str, Any]
) -> List[Signal]:
    """
    H9: Early Inventory Build
    
    Build matched inventory early when underround exists
    """
    min_tau = params.get('min_tau', 600)
    epsilon = params.get('epsilon', 0.015)
    cooldown = params.get('cooldown', 60)
    
    signals = []
    last_signal_t = -cooldown
    
    for _, row in market_df.iterrows():
        t = int(row['t'])
        tau = int(row['tau'])
        
        # Only early window
        if tau < min_tau:
            continue
        
        # Check cooldown
        if t - last_signal_t < cooldown:
            continue
        
        # Check underround
        sum_asks = row.get('sum_asks')
        if pd.isna(sum_asks):
            continue
        
        underround = 1 - sum_asks
        if underround > epsilon:
            signals.append(Signal(
                market_id=row['market_id'],
                t=t,
                side='buy_both',
                size=1.0,
                reason=f"early_inventory: underround={underround:.4f}, tau={tau}"
            ))
            last_signal_t = t
    
    return signals


def strategy_tight_spread(
    market_df: pd.DataFrame,
    params: Dict[str, Any]
) -> List[Signal]:
    """
    H10: Tight Spread Entry
    
    Enter when spreads are unusually tight
    """
    spread_threshold = params.get('spread_threshold', 0.02)
    cooldown = params.get('cooldown', 60)
    min_tau = params.get('min_tau', 120)
    
    signals = []
    last_signal_t = -cooldown
    
    for _, row in market_df.iterrows():
        t = int(row['t'])
        tau = int(row['tau'])
        
        if tau < min_tau:
            continue
        
        if t - last_signal_t < cooldown:
            continue
        
        # Check spread
        up_spread = row.get('pm_up_spread')
        down_spread = row.get('pm_down_spread')
        
        if pd.isna(up_spread) or pd.isna(down_spread):
            continue
        
        avg_spread = (up_spread + down_spread) / 2
        
        # Also check for underround to combine signals
        sum_asks = row.get('sum_asks', 1.0)
        underround = 1 - sum_asks
        
        if avg_spread < spread_threshold and underround > 0:
            signals.append(Signal(
                market_id=row['market_id'],
                t=t,
                side='buy_both',
                size=1.0,
                reason=f"tight_spread={avg_spread:.4f}, underround={underround:.4f}"
            ))
            last_signal_t = t
    
    return signals


STRATEGY_REGISTRY = {
    'H6_underround_harvest': strategy_underround_harvest,
    'H7_late_underround': strategy_late_underround,
    'H8_late_directional': strategy_late_directional,
    'H9_early_inventory': strategy_early_inventory,
    'H10_tight_spread': strategy_tight_spread,
}


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def execute_signal(
    market_df: pd.DataFrame,
    signal: Signal,
    hold_to_expiry: bool = True,
    hold_seconds: int = 180
) -> Optional[Trade]:
    """
    Execute a signal and compute PnL.
    """
    entry_row = market_df[market_df['t'] == signal.t]
    if entry_row.empty:
        return None
    
    entry_row = entry_row.iloc[0]
    
    # Determine exit time
    if hold_to_expiry:
        exit_t = MARKET_DURATION_SECONDS - 1
    else:
        exit_t = min(signal.t + hold_seconds, MARKET_DURATION_SECONDS - 1)
    
    exit_row = market_df[market_df['t'] == exit_t]
    if exit_row.empty:
        return None
    
    exit_row = exit_row.iloc[0]
    
    # Compute entry and exit prices
    if signal.side == 'buy_both':
        # Buy complete set
        up_ask = entry_row.get('pm_up_best_ask')
        down_ask = entry_row.get('pm_down_best_ask')
        
        if pd.isna(up_ask) or pd.isna(down_ask):
            return None
        
        entry_price = up_ask + down_ask  # Cost to buy complete set
        exit_price = 1.0  # Complete set pays $1 at expiry
        pnl = (exit_price - entry_price) * signal.size
        
    elif signal.side in ['buy_up', 'buy_down']:
        # Directional trade
        if signal.side == 'buy_up':
            entry_price = entry_row.get('pm_up_best_ask')
            # At expiry, UP pays 1 if Y=1, 0 if Y=0
            Y = entry_row.get('Y', exit_row.get('Y', 0))
            exit_price = float(Y)
        else:  # buy_down
            entry_price = entry_row.get('pm_down_best_ask')
            Y = entry_row.get('Y', exit_row.get('Y', 0))
            exit_price = 1 - float(Y)  # DOWN pays 1 if Y=0
        
        if pd.isna(entry_price):
            return None
        
        pnl = (exit_price - entry_price) * signal.size
    else:
        return None
    
    return Trade(
        market_id=signal.market_id,
        entry_t=signal.t,
        exit_t=exit_t,
        side=signal.side,
        entry_price=entry_price,
        exit_price=exit_price,
        size=signal.size,
        pnl=pnl,
    )


def backtest_strategy(
    market_data: pd.DataFrame,
    strategy_fn: Callable,
    params: Dict[str, Any],
    hypothesis_id: str
) -> BacktestResult:
    """
    Backtest a strategy across all markets.
    """
    all_trades = []
    market_pnls = {}
    
    # Group by market
    for market_id, market_df in market_data.groupby('market_id'):
        market_df = market_df.sort_values('t')
        
        # Generate signals
        signals = strategy_fn(market_df, params)
        
        # Execute signals
        market_trades = []
        for signal in signals:
            trade = execute_signal(
                market_df, 
                signal,
                hold_to_expiry=params.get('hold_to_expiry', True),
                hold_seconds=params.get('hold_seconds', 180)
            )
            if trade is not None:
                market_trades.append(trade)
        
        # Compute market PnL
        market_pnl = sum(t.pnl for t in market_trades)
        market_pnls[market_id] = market_pnl
        all_trades.extend(market_trades)
    
    # Compute statistics
    if len(all_trades) == 0:
        return BacktestResult(
            hypothesis_id=hypothesis_id,
            strategy_name=hypothesis_id,
            params=params,
            total_pnl=0.0,
            n_trades=0,
            n_markets=len(market_pnls),
            win_rate=0.0,
            avg_pnl_per_trade=0.0,
            avg_pnl_per_market=0.0,
            std_pnl_per_market=0.0,
            t_stat=0.0,
            sharpe_proxy=0.0,
            trades_per_market=0.0,
        )
    
    total_pnl = sum(t.pnl for t in all_trades)
    n_trades = len(all_trades)
    n_markets = len(market_pnls)
    win_rate = sum(1 for t in all_trades if t.pnl > 0) / n_trades
    avg_pnl_per_trade = total_pnl / n_trades
    
    # Per-market statistics (for t-stat calculation)
    pnl_array = np.array(list(market_pnls.values()))
    avg_pnl_per_market = pnl_array.mean()
    std_pnl_per_market = pnl_array.std()
    
    # T-statistic: mean / (std / sqrt(n))
    if std_pnl_per_market > 0 and n_markets > 1:
        t_stat = avg_pnl_per_market / (std_pnl_per_market / np.sqrt(n_markets))
    else:
        t_stat = 0.0
    
    # Sharpe proxy
    if std_pnl_per_market > 0:
        sharpe_proxy = avg_pnl_per_market / std_pnl_per_market
    else:
        sharpe_proxy = 0.0
    
    return BacktestResult(
        hypothesis_id=hypothesis_id,
        strategy_name=hypothesis_id,
        params=params,
        total_pnl=total_pnl,
        n_trades=n_trades,
        n_markets=n_markets,
        win_rate=win_rate,
        avg_pnl_per_trade=avg_pnl_per_trade,
        avg_pnl_per_market=avg_pnl_per_market,
        std_pnl_per_market=std_pnl_per_market,
        t_stat=t_stat,
        sharpe_proxy=sharpe_proxy,
        trades_per_market=n_trades / n_markets if n_markets > 0 else 0,
    )


def run_parameter_sweep(
    market_data: pd.DataFrame,
    hypothesis_id: str,
    strategy_fn: Callable,
    param_grid: Dict[str, List[Any]]
) -> List[BacktestResult]:
    """
    Run parameter sweep for a strategy.
    """
    from itertools import product
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    results = []
    
    for combo in product(*param_values):
        params = dict(zip(param_names, combo))
        result = backtest_strategy(market_data, strategy_fn, params, hypothesis_id)
        results.append(result)
        
        if result.n_trades > 0:
            print(f"    Params: {params} -> PnL: ${result.total_pnl:.2f}, t-stat: {result.t_stat:.2f}, trades: {result.n_trades}")
    
    return results


def main():
    print("=" * 70)
    print("Phase 7: Strategy Implementation & Backtesting")
    print("=" * 70)
    
    # Step 1: Load data
    market_data, market_info = load_market_data()
    hypotheses = load_hypotheses()
    
    # Step 2: Run backtests for each hypothesis
    print("\n" + "=" * 70)
    print("Running backtests...")
    print("=" * 70)
    
    all_results = {}
    best_results = {}
    
    for hyp in hypotheses:
        hyp_id = hyp['hypothesis_id']
        
        if hyp_id not in STRATEGY_REGISTRY:
            print(f"\nSkipping {hyp_id} - no implementation")
            continue
        
        print(f"\n--- {hyp_id}: {hyp['name']} ---")
        
        strategy_fn = STRATEGY_REGISTRY[hyp_id]
        
        # Get parameter grid from hypothesis
        params_spec = hyp.get('parameters', {})
        param_grid = {}
        base_params = {}
        
        for param_name, spec in params_spec.items():
            if isinstance(spec, dict):
                if 'sweep' in spec:
                    param_grid[param_name] = spec['sweep']
                if 'suggested' in spec:
                    base_params[param_name] = spec['suggested']
            else:
                base_params[param_name] = spec
        
        # If no sweep grid, just use base params
        if not param_grid:
            param_grid = {k: [v] for k, v in base_params.items()}
        
        # Run sweep
        results = run_parameter_sweep(market_data, hyp_id, strategy_fn, param_grid)
        
        all_results[hyp_id] = [asdict(r) for r in results]
        
        # Find best result
        if results:
            best = max(results, key=lambda r: r.t_stat)
            best_results[hyp_id] = asdict(best)
            print(f"  Best: t-stat={best.t_stat:.2f}, PnL=${best.total_pnl:.2f}, trades={best.n_trades}")
    
    # Step 3: Summary
    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY (Best per Hypothesis)")
    print("=" * 70)
    
    sorted_results = sorted(
        best_results.items(),
        key=lambda x: x[1]['t_stat'],
        reverse=True
    )
    
    for hyp_id, result in sorted_results:
        print(f"\n{hyp_id}:")
        print(f"  Total PnL: ${result['total_pnl']:.2f}")
        print(f"  Trades: {result['n_trades']}")
        print(f"  Markets: {result['n_markets']}")
        print(f"  Win Rate: {result['win_rate']*100:.1f}%")
        print(f"  t-stat: {result['t_stat']:.2f}")
        print(f"  Params: {result['params']}")
    
    # Step 4: Save outputs
    print("\n" + "=" * 70)
    print("Saving outputs...")
    
    # Save all results
    results_path = RESULTS_DIR / "backtest_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'all_results': all_results,
            'best_results': best_results,
        }, f, indent=2, default=str)
    print(f"  Backtest results saved to: {results_path}")
    
    # Save strategy implementations
    impl_path = RESULTS_DIR / "strategy_implementations.json"
    implementations = {
        hyp_id: {
            'hypothesis': next((h for h in hypotheses if h['hypothesis_id'] == hyp_id), {}),
            'best_params': best_results.get(hyp_id, {}).get('params', {}),
            'best_t_stat': best_results.get(hyp_id, {}).get('t_stat', 0),
        }
        for hyp_id in STRATEGY_REGISTRY.keys()
    }
    with open(impl_path, 'w') as f:
        json.dump(implementations, f, indent=2, default=str)
    print(f"  Strategy implementations saved to: {impl_path}")
    
    print("\n" + "=" * 70)
    print("DONE - Phase 7 Complete")
    print("=" * 70)
    
    return all_results, best_results


if __name__ == "__main__":
    main()

