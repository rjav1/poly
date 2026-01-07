"""
Event Study: Measure PM Response to CL Events

This module runs the event study to understand how Polymarket responds
to Chainlink price movements. This is the key discovery phase before
implementing trading strategies.

Key questions answered:
1. Does PM respond to CL moves?
2. How fast does PM respond?
3. Does response vary by time-to-expiry (tau)?
4. Is response stronger for strike-crossing events?
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

try:
    from .event_detection import Event, detect_cl_jumps, detect_strike_crosses, detect_all_events
except ImportError:
    from event_detection import Event, detect_cl_jumps, detect_strike_crosses, detect_all_events


@dataclass
class ResponseMeasurement:
    """Measurement of PM response at a specific lag."""
    market_id: str
    event_t: int
    event_type: str
    direction: str
    magnitude_bps: float
    tau: int
    delta_bps: float
    lag: int
    pm_up_response: float  # Change in pm_up_mid from baseline
    pm_down_response: float  # Change in pm_down_mid from baseline
    pm_directed_response: float  # Response in direction of event
    effective_spread: float  # Spread at this lag


def compute_pm_response(
    market_df: pd.DataFrame,
    event_t: int,
    max_lag: int = 30
) -> Dict[int, Dict[str, float]]:
    """
    Compute PM response at each lag after an event.
    
    Args:
        market_df: DataFrame for single market
        event_t: Time of event
        max_lag: Maximum lag to measure (seconds)
        
    Returns:
        Dict mapping lag to response measurements
    """
    responses = {}
    
    # Baseline: PM state just before event (t-1)
    baseline_row = market_df[market_df['t'] == event_t - 1]
    if baseline_row.empty:
        # Try event time as baseline if t-1 doesn't exist
        baseline_row = market_df[market_df['t'] == event_t]
        if baseline_row.empty:
            return responses
    
    baseline_up_mid = (baseline_row['pm_up_best_bid'].iloc[0] + baseline_row['pm_up_best_ask'].iloc[0]) / 2
    baseline_down_mid = (baseline_row['pm_down_best_bid'].iloc[0] + baseline_row['pm_down_best_ask'].iloc[0]) / 2
    
    for lag in range(0, max_lag + 1):
        target_row = market_df[market_df['t'] == event_t + lag]
        if target_row.empty:
            continue
        
        row = target_row.iloc[0]
        up_mid = (row['pm_up_best_bid'] + row['pm_up_best_ask']) / 2
        down_mid = (row['pm_down_best_bid'] + row['pm_down_best_ask']) / 2
        
        responses[lag] = {
            'pm_up_response': up_mid - baseline_up_mid,
            'pm_down_response': down_mid - baseline_down_mid,
            'up_bid': row['pm_up_best_bid'],
            'up_ask': row['pm_up_best_ask'],
            'down_bid': row['pm_down_best_bid'],
            'down_ask': row['pm_down_best_ask'],
            'effective_up_spread': row['pm_up_best_ask'] - row['pm_up_best_bid'],
            'effective_down_spread': row['pm_down_best_ask'] - row['pm_down_best_bid'],
        }
    
    return responses


def run_event_study(
    df: pd.DataFrame,
    events: List[Event],
    max_lag: int = 30
) -> pd.DataFrame:
    """
    Run event study across all events.
    
    For each event, measure PM response at lags 0..max_lag seconds.
    
    Args:
        df: Full DataFrame with all markets
        events: List of Event objects to study
        max_lag: Maximum lag to measure
        
    Returns:
        DataFrame with one row per (event, lag) pair
    """
    results = []
    
    for event in events:
        market_df = df[df['market_id'] == event.market_id]
        responses = compute_pm_response(market_df, event.t, max_lag)
        
        for lag, response in responses.items():
            # Directed response: positive if PM moved in same direction as event
            if event.direction == 'up':
                directed_response = response['pm_up_response']
            else:
                directed_response = response['pm_down_response']  # UP going down means DOWN going up
            
            results.append({
                'market_id': event.market_id,
                'event_t': event.t,
                'event_type': event.event_type,
                'direction': event.direction,
                'magnitude_bps': event.magnitude_bps,
                'tau': event.tau,
                'delta_bps': event.delta_bps,
                'lag': lag,
                'pm_up_response': response['pm_up_response'],
                'pm_down_response': response['pm_down_response'],
                'pm_directed_response': directed_response,
                'effective_up_spread': response['effective_up_spread'],
                'effective_down_spread': response['effective_down_spread'],
                # Regime classification
                'tau_bucket': get_tau_bucket(event.tau),
                'magnitude_bucket': get_magnitude_bucket(event.magnitude_bps),
                'delta_bucket': get_delta_bucket(event.delta_bps),
            })
    
    return pd.DataFrame(results)


def get_tau_bucket(tau: int) -> str:
    """Classify time-to-expiry."""
    if tau > 600:
        return 'early (10-15min)'
    elif tau > 300:
        return 'mid (5-10min)'
    else:
        return 'late (0-5min)'


def get_magnitude_bucket(magnitude_bps: float) -> str:
    """Classify event magnitude."""
    if magnitude_bps < 10:
        return 'small (<10bps)'
    elif magnitude_bps < 20:
        return 'medium (10-20bps)'
    else:
        return 'large (>20bps)'


def get_delta_bucket(delta_bps: float) -> str:
    """Classify distance to strike."""
    abs_delta = abs(delta_bps)
    if abs_delta <= 20:
        return 'near (<20bps)'
    elif abs_delta <= 100:
        return 'mid (20-100bps)'
    else:
        return 'far (>100bps)'


def compute_average_response_curve(
    study_df: pd.DataFrame,
    event_type: Optional[str] = None,
    direction: Optional[str] = None,
    tau_bucket: Optional[str] = None,
    magnitude_bucket: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute average response curve with confidence intervals.
    
    Args:
        study_df: Output from run_event_study
        event_type: Filter by event type
        direction: Filter by direction
        tau_bucket: Filter by tau bucket
        magnitude_bucket: Filter by magnitude bucket
        
    Returns:
        DataFrame with average response at each lag
    """
    filtered = study_df.copy()
    
    if event_type:
        filtered = filtered[filtered['event_type'] == event_type]
    if direction:
        filtered = filtered[filtered['direction'] == direction]
    if tau_bucket:
        filtered = filtered[filtered['tau_bucket'] == tau_bucket]
    if magnitude_bucket:
        filtered = filtered[filtered['magnitude_bucket'] == magnitude_bucket]
    
    if filtered.empty:
        return pd.DataFrame()
    
    # Group by lag and compute statistics
    grouped = filtered.groupby('lag').agg({
        'pm_directed_response': ['mean', 'std', 'count'],
        'pm_up_response': 'mean',
        'pm_down_response': 'mean',
        'effective_up_spread': 'mean',
        'effective_down_spread': 'mean',
    })
    
    # Flatten columns
    grouped.columns = [
        'mean_directed_response', 'std_directed_response', 'n_events',
        'mean_up_response', 'mean_down_response',
        'mean_up_spread', 'mean_down_spread',
    ]
    
    # Compute confidence intervals
    grouped['se'] = grouped['std_directed_response'] / np.sqrt(grouped['n_events'])
    grouped['ci_lower'] = grouped['mean_directed_response'] - 1.96 * grouped['se']
    grouped['ci_upper'] = grouped['mean_directed_response'] + 1.96 * grouped['se']
    
    return grouped.reset_index()


def compute_response_heatmap(
    study_df: pd.DataFrame,
    row_var: str = 'tau_bucket',
    col_var: str = 'magnitude_bucket',
    lag: int = 5
) -> pd.DataFrame:
    """
    Compute response heatmap at specific lag.
    
    Args:
        study_df: Output from run_event_study
        row_var: Variable for rows
        col_var: Variable for columns
        lag: Lag to measure response at
        
    Returns:
        Pivot table of average response
    """
    filtered = study_df[study_df['lag'] == lag]
    
    if filtered.empty:
        return pd.DataFrame()
    
    heatmap = filtered.pivot_table(
        values='pm_directed_response',
        index=row_var,
        columns=col_var,
        aggfunc='mean'
    )
    
    return heatmap


def compute_first_response_lag(
    study_df: pd.DataFrame,
    threshold: float = 0.005  # 0.5 cents
) -> pd.DataFrame:
    """
    Compute lag at which PM first responds significantly.
    
    Args:
        study_df: Output from run_event_study
        threshold: Minimum response to count as "significant"
        
    Returns:
        DataFrame with first response lag per event
    """
    results = []
    
    for (market_id, event_t), group in study_df.groupby(['market_id', 'event_t']):
        group = group.sort_values('lag')
        
        first_lag = None
        for _, row in group.iterrows():
            if abs(row['pm_directed_response']) >= threshold:
                first_lag = row['lag']
                break
        
        results.append({
            'market_id': market_id,
            'event_t': event_t,
            'event_type': group['event_type'].iloc[0],
            'direction': group['direction'].iloc[0],
            'tau': group['tau'].iloc[0],
            'magnitude_bps': group['magnitude_bps'].iloc[0],
            'first_response_lag': first_lag,
        })
    
    return pd.DataFrame(results)


def get_study_summary(study_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics from event study.
    
    Args:
        study_df: Output from run_event_study
        
    Returns:
        Summary dictionary
    """
    summary = {
        'total_events': study_df[['market_id', 'event_t']].drop_duplicates().shape[0],
        'total_observations': len(study_df),
        'n_markets': study_df['market_id'].nunique(),
    }
    
    # Response at key lags
    for lag in [0, 1, 2, 5, 10, 15, 30]:
        lag_data = study_df[study_df['lag'] == lag]
        if not lag_data.empty:
            summary[f'mean_response_lag{lag}'] = lag_data['pm_directed_response'].mean()
            summary[f'std_response_lag{lag}'] = lag_data['pm_directed_response'].std()
    
    # By event type
    for event_type in study_df['event_type'].unique():
        type_data = study_df[study_df['event_type'] == event_type]
        type_events = type_data[['market_id', 'event_t']].drop_duplicates().shape[0]
        summary[f'n_events_{event_type}'] = type_events
        
        # Response at lag 5
        lag5 = type_data[type_data['lag'] == 5]
        if not lag5.empty:
            summary[f'response_lag5_{event_type}'] = lag5['pm_directed_response'].mean()
    
    # By tau bucket
    for bucket in ['early (10-15min)', 'mid (5-10min)', 'late (0-5min)']:
        bucket_data = study_df[study_df['tau_bucket'] == bucket]
        if not bucket_data.empty:
            lag5 = bucket_data[bucket_data['lag'] == 5]
            if not lag5.empty:
                summary[f'response_lag5_{bucket.split()[0]}'] = lag5['pm_directed_response'].mean()
    
    return summary


# ==============================================================================
# Main Entry Point
# ==============================================================================

def run_full_event_study(
    df: pd.DataFrame,
    jump_threshold_bps: float = 10.0,
    near_strike_bps: float = 20.0,
    max_lag: int = 30
) -> Dict[str, Any]:
    """
    Run complete event study with all event types.
    
    Args:
        df: Full DataFrame with all markets
        jump_threshold_bps: Threshold for CL jump detection
        near_strike_bps: Threshold for near-strike regime
        max_lag: Maximum lag to measure
        
    Returns:
        Dictionary with all results
    """
    # Detect events
    events_df = detect_all_events(df, jump_threshold_bps, near_strike_bps)
    
    if events_df.empty:
        return {'error': 'No events detected'}
    
    # Convert to Event objects
    events = []
    for _, row in events_df.iterrows():
        events.append(Event(
            market_id=row['market_id'],
            t=row['t'],
            event_type=row['event_type'],
            direction=row['direction'],
            magnitude_bps=row['magnitude_bps'],
            tau=row['tau'],
            delta_bps=row['delta_bps'],
            cl_price=row['cl_price'],
            K=row['K'],
        ))
    
    # Run event study
    study_df = run_event_study(df, events, max_lag)
    
    # Compute summaries
    summary = get_study_summary(study_df)
    
    # Average response curves
    response_curves = {
        'all': compute_average_response_curve(study_df),
        'cl_jump': compute_average_response_curve(study_df, event_type='cl_jump'),
        'strike_cross': compute_average_response_curve(study_df, event_type='strike_cross'),
    }
    
    # Response by tau bucket
    for bucket in ['early (10-15min)', 'mid (5-10min)', 'late (0-5min)']:
        response_curves[f'tau_{bucket.split()[0]}'] = compute_average_response_curve(
            study_df, tau_bucket=bucket
        )
    
    # Heatmaps at different lags
    heatmaps = {}
    for lag in [1, 5, 10]:
        heatmaps[f'lag{lag}'] = compute_response_heatmap(study_df, lag=lag)
    
    # First response analysis
    first_response = compute_first_response_lag(study_df)
    
    return {
        'events_df': events_df,
        'study_df': study_df,
        'summary': summary,
        'response_curves': response_curves,
        'heatmaps': heatmaps,
        'first_response': first_response,
    }


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path
    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from scripts.backtest.data_loader import load_eth_markets, add_derived_columns
    
    print("Loading ETH markets...")
    df, market_info = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    print("\nRunning event study...")
    results = run_full_event_study(df, jump_threshold_bps=10.0)
    
    print("\n" + "="*60)
    print("EVENT STUDY RESULTS")
    print("="*60)
    
    summary = results['summary']
    print(f"\nTotal events: {summary['total_events']}")
    print(f"Markets: {summary['n_markets']}")
    
    print("\nEvents by type:")
    for event_type in ['cl_jump', 'strike_cross', 'near_strike_enter', 'near_strike_exit']:
        key = f'n_events_{event_type}'
        if key in summary:
            print(f"  {event_type}: {summary[key]}")
    
    print("\nAverage directed response at key lags:")
    for lag in [0, 1, 2, 5, 10, 15, 30]:
        key = f'mean_response_lag{lag}'
        if key in summary:
            print(f"  Lag {lag:2d}s: {summary[key]:+.4f}")
    
    print("\nResponse at lag 5s by tau bucket:")
    for bucket in ['early', 'mid', 'late']:
        key = f'response_lag5_{bucket}'
        if key in summary:
            print(f"  {bucket}: {summary[key]:+.4f}")
    
    # First response analysis
    first_resp = results['first_response']
    print(f"\nFirst significant response (>0.5c):")
    print(f"  Mean lag: {first_resp['first_response_lag'].mean():.1f}s")
    print(f"  Median lag: {first_resp['first_response_lag'].median():.1f}s")
    responded = first_resp['first_response_lag'].notna().sum()
    print(f"  Events with response: {responded}/{len(first_resp)} ({responded/len(first_resp)*100:.1f}%)")

