"""
Event Detection for Lead-Lag Analysis

Detects three types of events for the event study:
1. CL Jump Events: Price moves >= threshold_bps in one second
2. Strike-Crossing Events: CL crosses the strike price K
3. Near-Strike Regime: CL is within d bps of strike (most price-sensitive regime)
"""

from dataclasses import dataclass
from typing import List, Literal, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class Event:
    """A detected market event."""
    market_id: str
    t: int  # Time index within market
    event_type: Literal['cl_jump', 'strike_cross', 'near_strike_enter', 'near_strike_exit']
    direction: Literal['up', 'down']
    magnitude_bps: float  # Size of move (for jumps) or distance to strike (for cross)
    tau: int  # Time to expiry
    delta_bps: float  # Distance to strike at event time
    cl_price: float  # CL price at event time
    K: float  # Strike price
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'market_id': self.market_id,
            't': self.t,
            'event_type': self.event_type,
            'direction': self.direction,
            'magnitude_bps': self.magnitude_bps,
            'tau': self.tau,
            'delta_bps': self.delta_bps,
            'cl_price': self.cl_price,
            'K': self.K,
        }


def detect_cl_jumps(
    market_df: pd.DataFrame,
    threshold_bps: float
) -> List[Event]:
    """
    Detect CL price jumps exceeding threshold.
    
    A "jump" is a single-second move of >= threshold_bps.
    
    Args:
        market_df: DataFrame for single market (must have market_id, t, cl_mid, tau, K)
        threshold_bps: Minimum move size in bps
        
    Returns:
        List of Event objects
    """
    events = []
    
    if len(market_df) < 2:
        return events
    
    market_id = market_df['market_id'].iloc[0]
    K = market_df['K'].iloc[0]
    
    df = market_df.copy().sort_values('t')
    
    # Calculate returns in bps
    df['cl_return_bps'] = df['cl_mid'].pct_change() * 10000
    
    for idx, row in df.iterrows():
        if pd.isna(row['cl_return_bps']):
            continue
            
        if abs(row['cl_return_bps']) >= threshold_bps:
            events.append(Event(
                market_id=market_id,
                t=int(row['t']),
                event_type='cl_jump',
                direction='up' if row['cl_return_bps'] > 0 else 'down',
                magnitude_bps=abs(row['cl_return_bps']),
                tau=int(row['tau']),
                delta_bps=row['delta_bps'],
                cl_price=row['cl_mid'],
                K=K,
            ))
    
    return events


def detect_strike_crosses(market_df: pd.DataFrame) -> List[Event]:
    """
    Detect when CL crosses the strike price K.
    
    This is the economically critical event - when CL crosses K,
    the probability should snap toward 0 or 100%.
    
    Args:
        market_df: DataFrame for single market
        
    Returns:
        List of Event objects
    """
    events = []
    
    if len(market_df) < 2:
        return events
    
    market_id = market_df['market_id'].iloc[0]
    K = market_df['K'].iloc[0]
    
    df = market_df.copy().sort_values('t')
    
    # Determine if above or below strike
    df['above_strike'] = df['cl_mid'] > K
    
    # Detect crosses (where above_strike changes)
    df['prev_above'] = df['above_strike'].shift(1)
    df['crossed'] = (df['above_strike'] != df['prev_above']) & df['prev_above'].notna()
    
    for idx, row in df[df['crossed']].iterrows():
        # Direction: 'up' if we crossed from below to above, 'down' if above to below
        direction = 'up' if row['above_strike'] else 'down'
        
        events.append(Event(
            market_id=market_id,
            t=int(row['t']),
            event_type='strike_cross',
            direction=direction,
            magnitude_bps=abs(row['delta_bps']),  # How far we are now from strike
            tau=int(row['tau']),
            delta_bps=row['delta_bps'],
            cl_price=row['cl_mid'],
            K=K,
        ))
    
    return events


def detect_near_strike_entries(
    market_df: pd.DataFrame,
    threshold_bps: float = 20.0
) -> List[Event]:
    """
    Detect when CL enters the near-strike regime (within threshold_bps of K).
    
    The near-strike regime is where PM probability is most sensitive to CL moves.
    
    Args:
        market_df: DataFrame for single market
        threshold_bps: Definition of "near" in bps
        
    Returns:
        List of Event objects (enter and exit events)
    """
    events = []
    
    if len(market_df) < 2:
        return events
    
    market_id = market_df['market_id'].iloc[0]
    K = market_df['K'].iloc[0]
    
    df = market_df.copy().sort_values('t')
    
    # Flag near-strike regime
    df['near_strike'] = abs(df['delta_bps']) <= threshold_bps
    df['prev_near'] = df['near_strike'].shift(1)
    
    # Detect entries and exits
    for idx, row in df.iterrows():
        if pd.isna(row['prev_near']):
            continue
            
        # Entry: wasn't near, now is near
        if row['near_strike'] and not row['prev_near']:
            events.append(Event(
                market_id=market_id,
                t=int(row['t']),
                event_type='near_strike_enter',
                direction='up' if row['delta_bps'] > 0 else 'down',
                magnitude_bps=abs(row['delta_bps']),
                tau=int(row['tau']),
                delta_bps=row['delta_bps'],
                cl_price=row['cl_mid'],
                K=K,
            ))
        
        # Exit: was near, now isn't
        elif not row['near_strike'] and row['prev_near']:
            events.append(Event(
                market_id=market_id,
                t=int(row['t']),
                event_type='near_strike_exit',
                direction='up' if row['delta_bps'] > 0 else 'down',
                magnitude_bps=abs(row['delta_bps']),
                tau=int(row['tau']),
                delta_bps=row['delta_bps'],
                cl_price=row['cl_mid'],
                K=K,
            ))
    
    return events


def detect_all_events(
    df: pd.DataFrame,
    jump_threshold_bps: float = 10.0,
    near_strike_bps: float = 20.0
) -> pd.DataFrame:
    """
    Detect all event types across all markets.
    
    Args:
        df: Full DataFrame with all markets
        jump_threshold_bps: Threshold for CL jump detection
        near_strike_bps: Threshold for near-strike regime
        
    Returns:
        DataFrame with all events
    """
    all_events = []
    
    for market_id in df['market_id'].unique():
        market_df = df[df['market_id'] == market_id]
        
        # Detect each type
        all_events.extend(detect_cl_jumps(market_df, jump_threshold_bps))
        all_events.extend(detect_strike_crosses(market_df))
        all_events.extend(detect_near_strike_entries(market_df, near_strike_bps))
    
    # Convert to DataFrame
    if not all_events:
        return pd.DataFrame()
    
    events_df = pd.DataFrame([e.to_dict() for e in all_events])
    events_df = events_df.sort_values(['market_id', 't']).reset_index(drop=True)
    
    return events_df


def get_event_summary(events_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize detected events.
    
    Args:
        events_df: DataFrame of events
        
    Returns:
        Summary statistics
    """
    if events_df.empty:
        return {'total_events': 0}
    
    summary = {
        'total_events': len(events_df),
        'n_markets': events_df['market_id'].nunique(),
        'events_per_market': len(events_df) / events_df['market_id'].nunique(),
    }
    
    # By event type
    for event_type in ['cl_jump', 'strike_cross', 'near_strike_enter', 'near_strike_exit']:
        type_events = events_df[events_df['event_type'] == event_type]
        summary[f'n_{event_type}'] = len(type_events)
        summary[f'n_{event_type}_up'] = len(type_events[type_events['direction'] == 'up'])
        summary[f'n_{event_type}_down'] = len(type_events[type_events['direction'] == 'down'])
    
    # By tau bucket
    def get_tau_bucket(tau):
        if tau > 600:
            return 'early (10-15min)'
        elif tau > 300:
            return 'mid (5-10min)'
        else:
            return 'late (0-5min)'
    
    events_df = events_df.copy()
    events_df['tau_bucket'] = events_df['tau'].apply(get_tau_bucket)
    
    for bucket in ['early (10-15min)', 'mid (5-10min)', 'late (0-5min)']:
        bucket_events = events_df[events_df['tau_bucket'] == bucket]
        summary[f'n_events_{bucket.split()[0]}'] = len(bucket_events)
    
    return summary


def add_regime_flags(df: pd.DataFrame, near_strike_bps: float = 20.0) -> pd.DataFrame:
    """
    Add regime flags to DataFrame for analysis.
    
    Args:
        df: Full DataFrame
        near_strike_bps: Threshold for near-strike regime
        
    Returns:
        DataFrame with regime columns
    """
    df = df.copy()
    
    # Near-strike regime
    df['near_strike'] = abs(df['delta_bps']) <= near_strike_bps
    
    # Tau buckets
    def get_tau_bucket(tau):
        if tau > 600:
            return 'early'
        elif tau > 300:
            return 'mid'
        else:
            return 'late'
    
    df['tau_bucket'] = df['tau'].apply(get_tau_bucket)
    
    # Delta buckets (distance to strike)
    def get_delta_bucket(delta):
        abs_delta = abs(delta)
        if abs_delta <= 20:
            return 'near (<20bps)'
        elif abs_delta <= 100:
            return 'mid (20-100bps)'
        else:
            return 'far (>100bps)'
    
    df['delta_bucket'] = df['delta_bps'].apply(get_delta_bucket)
    
    # Above/below strike
    df['position'] = df['delta_bps'].apply(lambda x: 'above' if x > 0 else 'below')
    
    return df


# ==============================================================================
# Tests
# ==============================================================================

def test_cl_jump_detection():
    """Test CL jump detection."""
    # Create test data with a jump at t=5
    data = {
        'market_id': ['test'] * 10,
        't': list(range(10)),
        'tau': list(range(900, 890, -1)),
        'cl_mid': [100.0, 100.0, 100.0, 100.0, 100.0, 100.5, 100.5, 100.5, 100.5, 100.5],  # 50bps jump at t=5
        'K': [100.0] * 10,
        'delta_bps': [0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 50.0, 50.0, 50.0, 50.0],
    }
    df = pd.DataFrame(data)
    
    # Detect with 40bps threshold (should find the jump)
    events = detect_cl_jumps(df, threshold_bps=40.0)
    assert len(events) == 1, f"Expected 1 event, got {len(events)}"
    assert events[0].t == 5
    assert events[0].direction == 'up'
    assert abs(events[0].magnitude_bps - 50.0) < 0.1
    
    # Detect with 60bps threshold (should not find it)
    events = detect_cl_jumps(df, threshold_bps=60.0)
    assert len(events) == 0, f"Expected 0 events, got {len(events)}"
    
    print("[OK] CL jump detection test passed")


def test_strike_cross_detection():
    """Test strike crossing detection."""
    # Create test data crossing strike at t=5 and t=8
    data = {
        'market_id': ['test'] * 10,
        't': list(range(10)),
        'tau': list(range(900, 890, -1)),
        'cl_mid': [99.0, 99.5, 99.8, 99.9, 100.0, 100.1, 100.2, 100.1, 99.9, 99.8],
        'K': [100.0] * 10,
        'delta_bps': [-100, -50, -20, -10, 0, 10, 20, 10, -10, -20],
    }
    df = pd.DataFrame(data)
    
    events = detect_strike_crosses(df)
    assert len(events) == 2, f"Expected 2 events, got {len(events)}"
    
    # First cross: below to above at t=5
    assert events[0].t == 5
    assert events[0].direction == 'up'
    
    # Second cross: above to below at t=8
    assert events[1].t == 8
    assert events[1].direction == 'down'
    
    print("[OK] Strike cross detection test passed")


def test_near_strike_detection():
    """Test near-strike regime detection."""
    # Create test data entering and exiting near-strike regime
    data = {
        'market_id': ['test'] * 10,
        't': list(range(10)),
        'tau': list(range(900, 890, -1)),
        'cl_mid': [100.0] * 10,
        'K': [100.0] * 10,
        'delta_bps': [-50, -30, -15, -10, -5, 0, 5, 10, 25, 40],  # Enter at t=2, exit at t=8
    }
    df = pd.DataFrame(data)
    
    events = detect_near_strike_entries(df, threshold_bps=20.0)
    
    # Should have entry at t=2 and exit at t=8
    enter_events = [e for e in events if e.event_type == 'near_strike_enter']
    exit_events = [e for e in events if e.event_type == 'near_strike_exit']
    
    assert len(enter_events) == 1, f"Expected 1 enter, got {len(enter_events)}"
    assert len(exit_events) == 1, f"Expected 1 exit, got {len(exit_events)}"
    
    assert enter_events[0].t == 2
    assert exit_events[0].t == 8
    
    print("[OK] Near-strike detection test passed")


if __name__ == '__main__':
    test_cl_jump_detection()
    test_strike_cross_detection()
    test_near_strike_detection()
    print("\nAll event detection tests passed!")

