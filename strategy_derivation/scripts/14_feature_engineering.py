#!/usr/bin/env python3
"""
Phase 4: Feature Engineering for Policy Inversion

Creates rich feature set for modeling state -> action relationships.
Includes PM microstructure features, CL features, temporal features, and interactions.

Input:
- execution_enriched.parquet (trades with execution labels from Phase 3)

Output:
- feature_matrix.parquet (trades with full feature set)
- feature_descriptions.json (feature names and descriptions)
- feature_stats.json (feature statistics and importance hints)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MARKET_DURATION_SECONDS = 900


def load_execution_data() -> pd.DataFrame:
    """Load enriched execution data from Phase 3."""
    path = DATA_DIR / "execution_enriched.parquet"
    print(f"Loading execution data from: {path}")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} rows")
    print(f"  Columns: {len(df.columns)}")
    return df


def compute_pm_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PM-only microstructure features (no CL dependency).
    These are the most generalizable features.
    """
    print("\nComputing PM microstructure features...")
    df = df.copy()
    
    # -------------------------------------------------------------------------
    # Spread features
    # -------------------------------------------------------------------------
    # Spread width (already have from market data, but ensure)
    if 'mkt_pm_up_spread' in df.columns and 'mkt_pm_down_spread' in df.columns:
        df['feat_up_spread'] = df['mkt_pm_up_spread']
        df['feat_down_spread'] = df['mkt_pm_down_spread']
        df['feat_avg_spread'] = (df['mkt_pm_up_spread'] + df['mkt_pm_down_spread']) / 2
        df['feat_spread_diff'] = df['mkt_pm_up_spread'] - df['mkt_pm_down_spread']
    
    # Spread relative to mid
    if 'mkt_pm_up_mid' in df.columns and 'mkt_pm_up_spread' in df.columns:
        df['feat_up_spread_rel'] = np.where(
            df['mkt_pm_up_mid'] > 0,
            df['mkt_pm_up_spread'] / df['mkt_pm_up_mid'],
            np.nan
        )
    if 'mkt_pm_down_mid' in df.columns and 'mkt_pm_down_spread' in df.columns:
        df['feat_down_spread_rel'] = np.where(
            df['mkt_pm_down_mid'] > 0,
            df['mkt_pm_down_spread'] / df['mkt_pm_down_mid'],
            np.nan
        )
    
    # -------------------------------------------------------------------------
    # Underround/overround features (no-arb conditions)
    # -------------------------------------------------------------------------
    if 'mkt_sum_asks' in df.columns:
        df['feat_underround'] = 1 - df['mkt_sum_asks']  # Positive when sum_asks < 1
        df['feat_underround_positive'] = df['feat_underround'].clip(lower=0)
        df['feat_has_underround'] = (df['feat_underround'] > 0.005).astype(int)
    
    if 'mkt_sum_bids' in df.columns:
        df['feat_overround'] = df['mkt_sum_bids'] - 1  # Positive when sum_bids > 1
        df['feat_overround_positive'] = df['feat_overround'].clip(lower=0)
        df['feat_has_overround'] = (df['feat_overround'] > 0.005).astype(int)
    
    # -------------------------------------------------------------------------
    # Orderbook imbalance
    # -------------------------------------------------------------------------
    if 'mkt_sum_bids' in df.columns and 'mkt_sum_asks' in df.columns:
        denom = df['mkt_sum_bids'] + df['mkt_sum_asks']
        df['feat_book_imbalance'] = np.where(
            denom > 0,
            (df['mkt_sum_bids'] - df['mkt_sum_asks']) / denom,
            0
        )
    
    # -------------------------------------------------------------------------
    # Quote staleness (if available)
    # -------------------------------------------------------------------------
    if 'mkt_quote_staleness' in df.columns:
        df['feat_quote_staleness'] = df['mkt_quote_staleness']
        df['feat_stale_quote'] = (df['mkt_quote_staleness'] > 5).astype(int)
    
    # -------------------------------------------------------------------------
    # PM momentum features
    # -------------------------------------------------------------------------
    for k in [1, 5, 10, 30]:
        col = f'mkt_pm_momentum_{k}s'
        if col in df.columns:
            df[f'feat_pm_momentum_{k}s'] = df[col]
            df[f'feat_pm_momentum_{k}s_abs'] = df[col].abs()
            df[f'feat_pm_momentum_{k}s_sign'] = np.sign(df[col])
    
    # -------------------------------------------------------------------------
    # PM mid-price levels (probability proximity)
    # -------------------------------------------------------------------------
    if 'mkt_pm_up_mid' in df.columns:
        df['feat_up_mid'] = df['mkt_pm_up_mid']
        df['feat_up_mid_far_from_half'] = (df['mkt_pm_up_mid'] - 0.5).abs()
        df['feat_up_near_0'] = (df['mkt_pm_up_mid'] < 0.15).astype(int)
        df['feat_up_near_1'] = (df['mkt_pm_up_mid'] > 0.85).astype(int)
        df['feat_up_mid_range'] = (
            (df['mkt_pm_up_mid'] >= 0.35) & (df['mkt_pm_up_mid'] <= 0.65)
        ).astype(int)
    
    if 'mkt_pm_down_mid' in df.columns:
        df['feat_down_mid'] = df['mkt_pm_down_mid']
        df['feat_down_mid_far_from_half'] = (df['mkt_pm_down_mid'] - 0.5).abs()
    
    print(f"  Added PM microstructure features")
    return df


def compute_cl_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute CL (Chainlink) features for lead-lag strategies.
    """
    print("\nComputing CL features...")
    df = df.copy()
    
    # -------------------------------------------------------------------------
    # Delta from strike
    # -------------------------------------------------------------------------
    if 'mkt_delta_bps' in df.columns:
        df['feat_delta_bps'] = df['mkt_delta_bps']
        df['feat_delta_bps_abs'] = df['mkt_delta_bps'].abs()
        df['feat_delta_bps_sign'] = np.sign(df['mkt_delta_bps'])
        
        # Delta buckets
        df['feat_delta_tiny'] = (df['feat_delta_bps_abs'] < 5).astype(int)
        df['feat_delta_small'] = ((df['feat_delta_bps_abs'] >= 5) & (df['feat_delta_bps_abs'] < 15)).astype(int)
        df['feat_delta_medium'] = ((df['feat_delta_bps_abs'] >= 15) & (df['feat_delta_bps_abs'] < 30)).astype(int)
        df['feat_delta_large'] = (df['feat_delta_bps_abs'] >= 30).astype(int)
    
    # -------------------------------------------------------------------------
    # CL momentum
    # -------------------------------------------------------------------------
    for k in [1, 5, 10, 30]:
        col = f'mkt_cl_momentum_{k}s'
        if col in df.columns:
            df[f'feat_cl_momentum_{k}s'] = df[col]
            df[f'feat_cl_momentum_{k}s_abs'] = df[col].abs()
            df[f'feat_cl_momentum_{k}s_sign'] = np.sign(df[col])
    
    # -------------------------------------------------------------------------
    # Realized volatility
    # -------------------------------------------------------------------------
    if 'mkt_realized_vol_10s' in df.columns:
        df['feat_vol_10s'] = df['mkt_realized_vol_10s']
        median_vol = df['mkt_realized_vol_10s'].median()
        if not pd.isna(median_vol) and median_vol > 0:
            df['feat_high_vol'] = (df['mkt_realized_vol_10s'] > median_vol * 1.5).astype(int)
            df['feat_low_vol'] = (df['mkt_realized_vol_10s'] < median_vol * 0.5).astype(int)
    
    if 'mkt_realized_vol_30s' in df.columns:
        df['feat_vol_30s'] = df['mkt_realized_vol_30s']
    
    # -------------------------------------------------------------------------
    # Strike-cross indicator (how close to the strike)
    # -------------------------------------------------------------------------
    if 'mkt_delta_bps' in df.columns:
        df['feat_near_strike'] = (df['mkt_delta_bps'].abs() < 10).astype(int)
        df['feat_far_from_strike'] = (df['mkt_delta_bps'].abs() > 50).astype(int)
    
    print(f"  Added CL features")
    return df


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute temporal features related to time within market.
    """
    print("\nComputing temporal features...")
    df = df.copy()
    
    # -------------------------------------------------------------------------
    # Basic time coordinates
    # -------------------------------------------------------------------------
    df['feat_t'] = df['t']
    df['feat_tau'] = df['tau']
    df['feat_tau_normalized'] = df['tau'] / MARKET_DURATION_SECONDS  # 0-1 scale
    
    # -------------------------------------------------------------------------
    # Tau buckets (categorical)
    # -------------------------------------------------------------------------
    df['feat_tau_bucket'] = pd.cut(
        df['tau'],
        bins=[0, 60, 120, 300, 600, 900],
        labels=['0-60', '60-120', '120-300', '300-600', '600-900'],
        include_lowest=True
    )
    
    # Binary tau indicators
    df['feat_very_late'] = (df['tau'] <= 60).astype(int)
    df['feat_late'] = (df['tau'] <= 120).astype(int)
    df['feat_mid_window'] = ((df['tau'] > 120) & (df['tau'] <= 600)).astype(int)
    df['feat_early'] = (df['tau'] > 600).astype(int)
    
    # -------------------------------------------------------------------------
    # Time quadrants
    # -------------------------------------------------------------------------
    df['feat_first_quarter'] = (df['t'] < 225).astype(int)
    df['feat_second_quarter'] = ((df['t'] >= 225) & (df['t'] < 450)).astype(int)
    df['feat_third_quarter'] = ((df['t'] >= 450) & (df['t'] < 675)).astype(int)
    df['feat_fourth_quarter'] = (df['t'] >= 675).astype(int)
    
    # -------------------------------------------------------------------------
    # Time to key thresholds
    # -------------------------------------------------------------------------
    df['feat_secs_to_60'] = np.maximum(0, df['tau'] - 60)
    df['feat_secs_to_120'] = np.maximum(0, df['tau'] - 120)
    df['feat_secs_to_300'] = np.maximum(0, df['tau'] - 300)
    
    print(f"  Added temporal features")
    return df


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute interaction features combining multiple base features.
    """
    print("\nComputing interaction features...")
    df = df.copy()
    
    # -------------------------------------------------------------------------
    # Underround * capacity (edge-weighted capacity)
    # -------------------------------------------------------------------------
    if 'feat_underround_positive' in df.columns and 'mkt_achievable_capacity' in df.columns:
        df['feat_edge_capacity'] = df['feat_underround_positive'] * df['mkt_achievable_capacity']
    
    # -------------------------------------------------------------------------
    # Spread * tau (spread widens near expiry?)
    # -------------------------------------------------------------------------
    if 'feat_avg_spread' in df.columns:
        df['feat_spread_x_tau'] = df['feat_avg_spread'] * df['feat_tau_normalized']
    
    # -------------------------------------------------------------------------
    # Delta * tau (late directional bets?)
    # -------------------------------------------------------------------------
    if 'feat_delta_bps' in df.columns:
        df['feat_delta_x_tau'] = df['feat_delta_bps'] * df['feat_tau_normalized']
        df['feat_delta_x_late'] = df['feat_delta_bps'] * df['feat_late']
    
    # -------------------------------------------------------------------------
    # Underround * late window (late underround harvesting)
    # -------------------------------------------------------------------------
    if 'feat_underround_positive' in df.columns:
        df['feat_underround_x_late'] = df['feat_underround_positive'] * df['feat_late']
        df['feat_underround_x_very_late'] = df['feat_underround_positive'] * df['feat_very_late']
    
    # -------------------------------------------------------------------------
    # Momentum * tau (late momentum trading)
    # -------------------------------------------------------------------------
    if 'feat_cl_momentum_10s' in df.columns:
        df['feat_cl_mom_x_tau'] = df['feat_cl_momentum_10s'] * df['feat_tau_normalized']
        df['feat_cl_mom_x_late'] = df['feat_cl_momentum_10s'] * df['feat_late']
    
    # -------------------------------------------------------------------------
    # Volatility * spread (vol-adjusted spread)
    # -------------------------------------------------------------------------
    if 'feat_vol_10s' in df.columns and 'feat_avg_spread' in df.columns:
        df['feat_spread_vol_ratio'] = np.where(
            df['feat_vol_10s'] > 0,
            df['feat_avg_spread'] / (df['feat_vol_10s'] * 100),  # Scale vol
            np.nan
        )
    
    # -------------------------------------------------------------------------
    # Direction prediction features
    # -------------------------------------------------------------------------
    if 'feat_delta_bps' in df.columns and 'mkt_pm_up_mid' in df.columns:
        # Disagreement between CL and PM
        # If delta > 0 (CL above strike) but pm_up_mid < 0.5, there's disagreement
        df['feat_cl_pm_agree'] = (
            (np.sign(df['feat_delta_bps']) == np.sign(df['mkt_pm_up_mid'] - 0.5))
        ).astype(int)
        df['feat_cl_pm_disagree'] = 1 - df['feat_cl_pm_agree']
    
    print(f"  Added interaction features")
    return df


def compute_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features from position/execution context.
    """
    print("\nComputing position features...")
    df = df.copy()
    
    # -------------------------------------------------------------------------
    # Inventory state encoding
    # -------------------------------------------------------------------------
    if 'inventory_state' in df.columns:
        df['feat_is_long'] = (df['inventory_state'] == 'LONG').astype(int)
        df['feat_is_short'] = (df['inventory_state'] == 'SHORT').astype(int)
        df['feat_is_flat'] = (df['inventory_state'] == 'FLAT').astype(int)
    
    if 'inventory_size' in df.columns:
        df['feat_inventory_size'] = df['inventory_size']
        df['feat_has_inventory'] = (df['inventory_size'] > 0.001).astype(int)
    
    # -------------------------------------------------------------------------
    # Entry/exit encoding
    # -------------------------------------------------------------------------
    if 'is_entry' in df.columns:
        df['feat_is_entry'] = df['is_entry'].astype(int)
    if 'is_exit' in df.columns:
        df['feat_is_exit'] = df['is_exit'].astype(int)
    
    # -------------------------------------------------------------------------
    # Execution style encoding
    # -------------------------------------------------------------------------
    if 'execution_type' in df.columns:
        df['feat_is_taker'] = (df['execution_type'] == 'TAKER').astype(int)
        df['feat_is_maker'] = (df['execution_type'] == 'MAKER').astype(int)
    
    if 'aggressiveness_score' in df.columns:
        df['feat_aggressiveness'] = df['aggressiveness_score']
    
    print(f"  Added position features")
    return df


def get_feature_descriptions() -> Dict[str, str]:
    """Return descriptions for all features."""
    return {
        # PM microstructure
        'feat_up_spread': 'UP token bid-ask spread',
        'feat_down_spread': 'DOWN token bid-ask spread',
        'feat_avg_spread': 'Average of UP and DOWN spreads',
        'feat_spread_diff': 'UP spread minus DOWN spread',
        'feat_up_spread_rel': 'UP spread relative to mid price',
        'feat_down_spread_rel': 'DOWN spread relative to mid price',
        'feat_underround': 'Underround = 1 - sum_asks (positive = arb opportunity)',
        'feat_underround_positive': 'Underround clipped to positive values',
        'feat_has_underround': 'Binary: underround > 0.5%',
        'feat_overround': 'Overround = sum_bids - 1',
        'feat_overround_positive': 'Overround clipped to positive values',
        'feat_has_overround': 'Binary: overround > 0.5%',
        'feat_book_imbalance': 'Orderbook imbalance (bids-asks)/(bids+asks)',
        'feat_quote_staleness': 'Seconds since last quote change',
        'feat_stale_quote': 'Binary: quote staleness > 5s',
        'feat_pm_momentum_1s': 'PM mid-price change over 1 second',
        'feat_pm_momentum_5s': 'PM mid-price change over 5 seconds',
        'feat_pm_momentum_10s': 'PM mid-price change over 10 seconds',
        'feat_pm_momentum_30s': 'PM mid-price change over 30 seconds',
        'feat_up_mid': 'UP token mid price',
        'feat_up_mid_far_from_half': 'Absolute distance of UP mid from 0.5',
        'feat_up_near_0': 'Binary: UP mid < 0.15',
        'feat_up_near_1': 'Binary: UP mid > 0.85',
        'feat_up_mid_range': 'Binary: UP mid in [0.35, 0.65]',
        'feat_down_mid': 'DOWN token mid price',
        
        # CL features
        'feat_delta_bps': 'CL price delta from strike in basis points',
        'feat_delta_bps_abs': 'Absolute value of delta_bps',
        'feat_delta_bps_sign': 'Sign of delta_bps (+1/-1)',
        'feat_delta_tiny': 'Binary: |delta| < 5 bps',
        'feat_delta_small': 'Binary: 5 <= |delta| < 15 bps',
        'feat_delta_medium': 'Binary: 15 <= |delta| < 30 bps',
        'feat_delta_large': 'Binary: |delta| >= 30 bps',
        'feat_cl_momentum_1s': 'CL price change over 1 second',
        'feat_cl_momentum_5s': 'CL price change over 5 seconds',
        'feat_cl_momentum_10s': 'CL price change over 10 seconds',
        'feat_cl_momentum_30s': 'CL price change over 30 seconds',
        'feat_vol_10s': '10-second realized volatility',
        'feat_vol_30s': '30-second realized volatility',
        'feat_high_vol': 'Binary: vol > 1.5x median',
        'feat_low_vol': 'Binary: vol < 0.5x median',
        'feat_near_strike': 'Binary: |delta| < 10 bps',
        'feat_far_from_strike': 'Binary: |delta| > 50 bps',
        
        # Temporal
        'feat_t': 'Seconds from market start',
        'feat_tau': 'Seconds to expiry',
        'feat_tau_normalized': 'Tau normalized to [0,1]',
        'feat_tau_bucket': 'Categorical tau bucket',
        'feat_very_late': 'Binary: tau <= 60s',
        'feat_late': 'Binary: tau <= 120s',
        'feat_mid_window': 'Binary: 120s < tau <= 600s',
        'feat_early': 'Binary: tau > 600s',
        'feat_first_quarter': 'Binary: t in [0, 225)',
        'feat_second_quarter': 'Binary: t in [225, 450)',
        'feat_third_quarter': 'Binary: t in [450, 675)',
        'feat_fourth_quarter': 'Binary: t in [675, 900)',
        'feat_secs_to_60': 'Seconds until tau reaches 60',
        'feat_secs_to_120': 'Seconds until tau reaches 120',
        'feat_secs_to_300': 'Seconds until tau reaches 300',
        
        # Interactions
        'feat_edge_capacity': 'Underround * achievable capacity',
        'feat_spread_x_tau': 'Spread * normalized tau',
        'feat_delta_x_tau': 'Delta * normalized tau',
        'feat_delta_x_late': 'Delta * late window indicator',
        'feat_underround_x_late': 'Underround * late window',
        'feat_underround_x_very_late': 'Underround * very late window',
        'feat_cl_mom_x_tau': 'CL momentum * normalized tau',
        'feat_cl_mom_x_late': 'CL momentum * late window',
        'feat_spread_vol_ratio': 'Spread / volatility ratio',
        'feat_cl_pm_agree': 'Binary: CL and PM predictions agree',
        'feat_cl_pm_disagree': 'Binary: CL and PM predictions disagree',
        
        # Position
        'feat_is_long': 'Binary: currently long',
        'feat_is_short': 'Binary: currently short',
        'feat_is_flat': 'Binary: currently flat',
        'feat_inventory_size': 'Current inventory size',
        'feat_has_inventory': 'Binary: has non-zero inventory',
        'feat_is_entry': 'Binary: trade is position entry',
        'feat_is_exit': 'Binary: trade is position exit',
        'feat_is_taker': 'Binary: execution was taker',
        'feat_is_maker': 'Binary: execution was maker',
        'feat_aggressiveness': 'Aggressiveness score [0,1]',
    }


def compute_feature_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute feature statistics and importance hints."""
    print("\nComputing feature statistics...")
    
    # Get all feature columns
    feat_cols = [c for c in df.columns if c.startswith('feat_')]
    
    stats = {}
    for col in feat_cols:
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            stats[col] = {
                'dtype': str(df[col].dtype),
                'non_null': int(df[col].notna().sum()),
                'null_pct': float(df[col].isna().mean()),
                'mean': float(df[col].mean()) if df[col].notna().any() else None,
                'std': float(df[col].std()) if df[col].notna().any() else None,
                'min': float(df[col].min()) if df[col].notna().any() else None,
                'max': float(df[col].max()) if df[col].notna().any() else None,
                'median': float(df[col].median()) if df[col].notna().any() else None,
                'unique': int(df[col].nunique()),
            }
        else:
            stats[col] = {
                'dtype': str(df[col].dtype),
                'non_null': int(df[col].notna().sum()),
                'null_pct': float(df[col].isna().mean()),
                'unique': int(df[col].nunique()),
                'top_values': df[col].value_counts().head(5).to_dict(),
            }
    
    print(f"  Computed stats for {len(stats)} features")
    return stats


def main():
    print("=" * 70)
    print("Phase 4: Feature Engineering for Policy Inversion")
    print("=" * 70)
    
    # Step 1: Load execution data
    df = load_execution_data()
    
    # Step 2: Compute PM microstructure features
    df = compute_pm_microstructure_features(df)
    
    # Step 3: Compute CL features
    df = compute_cl_features(df)
    
    # Step 4: Compute temporal features
    df = compute_temporal_features(df)
    
    # Step 5: Compute interaction features
    df = compute_interaction_features(df)
    
    # Step 6: Compute position features
    df = compute_position_features(df)
    
    # Step 7: Get feature descriptions
    descriptions = get_feature_descriptions()
    
    # Step 8: Compute feature stats
    stats = compute_feature_stats(df)
    
    # Summary
    feat_cols = [c for c in df.columns if c.startswith('feat_')]
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total features: {len(feat_cols)}")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    
    # Feature category counts
    pm_feats = [c for c in feat_cols if 'spread' in c or 'underround' in c or 'overround' in c or 'imbalance' in c or 'pm_' in c or 'stale' in c]
    cl_feats = [c for c in feat_cols if 'delta' in c or 'cl_' in c or 'vol' in c or 'strike' in c]
    temporal_feats = [c for c in feat_cols if 'tau' in c or 't' in c or 'quarter' in c or 'late' in c or 'early' in c or 'secs_to' in c]
    position_feats = [c for c in feat_cols if 'inventory' in c or 'entry' in c or 'exit' in c or 'taker' in c or 'maker' in c or 'aggressive' in c or 'long' in c or 'short' in c or 'flat' in c]
    
    print(f"\nFeature categories:")
    print(f"  PM microstructure: {len(pm_feats)}")
    print(f"  CL/external: {len(cl_feats)}")
    print(f"  Temporal: {len(temporal_feats)}")
    print(f"  Position/execution: {len(position_feats)}")
    
    # Step 9: Save outputs
    print(f"\n{'='*70}")
    print("Saving outputs...")
    
    # Save feature matrix
    output_path = DATA_DIR / "feature_matrix.parquet"
    df.to_parquet(output_path, index=False)
    print(f"  Feature matrix saved to: {output_path}")
    print(f"  Shape: {df.shape}")
    
    # Save feature descriptions
    desc_path = RESULTS_DIR / "feature_descriptions.json"
    with open(desc_path, 'w') as f:
        json.dump(descriptions, f, indent=2)
    print(f"  Feature descriptions saved to: {desc_path}")
    
    # Save feature stats
    stats_path = RESULTS_DIR / "feature_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Feature stats saved to: {stats_path}")
    
    print(f"\n{'='*70}")
    print("DONE - Phase 4 Complete")
    print(f"{'='*70}")
    
    return df


if __name__ == "__main__":
    main()

