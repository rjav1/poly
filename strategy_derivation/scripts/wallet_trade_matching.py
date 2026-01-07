#!/usr/bin/env python3
"""
Wallet Trade Matching Analysis

Matches profitable trader trades to quote-level conditions at trade timestamps.
Validates whether traders captured underround opportunities.

Note: Vidarx only traded BTC, not ETH. We use PurpleThunderBicycleMountain and 
Account88888 who have ETH trades in our volume markets.

Output:
- wallet_trade_match_analysis.json: Matching statistics
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timezone

# Configuration
BASE_DIR = Path(__file__).parent.parent
RESEARCH_DIR = BASE_DIR.parent / "data_v2" / "research"
WALLET_DIR = BASE_DIR / "profitable_traders_wallet_data"
OUTPUT_DIR = BASE_DIR / "results"

# Volume markets time range
VOLUME_MARKETS_START = 1767717000  # 2026-01-06 16:30:00 UTC
VOLUME_MARKETS_END = 1767727800    # 2026-01-06 19:30:00 UTC

# Price filter (same as quote analysis)
MIN_PRICE = 0.05
MAX_PRICE = 0.95


def load_quote_data() -> pd.DataFrame:
    """Load quote data for volume markets."""
    df = pd.read_parquet(RESEARCH_DIR / "canonical_dataset_all_assets.parquet")
    
    # Filter to volume markets
    volume_market_ids = [
        '20260106_1630', '20260106_1645', '20260106_1700', '20260106_1715',
        '20260106_1730', '20260106_1745', '20260106_1800', '20260106_1815',
        '20260106_1830', '20260106_1845', '20260106_1900', '20260106_1915'
    ]
    df = df[df['market_id'].isin(volume_market_ids)].copy()
    
    # Parse timestamp and create epoch column for matching
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['ts_epoch'] = df['timestamp'].astype('int64') // 10**9
    
    # Compute derived columns
    df['underround'] = 1 - df['sum_asks']
    df['overround'] = df['sum_bids'] - 1
    df['achievable_capacity'] = np.minimum(
        df['pm_up_best_ask_size'].fillna(0),
        df['pm_down_best_ask_size'].fillna(0)
    )
    
    return df


def load_wallet_trades() -> pd.DataFrame:
    """Load wallet trades for ETH in volume markets time range."""
    df = pd.read_parquet(WALLET_DIR / "polymarket_multi_activity.parquet")
    
    # Filter to ETH
    df = df[df['title'].str.contains('Ethereum', case=False, na=False)].copy()
    
    # Convert timestamp to epoch
    df['ts_epoch'] = pd.to_datetime(df['timestamp_utc']).astype('int64') // 10**9
    
    # Filter to volume markets time range
    df = df[(df['ts_epoch'] >= VOLUME_MARKETS_START) & (df['ts_epoch'] <= VOLUME_MARKETS_END)].copy()
    
    # Filter to TRADE type and BUY side (underround harvesting = buying)
    df = df[(df['type'] == 'TRADE') & (df['side'] == 'BUY')].copy()
    
    print(f"Loaded {len(df)} ETH BUY trades in volume markets time range")
    print(f"Handles: {df['_handle'].value_counts().to_dict()}")
    
    return df


def extract_market_info_from_slug(slug: str) -> Dict[str, Any]:
    """Extract market start time from slug like 'eth-updown-15m-1767717000'."""
    if pd.isna(slug):
        return {'start_ts': None}
    
    parts = slug.split('-')
    if len(parts) >= 4:
        try:
            start_ts = int(parts[-1])
            return {'start_ts': start_ts}
        except ValueError:
            return {'start_ts': None}
    return {'start_ts': None}


def match_trades_to_quotes(trades_df: pd.DataFrame, quotes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match each trade to quote conditions at trade time.
    
    For each trade, find the quote row with matching:
    - Market (by start timestamp)
    - Time within market (by t = seconds from market start)
    """
    # Precompute market_start_ts for quotes (use .apply with timestamp() for proper conversion)
    quotes_df = quotes_df.copy()
    quotes_df['market_start_ts'] = quotes_df['market_start'].apply(
        lambda x: int(pd.Timestamp(x).timestamp()) if pd.notna(x) else None
    )
    
    # Create lookup dict for faster matching
    # Key: (market_start_ts, t) -> quote row
    quote_lookup = {}
    for _, row in quotes_df.iterrows():
        if pd.notna(row['market_start_ts']):
            key = (int(row['market_start_ts']), int(row['t']))
            quote_lookup[key] = row
    
    print(f"Built quote lookup with {len(quote_lookup)} entries")
    print(f"Sample keys: {list(quote_lookup.keys())[:3]}")
    
    matches = []
    skipped_no_slug = 0
    skipped_outside_window = 0
    skipped_no_match = 0
    
    for _, trade in trades_df.iterrows():
        # Extract market start from slug
        slug = trade.get('slug', '')
        if pd.isna(slug) or not slug:
            skipped_no_slug += 1
            continue
            
        market_info = extract_market_info_from_slug(slug)
        market_start_ts = market_info['start_ts']
        
        if market_start_ts is None:
            skipped_no_slug += 1
            continue
        
        # Compute t (seconds from market start)
        trade_ts = trade['ts_epoch']
        t = trade_ts - market_start_ts
        
        # Skip if trade is outside market window (0-900 seconds)
        if t < 0 or t >= 900:
            skipped_outside_window += 1
            continue
        
        # Lookup quote
        key = (market_start_ts, t)
        if key not in quote_lookup:
            skipped_no_match += 1
            continue
        
        quote = quote_lookup[key]
        
        matches.append({
            'handle': trade['_handle'],
            'trade_ts': trade_ts,
            'slug': slug,
            'market_start_ts': market_start_ts,
            't': t,
            'tau': 900 - t,
            'trade_side': trade['side'],
            'trade_outcome': trade['outcome'],
            'trade_price': trade['price'],
            'trade_size': trade['size'],
            # Quote conditions at trade time
            'quote_sum_asks': quote['sum_asks'],
            'quote_sum_bids': quote['sum_bids'],
            'quote_underround': quote['underround'],
            'quote_overround': quote['overround'],
            'quote_up_ask': quote['pm_up_best_ask'],
            'quote_down_ask': quote['pm_down_best_ask'],
            'quote_up_ask_size': quote['pm_up_best_ask_size'],
            'quote_down_ask_size': quote['pm_down_best_ask_size'],
            'quote_capacity': quote['achievable_capacity'],
            # Price validity
            'valid_prices': (
                (quote['pm_up_best_ask'] >= MIN_PRICE) &
                (quote['pm_up_best_ask'] <= MAX_PRICE) &
                (quote['pm_down_best_ask'] >= MIN_PRICE) &
                (quote['pm_down_best_ask'] <= MAX_PRICE)
            ),
        })
    
    print(f"Skipped: {skipped_no_slug} no slug, {skipped_outside_window} outside window, {skipped_no_match} no quote match")
    
    return pd.DataFrame(matches)


def analyze_paired_trades(matches_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze paired trades (UP + DOWN within same second).
    This is the signature of underround harvesting.
    """
    results = {}
    
    for handle in matches_df['handle'].unique():
        handle_df = matches_df[matches_df['handle'] == handle]
        
        # Group by (market_start_ts, t) to find paired trades
        paired_stats = []
        
        for (market_ts, t), group in handle_df.groupby(['market_start_ts', 't']):
            up_trades = group[group['trade_outcome'] == 'Up']
            down_trades = group[group['trade_outcome'] == 'Down']
            
            if len(up_trades) == 0 or len(down_trades) == 0:
                continue
            
            # Compute VWAP for each side
            up_vwap = (up_trades['trade_price'] * up_trades['trade_size']).sum() / up_trades['trade_size'].sum()
            down_vwap = (down_trades['trade_price'] * down_trades['trade_size']).sum() / down_trades['trade_size'].sum()
            
            set_cost = up_vwap + down_vwap
            realized_edge = 1 - set_cost
            
            # Get quote conditions at this time
            quote_row = group.iloc[0]
            
            paired_stats.append({
                't': t,
                'tau': 900 - t,
                'up_vwap': up_vwap,
                'down_vwap': down_vwap,
                'set_cost': set_cost,
                'realized_edge': realized_edge,
                'matched_qty': min(up_trades['trade_size'].sum(), down_trades['trade_size'].sum()),
                'quote_underround': quote_row['quote_underround'],
                'quote_sum_asks': quote_row['quote_sum_asks'],
                'valid_prices': quote_row['valid_prices'],
            })
        
        if not paired_stats:
            results[handle] = {
                'n_paired_seconds': 0,
                'n_positive_edge': 0,
                'median_realized_edge': None,
                'mean_realized_edge': None,
                'total_matched_qty': 0,
            }
            continue
        
        paired_df = pd.DataFrame(paired_stats)
        valid_pairs = paired_df[paired_df['valid_prices']]
        
        results[handle] = {
            'n_paired_seconds': len(paired_df),
            'n_valid_pairs': len(valid_pairs),
            'n_positive_edge': (paired_df['realized_edge'] > 0).sum(),
            'pct_positive_edge': (paired_df['realized_edge'] > 0).mean() * 100,
            'median_realized_edge': paired_df['realized_edge'].median(),
            'mean_realized_edge': paired_df['realized_edge'].mean(),
            'p25_realized_edge': paired_df['realized_edge'].quantile(0.25),
            'p75_realized_edge': paired_df['realized_edge'].quantile(0.75),
            'total_matched_qty': paired_df['matched_qty'].sum(),
            # Quote-trade alignment
            'median_quote_underround': valid_pairs['quote_underround'].median() if len(valid_pairs) > 0 else None,
            'correlation_realized_vs_quote': paired_df['realized_edge'].corr(paired_df['quote_underround']) if len(paired_df) > 1 else None,
        }
    
    return results


def main():
    print("=" * 70)
    print("WALLET TRADE MATCHING ANALYSIS")
    print("=" * 70)
    print()
    
    # Load data
    print("Loading quote data...")
    quotes_df = load_quote_data()
    print(f"Loaded {len(quotes_df)} quote rows from {quotes_df['market_id'].nunique()} markets")
    print()
    
    print("Loading wallet trades...")
    trades_df = load_wallet_trades()
    print()
    
    # Match trades to quotes
    print("Matching trades to quote conditions...")
    matches_df = match_trades_to_quotes(trades_df, quotes_df)
    print(f"Matched {len(matches_df)} trades to quotes")
    print()
    
    # Analyze by handle
    print("=" * 70)
    print("TRADE-QUOTE MATCHING BY HANDLE")
    print("=" * 70)
    
    match_stats = {}
    for handle in matches_df['handle'].unique():
        hdf = matches_df[matches_df['handle'] == handle]
        valid = hdf[hdf['valid_prices']]
        
        match_stats[handle] = {
            'total_matched_trades': len(hdf),
            'trades_at_valid_prices': len(valid),
            'trades_at_underround': (valid['quote_underround'] > 0).sum() if len(valid) > 0 else 0,
            'pct_at_underround': ((valid['quote_underround'] > 0).mean() * 100) if len(valid) > 0 else 0,
            'median_underround_at_trade': valid['quote_underround'].median() if len(valid) > 0 else None,
        }
        
        print(f"\n{handle}:")
        print(f"  Matched trades: {len(hdf)}")
        print(f"  Trades at valid prices: {len(valid)}")
        if len(valid) > 0:
            print(f"  Trades at underround: {match_stats[handle]['trades_at_underround']} ({match_stats[handle]['pct_at_underround']:.1f}%)")
            print(f"  Median underround at trade: ${match_stats[handle]['median_underround_at_trade']:.4f}")
    
    # Analyze paired trades
    print()
    print("=" * 70)
    print("PAIRED TRADE ANALYSIS (UNDERROUND HARVESTING SIGNATURE)")
    print("=" * 70)
    
    paired_results = analyze_paired_trades(matches_df)
    
    for handle, stats in paired_results.items():
        print(f"\n{handle}:")
        print(f"  Paired seconds: {stats['n_paired_seconds']}")
        if stats['n_paired_seconds'] > 0:
            print(f"  Positive edge: {stats['n_positive_edge']} ({stats['pct_positive_edge']:.1f}%)")
            print(f"  Median realized edge: ${stats['median_realized_edge']:.4f}")
            print(f"  Mean realized edge: ${stats['mean_realized_edge']:.4f}")
            print(f"  Total matched qty: {stats['total_matched_qty']:.1f}")
            if stats['correlation_realized_vs_quote'] is not None:
                print(f"  Correlation (realized vs quote): {stats['correlation_realized_vs_quote']:.3f}")
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    results = {
        'generated_at': datetime.now().isoformat(),
        'volume_markets_range': {
            'start': datetime.fromtimestamp(VOLUME_MARKETS_START, tz=timezone.utc).isoformat(),
            'end': datetime.fromtimestamp(VOLUME_MARKETS_END, tz=timezone.utc).isoformat(),
        },
        'total_matched_trades': len(matches_df),
        'match_stats_by_handle': match_stats,
        'paired_trade_analysis': paired_results,
    }
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "wallet_trade_match_analysis.json"
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return obj
    
    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [clean_dict(v) for v in d]
        else:
            return convert_numpy(d)
    
    with open(output_path, 'w') as f:
        json.dump(clean_dict(results), f, indent=2)
    
    print(f"\nSaved: {output_path}")
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()

