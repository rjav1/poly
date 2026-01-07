#!/usr/bin/env python3
"""
Phase 3: Strategy Parameter Extraction

Extracts strategy parameters from hypothesis test results.

Input: hypothesis_results.json
Output: strategy_params.json
"""

import json
from pathlib import Path
from typing import Dict, Any

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "results"  # hypothesis_results.json is in results/
OUTPUT_DIR = BASE_DIR / "results"


def load_hypothesis_results() -> Dict:
    """Load hypothesis test results."""
    path = DATA_DIR / "hypothesis_results.json"
    print(f"Loading: {path}")
    with open(path) as f:
        return json.load(f)


def extract_strategy_a_params(results: Dict) -> Dict[str, Any]:
    """
    Extract parameters for Strategy A: Underround Harvester.
    
    Based on vidarx and PurpleThunderBicycleMountain behavior.
    """
    h3 = results['H3_underround_harvesting']
    
    # Find wallets with positive edge
    positive_edge_wallets = {
        handle: data for handle, data in h3.items()
        if (data.get('pct_positive_edge') or 0) > 0.5
    }
    
    # Extract edge thresholds from successful harvesters
    edges = [data['median_edge'] for data in positive_edge_wallets.values() 
             if data['median_edge'] is not None]
    
    if edges:
        suggested_epsilon = min(edges)  # Conservative threshold
    else:
        suggested_epsilon = 0.02  # Default
    
    # Timing: avoid last 60s (most harvesters don't trade there)
    h1 = results['H1_late_window_concentration']
    harvester_handles = list(positive_edge_wallets.keys())
    
    if harvester_handles:
        # Check median tau for harvesters
        median_taus = [h1.get(h, {}).get('median_tau', 500) for h in harvester_handles]
        typical_tau = sum(median_taus) / len(median_taus)
    else:
        typical_tau = 500
    
    return {
        'strategy_name': 'UnderroundHarvester',
        'description': 'Buy both sides when sum_asks < 1 - epsilon, hold to expiry',
        'inspired_by': list(positive_edge_wallets.keys()),
        'parameters': {
            'epsilon': {
                'suggested': round(suggested_epsilon, 4),
                'sweep_range': [0.01, 0.02, 0.03, 0.04],
                'description': 'Minimum underround threshold to trigger entry'
            },
            'min_tau': {
                'suggested': 60,
                'sweep_range': [30, 60, 120],
                'description': 'Minimum time to expiry (avoid last N seconds)'
            },
            'max_tau': {
                'suggested': 840,
                'sweep_range': [600, 720, 840],
                'description': 'Maximum time to expiry (entry window start)'
            },
        },
        'evidence': {
            'vidarx_median_edge': h3.get('vidarx', {}).get('median_edge'),
            'vidarx_pct_positive': h3.get('vidarx', {}).get('pct_positive_edge'),
            'purple_median_edge': h3.get('PurpleThunderBicycleMountain', {}).get('median_edge'),
            'purple_pct_positive': h3.get('PurpleThunderBicycleMountain', {}).get('pct_positive_edge'),
        }
    }


def extract_strategy_b_params(results: Dict) -> Dict[str, Any]:
    """
    Extract parameters for Strategy B: Late Directional Taker (tsaiTop-style).
    
    Based on tsaiTop behavior.
    """
    h1 = results['H1_late_window_concentration']
    h2 = results['H2_two_sided_behavior']
    h4 = results['H4_short_horizon_scalping']
    
    # tsaiTop is the archetype for this strategy
    tsai_h1 = h1.get('tsaiTop', {})
    tsai_h2 = h2.get('tsaiTop', {})
    tsai_h4 = h4.get('tsaiTop', {})
    
    # Extract timing parameters
    share_last_300 = tsai_h1.get('share_last_300s', 0.63)
    median_tau = tsai_h1.get('median_tau', 159)
    
    # Hold time from matched trades
    median_hold = tsai_h4.get('median_hold_seconds', 190)
    
    return {
        'strategy_name': 'LateDirectionalTaker',
        'description': 'Take directional positions in last 5 minutes based on CL signal',
        'inspired_by': ['tsaiTop'],
        'parameters': {
            'tau_max': {
                'suggested': 300,
                'sweep_range': [120, 180, 300, 420],
                'description': 'Only trade when tau <= tau_max (last N seconds)'
            },
            'delta_threshold_bps': {
                'suggested': 10,
                'sweep_range': [5, 10, 15, 20],
                'description': 'Minimum |delta| in bps to trigger trade'
            },
            'hold_seconds': {
                'suggested': int(median_hold) if median_hold else 180,
                'sweep_range': [60, 120, 180, 240],
                'description': 'How long to hold position before exit'
            },
            'momentum_window': {
                'suggested': 10,
                'sweep_range': [5, 10, 20, 30],
                'description': 'Window for CL momentum calculation (seconds)'
            },
        },
        'evidence': {
            'tsai_share_last_300s': share_last_300,
            'tsai_median_tau': median_tau,
            'tsai_median_hold': median_hold,
            'tsai_pct_one_sided': 1 - tsai_h2.get('pct_both_outcomes', 0),
            'tsai_matched_pnl': tsai_h4.get('matched_pnl'),
        }
    }


def extract_strategy_c_params(results: Dict) -> Dict[str, Any]:
    """
    Extract parameters for Strategy C: Two-Sided Early, Tilt Late (Purple/Account-style).
    
    Based on PurpleThunderBicycleMountain and Account88888 behavior.
    """
    h1 = results['H1_late_window_concentration']
    h2 = results['H2_two_sided_behavior']
    h3 = results['H3_underround_harvesting']
    
    # These wallets trade both sides but still have timing concentration
    purple_h1 = h1.get('PurpleThunderBicycleMountain', {})
    account_h1 = h1.get('Account88888', {})
    
    purple_h2 = h2.get('PurpleThunderBicycleMountain', {})
    account_h2 = h2.get('Account88888', {})
    
    # Compute timing thresholds
    purple_share_300 = purple_h1.get('share_last_300s', 0.42)
    purple_share_120 = purple_h1.get('share_last_120s', 0.11)
    purple_median_tau = purple_h1.get('median_tau', 361)
    
    account_share_300 = account_h1.get('share_last_300s', 0.26)
    account_median_tau = account_h1.get('median_tau', 535)
    
    # Inventory phase: early in market (tau > 300)
    # Tilt phase: late in market (tau < 120-180)
    
    return {
        'strategy_name': 'TwoSidedEarlyTiltLate',
        'description': 'Build matched inventory early, add net exposure late based on CL sign',
        'inspired_by': ['PurpleThunderBicycleMountain', 'Account88888'],
        'parameters': {
            'inventory_phase_end': {
                'suggested': 300,
                'sweep_range': [180, 300, 420],
                'description': 'Stop building inventory when tau drops below this (seconds)'
            },
            'tilt_phase_start': {
                'suggested': 180,
                'sweep_range': [120, 180, 240],
                'description': 'Start adding net exposure when tau drops below this (seconds)'
            },
            'inventory_epsilon': {
                'suggested': 0.02,
                'sweep_range': [0.01, 0.02, 0.03],
                'description': 'Underround threshold for inventory building'
            },
            'tilt_delta_threshold_bps': {
                'suggested': 15,
                'sweep_range': [10, 15, 20],
                'description': 'Minimum |delta| in bps to tilt'
            },
            'tilt_size_ratio': {
                'suggested': 0.5,
                'sweep_range': [0.25, 0.5, 1.0],
                'description': 'Tilt position size as ratio of inventory'
            },
        },
        'evidence': {
            'purple_pct_both_outcomes': purple_h2.get('pct_both_outcomes'),
            'purple_share_last_300s': purple_share_300,
            'purple_share_last_120s': purple_share_120,
            'purple_median_tau': purple_median_tau,
            'account_pct_both_outcomes': account_h2.get('pct_both_outcomes'),
            'account_share_last_300s': account_share_300,
            'account_median_tau': account_median_tau,
        }
    }


def main():
    print("=" * 60)
    print("Phase 3: Strategy Parameter Extraction")
    print("=" * 60)
    
    # Load hypothesis results
    results = load_hypothesis_results()
    
    # Extract parameters for each strategy
    strategy_params = {
        'Strategy_A': extract_strategy_a_params(results),
        'Strategy_B': extract_strategy_b_params(results),
        'Strategy_C': extract_strategy_c_params(results),
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTED STRATEGY PARAMETERS")
    print("=" * 60)
    
    for strategy_key, strategy in strategy_params.items():
        print(f"\n### {strategy['strategy_name']} ({strategy_key})")
        print(f"Description: {strategy['description']}")
        print(f"Inspired by: {', '.join(strategy['inspired_by'])}")
        print("\nParameters:")
        for param_name, param_info in strategy['parameters'].items():
            print(f"  {param_name}:")
            print(f"    Suggested: {param_info['suggested']}")
            print(f"    Sweep: {param_info['sweep_range']}")
    
    # Save results
    output_path = OUTPUT_DIR / "strategy_params.json"
    with open(output_path, 'w') as f:
        json.dump(strategy_params, f, indent=2, default=str)
    print(f"\n\nResults saved to: {output_path}")
    
    # Also create a parameter summary table
    print("\n" + "=" * 60)
    print("PARAMETER SUMMARY TABLE")
    print("=" * 60)
    
    print("\n| Strategy | Key Parameter | Suggested | Sweep Range |")
    print("|----------|---------------|-----------|-------------|")
    
    for strategy_key, strategy in strategy_params.items():
        for i, (param_name, param_info) in enumerate(strategy['parameters'].items()):
            strategy_name = strategy['strategy_name'] if i == 0 else ""
            print(f"| {strategy_name:<20} | {param_name:<15} | {param_info['suggested']:<9} | {param_info['sweep_range']} |")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    
    return strategy_params


if __name__ == "__main__":
    main()

