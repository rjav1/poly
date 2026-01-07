#!/usr/bin/env python3
"""
Phase 6: Hypothesis Generation

Converts discovered rules into testable strategy hypotheses following 
the research-grade template with conditions, actions, mechanisms, and failure modes.

Input:
- feature_matrix.parquet (trades with features)
- policy_rules.json (extracted rules from Phase 5)
- execution_summary.json (execution style data)
- inventory_patterns.json (position data)

Output:
- hypotheses.json (ranked hypotheses)
- hypothesis_evidence.json (supporting evidence)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict
import hashlib

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"


@dataclass
class Hypothesis:
    """Research hypothesis with full specification."""
    hypothesis_id: str
    name: str
    category: str  # PM_ONLY, CL_PM_LEADLAG, INVENTORY, TIMING
    condition: str
    action: str
    mechanism: str
    failure_modes: List[str]
    evidence: Dict[str, Any]
    parameters: Dict[str, Any]
    ranking_score: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


def load_data() -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
    """Load all required data."""
    print("Loading data...")
    
    # Feature matrix
    fm_path = DATA_DIR / "feature_matrix.parquet"
    trades = pd.read_parquet(fm_path)
    print(f"  Trades: {len(trades):,}")
    
    # Policy rules
    rules_path = RESULTS_DIR / "policy_rules.json"
    with open(rules_path, 'r') as f:
        policy_rules = json.load(f)
    
    # Execution summary
    exec_path = RESULTS_DIR / "execution_summary.json"
    with open(exec_path, 'r') as f:
        exec_summary = json.load(f)
    
    # Inventory patterns
    inv_path = RESULTS_DIR / "inventory_patterns.json"
    with open(inv_path, 'r') as f:
        inventory_patterns = json.load(f)
    
    return trades, policy_rules, exec_summary, inventory_patterns


def generate_underround_hypotheses(
    trades: pd.DataFrame,
    inventory_patterns: Dict
) -> List[Hypothesis]:
    """Generate hypotheses related to underround harvesting."""
    hypotheses = []
    
    # H6: Basic underround harvesting
    # Find wallets that trade when underround exists
    if 'feat_underround_positive' in trades.columns:
        underround_trades = trades[trades['feat_underround_positive'] > 0.01]
        wallets_using_underround = underround_trades['wallet'].value_counts()
        
        # Calculate evidence
        if len(underround_trades) > 0:
            avg_underround = underround_trades['feat_underround_positive'].mean()
            pct_both_sides = (underround_trades['direction'] == 'BOTH').mean()
            
            # Which wallets show this behavior
            high_underround_wallets = []
            for wallet in wallets_using_underround.index[:5]:
                wallet_ur = underround_trades[underround_trades['wallet'] == wallet]
                if len(wallet_ur) > 10:
                    high_underround_wallets.append(wallet)
            
            hypotheses.append(Hypothesis(
                hypothesis_id="H6_underround_harvest",
                name="Underround Harvesting",
                category="PM_ONLY",
                condition="sum_asks < 1 - epsilon (underround > 1%)",
                action="Buy both UP and DOWN tokens simultaneously (complete set)",
                mechanism="When sum_asks < 1, buying both sides for less than $1 guarantees $1 at expiry regardless of outcome. The edge equals the underround magnitude.",
                failure_modes=[
                    "Insufficient capacity (size limits)",
                    "Execution slippage erodes edge",
                    "Quote staleness leads to stale underround",
                    "Competition from other arb traders"
                ],
                evidence={
                    'n_trades_with_underround': len(underround_trades),
                    'avg_underround_when_trading': float(avg_underround),
                    'pct_both_sides': float(pct_both_sides),
                    'wallets': high_underround_wallets,
                },
                parameters={
                    'epsilon': {'suggested': 0.01, 'sweep': [0.005, 0.01, 0.02, 0.03]},
                    'min_tau': {'suggested': 60, 'sweep': [30, 60, 120]},
                    'max_tau': {'suggested': 840, 'sweep': [720, 840]},
                    'min_capacity': {'suggested': 1, 'sweep': [0, 1, 5, 10]},
                },
                ranking_score=0.9 if len(high_underround_wallets) >= 2 else 0.7,
            ))
    
    # H7: Late-window underround (tau < 300)
    if 'feat_underround_x_late' in trades.columns:
        late_underround = trades[
            (trades['feat_underround_positive'] > 0.01) & 
            (trades['feat_late'] == 1)
        ]
        
        if len(late_underround) > 50:
            hypotheses.append(Hypothesis(
                hypothesis_id="H7_late_underround",
                name="Late Window Underround",
                category="PM_ONLY",
                condition="underround > 1% AND tau < 120s",
                action="Buy complete set in last 2 minutes",
                mechanism="Late-window underround opportunities may be more reliable as market-makers widen spreads near expiry, creating exploitable inefficiencies.",
                failure_modes=[
                    "Time pressure increases execution risk",
                    "Late spreads may be wider",
                    "Lower liquidity near expiry"
                ],
                evidence={
                    'n_late_underround_trades': len(late_underround),
                    'avg_underround': float(late_underround['feat_underround_positive'].mean()),
                },
                parameters={
                    'epsilon': {'suggested': 0.015, 'sweep': [0.01, 0.015, 0.02]},
                    'max_tau': {'suggested': 120, 'sweep': [60, 120, 180]},
                },
                ranking_score=0.75,
            ))
    
    return hypotheses


def generate_timing_hypotheses(
    trades: pd.DataFrame,
    inventory_patterns: Dict
) -> List[Hypothesis]:
    """Generate hypotheses related to trade timing."""
    hypotheses = []
    
    # H8: Late directional taker (tsaiTop-inspired)
    late_trades = trades[trades['feat_late'] == 1]
    
    if len(late_trades) > 0:
        # Check for directional bias in late trades
        late_directional = late_trades[late_trades['direction'].isin(['Up', 'Down'])]
        
        if len(late_directional) > 20:
            # Look for wallets with late directional pattern
            late_wallets = []
            for wallet in trades['wallet'].unique():
                w_trades = trades[trades['wallet'] == wallet]
                late_pct = (w_trades['feat_late'] == 1).mean()
                directional_pct = w_trades['direction'].isin(['Up', 'Down']).mean()
                
                if late_pct > 0.3 and directional_pct > 0.5:
                    late_wallets.append({
                        'wallet': wallet,
                        'late_pct': float(late_pct),
                        'directional_pct': float(directional_pct),
                    })
            
            hypotheses.append(Hypothesis(
                hypothesis_id="H8_late_directional",
                name="Late Directional Taker",
                category="TIMING",
                condition="tau < 300s AND |delta_bps| > threshold",
                action="Take directional position based on CL signal",
                mechanism="In late window, CL price movement has high predictive value for final outcome. Taking directional position captures this information advantage.",
                failure_modes=[
                    "CL signal is noise not signal",
                    "Spread cost exceeds expected edge",
                    "Late liquidity insufficient",
                    "Information already priced in"
                ],
                evidence={
                    'n_late_directional_trades': len(late_directional),
                    'late_wallets': late_wallets[:3],
                },
                parameters={
                    'max_tau': {'suggested': 300, 'sweep': [120, 180, 300, 420]},
                    'delta_threshold_bps': {'suggested': 10, 'sweep': [5, 10, 15, 20]},
                    'hold_seconds': {'suggested': 180, 'sweep': [60, 120, 180, 240]},
                },
                ranking_score=0.7 if late_wallets else 0.5,
            ))
    
    # H9: Early entry, late exit
    early_trades = trades[trades['feat_early'] == 1]
    
    if len(early_trades) > 100:
        # Check for inventory building early
        early_both = early_trades[early_trades['direction'] == 'BOTH']
        
        if len(early_both) > 50:
            hypotheses.append(Hypothesis(
                hypothesis_id="H9_early_inventory",
                name="Early Inventory Build",
                category="INVENTORY",
                condition="tau > 600s AND underround exists",
                action="Build matched inventory early, hold to expiry",
                mechanism="Building inventory early when spreads are tighter allows accumulating position at better prices. Hold to expiry for guaranteed payoff.",
                failure_modes=[
                    "Early prices may not be better",
                    "Capital tied up longer",
                    "Missing late opportunities"
                ],
                evidence={
                    'n_early_both_trades': len(early_both),
                    'early_underround_mean': float(early_both['feat_underround_positive'].mean()) if 'feat_underround_positive' in early_both.columns else 0,
                },
                parameters={
                    'min_tau': {'suggested': 600, 'sweep': [500, 600, 700]},
                    'epsilon': {'suggested': 0.015, 'sweep': [0.01, 0.015, 0.02]},
                },
                ranking_score=0.65,
            ))
    
    return hypotheses


def generate_spread_hypotheses(
    trades: pd.DataFrame,
    exec_summary: Dict
) -> List[Hypothesis]:
    """Generate hypotheses related to spread conditions."""
    hypotheses = []
    
    # H10: Tight spread entry
    if 'feat_avg_spread' in trades.columns:
        median_spread = trades['feat_avg_spread'].median()
        tight_spread_trades = trades[trades['feat_avg_spread'] < median_spread * 0.8]
        
        if len(tight_spread_trades) > 100:
            hypotheses.append(Hypothesis(
                hypothesis_id="H10_tight_spread",
                name="Tight Spread Entry",
                category="PM_ONLY",
                condition="spread < median_spread * 0.8",
                action="Enter position when spreads are unusually tight",
                mechanism="Tight spreads indicate either high liquidity or stale quotes. If it's liquidity, execution is cheaper. If it's staleness, there may be information advantage.",
                failure_modes=[
                    "Tight spreads may widen immediately",
                    "May indicate low volatility periods with no edge",
                    "Staleness means quotes may not be executable"
                ],
                evidence={
                    'n_tight_spread_trades': len(tight_spread_trades),
                    'median_spread': float(median_spread) if not pd.isna(median_spread) else 0,
                    'tight_threshold': float(median_spread * 0.8) if not pd.isna(median_spread) else 0,
                },
                parameters={
                    'spread_percentile': {'suggested': 20, 'sweep': [10, 20, 30]},
                    'min_spread_bps': {'suggested': 5, 'sweep': [0, 5, 10]},
                },
                ranking_score=0.55,
            ))
    
    return hypotheses


def generate_momentum_hypotheses(
    trades: pd.DataFrame,
    policy_rules: Dict
) -> List[Hypothesis]:
    """Generate hypotheses related to price momentum."""
    hypotheses = []
    
    # H11: CL momentum following
    if 'feat_cl_momentum_10s' in trades.columns:
        # Check if trades follow momentum direction
        mom_trades = trades[trades['feat_cl_momentum_10s_abs'] > 0]
        
        if len(mom_trades) > 100:
            # Check correlation between momentum sign and trade direction
            up_trades = mom_trades[mom_trades['direction'] == 'Up']
            down_trades = mom_trades[mom_trades['direction'] == 'Down']
            
            if len(up_trades) > 20 and len(down_trades) > 20:
                up_mom_mean = up_trades['feat_cl_momentum_10s'].mean()
                down_mom_mean = down_trades['feat_cl_momentum_10s'].mean()
                
                # If UP trades have positive momentum and DOWN have negative, signal following
                follows_momentum = up_mom_mean > down_mom_mean
                
                hypotheses.append(Hypothesis(
                    hypothesis_id="H11_momentum_follow",
                    name="CL Momentum Following",
                    category="CL_PM_LEADLAG",
                    condition="|CL_momentum_10s| > threshold",
                    action="Trade in direction of CL momentum",
                    mechanism="CL price momentum indicates direction of underlying asset movement. PM prices may lag CL, creating directional opportunity.",
                    failure_modes=[
                        "Momentum reversal",
                        "PM already priced in CL move",
                        "Momentum is noise",
                        "Spread cost exceeds momentum magnitude"
                    ],
                    evidence={
                        'n_momentum_trades': len(mom_trades),
                        'up_avg_momentum': float(up_mom_mean) if not pd.isna(up_mom_mean) else 0,
                        'down_avg_momentum': float(down_mom_mean) if not pd.isna(down_mom_mean) else 0,
                        'follows_momentum': follows_momentum,
                    },
                    parameters={
                        'momentum_threshold': {'suggested': 0.0001, 'sweep': [0.00005, 0.0001, 0.0002]},
                        'momentum_window': {'suggested': 10, 'sweep': [5, 10, 20]},
                    },
                    ranking_score=0.6 if follows_momentum else 0.4,
                ))
    
    return hypotheses


def generate_execution_hypotheses(
    trades: pd.DataFrame,
    exec_summary: Dict
) -> List[Hypothesis]:
    """Generate hypotheses related to execution style."""
    hypotheses = []
    
    # H12: Maker-only underround
    if 'execution_type' in trades.columns and 'feat_underround_positive' in trades.columns:
        maker_underround = trades[
            (trades['execution_type'] == 'MAKER') & 
            (trades['feat_underround_positive'] > 0.01)
        ]
        
        if len(maker_underround) > 50:
            hypotheses.append(Hypothesis(
                hypothesis_id="H12_maker_underround",
                name="Passive Underround Harvesting",
                category="PM_ONLY",
                condition="underround > 1% AND execution via limit orders",
                action="Post limit orders to capture underround passively",
                mechanism="Posting limit orders inside the underround allows earning the spread while still capturing the complete-set arbitrage. Lower execution risk than taker.",
                failure_modes=[
                    "Orders may not fill",
                    "Queue priority disadvantage",
                    "Other makers crowd out",
                    "Underround disappears before fill"
                ],
                evidence={
                    'n_maker_underround_trades': len(maker_underround),
                    'maker_pct_overall': float((trades['execution_type'] == 'MAKER').mean()),
                },
                parameters={
                    'limit_offset': {'suggested': 0.005, 'sweep': [0.002, 0.005, 0.01]},
                    'min_underround': {'suggested': 0.01, 'sweep': [0.005, 0.01, 0.015]},
                },
                ranking_score=0.6,
            ))
    
    return hypotheses


def rank_hypotheses(hypotheses: List[Hypothesis]) -> List[Hypothesis]:
    """Rank hypotheses by likelihood of being real and actionable."""
    # Score components:
    # 1. Category preference: PM_ONLY > INVENTORY > TIMING > CL_PM_LEADLAG
    # 2. Evidence strength: more wallets, more trades
    # 3. Mechanism clarity
    
    category_scores = {
        'PM_ONLY': 1.0,
        'INVENTORY': 0.8,
        'TIMING': 0.7,
        'CL_PM_LEADLAG': 0.6,
    }
    
    for h in hypotheses:
        # Base score from ranking_score
        score = h.ranking_score
        
        # Adjust by category
        score *= category_scores.get(h.category, 0.5)
        
        # Adjust by evidence
        evidence = h.evidence
        if 'wallets' in evidence and len(evidence['wallets']) >= 3:
            score *= 1.2
        if evidence.get('n_trades_with_underround', 0) > 1000:
            score *= 1.1
        
        h.ranking_score = min(1.0, score)
    
    # Sort by ranking score
    hypotheses.sort(key=lambda h: h.ranking_score, reverse=True)
    
    return hypotheses


def generate_all_hypotheses(
    trades: pd.DataFrame,
    policy_rules: Dict,
    exec_summary: Dict,
    inventory_patterns: Dict
) -> List[Hypothesis]:
    """Generate all hypotheses from different sources."""
    all_hypotheses = []
    
    print("\nGenerating hypotheses...")
    
    # Underround hypotheses
    ur_hyps = generate_underround_hypotheses(trades, inventory_patterns)
    print(f"  Underround hypotheses: {len(ur_hyps)}")
    all_hypotheses.extend(ur_hyps)
    
    # Timing hypotheses
    timing_hyps = generate_timing_hypotheses(trades, inventory_patterns)
    print(f"  Timing hypotheses: {len(timing_hyps)}")
    all_hypotheses.extend(timing_hyps)
    
    # Spread hypotheses
    spread_hyps = generate_spread_hypotheses(trades, exec_summary)
    print(f"  Spread hypotheses: {len(spread_hyps)}")
    all_hypotheses.extend(spread_hyps)
    
    # Momentum hypotheses
    mom_hyps = generate_momentum_hypotheses(trades, policy_rules)
    print(f"  Momentum hypotheses: {len(mom_hyps)}")
    all_hypotheses.extend(mom_hyps)
    
    # Execution hypotheses
    exec_hyps = generate_execution_hypotheses(trades, exec_summary)
    print(f"  Execution hypotheses: {len(exec_hyps)}")
    all_hypotheses.extend(exec_hyps)
    
    # Rank all hypotheses
    all_hypotheses = rank_hypotheses(all_hypotheses)
    
    return all_hypotheses


def main():
    print("=" * 70)
    print("Phase 6: Hypothesis Generation")
    print("=" * 70)
    
    # Step 1: Load all data
    trades, policy_rules, exec_summary, inventory_patterns = load_data()
    
    # Step 2: Generate all hypotheses
    hypotheses = generate_all_hypotheses(
        trades, policy_rules, exec_summary, inventory_patterns
    )
    
    # Step 3: Print summary
    print("\n" + "=" * 70)
    print("GENERATED HYPOTHESES (Ranked)")
    print("=" * 70)
    
    for i, h in enumerate(hypotheses):
        print(f"\n{i+1}. [{h.category}] {h.name} (ID: {h.hypothesis_id})")
        print(f"   Score: {h.ranking_score:.2f}")
        print(f"   Condition: {h.condition}")
        print(f"   Action: {h.action}")
        print(f"   Mechanism: {h.mechanism[:100]}...")
    
    # Step 4: Save outputs
    print("\n" + "=" * 70)
    print("Saving outputs...")
    
    # Save hypotheses
    hyp_path = RESULTS_DIR / "hypotheses.json"
    hypotheses_data = [h.to_dict() for h in hypotheses]
    with open(hyp_path, 'w') as f:
        json.dump(hypotheses_data, f, indent=2, default=str)
    print(f"  Hypotheses saved to: {hyp_path}")
    
    # Save evidence summary
    evidence_path = RESULTS_DIR / "hypothesis_evidence.json"
    evidence_data = {
        h.hypothesis_id: {
            'name': h.name,
            'category': h.category,
            'evidence': h.evidence,
            'parameters': h.parameters,
        }
        for h in hypotheses
    }
    with open(evidence_path, 'w') as f:
        json.dump(evidence_data, f, indent=2, default=str)
    print(f"  Evidence saved to: {evidence_path}")
    
    print(f"\nTotal hypotheses generated: {len(hypotheses)}")
    print(f"  PM_ONLY: {len([h for h in hypotheses if h.category == 'PM_ONLY'])}")
    print(f"  TIMING: {len([h for h in hypotheses if h.category == 'TIMING'])}")
    print(f"  INVENTORY: {len([h for h in hypotheses if h.category == 'INVENTORY'])}")
    print(f"  CL_PM_LEADLAG: {len([h for h in hypotheses if h.category == 'CL_PM_LEADLAG'])}")
    
    print("\n" + "=" * 70)
    print("DONE - Phase 6 Complete")
    print("=" * 70)
    
    return hypotheses


if __name__ == "__main__":
    main()

