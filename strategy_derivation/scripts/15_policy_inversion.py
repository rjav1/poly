#!/usr/bin/env python3
"""
Phase 5: Policy Inversion (State -> Action Modeling)

Infers what market conditions predict wallet actions using interpretable models.
Extracts rules and thresholds from model coefficients and tree structures.

Input:
- feature_matrix.parquet (trades with full feature set from Phase 4)

Output:
- policy_rules.json (extracted rules per wallet)
- policy_models_summary.json (model performance and feature importances)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESEARCH_DIR = BASE_DIR.parent / "data_v2" / "research"
MARKET_DURATION_SECONDS = 900

# Try to import sklearn
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not available. Using simplified rule extraction.")


def load_feature_matrix() -> pd.DataFrame:
    """Load feature matrix from Phase 4."""
    path = DATA_DIR / "feature_matrix.parquet"
    print(f"Loading feature matrix from: {path}")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} rows")
    return df


def load_market_data() -> pd.DataFrame:
    """Load canonical market data for negative sampling."""
    path = RESEARCH_DIR / "canonical_dataset_all_assets.parquet"
    print(f"\nLoading market data for negative sampling from: {path}")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} market-seconds")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns for modeling."""
    # Get all feat_ columns that are numeric
    feat_cols = []
    for col in df.columns:
        if col.startswith('feat_') and df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            # Skip categorical encoded columns that might have issues
            if df[col].isna().mean() < 0.5:  # Less than 50% missing
                feat_cols.append(col)
    return feat_cols


def prepare_trade_decision_dataset(
    trades: pd.DataFrame,
    market_data: pd.DataFrame,
    wallet: str,
    subsample_negatives: int = 10000
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Prepare dataset for modeling 'trade_now' decision.
    
    Positive samples: seconds when wallet traded
    Negative samples: seconds when wallet did NOT trade (subsampled)
    
    Returns:
        X: Feature matrix
        y: Binary target (1=traded, 0=did not trade)
    """
    # Get wallet trades
    wallet_trades = trades[trades['wallet'] == wallet].copy()
    
    if len(wallet_trades) == 0:
        return None, None
    
    # Get markets this wallet traded in
    traded_markets = wallet_trades['market_id'].unique()
    
    # Filter market data to those markets
    market_subset = market_data[market_data['market_id'].isin(traded_markets)].copy()
    
    if len(market_subset) == 0:
        return None, None
    
    # Create trade indicator
    trade_seconds = set(zip(wallet_trades['market_id'], wallet_trades['t'].astype(int)))
    
    # Mark which seconds had trades
    market_subset['traded'] = market_subset.apply(
        lambda row: (row['market_id'], int(row['t'])) in trade_seconds,
        axis=1
    ).astype(int)
    
    # Get positive samples (traded)
    positives = market_subset[market_subset['traded'] == 1]
    
    # Get negative samples (not traded) - subsample for balance
    negatives = market_subset[market_subset['traded'] == 0]
    n_neg = min(len(negatives), max(subsample_negatives, len(positives) * 3))
    negatives = negatives.sample(n=n_neg, random_state=42) if len(negatives) > n_neg else negatives
    
    # Combine
    combined = pd.concat([positives, negatives], axis=0)
    
    # Prepare features (use available market data features)
    feature_cols = []
    for col in combined.columns:
        if combined[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            if col not in ['traded', 't', 'tau', 'K', 'settlement', 'Y']:
                if combined[col].isna().mean() < 0.5:
                    feature_cols.append(col)
    
    X = combined[feature_cols].fillna(0)
    y = combined['traded'].values
    
    return X, y


def extract_rules_from_tree(
    tree: 'DecisionTreeClassifier',
    feature_names: List[str],
    class_names: List[str] = ['no_trade', 'trade'],
    max_rules: int = 10
) -> List[Dict[str, Any]]:
    """Extract interpretable rules from decision tree."""
    rules = []
    
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]
    
    def recurse(node, conditions):
        if tree_.feature[node] != -2:  # Not a leaf
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Left branch: feature <= threshold
            left_conditions = conditions + [(name, '<=', threshold)]
            recurse(tree_.children_left[node], left_conditions)
            
            # Right branch: feature > threshold
            right_conditions = conditions + [(name, '>', threshold)]
            recurse(tree_.children_right[node], right_conditions)
        else:
            # Leaf node
            values = tree_.value[node][0]
            total = sum(values)
            if total > 0:
                prob_trade = values[1] / total if len(values) > 1 else 0
                n_samples = int(total)
                
                # Only keep rules that predict trade with decent probability
                if prob_trade > 0.3 and n_samples > 10:
                    rules.append({
                        'conditions': conditions,
                        'prob_trade': float(prob_trade),
                        'n_samples': n_samples,
                        'confidence': float(prob_trade),
                    })
    
    recurse(0, [])
    
    # Sort by confidence and samples
    rules.sort(key=lambda x: (x['prob_trade'], x['n_samples']), reverse=True)
    
    return rules[:max_rules]


def format_rule_as_string(rule: Dict) -> str:
    """Convert rule dictionary to human-readable string."""
    conditions = rule['conditions']
    if not conditions:
        return "ALWAYS"
    
    parts = []
    for name, op, thresh in conditions:
        # Clean up feature name
        clean_name = name.replace('feat_', '').replace('mkt_', '')
        
        # Format threshold
        if isinstance(thresh, float):
            if abs(thresh) < 0.01:
                thresh_str = f"{thresh:.4f}"
            elif abs(thresh) < 1:
                thresh_str = f"{thresh:.3f}"
            else:
                thresh_str = f"{thresh:.1f}"
        else:
            thresh_str = str(thresh)
        
        parts.append(f"{clean_name} {op} {thresh_str}")
    
    return " AND ".join(parts)


def extract_logistic_rules(
    model: 'LogisticRegression',
    feature_names: List[str],
    feature_means: Dict[str, float],
    feature_stds: Dict[str, float],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """Extract rules from logistic regression coefficients."""
    rules = []
    
    # Get coefficients
    coefs = model.coef_[0]
    
    # Pair with feature names
    feat_coefs = list(zip(feature_names, coefs))
    
    # Sort by absolute coefficient
    feat_coefs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for name, coef in feat_coefs[:top_k]:
        if abs(coef) < 0.01:  # Skip near-zero coefficients
            continue
        
        # Determine threshold based on coefficient sign
        mean = feature_means.get(name, 0)
        std = feature_stds.get(name, 1)
        
        if coef > 0:
            # Higher values predict trading
            threshold = mean + 0.5 * std
            direction = 'high'
            condition = (name, '>', threshold)
        else:
            # Lower values predict trading
            threshold = mean - 0.5 * std
            direction = 'low'
            condition = (name, '<', threshold)
        
        rules.append({
            'feature': name,
            'coefficient': float(coef),
            'direction': direction,
            'threshold': float(threshold),
            'condition': condition,
            'importance': float(abs(coef)),
        })
    
    return rules


def analyze_wallet_policy_simple(
    trades: pd.DataFrame,
    wallet: str
) -> Dict[str, Any]:
    """
    Simple rule extraction without sklearn.
    Uses threshold analysis on feature distributions.
    """
    wallet_trades = trades[trades['wallet'] == wallet]
    
    if len(wallet_trades) < 10:
        return {'wallet': wallet, 'rules': [], 'error': 'insufficient_data'}
    
    # Get feature columns
    feat_cols = get_feature_columns(wallet_trades)
    
    rules = []
    
    # Analyze each feature
    for col in feat_cols[:30]:  # Limit to top 30 features
        values = wallet_trades[col].dropna()
        if len(values) < 10:
            continue
        
        mean = values.mean()
        std = values.std()
        median = values.median()
        p25 = values.quantile(0.25)
        p75 = values.quantile(0.75)
        
        # Determine if feature is skewed toward high or low values
        # Compare to what we'd expect from uniform distribution
        
        # Check for concentration at extremes
        if not pd.isna(p75) and not pd.isna(p25):
            # High concentration
            if p25 > mean:  # Most trades when feature is high
                rules.append({
                    'feature': col,
                    'pattern': 'high_concentration',
                    'threshold': float(p25),
                    'condition': f"{col.replace('feat_', '')} > {p25:.3f}",
                    'pct_trades': float((values > p25).mean()),
                })
            elif p75 < mean:  # Most trades when feature is low
                rules.append({
                    'feature': col,
                    'pattern': 'low_concentration',
                    'threshold': float(p75),
                    'condition': f"{col.replace('feat_', '')} < {p75:.3f}",
                    'pct_trades': float((values < p75).mean()),
                })
    
    # Sort by discriminative power
    rules.sort(key=lambda x: x.get('pct_trades', 0), reverse=True)
    
    return {
        'wallet': wallet,
        'n_trades': len(wallet_trades),
        'rules': rules[:15],
    }


def analyze_wallet_policy_sklearn(
    trades: pd.DataFrame,
    market_data: pd.DataFrame,
    wallet: str
) -> Dict[str, Any]:
    """
    Full policy inversion using sklearn models.
    """
    # Prepare dataset
    X, y = prepare_trade_decision_dataset(trades, market_data, wallet)
    
    if X is None or len(X) < 100:
        return analyze_wallet_policy_simple(trades, wallet)
    
    feature_names = list(X.columns)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Store means and stds for rule extraction
    feature_means = dict(zip(feature_names, scaler.mean_))
    feature_stds = dict(zip(feature_names, scaler.scale_))
    
    results = {
        'wallet': wallet,
        'n_positive': int(y.sum()),
        'n_negative': int(len(y) - y.sum()),
        'models': {},
        'rules': [],
    }
    
    # --------------------------------------------------------------------------
    # 1. Logistic Regression
    # --------------------------------------------------------------------------
    try:
        lr = LogisticRegression(
            penalty='l1',
            solver='saga',
            max_iter=1000,
            C=0.1,  # Regularization
            random_state=42
        )
        lr.fit(X_scaled, y)
        
        # Cross-validation score
        cv_scores = cross_val_score(lr, X_scaled, y, cv=3, scoring='roc_auc')
        
        # Extract rules
        lr_rules = extract_logistic_rules(
            lr, feature_names, feature_means, feature_stds
        )
        
        results['models']['logistic_regression'] = {
            'cv_auc_mean': float(cv_scores.mean()),
            'cv_auc_std': float(cv_scores.std()),
            'n_nonzero_coefs': int((lr.coef_[0] != 0).sum()),
            'top_features': lr_rules[:5],
        }
        
        # Add to rules
        for rule in lr_rules[:5]:
            results['rules'].append({
                'source': 'logistic_regression',
                'feature': rule['feature'],
                'direction': rule['direction'],
                'importance': rule['importance'],
                'rule_string': f"IF {rule['feature'].replace('feat_', '')} is {rule['direction']} THEN more likely to trade",
            })
    except Exception as e:
        results['models']['logistic_regression'] = {'error': str(e)}
    
    # --------------------------------------------------------------------------
    # 2. Decision Tree
    # --------------------------------------------------------------------------
    try:
        dt = DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=50,
            random_state=42
        )
        dt.fit(X_scaled, y)
        
        # Cross-validation score
        cv_scores = cross_val_score(dt, X_scaled, y, cv=3, scoring='roc_auc')
        
        # Extract rules
        dt_rules = extract_rules_from_tree(dt, feature_names)
        
        results['models']['decision_tree'] = {
            'cv_auc_mean': float(cv_scores.mean()),
            'cv_auc_std': float(cv_scores.std()),
            'tree_depth': int(dt.get_depth()),
            'n_leaves': int(dt.get_n_leaves()),
            'extracted_rules': [
                {
                    'rule_string': format_rule_as_string(r),
                    'prob_trade': r['prob_trade'],
                    'n_samples': r['n_samples'],
                }
                for r in dt_rules[:5]
            ],
        }
        
        # Feature importances
        importances = list(zip(feature_names, dt.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)
        results['models']['decision_tree']['feature_importances'] = [
            {'feature': f, 'importance': float(i)}
            for f, i in importances[:10]
        ]
        
        # Add to rules
        for rule in dt_rules[:3]:
            rule_str = format_rule_as_string(rule)
            results['rules'].append({
                'source': 'decision_tree',
                'rule_string': f"IF {rule_str} THEN trade (prob={rule['prob_trade']:.2f}, n={rule['n_samples']})",
                'confidence': rule['prob_trade'],
                'support': rule['n_samples'],
            })
    except Exception as e:
        results['models']['decision_tree'] = {'error': str(e)}
    
    return results


def analyze_direction_policy(
    trades: pd.DataFrame,
    wallet: str
) -> Dict[str, Any]:
    """
    Analyze what predicts direction choice (UP vs DOWN).
    """
    wallet_trades = trades[trades['wallet'] == wallet]
    
    # Filter to directional trades (not BOTH)
    directional = wallet_trades[wallet_trades['direction'].isin(['Up', 'Down'])]
    
    if len(directional) < 20:
        return {'wallet': wallet, 'direction_rules': [], 'error': 'insufficient_directional_trades'}
    
    # Get feature columns
    feat_cols = get_feature_columns(directional)
    
    rules = []
    
    # Analyze features that differ between UP and DOWN trades
    up_trades = directional[directional['direction'] == 'Up']
    down_trades = directional[directional['direction'] == 'Down']
    
    for col in feat_cols[:20]:
        up_mean = up_trades[col].mean()
        down_mean = down_trades[col].mean()
        
        if pd.isna(up_mean) or pd.isna(down_mean):
            continue
        
        diff = up_mean - down_mean
        pooled_std = directional[col].std()
        
        if pooled_std > 0:
            effect_size = abs(diff) / pooled_std
            
            if effect_size > 0.3:  # Moderate effect
                direction_chosen = 'UP' if diff > 0 else 'DOWN'
                rules.append({
                    'feature': col,
                    'effect_size': float(effect_size),
                    'up_mean': float(up_mean),
                    'down_mean': float(down_mean),
                    'rule': f"Higher {col.replace('feat_', '')} -> {direction_chosen}",
                })
    
    # Sort by effect size
    rules.sort(key=lambda x: x['effect_size'], reverse=True)
    
    return {
        'wallet': wallet,
        'n_up': len(up_trades),
        'n_down': len(down_trades),
        'direction_rules': rules[:10],
    }


def identify_cross_wallet_patterns(
    all_results: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Identify patterns that appear across multiple wallets.
    """
    print("\nIdentifying cross-wallet patterns...")
    
    # Collect all feature importances
    feature_wallets = defaultdict(list)
    
    for wallet, result in all_results.items():
        if 'models' in result:
            # From decision tree
            dt = result['models'].get('decision_tree', {})
            importances = dt.get('feature_importances', [])
            for fi in importances:
                feature_wallets[fi['feature']].append({
                    'wallet': wallet,
                    'importance': fi['importance'],
                    'source': 'decision_tree',
                })
            
            # From logistic regression
            lr = result['models'].get('logistic_regression', {})
            top_feats = lr.get('top_features', [])
            for tf in top_feats:
                feature_wallets[tf['feature']].append({
                    'wallet': wallet,
                    'importance': tf['importance'],
                    'source': 'logistic_regression',
                })
    
    # Find features important across multiple wallets
    cross_wallet_features = []
    for feature, wallet_data in feature_wallets.items():
        n_wallets = len(set(w['wallet'] for w in wallet_data))
        avg_importance = np.mean([w['importance'] for w in wallet_data])
        
        if n_wallets >= 2:  # At least 2 wallets
            cross_wallet_features.append({
                'feature': feature,
                'n_wallets': n_wallets,
                'wallets': list(set(w['wallet'] for w in wallet_data)),
                'avg_importance': float(avg_importance),
            })
    
    cross_wallet_features.sort(key=lambda x: (x['n_wallets'], x['avg_importance']), reverse=True)
    
    return {
        'universal_features': cross_wallet_features[:15],
        'total_features_analyzed': len(feature_wallets),
    }


def main():
    print("=" * 70)
    print("Phase 5: Policy Inversion (State -> Action Modeling)")
    print("=" * 70)
    
    # Step 1: Load feature matrix
    trades = load_feature_matrix()
    
    # Step 2: Load market data for negative sampling (if sklearn available)
    if SKLEARN_AVAILABLE:
        market_data = load_market_data()
    else:
        market_data = None
    
    # Step 3: Analyze each wallet's policy
    print("\n" + "=" * 70)
    print("Analyzing wallet policies...")
    print("=" * 70)
    
    wallets = trades['wallet'].unique()
    all_results = {}
    direction_results = {}
    
    for wallet in wallets:
        print(f"\n--- Analyzing: {wallet} ---")
        
        # Trade decision policy
        if SKLEARN_AVAILABLE and market_data is not None:
            result = analyze_wallet_policy_sklearn(trades, market_data, wallet)
        else:
            result = analyze_wallet_policy_simple(trades, wallet)
        
        all_results[wallet] = result
        
        # Direction policy
        dir_result = analyze_direction_policy(trades, wallet)
        direction_results[wallet] = dir_result
        
        # Print summary
        print(f"  Trades analyzed: {result.get('n_trades', result.get('n_positive', 0))}")
        if 'models' in result:
            lr_auc = result['models'].get('logistic_regression', {}).get('cv_auc_mean')
            dt_auc = result['models'].get('decision_tree', {}).get('cv_auc_mean')
            if lr_auc:
                print(f"  Logistic Regression AUC: {lr_auc:.3f}")
            if dt_auc:
                print(f"  Decision Tree AUC: {dt_auc:.3f}")
        
        n_rules = len(result.get('rules', []))
        print(f"  Rules extracted: {n_rules}")
    
    # Step 4: Identify cross-wallet patterns
    cross_wallet = identify_cross_wallet_patterns(all_results)
    
    # Step 5: Compile final rules
    print("\n" + "=" * 70)
    print("SUMMARY: Universal Features (appear in 2+ wallets)")
    print("=" * 70)
    
    for feat in cross_wallet['universal_features'][:10]:
        print(f"  {feat['feature']}: {feat['n_wallets']} wallets, avg importance {feat['avg_importance']:.3f}")
    
    # Step 6: Save outputs
    print("\n" + "=" * 70)
    print("Saving outputs...")
    
    # Compile all rules
    compiled_rules = {
        'per_wallet': all_results,
        'direction_analysis': direction_results,
        'cross_wallet_patterns': cross_wallet,
    }
    
    # Save policy rules
    rules_path = RESULTS_DIR / "policy_rules.json"
    with open(rules_path, 'w') as f:
        json.dump(compiled_rules, f, indent=2, default=str)
    print(f"  Policy rules saved to: {rules_path}")
    
    # Save model summary
    summary = {
        'wallets_analyzed': len(wallets),
        'sklearn_available': SKLEARN_AVAILABLE,
        'universal_features': cross_wallet['universal_features'][:10],
        'per_wallet_summary': {
            wallet: {
                'n_rules': len(result.get('rules', [])),
                'best_auc': max(
                    result.get('models', {}).get('logistic_regression', {}).get('cv_auc_mean', 0),
                    result.get('models', {}).get('decision_tree', {}).get('cv_auc_mean', 0)
                ) if 'models' in result else None,
            }
            for wallet, result in all_results.items()
        }
    }
    
    summary_path = RESULTS_DIR / "policy_models_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Policy models summary saved to: {summary_path}")
    
    print("\n" + "=" * 70)
    print("DONE - Phase 5 Complete")
    print("=" * 70)
    
    return compiled_rules


if __name__ == "__main__":
    main()

