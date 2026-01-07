#!/usr/bin/env python3
"""
Bayesian Analysis for Small Sample Sizes

With only 12 markets, frequentist t-tests are noisy. This module provides
Bayesian inference to answer:
1. P(mean > 0): Probability of positive edge
2. P(mean > $0.50): Probability of economically meaningful edge
3. Credible intervals

Uses analytical solution (conjugate prior) since normal assumption is reasonable.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class BayesianResults:
    """Results from Bayesian analysis."""
    # Data summary
    n_markets: int
    sample_mean: float
    sample_std: float
    
    # Posterior parameters (assuming unknown variance, t-distribution)
    posterior_mean: float
    posterior_std: float
    posterior_df: int  # Degrees of freedom
    
    # Key probabilities
    prob_positive: float  # P(mean > 0)
    prob_economically_meaningful: float  # P(mean > threshold)
    threshold: float  # Economic threshold used
    
    # Credible intervals
    ci_90: Tuple[float, float]  # 90% credible interval
    ci_95: Tuple[float, float]  # 95% credible interval
    
    # Comparison to frequentist
    frequentist_t_stat: float
    frequentist_p_value: float


def bayesian_inference(
    pnls: List[float],
    economic_threshold: float = 0.50,
    prior_mean: float = 0.0,
    prior_strength: float = 0.0,  # 0 = non-informative prior
) -> BayesianResults:
    """
    Perform Bayesian inference on per-market PnL values.
    
    Uses a weakly informative prior and computes posterior distribution
    for the mean PnL.
    
    With unknown variance and normal likelihood, the posterior for the mean
    follows a t-distribution.
    
    Args:
        pnls: List of per-market PnL values
        economic_threshold: Threshold for economic significance
        prior_mean: Prior mean (default: 0)
        prior_strength: Prior sample size (0 = non-informative)
        
    Returns:
        BayesianResults with all computed values
    """
    n = len(pnls)
    if n == 0:
        return BayesianResults(
            n_markets=0, sample_mean=0, sample_std=0,
            posterior_mean=0, posterior_std=0, posterior_df=0,
            prob_positive=0.5, prob_economically_meaningful=0,
            threshold=economic_threshold,
            ci_90=(0, 0), ci_95=(0, 0),
            frequentist_t_stat=0, frequentist_p_value=1.0,
        )
    
    # Sample statistics
    sample_mean = np.mean(pnls)
    sample_std = np.std(pnls, ddof=1) if n > 1 else 0.0
    sample_var = sample_std ** 2
    
    # Posterior with non-informative prior (Jeffrey's prior)
    # Posterior mean is approximately sample mean
    # Posterior follows t-distribution with df = n-1
    posterior_mean = sample_mean
    posterior_df = n - 1
    
    # Posterior standard error (standard error of the mean)
    posterior_std = sample_std / np.sqrt(n) if n > 0 else 0.0
    
    # Compute probabilities using t-distribution
    if posterior_std > 0 and posterior_df > 0:
        t_dist = stats.t(df=posterior_df, loc=posterior_mean, scale=posterior_std)
        
        # P(mean > 0)
        prob_positive = 1 - t_dist.cdf(0)
        
        # P(mean > threshold)
        prob_economically_meaningful = 1 - t_dist.cdf(economic_threshold)
        
        # Credible intervals
        ci_90 = (t_dist.ppf(0.05), t_dist.ppf(0.95))
        ci_95 = (t_dist.ppf(0.025), t_dist.ppf(0.975))
    else:
        prob_positive = 0.5
        prob_economically_meaningful = 0.0
        ci_90 = (sample_mean, sample_mean)
        ci_95 = (sample_mean, sample_mean)
    
    # Frequentist comparison
    if posterior_std > 0:
        frequentist_t_stat = sample_mean / posterior_std
        frequentist_p_value = 2 * (1 - stats.t.cdf(abs(frequentist_t_stat), df=posterior_df))
    else:
        frequentist_t_stat = 0
        frequentist_p_value = 1.0
    
    return BayesianResults(
        n_markets=n,
        sample_mean=sample_mean,
        sample_std=sample_std,
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        posterior_df=posterior_df,
        prob_positive=prob_positive,
        prob_economically_meaningful=prob_economically_meaningful,
        threshold=economic_threshold,
        ci_90=ci_90,
        ci_95=ci_95,
        frequentist_t_stat=frequentist_t_stat,
        frequentist_p_value=frequentist_p_value,
    )


def print_bayesian_report(results: BayesianResults):
    """Print a Bayesian analysis report."""
    print("\n" + "="*70)
    print("BAYESIAN ANALYSIS REPORT")
    print("="*70)
    
    print("\n1. DATA SUMMARY")
    print("-"*50)
    print(f"  Number of markets: {results.n_markets}")
    print(f"  Sample mean: ${results.sample_mean:.4f}")
    print(f"  Sample std: ${results.sample_std:.4f}")
    
    print("\n2. POSTERIOR DISTRIBUTION")
    print("-"*50)
    print(f"  Posterior mean: ${results.posterior_mean:.4f}")
    print(f"  Posterior std: ${results.posterior_std:.4f}")
    print(f"  Degrees of freedom: {results.posterior_df}")
    
    print("\n3. KEY PROBABILITIES")
    print("-"*50)
    print(f"  P(mean > $0.00): {results.prob_positive*100:.1f}%")
    print(f"  P(mean > ${results.threshold:.2f}): {results.prob_economically_meaningful*100:.1f}%")
    
    # Interpretation
    if results.prob_positive >= 0.95:
        print("\n  -> Strong evidence of positive edge (>95% probability)")
    elif results.prob_positive >= 0.80:
        print("\n  -> Moderate evidence of positive edge (80-95% probability)")
    elif results.prob_positive >= 0.50:
        print("\n  -> Weak/inconclusive evidence (50-80% probability)")
    else:
        print("\n  -> Evidence suggests NEGATIVE edge (<50% probability)")
    
    print("\n4. CREDIBLE INTERVALS")
    print("-"*50)
    print(f"  90% CI: [${results.ci_90[0]:.4f}, ${results.ci_90[1]:.4f}]")
    print(f"  95% CI: [${results.ci_95[0]:.4f}, ${results.ci_95[1]:.4f}]")
    
    # Check if CI excludes zero
    if results.ci_95[0] > 0:
        print("  -> 95% CI excludes zero: strong evidence of positive edge")
    elif results.ci_90[0] > 0:
        print("  -> 90% CI excludes zero: moderate evidence of positive edge")
    else:
        print("  -> CI includes zero: inconclusive")
    
    print("\n5. FREQUENTIST COMPARISON")
    print("-"*50)
    print(f"  t-statistic: {results.frequentist_t_stat:.2f}")
    print(f"  p-value (two-sided): {results.frequentist_p_value:.4f}")
    
    # Compare interpretations
    print("\n  Comparison:")
    if results.frequentist_p_value < 0.05:
        freq_result = "SIGNIFICANT (p<0.05)"
    elif results.frequentist_p_value < 0.10:
        freq_result = "MARGINAL (p<0.10)"
    else:
        freq_result = "NOT SIGNIFICANT (p>=0.10)"
    
    if results.prob_positive >= 0.95:
        bayes_result = "STRONG EVIDENCE (P>95%)"
    elif results.prob_positive >= 0.80:
        bayes_result = "MODERATE EVIDENCE (P>80%)"
    else:
        bayes_result = "WEAK/NO EVIDENCE (P<80%)"
    
    print(f"  Frequentist: {freq_result}")
    print(f"  Bayesian: {bayes_result}")
    
    print("\n" + "="*70)


def run_bayesian_analysis(
    df: pd.DataFrame,
    strategy: Any = None,
    economic_threshold: float = 0.50,
    volume_markets_only: bool = True,
    verbose: bool = True,
) -> BayesianResults:
    """
    Run Bayesian analysis on strategy results.
    
    Args:
        df: Market data
        strategy: Strategy to test (or default)
        economic_threshold: Threshold for economic significance
        volume_markets_only: Only use volume markets
        verbose: Print report
        
    Returns:
        BayesianResults
    """
    from scripts.backtest.strategies import SpreadCaptureStrategy
    from scripts.backtest.backtest_engine import run_maker_backtest
    from scripts.backtest.maker_execution import MakerExecutionConfig, FillModel
    
    if strategy is None:
        strategy = SpreadCaptureStrategy(
            spread_min=0.01,
            tau_min=60,
            tau_max=600,
            inventory_limit_up=10.0,
            inventory_limit_down=10.0,
        )
    
    config = MakerExecutionConfig(
        place_latency_ms=100,
        cancel_latency_ms=50,
        fill_model=FillModel.TOUCH_SIZE_PROXY,
        touch_trade_rate_per_second=0.03,
    )
    
    # Run backtest
    result = run_maker_backtest(df, strategy, config,
                                volume_markets_only=volume_markets_only,
                                verbose=verbose)
    
    # Extract per-market PnL
    market_pnls = result.get('metrics', {}).get('market_pnls', {})
    if not market_pnls:
        # Try to get from market_results
        market_results = result.get('market_results', {})
        market_pnls = {mid: r.get('pnl', 0) for mid, r in market_results.items()}
    
    pnls = list(market_pnls.values())
    
    # Run Bayesian inference
    results = bayesian_inference(pnls, economic_threshold=economic_threshold)
    
    if verbose:
        print_bayesian_report(results)
    
    return results


def main():
    """Run Bayesian analysis."""
    from scripts.backtest.data_loader import load_eth_markets, add_derived_columns
    
    print("Loading data...")
    df, _ = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    print(f"Loaded {len(df):,} rows, {df['market_id'].nunique()} markets")
    
    # Run analysis
    results = run_bayesian_analysis(df, volume_markets_only=True)


if __name__ == '__main__':
    main()

