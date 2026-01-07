"""
Fair Value Baseline Model

Purpose: Separate latency capture from directional momentum.

A pure "trade after CL event" strategy can profit for two reasons:
1. PM lags CL (latency capture - our hypothesis)
2. Price has short-term momentum (directional alpha)

This module builds a baseline fair value model to distinguish between them.
If our strategy's edge comes purely from momentum, it would show up in
the fair value model. If it comes from latency, the edge should be
orthogonal to fair value.

Key components:
- FairValueModel: Logistic regression based model
- BinnedFairValueModel: Empirical lookup table with smoothing (recommended)
- compute_realized_volatility: Backward-looking volatility calculation
"""

from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# REALIZED VOLATILITY CALCULATION
# ==============================================================================

def compute_realized_volatility(
    df: pd.DataFrame,
    window_size: int = 30,
    min_periods: int = 5,
    use_observed_only: bool = True
) -> pd.Series:
    """
    Compute backward-looking realized volatility of CL returns.
    
    Formula:
        r_t = 10000 * (CL_t / CL_{t-1} - 1)  [returns in bps]
        σ(t) = stdev(r_{t-W+1}, ..., r_t)    [rolling std]
    
    Args:
        df: DataFrame with 'cl_mid', 'market_id', and optionally 'cl_ffill' columns
        window_size: Number of seconds for rolling window (default 30s)
        min_periods: Minimum observations required (default 5)
        use_observed_only: If True, only use cl_ffill==0 rows for computation
        
    Returns:
        Series of realized volatility values (bps)
        
    Note:
        - First W-1 rows per market will have NaN
        - Uses backward-looking window only (no future data)
    """
    df = df.copy()
    
    # Compute 1-second returns in bps per market
    df['_cl_return_bps'] = df.groupby('market_id')['cl_mid'].pct_change() * 10000
    
    # If using observed data only, mask forward-filled returns
    if use_observed_only and 'cl_ffill' in df.columns:
        # Set returns to NaN where cl_ffill == 1 (forward-filled data)
        df.loc[df['cl_ffill'] == 1, '_cl_return_bps'] = np.nan
    
    # Compute rolling std (backward-looking only)
    realized_vol = df.groupby('market_id')['_cl_return_bps'].transform(
        lambda x: x.rolling(window=window_size, min_periods=min_periods).std()
    )
    
    return realized_vol


def add_realized_volatility_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add multiple realized volatility columns for different window sizes.
    
    Args:
        df: DataFrame with required columns
        
    Returns:
        DataFrame with added columns:
        - realized_vol_bps: 30s window (default)
        - realized_vol_15s: 15s window
        - realized_vol_60s: 60s window
    """
    df = df.copy()
    
    # Standard 30s window
    df['realized_vol_bps'] = compute_realized_volatility(df, window_size=30)
    
    # Alternative windows for sensitivity analysis
    df['realized_vol_15s'] = compute_realized_volatility(df, window_size=15)
    df['realized_vol_60s'] = compute_realized_volatility(df, window_size=60)
    
    return df


class FairValueModel:
    """
    Empirical fair value model for PM probabilities.
    
    Predicts P(Y=1) = P(settlement > K) using features:
    - delta_bps: Distance from strike in bps
    - tau: Time to expiry
    - cl_vol: Recent CL volatility
    - delta_tau_interaction: delta_bps * tau
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = ['delta_bps', 'tau', 'cl_vol', 'delta_tau']
        self.fitted = False
    
    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build feature matrix from DataFrame."""
        features = pd.DataFrame()
        
        features['delta_bps'] = df['delta_bps']
        features['tau'] = df['tau']
        
        # Recent CL volatility (30s rolling std of returns in bps)
        if 'cl_vol_30s' in df.columns:
            features['cl_vol'] = df['cl_vol_30s']
        else:
            features['cl_vol'] = df.groupby('market_id')['cl_mid'].transform(
                lambda x: x.pct_change().rolling(30, min_periods=5).std() * 10000
            )
        
        # Interaction term
        features['delta_tau'] = features['delta_bps'] * features['tau'] / 900  # Normalize
        
        # Fill NaN with 0 for volatility
        features = features.fillna(0)
        
        return features.values
    
    def fit(self, df: pd.DataFrame) -> 'FairValueModel':
        """
        Fit model on training markets.
        
        Args:
            df: Training DataFrame (must have market outcomes)
            
        Returns:
            self
        """
        # Get one row per second with the outcome
        df = df.copy()
        
        # Get market outcomes
        market_outcomes = df.groupby('market_id').first()[['Y']].reset_index()
        df = df.merge(market_outcomes, on='market_id', suffixes=('', '_outcome'))
        
        # Filter out rows with NaN in Y
        y_col = 'Y_outcome' if 'Y_outcome' in df.columns else 'Y'
        df = df.dropna(subset=[y_col])
        
        if len(df) == 0:
            raise ValueError("No valid rows with outcome labels after filtering NaN")
        
        # Build features
        X = self._build_features(df)
        y = df[y_col].values
        
        # Filter out rows with NaN in features
        valid_mask = ~np.isnan(X).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            raise ValueError("No valid rows after filtering NaN features")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit logistic regression
        self.model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        
        self.fitted = True
        
        # Store coefficients for interpretation
        self.coefficients = dict(zip(self.feature_names, self.model.coef_[0]))
        self.intercept = self.model.intercept_[0]
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict fair probability for each row.
        
        Args:
            df: DataFrame with required columns
            
        Returns:
            Array of P(Y=1) predictions
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = self._build_features(df)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary for documentation."""
        if not self.fitted:
            return {'error': 'Model not fitted'}
        
        return {
            'feature_names': self.feature_names,
            'coefficients': self.coefficients,
            'intercept': self.intercept,
            'interpretation': {
                'delta_bps': 'Higher = more likely UP wins (above strike)',
                'tau': 'More time = more uncertainty',
                'cl_vol': 'Higher vol = more uncertainty',
                'delta_tau': 'Interaction: far from strike matters more early',
            }
        }


# ==============================================================================
# BINNED FAIR VALUE MODEL (Option A - Recommended)
# ==============================================================================

class BinnedFairValueModel:
    """
    Empirical fair value model using binned lookup table.
    
    This approach is more robust than parametric models for small datasets.
    It directly estimates P(Y=1) for each (delta_bps, tau, realized_vol) bin.
    
    Binning scheme:
    - tau: bins of bin_tau_size seconds (default 30s)
    - delta_bps: bins of bin_delta_size bps (default 5bps)
    - realized_vol: terciles (low/med/high)
    
    Market-weighting:
    - Each row weighted by 1/900 to avoid pseudo-replication
    - Or sample every sample_every seconds (default 5)
    """
    
    def __init__(
        self,
        bin_tau_size: int = 30,
        bin_delta_size: float = 5.0,
        n_vol_bins: int = 3,
        min_samples_per_bin: int = 10,
        sample_every: int = 5,
        delta_range: Tuple[float, float] = (-500, 500),
    ):
        """
        Initialize binned model.
        
        Args:
            bin_tau_size: Size of tau bins in seconds (default 30)
            bin_delta_size: Size of delta bins in bps (default 5)
            n_vol_bins: Number of volatility bins (default 3 = terciles)
            min_samples_per_bin: Minimum samples for valid bin (default 10)
            sample_every: Sample every N seconds per market (default 5)
            delta_range: Range of delta_bps to consider (default -500 to 500)
        """
        self.bin_tau_size = bin_tau_size
        self.bin_delta_size = bin_delta_size
        self.n_vol_bins = n_vol_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.sample_every = sample_every
        self.delta_range = delta_range
        
        # Will be populated during fit
        self.lookup_table = None
        self.vol_bin_edges = None
        self.global_p = None
        self.fitted = False
        self.bin_stats = {}
    
    def _get_tau_bin(self, tau: float) -> int:
        """Get tau bin index (0 = last 30s of market)."""
        return int(tau // self.bin_tau_size)
    
    def _get_delta_bin(self, delta_bps: float) -> int:
        """Get delta bin index."""
        # Handle NaN
        if pd.isna(delta_bps):
            delta_bps = 0  # Default to at-strike
        # Clip to range
        delta_clipped = np.clip(delta_bps, self.delta_range[0], self.delta_range[1])
        # Shift so 0 is at center
        shifted = delta_clipped - self.delta_range[0]
        return int(shifted // self.bin_delta_size)
    
    def _get_vol_bin(self, vol: float) -> int:
        """Get volatility bin index based on fitted edges."""
        if self.vol_bin_edges is None:
            return 1  # Default to middle bin
        
        for i, edge in enumerate(self.vol_bin_edges[1:]):
            if vol <= edge:
                return i
        return self.n_vol_bins - 1
    
    def fit(self, df: pd.DataFrame) -> 'BinnedFairValueModel':
        """
        Fit binned model on training data.
        
        Args:
            df: Training DataFrame with columns:
                - market_id, t, tau, delta_bps, Y
                - realized_vol_bps (or cl_vol_30s)
                
        Returns:
            self
        """
        df = df.copy()
        
        # Get volatility column
        vol_col = 'realized_vol_bps' if 'realized_vol_bps' in df.columns else 'cl_vol_30s'
        if vol_col not in df.columns:
            df[vol_col] = compute_realized_volatility(df, window_size=30)
        
        # Sample every N seconds to reduce pseudo-replication
        if self.sample_every > 1:
            df = df[df['t'] % self.sample_every == 0].copy()
        
        # Get market outcomes (Y is same for all rows in a market)
        market_outcomes = df.groupby('market_id')['Y'].first().to_dict()
        df['Y_label'] = df['market_id'].map(market_outcomes)
        
        # Filter to observed data only (not forward-filled)
        if 'cl_ffill' in df.columns and 'pm_ffill' in df.columns:
            df = df[(df['cl_ffill'] == 0) & (df['pm_ffill'] == 0)]
        
        # Remove rows with NaN in key columns
        df = df.dropna(subset=['delta_bps', 'tau', vol_col, 'Y_label'])
        
        # Compute volatility bin edges (terciles)
        vol_values = df[vol_col].dropna()
        if len(vol_values) > 0:
            self.vol_bin_edges = np.percentile(
                vol_values, 
                np.linspace(0, 100, self.n_vol_bins + 1)
            )
        else:
            self.vol_bin_edges = np.array([0, 1, 2, 100])  # Fallback
        
        # Global probability (fallback for sparse bins)
        self.global_p = df['Y_label'].mean()
        
        # Assign bins
        df['tau_bin'] = df['tau'].apply(self._get_tau_bin)
        df['delta_bin'] = df['delta_bps'].apply(self._get_delta_bin)
        df['vol_bin'] = df[vol_col].apply(self._get_vol_bin)
        
        # Build lookup table
        # Key: (tau_bin, delta_bin, vol_bin)
        # Value: (p_hat, n_samples)
        self.lookup_table = {}
        
        grouped = df.groupby(['tau_bin', 'delta_bin', 'vol_bin'])
        
        for (tau_b, delta_b, vol_b), group in grouped:
            n_samples = len(group)
            p_hat = group['Y_label'].mean()
            
            self.lookup_table[(tau_b, delta_b, vol_b)] = {
                'p_hat': p_hat,
                'n_samples': n_samples,
                'is_valid': n_samples >= self.min_samples_per_bin
            }
        
        # Apply smoothing for sparse bins
        self._smooth_sparse_bins()
        
        # Store statistics
        self.bin_stats = {
            'n_bins': len(self.lookup_table),
            'n_valid_bins': sum(1 for v in self.lookup_table.values() if v['is_valid']),
            'global_p': self.global_p,
            'vol_bin_edges': self.vol_bin_edges.tolist(),
            'n_samples_total': len(df),
        }
        
        self.fitted = True
        return self
    
    def _smooth_sparse_bins(self):
        """Apply nearest-neighbor smoothing to sparse bins."""
        if self.lookup_table is None:
            return
        
        sparse_bins = [k for k, v in self.lookup_table.items() if not v['is_valid']]
        
        for key in sparse_bins:
            tau_b, delta_b, vol_b = key
            
            # Find neighboring bins
            neighbors = []
            for dt in [-1, 0, 1]:
                for dd in [-1, 0, 1]:
                    for dv in [-1, 0, 1]:
                        if dt == 0 and dd == 0 and dv == 0:
                            continue
                        neighbor_key = (tau_b + dt, delta_b + dd, vol_b + dv)
                        if neighbor_key in self.lookup_table:
                            neighbor = self.lookup_table[neighbor_key]
                            if neighbor['is_valid']:
                                neighbors.append(neighbor)
            
            # Average valid neighbors
            if neighbors:
                avg_p = np.mean([n['p_hat'] for n in neighbors])
                self.lookup_table[key]['p_hat'] = avg_p
                self.lookup_table[key]['smoothed'] = True
            else:
                # Fall back to global
                self.lookup_table[key]['p_hat'] = self.global_p
                self.lookup_table[key]['smoothed'] = True
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict fair probability for each row.
        
        Args:
            df: DataFrame with delta_bps, tau, and volatility columns
            
        Returns:
            Array of P(Y=1) predictions
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        df = df.copy()
        
        # Get volatility column
        vol_col = 'realized_vol_bps' if 'realized_vol_bps' in df.columns else 'cl_vol_30s'
        if vol_col not in df.columns:
            df[vol_col] = compute_realized_volatility(df, window_size=30)
        
        predictions = []
        
        for _, row in df.iterrows():
            # Handle NaN values
            if pd.isna(row['tau']) or pd.isna(row['delta_bps']):
                predictions.append(self.global_p)
                continue
            
            tau_b = self._get_tau_bin(row['tau'])
            delta_b = self._get_delta_bin(row['delta_bps'])
            vol = row[vol_col] if not pd.isna(row[vol_col]) else 0
            vol_b = self._get_vol_bin(vol)
            
            key = (tau_b, delta_b, vol_b)
            
            if key in self.lookup_table:
                predictions.append(self.lookup_table[key]['p_hat'])
            else:
                # Fall back to global probability
                predictions.append(self.global_p)
        
        return np.array(predictions)
    
    def predict_fast(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized prediction (faster than row-by-row).
        
        Args:
            df: DataFrame with delta_bps, tau, and volatility columns
            
        Returns:
            Array of P(Y=1) predictions
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get volatility column
        vol_col = 'realized_vol_bps' if 'realized_vol_bps' in df.columns else 'cl_vol_30s'
        
        # Handle NaN values - fill with defaults
        df_clean = df.copy()
        df_clean['tau'] = df_clean['tau'].fillna(450)  # Middle of range
        df_clean['delta_bps'] = df_clean['delta_bps'].fillna(0)  # At strike
        
        # Compute bins
        tau_bins = (df_clean['tau'] // self.bin_tau_size).astype(int)
        
        delta_clipped = np.clip(df_clean['delta_bps'], self.delta_range[0], self.delta_range[1])
        delta_shifted = delta_clipped - self.delta_range[0]
        delta_bins = (delta_shifted // self.bin_delta_size).astype(int)
        
        # Vol bins need special handling
        vol_values = df_clean[vol_col].fillna(0).values if vol_col in df_clean.columns else np.zeros(len(df_clean))
        vol_bins = np.digitize(vol_values, self.vol_bin_edges[1:-1]) if len(self.vol_bin_edges) > 2 else np.zeros(len(df_clean), dtype=int)
        
        # Lookup predictions
        predictions = np.full(len(df), self.global_p)
        
        # Mark rows with original NaN
        has_nan = df['tau'].isna() | df['delta_bps'].isna()
        
        for i in range(len(df)):
            if has_nan.iloc[i] if hasattr(has_nan, 'iloc') else has_nan[i]:
                predictions[i] = self.global_p
                continue
                
            key = (int(tau_bins.iloc[i]), int(delta_bins.iloc[i]), int(vol_bins[i]))
            if key in self.lookup_table:
                predictions[i] = self.lookup_table[key]['p_hat']
        
        return predictions
    
    def get_bin_stats(self) -> Dict[str, Any]:
        """Get binning statistics for validation."""
        return self.bin_stats
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary for documentation."""
        if not self.fitted:
            return {'error': 'Model not fitted'}
        
        return {
            'model_type': 'BinnedFairValueModel',
            'bin_tau_size': self.bin_tau_size,
            'bin_delta_size': self.bin_delta_size,
            'n_vol_bins': self.n_vol_bins,
            'delta_range': self.delta_range,
            'bin_stats': self.bin_stats,
            'interpretation': {
                'p_hat': 'Fair probability P(Y=1) estimated from training data',
                'binning': 'Empirical lookup with nearest-neighbor smoothing',
                'market_weighting': f'Sampled every {self.sample_every} seconds',
            }
        }


# ==============================================================================
# CALIBRATION METRICS
# ==============================================================================

def compute_brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Brier score (lower is better).
    
    Brier score = mean((y_pred - y_true)^2)
    
    Args:
        y_true: True binary outcomes (0 or 1)
        y_pred: Predicted probabilities
        
    Returns:
        Brier score (0 = perfect, 1 = worst), or np.nan if no valid data
    """
    # Filter out NaN values
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() == 0:
        return np.nan
    return np.mean((y_pred[valid] - y_true[valid]) ** 2)


def compute_expected_calibration_error(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures how well calibrated predictions are:
    - Bin predictions by predicted probability
    - Compute |avg_predicted - avg_actual| per bin
    - Weight by bin size
    
    Args:
        y_true: True binary outcomes
        y_pred: Predicted probabilities
        n_bins: Number of calibration bins
        
    Returns:
        ECE (lower is better, 0 = perfectly calibrated), or np.nan if no valid data
    """
    # Filter out NaN values
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() == 0:
        return np.nan
    
    y_true_clean = y_true[valid]
    y_pred_clean = y_pred[valid]
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true_clean)
    
    if total_samples == 0:
        return np.nan
    
    for i in range(n_bins):
        mask = (y_pred_clean >= bin_edges[i]) & (y_pred_clean < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_size = mask.sum()
            avg_pred = y_pred_clean[mask].mean()
            avg_true = y_true_clean[mask].mean()
            ece += (bin_size / total_samples) * abs(avg_pred - avg_true)
    
    return ece


def compute_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve data.
    
    Args:
        y_true: True binary outcomes
        y_pred: Predicted probabilities
        n_bins: Number of calibration bins
        
    Returns:
        Tuple of (bin_centers, actual_rates, bin_counts)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    actual_rates = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            actual_rates.append(y_true[mask].mean())
            bin_counts.append(mask.sum())
    
    return np.array(bin_centers), np.array(actual_rates), np.array(bin_counts)


def compute_mispricing(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Add mispricing columns to DataFrame.
    
    Args:
        df: DataFrame with PM prices
        model: Fitted FairValueModel
        
    Returns:
        DataFrame with p_hat and mispricing columns
    """
    df = df.copy()
    
    # Get fair value prediction
    df['p_hat'] = model.predict(df)
    
    # Compute PM mid price
    df['pm_up_mid'] = (df['pm_up_best_bid'] + df['pm_up_best_ask']) / 2
    
    # Mispricing: PM price - fair value
    df['mispricing'] = df['pm_up_mid'] - df['p_hat']
    df['abs_mispricing'] = df['mispricing'].abs()
    
    return df


def analyze_strategy_vs_fair_value(
    df: pd.DataFrame,
    trades: list,
    model: FairValueModel
) -> Dict[str, Any]:
    """
    Analyze if strategy edge comes from mispricing or momentum.
    
    Args:
        df: Full DataFrame
        trades: List of trade dicts from backtest
        model: Fitted FairValueModel
        
    Returns:
        Analysis results
    """
    if not trades:
        return {'error': 'No trades to analyze'}
    
    df = compute_mispricing(df, model)
    
    # For each trade, get mispricing at entry
    trade_mispricings = []
    for trade in trades:
        market_df = df[df['market_id'] == trade['market_id']]
        entry_row = market_df[market_df['t'] == trade['entry_t']]
        
        if not entry_row.empty:
            trade_mispricings.append({
                'pnl': trade['pnl'],
                'mispricing': entry_row['mispricing'].iloc[0],
                'p_hat': entry_row['p_hat'].iloc[0],
                'pm_up_mid': entry_row['pm_up_mid'].iloc[0],
                'side': trade['side'],
                'tau': trade.get('tau_at_entry', 900 - trade['entry_t']),
            })
    
    misprice_df = pd.DataFrame(trade_mispricings)
    
    # Analyze correlation between mispricing and PnL
    if len(misprice_df) > 5:
        # For buy_up trades, positive mispricing means PM is too high
        # We should be profitable if PM comes down
        buy_up_trades = misprice_df[misprice_df['side'] == 'buy_up']
        buy_down_trades = misprice_df[misprice_df['side'] == 'buy_down']
        
        # Correlation of mispricing with PnL
        overall_corr = misprice_df['mispricing'].corr(misprice_df['pnl'])
        
        analysis = {
            'n_trades': len(misprice_df),
            'avg_mispricing': misprice_df['mispricing'].mean(),
            'std_mispricing': misprice_df['mispricing'].std(),
            'avg_abs_mispricing': misprice_df['abs_mispricing'].mean() if 'abs_mispricing' in misprice_df else misprice_df['mispricing'].abs().mean(),
            'pnl_mispricing_correlation': overall_corr,
            'avg_p_hat': misprice_df['p_hat'].mean(),
            'avg_pm_price': misprice_df['pm_up_mid'].mean(),
            'interpretation': interpret_mispricing_correlation(overall_corr),
        }
        
        return analysis
    
    return {'error': 'Not enough trades for analysis'}


def interpret_mispricing_correlation(corr: float) -> str:
    """Interpret the correlation between mispricing and PnL."""
    if abs(corr) < 0.1:
        return "No relationship: Edge is likely from latency, not fair value deviation"
    elif corr > 0.3:
        return "Positive correlation: Strategy may be exploiting overpriced PM (momentum)"
    elif corr < -0.3:
        return "Negative correlation: Strategy may be exploiting underpriced PM (contrarian)"
    else:
        return "Weak relationship: Mix of latency and fair value effects"


def run_fair_value_analysis(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_trades: list,
    test_trades: list
) -> Dict[str, Any]:
    """
    Run complete fair value analysis.
    
    Args:
        train_df: Training data
        test_df: Test data
        train_trades: Trades on training data
        test_trades: Trades on test data
        
    Returns:
        Complete analysis results
    """
    # Fit model on train
    model = FairValueModel()
    model.fit(train_df)
    
    # Analyze train and test
    train_analysis = analyze_strategy_vs_fair_value(train_df, train_trades, model)
    test_analysis = analyze_strategy_vs_fair_value(test_df, test_trades, model)
    
    # Model summary
    model_summary = model.get_model_summary()
    
    return {
        'model_summary': model_summary,
        'train_analysis': train_analysis,
        'test_analysis': test_analysis,
    }


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from scripts.backtest.data_loader import load_eth_markets, add_derived_columns, get_train_test_split
    from scripts.backtest.strategies import StrikeCrossStrategy
    from scripts.backtest.backtest_engine import run_backtest, ExecutionConfig
    
    print("Loading data...")
    df, market_info = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    print("Splitting train/test...")
    train_df, test_df, _, _ = get_train_test_split(df)
    
    print("\nFitting fair value model...")
    model = FairValueModel()
    model.fit(train_df)
    
    print("\nModel Summary:")
    summary = model.get_model_summary()
    print(f"  Intercept: {summary['intercept']:.4f}")
    for feat, coef in summary['coefficients'].items():
        print(f"  {feat}: {coef:.4f}")
    
    print("\nRunning best strategy...")
    strategy = StrikeCrossStrategy(tau_max=600, hold_to_expiry=True)
    
    train_result = run_backtest(train_df, strategy, ExecutionConfig())
    test_result = run_backtest(test_df, strategy, ExecutionConfig())
    
    print("\nAnalyzing strategy vs fair value...")
    analysis = run_fair_value_analysis(
        train_df, test_df,
        train_result['trades'], test_result['trades']
    )
    
    print("\n" + "="*60)
    print("FAIR VALUE ANALYSIS")
    print("="*60)
    
    print("\nTrain Set:")
    for k, v in analysis['train_analysis'].items():
        print(f"  {k}: {v}")
    
    print("\nTest Set:")
    for k, v in analysis['test_analysis'].items():
        print(f"  {k}: {v}")

