"""
Visualizations for Backtest Results

Creates interactive HTML charts using Plotly.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Install with: pip install plotly")


def plot_response_curve(
    study_df: pd.DataFrame,
    event_type: Optional[str] = None,
    title: str = "PM Response to CL Events"
) -> Optional[go.Figure]:
    """
    Plot average PM response curve vs lag.
    
    Args:
        study_df: Event study results
        event_type: Filter by event type
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    df = study_df.copy()
    if event_type:
        df = df[df['event_type'] == event_type]
    
    # Compute average response at each lag
    avg_response = df.groupby('lag').agg({
        'pm_directed_response': ['mean', 'std', 'count']
    }).reset_index()
    avg_response.columns = ['lag', 'mean', 'std', 'n']
    avg_response['se'] = avg_response['std'] / np.sqrt(avg_response['n'])
    avg_response['ci_upper'] = avg_response['mean'] + 1.96 * avg_response['se']
    avg_response['ci_lower'] = avg_response['mean'] - 1.96 * avg_response['se']
    
    fig = go.Figure()
    
    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(avg_response['lag']) + list(avg_response['lag'][::-1]),
        y=list(avg_response['ci_upper']) + list(avg_response['ci_lower'][::-1]),
        fill='toself',
        fillcolor='rgba(68, 68, 68, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI'
    ))
    
    # Mean line
    fig.add_trace(go.Scatter(
        x=avg_response['lag'],
        y=avg_response['mean'],
        mode='lines+markers',
        name='Mean Response',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=title,
        xaxis_title="Lag (seconds)",
        yaxis_title="PM Response (directed)",
        template="plotly_white",
        height=500
    )
    
    return fig


def plot_latency_cliff(
    summary_df: pd.DataFrame,
    title: str = "Latency Cliff Analysis"
) -> Optional[go.Figure]:
    """
    Plot PnL vs execution latency.
    
    Args:
        summary_df: Latency analysis results
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=["Total PnL", "Hit Rate"])
    
    # PnL subplot
    fig.add_trace(go.Scatter(
        x=summary_df['latency'],
        y=summary_df['total_pnl'],
        mode='lines+markers',
        name='Total PnL',
        line=dict(color='#2ca02c', width=2),
        marker=dict(size=8)
    ), row=1, col=1)
    
    # Hit rate subplot
    fig.add_trace(go.Scatter(
        x=summary_df['latency'],
        y=summary_df['hit_rate'] * 100 if 'hit_rate' in summary_df else summary_df['hit_rate_per_market'] * 100,
        mode='lines+markers',
        name='Hit Rate (%)',
        line=dict(color='#ff7f0e', width=2),
        marker=dict(size=8)
    ), row=2, col=1)
    
    fig.update_layout(
        title=title,
        height=600,
        showlegend=False,
        template="plotly_white"
    )
    
    fig.update_xaxes(title_text="Latency (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="PnL ($)", row=1, col=1)
    fig.update_yaxes(title_text="Hit Rate (%)", row=2, col=1)
    
    return fig


def plot_parameter_heatmap(
    results_df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str = 'train_total_pnl',
    title: str = "Parameter Heatmap"
) -> Optional[go.Figure]:
    """
    Plot parameter sweep as heatmap.
    
    Args:
        results_df: Parameter sweep results
        x_param: Column for x-axis
        y_param: Column for y-axis
        metric: Column for color
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    # Create pivot table
    pivot = results_df.pivot_table(
        values=metric,
        index=y_param,
        columns=x_param,
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn',
        text=np.round(pivot.values, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title=metric)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_param,
        yaxis_title=y_param,
        height=500,
        template="plotly_white"
    )
    
    return fig


def plot_pnl_distribution(
    market_pnls: Dict[str, float],
    title: str = "Per-Market PnL Distribution"
) -> Optional[go.Figure]:
    """
    Plot histogram of per-market PnL.
    
    Args:
        market_pnls: Dict of market_id -> PnL
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    pnls = list(market_pnls.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=pnls,
        nbinsx=20,
        name='PnL',
        marker_color='#1f77b4'
    ))
    
    # Add mean line
    mean_pnl = np.mean(pnls)
    fig.add_vline(x=mean_pnl, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: ${mean_pnl:.3f}")
    
    fig.update_layout(
        title=title,
        xaxis_title="PnL ($)",
        yaxis_title="Count",
        height=400,
        template="plotly_white"
    )
    
    return fig


def plot_equity_curve(
    trades: List[Dict],
    title: str = "Cumulative PnL (Equity Curve)"
) -> Optional[go.Figure]:
    """
    Plot cumulative PnL over trades.
    
    Args:
        trades: List of trade dictionaries
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    if not trades:
        return None
    
    # Sort by entry time
    sorted_trades = sorted(trades, key=lambda x: (x['market_id'], x['entry_t']))
    
    cum_pnl = []
    running = 0
    for trade in sorted_trades:
        running += trade['pnl']
        cum_pnl.append(running)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cum_pnl) + 1)),
        y=cum_pnl,
        mode='lines',
        name='Cumulative PnL',
        line=dict(color='#2ca02c', width=2),
        fill='tozeroy',
        fillcolor='rgba(44, 160, 44, 0.2)'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=title,
        xaxis_title="Trade Number",
        yaxis_title="Cumulative PnL ($)",
        height=400,
        template="plotly_white"
    )
    
    return fig


def plot_placebo_comparison(
    real_pnl: float,
    placebo_pnls: List[float],
    title: str = "Real vs Placebo PnL"
) -> Optional[go.Figure]:
    """
    Plot real strategy vs placebo distribution.
    
    Args:
        real_pnl: PnL from real strategy
        placebo_pnls: List of PnL from placebo runs
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    fig = go.Figure()
    
    # Placebo histogram
    fig.add_trace(go.Histogram(
        x=placebo_pnls,
        nbinsx=15,
        name='Placebo',
        marker_color='#d62728',
        opacity=0.7
    ))
    
    # Real strategy line
    fig.add_vline(x=real_pnl, line_dash="solid", line_color="green", line_width=3,
                  annotation_text=f"Real: ${real_pnl:.2f}")
    
    # Placebo mean line
    placebo_mean = np.mean(placebo_pnls)
    fig.add_vline(x=placebo_mean, line_dash="dash", line_color="red",
                  annotation_text=f"Placebo Mean: ${placebo_mean:.2f}")
    
    fig.update_layout(
        title=title,
        xaxis_title="PnL ($)",
        yaxis_title="Count",
        height=400,
        template="plotly_white"
    )
    
    return fig


def plot_strategy_comparison(
    comparison_df: pd.DataFrame,
    title: str = "Strategy Comparison"
) -> Optional[go.Figure]:
    """
    Plot bar chart comparing strategies.
    
    Args:
        comparison_df: Strategy comparison DataFrame
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Total PnL", "t-statistic"])
    
    # PnL bars
    colors = ['#2ca02c' if pnl > 0 else '#d62728' for pnl in comparison_df['total_pnl']]
    fig.add_trace(go.Bar(
        x=comparison_df['strategy'],
        y=comparison_df['total_pnl'],
        marker_color=colors,
        name='Total PnL'
    ), row=1, col=1)
    
    # t-stat bars
    colors = ['#2ca02c' if t > 1.96 else '#ff7f0e' if t > 1.65 else '#d62728' 
              for t in comparison_df['t_stat']]
    fig.add_trace(go.Bar(
        x=comparison_df['strategy'],
        y=comparison_df['t_stat'],
        marker_color=colors,
        name='t-stat'
    ), row=1, col=2)
    
    # Significance lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=1.96, line_dash="dash", line_color="green", row=1, col=2,
                  annotation_text="95% sig")
    fig.add_hline(y=1.65, line_dash="dash", line_color="orange", row=1, col=2,
                  annotation_text="90% sig")
    
    fig.update_layout(
        title=title,
        height=500,
        showlegend=False,
        template="plotly_white"
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig


def plot_tau_pnl_heatmap(
    trades: List[Dict],
    tau_bins: List[int] = None,
    title: str = "PnL by Time-to-Expiry (Tau)"
) -> Optional[go.Figure]:
    """
    Plot heatmap/bar chart of PnL by time-to-expiry bucket.
    
    Args:
        trades: List of trade dictionaries with 'tau_at_entry' and 'pnl'
        tau_bins: Bin edges for tau (default: [0, 60, 120, 180, 300, 600, 900])
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE or not trades:
        return None
    
    if tau_bins is None:
        tau_bins = [0, 60, 120, 180, 300, 600, 900]
    
    df = pd.DataFrame(trades)
    
    if 'tau_at_entry' not in df.columns:
        return None
    
    # Create tau buckets
    df['tau_bucket'] = pd.cut(df['tau_at_entry'], bins=tau_bins, labels=[
        f"{tau_bins[i]}-{tau_bins[i+1]}s" for i in range(len(tau_bins)-1)
    ])
    
    # Aggregate by bucket
    agg = df.groupby('tau_bucket', observed=True).agg({
        'pnl': ['sum', 'mean', 'count']
    }).reset_index()
    agg.columns = ['tau_bucket', 'total_pnl', 'avg_pnl', 'n_trades']
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Total PnL by Tau", "Trade Count by Tau"])
    
    colors = ['#2ca02c' if p > 0 else '#d62728' for p in agg['total_pnl']]
    
    fig.add_trace(go.Bar(
        x=agg['tau_bucket'].astype(str),
        y=agg['total_pnl'],
        marker_color=colors,
        name='Total PnL',
        text=[f"${p:.3f}" for p in agg['total_pnl']],
        textposition='auto'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=agg['tau_bucket'].astype(str),
        y=agg['n_trades'],
        marker_color='#1f77b4',
        name='N Trades'
    ), row=1, col=2)
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    fig.update_layout(
        title=title,
        height=450,
        template="plotly_white",
        showlegend=False
    )
    
    fig.update_xaxes(tickangle=45, title_text="Time-to-Expiry")
    fig.update_yaxes(title_text="PnL ($)", row=1, col=1)
    fig.update_yaxes(title_text="Trade Count", row=1, col=2)
    
    return fig


def plot_delta_pnl_heatmap(
    trades: List[Dict],
    delta_bins: List[float] = None,
    title: str = "PnL by Distance-to-Strike (Delta)"
) -> Optional[go.Figure]:
    """
    Plot heatmap/bar chart of PnL by distance-to-strike bucket.
    
    Args:
        trades: List of trade dictionaries with 'delta_bps' and 'pnl'
        delta_bins: Bin edges for delta in bps (default: [-inf, -50, -20, -10, 10, 20, 50, inf])
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE or not trades:
        return None
    
    if delta_bins is None:
        delta_bins = [-np.inf, -100, -50, -20, 0, 20, 50, 100, np.inf]
    
    df = pd.DataFrame(trades)
    
    if 'delta_bps' not in df.columns:
        return None
    
    # Create labels for bins
    labels = []
    for i in range(len(delta_bins)-1):
        low = delta_bins[i]
        high = delta_bins[i+1]
        if low == -np.inf:
            labels.append(f"<{high}")
        elif high == np.inf:
            labels.append(f">{low}")
        else:
            labels.append(f"{int(low)} to {int(high)}")
    
    df['delta_bucket'] = pd.cut(df['delta_bps'], bins=delta_bins, labels=labels)
    
    # Aggregate by bucket
    agg = df.groupby('delta_bucket', observed=True).agg({
        'pnl': ['sum', 'mean', 'count']
    }).reset_index()
    agg.columns = ['delta_bucket', 'total_pnl', 'avg_pnl', 'n_trades']
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Total PnL by Delta", "Trade Count by Delta"])
    
    colors = ['#2ca02c' if p > 0 else '#d62728' for p in agg['total_pnl']]
    
    fig.add_trace(go.Bar(
        x=agg['delta_bucket'].astype(str),
        y=agg['total_pnl'],
        marker_color=colors,
        name='Total PnL',
        text=[f"${p:.3f}" for p in agg['total_pnl']],
        textposition='auto'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=agg['delta_bucket'].astype(str),
        y=agg['n_trades'],
        marker_color='#1f77b4',
        name='N Trades'
    ), row=1, col=2)
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    fig.update_layout(
        title=title,
        height=450,
        template="plotly_white",
        showlegend=False
    )
    
    fig.update_xaxes(tickangle=45, title_text="Delta (bps from strike)")
    fig.update_yaxes(title_text="PnL ($)", row=1, col=1)
    fig.update_yaxes(title_text="Trade Count", row=1, col=2)
    
    return fig


def plot_exposure_turnover(
    trades: List[Dict],
    df: pd.DataFrame = None,
    title: str = "Position Exposure & Turnover"
) -> Optional[go.Figure]:
    """
    Plot position exposure and turnover over time.
    
    Args:
        trades: List of trade dictionaries
        df: Full DataFrame for time context
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE or not trades:
        return None
    
    trades_df = pd.DataFrame(trades)
    
    # Sort by market order and entry time
    trades_df = trades_df.sort_values(['market_id', 'entry_t'])
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Cumulative Exposure", "Trade Size Over Time"],
        row_heights=[0.6, 0.4]
    )
    
    # Compute cumulative exposure (simple: +1 for buy, -1 for sell)
    exposure = []
    running = 0
    for _, trade in trades_df.iterrows():
        if trade['side'].startswith('buy'):
            running += 1
        else:
            running -= 1
        exposure.append(running)
    
    trades_df['exposure'] = exposure
    trades_df['trade_num'] = range(1, len(trades_df) + 1)
    
    # Exposure over trades
    fig.add_trace(go.Scatter(
        x=trades_df['trade_num'],
        y=trades_df['exposure'],
        mode='lines',
        name='Net Exposure',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ), row=1, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    # PnL per trade (bar chart)
    colors = ['#2ca02c' if p > 0 else '#d62728' for p in trades_df['pnl']]
    fig.add_trace(go.Bar(
        x=trades_df['trade_num'],
        y=trades_df['pnl'],
        marker_color=colors,
        name='Trade PnL'
    ), row=2, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    fig.update_layout(
        title=title,
        height=600,
        template="plotly_white",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Trade Number", row=2, col=1)
    fig.update_yaxes(title_text="Net Position", row=1, col=1)
    fig.update_yaxes(title_text="PnL ($)", row=2, col=1)
    
    return fig


def plot_pnl_by_market(
    market_pnls: Dict[str, float],
    title: str = "PnL by Market"
) -> Optional[go.Figure]:
    """
    Plot sorted bar chart of PnL by market.
    
    Args:
        market_pnls: Dict of market_id -> PnL
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE or not market_pnls:
        return None
    
    # Sort by PnL
    sorted_markets = sorted(market_pnls.items(), key=lambda x: x[1])
    markets = [m[0][:15] + '...' if len(m[0]) > 15 else m[0] for m in sorted_markets]
    pnls = [m[1] for m in sorted_markets]
    colors = ['#2ca02c' if p > 0 else '#d62728' for p in pnls]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=markets,
        x=pnls,
        orientation='h',
        marker_color=colors,
        text=[f"${p:.3f}" for p in pnls],
        textposition='auto'
    ))
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    # Add mean line
    mean_pnl = np.mean(pnls)
    fig.add_vline(x=mean_pnl, line_dash="dot", line_color="blue",
                  annotation_text=f"Mean: ${mean_pnl:.3f}")
    
    fig.update_layout(
        title=title,
        xaxis_title="PnL ($)",
        height=max(400, len(markets) * 25),
        template="plotly_white",
        margin=dict(l=150)
    )
    
    return fig


def plot_equity_curve_with_bootstrap(
    trades: List[Dict],
    n_bootstrap: int = 100,
    title: str = "Equity Curve with Bootstrap CI"
) -> Optional[go.Figure]:
    """
    Plot equity curve with bootstrap confidence intervals.
    
    Args:
        trades: List of trade dictionaries
        n_bootstrap: Number of bootstrap samples
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE or not trades:
        return None
    
    # Sort and compute actual equity curve
    sorted_trades = sorted(trades, key=lambda x: (x['market_id'], x['entry_t']))
    pnls = [t['pnl'] for t in sorted_trades]
    cum_pnl = np.cumsum(pnls)
    
    # Bootstrap
    rng = np.random.RandomState(42)
    bootstrap_curves = []
    for _ in range(n_bootstrap):
        # Resample markets (block bootstrap)
        market_ids = list(set(t['market_id'] for t in sorted_trades))
        sampled_markets = rng.choice(market_ids, size=len(market_ids), replace=True)
        
        # Get trades for sampled markets
        sampled_pnls = []
        for m in sampled_markets:
            market_trades = [t for t in sorted_trades if t['market_id'] == m]
            sampled_pnls.extend([t['pnl'] for t in market_trades])
        
        bootstrap_curves.append(np.cumsum(sampled_pnls))
    
    # Compute percentiles (pad shorter curves)
    max_len = max(len(c) for c in bootstrap_curves)
    padded = np.array([np.pad(c, (0, max_len - len(c)), 'edge') for c in bootstrap_curves])
    
    ci_lower = np.percentile(padded, 2.5, axis=0)
    ci_upper = np.percentile(padded, 97.5, axis=0)
    
    fig = go.Figure()
    
    # CI band
    x_range = list(range(1, len(ci_lower) + 1))
    fig.add_trace(go.Scatter(
        x=x_range + x_range[::-1],
        y=list(ci_upper) + list(ci_lower[::-1]),
        fill='toself',
        fillcolor='rgba(44, 160, 44, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI'
    ))
    
    # Actual curve
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cum_pnl) + 1)),
        y=cum_pnl.tolist(),
        mode='lines',
        name='Actual',
        line=dict(color='#2ca02c', width=2)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=title,
        xaxis_title="Trade Number",
        yaxis_title="Cumulative PnL ($)",
        height=500,
        template="plotly_white"
    )
    
    return fig


def save_all_plots(
    output_dir: Path,
    event_study_df: pd.DataFrame = None,
    latency_df: pd.DataFrame = None,
    sweep_df: pd.DataFrame = None,
    trades: List[Dict] = None,
    market_pnls: Dict[str, float] = None,
    placebo_results: Dict = None,
    df: pd.DataFrame = None
) -> List[str]:
    """
    Generate and save all plots including new tau/delta heatmaps and exposure.
    
    Returns:
        List of saved file paths
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, skipping visualizations")
        return []
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved = []
    
    # Response curve
    if event_study_df is not None and not event_study_df.empty:
        fig = plot_response_curve(event_study_df)
        if fig:
            path = output_dir / 'response_curve.html'
            fig.write_html(str(path))
            saved.append(str(path))
    
    # Latency cliff
    if latency_df is not None and not latency_df.empty:
        fig = plot_latency_cliff(latency_df)
        if fig:
            path = output_dir / 'latency_cliff.html'
            fig.write_html(str(path))
            saved.append(str(path))
    
    # PnL distribution
    if market_pnls:
        fig = plot_pnl_distribution(market_pnls)
        if fig:
            path = output_dir / 'pnl_distribution.html'
            fig.write_html(str(path))
            saved.append(str(path))
        
        # PnL by market (sorted bar chart)
        fig = plot_pnl_by_market(market_pnls)
        if fig:
            path = output_dir / 'pnl_by_market.html'
            fig.write_html(str(path))
            saved.append(str(path))
    
    # Equity curve (simple)
    if trades:
        fig = plot_equity_curve(trades)
        if fig:
            path = output_dir / 'equity_curve.html'
            fig.write_html(str(path))
            saved.append(str(path))
        
        # Equity curve with bootstrap CI
        fig = plot_equity_curve_with_bootstrap(trades)
        if fig:
            path = output_dir / 'equity_curve_bootstrap.html'
            fig.write_html(str(path))
            saved.append(str(path))
        
        # PnL by tau (time-to-expiry)
        fig = plot_tau_pnl_heatmap(trades)
        if fig:
            path = output_dir / 'pnl_by_tau.html'
            fig.write_html(str(path))
            saved.append(str(path))
        
        # PnL by delta (distance-to-strike)
        fig = plot_delta_pnl_heatmap(trades)
        if fig:
            path = output_dir / 'pnl_by_delta.html'
            fig.write_html(str(path))
            saved.append(str(path))
        
        # Exposure and turnover
        fig = plot_exposure_turnover(trades, df)
        if fig:
            path = output_dir / 'exposure_turnover.html'
            fig.write_html(str(path))
            saved.append(str(path))
    
    # Placebo comparison
    if placebo_results:
        fig = plot_placebo_comparison(
            placebo_results['real_result']['total_pnl'],
            placebo_results['placebo_df']['total_pnl'].tolist()
        )
        if fig:
            path = output_dir / 'placebo_comparison.html'
            fig.write_html(str(path))
            saved.append(str(path))
    
    return saved


# ==============================================================================
# MAKER STRATEGY VISUALIZATIONS
# ==============================================================================

def plot_maker_fill_rate_by_tau(
    diagnostics: Dict[str, Any],
    title: str = "Fill Rate by Time-to-Expiry"
) -> Optional[go.Figure]:
    """
    Plot fill rate as a function of tau (time-to-expiry).
    
    Args:
        diagnostics: Results from maker_diagnostics
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    fill_by_tau = diagnostics.get('fill_rate_by_tau', [])
    if not fill_by_tau:
        return None
    
    df = pd.DataFrame(fill_by_tau)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['tau_label'],
        y=df['n_fills'],
        marker_color='#1f77b4',
        name='Fills'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time-to-Expiry Window",
        yaxis_title="Number of Fills",
        template="plotly_white",
        height=400
    )
    
    return fig


def plot_maker_fill_rate_by_spread(
    diagnostics: Dict[str, Any],
    title: str = "Fill Rate by Spread Width"
) -> Optional[go.Figure]:
    """
    Plot fill rate as a function of spread width.
    
    Args:
        diagnostics: Results from maker_diagnostics
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    fill_by_spread = diagnostics.get('fill_rate_by_spread', [])
    if not fill_by_spread:
        return None
    
    df = pd.DataFrame(fill_by_spread)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['spread_label'],
        y=df['pct_of_fills'] * 100,
        marker_color='#2ca02c',
        name='% of Fills'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Spread Width",
        yaxis_title="Percentage of Fills",
        template="plotly_white",
        height=400
    )
    
    return fig


def plot_maker_pnl_decomposition(
    metrics: Dict[str, Any],
    title: str = "PnL Decomposition"
) -> Optional[go.Figure]:
    """
    Plot PnL decomposition into spread captured, adverse selection, inventory carry.
    
    Args:
        metrics: Metrics from maker backtest
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    components = [
        ('Spread Captured', metrics.get('spread_captured_total', 0)),
        ('Adverse Selection', -metrics.get('adverse_selection_total', 0)),
        ('Inventory Carry', metrics.get('inventory_carry_total', 0)),
    ]
    
    names = [c[0] for c in components]
    values = [c[1] for c in components]
    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=names,
        y=values,
        marker_color=colors,
        text=[f"${v:.4f}" for v in values],
        textposition='outside'
    ))
    
    total_pnl = sum(values)
    fig.add_hline(y=total_pnl, line_dash="dash", line_color="black",
                  annotation_text=f"Total: ${total_pnl:.4f}")
    
    fig.update_layout(
        title=title,
        xaxis_title="Component",
        yaxis_title="PnL ($)",
        template="plotly_white",
        height=400
    )
    
    return fig


def plot_maker_latency_cliff(
    latency_results: Dict[str, Any],
    title: str = "Maker Latency Cliff Analysis"
) -> Optional[go.Figure]:
    """
    Plot PnL vs placement latency for maker strategies.
    
    Args:
        latency_results: Results from maker latency sweep
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    place_summary = latency_results.get('place_latency_summary')
    if place_summary is None or len(place_summary) == 0:
        return None
    
    if isinstance(place_summary, pd.DataFrame):
        df = place_summary
    else:
        df = pd.DataFrame(place_summary)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('PnL vs Placement Latency', 'Fill Rate vs Placement Latency'),
        horizontal_spacing=0.15
    )
    
    # PnL line
    fig.add_trace(
        go.Scatter(
            x=df['place_latency_ms'],
            y=df['total_pnl'],
            mode='lines+markers',
            name='Total PnL',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Zero line for PnL
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    # Fill rate line
    fig.add_trace(
        go.Scatter(
            x=df['place_latency_ms'],
            y=df['fill_rate'] * 100,
            mode='lines+markers',
            name='Fill Rate',
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=8)
        ),
        row=1, col=2
    )
    
    # Mark cliff point
    cliff = latency_results.get('cliff_place_latency_ms', 0)
    if cliff > 0:
        fig.add_vline(x=cliff, line_dash="dash", line_color="red",
                      annotation_text=f"Cliff: {cliff}ms", row=1, col=1)
    
    fig.update_xaxes(title_text="Placement Latency (ms)", row=1, col=1)
    fig.update_xaxes(title_text="Placement Latency (ms)", row=1, col=2)
    fig.update_yaxes(title_text="PnL ($)", row=1, col=1)
    fig.update_yaxes(title_text="Fill Rate (%)", row=1, col=2)
    
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    return fig


def plot_maker_stress_test(
    stress_results: Dict[str, Any],
    title: str = "Stress Test Results"
) -> Optional[go.Figure]:
    """
    Plot stress test results (slippage, spread widening, volatility removal).
    
    Args:
        stress_results: Results from stress tests
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Slippage Tolerance',
            'Spread Widening',
            'Volatility Removal',
            'Fill Rate Sensitivity'
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.15
    )
    
    # 1. Slippage
    slip = stress_results.get('slippage_test', {}).get('results', [])
    if slip:
        slip_df = pd.DataFrame(slip)
        fig.add_trace(
            go.Scatter(
                x=slip_df['slippage_bps'],
                y=slip_df['total_pnl'],
                mode='lines+markers',
                name='Slippage',
                line=dict(color='#1f77b4')
            ),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    # 2. Spread widening
    spr = stress_results.get('spread_test', {}).get('results', [])
    if spr:
        spr_df = pd.DataFrame(spr)
        fig.add_trace(
            go.Scatter(
                x=spr_df['widen_factor'],
                y=spr_df['total_pnl'],
                mode='lines+markers',
                name='Spread',
                line=dict(color='#2ca02c')
            ),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
    
    # 3. Volatility removal
    vol = stress_results.get('volatility_test', {}).get('results', [])
    if vol:
        vol_df = pd.DataFrame(vol)
        fig.add_trace(
            go.Scatter(
                x=vol_df['pct_removed'] * 100,
                y=vol_df['total_pnl'],
                mode='lines+markers',
                name='Volatility',
                line=dict(color='#d62728')
            ),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # 4. Fill rate sensitivity
    fr = stress_results.get('fill_rate_test', {}).get('results', [])
    if fr:
        fr_df = pd.DataFrame(fr)
        fig.add_trace(
            go.Scatter(
                x=fr_df['touch_trade_rate'],
                y=fr_df['total_pnl'],
                mode='lines+markers',
                name='Fill Rate',
                line=dict(color='#9467bd')
            ),
            row=2, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)
    
    fig.update_xaxes(title_text="Slippage (bps)", row=1, col=1)
    fig.update_xaxes(title_text="Spread Factor", row=1, col=2)
    fig.update_xaxes(title_text="% Removed", row=2, col=1)
    fig.update_xaxes(title_text="Touch Trade Rate", row=2, col=2)
    
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_yaxes(title_text="PnL ($)", row=i, col=j)
    
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=600,
        showlegend=False
    )
    
    return fig


def plot_maker_pnl_by_market(
    backtest_result: Dict[str, Any],
    title: str = "PnL by Market"
) -> Optional[go.Figure]:
    """
    Plot PnL for each market in the maker backtest.
    
    Args:
        backtest_result: Results from run_maker_backtest
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    market_results = backtest_result.get('market_results', {})
    if not market_results:
        return None
    
    # Extract PnLs
    pnls = [(mid, r.get('pnl', 0)) for mid, r in market_results.items()]
    pnls.sort(key=lambda x: x[1])  # Sort by PnL
    
    markets = [p[0] for p in pnls]
    values = [p[1] for p in pnls]
    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=markets,
        y=values,
        marker_color=colors,
    ))
    
    fig.add_hline(y=0, line_dash="solid", line_color="black")
    
    # Add mean line
    mean_pnl = np.mean(values)
    fig.add_hline(y=mean_pnl, line_dash="dash", line_color="blue",
                  annotation_text=f"Mean: ${mean_pnl:.4f}")
    
    fig.update_layout(
        title=title,
        xaxis_title="Market ID",
        yaxis_title="PnL ($)",
        template="plotly_white",
        height=400,
        xaxis_tickangle=45
    )
    
    return fig


def save_maker_plots(
    output_dir: Path,
    backtest_result: Dict[str, Any],
    diagnostics: Optional[Dict[str, Any]] = None,
    latency_results: Optional[Dict[str, Any]] = None,
    stress_results: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Save all maker strategy plots to output directory.
    
    Args:
        output_dir: Where to save plots
        backtest_result: Results from run_maker_backtest
        diagnostics: Results from maker_diagnostics
        latency_results: Results from maker latency sweep
        stress_results: Results from stress tests
        
    Returns:
        List of saved file paths
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, skipping visualizations")
        return []
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved = []
    
    metrics = backtest_result.get('metrics', {})
    
    # PnL decomposition
    fig = plot_maker_pnl_decomposition(metrics)
    if fig:
        path = output_dir / 'maker_pnl_decomposition.html'
        fig.write_html(str(path))
        saved.append(str(path))
    
    # PnL by market
    fig = plot_maker_pnl_by_market(backtest_result)
    if fig:
        path = output_dir / 'maker_pnl_by_market.html'
        fig.write_html(str(path))
        saved.append(str(path))
    
    # Fill rate by tau
    if diagnostics:
        fig = plot_maker_fill_rate_by_tau(diagnostics)
        if fig:
            path = output_dir / 'maker_fill_rate_by_tau.html'
            fig.write_html(str(path))
            saved.append(str(path))
        
        fig = plot_maker_fill_rate_by_spread(diagnostics)
        if fig:
            path = output_dir / 'maker_fill_rate_by_spread.html'
            fig.write_html(str(path))
            saved.append(str(path))
    
    # Latency cliff
    if latency_results:
        fig = plot_maker_latency_cliff(latency_results)
        if fig:
            path = output_dir / 'maker_latency_cliff.html'
            fig.write_html(str(path))
            saved.append(str(path))
    
    # Stress tests
    if stress_results:
        fig = plot_maker_stress_test(stress_results)
        if fig:
            path = output_dir / 'maker_stress_tests.html'
            fig.write_html(str(path))
            saved.append(str(path))
    
    return saved


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    import sys
    
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from scripts.backtest.data_loader import load_eth_markets, add_derived_columns
    from scripts.backtest.strategies import StrikeCrossStrategy
    from scripts.backtest.backtest_engine import run_backtest, ExecutionConfig
    from scripts.backtest.event_study import run_full_event_study
    from scripts.backtest.latency_cliff import run_strategy_latency_analysis
    from scripts.backtest.placebo_tests import run_placebo_test_random
    
    print("Loading data...")
    df, market_info = load_eth_markets(min_coverage=90.0)
    df = add_derived_columns(df)
    
    output_dir = project_root / 'data_v2' / 'backtest_results' / 'plots'
    
    print("\nGenerating plots...")
    
    # Event study
    print("  Running event study...")
    study_results = run_full_event_study(df)
    
    # Strategy backtest
    print("  Running backtest...")
    strategy = StrikeCrossStrategy(tau_max=600, hold_to_expiry=True)
    result = run_backtest(df, strategy, ExecutionConfig())
    
    # Latency analysis
    print("  Running latency analysis...")
    latency_results = run_latency_analysis(df)
    
    # Placebo test
    print("  Running placebo test...")
    placebo_results = run_placebo_test_random(df, strategy, n_iterations=10)
    
    # Extract market PnLs
    market_pnls = {m: v['pnl'] for m, v in result['market_breakdown'].items()}
    
    print("\nSaving plots...")
    saved = save_all_plots(
        output_dir,
        event_study_df=study_results['study_df'],
        latency_df=latency_results['summary_df'],
        trades=result['trades'],
        market_pnls=market_pnls,
        placebo_results=placebo_results
    )
    
    print(f"\nSaved {len(saved)} plots to {output_dir}")
    for path in saved:
        print(f"  {path}")

