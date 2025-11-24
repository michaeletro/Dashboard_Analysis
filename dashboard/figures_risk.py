"""
Risk Management Dashboard Figures
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import lru_cache

from .config import THEME, style_figure
from . import data


@lru_cache(maxsize=32)
def make_risk_dashboard_figure() -> go.Figure:
    """Create individual dataset risk analysis dashboard (not unified portfolio)."""
    data.ensure_data_loaded()
    df_equity = data.df_equity
    df_bonds = data.df_bonds
    engine = data.engine
    
    # Create 2x2 subplot layout for separate dataset analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Equity Dataset: Return Distribution Analysis', 
            'Equity Dataset: Factor Score Analysis',
            'Bond Dataset: Duration Risk Distribution',
            'Time Series Dataset: Individual Asset Volatilities'
        ],
        specs=[
            [{'type': 'histogram'}, {'type': 'bar'}],
            [{'type': 'histogram'}, {'type': 'scatter'}]
        ]
    )
    
    try:
        # 1. EQUITY DATASET: Individual Stock Return Distribution Analysis
        if df_equity is not None and not df_equity.empty:
            # Try different possible return column names
            return_cols = ['1M % Change', '1M Change', 'Monthly Return', '1M Return']
            return_col = None
            for col in return_cols:
                if col in df_equity.columns:
                    return_col = col
                    break
            
            if return_col and not df_equity[return_col].isna().all():
                returns = df_equity[return_col].dropna().values
                
                if len(returns) > 0:
                    # Distribution of individual stock returns (not portfolio)
                    fig.add_trace(
                        go.Histogram(
                            x=returns,
                            nbinsx=min(25, max(10, len(returns) // 10)),
                            name='Individual Stock Returns',
                            showlegend=False,
                            marker_color=THEME['accent2'],
                            opacity=0.7
                        ),
                        row=1, col=1
                    )
                    
                    # Add statistics annotations
                    mean_ret = np.mean(returns)
                    std_ret = np.std(returns)
                    percentile_5 = np.percentile(returns, 5)
                    percentile_95 = np.percentile(returns, 95)
                    
                    fig.add_annotation(
                        x=0.23, y=0.95, xref="paper", yref="paper",
                        text=f"Cross-Sectional Stats:<br>Mean: {mean_ret:.2f}%<br>Std: {std_ret:.2f}%<br>5th Pctile: {percentile_5:.2f}%<br>95th Pctile: {percentile_95:.2f}%<br>N={len(returns)} stocks",
                        showarrow=False,
                        align="left",
                        font=dict(size=10, color=THEME['text'])
                    )
                else:
                    fig.add_annotation(
                        x=0.25, y=0.75, xref="paper", yref="paper",
                        text="No valid return data found",
                        showarrow=False,
                        font=dict(size=12, color=THEME['muted'])
                    )
            else:
                fig.add_annotation(
                    x=0.25, y=0.75, xref="paper", yref="paper",
                    text="Return data not available",
                    showarrow=False,
                    font=dict(size=12, color=THEME['muted'])
                )
        
        # 2. EQUITY DATASET: Factor Score Analysis (average scores across dataset)
        if df_equity is not None and not df_equity.empty:
            factor_cols = [col for col in df_equity.columns if col.endswith('Score') and not df_equity[col].isna().all()]
            
            if len(factor_cols) >= 1:
                factor_means = []
                factor_stds = []
                factor_names = []
                
                for factor in factor_cols[:6]:  # Limit to top 6 factors for readability
                    scores = df_equity[factor].dropna().values
                    if len(scores) > 0:
                        factor_means.append(np.mean(scores))
                        factor_stds.append(np.std(scores))
                        factor_names.append(factor.replace('Score', '').replace('_', ' '))
                
                if factor_means:
                    fig.add_trace(
                        go.Bar(
                            x=factor_names,
                            y=factor_means,
                            error_y=dict(type='data', array=factor_stds, visible=True),
                            name='Avg Factor Scores',
                            showlegend=False,
                            marker_color=THEME['accent'],
                            text=[f'{score:.3f}' for score in factor_means],
                            textposition='auto'
                        ),
                        row=1, col=2
                    )
            else:
                fig.add_annotation(
                    x=0.75, y=0.75, xref="paper", yref="paper",
                    text="No factor scores found",
                    showarrow=False,
                    font=dict(size=12, color=THEME['muted'])
                )
        
        # 3. BOND DATASET: Duration Risk Distribution Analysis
        if df_bonds is not None and not df_bonds.empty and 'Duration' in df_bonds.columns:
            durations = df_bonds['Duration'].dropna().values
            
            if len(durations) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=durations,
                        nbinsx=min(20, max(5, len(durations) // 5)),
                        name='Bond Duration Distribution',
                        showlegend=False,
                        marker_color=THEME['accent'],
                        opacity=0.7
                    ),
                    row=2, col=1
                )
                
                # Duration statistics
                mean_dur = np.mean(durations)
                median_dur = np.median(durations)
                fig.add_vline(
                    x=mean_dur,
                    line=dict(color='white', width=2, dash='dash'),
                    annotation_text=f'Mean: {mean_dur:.1f}y',
                    row=2, col=1
                )
                fig.add_vline(
                    x=median_dur,
                    line=dict(color=THEME['accent2'], width=2, dash='dot'),
                    annotation_text=f'Median: {median_dur:.1f}y',
                    row=2, col=1
                )
        else:
            fig.add_annotation(
                x=0.25, y=0.25, xref="paper", yref="paper",
                text="Bond duration data not available",
                showarrow=False,
                font=dict(size=12, color=THEME['muted'])
            )
        
        # 4. TIME SERIES DATASET: Individual Asset Volatility Analysis  
        if engine and hasattr(engine, 'returns') and engine.returns is not None:
            returns_ts = engine.returns
            if len(returns_ts.columns) > 0:
                # Calculate annualized volatilities for each asset
                vol_annual = returns_ts.std() * np.sqrt(252) * 100  # Convert to percentage
                
                # Limit to first 15 assets for readability
                n_assets = min(15, len(vol_annual))
                asset_names = list(vol_annual.index)[:n_assets] 
                volatilities = vol_annual.values[:n_assets]
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(volatilities))),
                        y=volatilities,
                        mode='markers+text',
                        marker=dict(
                            size=10,
                            color=volatilities,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Vol %", len=0.4, y=0.3, x=1.02)
                        ),
                        text=[name[:8] + '...' if len(name) > 8 else name for name in asset_names],
                        textposition='top center',
                        hovertemplate='Asset: %{text}<br>Volatility: %{y:.1f}%<extra></extra>',
                        name='Asset Volatilities',
                        showlegend=False
                    ),
                    row=2, col=2
                )
                
                # Add average volatility line
                avg_vol = np.mean(volatilities)
                fig.add_hline(
                    y=avg_vol,
                    line=dict(color='white', width=1, dash='dash'),
                    annotation_text=f'Avg: {avg_vol:.1f}%',
                    row=2, col=2
                )
        else:
            fig.add_annotation(
                x=0.75, y=0.25, xref="paper", yref="paper",
                text="Time series data not available",
                showarrow=False,
                font=dict(size=12, color=THEME['muted'])
            )
                
    except Exception as e:
        # Add error message if calculations fail
        fig.add_annotation(
            text=f"Dataset analysis error: {str(e)[:100]}...",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#ff4444")
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title=dict(
            text="Individual Dataset Risk Analysis (Not Unified Portfolio)",
            font=dict(size=18, color=THEME['text']),
            x=0.5
        ),
        paper_bgcolor=THEME['bg'],
        plot_bgcolor=THEME['panel'],
        font=dict(color=THEME['text'])
    )
    
    # Update axes with appropriate labels for each dataset
    fig.update_xaxes(title_text="Return (%)", row=1, col=1, gridcolor=THEME['grid'], color=THEME['text'])
    fig.update_yaxes(title_text="Number of Stocks", row=1, col=1, gridcolor=THEME['grid'], color=THEME['text'])
    
    fig.update_xaxes(title_text="Factors", row=1, col=2, gridcolor=THEME['grid'], color=THEME['text'])
    fig.update_yaxes(title_text="Average Score", row=1, col=2, gridcolor=THEME['grid'], color=THEME['text'])
    
    fig.update_xaxes(title_text="Duration (Years)", row=2, col=1, gridcolor=THEME['grid'], color=THEME['text'])
    fig.update_yaxes(title_text="Number of Bonds", row=2, col=1, gridcolor=THEME['grid'], color=THEME['text'])
    
    fig.update_xaxes(title_text="Asset Index", row=2, col=2, gridcolor=THEME['grid'], color=THEME['text'])
    fig.update_yaxes(title_text="Annualized Volatility (%)", row=2, col=2, gridcolor=THEME['grid'], color=THEME['text'])
    
    return fig


@lru_cache(maxsize=32)  
def make_correlation_heatmap_figure() -> go.Figure:
    """Create correlation heatmap for equity dataset factors and metrics."""
    data.ensure_data_loaded()
    df_equity = data.df_equity
    
    fig = go.Figure()
    
    try:
        if df_equity is None or df_equity.empty:
            fig.add_annotation(
                text="No equity data available for correlation analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color=THEME['text'])
            )
            return style_figure(fig, height=600, title="Factor & Metric Correlation Matrix")
        
        # Get factor scores and key metrics that exist in the data
        factor_cols = [col for col in df_equity.columns if col.endswith('Score') and not df_equity[col].isna().all()]
        
        # Check for various possible metric column names
        possible_metrics = {
            'P/E (TTM)': ['P/E (TTM)', 'P/E', 'PE Ratio', 'Price/Earnings'],
            'P/B (TTM)': ['P/B (TTM)', 'P/B', 'PB Ratio', 'Price/Book'],
            'ROE (%)': ['ROE (%)', 'ROE', 'Return on Equity', 'ROE %'],
            'Debt/Equity': ['Debt/Equity', 'D/E', 'Debt to Equity', 'Debt Equity Ratio'],
            '1M % Change': ['1M % Change', '1M Change', '1M Return', 'Monthly Return']
        }
        
        key_metrics = []
        for display_name, possible_names in possible_metrics.items():
            for name in possible_names:
                if name in df_equity.columns and not df_equity[name].isna().all():
                    key_metrics.append(name)
                    break
        
        # Combine all available columns for analysis
        analysis_cols = factor_cols + key_metrics
        
        if len(analysis_cols) < 3:
            available_cols = list(df_equity.columns)
            fig.add_annotation(
                text=f"Insufficient numeric data for correlation analysis.\nFound {len(analysis_cols)} suitable columns.\nAvailable columns: {', '.join(available_cols[:10])}{'...' if len(available_cols) > 10 else ''}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=12, color=THEME['text'])
            )
            return style_figure(fig, height=600, title="Factor & Metric Correlation Matrix")
        
        # Select numeric data and remove rows with all NaN
        correlation_data = df_equity[analysis_cols].select_dtypes(include=[np.number])
        
        # Remove columns with insufficient data (less than 10 non-null values)
        valid_cols = []
        for col in correlation_data.columns:
            if correlation_data[col].notna().sum() >= 10:
                valid_cols.append(col)
        
        if len(valid_cols) < 3:
            fig.add_annotation(
                text=f"Insufficient valid data points for correlation.\nNeed at least 10 non-null values per column.\nValid columns: {len(valid_cols)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=12, color=THEME['text'])
            )
            return style_figure(fig, height=600, title="Factor & Metric Correlation Matrix")
        
        # Calculate correlation matrix
        corr_data = correlation_data[valid_cols].corr()
        
        # Clean column names for display
        clean_labels = [col.replace('Score', '').replace(' (%)', '').replace(' (TTM)', '') for col in valid_cols]
        
        # Create heatmap
        fig.add_trace(
            go.Heatmap(
                z=corr_data.values,
                x=clean_labels,
                y=clean_labels,
                colorscale=[
                    [0.0, '#1e3a8a'],    # Dark blue for negative
                    [0.25, '#3b82f6'],   # Blue  
                    [0.5, '#ffffff'],    # White for zero
                    [0.75, '#f59e0b'],   # Orange
                    [1.0, '#dc2626']     # Red for positive
                ],
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.around(corr_data.values, 2),
                texttemplate='%{text}',
                textfont={"size": 11, "color": "black"},
                hoverongaps=False,
                colorbar=dict(
                    title="Correlation",
                    title_font=dict(color=THEME['text'], size=12),
                    tickfont=dict(color=THEME['text'], size=10),
                    len=0.8
                )
            )
        )
        
        # Add info annotation
        fig.add_annotation(
            text=f"Analyzing {len(valid_cols)} variables with sufficient data",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10, color=THEME['muted']),
            xanchor="left", yanchor="top"
        )
            
    except Exception as e:
        fig.add_annotation(
            text=f"Correlation analysis error: {str(e)[:100]}...",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#ff4444")
        )
    
    return style_figure(
        fig,
        height=600,
        title="Equity Dataset: Factor & Metric Correlations"
    )


@lru_cache(maxsize=32)
def make_bond_duration_risk_figure() -> go.Figure:
    """Create enhanced bond duration risk analysis figure."""
    data.ensure_data_loaded()
    df_bonds = data.df_bonds
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Duration Distribution', 'Interest Rate Sensitivity'],
        specs=[[{'type': 'histogram'}, {'type': 'scatter'}]]
    )
    
    try:
        if df_bonds is not None and not df_bonds.empty:
            # Check for duration column with various possible names
            duration_cols = ['Duration', 'Modified Duration', 'Mod Duration', 'duration']
            duration_col = None
            for col in duration_cols:
                if col in df_bonds.columns:
                    duration_col = col
                    break
            
            if duration_col and not df_bonds[duration_col].isna().all():
                durations = df_bonds[duration_col].dropna().values
                
                if len(durations) > 0:
                    # Duration histogram
                    fig.add_trace(
                        go.Histogram(
                            x=durations,
                            nbinsx=min(20, max(5, len(durations) // 3)),
                            name='Duration Distribution',
                            showlegend=False,
                            marker_color=THEME['accent2'],
                            opacity=0.7,
                            hovertemplate='Duration: %{x:.1f} years<br>Count: %{y}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
                    # Calculate duration statistics
                    mean_dur = np.mean(durations)
                    median_dur = np.median(durations)
                    std_dur = np.std(durations)
                    
                    # Portfolio duration calculation
                    if 'Weight' in df_bonds.columns and not df_bonds['Weight'].isna().all():
                        weights = df_bonds['Weight'].fillna(0).values[:len(durations)]
                        if np.sum(weights) > 0:
                            weights = weights / np.sum(weights)
                            portfolio_duration = np.sum(weights * durations)
                            weight_info = f"Weighted Duration: {portfolio_duration:.2f}y"
                        else:
                            portfolio_duration = mean_dur
                            weight_info = f"Equal Weight Duration: {portfolio_duration:.2f}y"
                    else:
                        portfolio_duration = mean_dur
                        weight_info = f"Average Duration: {portfolio_duration:.2f}y"
                    
                    # Interest rate sensitivity analysis
                    rate_shocks = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
                    price_changes = -portfolio_duration * rate_shocks
                    
                    fig.add_trace(
                        go.Scatter(
                            x=rate_shocks,
                            y=price_changes,
                            mode='lines+markers',
                            name='Price Sensitivity',
                            line=dict(color=THEME['accent'], width=3),
                            marker=dict(size=8, color=THEME['accent']),
                            showlegend=False,
                            hovertemplate='Rate Change: %{x}%<br>Price Change: %{y:.2f}%<extra></extra>'
                        ),
                        row=1, col=2
                    )
                    
                    # Add reference lines
                    fig.add_hline(y=0, line=dict(color='white', width=1, dash='dash'), row=1, col=2)
                    fig.add_vline(x=0, line=dict(color='white', width=1, dash='dash'), row=1, col=2)
                    
                    # Add annotations
                    fig.add_annotation(
                        x=0.23, y=0.95, xref="paper", yref="paper",
                        text=f"Stats:<br>Mean: {mean_dur:.2f}y<br>Median: {median_dur:.2f}y<br>Std: {std_dur:.2f}y",
                        showarrow=False, align="left",
                        font=dict(size=10, color=THEME['text'])
                    )
                    
                    fig.add_annotation(
                        x=0.73, y=0.95, xref="paper", yref="paper",
                        text=f"{weight_info}<br>Sensitivity:<br>+100bp: {-portfolio_duration:.2f}%<br>-100bp: {portfolio_duration:.2f}%",
                        showarrow=False, align="left",
                        font=dict(size=10, color=THEME['text'])
                    )
                    
                else:
                    fig.add_annotation(
                        text="No valid duration data found",
                        xref="paper", yref="paper", x=0.5, y=0.5,
                        showarrow=False, font=dict(size=14, color=THEME['muted'])
                    )
            else:
                fig.add_annotation(
                    text="Duration column not found in bond data",
                    xref="paper", yref="paper", x=0.5, y=0.5,
                    showarrow=False, font=dict(size=14, color=THEME['muted'])
                )
        else:
            fig.add_annotation(
                text="Bond data not available",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16, color=THEME['muted'])
            )
            
    except Exception as e:
        fig.add_annotation(
            text=f"Bond analysis error: {str(e)[:60]}...",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#ff4444")
        )
    
    # Update layout
    fig.update_layout(
        height=500,
        title=dict(
            text="Bond Portfolio: Duration Risk Analysis",
            font=dict(size=16, color=THEME['accent']),
            x=0.5
        ),
        paper_bgcolor=THEME['bg'],
        plot_bgcolor=THEME['panel'],
        font=dict(color=THEME['text'])
    )
    
    # Update axes
    fig.update_xaxes(title_text="Duration (Years)", gridcolor=THEME['grid'], color=THEME['text'], row=1, col=1)
    fig.update_yaxes(title_text="Number of Bonds", gridcolor=THEME['grid'], color=THEME['text'], row=1, col=1)
    fig.update_xaxes(title_text="Interest Rate Change (%)", gridcolor=THEME['grid'], color=THEME['text'], row=1, col=2)
    fig.update_yaxes(title_text="Portfolio Price Change (%)", gridcolor=THEME['grid'], color=THEME['text'], row=1, col=2)
    
    return fig


def clear_risk_fig_caches() -> None:
    """Clear all risk figure caches."""
    make_risk_dashboard_figure.cache_clear()
    make_correlation_heatmap_figure.cache_clear()
    make_bond_duration_risk_figure.cache_clear()