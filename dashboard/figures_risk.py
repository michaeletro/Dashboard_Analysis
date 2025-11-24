"""
Risk Management Dashboard Figures
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import lru_cache

from ..config import THEME, style_figure
from .. import data


@lru_cache(maxsize=32)
def make_risk_dashboard_figure() -> go.Figure:
    """Create comprehensive risk management dashboard."""
    data.ensure_data_loaded()
    df_equity = data.df_equity
    df_bonds = data.df_bonds
    
    # Create 2x2 subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Value at Risk Analysis', 
            'Factor Risk Attribution',
            'Sector Concentration Risk',
            'Portfolio Stress Tests'
        ],
        specs=[
            [{'type': 'histogram'}, {'type': 'bar'}],
            [{'type': 'bar'}, {'type': 'scatter'}]
        ]
    )
    
    try:
        # 1. VaR Analysis using 1M returns
        if '1M % Change' in df_equity.columns:
            returns = df_equity['1M % Change'].values
            returns = returns[~np.isnan(returns)]  # Remove NaN values
            
            if len(returns) > 0:
                # Historical VaR (5%)
                var_95 = np.percentile(returns, 5)
                
                # Create histogram of returns
                fig.add_trace(
                    go.Histogram(
                        x=returns,
                        nbinsx=30,
                        name='Return Distribution',
                        showlegend=False,
                        marker_color=THEME['accent2'],
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                
                # Add VaR line
                fig.add_vline(
                    x=var_95,
                    line=dict(color='red', width=2, dash='dash'),
                    annotation_text=f'95% VaR: {var_95:.2f}%',
                    row=1, col=1
                )
        
        # 2. Factor Risk Attribution
        factor_cols = [col for col in df_equity.columns if col.endswith('Score')]
        if len(factor_cols) >= 3:
            # Calculate portfolio factor exposures (equal weight)
            portfolio_weights = np.ones(len(df_equity)) / len(df_equity)
            factor_exposures = []
            factor_names = []
            
            for factor in factor_cols[:4]:  # Top 4 factors
                exposure = np.sum(portfolio_weights * df_equity[factor].values)
                factor_exposures.append(exposure)
                factor_names.append(factor.replace('Score', ''))
            
            fig.add_trace(
                go.Bar(
                    x=factor_names,
                    y=factor_exposures,
                    name='Factor Exposures',
                    showlegend=False,
                    marker_color=THEME['accent']
                ),
                row=1, col=2
            )
        
        # 3. Sector Concentration
        if 'Sector' in df_equity.columns:
            sector_counts = df_equity['Sector'].value_counts()
            sector_weights = sector_counts / len(df_equity)
            
            # Calculate HHI (Herfindahl-Hirschman Index)
            hhi = (sector_weights ** 2).sum()
            
            fig.add_trace(
                go.Bar(
                    x=sector_weights.index[:8],  # Top 8 sectors
                    y=sector_weights.values[:8] * 100,  # Convert to percentage
                    name='Sector Weights',
                    showlegend=False,
                    marker_color=THEME['accent2'],
                    text=[f'{w:.1f}%' for w in sector_weights.values[:8] * 100],
                    textposition='auto'
                ),
                row=2, col=1
            )
        
        # 4. Stress Test Scenarios
        if '1M % Change' in df_equity.columns:
            base_return = np.mean(df_equity['1M % Change'].values)
            base_vol = np.std(df_equity['1M % Change'].values)
            
            scenarios = {
                'Normal': base_return,
                'Market Crash (-30%)': base_return - 30,
                'High Vol (2x)': base_return - 2 * base_vol,
                'Recession': base_return - 15,
                'Black Swan': base_return - 50
            }
            
            scenario_names = list(scenarios.keys())
            scenario_returns = list(scenarios.values())
            colors = ['green' if r >= 0 else 'red' for r in scenario_returns]
            
            fig.add_trace(
                go.Scatter(
                    x=scenario_names,
                    y=scenario_returns,
                    mode='markers+text',
                    marker=dict(
                        size=[15 if abs(r) < 10 else 20 + abs(r)/2 for r in scenario_returns],
                        color=colors,
                        opacity=0.8
                    ),
                    text=[f'{r:.1f}%' for r in scenario_returns],
                    textposition='top center',
                    name='Stress Scenarios',
                    showlegend=False
                ),
                row=2, col=2
            )
            
    except Exception as e:
        # Add error message if calculations fail
        fig.add_annotation(
            text=f"Risk calculations unavailable: {str(e)[:50]}...",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color=THEME['text'])
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title=dict(
            text="Portfolio Risk Management Dashboard",
            font=dict(size=18, color=THEME['text']),
            x=0.5
        ),
        paper_bgcolor=THEME['bg'],
        plot_bgcolor=THEME['panel'],
        font=dict(color=THEME['text'])
    )
    
    # Update axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(
                gridcolor=THEME['grid'],
                color=THEME['text'],
                row=i, col=j
            )
            fig.update_yaxes(
                gridcolor=THEME['grid'],
                color=THEME['text'],
                row=i, col=j
            )
    
    return fig


@lru_cache(maxsize=32)  
def make_correlation_heatmap_figure() -> go.Figure:
    """Create correlation heatmap for risk factors."""
    data.ensure_data_loaded()
    df_equity = data.df_equity
    
    fig = go.Figure()
    
    try:
        # Get factor scores and key metrics
        factor_cols = [col for col in df_equity.columns if col.endswith('Score')]
        key_metrics = ['P/E (TTM)', 'P/B (TTM)', 'ROE (%)', 'Debt/Equity', '1M % Change']
        
        # Combine available columns
        analysis_cols = []
        for col in factor_cols + key_metrics:
            if col in df_equity.columns:
                analysis_cols.append(col)
        
        if len(analysis_cols) >= 3:
            # Calculate correlation matrix
            corr_data = df_equity[analysis_cols].corr()
            
            # Create heatmap
            fig.add_trace(
                go.Heatmap(
                    z=corr_data.values,
                    x=corr_data.columns,
                    y=corr_data.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_data.round(2).values,
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False,
                    colorbar=dict(
                        title="Correlation",
                        titlefont=dict(color=THEME['text']),
                        tickfont=dict(color=THEME['text'])
                    )
                )
            )
        else:
            fig.add_annotation(
                text="Insufficient data for correlation analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color=THEME['text'])
            )
            
    except Exception as e:
        fig.add_annotation(
            text=f"Correlation analysis failed: {str(e)[:50]}...",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color=THEME['text'])
        )
    
    fig = style_figure(
        fig,
        height=600,
        title="Factor & Metric Correlation Matrix"
    )
    
    return fig


@lru_cache(maxsize=32)
def make_bond_duration_risk_figure() -> go.Figure:
    """Create bond duration risk analysis figure."""
    data.ensure_data_loaded()
    df_bonds = data.df_bonds
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Duration Distribution', 'Interest Rate Sensitivity'],
        specs=[[{'type': 'histogram'}, {'type': 'scatter'}]]
    )
    
    try:
        if df_bonds is not None and 'Duration' in df_bonds.columns:
            durations = df_bonds['Duration'].values
            durations = durations[~np.isnan(durations)]
            
            if len(durations) > 0:
                # 1. Duration histogram
                fig.add_trace(
                    go.Histogram(
                        x=durations,
                        nbinsx=20,
                        name='Duration Distribution',
                        showlegend=False,
                        marker_color=THEME['accent2'],
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                
                # 2. Interest rate sensitivity analysis
                rate_shocks = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])  # Rate changes in %
                
                # Calculate portfolio duration if weights available
                if 'Weight' in df_bonds.columns:
                    weights = df_bonds['Weight'].values
                    portfolio_duration = np.sum(weights * durations)
                else:
                    portfolio_duration = np.mean(durations)
                
                # Price changes for different rate scenarios
                price_changes = -portfolio_duration * rate_shocks
                
                fig.add_trace(
                    go.Scatter(
                        x=rate_shocks,
                        y=price_changes,
                        mode='lines+markers',
                        name='Portfolio Price Change',
                        line=dict(color=THEME['accent'], width=3),
                        marker=dict(size=8),
                        showlegend=False,
                        hovertemplate='Rate Change: %{x}%<br>Price Change: %{y:.2f}%<extra></extra>'
                    ),
                    row=1, col=2
                )
                
                # Add zero line
                fig.add_hline(
                    y=0, line=dict(color='white', width=1, dash='dash'),
                    row=1, col=2
                )
                
        else:
            fig.add_annotation(
                text="Bond duration data not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color=THEME['text'])
            )
            
    except Exception as e:
        fig.add_annotation(
            text=f"Bond analysis failed: {str(e)[:50]}...",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color=THEME['text'])
        )
    
    # Update layout
    fig.update_layout(
        height=500,
        title=dict(
            text="Bond Duration Risk Analysis",
            font=dict(size=16, color=THEME['text']),
            x=0.5
        ),
        paper_bgcolor=THEME['bg'],
        plot_bgcolor=THEME['panel'],
        font=dict(color=THEME['text'])
    )
    
    # Update axes
    fig.update_xaxes(
        title_text="Duration (Years)",
        gridcolor=THEME['grid'],
        color=THEME['text'],
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Frequency",
        gridcolor=THEME['grid'],
        color=THEME['text'],
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="Interest Rate Change (%)",
        gridcolor=THEME['grid'],
        color=THEME['text'],
        row=1, col=2
    )
    fig.update_yaxes(
        title_text="Portfolio Price Change (%)",
        gridcolor=THEME['grid'],
        color=THEME['text'],
        row=1, col=2
    )
    
    return fig


def clear_risk_fig_caches() -> None:
    """Clear all risk figure caches."""
    make_risk_dashboard_figure.cache_clear()
    make_correlation_heatmap_figure.cache_clear()
    make_bond_duration_risk_figure.cache_clear()