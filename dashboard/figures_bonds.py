from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .config import THEME, style_figure
from . import data


def make_yield_curve_figure() -> go.Figure:
    """Create yield curve visualization showing maturity vs yield."""
    data.ensure_data_loaded()
    
    df_bonds = data.df_bonds
    
    fig = go.Figure()
    
    # Scatter plot of individual bonds
    fig.add_trace(
        go.Scatter(
            x=df_bonds['Maturity_Years'],
            y=df_bonds['Yield'] * 100,  # Convert to percentage
            mode='markers',
            marker=dict(
                size=8,
                color=df_bonds['Credit_Spread'] * 100,
                colorscale='Viridis',
                colorbar=dict(title='Credit Spread (bps)', x=1.02),
                showscale=True
            ),
            text=df_bonds['Rating'],
            hovertemplate=(
                'Maturity: %{x:.1f}Y<br>'
                'Yield: %{y:.2f}%<br>'
                'Rating: %{text}<br>'
                'Credit Spread: %{marker.color:.0f}bps<extra></extra>'
            ),
            name='Bonds'
        )
    )
    
    # Add fitted yield curve
    maturities_smooth = np.linspace(df_bonds['Maturity_Years'].min(), 
                                   df_bonds['Maturity_Years'].max(), 100)
    # Simple yield curve fit: y = a + b*(1 - exp(-c*x))
    treasury_curve = 2.0 + 2.5 * (1 - np.exp(-maturities_smooth / 5))
    
    fig.add_trace(
        go.Scatter(
            x=maturities_smooth,
            y=treasury_curve,
            mode='lines',
            line=dict(color=THEME['accent'], width=3, dash='dash'),
            name='Treasury Curve',
            hovertemplate='Maturity: %{x:.1f}Y<br>Treasury Yield: %{y:.2f}%<extra></extra>'
        )
    )
    
    fig = style_figure(fig, height=450, title='Yield Curve Analysis')
    fig.update_layout(
        xaxis_title='Maturity (Years)',
        yaxis_title='Yield (%)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
    )
    
    return fig


def make_duration_distribution_figure() -> go.Figure:
    """Create duration distribution and risk analysis."""
    data.ensure_data_loaded()
    
    df_bonds = data.df_bonds
    duration_stats = data.bond_duration_stats
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Duration Distribution', 'Duration vs Yield'],
        specs=[[{'secondary_y': False}, {'secondary_y': False}]]
    )
    
    # Duration histogram
    fig.add_trace(
        go.Histogram(
            x=df_bonds['Duration'],
            nbinsx=20,
            marker=dict(color=THEME['accent2'], opacity=0.7),
            name='Duration Hist',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add vertical line for portfolio duration
    port_duration = duration_stats['portfolio_duration']
    fig.add_vline(
        x=port_duration,
        line=dict(color=THEME['accent'], width=2, dash='dash'),
        annotation_text=f'Portfolio: {port_duration:.2f}',
        row=1, col=1
    )
    
    # Duration vs Yield scatter
    fig.add_trace(
        go.Scatter(
            x=df_bonds['Duration'],
            y=df_bonds['Yield'] * 100,
            mode='markers',
            marker=dict(
                size=8,
                color=df_bonds['Weight'] * 100,
                colorscale='Plasma',
                showscale=False
            ),
            text=df_bonds['Rating'],
            hovertemplate=(
                'Duration: %{x:.2f}<br>'
                'Yield: %{y:.2f}%<br>'
                'Weight: %{marker.color:.2f}%<br>'
                'Rating: %{text}<extra></extra>'
            ),
            name='Bonds',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Style subplots
    fig.update_xaxes(title_text='Duration', row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    fig.update_xaxes(title_text='Duration', row=1, col=2)
    fig.update_yaxes(title_text='Yield (%)', row=1, col=2)
    
    fig.update_layout(
        height=450,
        title=dict(text='Duration Risk Analysis', font=dict(size=16, color=THEME['accent'])),
        paper_bgcolor=THEME['card'],
        plot_bgcolor=THEME['panel'],
        font=dict(color=THEME['text']),
        showlegend=False
    )
    
    return fig


def make_credit_analysis_figure() -> go.Figure:
    """Create credit quality and sector analysis charts."""
    data.ensure_data_loaded()
    
    df_bonds = data.df_bonds
    credit_analysis = data.bond_credit_analysis
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Rating Distribution', 'Sector Allocation', 
                       'Credit Spreads by Rating', 'Yield vs Credit Spread'],
        specs=[[{'type': 'bar'}, {'type': 'pie'}],
               [{'type': 'bar'}, {'type': 'scatter'}]]
    )
    
    # Rating distribution
    if not credit_analysis['rating_distribution'].empty:
        ratings = credit_analysis['rating_distribution'].index
        counts = credit_analysis['rating_distribution'].values
        
        fig.add_trace(
            go.Bar(
                x=ratings,
                y=counts,
                marker=dict(color=THEME['accent2']),
                name='Rating Count',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Sector pie chart
    sector_weights = df_bonds.groupby('Sector')['Weight'].sum()
    fig.add_trace(
        go.Pie(
            labels=sector_weights.index,
            values=sector_weights.values,
            name='Sector Weight',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Credit spreads by rating
    spread_by_rating = df_bonds.groupby('Rating')['Credit_Spread'].mean() * 10000  # Convert to bps
    fig.add_trace(
        go.Bar(
            x=spread_by_rating.index,
            y=spread_by_rating.values,
            marker=dict(color=THEME['accent']),
            name='Avg Spread (bps)',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Yield vs Credit Spread
    fig.add_trace(
        go.Scatter(
            x=df_bonds['Credit_Spread'] * 10000,  # bps
            y=df_bonds['Yield'] * 100,  # percentage
            mode='markers',
            marker=dict(
                size=8,
                color=df_bonds['Duration'],
                colorscale='Viridis',
                showscale=False
            ),
            text=df_bonds['Rating'],
            hovertemplate=(
                'Credit Spread: %{x:.0f}bps<br>'
                'Yield: %{y:.2f}%<br>'
                'Duration: %{marker.color:.2f}<br>'
                'Rating: %{text}<extra></extra>'
            ),
            name='Bonds',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update subplot axes
    fig.update_xaxes(title_text='Rating', row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    fig.update_xaxes(title_text='Rating', row=2, col=1)
    fig.update_yaxes(title_text='Spread (bps)', row=2, col=1)
    fig.update_xaxes(title_text='Credit Spread (bps)', row=2, col=2)
    fig.update_yaxes(title_text='Yield (%)', row=2, col=2)
    
    fig.update_layout(
        height=600,
        title=dict(text='Credit Quality Analysis', font=dict(size=16, color=THEME['accent'])),
        paper_bgcolor=THEME['card'],
        plot_bgcolor=THEME['panel'],
        font=dict(color=THEME['text'])
    )
    
    return fig


def make_bond_performance_figure() -> go.Figure:
    """Create bond performance attribution analysis."""
    data.ensure_data_loaded()
    
    df_bonds = data.df_bonds
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Yield Changes (1M)', 'Portfolio Risk Decomposition'],
        specs=[[{'type': 'box'}, {'type': 'bar'}]]
    )
    
    # Box plot of yield changes by rating
    ratings = df_bonds['Rating'].unique()
    for rating in ratings:
        rating_data = df_bonds[df_bonds['Rating'] == rating]
        fig.add_trace(
            go.Box(
                y=rating_data['1M_Yield_Change'] * 100,  # Convert to bps
                name=rating,
                boxmean=True,
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Risk decomposition by sector
    sectors = df_bonds['Sector'].unique()
    sector_risk = []
    sector_names = []
    
    for sector in sectors:
        sector_bonds = df_bonds[df_bonds['Sector'] == sector]
        sector_duration = np.average(sector_bonds['Duration'], weights=sector_bonds['Weight'])
        sector_weight = sector_bonds['Weight'].sum()
        contribution = sector_duration * sector_weight
        sector_risk.append(contribution)
        sector_names.append(sector)
    
    fig.add_trace(
        go.Bar(
            x=sector_names,
            y=sector_risk,
            marker=dict(color=THEME['accent']),
            name='Duration Contribution',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text='Rating', row=1, col=1)
    fig.update_yaxes(title_text='Yield Change (bps)', row=1, col=1)
    fig.update_xaxes(title_text='Sector', row=1, col=2)
    fig.update_yaxes(title_text='Duration Contribution', row=1, col=2)
    
    fig.update_layout(
        height=450,
        title=dict(text='Bond Performance Analysis', font=dict(size=16, color=THEME['accent'])),
        paper_bgcolor=THEME['card'],
        plot_bgcolor=THEME['panel'],
        font=dict(color=THEME['text'])
    )
    
    return fig