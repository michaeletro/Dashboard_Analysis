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
    
    # Determine maturity column
    maturity_col = None
    for col in ['Maturity_Years', 'Duration']:
        if col in df_bonds.columns and not df_bonds[col].isna().all():
            maturity_col = col
            break
    
    if maturity_col is None:
        # No valid maturity data, show error
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="No maturity data available", showarrow=False,
            font=dict(size=16, color=THEME['text_color'])
        )
        fig.update_layout(
            title="Bond Yield Curve",
            xaxis_title="Maturity",
            yaxis_title="Yield (%)"
        )
        return style_figure(fig, title="Bond Yield Curve")
    
    # Get yield column
    yield_col = 'Yield' if 'Yield' in df_bonds.columns else None
    if yield_col is None:
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="No yield data available", showarrow=False,
            font=dict(size=16, color=THEME['text_color'])
        )
        fig.update_layout(
            title="Bond Yield Curve",
            xaxis_title="Maturity", 
            yaxis_title="Yield (%)"
        )
        return style_figure(fig, title="Bond Yield Curve")
    
    # Color by credit spread if available, otherwise use duration
    color_col = None
    color_title = "Value"
    if 'Credit_Spread' in df_bonds.columns and not df_bonds['Credit_Spread'].isna().all():
        color_col = 'Credit_Spread'
        color_values = df_bonds[color_col] * 10000  # Convert to bps
        color_title = "Credit Spread (bps)"
    elif 'Duration' in df_bonds.columns:
        color_col = 'Duration'
        color_values = df_bonds[color_col]
        color_title = "Duration (Years)"
    else:
        color_values = df_bonds.index  # Use index as fallback
        color_title = "Bond Index"
    
    # Scatter plot of individual bonds
    fig.add_trace(
        go.Scatter(
            x=df_bonds[maturity_col],
            y=df_bonds[yield_col] * 100,  # Convert to percentage
            mode='markers',
            marker=dict(
                size=8,
                color=color_values,
                colorscale='Viridis',
                colorbar=dict(title=color_title, x=1.02),
                showscale=True
            ),
            text=df_bonds.get('Rating', 'N/A'),
            hovertemplate=(
                f'{maturity_col}: %{{x:.1f}}<br>'
                'Yield: %{y:.2f}%<br>'
                'Rating: %{text}<br>'
                f'{color_title}: %{{marker.color:.1f}}<extra></extra>'
            ),
            name='Bonds'
        )
    )
    
        # Add fitted yield curve if we have sufficient data
    if len(df_bonds) > 5:
        try:
            # Fit curve to actual bond data
            maturities_for_fit = np.asarray(df_bonds[maturity_col].values, dtype=float)
            yields_for_fit = np.asarray(df_bonds[yield_col].values, dtype=float) * 100  # Convert to %
            
            # Remove any NaN values
            valid_mask = ~(np.isnan(maturities_for_fit) | np.isnan(yields_for_fit))
            if valid_mask.sum() > 3:  # Need at least 3 points
                mat_clean = maturities_for_fit[valid_mask]
                yield_clean = yields_for_fit[valid_mask]
                
                # Try Nelson-Siegel approximation first, fall back to polynomial
                try:
                    # Simple Nelson-Siegel approximation without optimization
                    beta0 = yield_clean[mat_clean > 10].mean() if (mat_clean > 10).any() else yield_clean.mean()
                    beta1 = yield_clean[mat_clean < 2].mean() - beta0 if (mat_clean < 2).any() else 0
                    beta2 = 0  # Simplified
                    lam = 1.0  # Fixed decay parameter
                    
                    # Generate smooth curve
                    maturities_smooth = np.linspace(mat_clean.min(), mat_clean.max(), 100)
                    exp_term = np.exp(-lam * maturities_smooth)
                    factor1 = (1 - exp_term) / (lam * maturities_smooth + 1e-6)  # Avoid division by zero
                    factor2 = factor1 - exp_term
                    fitted_curve = beta0 + beta1 * factor1 + beta2 * factor2
                    
                    curve_name = 'Nelson-Siegel Fit'
                except:
                    # Fallback to polynomial fit
                    coeffs = np.polyfit(mat_clean, yield_clean, min(2, len(mat_clean)-1))
                    maturities_smooth = np.linspace(mat_clean.min(), mat_clean.max(), 100)
                    fitted_curve = np.polyval(coeffs, maturities_smooth)
                    curve_name = 'Polynomial Fit'
                
                fig.add_trace(
                    go.Scatter(
                        x=maturities_smooth,
                        y=fitted_curve,
                        mode='lines',
                        line=dict(color=THEME['accent'], width=3, dash='dash'),
                        name=curve_name,
                        hovertemplate=f'Maturity: %{{x:.1f}}Y<br>{curve_name}: %{{y:.2f}}%<extra></extra>'
                    )
                )
        except Exception as e:
            # Skip curve fitting if it fails
            print(f"Yield curve fitting failed: {e}")
    
    fig = style_figure(fig, height=450, title='Yield Curve Analysis')
    fig.update_layout(
        xaxis_title=f'{maturity_col.replace("_", " ")}',
        yaxis_title='Yield (%)',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
    )
    
    return fig


def make_duration_distribution_figure() -> go.Figure:
    """Create duration distribution and risk analysis."""
    data.ensure_data_loaded()
    
    df_bonds = data.df_bonds
    duration_stats = data.bond_duration_stats
    
    if 'Duration' not in df_bonds.columns:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="No duration data available", showarrow=False,
            font=dict(size=16, color=THEME['text_color'])
        )
        return style_figure(fig, title="Duration Distribution Analysis")
    
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
    
    # Add vertical line for portfolio duration if available
    if duration_stats and 'portfolio_duration' in duration_stats:
        port_duration = duration_stats['portfolio_duration']
        fig.add_vline(
            x=port_duration,
            line=dict(color=THEME['accent'], width=2, dash='dash'),
            annotation_text=f'Portfolio: {port_duration:.2f}',
            row=1, col=1
        )
    
    # Duration vs Yield scatter (if yield data available)
    if 'Yield' in df_bonds.columns:
        # Determine color column
        color_col = None
        color_title = "Value"
        if 'Weight' in df_bonds.columns:
            color_values = df_bonds['Weight'] * 100
            color_title = "Weight (%)"
        elif 'Credit_Spread' in df_bonds.columns:
            color_values = df_bonds['Credit_Spread'] * 10000
            color_title = "Credit Spread (bps)"
        else:
            color_values = df_bonds.index
            color_title = "Bond Index"
        
        fig.add_trace(
            go.Scatter(
                x=df_bonds['Duration'],
                y=df_bonds['Yield'] * 100,
                mode='markers',
                marker=dict(
                    size=8,
                    color=color_values,
                    colorscale='Plasma',
                    showscale=False
                ),
                text=df_bonds.get('Rating', 'N/A'),
                hovertemplate=(
                    'Duration: %{x:.2f}<br>'
                    'Yield: %{y:.2f}%<br>'
                    f'{color_title}: %{{marker.color:.1f}}<br>'
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
    
    fig = style_figure(fig, height=450, title='Duration Risk Analysis')
    
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
                y=rating_data['1M_Yield_Change'] * 100,  # Convert % to bps
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