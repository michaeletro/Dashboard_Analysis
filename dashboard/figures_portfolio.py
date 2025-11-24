from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from functools import lru_cache
from .config import THEME, style_figure
from . import data


@lru_cache(maxsize=128)
def make_frontier_figure(n_points: int, show_tangency: bool) -> go.Figure:
    """
    Efficient frontier in annualised space, with optional tangency point.
    Uses the engine annualised mean and covariance for consistency.
    """
    data.ensure_data_loaded()
    engine = data.engine
    # frontier in annualised space
    r, vol, _ = engine.efficient_frontier(n_points=n_points, annualised=True)

    fig = go.Figure()

    # efficient frontier line
    fig.add_trace(
        go.Scatter(
            x=vol,
            y=r,
            mode="lines",
            name="efficient frontier",
            line=dict(width=3, color=THEME["accent2"]),
            hovertemplate="Vol: %{x:.4f}<br>Return: %{y:.4f}<extra></extra>",
        )
    )

    # individual assets (annualised)
    mu_assets = engine.mu_annualised
    Sigma_assets = engine.Sigma_annualised
    asset_vols = np.sqrt(np.diag(Sigma_assets))

    fig.add_trace(
        go.Scatter(
            x=asset_vols,
            y=mu_assets,
            mode="markers+text",
            name="assets",
            text=engine.assets,
            textposition="top center",
            marker=dict(size=8, color=THEME["muted"]),
            hovertemplate=(
                "Asset: %{text}<br>"
                "Vol: %{x:.4f}<br>"
                "Return: %{y:.4f}<extra></extra>"
            ),
        )
    )

    # tangency portfolio point
    if show_tangency:
        if engine.w_tangency_ is None:
            engine.tangency_weights()

        w = engine.w_tangency_
        mu_p, vol_p, _ = engine.portfolio_risk_return(
            w,
            annualised=True,
        )

        fig.add_trace(
            go.Scatter(
                x=[vol_p],
                y=[mu_p],
                mode="markers",
                marker=dict(size=11, color=THEME["accent"], symbol="star"),
                name="tangency",
                hovertemplate=(
                    "Tangency portfolio<br>"
                    "Vol: %{x:.4f}<br>"
                    "Return: %{y:.4f}<extra></extra>"
                ),
            )
        )

    fig = style_figure(
        fig,
        height=430,
        title="Efficient frontier",
    )
    fig.update_layout(
        xaxis_title="Volatility annual",
        yaxis_title="Expected return annual",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    return fig


@lru_cache(maxsize=64)
def make_paths_figure(
    n_steps: int,
    n_paths: int,
    portfolio_type: str,
    max_paths: int,
) -> go.Figure:
    """
    Simulated forward portfolio paths starting from current prices,
    normalised so that t = 0 is roughly 1 for all paths.
    
    Args:
        portfolio_type: 'tangency', 'equal', or 'black_litterman'
    """
    data.ensure_data_loaded()
    engine = data.engine
    
    # choose weights based on portfolio type
    if portfolio_type == "tangency":
        if engine.w_tangency_ is None:
            engine.tangency_weights()
        w = engine.w_tangency_
        title_suffix = "Tangency Portfolio"
    elif portfolio_type == "equal":
        w = np.ones(engine.n_assets, dtype=float) / engine.n_assets
        title_suffix = "Equal Weight Portfolio"
    elif portfolio_type == "black_litterman":
        if data.w_bl is None:
            # Try to compute Black-Litterman weights if not available
            try:
                data.ensure_data_loaded()  # This should compute BL weights
                w = data.w_bl
                if w is None:
                    raise ValueError("Black-Litterman weights unavailable")
            except Exception:
                # Fall back to tangency if BL fails
                if engine.w_tangency_ is None:
                    engine.tangency_weights()
                w = engine.w_tangency_
                title_suffix = "Tangency Portfolio (BL fallback)"
            else:
                title_suffix = "Black-Litterman Portfolio"
        else:
            w = data.w_bl
            title_suffix = "Black-Litterman Portfolio"
    else:
        # Default to equal weight for unknown types
        w = np.ones(engine.n_assets, dtype=float) / engine.n_assets
        title_suffix = "Equal Weight Portfolio"

    # start from last observed prices
    S0 = engine.level.iloc[-1].values

    # simulate normalised paths
    V = engine.simulate_portfolio_paths(
        weights=w,
        n_steps=n_steps,
        n_paths=n_paths,
        S0=S0,
    )

    n_paths_sim, n_steps_plus_1 = V.shape
    x_vals = np.arange(n_steps_plus_1)

    fig = go.Figure()

    # subset of paths to plot
    n_plot = min(n_paths_sim, max_paths)
    idx = np.linspace(0, n_paths_sim - 1, n_plot, dtype=int)

    for pid in idx:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=V[pid, :],
                mode="lines",
                line=dict(width=1, color="rgba(120, 140, 255, 0.15)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # mean simulated path
    mean_path = V.mean(axis=0)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=mean_path,
            mode="lines",
            line=dict(width=3, color=THEME["accent"]),
            name="mean simulated",
            hovertemplate="Step: %{x}<br>Value: %{y:.4f}<extra></extra>",
        )
    )

    fig = style_figure(
        fig,
        height=430,
        title=f"Simulated portfolio paths - {title_suffix}",
    )
    fig.update_layout(
        xaxis_title="Time step",
        yaxis_title="Portfolio value normalised",
    )
    return fig


@lru_cache(maxsize=32)
def make_tangency_levels_figure(
    n_paths: int = 500,
    max_paths_plot: int = 50,
) -> go.Figure:
    """
    Historical tangency portfolio level versus GBM simulations mapped to level space,
    rendered using StochasticPortfolioEngine.plot_portfolio_paths_2d so we get:

      - simulated paths
      - mean path
      - golden run path
      - realised historical path
      - horizon return distribution
    """
    data.ensure_data_loaded()
    engine = data.engine
    w_tan = data.w_tan
    # realised tangency path in level terms
    real_path_level = engine.historical_portfolio_path(w_tan, normalise=False)
    base_level = float(real_path_level[0])

    # same horizon as history
    n_steps = len(real_path_level) - 1

    # simulate normalised GBM portfolio paths
    V_norm = engine.simulate_portfolio_paths(
        weights=w_tan,
        n_steps=n_steps,
        n_paths=n_paths,
    )

    # map to level units
    V_level = engine.apply_growth_to_level(V_norm, base_level=base_level)

    time_idx = engine.level.index[: V_level.shape[1]]

    # use the engine 2D plotting method to build the full subplot figure
    fig = engine.plot_portfolio_paths_2d(
        V=V_level,
        time_index=time_idx,
        max_paths=max_paths_plot,
        show_mean=True,
        real_path=real_path_level,
        real_label="historical level",
        title="Tangency portfolio simulated vs historical levels",
        golden_run=True,
        plot_portfolio=False,     # do not overlay individual asset levels here
        normalise_at_start=False, # we are already in level space
        plot_nth_percentile=0.9,
        show=False,               # critical for Dash: do not call fig.show()
    )

    # optional: harmonise with the app theme
    fig.update_layout(
        paper_bgcolor=THEME["card"],
        plot_bgcolor=THEME["card"],
        font=dict(color=THEME["text"]),
        height=430,
    )

    # y axis for left panel is portfolio level now
    fig.update_yaxes(
        title_text="Portfolio level",
        row=1,
        col=1,
    )
    # y axis for right panel remains horizon return
    fig.update_yaxes(
        title_text="Horizon return",
        row=1,
        col=2,
    )

    return fig
@lru_cache(maxsize=64)
def make_portfolio_paths_3d_figure(
    port_type: str,
    n_steps: int,
    n_paths: int,
    max_paths: int,
) -> go.Figure:
    """
    Three dimensional view of simulated portfolio paths, using the same engine
    simulation logic (normalised paths).
    """
    data.ensure_data_loaded()
    engine = data.engine
    w_tan = data.w_tan
    w_bl = data.w_bl
    if port_type == "tangency":
        w = w_tan
        title_prefix = "Tangency"
    elif port_type == "black_litterman":
        if w_bl is None:
            from .figures_cross_section import make_cluster_placeholder

            return make_cluster_placeholder(
                "Black Litterman weights unavailable. "
                "Ensure black_litterman_tangency_weights is implemented."
            )
        w = w_bl
        title_prefix = "Black Litterman"
    else:
        # default fallback
        w = w_tan
        title_prefix = "Tangency"

    # simulate normalised portfolio paths
    V = engine.simulate_portfolio_paths(
        weights=w,
        n_steps=n_steps,
        n_paths=n_paths,
    )

    n_paths_sim, n_steps_plus_1 = V.shape
    time_idx = engine.returns.index[: n_steps_plus_1]

    fig = go.Figure()

    n_plot = min(n_paths_sim, max_paths)
    idx = np.linspace(0, n_paths_sim - 1, n_plot, dtype=int)

    for j, pid in enumerate(idx):
        fig.add_trace(
            go.Scatter3d(
                x=time_idx,
                y=np.full(n_steps_plus_1, pid),
                z=V[pid, :],
                mode="lines",
                line=dict(width=1),
                opacity=0.55,
                showlegend=(j == 0),
                name="simulated paths" if j == 0 else None,
            )
        )

    fig.update_layout(
        title=dict(
            text=f"{title_prefix} portfolio value paths 3D",
            font=dict(size=14, color=THEME["accent"]),
            x=0.5,  # Center the title
            y=0.98,  # Position near top but with space
            xanchor='center',
            yanchor='top'
        ),
        scene=dict(
            xaxis=dict(
                title="Time",
                backgroundcolor=THEME["panel"],
                gridcolor=THEME["grid"],
                zeroline=False,
                color=THEME["muted"],
            ),
            yaxis=dict(
                title="Path id",
                backgroundcolor=THEME["panel"],
                gridcolor=THEME["grid"],
                zeroline=False,
                color=THEME["muted"],
            ),
            zaxis=dict(
                title="Portfolio value normalised",
                backgroundcolor=THEME["panel"],
                gridcolor=THEME["grid"],
                zeroline=False,
                color=THEME["muted"],
            ),
            bgcolor=THEME["card"],
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)  # Better default viewing angle
            )
        ),
        margin=dict(l=10, r=10, b=10, t=60),  # Increased top margin for title
        height=480,  # Slightly increased height to accommodate title
        paper_bgcolor=THEME["card"],
        font=dict(color=THEME["text"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.02,  # Position legend at bottom
            xanchor="left",
            x=0,
            bgcolor="rgba(0, 0, 0, 0.7)",
            bordercolor=THEME["border"],
            borderwidth=1
        ),
        showlegend=True,
    )

    return fig


def clear_fig_caches() -> None:
    """Clear cached figure results (used after data reset)."""
    make_frontier_figure.cache_clear()
    make_paths_figure.cache_clear()
    make_tangency_levels_figure.cache_clear()
    make_portfolio_paths_3d_figure.cache_clear()
