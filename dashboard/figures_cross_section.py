from __future__ import annotations

from math import erf, sqrt

import numpy as np
import plotly.graph_objects as go

from .config import THEME, style_figure
from . import data


def make_cluster_placeholder(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=msg,
        showarrow=False,
        font=dict(color=THEME["accent"], size=13),
    )
    return style_figure(fig, height=420, title="Factor clustering (3D PCA space)")


def get_cluster_fig() -> go.Figure:
    data.ensure_data_loaded()
    df_equity = data.df_equity
    clusterer = data.clusterer
    raw_cluster_fig = clusterer.plot_clusters_3d(df_equity, show_plot=False)

    if raw_cluster_fig is None:
        factor_score_cols = [c for c in df_equity.columns if str(c).endswith("Score")]
        if len(factor_score_cols) < 3:
            msg = (
                "Cannot build 3D PCA cluster plot. "
                "Need at least three factor columns ending with 'Score'. "
                f"Found: {', '.join(factor_score_cols) or 'none'}"
            )
        else:
            msg = (
                "3D cluster figure unavailable. "
                "Check that scikit learn and plotly are installed "
                "and that FactorClusterer.fit_predict ran successfully."
            )
        return make_cluster_placeholder(msg)
    else:
        raw_cluster_fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        return raw_cluster_fig


def make_regression_figure() -> go.Figure:
    data.ensure_data_loaded()
    df_equity = data.df_equity
    y_actual = data.y_actual
    y_hat_all = data.y_hat_all
    y_hat_m = data.y_hat_m
    nmise_all = data.nmise_all
    mape_all = data.mape_all
    nmise_mom = data.nmise_mom
    mape_mom = data.mape_mom
    x_axis = df_equity.index if df_equity.index.is_unique else np.arange(len(df_equity))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=y_actual,
            mode="lines",
            name="Actual 1M % Change",
            line=dict(width=2, color=THEME["accent"]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=y_hat_all.flatten() * 100.0,
            mode="lines",
            name="Predicted all factors",
            line=dict(width=1.5, color=THEME["accent2"]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=y_hat_m.flatten() * 100.0,
            mode="lines",
            name="Predicted momentum only",
            line=dict(width=1.5, color="rgba(255, 80, 80, 0.8)"),
        )
    )

    subtitle = (
        f"All factors: MAPE {mape_all:.2f}%, NMISE {nmise_all:.4f} | "
        f"Momentum only: MAPE {mape_mom:.2f}%, NMISE {nmise_mom:.4f}"
    )

    fig = style_figure(
        fig,
        height=360,
        title=(
            "Cross sectional factor regression"
            "<br><span style='font-size:11px'>"
            f"{subtitle}</span>"
        ),
    )
    fig.update_layout(
        xaxis_title="Name or index",
        yaxis_title="1M % Change basis points",
    )
    return fig


 # compute on demand via function above


def make_peer_sector_figure() -> go.Figure:
    data.ensure_data_loaded()
    sector_summary = data.sector_summary
    if sector_summary is None:
        return make_cluster_placeholder("No sector summary available")
    if "Sector" in sector_summary.columns:
        sectors = sector_summary["Sector"].values
        numeric_cols = sector_summary.select_dtypes(include=[np.number]).columns
        metric_col = numeric_cols[0] if len(numeric_cols) > 0 else None
        if metric_col is None:
            y_vals = np.ones(len(sectors))
        else:
            y_vals = sector_summary[metric_col].values
    else:
        sectors = sector_summary.index
        numeric_cols = sector_summary.select_dtypes(include=[np.number]).columns
        metric_col = numeric_cols[0] if len(numeric_cols) > 0 else None
        if metric_col is None:
            y_vals = np.ones(len(sectors))
        else:
            y_vals = sector_summary[metric_col].values

    fig = go.Figure(
        data=go.Bar(
            x=sectors,
            y=y_vals,
            marker_color=THEME["accent"],
        )
    )
    fig = style_figure(fig, height=360, title="Peer analysis by sector")
    fig.update_layout(
        xaxis_title="Sector",
        yaxis_title="Score",
    )
    return fig


 # compute on demand via function above


def make_regression_bar_figure() -> go.Figure:
    """
    Bar chart of regression coefficients and t statistics
    from the results_all dataframe.
    """
    data.ensure_data_loaded()
    df = data.results_all.copy()

    if "Factor" in df.columns:
        factors = df["Factor"].astype(str).values
    else:
        factors = df.index.astype(str).values

    coef_col = next((c for c in df.columns if "coef" in c.lower()), None)
    t_col = next((c for c in df.columns if c.lower().startswith("t")), None)

    if coef_col is None or t_col is None:
        raise ValueError(
            f"Could not find coefficient or t statistic columns in results_all. "
            f"Columns are: {list(df.columns)}"
        )

    coef_vals = df[coef_col].values
    t_vals = df[t_col].values

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=factors,
            y=coef_vals,
            name="Coefficient",
            marker_color=THEME["accent"],
        )
    )
    fig = style_figure(
        fig,
        height=420,
        title="Regression coefficients",
    )
    fig.update_layout(
        xaxis_title="Factor",
        yaxis_title="Value",
        barmode="group",
        xaxis=dict(tickangle=45),
    )
    return fig


 # compute on demand via function above


def normal_pdf(x: np.ndarray) -> np.ndarray:
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x**2)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def make_tstat_distribution_figure() -> go.Figure:
    """
    Plot standard normal density and mark each regression t statistic
    as a vertical line with its two sided p value.
    """
    data.ensure_data_loaded()
    df = data.results_all.copy()

    # factor labels
    if "Factor" in df.columns:
        factors = df["Factor"].astype(str).values
    else:
        factors = df.index.astype(str).values

    # find t stat column
    t_col = next((c for c in df.columns if c.lower().startswith("t")), None)
    if t_col is None:
        raise ValueError(
            f"Could not find t statistic column in results_all. "
            f"Columns are: {list(df.columns)}"
        )

    t_vals = df[t_col].to_numpy(dtype=float)

    # keep only finite values to avoid NaN or inf
    mask = np.isfinite(t_vals)
    t_vals = t_vals[mask]
    factors = np.array(factors)[mask]

    if t_vals.size == 0:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No finite t statistics to display",
            showarrow=False,
            font=dict(color=THEME["accent"], size=13),
        )
        return style_figure(fig, height=260, title="t statistics on standard normal")

    # determine x range from t stats
    max_abs_t = float(np.max(np.abs(t_vals)))
    x_limit = max(4.0, min(10.0, max_abs_t * 1.2))

    # grid for the standard normal density
    x = np.linspace(-x_limit, x_limit, 800)
    y = normal_pdf(x)

    fig = go.Figure()

    # standard normal curve
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="Standard normal pdf",
            line=dict(width=2, color=THEME["muted"]),
            hoverinfo="skip",
        )
    )

    # vertical lines for each t stat
    for factor, t in zip(factors, t_vals):
        t_true = float(t)
        t_plot = float(np.clip(t_true, -x_limit * 0.98, x_limit * 0.98))

        pdf_t = normal_pdf(np.array([t_plot]))[0]
        p_two = 2.0 * (1.0 - normal_cdf(abs(t_true)))
        temp_color = THEME["accent2"]
        if factor == "MomentumScore":
            pdf_t = 0.1
        if abs(t_true) > 2.0:
            temp_color = "rgba(255, 80, 80, 0.8)"

        fig.add_trace(
            go.Scatter(
                x=[t_plot, t_plot],
                y=[0.0, pdf_t],
                mode="lines",
                name=f"{factor}  t={t_true:.2f}  p={p_two:.3f}",
                line=dict(width=2, color=temp_color),
                hovertemplate=(
                    f"Factor: {factor}<br>"
                    f"t statistic: {t_true:.3f}<br>"
                    f"two sided p: {p_two:.4f}<extra></extra>"
                ),
            )
        )

    # apply theme first
    fig = style_figure(
        fig,
        height=260,  # slightly taller to give space for legend + plot
        title="t statistics on standard normal",
    )

    # then override axes, margins, and legend placement
    fig.update_xaxes(range=[-x_limit, x_limit])
    fig.update_layout(
        xaxis_title="t value",
        yaxis_title="Density",
        margin=dict(l=40, r=20, b=80, t=60),  # extra bottom space for legend
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.25,          # place legend below the x axis, outside plot
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig
