from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


THEME = {
    "bg": "#000000",
    "panel": "#111111",
    "card": "#151515",
    "accent": "#f8e71c",
    "accent2": "#00e6ff",
    "grid": "#333333",
    "text": "#f5f5f5",
    "muted": "#999999",
    "border": "#2a2a2a",
}

APP_TITLE = "Bloomberg style portfolio dashboard"

# frontier slider
FRONTIER_MIN = 10
FRONTIER_MAX = 200
FRONTIER_STEP = 5
FRONTIER_DEFAULT = 80

# simulation sliders generic
SIM_STEPS_MIN = 10
SIM_STEPS_MAX = 500
SIM_STEPS_STEP = 10
SIM_STEPS_DEFAULT = 252

SIM_NPATHS_MIN = 10
SIM_NPATHS_MAX = 1000
SIM_NPATHS_STEP = 10
SIM_NPATHS_DEFAULT = 200

SIM_MAXPATHS_MIN = 10
SIM_MAXPATHS_MAX = 200
SIM_MAXPATHS_STEP = 10
SIM_MAXPATHS_DEFAULT = 50

GRAPH_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

CARD_PADDING = "16px 18px"
CARD_MIN_HEIGHT = "460px"


def nmise_mape(y_true, y_pred):
    """
    Compute NMISE and MAPE between truth and prediction vectors.

    NMISE = mean( (y_pred - y_true)**2 ) / mean( y_true**2 )
    MAPE  = mean( |y_pred - y_true| / |y_true| ) * 100
           computed only on entries where y_true != 0
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes must match, got {y_true.shape} and {y_pred.shape}")

    mse = np.mean((y_pred - y_true) ** 2)
    denom = np.mean(y_true ** 2)
    nmise = np.nan if denom == 0 else mse / denom

    mask = y_true != 0
    if not np.any(mask):
        mape = np.nan
    else:
        mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100.0

    return nmise, mape


def style_figure(fig: go.Figure, height: int = 450, title: str | None = None) -> go.Figure:
    """
    Apply Bloomberg style to a 2D plotly figure.
    """
    fig.update_layout(
        template="plotly_dark",
        title=title if title is not None else (fig.layout.title.text if fig.layout.title else None),
        title_font=dict(size=16, color=THEME["accent"]),
        font=dict(color=THEME["text"]),
        plot_bgcolor=THEME["panel"],
        paper_bgcolor=THEME["card"],
        margin=dict(l=40, r=20, b=40, t=60),
        height=height,
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=THEME["grid"],
        zeroline=False,
        linecolor=THEME["border"],
        tickfont=dict(color=THEME["muted"]),
        title_font=dict(color=THEME["muted"]),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=THEME["grid"],
        zeroline=False,
        linecolor=THEME["border"],
        tickfont=dict(color=THEME["muted"]),
        title_font=dict(color=THEME["muted"]),
    )
    return fig
