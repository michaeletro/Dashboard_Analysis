from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from common.metrics import nmise_mape


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


 # nmise_mape is now provided by common.metrics and imported above


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
