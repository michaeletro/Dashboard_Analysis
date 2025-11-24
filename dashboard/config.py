from __future__ import annotations

import os
import numpy as np
import plotly.graph_objects as go
from common.metrics import nmise_mape


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


THEME = {
    "bg": _env("APP_BG_COLOR", "#000000"),
    "panel": _env("APP_PANEL_COLOR", "#111111"),
    "card": _env("APP_CARD_COLOR", "#151515"),
    "accent": _env("APP_ACCENT_COLOR", "#f8e71c"),
    "accent2": _env("APP_ACCENT2_COLOR", "#00e6ff"),
    "grid": _env("APP_GRID_COLOR", "#333333"),
    "text": _env("APP_TEXT_COLOR", "#f5f5f5"),
    "muted": _env("APP_MUTED_COLOR", "#999999"),
    "border": _env("APP_BORDER_COLOR", "#2a2a2a"),
}

APP_TITLE = _env("APP_TITLE", "Bloomberg style portfolio dashboard")

# frontier slider
FRONTIER_MIN = int(_env("FRONTIER_MIN", "10"))
FRONTIER_MAX = int(_env("FRONTIER_MAX", "200"))
FRONTIER_STEP = int(_env("FRONTIER_STEP", "5"))
FRONTIER_DEFAULT = int(_env("FRONTIER_DEFAULT", "80"))

# simulation sliders generic
SIM_STEPS_MIN = int(_env("SIM_STEPS_MIN", "10"))
SIM_STEPS_MAX = int(_env("SIM_STEPS_MAX", "500"))
SIM_STEPS_STEP = int(_env("SIM_STEPS_STEP", "10"))
SIM_STEPS_DEFAULT = int(_env("SIM_STEPS_DEFAULT", "252"))

SIM_NPATHS_MIN = int(_env("SIM_NPATHS_MIN", "10"))
SIM_NPATHS_MAX = int(_env("SIM_NPATHS_MAX", "1000"))
SIM_NPATHS_STEP = int(_env("SIM_NPATHS_STEP", "10"))
SIM_NPATHS_DEFAULT = int(_env("SIM_NPATHS_DEFAULT", "200"))

SIM_MAXPATHS_MIN = int(_env("SIM_MAXPATHS_MIN", "10"))
SIM_MAXPATHS_MAX = int(_env("SIM_MAXPATHS_MAX", "200"))
SIM_MAXPATHS_STEP = int(_env("SIM_MAXPATHS_STEP", "10"))
SIM_MAXPATHS_DEFAULT = int(_env("SIM_MAXPATHS_DEFAULT", "50"))

GRAPH_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

CARD_PADDING = _env("CARD_PADDING", "16px 18px")
CARD_MIN_HEIGHT = _env("CARD_MIN_HEIGHT", "460px")


def get_runtime_settings() -> dict:
    """Return a snapshot of current runtime settings (theme + slider defaults).

    Useful for dynamic layout generation or exporting configuration.
    """
    return {
        "APP_TITLE": APP_TITLE,
        "THEME": THEME.copy(),
        "FRONTIER": {
            "min": FRONTIER_MIN,
            "max": FRONTIER_MAX,
            "step": FRONTIER_STEP,
            "default": FRONTIER_DEFAULT,
        },
        "SIM_STEPS": {
            "min": SIM_STEPS_MIN,
            "max": SIM_STEPS_MAX,
            "step": SIM_STEPS_STEP,
            "default": SIM_STEPS_DEFAULT,
        },
        "SIM_NPATHS": {
            "min": SIM_NPATHS_MIN,
            "max": SIM_NPATHS_MAX,
            "step": SIM_NPATHS_STEP,
            "default": SIM_NPATHS_DEFAULT,
        },
        "SIM_MAXPATHS": {
            "min": SIM_MAXPATHS_MIN,
            "max": SIM_MAXPATHS_MAX,
            "step": SIM_MAXPATHS_STEP,
            "default": SIM_MAXPATHS_DEFAULT,
        },
        "CARD": {
            "padding": CARD_PADDING,
            "min_height": CARD_MIN_HEIGHT,
        },
    }


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
