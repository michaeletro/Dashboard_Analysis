from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, State, html

from .app_instance import app
from .config import (
    FRONTIER_MIN,
    FRONTIER_MAX,
    SIM_STEPS_MIN,
    SIM_STEPS_MAX,
    SIM_NPATHS_MIN,
    SIM_NPATHS_MAX,
    SIM_MAXPATHS_MIN,
    SIM_MAXPATHS_MAX,
)
from .layouts import layout_tab_portfolio, layout_tab_other
from .figures_portfolio import (
    make_frontier_figure,
    make_paths_figure,
    make_tangency_levels_figure,
    make_portfolio_paths_3d_figure,
)


@app.callback(
    Output("tabs-content", "children"),
    Input("tabs", "value"),
)
def render_tab(tab_value):
    if tab_value == "tab-portfolio":
        return layout_tab_portfolio()
    if tab_value == "tab-other":
        return layout_tab_other()
    return html.Div("Unknown tab")


@app.callback(
    Output("frontier-graph", "figure"),
    Input("frontier-npoints", "value"),
    Input("frontier-show-tangency", "value"),
)
def update_frontier(n_points, show_tangency_values):
    n_points = int(max(FRONTIER_MIN, min(FRONTIER_MAX, n_points)))
    show_tan = "show" in (show_tangency_values or [])
    fig = make_frontier_figure(n_points=n_points, show_tangency=show_tan)
    return fig


@app.callback(
    Output("sim-graph", "figure"),
    Input("sim-run-button", "n_clicks"),
    State("sim-nsteps", "value"),
    State("sim-npaths", "value"),
    State("sim-maxpaths", "value"),
    State("sim-weights-type", "value"),
)
def update_simulation(n_clicks, n_steps, n_paths, max_paths, weights_type):
    if not n_clicks:
        return go.Figure()

    n_steps = int(max(SIM_STEPS_MIN, min(SIM_STEPS_MAX, n_steps)))
    n_paths = int(max(SIM_NPATHS_MIN, min(SIM_NPATHS_MAX, n_paths)))
    max_paths = int(max(SIM_MAXPATHS_MIN, min(SIM_MAXPATHS_MAX, max_paths)))

    use_tangency = weights_type == "tangency"

    fig = make_paths_figure(
        n_steps=n_steps,
        n_paths=n_paths,
        use_tangency=use_tangency,
        max_paths=max_paths,
    )
    return fig


@app.callback(
    Output("tan-levels-graph", "figure"),
    Input("tan-levels-npaths", "value"),
)
def update_tangency_levels(n_paths):
    n_paths = int(max(100, min(1000, n_paths)))
    fig = make_tangency_levels_figure(n_paths=n_paths, max_paths_plot=50)
    return fig


@app.callback(
    Output("portfolio-3d-graph", "figure"),
    Input("portfolio-3d-run-button", "n_clicks"),
    State("portfolio-3d-type", "value"),
    State("portfolio-3d-nsteps", "value"),
    State("portfolio-3d-npaths", "value"),
    State("portfolio-3d-maxpaths", "value"),
)
def update_portfolio_3d(n_clicks, port_type, n_steps, n_paths, max_paths):
    if not n_clicks:
        return go.Figure()

    n_steps = int(max(SIM_STEPS_MIN, min(SIM_STEPS_MAX, n_steps)))
    n_paths = int(max(50, min(500, n_paths)))
    max_paths = int(max(10, min(100, max_paths)))

    fig = make_portfolio_paths_3d_figure(
        port_type=port_type,
        n_steps=n_steps,
        n_paths=n_paths,
        max_paths=max_paths,
    )
    return fig
