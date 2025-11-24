from __future__ import annotations

from dash import dcc, html

from .config import (
    THEME,
    APP_TITLE,
    GRAPH_CONFIG,
    CARD_PADDING,
    CARD_MIN_HEIGHT,
    FRONTIER_MIN,
    FRONTIER_MAX,
    FRONTIER_STEP,
    FRONTIER_DEFAULT,
    SIM_STEPS_MIN,
    SIM_STEPS_MAX,
    SIM_STEPS_STEP,
    SIM_STEPS_DEFAULT,
    SIM_NPATHS_MIN,
    SIM_NPATHS_MAX,
    SIM_NPATHS_STEP,
    SIM_NPATHS_DEFAULT,
    SIM_MAXPATHS_MIN,
    SIM_MAXPATHS_MAX,
    SIM_MAXPATHS_STEP,
    SIM_MAXPATHS_DEFAULT,
)
from .data import (
    cluster_metrics,
    nmise_all,
    mape_all,
    nmise_mom,
    mape_mom,
    r2_all,
    r2_mom,
)
from .figures_cross_section import (
    get_cluster_fig,
    make_regression_figure,
    make_peer_sector_figure,
    make_regression_bar_figure,
    make_tstat_distribution_figure,
)
from .figures_portfolio import (
    make_frontier_figure,
    make_paths_figure,
    make_tangency_levels_figure,
    make_portfolio_paths_3d_figure,
)


# ---------- LaTeX markdown blocks for each tab ----------

portfolio_math_md = r"""
**Underlying mathematics**

**Return and risk estimation**

Let $r_t \in \mathbb{R}^n$ be the vector of asset log returns at time $t$.

Sample mean:
$$
\hat{\mu} = \frac{1}{T} \sum_{t=1}^{T} r_t
$$

Sample covariance:
$$
\hat{\Sigma} = \frac{1}{T - 1} \sum_{t=1}^{T} (r_t - \hat{\mu})(r_t - \hat{\mu})^{\top}
$$

Annualisation uses
$$
\mu_{\text{ann}} = \frac{\hat{\mu}}{\Delta t},
\qquad
\Sigma_{\text{ann}} = \frac{\hat{\Sigma}}{\Delta t},
$$
where $\Delta t$ is the length of one return step (for example $1/252$ for daily data).

---

**Calibration of GBM parameters**

For each asset price $S_t$, define log prices $X_t = \log S_t$ and log increments
$$
\Delta X_t = X_{t+\Delta t} - X_t.
$$

Under a geometric Brownian motion,
$$
\Delta X_t \sim \mathcal{N}\bigl( (\mu - \tfrac{1}{2}\sigma^2)\Delta t,\; \sigma^2 \Delta t \bigr).
$$

From historical data we compute the sample mean $m$ and variance $v$ of $\Delta X_t$:
$$
m = \frac{1}{T} \sum_{t=1}^{T} \Delta X_t,
\qquad
v = \frac{1}{T - 1} \sum_{t=1}^{T} (\Delta X_t - m)^2.
$$

The maximum likelihood estimators are
$$
\hat{\sigma}^2 = \frac{v}{\Delta t},
\qquad
\hat{\mu} = \frac{m}{\Delta t} + \frac{1}{2}\hat{\sigma}^2.
$$

In the multivariate case, let $\Delta X_t \in \mathbb{R}^n$ be the vector of log increments:

* Sample mean vector $m \in \mathbb{R}^n$  
* Sample covariance matrix $C \in \mathbb{R}^{n \times n}$

Then the GBM covariance for log prices is
$$
\hat{\Sigma}_{\log} = \frac{C}{\Delta t},
$$
and elementwise drift for each asset $i$ is
$$
\hat{\mu}_{\log,i} = \frac{m_i}{\Delta t},
\qquad
\hat{\mu}_i = \hat{\mu}_{\log,i} + \tfrac{1}{2}\hat{\Sigma}_{\log,ii}.
$$

These calibrated $\hat{\mu}$ and $\hat{\Sigma}$ are exactly what the engine uses as inputs for the Markowitz frontier and GBM simulations.

---

**Correlation and simulation of shocks**

From the covariance matrix $\hat{\Sigma}_{\log}$ we obtain a Cholesky factor $L$:
$$
\hat{\Sigma}_{\log} = L L^{\top},
\quad
L \text{ lower triangular}.
$$

To simulate correlated shocks at each step:

1. Draw $Z_t \sim \mathcal{N}(0, I_n)$ (independent components)  
2. Set correlated shocks $\varepsilon_t = L Z_t$  

The GBM step is then applied componentwise using these correlated shocks.

---

**Markowitz frontier**

For a portfolio with weights $w$:

Expected return:
$$
\mu_p = w^{\top} \mu
$$

Volatility:
$$
\sigma_p = \sqrt{w^{\top} \Sigma w}
$$

The efficient frontier solves
$$
\min_w \; w^{\top} \Sigma w
\quad \text{s.t.} \quad
w^{\top} \mu = \mu^\*, \; \mathbf{1}^{\top} w = 1
$$

for different targets $\mu^\*$.

---

**Tangency portfolio (max Sharpe)**

With risk free rate $r_f$ and ones vector $\mathbf{1}$,
$$
w^\* \propto \Sigma^{-1}(\mu - r_f \mathbf{1}),
\qquad
w^\* = \frac{\Sigma^{-1}(\mu - r_f \mathbf{1})}{\mathbf{1}^{\top} \Sigma^{-1}(\mu - r_f \mathbf{1})}.
$$

This $w^\*$ is the tangency portfolio used for both the frontier highlight and the GBM portfolio simulations.

---

**GBM simulation of asset and portfolio paths**

Each asset follows the discretised GBM
$$
S_{t+\Delta t} = S_t \exp\Big(
(\mu - \tfrac{1}{2} \sigma^2)\Delta t + \sigma \sqrt{\Delta t}\, Z_t
\Big),
\quad Z_t \sim \mathcal{N}(0, 1),
$$
or in the multivariate case using correlated $\varepsilon_t$ as described above.

Portfolio value at each step is
$$
V_t = w^{\top} S_t,
$$
and in the simulation engine paths are often normalised by $V_0$ so that $V_0 = 1$.

---

**Historical vs simulated tangency levels**

1. Compute realised tangency portfolio level $V^{\text{hist}}_t$ from actual prices  
2. Simulate normalised paths $\tilde V^{(k)}_t$ under GBM  
3. Map to level space with starting level $V^{\text{hist}}_0$ via  
   $V^{(k)}_t = V^{\text{hist}}_0 \tilde V^{(k)}_t$  

The simulated distribution, mean path, and a “golden run” path are then compared with $V^{\text{hist}}_t$.

---

**Golden run fit metrics (NMISE and MAPE)**

For a given simulated path $V^{(k)}_t$ and realised path $V^{\text{hist}}_t$:

Normalised mean integrated squared error:
$$
\text{NMISE}^{(k)} =
\frac{\sum_{t} \bigl( V^{(k)}_t - V^{\text{hist}}_t \bigr)^2}
     {\sum_{t} \bigl( V^{\text{hist}}_t - \bar{V}^{\text{hist}} \bigr)^2},
\quad
\bar{V}^{\text{hist}} = \frac{1}{T} \sum_{t} V^{\text{hist}}_t.
$$

Mean absolute percent error:
$$
\text{MAPE}^{(k)} =
\frac{100}{T} \sum_{t}
\left\lvert
\frac{V^{(k)}_t - V^{\text{hist}}_t}{V^{\text{hist}}_t}
\right\rvert.
$$

The “golden run” highlighted in the plot is the path $k$ that minimises $\text{NMISE}^{(k)}$ among all simulated paths.
"""

cross_section_math_md = r"""
**Underlying mathematics**

**Factor construction and preprocessing**

Let $X_{ij}$ be raw descriptor $j$ for stock $i$ at a given observation date  
(for example value, quality, momentum, size).

Preprocessing steps

1. Outlier control: clip each $X_{ij}$ at chosen lower and upper percentiles inside the universe.
2. Optional neutralisation: regress descriptors on controls such as size or sector dummies and keep residuals to remove simple biases.
3. Standardisation:
$$
Z_{ij} = \frac{X_{ij} - \bar X_j}{s_j},
$$
where $\bar X_j$ and $s_j$ are the cross sectional mean and standard deviation of descriptor $j$.

Many descriptors are then combined into a smaller set of composite factors.  
If $W$ is a matrix of weights with one column per factor, the factor score for stock $i$ is
$$
F_i = Z_i W,
$$
where $Z_i$ is the row vector of standardised descriptors for stock $i$.

Typical choices

• Value: combinations of book to price, earnings to price, cash flow to price, dividend yield, arranged so that higher scores mean cheaper stocks.  
• Quality: return on equity, profitability margins, earnings stability, balance sheet strength.  
• Momentum: cumulative return over a window such as twelve months excluding the most recent month, then standardised across the universe.  
• Size or growth: log market capitalisation, sales growth, earnings growth and similar variables.

The analytics treat the vector of factor scores for stock $i$ as $F_i \in \mathbb R^p$.

---

**Clustering of factor profiles**

The model clusters stocks in the space of factor scores.  
With factor vector $F_i \in \mathbb R^p$ and $K$ clusters, the k means objective is
$$
\min_{\{C_k\}} \sum_{k=1}^{K} \sum_{i \in C_k}
\lVert F_i - \mu_k \rVert^2,
$$
where $\mu_k$ is the centroid for cluster $C_k$.

Cluster quality is summarised by

• Silhouette index  
• Calinski Harabasz index  
• Davies Bouldin index  
• Inertia (total within cluster sum of squares)

Good solutions have high silhouette and Calinski Harabasz scores, low Davies Bouldin, and relatively low inertia for the chosen number of clusters. The three dimensional PCA plot displays the clusters after reducing factor space to its first few principal components.

---

**Cross sectional regression of returns on factors**

For a single observation date, let $y_i$ be the one period forward return for stock $i$ and let $X_i$ be its vector of factor scores. Stack these into
$$
y \in \mathbb R^N, \quad X \in \mathbb R^{N \times (p+1)},
$$
where the first column of $X$ is a column of ones for the intercept.

The cross sectional factor model is
$$
y = X \beta + \varepsilon,
$$
with ordinary least squares estimator
$$
\hat \beta = (X^\top X)^{-1} X^\top y.
$$

The first component of $\hat \beta$ is the intercept and the remaining components are factor premia for that date. The dashboard contrasts

• Full factor set: all available factor scores in $X$  
• Momentum only: a restricted regression that uses the momentum score as the single explanatory variable

For each coefficient $\hat \beta_j$ we compute the standard error $\text{se}(\hat \beta_j)$ from the OLS covariance estimate and the associated t statistic
$$
t_j = \frac{\hat \beta_j}{\text{se}(\hat \beta_j)}.
$$

These t statistics are displayed both as a bar chart and as a distribution to show which signals have economically and statistically meaningful payoffs.

The fitted cross sectional return is
$$
\hat y = X \hat \beta,
$$
and the coefficient of determination is
$$
R^2 = 1 − \frac{\sum_i (y_i − \hat y_i)^2}{\sum_i (y_i − \bar y)^2},
$$
where $\bar y$ is the cross sectional mean of realised returns.

Over time, repeating this cross sectional regression on each date produces a time series of factor premia and diagnostics, though the dashboard here focuses on the current snapshot.

---

**Error measures used in the summary**

Let $y_i$ be realised return and $\hat y_i$ the model prediction for stock $i$.

Mean absolute percent error
$$
\text{MAPE} = \frac{100}{N} \sum_{i=1}^{N}
\left\lvert
\frac{y_i − \hat y_i}{y_i}
\right\rvert.
$$

Normalised mean integrated squared error compares the squared error to the variance of realised returns
$$
\text{NMISE} =
\frac{\sum_i (y_i − \hat y_i)^2}
{\sum_i (y_i − \bar y)^2}.
$$

Smaller values indicate better cross sectional fit. The application reports these metrics for both the full factor model and the momentum only model.

---

**Peer valuation and sector dispersion**

Within each sector $s$, for a valuation ratio $M_i$ (for example price over earnings or enterprise value over EBITDA) we compute a sector reference statistic, typically the median $M^{\text{sector}}_s$. For stock $i$ in sector $s$,
$$
\text{Valuation\_vs\_Peers}_i
=
\frac{M_i − M^{\text{sector}}_s}{M^{\text{sector}}_s}.
$$

Negative values correspond to discounts versus sector peers and positive values to premia.

The peer dispersion plot groups $\text{Valuation\_vs\_Peers}_i$ by sector and displays their distribution, highlighting sectors where valuation is especially stretched or compressed relative to internal peers. This ties directly back to the factor and regression views,
since extreme valuation outliers often align with strong value or growth factor exposures.
"""


# ---------- Shared card helper ----------

def card(children, title=None):
    header = (
        html.Div(
            title,
            style={
                "color": THEME["accent"],
                "fontWeight": "bold",
                "marginBottom": "6px",
                "fontSize": "15px",
            },
        )
        if title
        else None
    )

    if isinstance(children, (list, tuple)):
        flat_children = list(children)
    else:
        flat_children = [children]

    if header is not None:
        flat_children = [header] + flat_children

    return html.Div(
        style={
            "backgroundColor": THEME["card"],
            "border": f"1px solid {THEME['border']}",
            "borderRadius": "4px",
            "padding": CARD_PADDING,
            "minHeight": CARD_MIN_HEIGHT,
            "boxShadow": "0 0 10px rgba(0, 0, 0, 0.6)",
            "display": "flex",
            "flexDirection": "column",
        },
        children=flat_children,
    )


# ---------- Portfolio tab ----------

def layout_tab_portfolio():
    # maths explainer at top of portfolio tab
    math_card = card(
        children=dcc.Markdown(
            portfolio_math_md,
            mathjax=True,
            style={
                "fontSize": "12px",
                "color": THEME["muted"],
                "whiteSpace": "pre-wrap",
            },
        ),
        title="Underlying mathematics",
    )

    # row 1 frontier and GBM sim
    row1 = html.Div(
        style={
            "display": "flex",
            "gap": "24px",
            "alignItems": "flex-start",
            "marginBottom": "18px",
        },
        children=[
            html.Div(
                style={"flex": "1"},
                children=[
                    card(
                        title="Efficient frontier",
                        children=[
                            html.Div(
                                [
                                    html.Label(
                                        "Number of frontier points",
                                        style={"color": THEME["muted"]},
                                    ),
                                    dcc.Slider(
                                        id="frontier-npoints",
                                        min=FRONTIER_MIN,
                                        max=FRONTIER_MAX,
                                        step=FRONTIER_STEP,
                                        value=FRONTIER_DEFAULT,
                                        marks={
                                            FRONTIER_MIN: str(FRONTIER_MIN),
                                            50: "50",
                                            100: "100",
                                            150: "150",
                                            FRONTIER_MAX: str(FRONTIER_MAX),
                                        },
                                        tooltip={"placement": "bottom"},
                                    ),
                                    html.Br(),
                                    dcc.Checklist(
                                        id="frontier-show-tangency",
                                        options=[
                                            {
                                                "label": "Show tangency portfolio",
                                                "value": "show",
                                            }
                                        ],
                                        value=["show"],
                                        style={"color": THEME["text"]},
                                    ),
                                ]
                            ),
                            html.Br(),
                            dcc.Loading(
                                type="dot",
                                color=THEME["accent"],
                                children=dcc.Graph(
                                    id="frontier-graph",
                                    figure=make_frontier_figure(
                                        n_points=FRONTIER_DEFAULT,
                                        show_tangency=True,
                                    ),
                                    style={"height": "460px", "width": "100%"},
                                    config=GRAPH_CONFIG,
                                ),
                            ),
                        ],
                    )
                ],
            ),
            html.Div(
                style={"flex": "1"},
                children=[
                    card(
                        title="GBM simulation",
                        children=[
                            html.Div(
                                [
                                    html.Label(
                                        "Simulation steps",
                                        style={"color": THEME["muted"]},
                                    ),
                                    dcc.Slider(
                                        id="sim-nsteps",
                                        min=SIM_STEPS_MIN,
                                        max=SIM_STEPS_MAX,
                                        step=SIM_STEPS_STEP,
                                        value=SIM_STEPS_DEFAULT,
                                        marks={
                                            SIM_STEPS_MIN: str(SIM_STEPS_MIN),
                                            100: "100",
                                            252: "252",
                                            SIM_STEPS_MAX: str(SIM_STEPS_MAX),
                                        },
                                        tooltip={"placement": "bottom"},
                                    ),
                                    html.Br(),
                                    html.Label(
                                        "Number of paths",
                                        style={"color": THEME["muted"]},
                                    ),
                                    dcc.Slider(
                                        id="sim-npaths",
                                        min=SIM_NPATHS_MIN,
                                        max=SIM_NPATHS_MAX,
                                        step=SIM_NPATHS_STEP,
                                        value=SIM_NPATHS_DEFAULT,
                                        marks={
                                            SIM_NPATHS_MIN: str(SIM_NPATHS_MIN),
                                            100: "100",
                                            500: "500",
                                            SIM_NPATHS_MAX: str(SIM_NPATHS_MAX),
                                        },
                                        tooltip={"placement": "bottom"},
                                    ),
                                    html.Br(),
                                    html.Label(
                                        "Max displayed paths",
                                        style={"color": THEME["muted"]},
                                    ),
                                    dcc.Slider(
                                        id="sim-maxpaths",
                                        min=SIM_MAXPATHS_MIN,
                                        max=SIM_MAXPATHS_MAX,
                                        step=SIM_MAXPATHS_STEP,
                                        value=SIM_MAXPATHS_DEFAULT,
                                        marks={
                                            SIM_MAXPATHS_MIN: str(SIM_MAXPATHS_MIN),
                                            50: "50",
                                            100: "100",
                                            SIM_MAXPATHS_MAX: str(SIM_MAXPATHS_MAX),
                                        },
                                        tooltip={"placement": "bottom"},
                                    ),
                                    html.Br(),
                                    dcc.RadioItems(
                                        id="sim-weights-type",
                                        options=[
                                            {
                                                "label": "Tangency portfolio",
                                                "value": "tangency",
                                            },
                                            {
                                                "label": "Equal weight",
                                                "value": "equal",
                                            },
                                        ],
                                        value="tangency",
                                        inline=True,
                                        style={"color": THEME["text"]},
                                    ),
                                    html.Br(),
                                    html.Button(
                                        "Run simulation",
                                        id="sim-run-button",
                                        n_clicks=0,
                                        style={
                                            "marginBottom": "8px",
                                            "backgroundColor": THEME["accent"],
                                            "color": "#000000",
                                            "border": "none",
                                            "padding": "6px 12px",
                                            "cursor": "pointer",
                                            "fontWeight": "bold",
                                        },
                                    ),
                                ]
                            ),
                            dcc.Loading(
                                type="dot",
                                color=THEME["accent"],
                                children=dcc.Graph(
                                    id="sim-graph",
                                    figure=make_paths_figure(
                                        n_steps=SIM_STEPS_DEFAULT,
                                        n_paths=SIM_NPATHS_DEFAULT,
                                        use_tangency=True,
                                        max_paths=SIM_MAXPATHS_DEFAULT,
                                    ),
                                    style={"height": "460px", "width": "100%"},
                                    config=GRAPH_CONFIG,
                                ),
                            ),
                        ],
                    )
                ],
            ),
        ],
    )

    # row 2 tangency levels and 3D paths
    row2 = html.Div(
        style={
            "display": "flex",
            "gap": "24px",
            "alignItems": "flex-start",
        },
        children=[
            html.Div(
                style={"flex": "1"},
                children=[
                    card(
                        title="Tangency portfolio simulated vs historical levels",
                        children=[
                            html.Div(
                                [
                                    html.Label(
                                        "Number of simulated paths",
                                        style={"color": THEME["muted"]},
                                    ),
                                    dcc.Slider(
                                        id="tan-levels-npaths",
                                        min=100,
                                        max=1000,
                                        step=50,
                                        value=500,
                                        marks={
                                            100: "100",
                                            250: "250",
                                            500: "500",
                                            750: "750",
                                            1000: "1000",
                                        },
                                        tooltip={"placement": "bottom"},
                                    ),
                                ]
                            ),
                            html.Br(),
                            dcc.Loading(
                                type="dot",
                                color=THEME["accent"],
                                children=dcc.Graph(
                                    id="tan-levels-graph",
                                    figure=make_tangency_levels_figure(
                                        n_paths=500,
                                        max_paths_plot=50,
                                    ),
                                    style={"height": "430px", "width": "100%"},
                                    config=GRAPH_CONFIG,
                                ),
                            ),
                        ],
                    )
                ],
            ),
            html.Div(
                style={"flex": "1"},
                children=[
                    card(
                        title="3D portfolio value paths",
                        children=[
                            html.Div(
                                [
                                    html.Label(
                                        "Portfolio type",
                                        style={"color": THEME["muted"]},
                                    ),
                                    dcc.RadioItems(
                                        id="portfolio-3d-type",
                                        options=[
                                            {"label": "Tangency", "value": "tangency"},
                                            {
                                                "label": "Black Litterman",
                                                "value": "black_litterman",
                                            },
                                        ],
                                        value="tangency",
                                        inline=True,
                                        style={
                                            "color": THEME["text"],
                                            "marginBottom": "6px",
                                        },
                                    ),
                                    html.Label(
                                        "Simulation steps",
                                        style={"color": THEME["muted"]},
                                    ),
                                    dcc.Slider(
                                        id="portfolio-3d-nsteps",
                                        min=SIM_STEPS_MIN,
                                        max=SIM_STEPS_MAX,
                                        step=SIM_STEPS_STEP,
                                        value=252,
                                        marks={
                                            SIM_STEPS_MIN: str(SIM_STEPS_MIN),
                                            126: "126",
                                            252: "252",
                                            SIM_STEPS_MAX: str(SIM_STEPS_MAX),
                                        },
                                        tooltip={"placement": "bottom"},
                                    ),
                                    html.Br(),
                                    html.Label(
                                        "Number of simulated paths",
                                        style={"color": THEME["muted"]},
                                    ),
                                    dcc.Slider(
                                        id="portfolio-3d-npaths",
                                        min=50,
                                        max=500,
                                        step=25,
                                        value=200,
                                        marks={
                                            50: "50",
                                            100: "100",
                                            200: "200",
                                            500: "500",
                                        },
                                        tooltip={"placement": "bottom"},
                                    ),
                                    html.Br(),
                                    html.Label(
                                        "Max displayed paths",
                                        style={"color": THEME["muted"]},
                                    ),
                                    dcc.Slider(
                                        id="portfolio-3d-maxpaths",
                                        min=10,
                                        max=100,
                                        step=5,
                                        value=30,
                                        marks={
                                            10: "10",
                                            30: "30",
                                            50: "50",
                                            100: "100",
                                        },
                                        tooltip={"placement": "bottom"},
                                    ),
                                    html.Br(),
                                    html.Button(
                                        "Run 3D simulation",
                                        id="portfolio-3d-run-button",
                                        n_clicks=0,
                                        style={
                                            "marginBottom": "8px",
                                            "backgroundColor": THEME["accent"],
                                            "color": "#000000",
                                            "border": "none",
                                            "padding": "6px 12px",
                                            "cursor": "pointer",
                                            "fontWeight": "bold",
                                        },
                                    ),
                                ]
                            ),
                            dcc.Loading(
                                type="dot",
                                color=THEME["accent"],
                                children=dcc.Graph(
                                    id="portfolio-3d-graph",
                                    figure=make_portfolio_paths_3d_figure(
                                        port_type="tangency",
                                        n_steps=252,
                                        n_paths=200,
                                        max_paths=30,
                                    ),
                                    style={"height": "430px", "width": "100%"},
                                    config=GRAPH_CONFIG,
                                ),
                            ),
                        ],
                    )
                ],
            ),
        ],
    )

    return html.Div(
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "12px",
        },
        children=[math_card, row1, row2],
    )


# ---------- Cross sectional tab ----------

def layout_tab_other():
    metrics_text = []
    if cluster_metrics is not None:
        metrics_text.append(f"Silhouette: {cluster_metrics.silhouette:.3f}")
        metrics_text.append(
            f"Calinski Harabasz: {cluster_metrics.calinski_harabasz:.1f}"
        )
        metrics_text.append(
            f"Davies Bouldin: {cluster_metrics.davies_bouldin:.3f}"
        )
        metrics_text.append(f"Inertia: {cluster_metrics.inertia:.1f}")
    metrics_joined = " | ".join(metrics_text) if metrics_text else "No metrics available"

    # maths explainer at top of EQ tab
    math_block = card(
        title="Underlying mathematics",
        children=dcc.Markdown(
            cross_section_math_md,
            mathjax=True,
            style={
                "fontSize": "12px",
                "color": THEME["muted"],
                "whiteSpace": "pre-wrap",
            },
        ),
    )

    # row 1 clustering plus regression lines, tall row for 3D
    row1 = html.Div(
        style={"display": "flex", "gap": "24px", "height": "700px"},
        children=[
            html.Div(
                style={"flex": "1", "height": "100%"},
                children=[
                    card(
                        title="Factor clustering PCA 3D",
                        children=[
                            html.Div(
                                metrics_joined,
                                style={
                                    "color": THEME["muted"],
                                    "fontSize": "11px",
                                    "marginBottom": "4px",
                                },
                            ),
                            dcc.Graph(
                                id="cluster-3d-graph",
                                figure=get_cluster_fig(),
                                config=GRAPH_CONFIG,
                                style={
                                    "flex": "1 1 auto",
                                    "height": "100%",
                                    "width": "100%",
                                },
                            ),
                        ],
                    )
                ],
            ),
            html.Div(
                style={"flex": "1", "height": "100%"},
                children=[
                    card(
                        title="Factor regression actual vs predicted",
                        children=[
                            dcc.Graph(
                                id="regression-graph",
                                figure=make_regression_figure(),
                                style={
                                    "flex": "1 1 auto",
                                    "height": "100%",
                                    "width": "100%",
                                },
                                config=GRAPH_CONFIG,
                            ),
                        ],
                    )
                ],
            ),
        ],
    )

    # row 2 peer analysis plus error summary
    row2 = html.Div(
        style={"display": "flex", "gap": "24px"},
        children=[
            html.Div(
                style={"flex": "1"},
                children=[
                    card(
                        title="Peer analysis by sector",
                        children=[
                            dcc.Graph(
                                id="peer-sector-graph",
                                figure=make_peer_sector_figure(),
                                style={"height": "430px", "width": "100%"},
                                config=GRAPH_CONFIG,
                            ),
                        ],
                    )
                ],
            ),
            html.Div(
                style={"flex": "1"},
                children=[
                    card(
                        title="Regression error summary",
                        children=[
                            html.Ul(
                                [
                                    html.Li(
                                        f"All factors: MAPE {mape_all:.2f}%, "
                                        f"NMISE {nmise_all:.4f}, R² {r2_all:.3f}",
                                        style={
                                            "color": THEME["text"],
                                            "fontSize": "12px",
                                        },
                                    ),
                                    html.Li(
                                        f"Momentum only: MAPE {mape_mom:.2f}%, "
                                        f"NMISE {nmise_mom:.4f}, R² {r2_mom:.3f}",
                                        style={
                                            "color": THEME["text"],
                                            "fontSize": "12px",
                                        },
                                    ),
                                ],
                                style={"paddingLeft": "18px"},
                            ),
                            html.Div(
                                "Compare marginal value of full factor set vs pure momentum.",
                                style={
                                    "color": THEME["muted"],
                                    "fontSize": "11px",
                                    "marginTop": "6px",
                                },
                            ),
                            html.Div(
                                style={"flex": "1"},
                                children=[
                                    dcc.Graph(
                                        id="coef-tstat-bar-graph",
                                        figure=make_regression_bar_figure(),
                                        style={"height": "420px", "width": "100%"},
                                        config=GRAPH_CONFIG,
                                    ),
                                ],
                            ),
                        ],
                    )
                ],
            ),
        ],
    )

    # row 3 t statistics distribution
    row3 = html.Div(
        style={"display": "flex", "gap": "24px"},
        children=[
            html.Div(
                style={"flex": "1"},
                children=[
                    card(
                        title="t statistics and error likelihood",
                        children=[
                            dcc.Graph(
                                id="tstat-dist-graph",
                                figure=make_tstat_distribution_figure(),
                                style={"height": "420px", "width": "100%"},
                                config=GRAPH_CONFIG,
                            ),
                        ],
                    )
                ],
            ),
        ],
    )

    return html.Div(
        style={"display": "flex", "flexDirection": "column", "gap": "18px"},
        children=[
            html.Div(
                "Cross sectional factor and peer analysis",
                style={
                    "color": THEME["muted"],
                    "marginBottom": "4px",
                    "fontSize": "14px",
                },
            ),
            math_block,
            row1,
            row2,
            row3,
        ],
    )


# ---------- Root layout and index_string ----------

def create_root_layout():
    return html.Div(
        style={
            "fontFamily": "Arial",
            "margin": "0",
            "padding": "10px 14px",
            "backgroundColor": THEME["bg"],
            "color": THEME["text"],
            "width": "100vw",
            "height": "100vh",
            "boxSizing": "border-box",
            "display": "flex",
            "flexDirection": "column",
        },
        children=[
            html.Div(
                APP_TITLE,
                style={
                    "fontSize": "22px",
                    "fontWeight": "bold",
                    "color": THEME["accent"],
                    "marginBottom": "6px",
                },
            ),
            html.Div(
                "Bloomberg themed analytics",
                style={
                    "fontSize": "12px",
                    "color": THEME["muted"],
                    "marginBottom": "12px",
                    "width": "100%",
                    "borderBottom": f"1px solid {THEME['border']}",
                },
            ),
            dcc.Tabs(
                id="tabs",
                value="tab-portfolio",
                parent_className="custom-tabs",
                className="custom-tabs-container",
                children=[
                    dcc.Tab(
                        label="Portfolio optimisation",
                        value="tab-portfolio",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        label="EQ Dataframe analysis",
                        value="tab-other",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                ],
                style={"borderBottom": f"1px solid {THEME['border']}", "marginBottom": "12px", "backgroundColor": THEME["bg"]},
            ),
            html.Div(
                id="tabs-content",
                style={
                    "marginTop": "12px",
                    "flex": "1",
                    "overflowY": "auto",
                },
            ),
        ],
    )


index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>""" + APP_TITLE + """</title>
        {%favicon%}
        {%css%}
        <script>
          MathJax = {
            tex: {
              inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
              displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            },
            svg: { fontCache: 'global' }
          };
        </script>
        <script id="MathJax-script" async
                src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
        </script>
        <style>
            .custom-tabs-container {
                color: """ + THEME["muted"] + """;
            }
            .custom-tab {
                background-color: """ + THEME["bg"] + """;
                border: 1px solid """ + THEME["border"] + """;
                padding: 6px 12px;
            }
            .custom-tab--selected {
                background-color: """ + THEME["card"] + """;
                border-bottom: 2px solid """ + THEME["accent"] + """;
                color: """ + THEME["accent"] + """;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""
