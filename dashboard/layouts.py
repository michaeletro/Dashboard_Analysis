from __future__ import annotations

from dash import dcc, html

from .config import (
    THEME,
    GRAPH_CONFIG,
    get_runtime_settings,
)
from .config import (
    THEME,
    GRAPH_CONFIG,
    get_runtime_settings,
)
from . import data

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
from .figures_bonds import (
    make_yield_curve_figure,
    make_duration_distribution_figure,
    make_credit_analysis_figure,
    make_bond_performance_figure,
)
from .figures_risk import (
    make_risk_dashboard_figure,
    make_correlation_heatmap_figure,
    make_bond_duration_risk_figure,
)


# ---------- LaTeX markdown blocks for each tab ----------

portfolio_math_md = r"""
**Underlying mathematics**

### Return and risk estimation

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
\mu_{\text{ann}} = \frac{\hat{\mu}}{\Delta t}, \qquad
\Sigma_{\text{ann}} = \frac{\hat{\Sigma}}{\Delta t},
$$
where $\Delta t$ is the length of one return step (e.g. $1/252$ for daily data).

---

### Calibration of GBM parameters

For each asset price $S_t$, define log price $X_t = \log S_t$ and increment
$$
\Delta X_t = X_{t+\Delta t} - X_t.
$$
Under GBM,
$$
\Delta X_t \sim \mathcal{N}\bigl( (\mu - \tfrac{1}{2}\sigma^2)\Delta t,\; \sigma^2 \Delta t \bigr).
$$

Compute sample mean $m$ and variance $v$ of $\Delta X_t$:
$$
m = \frac{1}{T} \sum_{t=1}^{T} \Delta X_t, \qquad
v = \frac{1}{T - 1} \sum_{t=1}^{T} (\Delta X_t - m)^2.
$$
MLE estimators:
$$
\hat{\sigma}^2 = \frac{v}{\Delta t}, \qquad
\hat{\mu} = \frac{m}{\Delta t} + \frac{1}{2}\hat{\sigma}^2.
$$

Multivariate case: $\Delta X_t \in \mathbb{R}^n$ with mean vector $m$ and covariance $C$.
$$
\hat{\Sigma}_{\log} = \frac{C}{\Delta t}, \qquad
\hat{\mu}_{\log,i} = \frac{m_i}{\Delta t}, \qquad
\hat{\mu}_i = \hat{\mu}_{\log,i} + \tfrac{1}{2}\hat{\Sigma}_{\log,ii}.
$$

---

### Correlated shocks
Cholesky (or other) factorisation:
$$
\hat{\Sigma}_{\log} = L L^{\top}, \quad L \text{ lower triangular}.
$$
Simulate step:
1. Draw $Z_t \sim \mathcal{N}(0, I_n)$.
2. Set $\varepsilon_t = L Z_t$.

---

### Markowitz frontier
Portfolio weights $w$ give expected return and volatility:
$$
\mu_p = w^{\top} \mu, \qquad \sigma_p = \sqrt{w^{\top} \Sigma w}.
$$
Frontier solves
$$
\min_w \; w^{\top} \Sigma w \quad \text{s.t.} \quad w^{\top} \mu = \mu^*,\; \mathbf{1}^{\top} w = 1.
$$

### Portfolio Construction Methods

#### 1. Equal Weight Portfolio
Simplest diversification:
$$
w_i = \frac{1}{n}, \quad i = 1, \ldots, n
$$
where $n$ is the number of assets.

#### 2. Tangency (Max Sharpe) Portfolio
Optimal risk-adjusted portfolio with risk-free rate $r_f$:
$$
w^{\text{tan}} = \frac{\Sigma^{-1}(\mu - r_f \mathbf{1})}{\mathbf{1}^{\top} \Sigma^{-1}(\mu - r_f \mathbf{1})}
$$
Maximises Sharpe ratio: $\text{SR} = \frac{\mu_p - r_f}{\sigma_p}$.

#### 3. Black-Litterman Portfolio
Bayesian approach combining market equilibrium with investor views:

**Prior (Market Equilibrium):**
$$
\mu_{\text{prior}} = \lambda \Sigma w_{\text{mkt}}
$$
where $\lambda$ is risk aversion, $w_{\text{mkt}}$ are market cap weights.

**Investor Views:** $P \mu = Q + \varepsilon$, where:
- $P$: picking matrix (which assets the views concern)
- $Q$: expected returns from views
- $\varepsilon \sim \mathcal{N}(0, \Omega)$: view uncertainty

**Posterior (Black-Litterman):**
$$
\mu_{\text{BL}} = \left[(\tau\Sigma)^{-1} + P^{\top}\Omega^{-1}P\right]^{-1} \left[(\tau\Sigma)^{-1}\mu_{\text{prior}} + P^{\top}\Omega^{-1}Q\right]
$$
$$
\Sigma_{\text{BL}} = \left[(\tau\Sigma)^{-1} + P^{\top}\Omega^{-1}P\right]^{-1}
$$
where $\tau$ controls confidence in prior.

**Black-Litterman Tangency:**
$$
w^{\text{BL}} = \frac{\Sigma_{\text{BL}}^{-1}(\mu_{\text{BL}} - r_f \mathbf{1})}{\mathbf{1}^{\top} \Sigma_{\text{BL}}^{-1}(\mu_{\text{BL}} - r_f \mathbf{1})}
$$

---

### GBM simulation
Single asset discretisation:
$$
S_{t+\Delta t} = S_t \exp\Big((\mu - \tfrac{1}{2}\sigma^2)\Delta t + \sigma \sqrt{\Delta t}\, Z_t\Big),\quad Z_t \sim \mathcal{N}(0,1).
$$
Portfolio value:
$$
V_t = w^{\top} S_t.
$$

---

### Historical vs simulated tangency levels
Normalise simulated paths $\tilde V^{(k)}_t$ to actual starting level: $V^{(k)}_t = V^{\text{hist}}_0 \tilde V^{(k)}_t$ and compare distribution with historical $V^{\text{hist}}_t$.

### Golden run fit metrics
Normalised MISE:
$$
	ext{NMISE}^{(k)} = \frac{\sum_t ( V^{(k)}_t - V^{\text{hist}}_t )^2}{\sum_t ( V^{\text{hist}}_t - \bar V^{\text{hist}} )^2}, \qquad \bar V^{\text{hist}} = \frac{1}{T}\sum_t V^{\text{hist}}_t.
$$
MAPE:
$$
	ext{MAPE}^{(k)} = \frac{100}{T} \sum_t \left| \frac{V^{(k)}_t - V^{\text{hist}}_t}{V^{\text{hist}}_t} \right|.
$$
Golden run = path with smallest NMISE.
"""

bond_math_md = r"""
**Underlying mathematics**

### Duration and Convexity

Modified duration measures price sensitivity to yield changes:
$$
D_{\text{mod}} = \frac{D}{1 + y}
$$
where $D$ is Macaulay duration and $y$ is yield to maturity.

Price change approximation:
$$
\frac{\Delta P}{P} \approx -D_{\text{mod}} \cdot \Delta y + \frac{1}{2} C \cdot (\Delta y)^2
$$
where $C$ is convexity.

### Portfolio Duration

Portfolio duration as weighted average:
$$
D_p = \sum_{i=1}^n w_i \cdot D_i
$$
where $w_i$ are portfolio weights and $D_i$ individual bond durations.

### Credit Risk Metrics

Credit spread over risk-free rate:
$$
s_i = y_i - r_{\text{rf}}(T_i)
$$
where $y_i$ is bond yield, $r_{\text{rf}}(T_i)$ is treasury yield at maturity $T_i$.

Expected loss estimation:
$$
\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}
$$
PD = Probability of Default, LGD = Loss Given Default, EAD = Exposure at Default.

### Yield Curve Analysis

**Nelson-Siegel yield curve model (Nelson & Siegel, 1987):**
$$
y(\tau) = \beta_0 + \beta_1 \left(\frac{1 - e^{-\lambda \tau}}{\lambda \tau}\right) + \beta_2 \left(\frac{1 - e^{-\lambda \tau}}{\lambda \tau} - e^{-\lambda \tau}\right)
$$

**Factor interpretation:**
- $\beta_0$ = long-term level factor
- $\beta_1$ = slope factor  
- $\beta_2$ = curvature factor
- $\lambda$ = decay parameter (> 0)

**Loading functions:**
- $L_1(\tau) = 1$ (constant, level)
- $L_2(\tau) = \frac{1 - e^{-\lambda \tau}}{\lambda \tau}$ (monotonic, slope)
- $L_3(\tau) = \frac{1 - e^{-\lambda \tau}}{\lambda \tau} - e^{-\lambda \tau}$ (hump-shaped, curvature)

**Limit behavior:**
- As $\tau \to 0$: $y(0) = \beta_0 + \beta_1$ (short rate)
- As $\tau \to \infty$: $y(\infty) = \beta_0$ (long rate)
- $L_2(0) = 1$, $L_2(\infty) = 0$
- $L_3(0) = 0$, $L_3(\infty) = 0$, $L_3$ peaks at $\tau^* = \ln(2)/\lambda$

**Svensson extension (4-factor):**
$$
y(m) = \beta_0 + \beta_1 \frac{1 - e^{-\lambda_1 m}}{\lambda_1 m} + \beta_2 \left(\frac{1 - e^{-\lambda_1 m}}{\lambda_1 m} - e^{-\lambda_1 m}\right) + \beta_3 \left(\frac{1 - e^{-\lambda_2 m}}{\lambda_2 m} - e^{-\lambda_2 m}\right)
$$

### Performance Attribution

Return decomposition:
$$
R_p = \sum_i w_i R_i = \sum_i w_i (\text{Duration Effect}_i + \text{Credit Effect}_i + \text{Selection Effect}_i)
$$
Duration effect captures interest rate risk, credit effect captures spread changes.
"""

risk_math_md = r"""
**Risk Management Mathematics**

### Value at Risk (VaR)

**Historical VaR**: $(1-\alpha)$ percentile of return distribution
$$
\text{VaR}_\alpha = -F^{-1}(\alpha)
$$
where $F^{-1}$ is the inverse CDF of portfolio returns.

**Parametric VaR**: Assumes normal distribution
$$
\text{VaR}_\alpha = -(\mu + \sigma \Phi^{-1}(\alpha))
$$
where $\Phi^{-1}$ is the inverse standard normal CDF.

**Expected Shortfall (CVaR)**: Expected loss beyond VaR
$$
\text{ES}_\alpha = -E[R | R \leq -\text{VaR}_\alpha]
$$

### Factor Risk Attribution

Portfolio variance decomposition:
$$
\sigma_p^2 = w^T \Sigma w = \sum_{i,j} w_i w_j \sigma_{ij}
$$

**Factor model**: $R_i = \alpha_i + \sum_k \beta_{ik} F_k + \epsilon_i$

Portfolio factor exposure:
$$
\beta_{pk} = \sum_i w_i \beta_{ik}
$$

Factor contribution to risk:
$$
\text{Risk Contrib}_k = \frac{\beta_{pk} \sigma_{kk} \beta_{pk}}{\sigma_p^2}
$$

### Concentration Risk Metrics

**Herfindahl-Hirschman Index (HHI)**:
$$
\text{HHI} = \sum_{i=1}^N w_i^2
$$
where $w_i$ are portfolio weights. Range: $[1/N, 1]$

**Effective Number of Holdings**:
$$
N_{\text{eff}} = \frac{1}{\text{HHI}}
$$

### Correlation Analysis

**Dynamic conditional correlation**:
$$
\rho_{ij,t} = \frac{Q_{ij,t}}{\sqrt{Q_{ii,t} Q_{jj,t}}}
$$

**Regime-dependent correlation**: $\rho_t = \rho_1 S_t + \rho_2 (1-S_t)$

### Stress Testing

**Scenario analysis**: Apply shocks $\Delta F$ to risk factors:
$$
\Delta P = \sum_k \frac{\partial P}{\partial F_k} \Delta F_k + \frac{1}{2} \sum_{k,l} \frac{\partial^2 P}{\partial F_k \partial F_l} \Delta F_k \Delta F_l
$$

**Monte Carlo**: Generate scenarios from multivariate distribution:
$$
\Delta F \sim \mathcal{N}(0, \Sigma_F)
$$
"""

individual_dataset_risk_math_md = r"""
**Individual Dataset Risk Analysis Mathematics**

### Equity Dataset: Cross-Sectional Return Analysis

**Cross-sectional return distribution**: For $N$ stocks at time $t$:
$$
R_{i,t} \sim \text{Distribution}(\mu_t, \sigma_t^2)
$$

**Cross-sectional moments**:
- Mean return: $\bar{R}_t = \frac{1}{N} \sum_{i=1}^N R_{i,t}$
- Cross-sectional volatility: $\sigma_{cs,t} = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (R_{i,t} - \bar{R}_t)^2}$
- Percentiles: $P_\alpha = F^{-1}(\alpha)$ where $F$ is empirical CDF

**Factor Score Analysis**: Average factor exposures across universe:
$$
\bar{\beta}_k = \frac{1}{N} \sum_{i=1}^N \beta_{i,k}
$$

Standard deviation of factor scores:
$$
\sigma_{\beta_k} = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (\beta_{i,k} - \bar{\beta}_k)^2}
$$

### Bond Dataset: Duration Risk Distribution

**Modified duration distribution**: For bond portfolio components:
$$
D_{\text{mod},i} = \frac{1}{P_i} \frac{\partial P_i}{\partial y}
$$

**Duration statistics across bonds**:
- Average duration: $\bar{D} = \frac{1}{M} \sum_{i=1}^M D_i$
- Duration volatility: $\sigma_D = \sqrt{\frac{1}{M-1} \sum_{i=1}^M (D_i - \bar{D})^2}$

**Interest rate sensitivity**: Price change for parallel yield shift:
$$
\frac{\Delta P_i}{P_i} \approx -D_{\text{mod},i} \Delta y + \frac{1}{2} C_i (\Delta y)^2
$$

where $C_i$ is convexity.

### Time Series Dataset: Individual Asset Volatility

**Rolling volatility estimation** for asset $i$:
$$
\hat{\sigma}_{i,t} = \sqrt{\frac{252}{T} \sum_{s=t-T+1}^t (r_{i,s} - \bar{r}_{i,t})^2}
$$

**Cross-asset volatility statistics**:
- Mean volatility: $\bar{\sigma} = \frac{1}{K} \sum_{i=1}^K \sigma_i$  
- Volatility dispersion: $\sigma_{\sigma} = \sqrt{\frac{1}{K-1} \sum_{i=1}^K (\sigma_i - \bar{\sigma})^2}$

**Risk ranking**: Assets sorted by $\sigma_i$ to identify highest/lowest risk

### Key Distinctions from Portfolio Analysis

❌ **NOT calculated**: Portfolio-level VaR, unified correlation matrix, aggregate factor exposures
✅ **Calculated**: Individual asset/security risk characteristics, cross-sectional distributions, dataset-specific risk metrics

Each dataset represents independent analytical universe requiring separate risk assessment methodologies.
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
\text{Valuation_vs_Peers}_i
=
\frac{M_i − M^{\text{sector}}_s}{M^{\text{sector}}_s}.
$$

Negative values correspond to discounts versus sector peers and positive values to premia.

The peer dispersion plot groups $\text{Valuation\_vs\_Peers}_i$ by sector and displays their distribution, highlighting sectors where valuation is especially stretched or compressed relative to internal peers. This ties directly back to the factor and regression views,
since extreme valuation outliers often align with strong value or growth factor exposures.
"""


# ---------- Shared card helper ----------

def card(children, title=None):
    """Bloomberg-style card component with terminal aesthetics."""
    header = (
        html.Div(
            style={
                "borderBottom": f"1px solid {THEME['border']}",
                "paddingBottom": "10px",
                "marginBottom": "16px",
                "background": f"linear-gradient(90deg, transparent, {THEME['accent']}22, transparent)",
                "position": "relative"
            },
            children=[
                html.Div(
                    title,
                    style={
                        "color": THEME["accent"],
                        "fontWeight": "700",
                        "fontSize": "13px",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.8px",
                        "fontFamily": "'Bloomberg', monospace"
                    },
                ),
                # Accent underline
                html.Div(style={
                    "position": "absolute",
                    "bottom": "0",
                    "left": "0",
                    "width": "40px",
                    "height": "2px",
                    "background": f"linear-gradient(90deg, {THEME['accent']}, {THEME['accent2']})"
                })
            ]
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

    _settings = get_runtime_settings()
    return html.Div(
        style={
            "backgroundColor": THEME["card"],
            "border": f"1px solid {THEME['border']}",
            "borderRadius": "6px",
            "padding": _settings["CARD"]["padding"],
            "minHeight": _settings["CARD"]["min_height"],
            "boxShadow": "0 4px 16px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05)",
            "background": f"linear-gradient(135deg, {THEME['card']}, #1a1a1a)",
            "display": "flex",
            "flexDirection": "column",
            "position": "relative",
            "overflow": "hidden"
        },
        children=[
            *flat_children,
            # Bloomberg corner accent
            html.Div(style={
                "position": "absolute",
                "top": "0",
                "right": "0",
                "width": "2px",
                "height": "100%",
                "background": f"linear-gradient(180deg, {THEME['accent']}, {THEME['accent2']})",
                "opacity": "0.6"
            })
        ]
    )


# ---------- Portfolio tab ----------

def layout_tab_portfolio():
    # maths explainer at top of portfolio tab
    settings = get_runtime_settings()
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
                                        min=settings["FRONTIER"]["min"],
                                        max=settings["FRONTIER"]["max"],
                                        step=settings["FRONTIER"]["step"],
                                        value=settings["FRONTIER"]["default"],
                                        marks={
                                            settings["FRONTIER"]["min"]: str(settings["FRONTIER"]["min"]),
                                            50: "50",
                                            100: "100",
                                            150: "150",
                                            settings["FRONTIER"]["max"]: str(settings["FRONTIER"]["max"]),
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
                                        n_points=settings["FRONTIER"]["default"],
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
                                        min=settings["SIM_STEPS"]["min"],
                                        max=settings["SIM_STEPS"]["max"],
                                        step=settings["SIM_STEPS"]["step"],
                                        value=settings["SIM_STEPS"]["default"],
                                        marks={
                                            settings["SIM_STEPS"]["min"]: str(settings["SIM_STEPS"]["min"]),
                                            100: "100",
                                            252: "252",
                                            settings["SIM_STEPS"]["max"]: str(settings["SIM_STEPS"]["max"]),
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
                                        min=settings["SIM_NPATHS"]["min"],
                                        max=settings["SIM_NPATHS"]["max"],
                                        step=settings["SIM_NPATHS"]["step"],
                                        value=settings["SIM_NPATHS"]["default"],
                                        marks={
                                            settings["SIM_NPATHS"]["min"]: str(settings["SIM_NPATHS"]["min"]),
                                            100: "100",
                                            500: "500",
                                            settings["SIM_NPATHS"]["max"]: str(settings["SIM_NPATHS"]["max"]),
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
                                        min=settings["SIM_MAXPATHS"]["min"],
                                        max=settings["SIM_MAXPATHS"]["max"],
                                        step=settings["SIM_MAXPATHS"]["step"],
                                        value=settings["SIM_MAXPATHS"]["default"],
                                        marks={
                                            settings["SIM_MAXPATHS"]["min"]: str(settings["SIM_MAXPATHS"]["min"]),
                                            50: "50",
                                            100: "100",
                                            settings["SIM_MAXPATHS"]["max"]: str(settings["SIM_MAXPATHS"]["max"]),
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
                                            {
                                                "label": "Black-Litterman",
                                                "value": "black_litterman",
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
                                        n_steps=settings["SIM_STEPS"]["default"],
                                        n_paths=settings["SIM_NPATHS"]["default"],
                                        portfolio_type="tangency",
                                        max_paths=settings["SIM_MAXPATHS"]["default"],
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
                                        min=settings["SIM_STEPS"]["min"],
                                        max=settings["SIM_STEPS"]["max"],
                                        step=settings["SIM_STEPS"]["step"],
                                        value=252,
                                        marks={
                                            settings["SIM_STEPS"]["min"]: str(settings["SIM_STEPS"]["min"]),
                                            126: "126",
                                            252: "252",
                                            settings["SIM_STEPS"]["max"]: str(settings["SIM_STEPS"]["max"]),
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
    if data.cluster_metrics is not None:
        metrics_text.append(f"Silhouette: {data.cluster_metrics.silhouette:.3f}")
        metrics_text.append(
            f"Calinski Harabasz: {data.cluster_metrics.calinski_harabasz:.1f}"
        )
        metrics_text.append(
            f"Davies Bouldin: {data.cluster_metrics.davies_bouldin:.3f}"
        )
        metrics_text.append(f"Inertia: {data.cluster_metrics.inertia:.1f}")
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
                                        f"All factors: MAPE {data.mape_all:.2f}%, "
                                        f"NMISE {data.nmise_all:.4f}, R² {data.r2_all:.3f}",
                                        style={
                                            "color": THEME["text"],
                                            "fontSize": "12px",
                                        },
                                    ),
                                    html.Li(
                                        f"Momentum only: MAPE {data.mape_mom:.2f}%, "
                                        f"NMISE {data.nmise_mom:.4f}, R² {data.r2_mom:.3f}",
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


# ---------- Bond analysis tab ----------

def layout_tab_bonds():
    """Bond portfolio analysis tab with fixed income analytics."""
    # Math explainer at top
    math_block = card(
        title="Underlying mathematics",
        children=dcc.Markdown(
            bond_math_md,
            mathjax=True,
            style={
                "fontSize": "12px",
                "color": THEME["muted"],
                "whiteSpace": "pre-wrap",
            },
        ),
    )

    # Portfolio summary stats
    if data.is_ready():
        portfolio_risk = data.bond_portfolio_risk
        duration_stats = data.bond_duration_stats
        
        summary_text = [
            f"Portfolio Duration: {portfolio_risk['portfolio_duration']:.2f}",
            f"Portfolio Yield: {portfolio_risk['portfolio_yield']*100:.2f}%",
            f"Number of Bonds: {portfolio_risk['number_of_bonds']}",
            f"Total Face Value: ${portfolio_risk['total_face_value']/1e6:.1f}M",
        ]
        summary_joined = " | ".join(summary_text)
    else:
        summary_joined = "Loading bond data..."

    # Row 1: Yield curve and duration analysis
    row1 = html.Div(
        style={"display": "flex", "gap": "24px", "height": "500px"},
        children=[
            html.Div(
                style={"flex": "1", "height": "100%"},
                children=[
                    card(
                        title="Yield Curve Analysis",
                        children=[
                            dcc.Graph(
                                id="yield-curve-graph",
                                figure=make_yield_curve_figure(),
                                config=GRAPH_CONFIG,
                                style={"flex": "1 1 auto", "height": "100%", "width": "100%"},
                            ),
                        ],
                    )
                ],
            ),
            html.Div(
                style={"flex": "1", "height": "100%"},
                children=[
                    card(
                        title="Duration Risk Analysis",
                        children=[
                            dcc.Graph(
                                id="duration-graph",
                                figure=make_duration_distribution_figure(),
                                style={"flex": "1 1 auto", "height": "100%", "width": "100%"},
                                config=GRAPH_CONFIG,
                            ),
                        ],
                    )
                ],
            ),
        ],
    )

    # Row 2: Credit analysis and performance
    row2 = html.Div(
        style={"display": "flex", "gap": "24px"},
        children=[
            html.Div(
                style={"flex": "1"},
                children=[
                    card(
                        title="Credit Quality Analysis",
                        children=[
                            dcc.Graph(
                                id="credit-analysis-graph",
                                figure=make_credit_analysis_figure(),
                                style={"height": "600px", "width": "100%"},
                                config=GRAPH_CONFIG,
                            ),
                        ],
                    )
                ],
            ),
        ],
    )

    # Row 3: Performance analysis
    row3 = html.Div(
        style={"display": "flex", "gap": "24px"},
        children=[
            html.Div(
                style={"flex": "1"},
                children=[
                    card(
                        title="Performance Attribution",
                        children=[
                            html.Div(
                                summary_joined,
                                style={
                                    "color": THEME["muted"],
                                    "fontSize": "11px",
                                    "marginBottom": "8px",
                                },
                            ),
                            dcc.Graph(
                                id="bond-performance-graph",
                                figure=make_bond_performance_figure(),
                                style={"height": "450px", "width": "100%"},
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
                "Fixed income portfolio analysis",
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


def layout_tab_risk():
    """Individual dataset risk analysis tab (CORRECTED - not unified portfolio)."""
    # Math explainer at top
    math_block = card(
        title="Individual Dataset Risk Analysis Mathematics",
        children=dcc.Markdown(
            individual_dataset_risk_math_md,
            mathjax=True,
            style={
                "fontSize": "12px",
                "color": THEME["muted"],
                "whiteSpace": "pre-wrap",
            },
        ),
    )

    # Row 1: Main risk dashboard for individual datasets
    row1 = html.Div(
        style={"display": "flex", "gap": "24px"},
        children=[
            html.Div(
                style={"flex": "1"},
                children=[
                    card(
                        title="Individual Dataset Risk Analysis (Not Unified Portfolio)",
                        children=[
                            dcc.Graph(
                                id="risk-dashboard-graph",
                                figure=make_risk_dashboard_figure(),
                                style={"height": "800px", "width": "100%"},
                                config=GRAPH_CONFIG,
                            ),
                        ],
                    )
                ],
            ),
        ],
    )

    # Row 2: Correlation analysis and bond duration risk
    row2 = html.Div(
        style={"display": "flex", "gap": "24px"},
        children=[
            html.Div(
                style={"flex": "1"},
                children=[
                    card(
                        title="Factor & Metric Correlations",
                        children=[
                            dcc.Graph(
                                id="correlation-heatmap-graph", 
                                figure=make_correlation_heatmap_figure(),
                                style={"height": "600px", "width": "100%"},
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
                        title="Bond Duration Risk Analysis",
                        children=[
                            dcc.Graph(
                                id="bond-duration-risk-graph",
                                figure=make_bond_duration_risk_figure(),
                                style={"height": "600px", "width": "100%"},
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
                "Individual dataset risk analysis (equity, bonds, time series analyzed separately)",
                style={
                    "color": THEME["muted"],
                    "marginBottom": "4px",
                    "fontSize": "14px",
                },
            ),
            math_block,
            row1,
            row2,
        ],
    )


# Bloomberg loading and error components
def bloomberg_loading_overlay() -> html.Div:
    """Bloomberg-style loading overlay."""
    return html.Div(
        id="bloomberg-loading",
        className="bloomberg-loading",
        style={"display": "none"},
        children=[
            html.Div(className="bloomberg-spinner"),
            html.P("LOADING ANALYTICS...", style={
                "color": THEME["accent"],
                "fontSize": "12px",
                "fontWeight": "bold",
                "margin": "0",
                "letterSpacing": "1px"
            })
        ]
    )

def bloomberg_error_modal() -> html.Div:
    """Bloomberg-style error modal."""
    return html.Div(
        id="error-modal",
        style={"display": "none"},
        children=[
            html.Div(
                style={
                    "position": "fixed",
                    "top": "0",
                    "left": "0",
                    "width": "100%",
                    "height": "100%",
                    "backgroundColor": "rgba(0, 0, 0, 0.8)",
                    "zIndex": "9999",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center"
                },
                children=[
                    html.Div(
                        style={
                            "backgroundColor": THEME["card"],
                            "border": f"2px solid #ff4444",
                            "borderRadius": "8px",
                            "padding": "30px",
                            "maxWidth": "500px",
                            "textAlign": "center"
                        },
                        children=[
                            html.H3("⚠️ SYSTEM ERROR", style={
                                "color": "#ff4444",
                                "fontSize": "16px",
                                "marginBottom": "15px"
                            }),
                            html.P(id="error-message", style={
                                "color": THEME["text"],
                                "fontSize": "12px"
                            }),
                            html.Button("DISMISS", 
                                       id="error-dismiss",
                                       style={
                                           "backgroundColor": THEME["accent"],
                                           "color": "#000",
                                           "border": "none",
                                           "padding": "8px 16px",
                                           "borderRadius": "4px",
                                           "fontSize": "11px",
                                           "fontWeight": "bold",
                                           "cursor": "pointer"
                                       })
                        ]
                    )
                ]
            )
        ]
    )

def bloomberg_shortcuts_help() -> html.Div:
    """Bloomberg-style keyboard shortcuts help."""
    return html.Div(
        className="bloomberg-shortcut",
        children=[
            html.P("HOTKEYS: F1-Portfolio | F2-Equity | F3-Bonds | F4-Risk", style={
                "margin": "0",
                "color": THEME["text"],
                "fontSize": "9px"
            })
        ]
    )

# ---------- Root layout and index_string ----------

def create_root_layout():
    _settings = get_runtime_settings()
    # Settings panel (collapsible) components
    settings_panel = html.Div(
        id="settings-container",
        style={
            "display": "none",
            "backgroundColor": THEME["card"],
            "border": f"1px solid {THEME['border']}",
            "borderRadius": "4px",
            "padding": _settings["CARD"]["padding"],
            "marginBottom": "12px",
        },
        children=[
            html.Div(
                "Runtime settings snapshot",
                style={
                    "color": THEME["accent"],
                    "fontWeight": "bold",
                    "marginBottom": "6px",
                },
            ),
            html.Pre(
                id="settings-json",
                style={
                    "fontSize": "11px",
                    "color": THEME["muted"],
                    "whiteSpace": "pre-wrap",
                    "margin": 0,
                },
            ),
            html.Button(
                "Refresh settings",
                id="settings-refresh",
                n_clicks=0,
                style={
                    "marginTop": "8px",
                    "backgroundColor": THEME["accent"],
                    "color": "#000",
                    "border": "none",
                    "padding": "4px 10px",
                    "cursor": "pointer",
                    "fontWeight": "bold",
                },
            ),
        ],
    )

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
            "paddingBottom": "35px",  # Add padding to avoid overlap with status bar
            "display": "flex",
            "flexDirection": "column",
        },
        children=[
            html.Div(
                _settings["APP_TITLE"],
                style={
                    "fontSize": "22px",
                    "fontWeight": "bold",
                    "color": THEME["accent"],
                    "marginBottom": "6px",
                },
            ),
            html.Div(
                style={"display": "flex", "gap": "8px", "marginBottom": "12px"},
                children=[
                    html.Button(
                        "Toggle settings",
                        id="settings-toggle",
                        n_clicks=0,
                        style={
                            "backgroundColor": THEME["accent2"],
                            "color": "#000",
                            "border": "none",
                            "padding": "4px 10px",
                            "cursor": "pointer",
                            "fontWeight": "bold",
                        },
                    ),
                    html.Button(
                        "Clear caches",
                        id="clear-caches",
                        n_clicks=0,
                        style={
                            "backgroundColor": THEME["accent"],
                            "color": "#000",
                            "border": "none",
                            "padding": "4px 10px",
                            "cursor": "pointer",
                            "fontWeight": "bold",
                        },
                    ),
                    html.Div(
                        id="clear-caches-status",
                        style={"color": THEME["muted"], "fontSize": "11px", "alignSelf": "center"},
                    ),
                    html.Div(
                        "Environment + runtime configuration (view only)",
                        style={"color": THEME["muted"], "fontSize": "12px", "alignSelf": "center"},
                    ),
                ],
            ),
            settings_panel,
            # Bloomberg-style header section
            html.Div(
                style={
                    "background": f"linear-gradient(135deg, {THEME['card']}, {THEME['panel']})",
                    "border": f"1px solid {THEME['border']}",
                    "borderRadius": "6px",
                    "padding": "16px 20px",
                    "marginBottom": "20px",
                    "boxShadow": f"0 2px 10px rgba(248, 231, 28, 0.1)"
                },
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"},
                        children=[
                            html.Div([
                                html.H2(
                                    "📈 BLOOMBERG TERMINAL",
                                    style={
                                        "color": THEME["accent"],
                                        "fontSize": "20px",
                                        "fontWeight": "700",
                                        "letterSpacing": "1.2px",
                                        "margin": "0",
                                        "fontFamily": "'Bloomberg', monospace",
                                        "textShadow": f"0 0 20px {THEME['accent']}, 0 0 40px {THEME['accent']}",
                                        "animation": "terminal-flicker 3s infinite"
                                    }
                                ),
                                html.P(
                                    "Professional Analytics & Portfolio Management Suite",
                                    style={
                                        "color": THEME["text"],
                                        "fontSize": "12px",
                                        "margin": "4px 0 0 0",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.5px"
                                    }
                                )
                            ]),
                            html.Div([
                                html.Div(
                                    "●",
                                    style={
                                        "color": THEME["accent2"],
                                        "fontSize": "14px",
                                        "marginRight": "8px",
                                        "display": "inline-block"
                                    }
                                ),
                                html.Span(
                                    "LIVE",
                                    style={
                                        "color": THEME["accent2"],
                                        "fontSize": "11px",
                                        "fontWeight": "700",
                                        "letterSpacing": "1px"
                                    }
                                )
                            ], style={"display": "flex", "alignItems": "center"})
                        ]
                    )
                ]
            ),
            dcc.Tabs(
                id="tabs",
                value="tab-portfolio",
                parent_className="custom-tabs",
                className="custom-tabs-container",
                style={
                    "backgroundColor": THEME["bg"],
                    "border": "none",
                    "borderRadius": "8px 8px 0 0",
                    "overflow": "hidden"
                },
                children=[
                    dcc.Tab(
                        label="📊 PORTFOLIO OPTIMIZATION",
                        value="tab-portfolio",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        label="📈 EQUITY ANALYTICS",
                        value="tab-other",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        label="🏦 FIXED INCOME",
                        value="tab-bonds",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        label="⚠️ RISK ANALYTICS",
                        value="tab-risk",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                ]
            ),
            html.Div(
                id="tabs-content",
                style={
                    "marginTop": "12px",
                    "flex": "1",
                    "overflowY": "auto",
                },
            ),
            # Bloomberg enhancement components
            bloomberg_loading_overlay(),
            bloomberg_error_modal(),
            bloomberg_shortcuts_help(),
            # Bloomberg-style status bar
            html.Div(
                style={
                    "position": "fixed",
                    "bottom": "0",
                    "left": "0",
                    "right": "0",
                    "height": "28px",
                    "backgroundColor": THEME["panel"],
                    "borderTop": f"1px solid {THEME['border']}",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "padding": "0 16px",
                    "zIndex": "1000",
                    "fontSize": "11px",
                    "fontFamily": "'Bloomberg', monospace"
                },
                children=[
                    html.Div([
                        html.Span("BLOOMBERG TERMINAL", style={"color": THEME["accent"], "fontWeight": "bold", "marginRight": "12px"}),
                        html.Span("●", style={"color": "#00ff00", "marginRight": "4px"}),
                        html.Span("CONNECTED", style={"color": THEME["text"]}),
                    ]),
                    html.Div([
                        html.Span("MARKET STATUS: ", style={"color": THEME["muted"]}),
                        html.Span("LIVE", style={
                            "color": THEME["accent2"], 
                            "fontWeight": "bold",
                            "textShadow": f"0 0 10px {THEME['accent2']}",
                            "animation": "pulse 2s infinite"
                        }),
                        html.Span(" | ", style={"color": THEME["muted"], "margin": "0 8px"}),
                        html.Span("ANALYTICS READY", style={
                            "color": "#00ff00",
                            "textShadow": "0 0 8px #00ff00"
                        }),
                        html.Span(" | ", style={"color": THEME["muted"], "margin": "0 8px"}),
                        html.Span("USD/EUR: 1.0842", style={
                            "color": "#00ff88",
                            "fontSize": "10px",
                            "fontFamily": "monospace"
                        }),
                    ], style={"display": "flex", "alignItems": "center"})
                ]
            ),
        ]
    )


_app_title = get_runtime_settings()["APP_TITLE"]
index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>""" + _app_title + """</title>
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
                border-bottom: 1px solid """ + THEME["border"] + """;
                margin-bottom: 0px;
            }
            .custom-tab {
                background: linear-gradient(135deg, """ + THEME["bg"] + """, #0a0a0a) !important;
                border: 1px solid #2a2a2a !important;
                border-bottom: none !important;
                padding: 12px 20px !important;
                margin: 0px 2px !important;
                border-radius: 6px 6px 0px 0px !important;
                font-family: 'Bloomberg', 'Arial', sans-serif !important;
                font-weight: 600 !important;
                font-size: 13px !important;
                text-transform: uppercase !important;
                letter-spacing: 0.8px !important;
                color: #cccccc !important;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
                cursor: pointer !important;
                min-width: 140px !important;
                text-align: center !important;
                position: relative !important;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
            }
            .custom-tab:hover {
                background: linear-gradient(135deg, #1a1a1a, """ + THEME["panel"] + """) !important;
                color: """ + THEME["accent2"] + """ !important;
                border-color: """ + THEME["accent2"] + """ !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 16px rgba(0, 230, 255, 0.25), 0 0 20px rgba(0, 230, 255, 0.1) !important;
                text-shadow: 0 0 10px rgba(0, 230, 255, 0.5) !important;
            }
            .custom-tab--selected {
                background: linear-gradient(135deg, #1e1e1e, """ + THEME["card"] + """) !important;
                border-color: """ + THEME["accent"] + """ !important;
                border-bottom: 3px solid """ + THEME["accent"] + """ !important;
                color: """ + THEME["accent"] + """ !important;
                font-weight: 700 !important;
                box-shadow: 0 6px 20px rgba(248, 231, 28, 0.3), 0 0 30px rgba(248, 231, 28, 0.15), inset 0 1px 0 rgba(248, 231, 28, 0.2) !important;
                transform: translateY(-3px) !important;
                text-shadow: 0 0 15px rgba(248, 231, 28, 0.6) !important;
            }
            .custom-tab--selected:before {
                content: '';
                position: absolute !important;
                bottom: -3px !important;
                left: 0 !important;
                right: 0 !important;
                height: 3px !important;
                background: linear-gradient(90deg, """ + THEME["accent"] + """, """ + THEME["accent2"] + """) !important;
                animation: glow 2s ease-in-out infinite alternate !important;
            }
            @keyframes glow {
                from { box-shadow: 0 0 5px """ + THEME["accent"] + """; }
                to { box-shadow: 0 0 10px """ + THEME["accent"] + """, 0 0 15px """ + THEME["accent2"] + """; }
            }
            .custom-tabs-container .tab-content {
                border: none !important;
                background-color: """ + THEME["bg"] + """ !important;
                padding: 0 !important;
            }
            /* Bloomberg-style active indicator */
            .custom-tab--selected:after {
                content: '●' !important;
                position: absolute !important;
                top: 4px !important;
                right: 8px !important;
                color: """ + THEME["accent"] + """ !important;
                font-size: 8px !important;
                animation: pulse 1.5s ease-in-out infinite !important;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.7; transform: scale(1.2); }
            }
            @keyframes terminal-flicker {
                0%, 100% { opacity: 1; }
                98% { opacity: 1; }
                99% { opacity: 0.8; }
            }
            @keyframes data-flow {
                0% { transform: translateX(-100%); opacity: 0; }
                50% { opacity: 1; }
                100% { transform: translateX(100%); opacity: 0; }
            }
            /* Bloomberg terminal-style typography */
            body {
                font-family: 'Bloomberg', 'Monaco', 'Menlo', 'Consolas', monospace !important;
                background-color: """ + THEME["bg"] + """ !important;
                background-image: 
                    radial-gradient(circle at 1px 1px, rgba(248, 231, 28, 0.02) 1px, transparent 0);
                background-size: 20px 20px;
                animation: terminal-flicker 5s infinite;
            }
            /* Bloomberg data table styling */
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table {
                border-collapse: separate !important;
                border-spacing: 1px !important;
                background-color: """ + THEME["bg"] + """ !important;
            }
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
                background-color: """ + THEME["panel"] + """ !important;
                color: """ + THEME["accent"] + """ !important;
                font-weight: 700 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.5px !important;
                font-size: 11px !important;
                border: 1px solid """ + THEME["border"] + """ !important;
            }
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td {
                background-color: """ + THEME["card"] + """ !important;
                color: """ + THEME["text"] + """ !important;
                border: 1px solid """ + THEME["border"] + """ !important;
                font-family: 'Bloomberg', monospace !important;
                font-size: 10px !important;
            }
            /* Bloomberg scrollbar styling */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            ::-webkit-scrollbar-track {
                background: """ + THEME["bg"] + """;
            }
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(""" + THEME["accent"] + """, """ + THEME["accent2"] + """);
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(""" + THEME["accent2"] + """, """ + THEME["accent"] + """);
            }
            .custom-tab .tab-label {
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                gap: 6px !important;
            }
            /* Additional Bloomberg styling */
            .custom-tabs-container > .tabs-and-content-wrapper {
                border: none !important;
                background-color: """ + THEME["bg"] + """ !important;
            }
            /* Bloomberg keyboard shortcuts and interactions */
            .bloomberg-shortcut {
                position: fixed;
                bottom: 40px;
                right: 20px;
                background: linear-gradient(135deg, """ + THEME["panel"] + """, #2a2a2a);
                border: 1px solid """ + THEME["accent"] + """;
                padding: 8px 12px;
                border-radius: 6px;
                color: """ + THEME["text"] + """;
                font-size: 10px;
                font-family: 'Bloomberg', monospace;
                z-index: 1001;
                opacity: 0.8;
            }
            .bloomberg-loading {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: """ + THEME["panel"] + """;
                border: 2px solid """ + THEME["accent"] + """;
                padding: 30px;
                border-radius: 8px;
                z-index: 9999;
                text-align: center;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.7);
            }
            .bloomberg-spinner {
                border: 3px solid """ + THEME["border"] + """;
                border-top: 3px solid """ + THEME["accent"] + """;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
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
