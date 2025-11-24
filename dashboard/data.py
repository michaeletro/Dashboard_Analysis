from __future__ import annotations

import os
import numpy as np
import pandas as pd

from equity_data_frame_module.clustering import FactorClusterer
from equity_data_frame_module.regression import CrossSectionalRegressor
from equity_data_frame_module.peer_analysis import PeerAnalyzer
from equity_data_frame_module.factor_analysis import FactorAnalyzer
from equity_data_frame_module import DataLoader

from time_series_modules.StochasticPortfolioEngine import (
    StochasticPortfolioEngine,
)
from common.metrics import nmise_mape

# ---------------------------------------------------------------------
# Load data from Excel
# ---------------------------------------------------------------------

def _make_synthetic_prices(n_assets: int = 6, n_days: int = 252 * 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.08, 0.04, size=n_assets)   # annual drift
    sigma = rng.uniform(0.15, 0.35, size=n_assets)  # annual vol
    dt = 1.0 / 252.0

    S0 = rng.uniform(50, 200, size=n_assets)
    prices = np.zeros((n_days, n_assets), dtype=float)
    prices[0, :] = S0
    for t in range(1, n_days):
        z = rng.standard_normal(n_assets)
        incr = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        prices[t, :] = prices[t - 1, :] * np.exp(incr)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    cols = [f"Asset {i+1}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=dates, columns=cols)


excel_file_path = os.environ.get("DATA_XLSX", "../DBG Data Set Presentation Prep Doc.xlsx")

loader = DataLoader()
all_sheets = None
if os.path.exists(excel_file_path):
    all_sheets = loader.read_all_sheets(excel_file_path)
    if not all_sheets:
        all_sheets = None

if all_sheets is None:
    # Fallback to synthetic datasets
    df_time_series = _make_synthetic_prices()
    df_equity = loader.create_synthetic_data(n_stocks=150, seed=123)
else:
    # time series sheet for engine (third sheet)
    df_time_series = list(all_sheets.values())[2].copy()
    df_time_series.set_index(df_time_series.columns[0], inplace=True)

    # equity cross-section (first sheet)
    df_equity = all_sheets[list(all_sheets.keys())[0]].copy()

# ---------------------------------------------------------------------
# Time series engine and portfolio weights
# ---------------------------------------------------------------------

engine = StochasticPortfolioEngine(df_time_series, dt=1/252, rf=0.02, verbose=True)

# weights
w_tan = engine.tangency_weights().values

# realised portfolio level path from prices
real_path_level = engine.historical_portfolio_path(w_tan, normalise=False)
base_level = real_path_level[0]  # starting portfolio value implied by prices

# simulate normalised GBM portfolio paths
V_norm = engine.simulate_portfolio_paths(weights=w_tan, n_steps=len(real_path_level) - 1, n_paths=1000)

# apply growth rates to actual starting level
V_level = engine.apply_growth_to_level(V_norm, base_level=base_level)

# plot simulated vs realised in level terms
time_idx = engine.level.index[: V_level.shape[1]]

# tangency weights
w_tan_series = engine.tangency_weights()
w_tan = np.asarray(getattr(w_tan_series, "values", w_tan_series), dtype=float)

# Black Litterman tangency weights (optional)
try:
    # simple toy market portfolio: equal weight
    w_mkt = np.full(engine.n_assets, 1.0 / engine.n_assets)

    # single view on first asset
    P = np.zeros((1, engine.n_assets))
    P[0, 0] = 1.0

    # view: 5 percent annual excess return on that asset
    Q = np.array([0.05])

    w_bl_series = engine.black_litterman_tangency_weights(
        w_mkt=w_mkt,
        P=P,
        Q=Q,
    )
    w_bl = np.asarray(getattr(w_bl_series, "values", w_bl_series), dtype=float)
except Exception as bl_err:
    print(f"Black Litterman tangency weights not available: {bl_err}")
    w_bl = None


# ---------------------------------------------------------------------
# Equity cross section and factor construction
# ---------------------------------------------------------------------

factor_analyzer = FactorAnalyzer()
factor_scores = factor_analyzer.construct_factor_scores(df_equity)

for factor_name, scores in factor_scores.items():
    df_equity[factor_name] = scores

factor_cols = list(factor_scores.keys())

# ---------------------------------------------------------------------
# Clustering on factor space
# ---------------------------------------------------------------------

clusterer = FactorClusterer(
    n_clusters=4,
    random_state=42,
    verbose=False,
)

cluster_labels = clusterer.fit_predict(df_equity)
df_equity["FactorCluster"] = cluster_labels

cluster_report = clusterer.get_cluster_report()
cluster_metrics = cluster_report["metrics"] if cluster_report is not None else None

# ---------------------------------------------------------------------
# Cross sectional regression: all factors vs momentum only
# ---------------------------------------------------------------------

regressor = CrossSectionalRegressor()

# all factors
results_all, r2_all, coefs_all = regressor.fit(df_equity)
if results_all is None:
    raise RuntimeError("CrossSectionalRegressor.fit (all factors) returned None")

factors_all = results_all["Factor"].values
coefs_all_vec = results_all["Coefficient"].values

X_all = np.column_stack(
    [
        np.ones(len(df_equity)),
        df_equity[factors_all[1:]].values,  # skip intercept name
    ]
)
y_hat_all = X_all @ coefs_all_vec.reshape(-1, 1)

# momentum only
if "MomentumScore" not in df_equity.columns:
    raise RuntimeError(
        "MomentumScore column not found in df_equity. "
        "Ensure FactorAnalyzer constructed MomentumScore."
    )

results_mom, r2_mom, coefs_mom = regressor.fit(
    df_equity,
    factor_cols=["MomentumScore"],
)
if results_mom is None:
    raise RuntimeError("CrossSectionalRegressor.fit (momentum only) returned None")

factors_m = results_mom["Factor"].values
coefs_m_vec = results_mom["Coefficient"].values

X_m = np.column_stack(
    [
        np.ones(len(df_equity)),
        df_equity[factors_m[1:]].values,
    ]
)
y_hat_m = X_m @ coefs_m_vec.reshape(-1, 1)

# actual response (in percent)
y_actual = df_equity["1M % Change"].values

# use nmise_mape from the time series engine module
nmise_all, mape_all = nmise_mape(y_actual, y_hat_all.flatten() * 100.0)
nmise_mom, mape_mom = nmise_mape(y_actual, y_hat_m.flatten() * 100.0)

# ---------------------------------------------------------------------
# Peer analysis
# ---------------------------------------------------------------------

peer_analyzer = PeerAnalyzer()
df_equity = peer_analyzer.analyze(df_equity)
sector_summary = peer_analyzer.get_sector_summary(df_equity)
peer_analyzer_fig = peer_analyzer.plot_sector_dispersion(df_equity)
