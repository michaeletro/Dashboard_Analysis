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
from bond_series_modules.bond_analyzer import (
    BondAnalyzer,
    create_synthetic_bond_data,
    calculate_bond_portfolio_risk,
)
from common.metrics import nmise_mape
from .figures_portfolio import clear_fig_caches

# Public module attributes (populated lazily)
engine = None
w_tan = None
w_bl = None
df_equity = None
clusterer = None
cluster_metrics = None
results_all = None
r2_all = None
y_hat_all = None
results_mom = None
r2_mom = None
y_hat_m = None
y_actual = None
nmise_all = None
mape_all = None
nmise_mom = None
mape_mom = None
peer_analyzer = None
sector_summary = None
peer_analyzer_fig = None
# Bond analysis globals
bond_analyzer = None
df_bonds = None
bond_portfolio_risk = None
bond_duration_stats = None
bond_yield_curve = None
bond_credit_analysis = None

_INIT_DONE = False


def _make_synthetic_prices(n_assets: int = 6, n_days: int = 252 * 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.08, 0.04, size=n_assets)
    sigma = rng.uniform(0.15, 0.35, size=n_assets)
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


def ensure_data_loaded() -> None:
    """Initialise all heavy data and models once on demand."""
    global _INIT_DONE
    if _INIT_DONE:
        return

    global engine, w_tan, w_bl, df_equity
    global clusterer, cluster_metrics
    global results_all, r2_all, y_hat_all
    global results_mom, r2_mom, y_hat_m
    global y_actual, peer_analyzer, sector_summary, peer_analyzer_fig
    global bond_analyzer, df_bonds, bond_portfolio_risk
    global bond_duration_stats, bond_yield_curve, bond_credit_analysis

    excel_file_path = os.environ.get("DATA_XLSX", "../DBG Data Set Presentation Prep Doc.xlsx")

    loader = DataLoader()
    all_sheets = None
    if os.path.exists(excel_file_path):
        all_sheets = loader.read_all_sheets(excel_file_path)
        if not all_sheets:
            all_sheets = None

    if all_sheets is None:
        df_time_series = _make_synthetic_prices()
        df_equity = loader.create_synthetic_data(n_stocks=150, seed=123)
        df_bonds = create_synthetic_bond_data(n_bonds=80, seed=456)
    else:
        df_time_series = list(all_sheets.values())[2].copy()
        df_time_series.set_index(df_time_series.columns[0], inplace=True)
        df_equity = all_sheets[list(all_sheets.keys())[0]].copy()
        # Try to load bond data from second sheet, fallback to synthetic
        try:
            df_bonds = list(all_sheets.values())[1].copy()
        except (IndexError, KeyError):
            df_bonds = create_synthetic_bond_data(n_bonds=80, seed=456)

    # Engine and weights
    engine = StochasticPortfolioEngine(df_time_series, dt=1/252, rf=0.02, verbose=True)
    w_tan_series = engine.tangency_weights()
    w_tan = np.asarray(getattr(w_tan_series, "values", w_tan_series), dtype=float)

    try:
        w_mkt = np.full(engine.n_assets, 1.0 / engine.n_assets)
        P = np.zeros((1, engine.n_assets))
        P[0, 0] = 1.0
        Q = np.array([0.05])
        w_bl_series = engine.black_litterman_tangency_weights(w_mkt=w_mkt, P=P, Q=Q)
        w_bl = np.asarray(getattr(w_bl_series, "values", w_bl_series), dtype=float)
    except Exception as bl_err:
        print(f"Black Litterman tangency weights not available: {bl_err}")
        w_bl = None

    # Factors and clustering
    factor_analyzer = FactorAnalyzer()
    factor_scores = factor_analyzer.construct_factor_scores(df_equity)
    for factor_name, scores in factor_scores.items():
        df_equity[factor_name] = scores

    clusterer = FactorClusterer(n_clusters=4, random_state=42, verbose=False)
    cluster_labels = clusterer.fit_predict(df_equity)
    df_equity["FactorCluster"] = cluster_labels
    cluster_report = clusterer.get_cluster_report()
    cluster_metrics = cluster_report["metrics"] if cluster_report is not None else None

    # Cross-sectional regressions
    regressor = CrossSectionalRegressor()
    results_all, r2_all, coefs_all = regressor.fit(df_equity)
    if results_all is None:
        raise RuntimeError("CrossSectionalRegressor.fit (all factors) returned None")

    factors_all = results_all["Factor"].values
    coefs_all_vec = results_all["Coefficient"].values
    X_all = np.column_stack([np.ones(len(df_equity)), df_equity[factors_all[1:]].values])
    y_hat_all = X_all @ coefs_all_vec.reshape(-1, 1)

    if "MomentumScore" not in df_equity.columns:
        raise RuntimeError("MomentumScore column not found in df_equity. Ensure FactorAnalyzer constructed MomentumScore.")

    results_mom, r2_mom, coefs_mom = regressor.fit(df_equity, factor_cols=["MomentumScore"])
    if results_mom is None:
        raise RuntimeError("CrossSectionalRegressor.fit (momentum only) returned None")

    factors_m = results_mom["Factor"].values
    coefs_m_vec = results_mom["Coefficient"].values
    X_m = np.column_stack([np.ones(len(df_equity)), df_equity[factors_m[1:]].values])
    y_hat_m = X_m @ coefs_m_vec.reshape(-1, 1)

    y_actual = df_equity["1M % Change"].values
    # compute once for the summary
    _nm_all, _mp_all = nmise_mape(y_actual, y_hat_all.flatten() * 100.0)
    _nm_mom, _mp_mom = nmise_mape(y_actual, y_hat_m.flatten() * 100.0)
    # expose the computed values via globals with original names
    globals()["nmise_all"], globals()["mape_all"] = _nm_all, _mp_all
    globals()["nmise_mom"], globals()["mape_mom"] = _nm_mom, _mp_mom

    # Peer analysis
    peer_analyzer = PeerAnalyzer()
    df_equity = peer_analyzer.analyze(df_equity)
    sector_summary = peer_analyzer.get_sector_summary(df_equity)
    peer_analyzer_fig = peer_analyzer.plot_sector_dispersion(df_equity)

    # Bond analysis
    bond_analyzer = BondAnalyzer(verbose=True)
    bond_analyzer.load_data(df_bonds)
    bond_portfolio_risk = calculate_bond_portfolio_risk(df_bonds)
    bond_duration_stats = bond_analyzer.calculate_duration_statistics()
    bond_yield_curve = bond_analyzer.yield_curve_analysis()
    bond_credit_analysis = bond_analyzer.credit_analysis()

    _INIT_DONE = True


def is_ready() -> bool:
    return _INIT_DONE


def reset_cache() -> None:
    """Reset the module state (for debugging/tests)."""
    global _INIT_DONE
    for name in [
        "engine","w_tan","w_bl","df_equity","clusterer","cluster_metrics",
        "results_all","r2_all","y_hat_all","results_mom","r2_mom","y_hat_m",
        "y_actual","peer_analyzer","sector_summary","peer_analyzer_fig",
        "nmise_all","mape_all","nmise_mom","mape_mom",
        "bond_analyzer","df_bonds","bond_portfolio_risk","bond_duration_stats",
        "bond_yield_curve","bond_credit_analysis",
    ]:
        globals()[name] = None
    _INIT_DONE = False
    # also clear any cached figures
     # (avoid stale visuals referencing old data objects)
    try:
        clear_fig_caches()
    except Exception as _e:
        print(f"Warning: could not clear figure caches: {_e}")
