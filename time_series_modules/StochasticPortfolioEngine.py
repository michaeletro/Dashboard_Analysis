from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def nmise_mape(y_true, y_pred) -> Tuple[float, float]:
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

    return float(nmise), float(mape)


@dataclass
class GBMParams:
    mu: np.ndarray       # continuous drift per year
    sigma: np.ndarray    # volatility per year
    corr: np.ndarray     # correlation matrix


class StochasticPortfolioEngine:
    """
    End to end engine that operates on a DataFrame of price levels.

    Steps
    -----
    1  Fit from returns: estimate mean, covariance, correlation
    2  Compute tangency portfolio
    3  Compute Black Litterman posterior and tangency portfolio
    4  Calibrate GBM SDE parameters
    5  Simulate GBM paths and portfolio paths
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        dt: float = 1.0 / 252.0,
        rf: float = 0.0,
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        returns
            DataFrame of price levels, rows time index, columns assets
        dt
            Time step in years for each row
        rf
            Risk free rate per year, used for tangency portfolios
        verbose
            If True, print basic summaries
        """
        if returns.empty:
            raise ValueError("Input returns DataFrame (price levels) is empty.")

        # drop any rows with missing values across assets
        if returns.isna().any().any():
            returns = returns.dropna(axis=0, how="any")

        if returns.shape[0] < 3:
            raise ValueError("Need at least 3 observations to estimate returns.")

        # store level (price) data
        self.level = returns.copy()

        # simple returns from levels
        self.returns = returns.pct_change().dropna()

        if self.returns.shape[0] < 2:
            raise ValueError(
                "Not enough nonzero return observations after pct_change()."
            )

        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)
        self.dt = float(dt)
        self.rf = float(rf)
        self.verbose = bool(verbose)

        # basic estimates from simple returns per dt
        self.mu_hat = self.returns.mean().values     # per dt
        self.Sigma_hat = self.returns.cov().values   # per dt
        self.corr_hat = np.corrcoef(self.returns.values.T)

        # GBM parameters (continuous per year)
        self.gbm_params: Optional[GBMParams] = None

        # portfolio weights
        self.w_tangency_: Optional[np.ndarray] = None
        self.w_bl_: Optional[np.ndarray] = None

        if self.verbose:
            ann_mu = self.mu_hat / self.dt
            ann_vol = np.sqrt(np.diag(self.Sigma_hat)) / np.sqrt(self.dt)
            print("Fitted from returns")
            print("  assets:", self.assets)
            print("  annualised mean returns:", np.round(ann_mu, 4))
            print("  annualised volatilities:", np.round(ann_vol, 4))

    # ------------------------------------------------------------------
    # basic properties
    # ------------------------------------------------------------------

    @property
    def mu_annualised(self) -> np.ndarray:
        return self.mu_hat / self.dt

    @property
    def Sigma_annualised(self) -> np.ndarray:
        return self.Sigma_hat / self.dt

    # ------------------------------------------------------------------
    # Markowitz tangency portfolio
    # ------------------------------------------------------------------

    def tangency_weights(self) -> pd.Series:
        """
        Compute unconstrained tangency weights using sample mean and covariance.

        Returns
        -------
        pd.Series of weights indexed by asset name
        """
        n = self.n_assets
        ones = np.ones(n)

        # rf per dt
        rf_dt = self.rf * self.dt
        mu_excess = self.mu_hat - rf_dt * ones

        Sigma = self.Sigma_hat.copy()

        try:
            numer = np.linalg.solve(Sigma, mu_excess)
        except np.linalg.LinAlgError:
            Sigma_pinv = np.linalg.pinv(Sigma)
            numer = Sigma_pinv @ mu_excess

        denom = float(ones @ numer)
        if denom == 0.0:
            raise ValueError("Tangency denominator is zero, cannot normalise weights.")

        w = numer / denom
        self.w_tangency_ = w

        if self.verbose:
            print("\nTangency portfolio (sample)")
            self._print_portfolio_stats(w)

        return pd.Series(w, index=self.assets)

    # ------------------------------------------------------------------
    # Black Litterman
    # ------------------------------------------------------------------

    def black_litterman_posterior(
        self,
        w_mkt: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        tau: float = 0.025,
        lam: float = 2.5,
        Omega: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute Black Litterman posterior excess returns.

        Parameters
        ----------
        w_mkt
            Market portfolio weights (N vector)
        P
            View matrix shape K by N
        Q
            View returns (excess) of length K (per year)
        tau
            Prior uncertainty scale parameter
        lam
            Risk aversion parameter
        Omega
            K by K view covariance matrix. If None, uses diag of
            P (tau Sigma) P transpose.

        Returns
        -------
        Posterior excess return vector mu_bl (per year)
        """
        w_mkt = np.asarray(w_mkt, dtype=float).reshape(-1)
        P = np.asarray(P, dtype=float)
        Q = np.asarray(Q, dtype=float).reshape(-1)

        if w_mkt.shape[0] != self.n_assets:
            raise ValueError("w_mkt length must equal number of assets.")
        if P.shape[1] != self.n_assets:
            raise ValueError("P must have N columns equal to number of assets.")
        if P.shape[0] != Q.shape[0]:
            raise ValueError("P rows and Q length must match (same number of views).")

        # annualise Sigma from per dt (simple returns)
        Sigma_ann = self.Sigma_annualised

        # prior equilibrium excess returns
        pi = lam * Sigma_ann @ w_mkt

        if Omega is None:
            tauSigma = tau * Sigma_ann
            Omega = np.diag(np.diag(P @ tauSigma @ P.T))
        else:
            Omega = np.asarray(Omega, dtype=float)

        tauSigma_inv = np.linalg.inv(tau * Sigma_ann)
        Omega_inv = np.linalg.inv(Omega)

        A = tauSigma_inv + P.T @ Omega_inv @ P
        b = tauSigma_inv @ pi + P.T @ Omega_inv @ Q

        try:
            mu_bl = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            mu_bl = np.linalg.pinv(A) @ b

        return mu_bl

    def black_litterman_tangency_weights(
        self,
        w_mkt: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        tau: float = 0.025,
        lam: float = 2.5,
        Omega: Optional[np.ndarray] = None,
    ) -> pd.Series:
        """
        Compute tangency weights using Black Litterman posterior returns.
        """
        mu_bl = self.black_litterman_posterior(
            w_mkt=w_mkt,
            P=P,
            Q=Q,
            tau=tau,
            lam=lam,
            Omega=Omega,
        )

        rf_year = self.rf
        ones = np.ones(self.n_assets)
        mu_excess = mu_bl - rf_year * ones

        Sigma_ann = self.Sigma_annualised

        try:
            numer = np.linalg.solve(Sigma_ann, mu_excess)
        except np.linalg.LinAlgError:
            Sigma_pinv = np.linalg.pinv(Sigma_ann)
            numer = Sigma_pinv @ mu_excess

        denom = float(ones @ numer)
        if denom == 0.0:
            raise ValueError("Black Litterman tangency denominator is zero.")

        w_bl = numer / denom
        self.w_bl_ = w_bl

        if self.verbose:
            print("\nTangency portfolio (Black Litterman)")
            self._print_portfolio_stats(
                w_bl, mu_vec=mu_bl, Sigma=Sigma_ann, rf=rf_year
            )

        return pd.Series(w_bl, index=self.assets)

    # ------------------------------------------------------------------
    # GBM calibration and simulation
    # ------------------------------------------------------------------

    def calibrate_gbm(self) -> GBMParams:
        """
        Calibrate per asset GBM parameters from historical returns.

        Uses simple returns in self.returns, converts them to log returns
        and fits continuous GBM

            dS / S = mu dt + sigma dW

        Returns
        -------
        GBMParams dataclass with mu, sigma, corr (annualised)
        """
        r_simple = self.returns.values
        r_log = np.log1p(r_simple)  # log(1 + r)

        # mean and variance of log returns per dt
        m_log_dt = r_log.mean(axis=0)
        v_log_dt = r_log.var(axis=0, ddof=1)

        # scale to yearly drift and volatility
        sigma = np.sqrt(v_log_dt / self.dt)
        mu = m_log_dt / self.dt + 0.5 * sigma**2

        # correlation from log returns
        corr = np.corrcoef(r_log.T)
        params = GBMParams(mu=mu, sigma=sigma, corr=corr)

        self.gbm_params = params
        self.corr_hat = corr

        if self.verbose:
            print("\nGBM calibration (annualised)")
            print("  mu:", np.round(mu, 4))
            print("  sigma:", np.round(sigma, 4))

        return params

    def simulate_gbm_paths(
        self,
        n_steps: int,
        n_paths: int,
        S0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Simulate GBM paths for all assets.

        Parameters
        ----------
        n_steps
            Number of time steps to simulate
        n_paths
            Number of Monte Carlo paths
        S0
            Initial prices, length N. If None, uses all ones.

        Returns
        -------
        paths
            Array shape (n_paths, n_steps + 1, N)
        """
        if n_steps <= 0 or n_paths <= 0:
            raise ValueError("n_steps and n_paths must be positive integers.")

        if self.gbm_params is None:
            self.calibrate_gbm()

        assert self.gbm_params is not None
        mu = self.gbm_params.mu
        sigma = self.gbm_params.sigma
        corr = self.gbm_params.corr

        if S0 is None:
            S0 = np.ones(self.n_assets, dtype=float)
        else:
            S0 = np.asarray(S0, dtype=float)
            if S0.shape[0] != self.n_assets:
                raise ValueError("S0 length must equal number of assets.")

        # ensure correlation is positive semidefinite for Cholesky
        try:
            L = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            eps = 1e-10
            corr_reg = corr + eps * np.eye(self.n_assets)
            L = np.linalg.cholesky(corr_reg)

        paths = np.zeros((n_paths, n_steps + 1, self.n_assets), dtype=float)
        paths[:, 0, :] = S0

        dt_sim = self.dt
        drift = (mu - 0.5 * sigma**2) * dt_sim
        diff = sigma * np.sqrt(dt_sim)

        for t in range(1, n_steps + 1):
            z = np.random.normal(size=(n_paths, self.n_assets))
            z_corr = z @ L.T
            increment = drift + diff * z_corr
            paths[:, t, :] = paths[:, t - 1, :] * np.exp(increment)

        return paths

    def simulate_portfolio_paths(
        self,
        weights: np.ndarray,
        n_steps: int,
        n_paths: int,
        S0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Simulate GBM paths and aggregate into a portfolio.

        Parameters
        ----------
        weights
            N vector of portfolio weights
        n_steps
            Number of time steps
        n_paths
            Number of Monte Carlo paths
        S0
            Optional initial prices for assets. If None, uses ones.

        Returns
        -------
        V
            Array shape (n_paths, n_steps + 1) of portfolio values,
            normalised so that V[:, 0] is approximately 1
        """
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.shape[0] != self.n_assets:
            raise ValueError("weights length must equal number of assets.")

        paths = self.simulate_gbm_paths(
            n_steps=n_steps,
            n_paths=n_paths,
            S0=S0,
        )

        # portfolio value as weighted sum of asset prices
        S0_vec = paths[:, 0, :]
        V0 = float(np.sum(weights * S0_vec[0, :]))
        if V0 == 0.0:
            raise ValueError("Initial portfolio value V0 is zero, cannot normalise.")

        V = (paths @ weights) / V0
        return V

    # ------------------------------------------------------------------
    # Level integration: real path and growth application
    # ------------------------------------------------------------------

    def historical_portfolio_path(
        self,
        weights: np.ndarray,
        normalise: bool = False,
    ) -> np.ndarray:
        """
        Build realised portfolio value path from underlying price levels.

        Parameters
        ----------
        weights
            Length N vector of portfolio weights
        normalise
            If True, scale so that starting value is 1.0

        Returns
        -------
        Array of length T with portfolio values in level units,
        or normalised if normalise is True
        """
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.shape[0] != self.n_assets:
            raise ValueError("weights length must equal number of assets.")

        S = self.level.values  # T by N
        V = S @ weights        # T vector

        if normalise:
            if V[0] == 0.0:
                raise ValueError("Cannot normalise by zero starting value.")
            return V / V[0]

        return V

    def apply_growth_to_level(
        self,
        V: np.ndarray,
        base_level: float,
    ) -> np.ndarray:
        """
        Take simulated growth paths (starting at 1) and map them to a real level.

        Parameters
        ----------
        V
            Array shape (n_paths, n_steps + 1) normalised so V[:, 0] is about 1
        base_level
            Starting level at t0

        Returns
        -------
        Array same shape as V, expressed in level units
        """
        V = np.asarray(V, dtype=float)
        return V * float(base_level)

    # ------------------------------------------------------------------
    # helpers: stats
    # ------------------------------------------------------------------

    def portfolio_risk_return(
        self,
        w: np.ndarray,
        annualised: bool = True,
        mu_vec: Optional[np.ndarray] = None,
        Sigma: Optional[np.ndarray] = None,
        rf: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Compute (mean, vol, Sharpe) of a portfolio defined by weights w.
        """
        w = np.asarray(w, dtype=float).reshape(-1)
        if w.shape[0] != self.n_assets:
            raise ValueError("weights length must equal number of assets.")

        if mu_vec is None or Sigma is None or rf is None:
            if annualised:
                mu_vec = self.mu_annualised
                Sigma = self.Sigma_annualised
                rf = self.rf
            else:
                mu_vec = self.mu_hat
                Sigma = self.Sigma_hat
                rf = self.rf * self.dt

        mu_p = float(w @ mu_vec)
        var_p = float(w @ (Sigma @ w))
        vol_p = float(np.sqrt(var_p))
        sharpe = (mu_p - rf) / vol_p if vol_p > 0 else np.nan

        return mu_p, vol_p, sharpe

    def asset_risk_return_table(
        self,
        annualised: bool = True,
    ) -> pd.DataFrame:
        """
        Table of asset level risk return and Sharpe.

        Columns
        -------
        asset, mean, vol, sharpe
        """
        if annualised:
            mu = self.mu_annualised
            Sigma = self.Sigma_annualised
            rf = self.rf
        else:
            mu = self.mu_hat
            Sigma = self.Sigma_hat
            rf = self.rf * self.dt

        vols = np.sqrt(np.diag(Sigma))
        sharpes = np.where(vols > 0.0, (mu - rf) / vols, np.nan)

        return pd.DataFrame(
            {
                "asset": self.assets,
                "mean": mu,
                "vol": vols,
                "sharpe": sharpes,
            }
        )

    # ------------------------------------------------------------------
    # small helpers for printing
    # ------------------------------------------------------------------

    def _print_portfolio_stats(
        self,
        w: np.ndarray,
        mu_vec: Optional[np.ndarray] = None,
        Sigma: Optional[np.ndarray] = None,
        rf: Optional[float] = None,
    ) -> None:
        """
        Print basic stats for a given weight vector.
        """
        mu_p, vol_p, sharpe = self.portfolio_risk_return(
            w,
            annualised=True,
            mu_vec=mu_vec,
            Sigma=Sigma,
            rf=rf,
        )

        w = np.asarray(w, dtype=float).reshape(-1)

        print("  weights:", dict(zip(self.assets, np.round(w, 4))))
        print(f"  mean:   {mu_p:.4f}")
        print(f"  vol:    {vol_p:.4f}")
        print(f"  Sharpe: {sharpe:.3f}")

    # ------------------------------------------------------------------
    # portfolio path plots
    # ------------------------------------------------------------------

    def plot_portfolio_paths_3d(
        self,
        V: np.ndarray,
        time_index: Optional[pd.Index] = None,
        max_paths: int = 100,
        title: str = "Portfolio value paths (3D)",
        normalise_at_start: bool = True,
    ) -> None:
        """
        Plot simulated portfolio paths in three dimensions using Plotly, with two scenes.

        Left scene
            x  time
            y  path index
            z  portfolio value (or normalised value)

        Right scene
            x  path index
            y  value bin
            z  percentage of time the path spends in that value bin
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("plotly is not installed. Run `pip install plotly` to enable plotting.")
            return

        V = np.asarray(V, dtype=float)
        if V.ndim != 2:
            raise ValueError("V must be a 2D array of shape (n_paths, n_steps + 1).")

        n_paths, n_steps_plus_1 = V.shape

        # optional normalisation so all paths start at 1
        if normalise_at_start:
            V0 = V[:, [0]]
            if np.any(V0 == 0.0):
                raise ValueError(
                    "Some initial simulated portfolio values are zero, cannot normalise."
                )
            V = V / V0

        # time axis
        if time_index is not None:
            if len(time_index) != n_steps_plus_1:
                raise ValueError("time_index length must match V.shape[1].")
            x_time = np.array(time_index)
        else:
            x_time = np.arange(n_steps_plus_1)

        # two scenes
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            column_widths=[0.7, 0.3],
            horizontal_spacing=0.07,
        )

        # left scene paths
        n_plot = min(n_paths, max_paths)
        path_indices = np.linspace(0, n_paths - 1, n_plot, dtype=int)

        for pid in path_indices:
            y_path = np.full(n_steps_plus_1, pid)
            z_vals = V[pid, :]

            fig.add_trace(
                go.Scatter3d(
                    x=x_time,
                    y=y_path,
                    z=z_vals,
                    mode="lines",
                    line=dict(width=2),
                    name=f"Path {pid}",
                    hovertemplate=(
                        "Time: %{x}<br>"
                        "Path: %{y}<br>"
                        "Value: %{z:.4f}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        # right scene distribution surface

        # global value range
        v_min = float(np.nanmin(V))
        v_max = float(np.nanmax(V))
        if not np.isfinite(v_min) or not np.isfinite(v_max):
            raise ValueError(
                "Non finite values found in V for distribution plotting."
            )

        pad = 0.02 * (v_max - v_min) if v_max > v_min else 0.01
        v_min -= pad
        v_max += pad

        n_bins = 10
        bins = np.linspace(v_min, v_max, n_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # raw frequency matrix shape (n_bins, n_plot)
        freq_mat = np.zeros((n_bins, n_plot), dtype=float)

        for j, pid in enumerate(path_indices):
            series = V[pid, :]
            counts, _ = np.histogram(series, bins=bins)
            freq_pct = counts / series.size * 100.0
            freq_mat[:, j] = freq_pct

        # smoothing helpers

        def _gaussian_kernel(radius: int, sigma: float) -> np.ndarray:
            """
            Build one dimensional Gaussian kernel with given radius and sigma.
            """
            if radius <= 0:
                return np.array([1.0], dtype=float)
            x = np.arange(-radius, radius + 1, dtype=float)
            k = np.exp(-0.5 * (x / sigma) ** 2)
            k_sum = k.sum()
            if k_sum == 0.0:
                return np.array([1.0], dtype=float)
            return k / k_sum

        def _convolve_along_axis(
            arr: np.ndarray,
            kernel: np.ndarray,
            axis: int,
        ) -> np.ndarray:
            """
            Separable convolution along one axis with edge padding.
            """
            arr = np.asarray(arr, dtype=float)
            k = np.asarray(kernel, dtype=float).reshape(-1)
            if k.size == 1:
                return arr.copy()

            pad = k.size // 2
            if axis == 0:
                padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
                out = np.empty_like(arr)
                for i in range(arr.shape[0]):
                    window = padded[i : i + k.size, :]
                    out[i, :] = (window * k[:, None]).sum(axis=0)
            elif axis == 1:
                padded = np.pad(arr, ((0, 0), (pad, pad)), mode="edge")
                out = np.empty_like(arr)
                for j in range(arr.shape[1]):
                    window = padded[:, j : j + k.size]
                    out[:, j] = (window * k[None, :]).sum(axis=1)
            else:
                raise ValueError("axis must be 0 or 1 for _convolve_along_axis.")
            return out

        # kernels
        kernel_value = _gaussian_kernel(radius=1, sigma=0.5)
        kernel_path = _gaussian_kernel(radius=1, sigma=0.3)

        freq_smooth = _convolve_along_axis(freq_mat, kernel_value, axis=0)
        freq_smooth = _convolve_along_axis(freq_smooth, kernel_path, axis=1)
        freq_smooth = np.maximum(freq_smooth, 0.0)

        x_surface = path_indices.astype(float)
        y_surface = bin_centers
        z_surface = freq_smooth

        fig.add_trace(
            go.Surface(
                x=x_surface,
                y=y_surface,
                z=z_surface,
                name="pathwise value distribution",
                hovertemplate=(
                    "Path: %{x}<br>"
                    "Value bin: %{y:.4f}<br>"
                    "Pct time: %{z:.2f}%<extra></extra>"
                ),
                showscale=True,
            ),
            row=1,
            col=2,
        )

        # layout

        fig.update_layout(
            title=title,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=0, r=0, b=0, t=50),
        )

        fig.update_scenes(
            dict(
                xaxis_title="Time",
                yaxis_title="Path index",
                zaxis_title=(
                    "Normalised value (start = 1)"
                    if normalise_at_start
                    else "Portfolio value"
                ),
                xaxis=dict(showgrid=True, gridcolor="lightgray"),
                yaxis=dict(showgrid=True, gridcolor="lightgray"),
                zaxis=dict(showgrid=True, gridcolor="lightgray"),
                bgcolor="white",
            ),
            row=1,
            col=1,
        )

        fig.update_scenes(
            dict(
                xaxis_title="Path index",
                yaxis_title="Normalised value" if normalise_at_start else "Value",
                zaxis_title="Pct of time in value bin",
                xaxis=dict(showgrid=True, gridcolor="lightgray"),
                yaxis=dict(showgrid=True, gridcolor="lightgray"),
                zaxis=dict(showgrid=True, gridcolor="lightgray"),
                bgcolor="white",
            ),
            row=1,
            col=2,
        )

        fig.show()

    def plot_portfolio_paths_2d(
        self,
        V: np.ndarray,
        time_index: Optional[pd.Index] = None,
        max_paths: int = 50,
        show_mean: bool = True,
        real_path: Optional[np.ndarray] = None,
        real_label: str = "historical path",
        title: str = "Portfolio value paths",
        golden_run: bool = True,
        plot_portfolio: bool = True,
        normalise_at_start: bool = True,
        plot_nth_percentile: Optional[float] = 0.9,
        show: bool = False,   # <<< NEW
    ):
        """
        Plot simulated portfolio paths in 2D using Plotly.

        Left panel
            simulated paths, optional mean, golden run, realised path,
            and optional evolving percentile path.

        Right panel
            distribution of horizon returns across simulated paths,
            with optional line for realised horizon return and the same
            percentile of horizon returns.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("plotly is not installed. Run `pip install plotly` to enable plotting.")
            return

        V = np.asarray(V, dtype=float)
        if V.ndim != 2:
            raise ValueError(
                "V must be a 2D array of shape (n_paths, n_steps_plus_1)."
            )

        n_paths, n_steps_plus_1 = V.shape

        # optional normalisation
        if normalise_at_start:
            V0 = V[:, [0]]
            if np.any(V0 == 0.0):
                raise ValueError(
                    "Some initial simulated portfolio values are zero, cannot normalise."
                )
            V = V / V0

        # time axis
        if time_index is not None:
            if len(time_index) != n_steps_plus_1:
                raise ValueError("time_index length must match V.shape[1].")
            x_vals = np.array(time_index)
        else:
            x_vals = np.arange(n_steps_plus_1)

        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.7, 0.3],
            horizontal_spacing=0.07,
            specs=[[{"type": "xy"}, {"type": "xy"}]],
        )

        # left panel paths
        n_plot = min(n_paths, max_paths)
        if n_plot > 0:
            path_indices = np.linspace(0, n_paths - 1, n_plot, dtype=int)
            for pid in path_indices:
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=V[pid, :],
                        mode="lines",
                        line=dict(width=1, color="rgba(0, 0, 150, 0.15)"),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

        # mean simulated path
        if show_mean:
            mean_path = V.mean(axis=0)
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=mean_path,
                    mode="lines",
                    line=dict(width=3, color="rgba(0, 0, 0, 0.9)"),
                    name="mean simulated",
                    hovertemplate="Time: %{x}<br>Value: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # evolving percentile path across time
        if plot_nth_percentile is not None:
            if not (0.0 < plot_nth_percentile < 1.0):
                raise ValueError("plot_nth_percentile must be between 0 and 1.")
            q = plot_nth_percentile * 100.0
            perc_path = np.percentile(V, q, axis=0)

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=perc_path,
                    mode="lines",
                    line=dict(width=2, color="rgba(0, 150, 0, 0.8)", dash="dot"),
                    name=f"{int(q)}th percentile path",
                    hovertemplate=(
                        "Time: %{x}<br>"
                        f"{int(q)}th pct value: %<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=np.percentile(V, abs(q - 100), axis=0),
                    mode="lines",
                    line=dict(width=2, color="rgba(150, 0, 0, 0.8)", dash="dot"),
                    name=f"{int(100 - q)}th percentile path",
                    hovertemplate=(
                        "Time: %{x}<br>"
                        f"{int(100 - q)}th pct value: %<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

        # realised path handling
        real_arr = None
        if real_path is not None:
            real_arr = np.asarray(real_path, dtype=float)
            if real_arr.shape[0] != n_steps_plus_1:
                raise ValueError(
                    "real_path length must match V.shape[1] (time dimension)."
                )
            if normalise_at_start:
                if real_arr[0] == 0.0:
                    raise ValueError("real_path initial value is zero, cannot normalise.")
                real_arr = real_arr / real_arr[0]
        elif golden_run:
            raise ValueError("golden_run=True requires real_path to be provided.")

        # golden run
        if golden_run and real_arr is not None:
            nmise_list = []
            mape_list = []
            for z in range(n_paths):
                nmise_z, mape_z = nmise_mape(real_arr, V[z])
                nmise_list.append(nmise_z)
                mape_list.append(mape_z)

            nmise_arr = np.asarray(nmise_list, dtype=float)

            if np.all(np.isnan(nmise_arr)):
                print("All NMISE values are NaN; skipping golden run highlight.")
            else:
                lowest_nmise_index = int(np.nanargmin(nmise_arr))
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=V[lowest_nmise_index],
                        mode="lines",
                        line=dict(width=3, color="rgba(0, 150, 0, 0.9)"),
                        name="golden run",
                        hovertemplate=(
                            "Time: %{x}<br>Value: %{y:.4f}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )
                print(
                    f"Golden run path index: {lowest_nmise_index}, "
                    f"NMISE: {nmise_arr[lowest_nmise_index]:.6f}, "
                    f"MAPE: {mape_list[lowest_nmise_index]:.4f}%"
                )

        # realised path overlay
        if real_arr is not None:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=real_arr,
                    mode="lines",
                    line=dict(width=3, color="red"),
                    name=real_label,
                    hovertemplate="Time: %{x}<br>Value: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # overlay underlying level series
        if plot_portfolio:
            for col in self.level.columns:
                y_under = np.asarray(self.level[col].values, dtype=float)
                if normalise_at_start:
                    if y_under[0] == 0.0:
                        continue
                    y_under = y_under / y_under[0]

                fig.add_trace(
                    go.Scatter(
                        x=self.level.index,
                        y=y_under,
                        mode="lines",
                        line=dict(width=1, color="rgba(150, 150, 150, 0.2)"),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

        # right panel horizon return distribution
        if normalise_at_start:
            ret_sim = V[:, -1] - 1.0
        else:
            ret_sim = V[:, -1] / V[:, 0] - 1.0

        fig.add_trace(
            go.Histogram(
                y=ret_sim,
                nbinsy=40,
                orientation="h",
                name="simulated horizon returns",
                marker=dict(color="rgba(0, 0, 150, 0.5)"),
                opacity=0.7,
                hovertemplate="Return: %{y:.4f}<br>Count: %{x}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # percentile line for horizon returns
        if plot_nth_percentile is not None:
            if not (0.0 < plot_nth_percentile < 1.0):
                raise ValueError("plot_nth_percentile must be between 0 and 1.")
            q = plot_nth_percentile * 100.0
            perc_value = np.percentile(ret_sim, q)

            fig.add_shape(
                type="line",
                x0=0,
                x1=1,
                xref="paper",
                y0=perc_value,
                y1=perc_value,
                yref="y2",
                line=dict(color="green", width=2, dash="dot"),
            )
            fig.add_annotation(
                x=1.02,
                y=perc_value,
                xref="paper",
                yref="y2",
                text=f"{int(q)}th percentile",
                showarrow=False,
                font=dict(color="green", size=10),
                xanchor="left",
            )

        # realised horizon return line
        if real_arr is not None:
            if normalise_at_start:
                ret_real = real_arr[-1] - 1.0
            else:
                ret_real = real_arr[-1] / real_arr[0] - 1.0

            fig.add_shape(
                type="line",
                x0=0,
                x1=1,
                xref="paper",
                y0=ret_real,
                y1=ret_real,
                yref="y2",
                line=dict(color="red", width=2, dash="dash"),
            )
            fig.add_annotation(
                x=1.02,
                y=ret_real,
                xref="paper",
                yref="y2",
                text="realised",
                showarrow=False,
                font=dict(color="red", size=10),
                xanchor="left",
            )

        fig.update_layout(
            title=title,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=0, r=0, b=0, t=50),
            barmode="overlay",
        )

        fig.update_xaxes(
            title_text="Time",
            showgrid=True,
            gridcolor="lightgray",
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title_text=(
                "Normalised value (start = 1)"
                if normalise_at_start
                else "Portfolio value"
            ),
            showgrid=True,
            gridcolor="lightgray",
            row=1,
            col=1,
        )

        fig.update_xaxes(
            title_text="Frequency",
            showgrid=True,
            gridcolor="lightgray",
            row=1,
            col=2,
        )
        fig.update_yaxes(
            title_text="Horizon return",
            showgrid=True,
            gridcolor="lightgray",
            row=1,
            col=2,
        )

        if show:
            fig.show()

        return fig

    # ------------------------------------------------------------------
    # efficient frontier
    # ------------------------------------------------------------------

    def efficient_frontier(
        self,
        n_points: int = 50,
        annualised: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Markowitz efficient frontier for unconstrained portfolios
        with full investment sum of weights equals one.

        Parameters
        ----------
        n_points
            Number of points along the frontier
        annualised
            If True, returns and vol are annualised

        Returns
        -------
        target_returns
        target_vols
        weights
        """
        if n_points <= 1:
            raise ValueError("n_points must be greater than 1.")

        mu = self.mu_annualised
        Sigma = self.Sigma_annualised

        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.pinv(Sigma)

        ones = np.ones(self.n_assets)

        A = float(ones.T @ Sigma_inv @ ones)
        B = float(ones.T @ Sigma_inv @ mu)
        C = float(mu.T @ Sigma_inv @ mu)
        delta = A * C - B**2

        if delta <= 0:
            raise ValueError("Delta is non positive, covariance may be singular.")

        r_min = float(mu.min())
        r_max = float(mu.max())
        target_returns = np.linspace(r_min, r_max, n_points)

        weights = np.zeros((n_points, self.n_assets))
        target_vars = np.zeros(n_points)

        for i, r in enumerate(target_returns):
            lambda_1 = (C - B * r) / delta
            lambda_2 = (A * r - B) / delta

            w = Sigma_inv @ (lambda_1 * ones + lambda_2 * mu)
            weights[i, :] = w

            target_vars[i] = (A * r**2 - 2 * B * r + C) / delta

        target_vols = np.sqrt(target_vars)

        if not annualised:
            target_returns = target_returns * self.dt
            target_vols = target_vols * np.sqrt(self.dt)

        return target_returns, target_vols, weights

    def plot_efficient_frontier(
        self,
        n_points: int = 50,
        annualised: bool = True,
        show_tangency: bool = True,
        title: str = "Efficient frontier",
    ) -> None:
        """
        Plot Markowitz efficient frontier using Plotly, with individual assets,
        tangency portfolio and capital market line.

        Axes
        ----
        x  volatility
        y  expected return
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("plotly is not installed. Run `pip install plotly` to enable plotting.")
            return

        r_frontier, vol_frontier, _ = self.efficient_frontier(
            n_points=n_points,
            annualised=annualised,
        )

        if annualised:
            mu = self.mu_annualised
            Sigma = self.Sigma_annualised
            rf = self.rf
            x_label = "Volatility (annualised)"
            y_label = "Expected return (annualised)"
        else:
            mu = self.mu_hat
            Sigma = self.Sigma_hat
            rf = self.rf * self.dt
            x_label = "Volatility (per step)"
            y_label = "Expected return (per step)"

        fig = go.Figure()

        # frontier
        fig.add_trace(
            go.Scatter(
                x=vol_frontier,
                y=r_frontier,
                mode="lines",
                name="efficient frontier",
                line=dict(width=3),
                hovertemplate=(
                    "Vol: %{x:.4f}<br>"
                    "Return: %{y:.4f}<extra></extra>"
                ),
            )
        )

        # assets
        asset_vols = np.sqrt(np.diag(Sigma))
        asset_returns = mu

        fig.add_trace(
            go.Scatter(
                x=asset_vols,
                y=asset_returns,
                mode="markers+text",
                name="assets",
                text=self.assets,
                textposition="top center",
                marker=dict(size=8, color="rgba(80, 80, 80, 0.9)"),
                hovertemplate=(
                    "Asset: %{text}<br>"
                    "Vol: %{x:.4f}<br>"
                    "Return: %{y:.4f}<extra></extra>"
                ),
            )
        )

        # tangency and capital market line
        if show_tangency:
            if self.w_tangency_ is None:
                self.tangency_weights()

            w = np.asarray(self.w_tangency_, dtype=float).reshape(-1)
            mu_p, vol_p, sharpe_p = self.portfolio_risk_return(
                w,
                annualised=annualised,
                mu_vec=mu,
                Sigma=Sigma,
                rf=rf,
            )

            fig.add_trace(
                go.Scatter(
                    x=[vol_p],
                    y=[mu_p],
                    mode="markers",
                    marker=dict(size=11, color="red", symbol="star"),
                    name="tangency portfolio",
                    hovertemplate=(
                        "Tangency portfolio<br>"
                        "Vol: %{x:.4f}<br>"
                        "Return: %{y:.4f}<extra></extra>"
                    ),
                )
            )

            if vol_p > 0 and np.isfinite(sharpe_p):
                x_max = float(
                    1.05
                    * max(
                        float(vol_frontier.max()),
                        float(asset_vols.max()),
                        vol_p,
                    )
                )
                x_cml = np.linspace(0.0, x_max, 80)
                y_cml = rf + sharpe_p * x_cml

                fig.add_trace(
                    go.Scatter(
                        x=x_cml,
                        y=y_cml,
                        mode="lines",
                        name="capital market line",
                        line=dict(
                            width=3,
                            color="rgba(150, 0, 0, 0.7)",
                            dash="dash",
                        ),
                        hovertemplate=(
                            "Vol: %{x:.4f}<br>"
                            "Return: %{y:.4f}<extra></extra>"
                        ),
                    )
                )

        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=0, r=0, b=0, t=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
            ),
        )
        fig.update_xaxes(showgrid=True, gridcolor="lightgray")
        fig.update_yaxes(showgrid=True, gridcolor="lightgray")

        fig.show()

    def plot_efficient_frontier_3d(
        self,
        n_points: int = 50,
        annualised: bool = True,
        show_tangency: bool = True,
        title: str = "Efficient frontier 3D view",
    ) -> None:
        """
        Three dimensional view of the efficient frontier.

        Axes
        ----
        x  volatility
        y  expected return
        z  Sharpe ratio
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("plotly is not installed. Run `pip install plotly` to enable plotting.")
            return

        r_frontier, vol_frontier, _ = self.efficient_frontier(
            n_points=n_points,
            annualised=annualised,
        )

        if annualised:
            mu = self.mu_annualised
            Sigma = self.Sigma_annualised
            rf = self.rf
        else:
            mu = self.mu_hat
            Sigma = self.Sigma_hat
            rf = self.rf * self.dt

        sharpe_frontier = np.where(
            vol_frontier > 0.0,
            (r_frontier - rf) / vol_frontier,
            np.nan,
        )

        fig = go.Figure()

        # frontier curve
        fig.add_trace(
            go.Scatter3d(
                x=vol_frontier,
                y=r_frontier,
                z=sharpe_frontier,
                mode="lines+markers",
                name="efficient frontier",
                line=dict(width=3),
                marker=dict(size=4),
                hovertemplate=(
                    "Vol: %{x:.4f}<br>"
                    "Return: %{y:.4f}<br>"
                    "Sharpe: %{z:.4f}<extra></extra>"
                ),
            )
        )

        # assets
        asset_vols = np.sqrt(np.diag(Sigma))
        asset_returns = mu
        asset_sharpes = np.where(
            asset_vols > 0.0,
            (asset_returns - rf) / asset_vols,
            np.nan,
        )

        fig.add_trace(
            go.Scatter3d(
                x=asset_vols,
                y=asset_returns,
                z=asset_sharpes,
                mode="markers+text",
                name="assets",
                text=self.assets,
                textposition="top center",
                marker=dict(size=8, color="rgba(80, 80, 80, 0.9)"),
                hovertemplate=(
                    "Asset: %{text}<br>"
                    "Vol: %{x:.4f}<br>"
                    "Return: %{y:.4f}<br>"
                    "Sharpe: %{z:.4f}<extra></extra>"
                ),
            )
        )

        if show_tangency:
            if self.w_tangency_ is None:
                self.tangency_weights()

            w = np.asarray(self.w_tangency_, dtype=float).reshape(-1)
            mu_p, vol_p, sharpe_p = self.portfolio_risk_return(
                w,
                annualised=annualised,
                mu_vec=mu,
                Sigma=Sigma,
                rf=rf,
            )

            fig.add_trace(
                go.Scatter3d(
                    x=[vol_p],
                    y=[mu_p],
                    z=[sharpe_p],
                    mode="markers",
                    marker=dict(size=11, color="red", symbol="star"),
                    name="tangency portfolio",
                    hovertemplate=(
                        "Tangency portfolio<br>"
                        "Vol: %{x:.4f}<br>"
                        "Return: %{y:.4f}<br>"
                        "Sharpe: %{z:.4f}<extra></extra>"
                    ),
                )
            )

            if vol_p > 0.0 and np.isfinite(sharpe_p):
                x_max = float(
                    1.05
                    * max(
                        float(vol_frontier.max()),
                        float(asset_vols.max()),
                        vol_p,
                    )
                )
                x_cml = np.linspace(0.0, x_max, 60)
                y_cml = rf + sharpe_p * x_cml
                z_cml = np.full_like(x_cml, sharpe_p)

                fig.add_trace(
                    go.Scatter3d(
                        x=x_cml,
                        y=y_cml,
                        z=z_cml,
                        mode="lines",
                        name="capital market line",
                        line=dict(
                            width=3,
                            color="rgba(150, 0, 0, 0.7)",
                            dash="dash",
                        ),
                        hovertemplate=(
                            "Vol: %{x:.4f}<br>"
                            "Return: %{y:.4f}<br>"
                            "Sharpe: %{z:.4f}<extra></extra>"
                        ),
                    )
                )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=(
                    "Volatility (annualised)"
                    if annualised
                    else "Volatility (per step)"
                ),
                yaxis_title=(
                    "Expected return (annualised)"
                    if annualised
                    else "Expected return (per step)"
                ),
                zaxis_title="Sharpe ratio",
                xaxis=dict(showgrid=True, gridcolor="lightgray"),
                yaxis=dict(showgrid=True, gridcolor="lightgray"),
                zaxis=dict(showgrid=True, gridcolor="lightgray"),
                bgcolor="white",
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
            ),
            paper_bgcolor="white",
        )

        fig.show()
