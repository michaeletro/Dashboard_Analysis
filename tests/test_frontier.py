import numpy as np

from dashboard import data

def setup_module(module):
    data.reset_cache()
    data.ensure_data_loaded()


def test_efficient_frontier_monotonic_vol():
    engine = data.engine
    r, vol, _ = engine.efficient_frontier(n_points=50, annualised=True)
    # Vols should be strictly increasing or non-decreasing
    diffs = np.diff(vol)
    assert np.all(diffs >= -1e-10), "Frontier volatilities not monotonic"
    # Returns positive length and matching shapes
    assert len(r) == len(vol) == 50


def test_covariance_psd():
    engine = data.engine
    Sigma = engine.Sigma_annualised
    # Numerical PSD check: eigenvalues >= -tol
    eigs = np.linalg.eigvalsh(Sigma)
    assert eigs.min() >= -1e-8, f"Covariance not PSD: min eigenvalue {eigs.min()}"
