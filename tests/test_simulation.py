import numpy as np

from dashboard import data


def setup_module(module):
    data.reset_cache()
    data.ensure_data_loaded()


def test_simulation_shape_and_positive():
    engine = data.engine
    w = np.ones(engine.n_assets) / engine.n_assets
    V = engine.simulate_portfolio_paths(weights=w, n_steps=40, n_paths=25)
    assert V.shape == (25, 41)
    # Paths are normalised positive values
    assert np.all(V > 0), "Simulated portfolio paths contain non-positive values"


def test_tangency_weights_length():
    engine = data.engine
    w_tan = engine.tangency_weights()
    assert len(w_tan) == engine.n_assets
    # weights sum to 1 approximately
    assert abs(w_tan.sum() - 1.0) < 1e-6
