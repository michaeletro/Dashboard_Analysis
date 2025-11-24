from dashboard import data


def setup_module(module):
    data.reset_cache()
    data.ensure_data_loaded()


def test_regression_results_structure():
    # Using globals prepared in data.ensure_data_loaded
    assert data.results_all is not None, "All-factor regression results is None"
    cols = set(data.results_all.columns)
    required = {"Factor", "Coefficient"}
    assert required.issubset(cols), f"Regression result missing columns {required - cols}"


def test_r2_range():
    assert data.r2_all is not None and 0.0 <= data.r2_all <= 1.0
    assert data.r2_mom is not None and 0.0 <= data.r2_mom <= 1.0


def test_error_metrics_range():
    # NMISE >= 0, MAPE >= 0
    assert data.nmise_all >= 0
    assert data.mape_all >= 0
    assert data.nmise_mom >= 0
    assert data.mape_mom >= 0
