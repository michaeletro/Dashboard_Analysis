from dashboard import data


def setup_module(module):
    data.reset_cache()
    data.ensure_data_loaded()


def test_factor_scores_columns_present():
    df = data.df_equity
    # Expect at least these factors from FactorAnalyzer construction
    expected = {"ValueScore", "QualityScore", "MomentumScore"}
    assert expected.issubset(df.columns), f"Missing factor columns: {expected - set(df.columns)}"
    # Scores should be numeric and finite
    for col in expected:
        series = df[col]
        assert series.dtype.kind in "fi", f"Factor {col} is not numeric"
        assert series.isna().sum() == 0, f"Factor {col} contains NaNs"
