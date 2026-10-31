import pandas as pd
import numpy as np
from engine.profiler import profile_dataset
from engine.optional_analysis import run_clustering, run_feature_importance, run_forecast


def _blob_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": rng.normal(0, 1, 200), "y": rng.normal(0, 1, 200),
        "label": rng.choice(["a", "b"], 200),
    })


def test_clustering_runs():
    df = _blob_df()
    result = run_clustering(df, profile_dataset(df))
    assert result["available"] and result["k"] >= 2


def test_feature_importance_runs():
    df = _blob_df()
    df["target"] = (df["x"] + df["y"] > 0).astype(int).astype(str)
    result = run_feature_importance(df, profile_dataset(df), target="target")
    assert result["available"] and "top_features" in result


def test_forecast_runs():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    df = pd.DataFrame({"date": dates, "sales": np.arange(60) + np.random.randn(60)})
    result = run_forecast(df, profile_dataset(df), date_col="date", value_col="sales")
    assert result["available"] and len(result["forecast"]) == 6
