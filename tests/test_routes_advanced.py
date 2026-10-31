import pandas as pd
import numpy as np
from engine.optional_analysis import run_clustering, run_feature_importance
from engine.profiler import profile_dataset


def test_clustering_and_feature_importance_contract_match_route_expectations():
    df = pd.DataFrame({"x": np.random.randn(200), "y": np.random.randn(200)})
    cluster_result = run_clustering(df, profile_dataset(df))
    assert "plot_df" in cluster_result and {"pc1", "pc2", "cluster"} <= set(cluster_result["plot_df"].columns)

    df["target"] = (df["x"] + df["y"] > 0).astype(str)
    fi_result = run_feature_importance(df, profile_dataset(df), target="target")
    assert set(fi_result["top_features"].keys())
