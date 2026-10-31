import numpy as np
import pandas as pd
from engine.profiler import profile_dataset
from engine.charts import build_relationship_charts, build_segment_charts


def test_pearson_spearman_and_cramers_all_appear():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "cat_a": rng.choice(["p", "q"], n),
    })
    df["y"] = df["x"] * 2 + rng.normal(0, 0.1, n)
    df["cat_b"] = df["cat_a"]

    profile = profile_dataset(df)
    charts = build_relationship_charts(df, profile)
    titles = [c.title for c in charts]

    assert any("Pearson" in t for t in titles)
    assert any("Spearman" in t for t in titles)
    assert any("Categorical associations" in t for t in titles)


def test_segments_separate_from_relationships():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"val": rng.normal(0, 1, 200), "grp": rng.choice(["a", "b", "c"], 200)})
    profile = profile_dataset(df)
    segment_charts = build_segment_charts(df, profile)
    assert all(c.section == "segments" for c in segment_charts)
