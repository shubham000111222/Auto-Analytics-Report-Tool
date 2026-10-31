import pandas as pd
import numpy as np
from engine.profiler import profile_dataset
from engine.charts import build_all_charts, enforce_global_budget, MAX_CHARTS_PER_SECTION


def _sample_df(n_num=20, n_cat=5, rows=300):
    rng = np.random.default_rng(0)
    data = {f"num_{i}": rng.normal(0, 1, rows) for i in range(n_num)}
    data.update({f"cat_{i}": rng.choice(["a", "b", "c"], rows) for i in range(n_cat)})
    data["date"] = pd.date_range("2022-01-01", periods=rows, freq="D")
    return pd.DataFrame(data)


def test_charts_build_without_crashing():
    df = _sample_df()
    profile = profile_dataset(df)
    charts = build_all_charts(df, profile)
    assert isinstance(charts, dict)
    assert all(isinstance(v, list) for v in charts.values())


def test_wide_file_respects_per_section_cap():
    df = _sample_df(n_num=50, n_cat=0)
    profile = profile_dataset(df)
    charts = build_all_charts(df, profile)
    assert len(charts["numeric"]) <= MAX_CHARTS_PER_SECTION


def test_global_budget_enforced():
    fake_sections = {"data_quality": [1] * 5, "relationships": [1] * 20,
                      "trends": [1] * 10, "numeric": [1] * 12, "categorical": [1] * 12, "segments": [1] * 5}
    trimmed = enforce_global_budget(fake_sections, budget=20)
    assert sum(len(v) for v in trimmed.values()) <= 20
    assert len(trimmed["data_quality"]) == 5


def test_every_chart_has_a_takeaway():
    df = _sample_df()
    charts = build_all_charts(df, profile_dataset(df))
    for specs in charts.values():
        for spec in specs:
            assert spec.takeaway and isinstance(spec.takeaway, str)
