import numpy as np
import pandas as pd
from engine import loader
from engine.profiler import profile_dataset, apply_exact_stats


def test_exact_stats_match_full_dataset_not_sample(monkeypatch):
    monkeypatch.setattr(loader, "MAX_IN_MEMORY_BYTES", 1)
    monkeypatch.setattr(loader, "SAMPLE_ROWS_FOR_CHARTS", 50)

    n = 5000
    df = pd.DataFrame({
        "value": np.random.default_rng(0).normal(100, 15, n),
        "category": np.random.choice(["a", "b", "c"], n),
    })
    raw = df.to_csv(index=False).encode("utf-8")

    result = loader.load_file("big.csv", raw)
    assert result.is_sampled and len(result.df) <= 50
    assert result.row_count_full == n

    profile = profile_dataset(result.df)
    apply_exact_stats(profile, result)

    exact_mean = profile.columns["value"].stats["exact"]["mean"]
    assert abs(exact_mean - df["value"].mean()) < 0.01
    assert profile.n_rows == n
    assert profile.columns["value"].missing_count == 0

    loader.cleanup_load_result(result)
