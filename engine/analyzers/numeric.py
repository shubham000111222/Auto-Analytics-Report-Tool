from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as sstats


def _skew_label(skew: float) -> str:
    if skew > 1: return "highly right-skewed"
    if skew > 0.5: return "moderately right-skewed"
    if skew < -1: return "highly left-skewed"
    if skew < -0.5: return "moderately left-skewed"
    return "approximately symmetric"


def _fd_bins(s: pd.Series) -> int:
    n = len(s)
    if n < 2: return 1
    iqr = s.quantile(0.75) - s.quantile(0.25)
    if iqr == 0: return min(30, max(5, int(np.sqrt(n))))
    bw = 2 * iqr / (n ** (1 / 3))
    if bw <= 0: return 20
    return int(min(60, max(5, (s.max() - s.min()) / bw)))


def _mad_outliers(s: pd.Series) -> pd.Series:
    median = s.median()
    mad = (s - median).abs().median()
    if mad == 0:
        return s.iloc[0:0]
    modified_z = 0.6745 * (s - median) / mad
    return s[modified_z.abs() > 3.5]


def analyze_numeric(series: pd.Series) -> dict:
    s = series.dropna().astype(float)
    if s.empty:
        return {"empty": True}

    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_outliers = s[(s < lower) | (s > upper)]
    mad_outliers = _mad_outliers(s)

    std = s.std(ddof=0)
    z = (s - s.mean()) / std if std > 0 else pd.Series([0] * len(s))
    z_outliers = s[z.abs() > 3]

    skew = float(sstats.skew(s)) if len(s) > 2 else 0.0
    kurt = float(sstats.kurtosis(s)) if len(s) > 2 else 0.0

    n_unique = s.nunique()
    looks_ordinal = n_unique <= 10 and n_unique / len(s) < 0.05
    suggest_log_transform = skew > 1.0 and (s > 0).all()

    return {
        "count": int(s.count()), "mean": float(s.mean()), "median": float(s.median()),
        "std": float(std), "min": float(s.min()), "max": float(s.max()),
        "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
        "skew": skew, "kurtosis": kurt, "skew_label": _skew_label(skew),
        "iqr_outlier_count": int(len(iqr_outliers)),
        "iqr_outlier_pct": round(len(iqr_outliers) / len(s) * 100, 2),
        "mad_outlier_count": int(len(mad_outliers)),
        "z_outlier_count": int(len(z_outliers)),
        "zero_count": int((s == 0).sum()), "negative_count": int((s < 0).sum()),
        "bins_suggested": _fd_bins(s),
        "looks_ordinal": looks_ordinal,
        "suggest_log_transform": suggest_log_transform,
    }


def resolve_numeric_stats(df, col, column_profile) -> dict:
    """Prefers exact DuckDB stats (set by profiler.apply_exact_stats) over sample
    estimates, so every number reaching the narrative reflects the full dataset even
    though the chart itself is drawn from a sample."""
    exact = column_profile.stats.get("exact") if column_profile else None
    sample_stats = analyze_numeric(df[col])
    if not exact:
        return sample_stats

    merged = dict(sample_stats)
    for key in ("count", "mean", "std", "min", "max", "q1", "median", "q3", "iqr",
                "zero_count", "negative_count", "iqr_outlier_count", "iqr_outlier_pct",
                "skew", "kurtosis"):
        if exact.get(key) is not None:
            merged[key] = exact[key]
    merged["skew_label"] = _skew_label(merged["skew"])
    merged["stats_source"] = "duckdb_exact"
    return merged
