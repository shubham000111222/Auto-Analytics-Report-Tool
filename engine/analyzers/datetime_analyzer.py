from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def _freq(span_days: int) -> str:
    if span_days <= 60: return "D"
    if span_days <= 730: return "W"
    return "M"


def _gaps(s: pd.Series, freq: str) -> list:
    full_range = pd.period_range(s.min(), s.max(), freq=freq)
    present = set(s.dt.to_period(freq))
    return [p for p in full_range if p not in present]


def _max_gap_length(s: pd.Series, freq: str) -> int:
    full_range = pd.period_range(s.min(), s.max(), freq=freq)
    present = sorted(set(s.dt.to_period(freq)))
    present_idx = {p: i for i, p in enumerate(full_range)}
    present_positions = sorted(present_idx[p] for p in present if p in present_idx)
    if len(present_positions) < 2:
        return 0
    return max(b - a for a, b in zip(present_positions, present_positions[1:])) - 1


def analyze_datetime(series: pd.Series) -> dict:
    s = pd.to_datetime(series, errors="coerce").dropna().sort_values()
    if s.empty:
        return {"empty": True}

    span_days = (s.max() - s.min()).days
    freq = _freq(span_days)
    counts = s.dt.to_period(freq).value_counts().sort_index()
    gaps = _gaps(s, freq)

    if len(counts) > 2:
        tau, p_value = kendalltau(np.arange(len(counts)), counts.values)
        slope = float(np.polyfit(np.arange(len(counts)), counts.values, 1)[0])
    else:
        tau, p_value, slope = 0.0, 1.0, 0.0

    trend_significant = p_value < 0.05
    duplicate_timestamps = int(s.duplicated().sum())
    max_gap_periods = _max_gap_length(s, freq) if gaps else 0

    return {
        "min_date": str(s.min()), "max_date": str(s.max()), "span_days": span_days,
        "resample_freq": freq,
        "record_count_by_period": {str(k): int(v) for k, v in counts.items()},
        "trend_slope": slope, "trend_tau": round(float(tau), 3),
        "trend_p_value": round(float(p_value), 4), "trend_significant": trend_significant,
        "trend_direction": (
            "increasing" if slope > 0 and trend_significant else
            "decreasing" if slope < 0 and trend_significant else
            "no clear trend"
        ),
        "gap_count": len(gaps), "gaps": [str(g) for g in gaps[:10]],
        "max_gap_periods": max_gap_periods, "duplicate_timestamps": duplicate_timestamps,
        "weekday_distribution": s.dt.day_name().value_counts().to_dict(),
    }
