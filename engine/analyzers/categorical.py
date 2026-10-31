from __future__ import annotations
import pandas as pd


def analyze_categorical(series: pd.Series, top_n: int = 10) -> dict:
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return {"empty": True}

    counts = s.value_counts()
    top = counts.head(top_n)
    other_count = counts.iloc[top_n:].sum() if len(counts) > top_n else 0
    total = len(s)

    normalized_unique = s.str.lower().str.strip().nunique()
    raw_unique = s.nunique()

    return {
        "cardinality": int(raw_unique),
        "mode": counts.index[0],
        "mode_pct": round(counts.iloc[0] / total * 100, 2),
        "top_values": top.to_dict(),
        "top_value_pct": (top / total * 100).round(2).to_dict(),
        "other_count": int(other_count),
        "other_pct": round(other_count / total * 100, 2) if total else 0,
        "suggest_pie": raw_unique <= 5,
        "possible_inconsistent_categories": int(raw_unique - normalized_unique),
    }


def resolve_categorical_stats(df, col: str, column_profile, dataset_profile, top_n: int = 10) -> dict:
    """Prefers exact DuckDB top-value counts over the sampled estimate."""
    sample_stats = analyze_categorical(df[col], top_n=top_n)
    exact_top = column_profile.stats.get("exact_top_values") if column_profile else None
    if not exact_top:
        return sample_stats

    total_non_null = dataset_profile.n_rows - column_profile.missing_count
    top_sorted = dict(sorted(exact_top.items(), key=lambda kv: kv[1], reverse=True)[:top_n])
    if not top_sorted:
        return sample_stats
    mode, mode_count = next(iter(top_sorted.items()))
    other_count = max(0, total_non_null - sum(top_sorted.values()))

    merged = dict(sample_stats)
    merged.update({
        "cardinality": column_profile.unique_count,
        "mode": mode,
        "mode_pct": round(mode_count / total_non_null * 100, 2) if total_non_null else 0,
        "top_values": top_sorted,
        "top_value_pct": {k: round(v / total_non_null * 100, 2) for k, v in top_sorted.items()} if total_non_null else {},
        "other_count": other_count,
        "other_pct": round(other_count / total_non_null * 100, 2) if total_non_null else 0,
        "suggest_pie": column_profile.unique_count <= 5,
        "stats_source": "duckdb_exact",
    })
    return merged
