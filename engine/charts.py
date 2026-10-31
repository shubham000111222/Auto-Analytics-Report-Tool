from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .profiler import DatasetProfile, ColType
from .analyzers import numeric as num_an, categorical as cat_an
from .analyzers import datetime_analyzer as dt_an, relationships as rel_an

MAX_CHARTS_PER_SECTION = 12
GLOBAL_CHART_BUDGET = 40

DISPLAY_ORDER = ["data_quality", "numeric", "categorical", "relationships", "trends", "segments"]
TRIM_PRIORITY = ["data_quality", "relationships", "trends", "segments", "numeric", "categorical"]


@dataclass
class ChartSpec:
    section: str
    title: str
    fig: go.Figure
    takeaway: str
    facts: dict = field(default_factory=dict)


def _select_top_columns(df, cols, k):
    if len(cols) <= k:
        return cols
    scored = [(c, df[c].var()) for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    scored.sort(key=lambda x: (x[1] if pd.notna(x[1]) else 0), reverse=True)
    return [c for c, _ in scored[:k]] or cols[:k]


def build_quality_charts(df, profile: DatasetProfile) -> list[ChartSpec]:
    missing = {c: p.missing_count for c, p in profile.columns.items() if p.missing_count > 0}
    if not missing:
        return []
    s = pd.Series(missing).sort_values(ascending=False)
    fig = px.bar(s, orientation="h", labels={"index": "column", "value": "missing count"},
                 title="Missing values per column")
    worst = s.index[0]
    return [ChartSpec("data_quality", "Missing values per column", fig,
                       f"{worst} has the most missing data at {profile.columns[worst].missing_pct}%.",
                       {"worst_col": worst, "worst_pct": profile.columns[worst].missing_pct})]


def build_numeric_charts(df, profile: DatasetProfile) -> list[ChartSpec]:
    cols = _select_top_columns(df, [c for c, p in profile.columns.items() if p.dtype == ColType.NUMERIC],
                                MAX_CHARTS_PER_SECTION)
    charts = []
    for col in cols:
        stats = num_an.resolve_numeric_stats(df, col, profile.columns[col])
        if stats.get("empty"):
            continue
        fig = px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}")
        basis = " (exact, full dataset)" if stats.get("stats_source") == "duckdb_exact" else ""
        takeaway = (f"{col} is {stats['skew_label']} with {stats['iqr_outlier_count']} "
                    f"outlier(s) ({stats['iqr_outlier_pct']}%){basis}.")
        if stats.get("looks_ordinal"):
            takeaway += " Values look ordinal/discrete rather than continuous."
        charts.append(ChartSpec("numeric", f"Distribution of {col}", fig, takeaway, stats))
    return charts


def build_categorical_charts(df, profile: DatasetProfile) -> list[ChartSpec]:
    cols = [c for c, p in profile.columns.items() if p.dtype == ColType.CATEGORICAL][:MAX_CHARTS_PER_SECTION]
    charts = []
    for col in cols:
        stats = cat_an.resolve_categorical_stats(df, col, profile.columns[col], profile)
        if stats.get("empty"):
            continue
        items = list(stats["top_values"].items())
        if stats["other_count"] > 0:
            items.append(("Other", stats["other_count"]))
        plot_df = pd.DataFrame(items, columns=[col, "count"])
        fig = (px.pie(plot_df, names=col, values="count", title=f"Share by {col}")
               if stats["suggest_pie"] else
               px.bar(plot_df, x="count", y=col, orientation="h", title=f"Top values: {col}"))
        basis = " (exact, full dataset)" if stats.get("stats_source") == "duckdb_exact" else ""
        takeaway = f"'{stats['mode']}' is the most common value in {col} ({stats['mode_pct']}%){basis}."
        if stats.get("possible_inconsistent_categories"):
            takeaway += f" Possible inconsistent casing/spacing in {stats['possible_inconsistent_categories']} value(s)."
        charts.append(ChartSpec("categorical", f"Top values: {col}", fig, takeaway, stats))
    return charts


def build_datetime_charts(df, profile: DatasetProfile) -> list[ChartSpec]:
    dt_cols = [c for c, p in profile.columns.items() if p.dtype == ColType.DATETIME]
    num_cols = [c for c, p in profile.columns.items() if p.dtype == ColType.NUMERIC]
    if not dt_cols:
        return []
    date_col = dt_cols[0]
    dt_stats = dt_an.analyze_datetime(df[date_col])
    if dt_stats.get("empty"):
        return []

    df2 = df.copy()
    df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce")
    targets = num_cols[:MAX_CHARTS_PER_SECTION] or [None]
    charts = []
    for value_col in targets:
        if value_col is None:
            s = df2.set_index(date_col).resample(dt_stats["resample_freq"]).size().reset_index(name="count")
            fig = px.line(s, x=date_col, y="count", title="Record count over time")
            takeaway = f"Data spans {dt_stats['span_days']} days; trend is {dt_stats['trend_direction']}."
        else:
            s = df2.set_index(date_col)[value_col].resample(dt_stats["resample_freq"]).sum().reset_index()
            fig = px.line(s, x=date_col, y=value_col, title=f"{value_col} over time")
            peak = s.loc[s[value_col].idxmax()]
            avg = s[value_col].mean()
            ratio = round(peak[value_col] / avg, 1) if avg else 0
            takeaway = f"{value_col} peaks around {peak[date_col].date()}, {ratio}x the average period."
        if dt_stats.get("gap_count"):
            takeaway += f" {dt_stats['gap_count']} gap period(s) detected (largest: {dt_stats['max_gap_periods']})."
        charts.append(ChartSpec("trends", fig.layout.title.text, fig, takeaway,
                                 {**dt_stats, "value_col": value_col}))
    return charts


def build_relationship_charts(df, profile: DatasetProfile) -> list[ChartSpec]:
    charts = []
    num_cols = [c for c, p in profile.columns.items() if p.dtype == ColType.NUMERIC]
    cat_cols = [c for c, p in profile.columns.items() if p.dtype == ColType.CATEGORICAL]

    if len(num_cols) >= 2:
        for method in ("pearson", "spearman"):
            info = rel_an.numeric_correlations(df[num_cols], method=method)
            if not info["matrix"]:
                continue
            fig = px.imshow(pd.DataFrame(info["matrix"]), text_auto=".2f",
                             title=f"Correlation matrix ({method.title()})")
            top = info["top_pairs"][0] if info["top_pairs"] else None
            takeaway = (f"Strongest {method} relationship: {top['col_a']} vs {top['col_b']} (r={top['r']})."
                        if top else f"No strong {method} relationships found (threshold not met).")
            charts.append(ChartSpec("relationships", f"Correlation matrix ({method})", fig, takeaway, info))

            if method == "pearson":
                for pair in info["top_pairs"]:
                    sfig = px.scatter(df, x=pair["col_a"], y=pair["col_b"], trendline="ols",
                                       title=f"{pair['col_a']} vs {pair['col_b']} (r={pair['r']})")
                    charts.append(ChartSpec("relationships", f"{pair['col_a']} vs {pair['col_b']}", sfig,
                                             f"Correlation of {pair['r']} between {pair['col_a']} and {pair['col_b']}.",
                                             pair))

        vifs = rel_an.variance_inflation_factors(df, num_cols)
        if vifs:
            vdf = pd.Series(vifs).sort_values(ascending=False).reset_index()
            vdf.columns = ["column", "VIF"]
            fig = px.bar(vdf, x="VIF", y="column", orientation="h", title="Multicollinearity check (VIF)")
            worst_col, worst_vif = vdf.iloc[0]["column"], vdf.iloc[0]["VIF"]
            charts.append(ChartSpec("relationships", "Multicollinearity check", fig,
                                     f"{worst_col} is highly redundant with other numeric columns (VIF={worst_vif}).",
                                     {"vifs": vifs}))

    if len(cat_cols) >= 2:
        assoc = rel_an.categorical_associations(df, cat_cols)
        if assoc["top_pairs"]:
            cv_df = pd.DataFrame(assoc["top_pairs"])
            cv_df["pair"] = cv_df["col_a"] + " vs " + cv_df["col_b"]
            fig = px.bar(cv_df, x="cramers_v", y="pair", orientation="h",
                         title="Strongest categorical associations (Cramer's V)")
            top = assoc["top_pairs"][0]
            charts.append(ChartSpec("relationships", "Categorical associations", fig,
                           f"{top['col_a']} and {top['col_b']} are strongly associated "
                           f"(Cramer's V={top['cramers_v']}).", assoc))

    return charts[:MAX_CHARTS_PER_SECTION]


def build_segment_charts(df, profile: DatasetProfile) -> list[ChartSpec]:
    num_cols = [c for c, p in profile.columns.items() if p.dtype == ColType.NUMERIC]
    cat_cols = [c for c, p in profile.columns.items() if p.dtype == ColType.CATEGORICAL]
    ranked_pairs = rel_an.rank_categorical_numeric_pairs(df, cat_cols, num_cols)

    charts = []
    for pair in ranked_pairs:
        cat_col, num_col = pair["cat_col"], pair["num_col"]
        n_groups = df[cat_col].nunique(dropna=True)
        if n_groups <= 8:
            fig = px.box(df, x=cat_col, y=num_col, title=f"{num_col} distribution by {cat_col}")
        else:
            info = rel_an.group_comparison(df, cat_col, num_col)
            plot_df = pd.DataFrame(list(info["group_means"].items()), columns=[cat_col, "mean"])
            fig = px.bar(plot_df, x=cat_col, y="mean", title=f"Mean {num_col} by {cat_col}")
        takeaway = f"{cat_col} explains {round(pair['eta_squared'] * 100, 1)}% of the variance in {num_col}."
        charts.append(ChartSpec("segments", f"{num_col} by {cat_col}", fig, takeaway, pair))

    return charts[:MAX_CHARTS_PER_SECTION]


def enforce_global_budget(charts_by_section: dict[str, list[ChartSpec]],
                           budget: int = GLOBAL_CHART_BUDGET) -> dict[str, list[ChartSpec]]:
    total = sum(len(v) for v in charts_by_section.values())
    if total <= budget:
        return charts_by_section
    trimmed = {k: [] for k in charts_by_section}
    remaining = budget
    for section in TRIM_PRIORITY:
        specs = charts_by_section.get(section, [])
        take = min(len(specs), max(0, remaining))
        trimmed[section] = specs[:take]
        remaining -= take
    return trimmed


def build_all_charts(df, profile: DatasetProfile) -> dict[str, list[ChartSpec]]:
    charts = {
        "data_quality": build_quality_charts(df, profile),
        "numeric": build_numeric_charts(df, profile),
        "categorical": build_categorical_charts(df, profile),
        "relationships": build_relationship_charts(df, profile),
        "trends": build_datetime_charts(df, profile),
        "segments": build_segment_charts(df, profile),
    }
    return enforce_global_budget(charts)
