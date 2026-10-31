from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import pandas as pd

from .profiler import DatasetProfile, ColType
from .analyzers.numeric import analyze_numeric
from .analyzers.categorical import analyze_categorical


@dataclass
class CleaningSuggestion:
    id: str
    column: str | None
    issue_type: str
    action: str
    description: str
    affected_count: int
    affected_pct: float | None
    severity: str
    preview_before: list = field(default_factory=list)
    preview_after: list = field(default_factory=list)


def suggest_missing_value_fixes(df: pd.DataFrame, profile: DatasetProfile) -> list[CleaningSuggestion]:
    out = []
    for name, p in profile.columns.items():
        if p.missing_count == 0 or p.dtype == ColType.CONSTANT:
            continue
        if p.missing_pct > 60:
            action, desc, sev = "drop_column", f"{name} is {p.missing_pct}% missing -- consider dropping it.", "high"
        elif p.dtype == ColType.NUMERIC:
            action, desc, sev = "fill_median", f"Fill {p.missing_count} missing values in {name} with the median.", "medium"
        elif p.dtype in (ColType.CATEGORICAL, ColType.BOOLEAN):
            action, desc, sev = "fill_mode", f"Fill {p.missing_count} missing values in {name} with the most common value.", "low"
        elif p.dtype == ColType.DATETIME:
            action, desc, sev = "drop_rows_missing", f"Drop {p.missing_count} rows missing a date in {name}.", "medium"
        else:
            action, desc, sev = "fill_placeholder", f"Fill {p.missing_count} missing text values in {name} with 'Unknown'.", "low"

        out.append(CleaningSuggestion(
            id=f"missing::{name}", column=name, issue_type="missing_values", action=action,
            description=desc, affected_count=p.missing_count, affected_pct=p.missing_pct, severity=sev,
            preview_before=df[name].head(5).tolist(),
        ))
    return out


def suggest_duplicate_fixes(df: pd.DataFrame, profile: DatasetProfile) -> list[CleaningSuggestion]:
    if profile.duplicate_rows == 0:
        return []
    pct = round(profile.duplicate_rows / profile.n_rows * 100, 2) if profile.n_rows else 0
    return [CleaningSuggestion(
        id="duplicates::rows", column=None, issue_type="duplicates", action="drop_duplicates",
        description=f"Remove {profile.duplicate_rows} exact duplicate rows ({pct}% of data).",
        affected_count=profile.duplicate_rows, affected_pct=pct,
        severity="high" if pct > 10 else "medium",
    )]


def suggest_outlier_fixes(df: pd.DataFrame, profile: DatasetProfile) -> list[CleaningSuggestion]:
    out = []
    for name, p in profile.columns.items():
        if p.dtype != ColType.NUMERIC:
            continue
        stats = analyze_numeric(df[name])
        if stats.get("empty") or stats["iqr_outlier_count"] == 0:
            continue
        pct = stats["iqr_outlier_pct"]
        out.append(CleaningSuggestion(
            id=f"outliers::{name}", column=name, issue_type="outliers", action="cap_outliers_iqr",
            description=f"{stats['iqr_outlier_count']} outlier(s) in {name} ({pct}%) fall outside 1.5x IQR. "
                        f"Suggest capping (winsorizing) rather than deleting.",
            affected_count=stats["iqr_outlier_count"], affected_pct=pct,
            severity="high" if pct > 10 else "medium" if pct > 2 else "low",
        ))
    return out


def suggest_category_normalization(df: pd.DataFrame, profile: DatasetProfile) -> list[CleaningSuggestion]:
    out = []
    for name, p in profile.columns.items():
        if p.dtype != ColType.CATEGORICAL:
            continue
        stats = analyze_categorical(df[name])
        if stats.get("empty") or stats["possible_inconsistent_categories"] == 0:
            continue
        s = df[name].dropna().astype(str)
        groups = s.groupby(s.str.lower().str.strip()).apply(lambda x: sorted(x.unique().tolist()))
        examples = {k: v for k, v in groups.items() if len(v) > 1}
        out.append(CleaningSuggestion(
            id=f"categories::{name}", column=name, issue_type="inconsistent_categories",
            action="normalize_categories",
            description=f"{name} has {stats['possible_inconsistent_categories']} value(s) that differ only "
                        f"by case/whitespace, e.g. {list(examples.values())[:3]}. Suggest trimming + lowercasing.",
            affected_count=stats["possible_inconsistent_categories"], affected_pct=None,
            severity="medium",
        ))
    return out


def suggest_type_fixes(df: pd.DataFrame, profile: DatasetProfile) -> list[CleaningSuggestion]:
    out = []
    for name, p in profile.columns.items():
        if p.dtype == ColType.MIXED:
            out.append(CleaningSuggestion(
                id=f"mixed::{name}", column=name, issue_type="type_mismatch", action="coerce_numeric_text",
                description=f"{name} contains mixed types (e.g. numbers stored as text). "
                            f"Suggest coercing to numeric where possible, flagging the rest as missing.",
                affected_count=p.unique_count, affected_pct=None, severity="medium",
            ))
    return out


def generate_all_suggestions(df: pd.DataFrame, profile: DatasetProfile) -> list[CleaningSuggestion]:
    return (
        suggest_duplicate_fixes(df, profile)
        + suggest_missing_value_fixes(df, profile)
        + suggest_outlier_fixes(df, profile)
        + suggest_category_normalization(df, profile)
        + suggest_type_fixes(df, profile)
    )


def _act_drop_column(df, col): return df.drop(columns=[col])
def _act_fill_median(df, col): return df.assign(**{col: df[col].fillna(df[col].median())})
def _act_fill_mode(df, col):
    mode = df[col].mode(dropna=True)
    return df.assign(**{col: df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")})
def _act_drop_rows_missing(df, col): return df.dropna(subset=[col])
def _act_fill_placeholder(df, col): return df.assign(**{col: df[col].fillna("Unknown")})
def _act_drop_duplicates(df, col): return df.drop_duplicates()
def _act_cap_outliers_iqr(df, col):
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return df.assign(**{col: df[col].clip(lower, upper)})
def _act_normalize_categories(df, col):
    return df.assign(**{col: df[col].astype(str).str.strip().str.lower()})
def _act_coerce_numeric_text(df, col):
    return df.assign(**{col: pd.to_numeric(df[col], errors="coerce")})

ACTION_REGISTRY: dict[str, Callable[[pd.DataFrame, str | None], pd.DataFrame]] = {
    "drop_column": _act_drop_column, "fill_median": _act_fill_median, "fill_mode": _act_fill_mode,
    "drop_rows_missing": _act_drop_rows_missing, "fill_placeholder": _act_fill_placeholder,
    "drop_duplicates": _act_drop_duplicates, "cap_outliers_iqr": _act_cap_outliers_iqr,
    "normalize_categories": _act_normalize_categories, "coerce_numeric_text": _act_coerce_numeric_text,
}


def apply_cleaning(df: pd.DataFrame, suggestions: list[CleaningSuggestion],
                    accepted_ids: set[str]) -> tuple[pd.DataFrame, list[str]]:
    cleaned = df.copy()
    applied = []
    for s in suggestions:
        if s.id not in accepted_ids:
            continue
        fn = ACTION_REGISTRY.get(s.action)
        if not fn:
            continue
        before_rows, before_cols = cleaned.shape
        cleaned = fn(cleaned, s.column)
        applied.append(f"[{s.id}] {s.description} "
                        f"(shape {before_rows}x{before_cols} -> {cleaned.shape[0]}x{cleaned.shape[1]})")
    return cleaned, applied
