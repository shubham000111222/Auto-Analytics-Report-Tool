from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

from .exact_stats import (exact_missing_and_duplicates, exact_numeric_stats,
                           exact_cardinality, exact_top_categories)


class ColType(str, Enum):
    CONSTANT = "constant"
    BOOLEAN = "boolean"
    ID = "id"
    DATETIME = "datetime"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    MIXED = "mixed"


@dataclass
class ColumnProfile:
    name: str
    dtype: ColType
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    memory_bytes: int
    sample_values: list
    flags: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    duplicate_rows: int
    memory_bytes: int
    columns: dict[str, ColumnProfile]
    warnings: list[str] = field(default_factory=list)


ID_NAME_HINTS = ("id", "uuid", "guid", "_key", "code")


def _looks_like_id(name: str, series: pd.Series, n_rows: int) -> bool:
    name_hit = any(h in name.lower() for h in ID_NAME_HINTS)
    high_uniqueness = series.nunique(dropna=True) / max(n_rows, 1) > 0.95
    return high_uniqueness and (name_hit or series.dtype == object)


def _try_parse_datetime(series: pd.Series) -> tuple[bool, pd.Series | None]:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True, series
    if series.dtype != object:
        return False, None
    non_null = series.dropna()
    if non_null.empty:
        return False, None
    parsed = pd.to_datetime(non_null, errors="coerce", infer_datetime_format=True)
    success_rate = parsed.notna().mean()
    if success_rate > 0.9:
        return True, pd.to_datetime(series, errors="coerce")
    return False, None


def detect_column_type(name: str, series: pd.Series, n_rows: int) -> tuple[ColType, list[str]]:
    flags = []
    non_null = series.dropna()

    if non_null.empty:
        flags.append("entirely null")
        return ColType.CONSTANT, flags

    if series.nunique(dropna=True) == 1:
        flags.append("constant value")
        return ColType.CONSTANT, flags

    if pd.api.types.is_bool_dtype(series):
        return ColType.BOOLEAN, flags
    if set(non_null.unique().tolist()) <= {0, 1, "0", "1", True, False}:
        if series.nunique(dropna=True) == 2:
            return ColType.BOOLEAN, flags

    if _looks_like_id(name, series, n_rows):
        flags.append("high-cardinality identifier")
        return ColType.ID, flags

    is_dt, _ = _try_parse_datetime(series)
    if is_dt:
        return ColType.DATETIME, flags

    if pd.api.types.is_numeric_dtype(series):
        return ColType.NUMERIC, flags

    if series.dtype == object:
        types_in_col = non_null.map(type).nunique()
        if types_in_col > 1:
            flags.append("mixed types within column")
            return ColType.MIXED, flags

        uniqueness_ratio = series.nunique(dropna=True) / max(n_rows, 1)
        avg_len = non_null.astype(str).str.len().mean()
        if uniqueness_ratio < 0.5 or series.nunique(dropna=True) <= 50:
            return ColType.CATEGORICAL, flags
        if avg_len > 30:
            return ColType.TEXT, flags
        return ColType.CATEGORICAL, flags

    flags.append("unrecognized dtype")
    return ColType.MIXED, flags


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    n_rows, n_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    warnings = []

    if n_cols > 500:
        warnings.append("Very wide dataset (>500 columns); report will cap charted columns.")
    if n_rows == 0:
        warnings.append("Dataset has zero rows.")

    columns: dict[str, ColumnProfile] = {}
    for col in df.columns:
        series = df[col]
        col_type, flags = detect_column_type(str(col), series, n_rows)
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))

        columns[str(col)] = ColumnProfile(
            name=str(col), dtype=col_type, missing_count=missing,
            missing_pct=round(missing / n_rows * 100, 2) if n_rows else 0.0,
            unique_count=unique,
            unique_pct=round(unique / n_rows * 100, 2) if n_rows else 0.0,
            memory_bytes=int(series.memory_usage(deep=True)),
            sample_values=series.dropna().head(5).tolist(), flags=flags,
        )

    return DatasetProfile(n_rows=n_rows, n_cols=n_cols, duplicate_rows=duplicate_rows,
                           memory_bytes=int(df.memory_usage(deep=True).sum()),
                           columns=columns, warnings=warnings)


def apply_exact_stats(profile: DatasetProfile, load) -> None:
    """Overwrites sample-derived counts with exact SQL results from the full dataset.
    Call only when load.is_sampled is True and load.duckdb_con is available."""
    con, table = load.duckdb_con, load.duckdb_table
    if con is None or table is None:
        return

    col_names = list(profile.columns.keys())
    if len(col_names) > 300:
        profile.warnings.append("Too many columns for exact per-column DuckDB stats; "
                                 "missing/duplicate counts remain sample-based estimates.")
    else:
        agg = exact_missing_and_duplicates(con, table, col_names)
        profile.duplicate_rows = agg["duplicate_rows"]
        profile.n_rows = agg["total_rows"]
        for name, p in profile.columns.items():
            p.missing_count = agg["missing"].get(name, p.missing_count)
            p.missing_pct = round(p.missing_count / agg["total_rows"] * 100, 2) if agg["total_rows"] else 0.0

    for name, p in profile.columns.items():
        try:
            if p.dtype == ColType.NUMERIC:
                p.stats["exact"] = exact_numeric_stats(con, table, name)
                p.unique_count = exact_cardinality(con, table, name)
            elif p.dtype == ColType.CATEGORICAL:
                p.stats["exact_top_values"] = exact_top_categories(con, table, name)
                p.unique_count = exact_cardinality(con, table, name)
            else:
                continue
            p.unique_pct = round(p.unique_count / profile.n_rows * 100, 2) if profile.n_rows else 0.0
        except Exception as e:
            p.flags.append(f"exact stats unavailable: {e}")
