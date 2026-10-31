from __future__ import annotations
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import duckdb
from charset_normalizer import from_bytes

MAX_IN_MEMORY_BYTES = 50 * 1024 * 1024
MAX_TOTAL_BYTES = 500 * 1024 * 1024
SAMPLE_ROWS_FOR_CHARTS = 200_000
SUPPORTED_EXT = {".csv", ".tsv", ".xlsx", ".xls", ".json"}


@dataclass
class LoadResult:
    df: pd.DataFrame
    sheet_names: list[str] | None
    sheet_used: str | None
    row_count_full: int
    is_sampled: bool
    encoding_detected: str | None
    delimiter_detected: str | None
    warnings: list[str]
    duckdb_con: Any = None
    duckdb_table: str | None = None
    cleanup_path: str | None = None


def cleanup_load_result(load: LoadResult) -> None:
    if load.duckdb_con is not None:
        try:
            load.duckdb_con.close()
        except Exception:
            pass
    if load.cleanup_path and os.path.exists(load.cleanup_path):
        try:
            os.remove(load.cleanup_path)
        except Exception:
            pass


def _detect_encoding(raw: bytes) -> str:
    match = from_bytes(raw).best()
    return match.encoding if match else "utf-8"


def _sniff_delimiter(raw: bytes, encoding: str) -> str:
    import csv
    sample = raw[:65536].decode(encoding, errors="replace")
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def validate_file(filename: str, size_bytes: int) -> list[str]:
    errors = []
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXT:
        errors.append(f"Unsupported file type: {ext}")
    if size_bytes > MAX_TOTAL_BYTES:
        errors.append(f"File exceeds max size of {MAX_TOTAL_BYTES // (1024*1024)}MB")
    return errors


def list_excel_sheets(path_or_buffer) -> list[str]:
    return pd.ExcelFile(path_or_buffer, engine="openpyxl").sheet_names


def load_file(filename: str, raw_bytes: bytes, sheet_name: str | None = None) -> LoadResult:
    warnings: list[str] = []
    ext = Path(filename).suffix.lower()
    size = len(raw_bytes)

    if ext in (".csv", ".tsv", ".json"):
        return _load_text_like(ext, raw_bytes, size, warnings)
    if ext in (".xlsx", ".xls"):
        return _load_excel(raw_bytes, size, sheet_name, warnings)
    raise ValueError(f"Unsupported extension: {ext}")


def _load_text_like(ext: str, raw_bytes: bytes, size: int, warnings: list[str]) -> LoadResult:
    encoding = _detect_encoding(raw_bytes)

    if ext == ".json":
        df_full = pd.read_json(io.BytesIO(raw_bytes), encoding=encoding)
        _detect_header_issues(df_full, warnings)
        return LoadResult(df=df_full, sheet_names=None, sheet_used=None,
                           row_count_full=len(df_full), is_sampled=False,
                           encoding_detected=encoding, delimiter_detected=None, warnings=warnings)

    delimiter = "\t" if ext == ".tsv" else _sniff_delimiter(raw_bytes, encoding)

    if size > MAX_IN_MEMORY_BYTES:
        text = raw_bytes.decode(encoding, errors="replace")
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        tmp.write(text.encode("utf-8"))
        tmp.close()

        con = duckdb.connect()
        table = "data"
        safe_path = tmp.name.replace("'", "''")
        safe_delim = delimiter.replace("'", "''")
        con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_csv_auto('{safe_path}', "
                    f"delim='{safe_delim}', header=True)")
        row_count_full = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        sample_df = con.execute(
            f"SELECT * FROM {table} USING SAMPLE {SAMPLE_ROWS_FOR_CHARTS} ROWS (reservoir)"
        ).df()

        warnings.append(f"File has {row_count_full:,} rows -- charts use a "
                         f"{len(sample_df):,}-row sample, but all statistics in this report "
                         f"are computed exactly against the full dataset via SQL.")
        _detect_header_issues(sample_df, warnings)

        return LoadResult(df=sample_df, sheet_names=None, sheet_used=None,
                           row_count_full=row_count_full, is_sampled=True,
                           encoding_detected=encoding, delimiter_detected=delimiter,
                           warnings=warnings, duckdb_con=con, duckdb_table=table,
                           cleanup_path=tmp.name)

    try:
        df_full = pd.read_csv(io.BytesIO(raw_bytes), delimiter=delimiter, encoding=encoding)
    except UnicodeDecodeError:
        encoding = "latin-1"
        warnings.append("Fell back to latin-1 encoding after decode failure.")
        df_full = pd.read_csv(io.BytesIO(raw_bytes), delimiter=delimiter, encoding=encoding)

    _detect_header_issues(df_full, warnings)
    return LoadResult(df=df_full, sheet_names=None, sheet_used=None,
                       row_count_full=len(df_full), is_sampled=False,
                       encoding_detected=encoding, delimiter_detected=delimiter, warnings=warnings)


def _load_excel(raw_bytes: bytes, size: int, sheet_name: str | None, warnings: list[str]) -> LoadResult:
    buf = io.BytesIO(raw_bytes)
    sheets = list_excel_sheets(buf)
    chosen = sheet_name or sheets[0]
    if sheet_name is None and len(sheets) > 1:
        warnings.append(f"Workbook has {len(sheets)} sheets; defaulted to '{chosen}'.")

    df_full = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=chosen, engine="openpyxl")
    _detect_header_issues(df_full, warnings)
    full_len = len(df_full)

    if full_len <= SAMPLE_ROWS_FOR_CHARTS:
        return LoadResult(df=df_full, sheet_names=sheets, sheet_used=chosen,
                           row_count_full=full_len, is_sampled=False,
                           encoding_detected=None, delimiter_detected=None, warnings=warnings)

    con = duckdb.connect()
    con.register("tmp_view", df_full)
    con.execute("CREATE TABLE full_data AS SELECT * FROM tmp_view")
    con.unregister("tmp_view")
    sample_df = con.execute(
        f"SELECT * FROM full_data USING SAMPLE {SAMPLE_ROWS_FOR_CHARTS} ROWS (reservoir)"
    ).df()
    del df_full

    warnings.append(f"Sheet has {full_len:,} rows -- charts use a {len(sample_df):,}-row sample; "
                     f"statistics are computed exactly against the full sheet.")
    return LoadResult(df=sample_df, sheet_names=sheets, sheet_used=chosen,
                       row_count_full=full_len, is_sampled=True,
                       encoding_detected=None, delimiter_detected=None, warnings=warnings,
                       duckdb_con=con, duckdb_table="full_data")


def _detect_header_issues(df: pd.DataFrame, warnings: list[str]) -> None:
    unnamed = sum(1 for c in df.columns if str(c).startswith("Unnamed:"))
    if df.shape[1] and unnamed / df.shape[1] > 0.5:
        warnings.append("Many unnamed columns detected -- header row may be misaligned.")
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        warnings.append(f"{len(empty_cols)} fully empty column(s) detected: {empty_cols[:5]}")
