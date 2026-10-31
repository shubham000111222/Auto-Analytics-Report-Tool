from __future__ import annotations
import duckdb


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def exact_numeric_stats(con: duckdb.DuckDBPyConnection, table: str, col: str) -> dict:
    q = quote_ident(col)
    sql = f"""
        SELECT
            COUNT({q}) AS count, AVG({q}) AS mean, STDDEV_POP({q}) AS std,
            MIN({q}) AS min, MAX({q}) AS max,
            APPROX_QUANTILE({q}, 0.25) AS q1, APPROX_QUANTILE({q}, 0.5) AS median,
            APPROX_QUANTILE({q}, 0.75) AS q3,
            SKEWNESS({q}) AS skew, KURTOSIS({q}) AS kurtosis,
            SUM(CASE WHEN {q} = 0 THEN 1 ELSE 0 END) AS zero_count,
            SUM(CASE WHEN {q} < 0 THEN 1 ELSE 0 END) AS negative_count
        FROM {table}
    """
    cols = ["count", "mean", "std", "min", "max", "q1", "median", "q3",
            "skew", "kurtosis", "zero_count", "negative_count"]
    row = con.execute(sql).fetchone()
    result = dict(zip(cols, row))

    if result["q1"] is not None and result["q3"] is not None and result["count"]:
        iqr = result["q3"] - result["q1"]
        lower, upper = result["q1"] - 1.5 * iqr, result["q3"] + 1.5 * iqr
        n_outliers = con.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {q} < ? OR {q} > ?", [lower, upper]
        ).fetchone()[0]
        result["iqr"] = iqr
        result["iqr_outlier_count"] = int(n_outliers)
        result["iqr_outlier_pct"] = round(n_outliers / result["count"] * 100, 2)
    return result


def exact_missing_and_duplicates(con: duckdb.DuckDBPyConnection, table: str, columns: list[str]) -> dict:
    missing = {}
    for col in columns:
        q = quote_ident(col)
        n = con.execute(f"SELECT SUM(CASE WHEN {q} IS NULL THEN 1 ELSE 0 END) FROM {table}").fetchone()[0]
        missing[col] = int(n or 0)
    total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    distinct = con.execute(f"SELECT COUNT(*) FROM (SELECT DISTINCT * FROM {table})").fetchone()[0]
    return {"missing": missing, "duplicate_rows": int(total - distinct), "total_rows": int(total)}


def exact_cardinality(con: duckdb.DuckDBPyConnection, table: str, col: str) -> int:
    q = quote_ident(col)
    return con.execute(f"SELECT COUNT(DISTINCT {q}) FROM {table}").fetchone()[0]


def exact_top_categories(con: duckdb.DuckDBPyConnection, table: str, col: str, top_n: int = 10) -> dict:
    q = quote_ident(col)
    sql = f"""
        SELECT {q} AS val, COUNT(*) AS cnt FROM {table}
        WHERE {q} IS NOT NULL GROUP BY {q} ORDER BY cnt DESC LIMIT {top_n}
    """
    return {str(r[0]): int(r[1]) for r in con.execute(sql).fetchall()}
