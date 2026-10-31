from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression


def numeric_correlations(df: pd.DataFrame, method: str = "pearson",
                          threshold: float = 0.3, max_pairs: int = 5) -> dict:
    num_df = df.select_dtypes("number")
    if num_df.shape[1] < 2:
        return {"matrix": {}, "top_pairs": [], "method": method}

    corr = num_df.corr(method=method)
    pairs = []
    for a, b in itertools.combinations(corr.columns, 2):
        r = corr.loc[a, b]
        if pd.notna(r) and abs(r) >= threshold:
            pairs.append({"col_a": a, "col_b": b, "r": round(float(r), 3)})
    pairs.sort(key=lambda p: abs(p["r"]), reverse=True)
    return {"matrix": corr.round(3).to_dict(), "top_pairs": pairs[:max_pairs], "method": method}


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    confusion = pd.crosstab(x, y)
    if confusion.size == 0:
        return 0.0
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def categorical_associations(df: pd.DataFrame, cat_cols: list[str],
                              threshold: float = 0.3, max_pairs: int = 5) -> dict:
    pairs = []
    for a, b in itertools.combinations(cat_cols, 2):
        v = cramers_v(df[a].dropna(), df[b].dropna())
        if v >= threshold:
            pairs.append({"col_a": a, "col_b": b, "cramers_v": round(v, 3)})
    pairs.sort(key=lambda p: p["cramers_v"], reverse=True)
    return {"top_pairs": pairs[:max_pairs]}


def group_comparison(df: pd.DataFrame, cat_col: str, num_col: str, top_n: int = 10) -> dict:
    grouped = (df.groupby(cat_col)[num_col].agg(["mean", "median", "count"])
               .sort_values("mean", ascending=False).head(top_n))
    overall_mean = df[num_col].mean()
    return {
        "group_means": grouped["mean"].round(3).to_dict(),
        "group_counts": grouped["count"].to_dict(),
        "overall_mean": round(float(overall_mean), 3) if pd.notna(overall_mean) else None,
        "top_group": grouped.index[0] if not grouped.empty else None,
        "top_group_ratio_to_overall": round(float(grouped["mean"].iloc[0] / overall_mean), 2)
            if not grouped.empty and overall_mean else None,
    }


def eta_squared(categorical: pd.Series, numeric: pd.Series) -> float:
    data = pd.DataFrame({"cat": categorical, "num": numeric}).dropna()
    if data.empty or data["cat"].nunique() < 2:
        return 0.0
    grand_mean = data["num"].mean()
    ss_between = data.groupby("cat")["num"].apply(
        lambda x: len(x) * (x.mean() - grand_mean) ** 2
    ).sum()
    ss_total = ((data["num"] - grand_mean) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def rank_categorical_numeric_pairs(df: pd.DataFrame, cat_cols: list[str],
                                    num_cols: list[str], top_n: int = 8) -> list[dict]:
    pairs = []
    for c in cat_cols:
        for n in num_cols:
            eta2 = eta_squared(df[c], df[n])
            if eta2 > 0.01:
                pairs.append({"cat_col": c, "num_col": n, "eta_squared": round(eta2, 3)})
    pairs.sort(key=lambda p: p["eta_squared"], reverse=True)
    return pairs[:top_n]


def variance_inflation_factors(df: pd.DataFrame, num_cols: list[str]) -> dict:
    data = df[num_cols].dropna()
    if data.shape[1] < 2 or len(data) < 10:
        return {}
    vifs = {}
    for col in num_cols:
        X = data.drop(columns=[col])
        y = data[col]
        if X.shape[1] == 0:
            continue
        r2 = LinearRegression().fit(X, y).score(X, y)
        vifs[col] = float("inf") if r2 >= 0.9999 else round(1 / (1 - r2), 2)
    return {k: v for k, v in vifs.items() if v > 5}
