from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

from .profiler import DatasetProfile, ColType


def run_clustering(df: pd.DataFrame, profile: DatasetProfile, k_max: int = 6) -> dict:
    num_cols = [c for c, p in profile.columns.items() if p.dtype == ColType.NUMERIC]
    if len(num_cols) < 2:
        return {"available": False, "reason": "Need at least 2 numeric columns for clustering."}

    X = df[num_cols].dropna()
    if len(X) < 10:
        return {"available": False, "reason": "Not enough complete numeric rows for clustering."}

    Xs = StandardScaler().fit_transform(X)
    best_k, best_score, best_labels = 2, -1, None
    for k in range(2, min(k_max, len(X) - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xs)
        score = silhouette_score(Xs, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    coords = PCA(n_components=2, random_state=42).fit_transform(Xs)
    plot_df = pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1], "cluster": best_labels.astype(str)})

    sizes = pd.Series(best_labels).value_counts().sort_index()
    return {
        "available": True, "k": best_k, "silhouette_score": round(float(best_score), 3),
        "cluster_sizes": sizes.to_dict(), "plot_df": plot_df, "columns_used": num_cols,
    }


def run_feature_importance(df: pd.DataFrame, profile: DatasetProfile, target: str) -> dict:
    if target not in profile.columns:
        return {"available": False, "reason": f"{target} not found."}

    target_type = profile.columns[target].dtype
    is_classification = target_type in (ColType.CATEGORICAL, ColType.BOOLEAN)

    feature_cols = [c for c, p in profile.columns.items()
                    if c != target and p.dtype in (ColType.NUMERIC, ColType.CATEGORICAL, ColType.BOOLEAN)]
    if not feature_cols:
        return {"available": False, "reason": "No usable feature columns."}

    data = df[feature_cols + [target]].dropna()
    if len(data) < 30:
        return {"available": False, "reason": "Not enough complete rows (need 30+)."}

    X = pd.get_dummies(data[feature_cols], drop_first=True)
    y = data[target]

    if is_classification:
        if y.nunique() < 2:
            return {"available": False, "reason": "Target has fewer than 2 classes."}
        model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=8)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=8)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return {
        "available": True, "target": target, "task": "classification" if is_classification else "regression",
        "holdout_score": round(float(score), 3),
        "score_metric": "accuracy" if is_classification else "r2",
        "top_features": importances.head(10).round(3).to_dict(),
    }


def run_forecast(df: pd.DataFrame, profile: DatasetProfile, date_col: str, value_col: str,
                  periods_ahead: int = 6) -> dict:
    s = df[[date_col, value_col]].dropna()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s = s.dropna().sort_values(date_col)
    if len(s) < 8:
        return {"available": False, "reason": "Need at least 8 data points to forecast."}

    span_days = (s[date_col].max() - s[date_col].min()).days
    freq = "D" if span_days <= 60 else "W" if span_days <= 730 else "M"
    series = s.set_index(date_col)[value_col].resample(freq).sum()

    x = np.arange(len(series))
    slope, intercept = np.polyfit(x, series.values, 1)
    future_x = np.arange(len(series), len(series) + periods_ahead)
    forecast_values = slope * future_x + intercept

    freq_map = {"D": "D", "W": "W", "M": "MS"}
    future_index = pd.date_range(series.index[-1], periods=periods_ahead + 1, freq=freq_map[freq])[1:]

    return {
        "available": True, "freq": freq, "trend_slope": float(slope),
        "history": series.reset_index().rename(columns={value_col: "value"}).to_dict("records"),
        "forecast": [{"date": str(d), "value": float(v)} for d, v in zip(future_index, forecast_values)],
        "caveat": "Simple linear trend extrapolation -- treat as a directional estimate, not a precise prediction.",
    }
