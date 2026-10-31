from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .db import get_db
from .models import Report, ReportStatus
from engine.profiler import profile_dataset
from engine.optional_analysis import run_clustering, run_feature_importance, run_forecast

router = APIRouter(prefix="/reports", tags=["advanced-analysis"])


def _load_report_df(report_id: str, db: Session) -> pd.DataFrame:
    report = db.query(Report).get(report_id)
    if not report or report.status != ReportStatus.DONE or not report.result_data_path:
        raise HTTPException(404, "Report data not available")
    return pd.read_parquet(report.result_data_path)


@router.get("/{report_id}/columns")
def list_columns(report_id: str, db: Session = Depends(get_db)):
    df = _load_report_df(report_id, db)
    profile = profile_dataset(df)
    return {name: p.dtype.value for name, p in profile.columns.items()}


@router.post("/{report_id}/clustering")
def clustering(report_id: str, k_max: int = Query(6, ge=2, le=15), db: Session = Depends(get_db)):
    df = _load_report_df(report_id, db)
    result = run_clustering(df, profile_dataset(df), k_max=k_max)
    if not result["available"]:
        raise HTTPException(400, result["reason"])
    fig = px.scatter(result["plot_df"], x="pc1", y="pc2", color="cluster",
                      title=f"Clusters (k={result['k']}, silhouette={result['silhouette_score']})")
    return {"k": result["k"], "silhouette_score": result["silhouette_score"],
            "cluster_sizes": result["cluster_sizes"], "columns_used": result["columns_used"],
            "chart_html": fig.to_html(full_html=False, include_plotlyjs="cdn")}


@router.post("/{report_id}/feature-importance")
def feature_importance(report_id: str, target: str = Query(...), db: Session = Depends(get_db)):
    df = _load_report_df(report_id, db)
    result = run_feature_importance(df, profile_dataset(df), target=target)
    if not result["available"]:
        raise HTTPException(400, result["reason"])
    imp_df = pd.Series(result["top_features"]).sort_values().reset_index()
    imp_df.columns = ["feature", "importance"]
    fig = px.bar(imp_df, x="importance", y="feature", orientation="h",
                 title=f"Feature importance for {target}")
    return {**result, "chart_html": fig.to_html(full_html=False, include_plotlyjs="cdn")}


@router.post("/{report_id}/forecast")
def forecast(report_id: str, date_col: str = Query(...), value_col: str = Query(...),
             periods_ahead: int = Query(6, ge=1, le=52), db: Session = Depends(get_db)):
    df = _load_report_df(report_id, db)
    result = run_forecast(df, profile_dataset(df), date_col=date_col, value_col=value_col,
                           periods_ahead=periods_ahead)
    if not result["available"]:
        raise HTTPException(400, result["reason"])
    hist, fc = pd.DataFrame(result["history"]), pd.DataFrame(result["forecast"])
    fig = go.Figure()
    fig.add_scatter(x=hist[date_col], y=hist["value"], name="history")
    fig.add_scatter(x=fc["date"], y=fc["value"], name="forecast", line=dict(dash="dash"))
    return {**result, "chart_html": fig.to_html(full_html=False, include_plotlyjs="cdn")}
