import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine.loader import validate_file, list_excel_sheets
from engine.security import full_security_check
from engine.config import settings
from engine.pipeline import run_pipeline, export_html, export_pdf, export_excel
from engine.charts import DISPLAY_ORDER
from engine.optional_analysis import run_clustering, run_feature_importance, run_forecast

SECTION_TITLES = {
    "data_quality": "Data Quality", "numeric": "Numeric Columns", "categorical": "Categorical Columns",
    "relationships": "Relationships", "trends": "Trends", "segments": "Segment Comparisons",
}

st.set_page_config(page_title="Auto Analytics Report", layout="wide")
st.title("Auto Analytics Report Generator")
st.caption("Only aggregated statistics -- never raw rows -- are sent to the AI narrative engine.")

uploaded = st.file_uploader("Upload CSV / XLSX / XLS / TSV / JSON",
                             type=["csv", "xlsx", "xls", "tsv", "json"])

if uploaded:
    raw = uploaded.read()
    errors = validate_file(uploaded.name, len(raw)) + full_security_check(
        uploaded.name, raw, settings.allowed_extensions)
    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    sheet_name = None
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        sheet_name = st.selectbox("Sheet to analyze", list_excel_sheets(uploaded))

    if st.session_state.get("_file_key") != uploaded.name:
        with st.spinner("Running a safety pre-check, then parsing and profiling..."):
            try:
                st.session_state["pipeline_result"] = run_pipeline(
                    uploaded.name, raw, sheet_name=sheet_name, skip_narrative=True)
            except ValueError as e:
                st.error(str(e))
                st.stop()
        st.session_state["_file_key"] = uploaded.name
        st.session_state["accepted_ids"] = set()

    result = st.session_state["pipeline_result"]
    for w in result.load.warnings:
        st.warning(w)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{result.load.row_count_full:,}")
    c2.metric("Columns", result.profile.n_cols)
    c3.metric("Duplicate rows", result.profile.duplicate_rows)
    c4.metric("Memory", f"{result.profile.memory_bytes / 1e6:.1f} MB")

    tabs = st.tabs(["Cleaning"] + [SECTION_TITLES[k] for k in DISPLAY_ORDER] +
                    ["Advanced", "AI Summary", "Export"])

    with tabs[0]:
        st.caption("Nothing here changes your data until you select suggestions and rebuild.")
        accepted = set()
        for s in result.cleaning_suggestions:
            if st.checkbox(f"[{s.severity.upper()}] {s.description}", key=s.id,
                            value=s.id in st.session_state["accepted_ids"]):
                accepted.add(s.id)
        if st.button("Apply selected cleaning & rebuild report"):
            st.session_state["accepted_ids"] = accepted
            with st.spinner("Rebuilding with cleaning applied..."):
                st.session_state["pipeline_result"] = run_pipeline(
                    uploaded.name, raw, sheet_name=sheet_name,
                    accepted_cleaning_ids=accepted, skip_narrative=True)
            st.rerun()
        if result.cleaning_log:
            st.success("Applied so far:")
            for line in result.cleaning_log:
                st.text(line)

    for i, key in enumerate(DISPLAY_ORDER, start=1):
        with tabs[i]:
            specs = result.charts.get(key, [])
            if not specs:
                st.info("Nothing to show for this section.")
            for spec in specs:
                st.plotly_chart(spec.fig, use_container_width=True)
                st.caption(spec.takeaway)

    with tabs[len(DISPLAY_ORDER) + 1]:
        df, profile = result.cleaned_df, result.profile

        st.subheader("Clustering")
        if st.button("Run clustering"):
            with st.spinner("Clustering..."):
                r = run_clustering(df, profile)
            if r["available"]:
                st.plotly_chart(px.scatter(r["plot_df"], x="pc1", y="pc2", color="cluster",
                                 title=f"k={r['k']}, silhouette={r['silhouette_score']}"),
                                 use_container_width=True)
                st.write("Cluster sizes:", r["cluster_sizes"])
            else:
                st.info(r["reason"])

        st.divider()
        st.subheader("Feature importance")
        target_col = st.selectbox("Target column", options=list(df.columns), key="fi_target")
        if st.button("Run feature importance"):
            with st.spinner("Training a quick model..."):
                r = run_feature_importance(df, profile, target=target_col)
            if r["available"]:
                st.write(f"Holdout {r['score_metric']}: **{r['holdout_score']}**")
                imp = pd.Series(r["top_features"]).sort_values()
                st.plotly_chart(px.bar(imp, orientation="h", title=f"Feature importance for {target_col}"),
                                 use_container_width=True)
            else:
                st.info(r["reason"])

        st.divider()
        st.subheader("Forecast")
        date_cols = [c for c, p in profile.columns.items() if p.dtype.value == "datetime"]
        num_cols_f = [c for c, p in profile.columns.items() if p.dtype.value == "numeric"]
        if date_cols and num_cols_f:
            date_col = st.selectbox("Date column", date_cols)
            value_col = st.selectbox("Value column", num_cols_f)
            horizon = st.slider("Periods ahead", 1, 24, 6)
            if st.button("Run forecast"):
                with st.spinner("Forecasting..."):
                    r = run_forecast(df, profile, date_col=date_col, value_col=value_col, periods_ahead=horizon)
                if r["available"]:
                    hist, fc = pd.DataFrame(r["history"]), pd.DataFrame(r["forecast"])
                    fig = go.Figure()
                    fig.add_scatter(x=hist[date_col], y=hist["value"], name="history")
                    fig.add_scatter(x=fc["date"], y=fc["value"], name="forecast", line=dict(dash="dash"))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(r["caveat"])
                else:
                    st.info(r["reason"])
        else:
            st.info("Need at least one datetime and one numeric column to forecast.")

    with tabs[len(DISPLAY_ORDER) + 2]:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            st.warning("Set ANTHROPIC_API_KEY to enable the AI narrative.")
        elif result.narrative:
            st.markdown(result.narrative.text)
        elif st.button("Generate AI narrative"):
            with st.spinner("Writing narrative from computed statistics..."):
                full_result = run_pipeline(uploaded.name, raw, sheet_name=sheet_name,
                                            accepted_cleaning_ids=st.session_state["accepted_ids"],
                                            api_key=api_key)
            st.session_state["pipeline_result"] = full_result
            if not full_result.narrative.valid:
                st.warning("Some AI claims were unverifiable; showing a safe fallback summary instead.")
            st.markdown(full_result.narrative.text)

    with tabs[len(DISPLAY_ORDER) + 3]:
        html = export_html(result, uploaded.name)
        st.download_button("Download HTML report", html, "report.html", "text/html")
        try:
            st.download_button("Download PDF report", export_pdf(result, uploaded.name),
                                "report.pdf", "application/pdf")
        except Exception as e:
            st.caption(f"PDF export unavailable here: {e}")
        st.download_button("Download Excel workbook", export_excel(result), "report.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
