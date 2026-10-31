from __future__ import annotations
from .charts import DISPLAY_ORDER
from .exporters import fig_to_static_img

SECTION_TITLES = {
    "data_quality": "Data Quality",
    "numeric": "Numeric Column Profiles",
    "categorical": "Categorical Column Profiles",
    "relationships": "Relationships",
    "trends": "Trends Over Time",
    "segments": "Segment Comparisons",
}


def build_caveats(profile, load, cleaning_suggestions, cleaning_log) -> list[str]:
    caveats = []
    if load.is_sampled:
        caveats.append(
            f"This dataset has {load.row_count_full:,} rows. Missing-value counts, duplicate counts, "
            f"cardinality, and numeric aggregates (mean/median/std/outliers) were computed exactly "
            f"against the full dataset via SQL. Charts, categorical top-values, datetime trend, and "
            f"forecast analyses are drawn from a {len(load.df):,}-row sample."
        )
    caveats.extend(load.warnings)
    caveats.extend(profile.warnings)

    applied_ids = {line.split("]")[0].strip("[") for line in cleaning_log} if cleaning_log else set()
    unapplied = [s for s in cleaning_suggestions if s.id not in applied_ids]
    if unapplied:
        caveats.append(f"{len(unapplied)} cleaning suggestion(s) were identified but not applied -- "
                        f"see the Cleaning tab for details.")
    if cleaning_log:
        caveats.append(f"{len(cleaning_log)} cleaning action(s) were applied before this report was generated.")
    return caveats


def assemble_report_context(pipeline_result, filename: str) -> dict:
    sections = []
    for key in DISPLAY_ORDER:
        specs = pipeline_result.charts.get(key, [])
        if not specs:
            continue
        sections.append({
            "key": key, "title": SECTION_TITLES.get(key, key.title()),
            "charts": [{"title": c.title,
                        "html": c.fig.to_html(full_html=False, include_plotlyjs=True),
                        "takeaway": c.takeaway} for c in specs],
        })
    caveats = build_caveats(pipeline_result.profile, pipeline_result.load,
                             pipeline_result.cleaning_suggestions, pipeline_result.cleaning_log)
    return {
        "filename": filename, "profile": pipeline_result.profile,
        "narrative": pipeline_result.narrative.text if pipeline_result.narrative else None,
        "sections": sections, "caveats": caveats,
    }


def assemble_pdf_context(pipeline_result, filename: str) -> dict:
    sections = []
    for key in DISPLAY_ORDER:
        specs = pipeline_result.charts.get(key, [])
        if not specs:
            continue
        sections.append({
            "key": key, "title": SECTION_TITLES.get(key, key.title()),
            "charts": [{"title": c.title, "img_b64": fig_to_static_img(c.fig), "takeaway": c.takeaway}
                       for c in specs],
        })
    caveats = build_caveats(pipeline_result.profile, pipeline_result.load,
                             pipeline_result.cleaning_suggestions, pipeline_result.cleaning_log)
    return {
        "filename": filename, "profile": pipeline_result.profile,
        "narrative": pipeline_result.narrative.text if pipeline_result.narrative else None,
        "sections": sections, "caveats": caveats,
    }
