from __future__ import annotations
from dataclasses import dataclass

from .loader import load_file, cleanup_load_result, LoadResult
from .profiler import profile_dataset, apply_exact_stats, DatasetProfile
from .cleaner import generate_all_suggestions, apply_cleaning, CleaningSuggestion
from .charts import build_all_charts, ChartSpec
from .narrative import generate_narrative, NarrativeResult
from .report_sections import assemble_report_context, assemble_pdf_context
from .exporters import render_html, render_pdf_html, html_to_pdf_weasyprint, build_excel_export
from .preparse_check import safe_preparse


@dataclass
class PipelineResult:
    load: LoadResult
    profile: DatasetProfile
    cleaning_suggestions: list[CleaningSuggestion]
    cleaned_df: object
    cleaning_log: list[str]
    charts: dict[str, list[ChartSpec]]
    stats_blob: dict
    narrative: NarrativeResult | None


def build_stats_blob(load, profile, charts) -> dict:
    worst = max(profile.columns.values(), key=lambda c: c.missing_pct) if profile.columns else None
    return {
        "n_rows": load.row_count_full, "n_cols": profile.n_cols,
        "duplicate_rows": profile.duplicate_rows, "is_sampled": load.is_sampled,
        "data_quality": {"worst_col": worst.name if worst else None,
                          "worst_pct": worst.missing_pct if worst else 0},
        "column_types": {name: p.dtype.value for name, p in profile.columns.items()},
        "charts": {section: [{"title": c.title, "facts": c.facts} for c in specs]
                   for section, specs in charts.items()},
    }


def run_pipeline(filename: str, raw_bytes: bytes, sheet_name: str | None = None,
                  accepted_cleaning_ids: set[str] | None = None, api_key: str | None = None,
                  skip_narrative: bool = False, skip_preparse_check: bool = False) -> PipelineResult:
    if not skip_preparse_check:
        try:
            safe_preparse(filename, raw_bytes)
        except Exception as e:
            raise ValueError(f"File failed safety pre-check: {e}")

    load = load_file(filename, raw_bytes, sheet_name=sheet_name)
    try:
        profile = profile_dataset(load.df)
        if load.is_sampled:
            apply_exact_stats(profile, load)

        suggestions = generate_all_suggestions(load.df, profile)
        cleaned_df, cleaning_log = load.df, []
        if accepted_cleaning_ids:
            cleaned_df, cleaning_log = apply_cleaning(load.df, suggestions, accepted_cleaning_ids)
            profile = profile_dataset(cleaned_df)
            if load.is_sampled:
                profile.warnings.append("Post-cleaning statistics are computed on the in-memory "
                                         "sample, not re-verified against the full dataset.")

        charts = build_all_charts(cleaned_df, profile)
        stats_blob = build_stats_blob(load, profile, charts)

        narrative = None
        if not skip_narrative and api_key:
            narrative = generate_narrative(stats_blob, api_key)

        return PipelineResult(load=load, profile=profile, cleaning_suggestions=suggestions,
                               cleaned_df=cleaned_df, cleaning_log=cleaning_log, charts=charts,
                               stats_blob=stats_blob, narrative=narrative)
    finally:
        cleanup_load_result(load)


def export_html(result: PipelineResult, filename: str) -> str:
    return render_html(assemble_report_context(result, filename))


def export_pdf(result: PipelineResult, filename: str) -> bytes:
    return html_to_pdf_weasyprint(render_pdf_html(assemble_pdf_context(result, filename)))


def export_excel(result: PipelineResult) -> bytes:
    import pandas as pd
    profile_df = pd.DataFrame([{"column": p.name, "type": p.dtype.value,
                                 "missing_pct": p.missing_pct, "unique": p.unique_count}
                                for p in result.profile.columns.values()])
    quality_df = pd.DataFrame([{"column": p.name, "flags": ", ".join(p.flags)}
                                for p in result.profile.columns.values() if p.flags])
    return build_excel_export(result.cleaned_df, profile_df, quality_df)
