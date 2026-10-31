from __future__ import annotations
import base64
import io
from datetime import datetime
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape


def render_html(context: dict) -> str:
    env = Environment(loader=FileSystemLoader("templates"), autoescape=select_autoescape(["html"]))
    return env.get_template("report.html.j2").render(**context, generated_at=datetime.utcnow().isoformat())


def fig_to_static_img(fig, width: int = 900, height: int = 500, scale: int = 2) -> str:
    """Renders a Plotly figure to a base64 PNG via kaleido -- no browser or network required."""
    png_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
    return base64.b64encode(png_bytes).decode("ascii")


def render_pdf_html(context: dict) -> str:
    env = Environment(loader=FileSystemLoader("templates"), autoescape=select_autoescape(["html"]))
    return env.get_template("report_pdf.html.j2").render(**context, generated_at=datetime.utcnow().isoformat())


def html_to_pdf_weasyprint(pdf_html: str) -> bytes:
    """Primary PDF path: static HTML+images, no JS engine, no outbound network calls."""
    from weasyprint import HTML
    return HTML(string=pdf_html).write_pdf()


def html_to_pdf_playwright(html: str) -> bytes:
    """Optional fallback path, requires a Chromium install (see requirements-dev.txt)."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="networkidle")
        pdf = page.pdf(format="A4", print_background=True)
        browser.close()
    return pdf


def build_excel_export(df: pd.DataFrame, profile_summary: pd.DataFrame, quality_summary: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="cleaned_data", index=False)
        profile_summary.to_excel(w, sheet_name="column_profile", index=False)
        quality_summary.to_excel(w, sheet_name="data_quality", index=False)
    buf.seek(0)
    return buf.read()
