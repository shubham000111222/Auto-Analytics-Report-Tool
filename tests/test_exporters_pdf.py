import pandas as pd
from engine.pipeline import run_pipeline, export_pdf, export_html


def test_pdf_and_html_export_smoke():
    df = pd.DataFrame({"x": range(50), "cat": ["a", "b"] * 25})
    raw = df.to_csv(index=False).encode()
    result = run_pipeline("sample.csv", raw, skip_narrative=True, skip_preparse_check=True)

    html = export_html(result, "sample.csv")
    assert "<html" in html.lower()

    pdf_bytes = export_pdf(result, "sample.csv")
    assert pdf_bytes[:4] == b"%PDF"
