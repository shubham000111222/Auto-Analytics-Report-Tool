import os
from datetime import datetime
from engine.pipeline import run_pipeline, export_html, export_pdf
from .db import SessionLocal
from .models import Report, ReportStatus
from .storage import RESULT_DIR


def process_report(report_id: str, filepath: str, filename: str, api_key: str | None):
    db = SessionLocal()
    report = db.query(Report).get(report_id)
    try:
        report.status = ReportStatus.PROCESSING
        db.commit()

        with open(filepath, "rb") as f:
            raw = f.read()
        result = run_pipeline(filename, raw, api_key=api_key, skip_preparse_check=True)

        html = export_html(result, filename)
        html_path = os.path.join(RESULT_DIR, f"{report_id}.html")
        with open(html_path, "w") as f:
            f.write(html)

        pdf_path = os.path.join(RESULT_DIR, f"{report_id}.pdf")
        try:
            with open(pdf_path, "wb") as f:
                f.write(export_pdf(result, filename))
        except Exception:
            pdf_path = None

        data_path = os.path.join(RESULT_DIR, f"{report_id}.parquet")
        result.cleaned_df.to_parquet(data_path)

        report.status = ReportStatus.DONE
        report.result_html_path = html_path
        report.result_pdf_path = pdf_path
        report.result_data_path = data_path
        report.stats_json = result.stats_blob
        report.completed_at = datetime.utcnow()
        db.commit()
    except Exception as e:
        report.status = ReportStatus.FAILED
        report.error_message = str(e)
        db.commit()
    finally:
        db.close()
