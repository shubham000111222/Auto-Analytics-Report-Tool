import os
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .db import get_db
from .models import Report, ReportStatus
from .schemas import ReportOut
from .storage import save_upload
from .pipeline import process_report
from . import routes_advanced
from engine.loader import validate_file
from engine.security import full_security_check
from engine.preparse_check import safe_preparse
from engine.config import settings

app = FastAPI(title="Auto Analytics Report API")
app.include_router(routes_advanced.router)


@app.post("/reports/upload", response_model=ReportOut)
async def upload_report(background_tasks: BackgroundTasks, file: UploadFile = File(...),
                          db: Session = Depends(get_db)):
    raw = await file.read()
    errors = validate_file(file.filename, len(raw)) + full_security_check(
        file.filename, raw, settings.allowed_extensions)
    if errors:
        raise HTTPException(400, errors)
    try:
        safe_preparse(file.filename, raw)
    except Exception as e:
        raise HTTPException(400, f"File failed safety pre-check: {e}")

    path = save_upload(file.filename, raw)
    report = Report(filename=file.filename, storage_path=path, status=ReportStatus.PENDING)
    db.add(report)
    db.commit()
    db.refresh(report)

    background_tasks.add_task(process_report, report.id, path, file.filename,
                               os.environ.get("ANTHROPIC_API_KEY"))
    return report


@app.get("/reports/{report_id}", response_model=ReportOut)
def get_report(report_id: str, db: Session = Depends(get_db)):
    report = db.query(Report).get(report_id)
    if not report:
        raise HTTPException(404, "Report not found")
    return report


@app.get("/reports/{report_id}/html")
def get_html(report_id: str, db: Session = Depends(get_db)):
    report = db.query(Report).get(report_id)
    if not report or report.status != ReportStatus.DONE:
        raise HTTPException(404, "Report not ready")
    return FileResponse(report.result_html_path, media_type="text/html")


@app.get("/reports/{report_id}/pdf")
def get_pdf(report_id: str, db: Session = Depends(get_db)):
    report = db.query(Report).get(report_id)
    if not report or not report.result_pdf_path:
        raise HTTPException(404, "PDF not available")
    return FileResponse(report.result_pdf_path, media_type="application/pdf")


@app.delete("/reports/{report_id}")
def delete_report(report_id: str, db: Session = Depends(get_db)):
    report = db.query(Report).get(report_id)
    if not report:
        raise HTTPException(404, "Report not found")
    for p in (report.storage_path, report.result_html_path, report.result_pdf_path, report.result_data_path):
        if p and os.path.exists(p):
            os.remove(p)
    db.delete(report)
    db.commit()
    return {"deleted": report_id}
