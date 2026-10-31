import os
from datetime import datetime, timedelta
from .db import SessionLocal
from .models import Report
from .storage import RETENTION_HOURS


def cleanup():
    cutoff = datetime.utcnow() - timedelta(hours=RETENTION_HOURS)
    db = SessionLocal()
    for r in db.query(Report).filter(Report.created_at < cutoff).all():
        for p in (r.storage_path, r.result_html_path, r.result_pdf_path, r.result_data_path):
            if p and os.path.exists(p):
                os.remove(p)
        db.delete(r)
    db.commit()
    db.close()


if __name__ == "__main__":
    cleanup()
