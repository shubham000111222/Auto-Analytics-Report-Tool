import enum
import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, JSON, Enum as SAEnum
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ReportStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class Report(Base):
    __tablename__ = "reports"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    status = Column(SAEnum(ReportStatus), default=ReportStatus.PENDING)
    storage_path = Column(String)
    result_html_path = Column(String)
    result_pdf_path = Column(String)
    result_data_path = Column(String)
    error_message = Column(String)
    stats_json = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
