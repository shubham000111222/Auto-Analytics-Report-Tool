from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class ReportOut(BaseModel):
    id: str
    filename: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True
