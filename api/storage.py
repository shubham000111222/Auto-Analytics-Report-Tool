import os
import uuid

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./uploads")
RESULT_DIR = os.environ.get("RESULT_DIR", "./results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
RETENTION_HOURS = int(os.environ.get("RETENTION_HOURS", "24"))


def save_upload(filename: str, raw: bytes) -> str:
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{os.path.basename(filename)}")
    with open(path, "wb") as f:
        f.write(raw)
    return path
