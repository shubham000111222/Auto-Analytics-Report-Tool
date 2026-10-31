from __future__ import annotations
import io
import zipfile
from pathlib import Path

MAX_DECOMPRESSED_RATIO = 100
MAX_DECOMPRESSED_BYTES = 1 * 1024 * 1024 * 1024


def is_extension_allowed(filename: str, allowed: tuple[str, ...]) -> bool:
    return Path(filename).suffix.lower() in allowed


def reject_macro_enabled_workbook(filename: str) -> list[str]:
    errors = []
    if Path(filename).suffix.lower() in (".xlsm", ".xlsb"):
        errors.append("Macro-enabled workbooks are not accepted. Save as .xlsx and re-upload.")
    return errors


def check_zip_bomb(raw_bytes: bytes) -> list[str]:
    """.xlsx is a zip archive -- guard against maliciously inflated contents."""
    errors = []
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
            total_uncompressed = sum(i.file_size for i in z.infolist())
            total_compressed = sum(i.compress_size for i in z.infolist()) or 1
            if total_uncompressed > MAX_DECOMPRESSED_BYTES:
                errors.append("File expands beyond the allowed size limit when decompressed.")
            if total_uncompressed / total_compressed > MAX_DECOMPRESSED_RATIO:
                errors.append("File failed a compression-ratio safety check (possible zip bomb).")
    except zipfile.BadZipFile:
        pass
    return errors


def full_security_check(filename: str, raw_bytes: bytes, allowed: tuple[str, ...]) -> list[str]:
    errors = []
    if not is_extension_allowed(filename, allowed):
        errors.append(f"Extension not allowed: {Path(filename).suffix}")
    errors += reject_macro_enabled_workbook(filename)
    if Path(filename).suffix.lower() == ".xlsx":
        errors += check_zip_bomb(raw_bytes)
    return errors
