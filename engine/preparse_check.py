from __future__ import annotations
from .sandbox import run_sandboxed


def _preparse(filename: str, raw_bytes: bytes) -> dict:
    import io
    import pandas as pd
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext in ("csv", "tsv"):
        df = pd.read_csv(io.BytesIO(raw_bytes), nrows=5, sep=None, engine="python")
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(raw_bytes), nrows=5, engine="openpyxl")
    elif ext == "json":
        df = pd.read_json(io.BytesIO(raw_bytes))
        df = df.head(5)
    else:
        raise ValueError(f"Unsupported extension for pre-check: {ext}")
    return {"columns": list(map(str, df.columns)), "preview_rows": int(len(df))}


def safe_preparse(filename: str, raw_bytes: bytes, timeout_seconds: int = 15,
                   cpu_seconds: int = 10, memory_mb: int = 512) -> dict:
    """Fast, resource-capped sanity check run before the real pipeline touches a file."""
    return run_sandboxed(_preparse, args=(filename, raw_bytes),
                          timeout_seconds=timeout_seconds, cpu_seconds=cpu_seconds, memory_mb=memory_mb)
