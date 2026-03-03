# parquet_store.py
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd

PARQUET_DIR = "data_parquet"
PARQUET_FILE = os.path.join(PARQUET_DIR, "tool_audit.parquet")


def _ensure_dir() -> None:
    os.makedirs(PARQUET_DIR, exist_ok=True)


def append_audit_row(row: Dict[str, Any]) -> str:
    """
    Append a single tool call row to a Parquet file.
    For class demos, we rewrite the parquet file (simple and reliable).
    """
    _ensure_dir()

    row = dict(row)
    row["server_utc_ts"] = datetime.now(timezone.utc).isoformat()

    new_df = pd.DataFrame([row])

    if os.path.exists(PARQUET_FILE):
        old_df = pd.read_parquet(PARQUET_FILE)
        out_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out_df = new_df

    out_df.to_parquet(PARQUET_FILE, index=False)
    return PARQUET_FILE


def export_audit_copy(filename: str = "audit_export.parquet") -> str:
    """
    Export a copy of the audit parquet for submission.
    """
    _ensure_dir()
    out_path = os.path.join(PARQUET_DIR, filename)

    if os.path.exists(PARQUET_FILE):
        df = pd.read_parquet(PARQUET_FILE)
        df.to_parquet(out_path, index=False)
    else:
        pd.DataFrame([]).to_parquet(out_path, index=False)

    return out_path