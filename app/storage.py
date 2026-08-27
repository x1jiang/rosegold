"""Persistent annotation and export store.

On Cloud Run, ROSEGOLD_OUTPUT_DIR is the GCS volume mount
(/mnt/gcs/outputs). Locally it defaults to ./outputs.
Paths are resolved at call time so deploy env vars take effect.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def output_dir() -> str:
    return os.path.abspath(os.getenv("ROSEGOLD_OUTPUT_DIR", os.path.join(_ROOT, "outputs")))


def audit_log_path() -> str:
    return os.path.abspath(os.getenv("ROSEGOLD_AUDIT_LOG", os.path.join(output_dir(), "human_audit_log.jsonl")))


def batch_csv_path() -> str:
    return os.path.join(output_dir(), "rose_gold_adjudications.csv")


def batch_jsonl_path() -> str:
    return os.path.join(output_dir(), "rose_gold_adjudications.jsonl")


def batch_parquet_path() -> str:
    return os.path.join(output_dir(), "rose_gold_adjudications.parquet")


def omop_obs_path() -> str:
    return os.path.join(output_dir(), "omop_observation_rosegold.csv")


def criteria_path() -> str:
    return os.path.join(output_dir(), "custom_criteria.txt")


def ensure_output_dir() -> str:
    dest = output_dir()
    os.makedirs(dest, exist_ok=True)
    parent = os.path.dirname(audit_log_path())
    if parent:
        os.makedirs(parent, exist_ok=True)
    return dest


def is_durable() -> bool:
    bucket = os.getenv("ROSEGOLD_GCS_BUCKET", "").strip()
    if bucket:
        return True
    dest = output_dir()
    if dest.startswith("/mnt/gcs"):
        return True
    return os.path.ismount("/mnt/gcs")


def storage_status() -> Dict[str, Any]:
    ensure_output_dir()
    return {
        "output_dir": output_dir(),
        "audit_log": audit_log_path(),
        "durable": is_durable(),
        "gcs_bucket": os.getenv("ROSEGOLD_GCS_BUCKET") or None,
        "audit_exists": os.path.exists(audit_log_path()),
        "batch_exists": os.path.exists(batch_jsonl_path()),
        "writable": os.access(output_dir(), os.W_OK),
    }


def _fsync_path(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _flush_write(path: str, text: str, mode: str = "a") -> None:
    ensure_output_dir()
    with open(path, mode, encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    _fsync_path(path)


def normalize_audit(entry: Dict[str, Any]) -> Dict[str, Any]:
    status = entry.get("llm_status") or entry.get("adjudication_status") or ""
    decision = entry.get("human_decision") or ""
    if entry.get("human_positive") is None:
        if "Override" in decision:
            human_positive = "Positive" in decision
        else:
            human_positive = "POSITIVE" in str(status)
    else:
        human_positive = bool(entry.get("human_positive"))
    if entry.get("llm_positive") is None:
        llm_positive = "POSITIVE" in str(status)
    else:
        llm_positive = bool(entry.get("llm_positive"))
    return {
        "visit_occurrence_id": entry.get("visit_occurrence_id"),
        "person_id": entry.get("person_id"),
        "reviewer_id": entry.get("reviewer_id") or "physician_01",
        "adjudication_status": entry.get("adjudication_status") or status,
        "llm_status": status,
        "llm_confidence": entry.get("llm_confidence"),
        "human_decision": decision or None,
        "human_positive": human_positive,
        "llm_positive": llm_positive,
        "reviewer_agreement": bool(entry.get("reviewer_agreement")),
        "override_reason": entry.get("override_reason"),
        "comments": entry.get("comments"),
        "timestamp": entry.get("timestamp"),
    }


def append_audit(entry: Dict[str, Any]) -> str:
    import datetime

    payload = normalize_audit(entry)
    if not payload.get("timestamp"):
        payload["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path = audit_log_path()
    _flush_write(path, json.dumps(payload) + "\n", mode="a")
    return path


def read_audit() -> List[Dict[str, Any]]:
    path = audit_log_path()
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_batch_results(results: List[Dict[str, Any]], target_condition: str) -> Dict[str, str]:
    import pandas as pd
    from app.omop_export import export_to_omop_observation

    ensure_output_dir()
    paths = {
        "csv": batch_csv_path(),
        "jsonl": batch_jsonl_path(),
        "parquet": batch_parquet_path(),
        "omop": omop_obs_path(),
    }
    df = pd.DataFrame(results)
    df.to_csv(paths["csv"], index=False)
    _fsync_path(paths["csv"])
    try:
        df.to_parquet(paths["parquet"], index=False)
        _fsync_path(paths["parquet"])
    except Exception:
        pass
    _flush_write(paths["jsonl"], "".join(json.dumps(item) + "\n" for item in results), mode="w")
    obs = export_to_omop_observation(results, target_condition)
    obs.to_csv(paths["omop"], index=False)
    _fsync_path(paths["omop"])
    return paths


def load_batch_results() -> List[Dict[str, Any]]:
    path = batch_jsonl_path()
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_criteria(text: str) -> str:
    path = criteria_path()
    _flush_write(path, text or "", mode="w")
    return path


def load_criteria() -> Optional[str]:
    path = criteria_path()
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read().strip()
    return text or None


def default_batch_csv() -> str:
    ensure_output_dir()
    return batch_csv_path()
