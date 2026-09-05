import os
import datetime
import threading
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from contextlib import asynccontextmanager

from app.schemas import RoseGoldAdjudication
from app.engine import AdjudicationEngine
from app.omop_loader import load_omop_data, load_visit_index
from app.mimic_ext_notes import looks_like_mimic_ext_notes_path
from app.config_loader import resolve_criteria
from app.storage import (
    append_audit,
    batch_csv_path,
    batch_jsonl_path,
    criteria_path,
    load_batch_results,
    load_criteria,
    omop_obs_path,
    read_audit,
    save_batch_results,
    save_criteria,
    storage_status,
)

DATA_DIR = os.path.abspath(os.getenv("ROSEGOLD_DATA_DIR", "data"))
NOTES_PATH = os.path.abspath(os.getenv("ROSEGOLD_NOTES_PATH", os.path.join(DATA_DIR, "synthetic_notes.csv")))
VISITS_PATH = os.path.abspath(os.getenv("ROSEGOLD_VISITS_PATH", os.path.join(DATA_DIR, "synthetic_visits.csv")))

engine = AdjudicationEngine(
    model_name=os.getenv("ROSEGOLD_MODEL_NAME", "auto")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=lambda: engine.backend_status(init=True), daemon=True).start()
    yield


app = FastAPI(
    title="Rose Gold Clinical Adjudication API",
    description="High-Throughput On-Premises OMOP Note Adjudication & Phenotyping Service",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ROSEGOLD_CORS_ORIGINS", "*").split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_data_path(user_path: Optional[str], default_path: str) -> str:
    if not user_path:
        return default_path
    candidate = os.path.abspath(user_path)
    if candidate.startswith(DATA_DIR) and os.path.exists(candidate):
        return candidate
    return default_path


class SingleAdjudicationRequest(BaseModel):
    visit_occurrence_id: Optional[int] = None
    person_id: Optional[int] = 1001
    notes_formatted_text: Optional[str] = None
    target_condition: str = "Sepsis / Septic Shock"
    clinical_criteria: Optional[str] = None


class BatchAdjudicationRequest(BaseModel):
    visit_occurrence_ids: Optional[List[int]] = None
    target_condition: str = "Sepsis / Septic Shock"
    clinical_criteria: Optional[str] = None
    notes_path: Optional[str] = None
    visits_path: Optional[str] = None


class PhysicianFeedbackRequest(BaseModel):
    visit_occurrence_id: int
    person_id: int
    reviewer_id: str = "physician_01"
    adjudication_status: str
    reviewer_agreement: bool
    override_reason: Optional[str] = None
    comments: Optional[str] = None
    llm_status: Optional[str] = None
    llm_confidence: Optional[float] = None
    human_decision: Optional[str] = None
    human_positive: Optional[bool] = None
    llm_positive: Optional[bool] = None


class CriteriaRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "service": "Rose Gold Clinical Adjudication",
        "docs": "/docs",
        "health": "/health",
        "ui_hint": "Streamlit dashboard on port 8501 when start_services.sh / Cloud Run sidecar is used.",
    }


@app.get("/health")
def health_check():
    try:
        backend = engine.backend_status(init=False)
        ready = True
        status = "healthy"
        error = backend.get("error")
    except Exception as exc:
        backend = {"backend": "unavailable", "model_name": None, "llm_real": False}
        ready = False
        status = "degraded"
        error = str(exc)
    return {
        "status": status,
        "engine_ready": ready,
        "vllm_active": engine.is_vllm_available,
        "backend": backend.get("backend"),
        "llm_real": bool(backend.get("llm_real")),
        "model_name": backend.get("model_name") or engine.model_name,
        "device": engine.hardware_info.get("device"),
        "notes_file": NOTES_PATH,
        "visits_file": VISITS_PATH,
        "storage": storage_status(),
        "error": error,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


@app.get("/api/visits")
def get_visits():
    """Returns list of visits available in the configured OMOP dataset."""
    if not os.path.exists(NOTES_PATH):
        raise HTTPException(status_code=404, detail="OMOP data files not found.")
    if not looks_like_mimic_ext_notes_path(NOTES_PATH) and not os.path.exists(VISITS_PATH):
        raise HTTPException(status_code=404, detail="OMOP data files not found.")
    return load_visit_index(NOTES_PATH, VISITS_PATH)


@app.get("/api/notes/{visit_occurrence_id}")
def get_visit_notes(visit_occurrence_id: int):
    """Returns the full chronological note trajectory for a specific visit encounter."""
    records = load_omop_data(NOTES_PATH, VISITS_PATH, target_visits=[visit_occurrence_id])
    if not records:
        raise HTTPException(status_code=404, detail=f"Visit {visit_occurrence_id} not found.")
    return records[0]


@app.post("/api/adjudicate/single", response_model=RoseGoldAdjudication)
def adjudicate_single_visit(req: SingleAdjudicationRequest):
    """Adjudicates a single visit encounter either by ID or raw notes text."""
    if req.notes_formatted_text:
        record = {
            "person_id": req.person_id or 1,
            "visit_occurrence_id": req.visit_occurrence_id or 99999,
            "visit_start_date": "Unknown",
            "visit_end_date": "Unknown",
            "notes_formatted_text": req.notes_formatted_text,
        }
    elif req.visit_occurrence_id is not None:
        records = load_omop_data(NOTES_PATH, VISITS_PATH, target_visits=[req.visit_occurrence_id])
        if not records:
            raise HTTPException(status_code=404, detail=f"Visit {req.visit_occurrence_id} not found.")
        record = records[0]
    else:
        raise HTTPException(status_code=400, detail="Must provide either visit_occurrence_id or notes_formatted_text.")

    criteria = resolve_criteria(req.target_condition, req.clinical_criteria)
    try:
        return engine.adjudicate_single(record, req.target_condition, criteria)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Adjudication error: {exc}")


@app.post("/api/adjudicate/batch", response_model=List[RoseGoldAdjudication])
def adjudicate_batch_visits(req: BatchAdjudicationRequest):
    """Executes high-throughput batch adjudication over multiple visits."""
    n_path = _safe_data_path(req.notes_path, NOTES_PATH)
    v_path = _safe_data_path(req.visits_path, VISITS_PATH)
    records = load_omop_data(n_path, v_path, target_visits=req.visit_occurrence_ids)
    if not records:
        raise HTTPException(status_code=400, detail="No matching visit records found.")

    criteria = resolve_criteria(req.target_condition, req.clinical_criteria)
    try:
        results = engine.adjudicate_batch(records, req.target_condition, criteria)
        save_batch_results(results, req.target_condition)
        return results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch adjudication error: {exc}")


@app.get("/api/audit")
def get_audit_log():
    return {"durable": storage_status()["durable"], "entries": read_audit()}


@app.get("/api/batch")
def get_batch_results():
    status = storage_status()
    return {
        "durable": status["durable"],
        "results": load_batch_results(),
        "paths": {
            "csv": batch_csv_path(),
            "jsonl": batch_jsonl_path(),
            "omop": omop_obs_path(),
        },
    }


@app.get("/api/criteria")
def get_saved_criteria():
    return {
        "text": load_criteria(),
        "path": criteria_path(),
        "durable": storage_status()["durable"],
    }


@app.post("/api/criteria")
def record_criteria(req: CriteriaRequest):
    path = save_criteria(req.text)
    return {
        "status": "success",
        "path": path,
        "durable": storage_status()["durable"],
    }


@app.post("/api/feedback")
def record_physician_feedback(feedback: PhysicianFeedbackRequest):
    """Appends physician review agreement/override to the durable audit log."""
    path = append_audit(feedback.model_dump())
    status = storage_status()
    return {
        "status": "success",
        "message": "Feedback recorded in audit log.",
        "path": path,
        "durable": status["durable"],
    }
