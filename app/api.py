"""Rose Gold FastAPI service.

Hardening notes
---------------
* Request bodies are capped (``ROSEGOLD_MAX_BODY_BYTES``) before the JSON parser runs.
* Every free-text field carries an explicit length bound so a single request cannot
  blow up prompt size or the on-disk audit log.
* User-supplied data paths must resolve (symlinks included) to a regular file inside
  ``ROSEGOLD_DATA_DIR``; anything else is rejected with 400 instead of silently
  substituting the default dataset.
* Unexpected exceptions are logged with a correlation id and returned as a generic
  500 so internal paths and stack traces never reach clients.
* Optional shared-secret auth: set ``ROSEGOLD_API_KEY`` and every ``/api/*`` route
  requires ``X-API-Key`` (or ``Authorization: Bearer``).
* CORS is closed by default; opt in with ``ROSEGOLD_CORS_ORIGINS``.
"""

from __future__ import annotations

import datetime
import hmac
import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.config_loader import resolve_criteria
from app.engine import AdjudicationEngine
from app.mimic_ext_notes import MimicExtNotesError, looks_like_mimic_ext_notes_path
from app.omop_loader import load_omop_data, load_visit_index
from app.schemas import RoseGoldAdjudication
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

logger = logging.getLogger("rosegold.api")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = os.path.realpath(os.getenv("ROSEGOLD_DATA_DIR", "data"))
NOTES_PATH = os.path.abspath(os.getenv("ROSEGOLD_NOTES_PATH", os.path.join(DATA_DIR, "synthetic_notes.csv")))
VISITS_PATH = os.path.abspath(os.getenv("ROSEGOLD_VISITS_PATH", os.path.join(DATA_DIR, "synthetic_visits.csv")))


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


MAX_BODY_BYTES = _int_env("ROSEGOLD_MAX_BODY_BYTES", 8 * 1024 * 1024)
MAX_NOTES_CHARS = _int_env("ROSEGOLD_MAX_NOTES_CHARS", 400_000)
MAX_CRITERIA_CHARS = _int_env("ROSEGOLD_MAX_CRITERIA_CHARS", 20_000)
MAX_BATCH_VISITS = _int_env("ROSEGOLD_MAX_BATCH_VISITS", 500)
MAX_CONDITION_CHARS = 200
MAX_SHORT_TEXT = 256
MAX_COMMENT_CHARS = 4_000
MAX_PATH_CHARS = 1_024


def _api_key() -> str:
    """Read at call time so tests and rotated secrets take effect without restart."""
    return os.getenv("ROSEGOLD_API_KEY", "").strip()


def require_api_key(request: Request) -> None:
    expected = _api_key()
    if not expected:
        return
    supplied = request.headers.get("x-api-key", "")
    if not supplied:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
    if not supplied or not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


engine = AdjudicationEngine(model_name=os.getenv("ROSEGOLD_MODEL_NAME", "auto"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=lambda: engine.backend_status(init=True), daemon=True, name="rosegold-backend-init").start()
    yield


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Pure ASGI middleware: reject bodies above ``max_bytes`` with 413.

    Checks ``Content-Length`` up front and also counts streamed chunks so a
    chunked upload without a length header cannot bypass the limit.
    """

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = int(max_bytes)

    async def _reject(self, send) -> None:
        body = b'{"detail":"Request body too large."}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers") or []:
            if name == b"content-length":
                try:
                    if int(value) > self.max_bytes:
                        await self._reject(send)
                        return
                except ValueError:
                    pass
                break

        received = 0
        response_started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b"") or b"")
                if received > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not response_started:
                await self._reject(send)


class SecurityHeadersMiddleware:
    """Defensive response headers. ``/docs`` keeps framing/caching defaults so Swagger works."""

    _STATIC = (
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"referrer-policy", b"no-referrer"),
        (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
    )

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        no_store = path.startswith("/api/") or path == "/health"

        async def send_with_headers(message):
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers") or [])
                present = {k.lower() for k, _ in headers}
                for key, value in self._STATIC:
                    if key not in present:
                        headers.append((key, value))
                if no_store and b"cache-control" not in present:
                    headers.append((b"cache-control", b"no-store"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


app = FastAPI(
    title="Rose Gold Clinical Adjudication API",
    description="High-Throughput On-Premises OMOP Note Adjudication & Phenotyping Service",
    version="1.2.0",
    lifespan=lifespan,
)

_cors_origins = [o.strip() for o in os.getenv("ROSEGOLD_CORS_ORIGINS", "").split(",") if o.strip()]
if _cors_origins:
    # Browsers refuse credentials with a wildcard origin; never advertise that combination.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials="*" not in _cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    )
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_BODY_BYTES)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    error_id = secrets.token_hex(6)
    logger.exception("Unhandled error %s on %s %s", error_id, request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error.", "error_id": error_id})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_data_path(user_path: Optional[str], default_path: str) -> str:
    """Resolve a caller-supplied dataset path or reject it.

    The resolved (symlink-free) path must be a regular file strictly inside
    ``DATA_DIR``. ``DATA_DIR + "_evil"`` style prefix collisions and ``..``
    traversal are both rejected.
    """
    if not user_path:
        return default_path
    if len(user_path) > MAX_PATH_CHARS or "\x00" in user_path:
        raise HTTPException(status_code=400, detail="Invalid data path.")
    candidate = os.path.realpath(user_path)
    try:
        inside = os.path.commonpath([DATA_DIR, candidate]) == DATA_DIR
    except ValueError:
        inside = False
    if not inside or candidate == DATA_DIR or not os.path.isfile(candidate):
        raise HTTPException(status_code=400, detail="Data path must be a file inside the configured data directory.")
    return candidate


def _load_records(notes_path: str, visits_path: str, target_visits: Optional[List[int]]) -> List[Dict[str, Any]]:
    try:
        return load_omop_data(notes_path, visits_path, target_visits=target_visits)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="OMOP data files not found.")
    except (MimicExtNotesError, ValueError, KeyError) as exc:
        logger.warning("Rejected dataset %s: %s", notes_path, exc)
        raise HTTPException(status_code=400, detail="Dataset is malformed or missing required columns.")


def _run_engine(fn, *args):
    """Run an engine call, mapping backend-unavailable to 503 and everything else to a logged 500."""
    try:
        return fn(*args)
    except RuntimeError as exc:
        # Engine raises RuntimeError only for "no real backend" conditions; message is operational, not sensitive.
        raise HTTPException(status_code=503, detail=f"Adjudication backend unavailable: {str(exc)[:300]}")
    except HTTPException:
        raise
    except Exception:
        error_id = secrets.token_hex(6)
        logger.exception("Adjudication failed (error %s)", error_id)
        raise HTTPException(status_code=500, detail=f"Adjudication error (id {error_id}).")


def _strip_nonempty(value: str, field: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        raise ValueError(f"{field} must not be empty")
    return cleaned


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SingleAdjudicationRequest(BaseModel):
    visit_occurrence_id: Optional[int] = None
    person_id: Optional[int] = 1001
    notes_formatted_text: Optional[str] = Field(None, max_length=MAX_NOTES_CHARS)
    target_condition: str = Field("Sepsis / Septic Shock", min_length=1, max_length=MAX_CONDITION_CHARS)
    clinical_criteria: Optional[str] = Field(None, max_length=MAX_CRITERIA_CHARS)

    @field_validator("target_condition")
    @classmethod
    def _condition(cls, value: str) -> str:
        return _strip_nonempty(value, "target_condition")


class BatchAdjudicationRequest(BaseModel):
    visit_occurrence_ids: Optional[List[int]] = Field(None, max_length=MAX_BATCH_VISITS)
    target_condition: str = Field("Sepsis / Septic Shock", min_length=1, max_length=MAX_CONDITION_CHARS)
    clinical_criteria: Optional[str] = Field(None, max_length=MAX_CRITERIA_CHARS)
    notes_path: Optional[str] = Field(None, max_length=MAX_PATH_CHARS)
    visits_path: Optional[str] = Field(None, max_length=MAX_PATH_CHARS)

    @field_validator("target_condition")
    @classmethod
    def _condition(cls, value: str) -> str:
        return _strip_nonempty(value, "target_condition")


class PhysicianFeedbackRequest(BaseModel):
    visit_occurrence_id: int
    person_id: int
    reviewer_id: str = Field("physician_01", min_length=1, max_length=MAX_SHORT_TEXT)
    adjudication_status: str = Field(..., min_length=1, max_length=MAX_SHORT_TEXT)
    reviewer_agreement: bool
    override_reason: Optional[str] = Field(None, max_length=MAX_COMMENT_CHARS)
    comments: Optional[str] = Field(None, max_length=MAX_COMMENT_CHARS)
    llm_status: Optional[str] = Field(None, max_length=MAX_SHORT_TEXT)
    llm_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    human_decision: Optional[str] = Field(None, max_length=MAX_SHORT_TEXT)
    human_positive: Optional[bool] = None
    llm_positive: Optional[bool] = None


class CriteriaRequest(BaseModel):
    text: str = Field(..., max_length=MAX_CRITERIA_CHARS)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


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
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Health probe failed")
        backend = {"backend": "unavailable", "model_name": None, "llm_real": False}
        ready = False
        status = "degraded"
        error = type(exc).__name__
    return {
        "status": status,
        "engine_ready": ready,
        "backend_initializing": engine.is_initializing,
        "vllm_active": engine.is_vllm_available,
        "backend": backend.get("backend"),
        "llm_real": bool(backend.get("llm_real")),
        "model_name": backend.get("model_name") or engine.model_name,
        "device": engine.hardware_info.get("device"),
        "notes_file": NOTES_PATH,
        "visits_file": VISITS_PATH,
        "storage": storage_status(),
        "auth_required": bool(_api_key()),
        "error": error,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


api = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])


@api.get("/visits")
def get_visits():
    """Returns list of visits available in the configured OMOP dataset."""
    if not os.path.isfile(NOTES_PATH):
        raise HTTPException(status_code=404, detail="OMOP data files not found.")
    try:
        is_mimic = looks_like_mimic_ext_notes_path(NOTES_PATH)
    except Exception:
        raise HTTPException(status_code=400, detail="Dataset is malformed or missing required columns.")
    if not is_mimic and not os.path.isfile(VISITS_PATH):
        raise HTTPException(status_code=404, detail="OMOP data files not found.")
    try:
        return load_visit_index(NOTES_PATH, VISITS_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="OMOP data files not found.")
    except (MimicExtNotesError, ValueError, KeyError):
        raise HTTPException(status_code=400, detail="Dataset is malformed or missing required columns.")


@api.get("/notes/{visit_occurrence_id}")
def get_visit_notes(visit_occurrence_id: int):
    """Returns the full chronological note trajectory for a specific visit encounter."""
    records = _load_records(NOTES_PATH, VISITS_PATH, [visit_occurrence_id])
    if not records:
        raise HTTPException(status_code=404, detail=f"Visit {visit_occurrence_id} not found.")
    return records[0]


@api.post("/adjudicate/single", response_model=RoseGoldAdjudication)
def adjudicate_single_visit(req: SingleAdjudicationRequest):
    """Adjudicates a single visit encounter either by ID or raw notes text."""
    if req.notes_formatted_text and req.notes_formatted_text.strip():
        record = {
            "person_id": req.person_id or 1,
            "visit_occurrence_id": req.visit_occurrence_id or 99999,
            "visit_start_date": "Unknown",
            "visit_end_date": "Unknown",
            "notes_formatted_text": req.notes_formatted_text,
        }
    elif req.visit_occurrence_id is not None:
        records = _load_records(NOTES_PATH, VISITS_PATH, [req.visit_occurrence_id])
        if not records:
            raise HTTPException(status_code=404, detail=f"Visit {req.visit_occurrence_id} not found.")
        record = records[0]
    else:
        raise HTTPException(status_code=400, detail="Must provide either visit_occurrence_id or notes_formatted_text.")

    criteria = resolve_criteria(req.target_condition, req.clinical_criteria)
    return _run_engine(engine.adjudicate_single, record, req.target_condition, criteria)


@api.post("/adjudicate/batch", response_model=List[RoseGoldAdjudication])
def adjudicate_batch_visits(req: BatchAdjudicationRequest):
    """Executes high-throughput batch adjudication over multiple visits."""
    n_path = _safe_data_path(req.notes_path, NOTES_PATH)
    v_path = _safe_data_path(req.visits_path, VISITS_PATH)
    records = _load_records(n_path, v_path, req.visit_occurrence_ids)
    if not records:
        raise HTTPException(status_code=400, detail="No matching visit records found.")
    if len(records) > MAX_BATCH_VISITS:
        raise HTTPException(
            status_code=400,
            detail=f"Batch too large ({len(records)} visits); pass visit_occurrence_ids with at most {MAX_BATCH_VISITS}.",
        )

    criteria = resolve_criteria(req.target_condition, req.clinical_criteria)

    def _run():
        results = engine.adjudicate_batch(records, req.target_condition, criteria)
        save_batch_results(results, req.target_condition)
        return results

    return _run_engine(_run)


@api.get("/audit")
def get_audit_log():
    return {"durable": storage_status()["durable"], "entries": read_audit()}


@api.get("/batch")
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


@api.get("/criteria")
def get_saved_criteria():
    return {
        "text": load_criteria(),
        "path": criteria_path(),
        "durable": storage_status()["durable"],
    }


@api.post("/criteria")
def record_criteria(req: CriteriaRequest):
    path = save_criteria(req.text)
    return {
        "status": "success",
        "path": path,
        "durable": storage_status()["durable"],
    }


@api.post("/feedback")
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


app.include_router(api)
