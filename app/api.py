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
  requires ``X-API-Key`` (or ``Authorization: Bearer``). When a key is configured,
  ``/health`` hides filesystem paths and backend error text from unauthenticated callers.
* CORS is closed by default; opt in with ``ROSEGOLD_CORS_ORIGINS``.
* Every response carries ``X-Request-ID`` (echoed from the caller when well-formed,
  generated otherwise); the same id is used in error logs and 500 bodies.
* Optional per-client rate limit (``ROSEGOLD_RATE_LIMIT_PER_MIN``) on ``/api/*``.
* ``/ready`` returns 503 until the configured LLM backend is actually serving, so an
  orchestrator never routes traffic to an instance that would answer with 503s.
"""

from __future__ import annotations

import datetime
import hmac
import logging
import os
import re
import secrets
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Deque, Dict, List, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.config_loader import resolve_criteria
from app.engine import AdjudicationEngine
from app.logging_setup import configure_logging
from app.mimic_ext_notes import MimicExtNotesError, looks_like_mimic_ext_notes_path
from app.omop_loader import load_omop_data, load_visit_index
from app.paths import UnsafePathError, resolve_data_file
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

configure_logging()
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
# 0 disables the limiter (default). Applies per client IP to /api/* only.
RATE_LIMIT_PER_MIN = _int_env("ROSEGOLD_RATE_LIMIT_PER_MIN", 0, minimum=0)


def _api_key() -> str:
    """Read at call time so tests and rotated secrets take effect without restart."""
    return os.getenv("ROSEGOLD_API_KEY", "").strip()


def _supplied_key(request: Request) -> str:
    supplied = request.headers.get("x-api-key", "")
    if not supplied:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
    return supplied


def _is_authenticated(request: Request) -> bool:
    """True when no key is configured (open deployment) or the caller presented the right one."""
    expected = _api_key()
    if not expected:
        return True
    supplied = _supplied_key(request)
    return bool(supplied) and hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def require_api_key(request: Request) -> None:
    if not _is_authenticated(request):
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


_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _request_id_from_scope(scope) -> str:
    return (scope.get("state") or {}).get("request_id") or ""


class RequestIDMiddleware:
    """Attach a request id to every request/response.

    A well-formed inbound ``X-Request-ID`` (load balancer, Cloud Run, client) is
    reused so logs can be joined across hops; anything else is replaced with a
    fresh random id. The id is stored in ``scope["state"]`` for handlers.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        inbound = ""
        for name, value in scope.get("headers") or []:
            if name == b"x-request-id":
                inbound = value.decode("latin-1", "replace")
                break
        request_id = inbound if _REQUEST_ID_RE.match(inbound) else secrets.token_hex(8)
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_id(message):
            if message.get("type") == "http.response.start":
                headers = [(k, v) for k, v in (message.get("headers") or []) if k.lower() != b"x-request-id"]
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_id)


class SecurityHeadersMiddleware:
    """Defensive response headers. ``/docs`` keeps framing/caching defaults so Swagger works."""

    _STATIC = (
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"referrer-policy", b"no-referrer"),
        (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
    )
    # Pure JSON routes never need to load anything; a locked-down CSP makes a
    # reflected-content bug on them unexploitable in a browser.
    _API_CSP = b"default-src 'none'; frame-ancestors 'none'"

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        api_route = path.startswith("/api/") or path in ("/health", "/ready")

        async def send_with_headers(message):
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers") or [])
                present = {k.lower() for k, _ in headers}
                for key, value in self._STATIC:
                    if key not in present:
                        headers.append((key, value))
                if api_route:
                    if b"cache-control" not in present:
                        headers.append((b"cache-control", b"no-store"))
                    if b"content-security-policy" not in present:
                        headers.append((b"content-security-policy", self._API_CSP))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


class RateLimitMiddleware:
    """Fixed-window per-client limiter for ``/api/*``; 429 with ``Retry-After`` when exceeded.

    In-memory and per-process, which matches the single-worker deployment used
    here. Client identity is the transport peer address; ``X-Forwarded-For`` is
    honoured only when ``ROSEGOLD_TRUST_PROXY=1`` because otherwise any caller
    could spoof their way past the limit.
    """

    def __init__(self, app, per_minute: int, trust_proxy: Optional[bool] = None, max_clients: int = 10_000):
        self.app = app
        self.per_minute = int(per_minute)
        self.trust_proxy = (
            trust_proxy
            if trust_proxy is not None
            else os.getenv("ROSEGOLD_TRUST_PROXY", "").lower() in {"1", "true", "yes"}
        )
        self.max_clients = max_clients
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def _client_key(self, scope) -> str:
        if self.trust_proxy:
            for name, value in scope.get("headers") or []:
                if name == b"x-forwarded-for":
                    first = value.decode("latin-1", "replace").split(",")[0].strip()
                    if first:
                        return first[:64]
                    break
        client = scope.get("client") or ("unknown", 0)
        return str(client[0])

    def _allow(self, key: str, now: float) -> tuple[bool, float]:
        window = 60.0
        with self._lock:
            bucket = self._hits.get(key)
            if bucket is None:
                if len(self._hits) >= self.max_clients:
                    # Evict the stalest client rather than grow without bound.
                    stalest = min(self._hits, key=lambda k: self._hits[k][-1] if self._hits[k] else 0.0)
                    self._hits.pop(stalest, None)
                bucket = deque()
                self._hits[key] = bucket
            while bucket and now - bucket[0] >= window:
                bucket.popleft()
            if len(bucket) >= self.per_minute:
                return False, max(1.0, window - (now - bucket[0]))
            bucket.append(now)
            return True, 0.0

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or self.per_minute <= 0 or not scope.get("path", "").startswith("/api/"):
            await self.app(scope, receive, send)
            return
        allowed, retry_after = self._allow(self._client_key(scope), time.monotonic())
        if allowed:
            await self.app(scope, receive, send)
            return
        body = b'{"detail":"Rate limit exceeded."}'
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"retry-after", str(int(retry_after + 0.999)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


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
app.add_middleware(RateLimitMiddleware, per_minute=RATE_LIMIT_PER_MIN)
# Outermost so every response, including 413/429 short-circuits, carries the id.
app.add_middleware(RequestIDMiddleware)


def _error_id(request: Request) -> str:
    rid = _request_id_from_scope(request.scope)
    return rid or secrets.token_hex(6)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    error_id = _error_id(request)
    logger.exception("Unhandled error %s on %s %s", error_id, request.method, request.url.path)
    # Starlette's ServerErrorMiddleware sends this response outside the user
    # middleware stack, so the request-id header has to be set here explicitly.
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "error_id": error_id},
        headers={"X-Request-ID": error_id, "Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_data_path(user_path: Optional[str], default_path: str) -> str:
    """Resolve a caller-supplied dataset path or reject it with 400.

    Delegates to :func:`app.paths.resolve_data_file`: the resolved (symlink-free)
    path must be a regular ``.csv``/``.parquet`` file strictly inside ``DATA_DIR``.
    """
    if not user_path:
        return default_path
    try:
        return resolve_data_file(user_path, root=DATA_DIR)
    except UnsafePathError:
        raise HTTPException(status_code=400, detail="Data path must be a file inside the configured data directory.")


def _load_records(notes_path: str, visits_path: str, target_visits: Optional[List[int]]) -> List[Dict[str, Any]]:
    try:
        return load_omop_data(notes_path, visits_path, target_visits=target_visits)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="OMOP data files not found.")
    except (MimicExtNotesError, ValueError, KeyError) as exc:
        logger.warning("Rejected dataset %s: %s", notes_path, exc)
        raise HTTPException(status_code=400, detail="Dataset is malformed or missing required columns.")


def _run_engine(request: Request, fn, *args):
    """Run an engine call, mapping backend-unavailable to 503 and everything else to a logged 500.

    Backend error text can contain URLs, filesystem paths or library internals,
    so it is logged with the request id and never echoed to the client.
    """
    error_id = _error_id(request)
    try:
        return fn(*args)
    except RuntimeError as exc:
        logger.error("Adjudication backend unavailable (request %s): %s", error_id, exc)
        raise HTTPException(
            status_code=503,
            detail="Adjudication backend unavailable. Check /ready or the service logs.",
            headers={"Retry-After": "30"},
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Adjudication failed (request %s)", error_id)
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


def _backend_snapshot() -> Dict[str, Any]:
    try:
        backend = engine.backend_status(init=False)
        backend["probe_ok"] = True
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Health probe failed")
        backend = {"backend": "unavailable", "model_name": None, "llm_real": False, "error": type(exc).__name__, "probe_ok": False}
    return backend


def _backend_serving(backend: Dict[str, Any]) -> bool:
    """True when the instance can answer adjudication requests without a 503.

    Either a real LLM backend is loaded, or none was required (rules/mock mode).
    """
    if not backend.get("probe_ok", True):
        return False
    if engine.is_initializing:
        return False
    if engine.wants_real_backend():
        return bool(backend.get("llm_real"))
    return True


@app.get("/health")
def health_check(request: Request):
    """Liveness plus a status summary.

    Always 200 while the process is up. Filesystem paths, storage layout and raw
    backend error text are only included for authenticated callers (or when no
    API key is configured at all, i.e. a local/dev deployment).
    """
    backend = _backend_snapshot()
    serving = _backend_serving(backend)
    payload: Dict[str, Any] = {
        "status": "healthy" if serving else "degraded",
        "engine_ready": bool(backend.get("probe_ok", True)),
        "ready": serving,
        "backend_initializing": engine.is_initializing,
        "vllm_active": engine.is_vllm_available,
        "backend": backend.get("backend"),
        "llm_real": bool(backend.get("llm_real")),
        "model_name": backend.get("model_name") or engine.model_name,
        "device": engine.hardware_info.get("device"),
        "auth_required": bool(_api_key()),
        "version": app.version,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if _is_authenticated(request):
        payload.update(
            {
                "notes_file": NOTES_PATH,
                "visits_file": VISITS_PATH,
                "storage": storage_status(),
                "error": backend.get("error"),
            }
        )
    else:
        payload["storage"] = {"durable": storage_status()["durable"]}
    return payload


@app.get("/ready")
def readiness_check():
    """Readiness probe: 200 only when adjudication requests would succeed right now.

    Returns 503 while a required LLM backend is still loading or failed to load,
    so load balancers / Cloud Run startup probes hold traffic instead of letting
    clients collect 503s from ``/api/adjudicate/*``.
    """
    backend = _backend_snapshot()
    serving = _backend_serving(backend)
    body = {
        "ready": serving,
        "backend": backend.get("backend"),
        "llm_real": bool(backend.get("llm_real")),
        "backend_initializing": engine.is_initializing,
    }
    if serving:
        return body
    return JSONResponse(status_code=503, content=body, headers={"Retry-After": "10"})


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
def adjudicate_single_visit(req: SingleAdjudicationRequest, request: Request):
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
    return _run_engine(request, engine.adjudicate_single, record, req.target_condition, criteria)


@api.post("/adjudicate/batch", response_model=List[RoseGoldAdjudication])
def adjudicate_batch_visits(req: BatchAdjudicationRequest, request: Request):
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

    return _run_engine(request, _run)


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
