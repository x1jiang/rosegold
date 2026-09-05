"""Regression tests for the security hardening layer."""

import hashlib
import io
import json
import os
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api as api_module
from app import model_store, storage
from app.api import BodySizeLimitMiddleware, _safe_data_path, app
from app.engine import AdjudicationEngine, _trust_remote_code
from app.prompts import SYSTEM_PROMPT
from fastapi import HTTPException


@pytest.fixture
def client():
    return TestClient(app)


# --------------------------------------------------------------------------
# Path validation
# --------------------------------------------------------------------------


def test_safe_data_path_rejects_traversal_prefix_collision_and_dirs(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "ok.csv").write_text("a,b\n1,2\n")
    (data_dir / "sub").mkdir()
    evil_dir = tmp_path / "data_evil"
    evil_dir.mkdir()
    (evil_dir / "n.csv").write_text("a\n")
    outside = tmp_path / "outside.csv"
    outside.write_text("a\n")
    monkeypatch.setattr(api_module, "DATA_DIR", os.path.realpath(str(data_dir)))

    assert _safe_data_path(str(data_dir / "ok.csv"), "default") == os.path.realpath(str(data_dir / "ok.csv"))
    assert _safe_data_path(None, "default") == "default"
    assert _safe_data_path("", "default") == "default"

    for bad in (
        str(evil_dir / "n.csv"),                       # DATA_DIR prefix collision
        str(data_dir / ".." / "outside.csv"),          # traversal
        str(outside),                                  # outside entirely
        str(data_dir),                                 # the directory itself
        str(data_dir / "sub"),                         # a directory inside
        str(data_dir / "missing.csv"),                 # nonexistent
        "x" * 5000,                                    # absurd length
        "a\x00b",                                      # NUL byte
    ):
        with pytest.raises(HTTPException) as excinfo:
            _safe_data_path(bad, "default")
        assert excinfo.value.status_code == 400


def test_safe_data_path_rejects_symlink_escape(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    secret = tmp_path / "secret.csv"
    secret.write_text("x\n")
    link = data_dir / "link.csv"
    link.symlink_to(secret)
    monkeypatch.setattr(api_module, "DATA_DIR", os.path.realpath(str(data_dir)))
    with pytest.raises(HTTPException):
        _safe_data_path(str(link), "default")


def test_batch_endpoint_rejects_bad_path(client):
    res = client.post(
        "/api/adjudicate/batch",
        json={"visit_occurrence_ids": [20001], "notes_path": "../../etc/passwd"},
    )
    assert res.status_code == 400
    assert "data directory" in res.json()["detail"]


# --------------------------------------------------------------------------
# Input bounds
# --------------------------------------------------------------------------


def test_oversized_notes_text_rejected(client):
    res = client.post(
        "/api/adjudicate/single",
        json={"notes_formatted_text": "x" * (api_module.MAX_NOTES_CHARS + 1)},
    )
    assert res.status_code == 422


def test_blank_target_condition_rejected(client):
    res = client.post("/api/adjudicate/single", json={"notes_formatted_text": "sepsis", "target_condition": "   "})
    assert res.status_code == 422


def test_feedback_confidence_bounds(client):
    base = {
        "visit_occurrence_id": 1,
        "person_id": 1,
        "adjudication_status": "CONFIRMED_POSITIVE",
        "reviewer_agreement": True,
    }
    assert client.post("/api/feedback", json={**base, "llm_confidence": 1.5}).status_code == 422
    assert client.post("/api/feedback", json={**base, "comments": "x" * 5000}).status_code == 422
    assert client.post("/api/feedback", json={**base, "llm_confidence": 0.5}).status_code == 200


def test_criteria_length_bound(client):
    assert client.post("/api/criteria", json={"text": "x" * (api_module.MAX_CRITERIA_CHARS + 1)}).status_code == 422


def test_batch_id_list_bound(client):
    ids = list(range(api_module.MAX_BATCH_VISITS + 1))
    assert client.post("/api/adjudicate/batch", json={"visit_occurrence_ids": ids}).status_code == 422


# --------------------------------------------------------------------------
# Body-size middleware
# --------------------------------------------------------------------------


def _tiny_app(limit: int) -> TestClient:
    inner = FastAPI()

    @inner.post("/echo")
    async def echo(payload: dict):
        return {"n": len(json.dumps(payload))}

    inner.add_middleware(BodySizeLimitMiddleware, max_bytes=limit)
    return TestClient(inner)


def test_body_limit_content_length_and_streamed():
    tc = _tiny_app(64)
    small = tc.post("/echo", json={"a": "b"})
    assert small.status_code == 200
    big = tc.post("/echo", json={"a": "x" * 500})
    assert big.status_code == 413
    # Chunked body without a Content-Length header.
    chunked = tc.post("/echo", content=io.BytesIO(json.dumps({"a": "x" * 500}).encode()), headers={"content-type": "application/json"})
    assert chunked.status_code == 413


def test_real_app_body_limit_via_header(client):
    res = client.post(
        "/api/adjudicate/single",
        content=b"{}",
        headers={"content-type": "application/json", "content-length": str(api_module.MAX_BODY_BYTES + 1)},
    )
    assert res.status_code == 413


# --------------------------------------------------------------------------
# Headers, CORS, auth
# --------------------------------------------------------------------------


def test_security_headers_present(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.headers["x-content-type-options"] == "nosniff"
    assert res.headers["x-frame-options"] == "DENY"
    assert res.headers["referrer-policy"] == "no-referrer"
    assert res.headers["cache-control"] == "no-store"
    assert res.json()["auth_required"] is False


def test_cors_closed_by_default(client):
    res = client.get("/api/visits", headers={"Origin": "https://evil.example"})
    assert res.status_code == 200
    assert "access-control-allow-origin" not in {k.lower() for k in res.headers}


def test_api_key_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("ROSEGOLD_API_KEY", "s3cret-key")
    assert client.get("/health").status_code == 200
    assert client.get("/health").json()["auth_required"] is True
    assert client.get("/api/visits").status_code == 401
    assert client.get("/api/visits", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/visits", headers={"X-API-Key": "s3cret-key"}).status_code == 200
    assert client.get("/api/visits", headers={"Authorization": "Bearer s3cret-key"}).status_code == 200
    assert client.post("/api/feedback", json={
        "visit_occurrence_id": 1, "person_id": 1, "adjudication_status": "X", "reviewer_agreement": True,
    }).status_code == 401


def test_unhandled_exception_is_sanitized(client, monkeypatch):
    def boom(*_args, **_kwargs):
        raise KeyError("/secret/internal/path")

    monkeypatch.setattr(api_module, "read_audit", boom)
    tc = TestClient(app, raise_server_exceptions=False)
    res = tc.get("/api/audit")
    assert res.status_code == 500
    body = res.json()
    assert body["detail"] == "Internal server error."
    assert "secret" not in json.dumps(body)
    # The correlation id in the body is the request id on the response so
    # clients can quote one value and operators grep one value in the logs.
    assert body["error_id"] == res.headers["x-request-id"]
    assert len(body["error_id"]) >= 12


def test_missing_visit_returns_404_not_500(client):
    assert client.get("/api/notes/424242424242").status_code == 404


# --------------------------------------------------------------------------
# Storage resilience
# --------------------------------------------------------------------------


def test_read_audit_skips_corrupt_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("ROSEGOLD_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("ROSEGOLD_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    storage.append_audit({"visit_occurrence_id": 1, "person_id": 1, "adjudication_status": "CONFIRMED_POSITIVE", "reviewer_agreement": True})
    with open(tmp_path / "audit.jsonl", "a", encoding="utf-8") as handle:
        handle.write('{"visit_occurrence_id": 2, "trunc')  # interrupted write, no newline
        handle.write("\n[1,2,3]\n\n")
    storage.append_audit({"visit_occurrence_id": 3, "person_id": 1, "adjudication_status": "CONFIRMED_NEGATIVE", "reviewer_agreement": False})
    rows = storage.read_audit()
    assert [r["visit_occurrence_id"] for r in rows] == [1, 3]
    assert storage.load_batch_results() == []


# --------------------------------------------------------------------------
# Model download hardening
# --------------------------------------------------------------------------


def test_download_url_must_be_https():
    for bad in ("http://example.com/x.gguf", "ftp://example.com/x.gguf", "file:///etc/passwd", "x.gguf", "https:///nohost"):
        with pytest.raises(ValueError):
            model_store.validate_download_url(bad)
    assert model_store.validate_download_url("https://huggingface.co/a/b.gguf")


def test_gguf_filename_strips_directories(monkeypatch):
    monkeypatch.setenv("ROSEGOLD_LLAMA_GGUF", "../../etc/passwd")
    assert model_store.gguf_filename() == "passwd"


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


def test_download_verifies_sha256_and_cleans_up(tmp_path, monkeypatch):
    payload = b"GGUF" * 64
    good_sha = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(model_store, "MIN_BYTES", 8)
    monkeypatch.setattr(model_store.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload))
    dest = tmp_path / "m.gguf"

    monkeypatch.setenv("ROSEGOLD_LLAMA_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        model_store._download("https://example.com/m.gguf", str(dest))
    assert not dest.exists()
    assert not (tmp_path / "m.gguf.part").exists()

    monkeypatch.setenv("ROSEGOLD_LLAMA_SHA256", good_sha)
    model_store._download("https://example.com/m.gguf", str(dest))
    assert dest.read_bytes() == payload
    assert not (tmp_path / "m.gguf.part").exists()


def test_download_rejects_too_small(tmp_path, monkeypatch):
    monkeypatch.delenv("ROSEGOLD_LLAMA_SHA256", raising=False)
    monkeypatch.setattr(model_store, "MIN_BYTES", 10_000)
    monkeypatch.setattr(model_store.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(b"tiny"))
    dest = tmp_path / "m.gguf"
    with pytest.raises(RuntimeError, match="too small"):
        model_store._download("https://example.com/m.gguf", str(dest))
    assert not dest.exists()


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


def test_trust_remote_code_off_by_default(monkeypatch):
    monkeypatch.delenv("ROSEGOLD_TRUST_REMOTE_CODE", raising=False)
    assert _trust_remote_code() is False
    monkeypatch.setenv("ROSEGOLD_TRUST_REMOTE_CODE", "1")
    assert _trust_remote_code() is True


def test_engine_concurrent_first_use_is_safe(monkeypatch):
    monkeypatch.delenv("ROSEGOLD_LLM_BACKEND", raising=False)
    monkeypatch.delenv("K_SERVICE", raising=False)
    engine = AdjudicationEngine(model_name="auto")
    record = {"person_id": 1, "visit_occurrence_id": 1, "notes_formatted_text": "Patient with septic shock on norepinephrine."}
    errors = []
    outputs = []

    def worker():
        try:
            outputs.append(engine.adjudicate_single(record, "Sepsis / Septic Shock", "criteria"))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(outputs) == 8
    assert engine.is_initializing is False
    assert engine.adjudicate_batch([], "Sepsis / Septic Shock", "criteria") == []


def test_prompt_declares_notes_as_data():
    assert "not instructions" in SYSTEM_PROMPT
