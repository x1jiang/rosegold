"""Regression tests for the production-readiness layer (second hardening pass).

Covers: shared path guard, request ids, CSP, rate limiting, /health redaction,
/ready semantics, generic 503s, backend retry-with-cooldown, hybrid endpoint
policy and honest provenance, Vertex per-record isolation, atomic storage
writes, loader NaN handling, and JSON logging.
"""

import json
import logging
import os
import threading

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api as api_module
import app.engine as engine_module
from app import hybrid_engine, logging_setup, paths, storage
from app.api import RateLimitMiddleware, app
from app.engine import AdjudicationEngine
from app.omop_loader import load_omop_data
from app.vertex_engine import VertexGeminiEngine


@pytest.fixture
def client():
    return TestClient(app)


# --------------------------------------------------------------------------
# Shared path guard
# --------------------------------------------------------------------------


def test_resolve_data_file_rules(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "ok.csv").write_text("a\n")
    (root / "ok.parquet").write_bytes(b"PAR1")
    (root / "notes.txt").write_text("a\n")
    (root / "sub").mkdir()
    (tmp_path / "data_evil").mkdir()
    (tmp_path / "data_evil" / "x.csv").write_text("a\n")
    (tmp_path / "outside.csv").write_text("a\n")
    link = root / "link.csv"
    link.symlink_to(tmp_path / "outside.csv")

    assert paths.resolve_data_file(str(root / "ok.csv"), root=str(root)) == os.path.realpath(str(root / "ok.csv"))
    assert paths.resolve_data_file(str(root / "ok.parquet"), root=str(root)).endswith("ok.parquet")
    for bad in (
        None,
        "",
        "   ",
        str(root),
        str(root / "sub"),
        str(root / "notes.txt"),               # wrong extension
        str(root / "missing.csv"),
        str(root / ".." / "outside.csv"),
        str(tmp_path / "data_evil" / "x.csv"),  # prefix collision
        str(link),                              # symlink escape
        "a\x00b.csv",
        "x" * 3000 + ".csv",
    ):
        with pytest.raises(paths.UnsafePathError):
            paths.resolve_data_file(bad, root=str(root))
        assert paths.is_safe_data_file(bad, root=str(root)) is False


def test_data_dir_reads_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ROSEGOLD_DATA_DIR", str(tmp_path))
    assert paths.data_dir() == os.path.realpath(str(tmp_path))


# --------------------------------------------------------------------------
# Request ids, CSP, and 503 sanitisation
# --------------------------------------------------------------------------


def test_request_id_generated_and_echoed(client):
    res = client.get("/health")
    generated = res.headers["x-request-id"]
    assert len(generated) == 16

    res = client.get("/health", headers={"X-Request-ID": "trace-abc.123_XYZ"})
    assert res.headers["x-request-id"] == "trace-abc.123_XYZ"

    # Malformed inbound ids are replaced, never reflected.
    res = client.get("/health", headers={"X-Request-ID": "<script>alert(1)</script>"})
    assert res.headers["x-request-id"] != "<script>alert(1)</script>"
    assert "<" not in res.headers["x-request-id"]


def test_request_id_present_on_short_circuit_responses(client):
    res = client.post(
        "/api/adjudicate/single",
        content=b"{}",
        headers={"content-type": "application/json", "content-length": str(api_module.MAX_BODY_BYTES + 1), "X-Request-ID": "big-1"},
    )
    assert res.status_code == 413
    assert res.headers["x-request-id"] == "big-1"


def test_csp_on_api_routes_only(client):
    assert client.get("/health").headers["content-security-policy"].startswith("default-src 'none'")
    assert client.get("/api/visits").headers["content-security-policy"].startswith("default-src 'none'")
    assert "content-security-policy" not in {k.lower() for k in client.get("/docs").headers}


def test_backend_unavailable_message_is_generic(client, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("https://internal-host:7790/v1/completions refused /mnt/gcs/models/x.gguf")

    monkeypatch.setattr(api_module.engine, "adjudicate_single", boom)
    res = client.post("/api/adjudicate/single", json={"notes_formatted_text": "sepsis"})
    assert res.status_code == 503
    body = json.dumps(res.json())
    assert "internal-host" not in body and "/mnt/gcs" not in body
    assert res.headers["retry-after"] == "30"


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


def _limited_app(per_minute: int, trust_proxy: bool = False) -> TestClient:
    inner = FastAPI()

    @inner.get("/api/ping")
    async def ping():
        return {"ok": True}

    @inner.get("/health")
    async def health():
        return {"ok": True}

    inner.add_middleware(RateLimitMiddleware, per_minute=per_minute, trust_proxy=trust_proxy)
    return TestClient(inner)


def test_rate_limit_enforced_on_api_routes():
    tc = _limited_app(3)
    assert [tc.get("/api/ping").status_code for _ in range(3)] == [200, 200, 200]
    res = tc.get("/api/ping")
    assert res.status_code == 429
    assert int(res.headers["retry-after"]) >= 1
    assert res.json()["detail"] == "Rate limit exceeded."
    # Probes are never limited.
    assert tc.get("/health").status_code == 200


def test_rate_limit_ignores_forwarded_for_unless_trusted():
    tc = _limited_app(2, trust_proxy=False)
    for _ in range(2):
        tc.get("/api/ping", headers={"X-Forwarded-For": "1.1.1.1"})
    # Spoofing a new address does not reset the bucket without ROSEGOLD_TRUST_PROXY.
    assert tc.get("/api/ping", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429

    trusted = _limited_app(2, trust_proxy=True)
    for _ in range(2):
        trusted.get("/api/ping", headers={"X-Forwarded-For": "1.1.1.1"})
    assert trusted.get("/api/ping", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 429
    assert trusted.get("/api/ping", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200


def test_rate_limit_disabled_by_default(client):
    assert api_module.RATE_LIMIT_PER_MIN == 0
    assert all(client.get("/api/visits").status_code == 200 for _ in range(5))


# --------------------------------------------------------------------------
# /health redaction and /ready
# --------------------------------------------------------------------------


def test_health_redacts_paths_when_key_configured(client, monkeypatch):
    monkeypatch.delenv("ROSEGOLD_API_KEY", raising=False)
    open_payload = client.get("/health").json()
    assert "notes_file" in open_payload and "audit_log" in open_payload["storage"]

    monkeypatch.setenv("ROSEGOLD_API_KEY", "k")
    anon = client.get("/health")
    assert anon.status_code == 200
    body = anon.json()
    assert body["auth_required"] is True
    assert "notes_file" not in body and "visits_file" not in body and "error" not in body
    assert set(body["storage"]) == {"durable"}

    authed = client.get("/health", headers={"X-API-Key": "k"}).json()
    assert "notes_file" in authed and "output_dir" in authed["storage"]


def test_ready_ok_in_rules_mode(client, monkeypatch):
    monkeypatch.delenv("ROSEGOLD_LLM_BACKEND", raising=False)
    monkeypatch.delenv("K_SERVICE", raising=False)
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json()["ready"] is True
    assert client.get("/health").json()["ready"] is True


def test_ready_503_when_required_backend_missing(client, monkeypatch):
    monkeypatch.setenv("ROSEGOLD_LLM_BACKEND", "hybrid")
    monkeypatch.delenv("ROSEGOLD_MUSE_URL", raising=False)
    monkeypatch.delenv("ROSEGOLD_ALLOW_MOCK", raising=False)
    fresh = AdjudicationEngine(model_name="auto")
    monkeypatch.setattr(api_module, "engine", fresh)

    # Before init: loading -> not ready.
    res = client.get("/ready")
    assert res.status_code == 503
    assert res.headers["retry-after"] == "10"
    assert res.json()["backend"] == "loading"

    # After a failed init: still not ready, health says degraded but stays 200 (liveness).
    fresh.backend_status(init=True)
    assert client.get("/ready").status_code == 503
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["ready"] is False

    # And adjudication refuses to hand out keyword-rule labels.
    res = client.post("/api/adjudicate/single", json={"notes_formatted_text": "septic shock"})
    assert res.status_code == 503


# --------------------------------------------------------------------------
# Engine: retry with cooldown, hybrid honesty
# --------------------------------------------------------------------------


def test_engine_retries_failed_backend_after_cooldown(monkeypatch):
    monkeypatch.setenv("ROSEGOLD_LLM_BACKEND", "hybrid")
    monkeypatch.delenv("ROSEGOLD_MUSE_URL", raising=False)
    monkeypatch.delenv("ROSEGOLD_ALLOW_MOCK", raising=False)
    monkeypatch.setenv("ROSEGOLD_BACKEND_RETRY_SECONDS", "5")

    clock = {"now": 1000.0}
    monkeypatch.setattr(engine_module.time, "monotonic", lambda: clock["now"])
    engine = AdjudicationEngine(model_name="auto")
    record = {"person_id": 1, "visit_occurrence_id": 1, "notes_formatted_text": "septic shock on pressors"}

    with pytest.raises(RuntimeError):
        engine.adjudicate_single(record, "Sepsis / Septic Shock", "criteria")
    assert engine.wants_real_backend() and not engine.has_real_backend()
    assert engine.backend_status()["backend"] == "keyword_rules"

    # Within the cooldown nothing is retried.
    calls = {"n": 0}
    original = engine._init_backend_locked

    def counting():
        calls["n"] += 1
        return original()

    monkeypatch.setattr(engine, "_init_backend_locked", counting)
    with pytest.raises(RuntimeError):
        engine.adjudicate_single(record, "Sepsis / Septic Shock", "criteria")
    assert calls["n"] == 0

    # Operator fixes configuration; after the cooldown the next request re-inits and succeeds.
    monkeypatch.setenv("ROSEGOLD_MUSE_URL", "https://muse.internal.example/v1/completions")
    clock["now"] += 6
    result = engine.adjudicate_single(record, "Sepsis / Septic Shock", "criteria")
    assert calls["n"] == 1
    assert engine.has_real_backend()
    assert result["inference_backend"].startswith("hybrid:")
    assert engine._next_retry_at is None


def test_hybrid_url_policy(monkeypatch):
    monkeypatch.delenv("ROSEGOLD_MUSE_ALLOW_HTTP", raising=False)
    monkeypatch.delenv("ROSEGOLD_MUSE_URL", raising=False)
    with pytest.raises(ValueError, match="not set"):
        hybrid_engine.muse_url()
    with pytest.raises(ValueError, match="plain http"):
        hybrid_engine.validate_muse_url("http://10.0.0.5:7790/v1/completions")
    with pytest.raises(ValueError, match="plain http"):
        hybrid_engine.validate_muse_url("http://129.106.31.72:7790/v1/completions")
    for bad in ("ftp://x/y", "muse.internal", "https://"):
        with pytest.raises(ValueError):
            hybrid_engine.validate_muse_url(bad)
    assert hybrid_engine.validate_muse_url("https://muse.internal/v1/completions")
    assert hybrid_engine.validate_muse_url("http://127.0.0.1:8001/v1/completions")
    assert hybrid_engine.validate_muse_url("http://localhost:8001/v1/completions")
    monkeypatch.setenv("ROSEGOLD_MUSE_ALLOW_HTTP", "1")
    assert hybrid_engine.validate_muse_url("http://10.0.0.5:7790/v1/completions")


class _FakeSession:
    def __init__(self, status=200, text="true", raise_exc=None):
        self.status = status
        self.text = text
        self.raise_exc = raise_exc
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers or {}, "json": json, "timeout": timeout})
        if self.raise_exc:
            raise self.raise_exc

        class R:
            status_code = self.status
            headers = {}

            def json(inner):
                return {"choices": [{"text": self.text}]}

        return R()


_SEPTIC = {
    "person_id": 1,
    "visit_occurrence_id": 7,
    "visit_start_date": "2026-01-01",
    "visit_end_date": "2026-01-03",
    "notes_formatted_text": "Patient in septic shock, lactate 4.2, started on norepinephrine for MAP < 65.",
}


def test_hybrid_reports_rules_only_when_llm_fails(monkeypatch):
    import requests

    monkeypatch.setenv("ROSEGOLD_MUSE_API_KEY", "muse-token")
    session = _FakeSession(raise_exc=requests.ConnectionError("refused"))
    engine = hybrid_engine.HybridAdjudicationEngine(
        endpoint_url="https://muse.internal/v1/completions", session=session
    )
    out = engine.adjudicate_single(_SEPTIC, "Sepsis / Septic Shock", "criteria")
    assert out["inference_backend"] == "hybrid:rules_only(ConnectionError)"
    assert out["confidence_score"] == hybrid_engine.RULES_ONLY_CONFIDENCE
    assert "rules only" in out["clinical_rationale"]
    assert session.calls[0]["headers"]["Authorization"] == "Bearer muse-token"

    session = _FakeSession(status=500)
    engine = hybrid_engine.HybridAdjudicationEngine(endpoint_url="https://muse.internal/v1/completions", session=session)
    out = engine.adjudicate_single(_SEPTIC, "Sepsis / Septic Shock", "criteria")
    assert out["inference_backend"] == "hybrid:rules_only(http_500)"


def test_hybrid_reports_llm_backend_when_verified(monkeypatch):
    monkeypatch.delenv("ROSEGOLD_MUSE_API_KEY", raising=False)
    session = _FakeSession(status=200, text="true, \"reason\": \"documented septic shock\"")
    engine = hybrid_engine.HybridAdjudicationEngine(
        endpoint_url="https://muse.internal/v1/completions", model_name="Muse-Test", session=session
    )
    out = engine.adjudicate_single(_SEPTIC, "Sepsis / Septic Shock", "criteria")
    assert out["inference_backend"] == "hybrid:Muse-Test"
    assert out["condition_present"] is True
    assert "Muse LLM Verification" in out["clinical_rationale"]
    assert "Authorization" not in session.calls[0]["headers"]


# --------------------------------------------------------------------------
# Vertex per-record isolation
# --------------------------------------------------------------------------


def test_vertex_one_bad_response_does_not_abort_batch():
    class _Models:
        def __init__(self):
            self.n = 0

        def generate_content(self, **_kwargs):
            self.n += 1
            if self.n == 1:
                raise TimeoutError("deadline")

            class R:
                text = json.dumps(
                    {
                        "clinical_rationale": "Documented septic shock.",
                        "primary_criteria_met": ["infection", "organ dysfunction"],
                        "key_evidence": [],
                        "phenotype_status": "CONFIRMED_POSITIVE",
                        "condition_present": True,
                        "confidence_score": 0.9,
                    }
                )

            return R()

    class _Client:
        models = _Models()

    pytest.importorskip("google.genai")
    engine = VertexGeminiEngine(model_name="gemini-test", project="p", location="l", client=_Client())
    records = [dict(_SEPTIC, visit_occurrence_id=1), dict(_SEPTIC, visit_occurrence_id=2)]
    out = engine.adjudicate_batch(records, "Sepsis / Septic Shock", "criteria")
    assert [r["visit_occurrence_id"] for r in out] == [1, 2]
    assert out[0]["phenotype_status"] == "INDETERMINATE_INSUFFICIENT_DATA"
    assert "TimeoutError" in out[0]["clinical_rationale"]
    assert out[1]["phenotype_status"] == "CONFIRMED_POSITIVE"
    assert out[1]["inference_backend"] == "vertex:gemini-test"


# --------------------------------------------------------------------------
# Storage: atomic whole-file writes, no temp residue, concurrent appends
# --------------------------------------------------------------------------


def test_save_batch_results_is_atomic_and_leaves_no_temp_files(tmp_path, monkeypatch):
    monkeypatch.setenv("ROSEGOLD_OUTPUT_DIR", str(tmp_path))
    results = [
        {
            "person_id": 1,
            "visit_occurrence_id": 1,
            "condition_present": True,
            "phenotype_status": "CONFIRMED_POSITIVE",
            "confidence_score": 0.9,
            "primary_criteria_met": [],
            "key_evidence": [],
            "clinical_rationale": "x",
        }
    ]
    paths_out = storage.save_batch_results(results, "Sepsis / Septic Shock")
    for key in ("csv", "jsonl", "omop"):
        assert os.path.isfile(paths_out[key])
    assert storage.load_batch_results() == results
    assert not [p for p in os.listdir(tmp_path) if p.startswith(".tmp-")]

    # A failure mid-write leaves the previous good file intact.
    def exploding(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(storage, "_atomic_write_bytes", exploding)
    with pytest.raises(OSError):
        storage.save_criteria("new text")
    monkeypatch.undo()
    monkeypatch.setenv("ROSEGOLD_OUTPUT_DIR", str(tmp_path))
    assert storage.load_batch_results() == results
    assert not [p for p in os.listdir(tmp_path) if p.startswith(".tmp-")]


def test_concurrent_audit_appends_do_not_interleave(tmp_path, monkeypatch):
    monkeypatch.setenv("ROSEGOLD_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("ROSEGOLD_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    comment = "c" * 20_000

    def worker(i):
        storage.append_audit(
            {"visit_occurrence_id": i, "person_id": 1, "adjudication_status": "CONFIRMED_POSITIVE", "reviewer_agreement": True, "comments": comment}
        )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = storage.read_audit()
    assert sorted(r["visit_occurrence_id"] for r in rows) == list(range(16))
    assert all(r["comments"] == comment for r in rows)


# --------------------------------------------------------------------------
# Loader: NaN person_id no longer crashes
# --------------------------------------------------------------------------


def test_loader_tolerates_missing_person_id(tmp_path):
    notes = tmp_path / "notes.csv"
    visits = tmp_path / "visits.csv"
    pd.DataFrame(
        {
            "note_id": [1, 2],
            "person_id": [10, None],
            "visit_occurrence_id": [100, 200],
            "note_date": ["2026-01-01", "2026-01-02"],
            "note_text": ["fever and hypotension", "knee replacement"],
        }
    ).to_csv(notes, index=False)
    pd.DataFrame(
        {
            "visit_occurrence_id": [100, 200],
            "person_id": [10, None],
            "visit_start_date": ["2026-01-01", "2026-01-02"],
            "visit_end_date": ["2026-01-03", "2026-01-04"],
        }
    ).to_csv(visits, index=False)
    records = load_omop_data(str(notes), str(visits))
    assert [r["visit_occurrence_id"] for r in records] == [100, 200]
    assert records[0]["person_id"] == 10
    assert records[1]["person_id"] == 0


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------


def test_json_log_formatter_emits_one_object_per_line():
    formatter = logging_setup._JsonFormatter()
    record = logging.LogRecord("rosegold.test", logging.WARNING, __file__, 1, "hello %s", ("world",), None)
    line = formatter.format(record)
    parsed = json.loads(line)
    assert parsed["level"] == "WARNING"
    assert parsed["message"] == "hello world"
    assert parsed["logger"] == "rosegold.test"
    assert parsed["ts"].endswith("Z")


def test_configure_logging_is_idempotent():
    logging_setup.configure_logging()
    before = list(logging.getLogger().handlers)
    logging_setup.configure_logging()
    assert logging.getLogger().handlers == before
