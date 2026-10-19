"""Required Muse calls must preserve uncertainty and fail without rule substitution."""
import json

from app.hybrid_engine import HybridAdjudicationEngine


class Session:
    def __init__(self, payload=None, status=200, finish="stop", model="Muse-Test"):
        self.payload, self.status, self.finish, self.model = payload, status, finish, model
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        owner = self

        class Response:
            status_code = owner.status

            def json(self):
                return {"model": owner.model, "choices": [{"finish_reason": owner.finish,
                        "message": {"content": json.dumps(owner.payload)}}]}

        return Response()


RECORD = {"person_id": 1, "visit_occurrence_id": 7,
          "notes_formatted_text": "Observed creatinine rise over 48 hours: 0.5 mg/dl."}


def payload(quote=None):
    return {"clinical_rationale": "Reported creatinine rise supports suspected acute kidney injury.",
            "phenotype_status": "SUSPECTED_PROBABLE", "condition_present": True,
            "confidence_score": 0.7, "primary_criteria_met": [],
            "key_evidence": [{"evidence_quote": quote or RECORD["notes_formatted_text"]}]}


def run(session, endpoint="https://muse.internal/v1/completions"):
    return HybridAdjudicationEngine(endpoint_url=endpoint, model_name="Muse-Test",
                                   session=session, require_llm=True).adjudicate_single(RECORD, "AKI", "criteria")


def test_required_mode_calls_muse_even_without_rule_trigger():
    session = Session(payload())
    result = run(session)
    assert len(session.calls) == 1
    assert session.calls[0][0] == "https://muse.internal/v1/chat/completions"
    assert session.calls[0][1]["json"]["max_tokens"] == 4096
    assert result["phenotype_status"] == "SUSPECTED_PROBABLE"
    assert result["inference_backend"] == "hybrid:Muse-Test:required_llm"
    assert result["key_evidence"][0]["evidence_quote"] == RECORD["notes_formatted_text"]


def test_required_mode_fails_closed():
    for session in [Session(status=503), Session(payload(), finish="length"),
                    Session(payload(), model="Different-model"), Session({"invalid": True})]:
        result = run(session)
        assert result["phenotype_status"] == "INDETERMINATE_INSUFFICIENT_DATA"
        assert not result["condition_present"]
        assert result["key_evidence"] == []
        assert result["inference_backend"].startswith("hybrid:llm_failed(")


def test_required_mode_rejects_ungrounded_evidence_and_handles_chat_url():
    session = Session(payload("Completely invented laboratory test."))
    result = run(session, "https://muse.internal/v1/chat/completions")
    assert session.calls[0][0] == "https://muse.internal/v1/chat/completions"
    assert result["phenotype_status"] == "INDETERMINATE_INSUFFICIENT_DATA"
    assert result["key_evidence"] == []


def test_required_mode_runs_through_rosegold_engine(monkeypatch):
    from app.engine import AdjudicationEngine
    from app import hybrid_engine

    session = Session(payload())
    monkeypatch.setenv("ROSEGOLD_LLM_BACKEND", "hybrid")
    monkeypatch.setenv("ROSEGOLD_MUSE_URL", "https://muse.internal/v1/completions")
    monkeypatch.setenv("ROSEGOLD_MUSE_MODEL", "Muse-Test")
    monkeypatch.setenv("ROSEGOLD_HYBRID_REQUIRE_LLM", "1")
    monkeypatch.setenv("ROSEGOLD_ALLOW_MOCK", "0")
    monkeypatch.setattr(hybrid_engine.requests, "Session", lambda: session)
    engine = AdjudicationEngine(model_name="Muse-Test")
    result = engine.adjudicate_single(RECORD, "AKI", "criteria")
    assert result["inference_backend"] == "hybrid:Muse-Test:required_llm"
    assert len(session.calls) == 1


def test_masked_inference_preserves_result_ids_without_exposing_them():
    first = Session(payload())
    second = Session(payload())
    record_a = {**RECORD, "inference_identity_masked": True}
    record_b = {**record_a, "person_id": 918273, "visit_occurrence_id": 564738}
    outputs = []
    for session, record in [(first, record_a), (second, record_b)]:
        engine = HybridAdjudicationEngine(endpoint_url="https://muse.internal/v1/completions",
                                         model_name="Muse-Test", session=session, require_llm=True)
        outputs.append(engine.adjudicate_single(record, "AKI", "criteria"))
    assert first.calls[0][1]["json"] == second.calls[0][1]["json"]
    assert outputs[0]["person_id"] == RECORD["person_id"]
    assert outputs[1]["person_id"] == 918273
    assert outputs[1]["visit_occurrence_id"] == 564738
