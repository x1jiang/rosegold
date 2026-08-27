import json

from app.vertex_engine import VertexGeminiEngine, _parse_model_json


class _FakeResponse:
    def __init__(self, payload):
        self.text = json.dumps(payload)


class _FakeModels:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self.payload)


class _FakeClient:
    def __init__(self, payload):
        self.models = _FakeModels(payload)


def test_parse_wrapped_schema_object():
    parsed = _parse_model_json("""{
      "RoseGoldAdjudication": {
        "clinical_rationale": "Urosepsis with vasopressors meets septic shock.",
        "primary_criteria_met": ["infection"],
        "key_evidence": [{"evidence_quote": "Severe Sepsis and Septic Shock", "interpretation": "dx"}],
        "phenotype_status": "CONFIRMED_POSITIVE",
        "condition_present": true,
        "confidence_score": 0.8
      }
    }""")
    assert parsed.phenotype_status == "CONFIRMED_POSITIVE"
    assert "Urosepsis" in parsed.clinical_rationale


def test_parse_fenced_json():
    parsed = _parse_model_json("""```json
{"clinical_rationale": "notes mention septic shock and vasopressors", "primary_criteria_met": ["infection"], "key_evidence": [{"evidence_quote": "Severe Sepsis and Septic Shock secondary to Klebsiella", "interpretation": "explicit diagnosis"}], "phenotype_status": "CONFIRMED_POSITIVE", "condition_present": true, "confidence_score": 0.81}
```""")
    assert parsed.phenotype_status == "CONFIRMED_POSITIVE"
    assert parsed.confidence_score == 0.81


def test_vertex_engine_uses_client_and_tags_backend():
    payload = {
        "clinical_rationale": "Chart documents Klebsiella urosepsis with shock requiring Levophed.",
        "primary_criteria_met": ["infection", "vasopressors"],
        "key_evidence": [{
            "note_id": 40003,
            "note_date": "2026-03-08",
            "evidence_quote": "Severe Sepsis and Septic Shock secondary to Klebsiella urosepsis",
            "interpretation": "Discharge diagnosis",
        }],
        "phenotype_status": "CONFIRMED_POSITIVE",
        "condition_present": True,
        "confidence_score": 0.83,
    }
    engine = VertexGeminiEngine(model_name="gemini-2.5-flash", client=_FakeClient(payload))
    rows = engine.adjudicate_batch(
        [{
            "person_id": 1001,
            "visit_occurrence_id": 20001,
            "visit_start_date": "2026-03-01",
            "visit_end_date": "2026-03-08",
            "notes_formatted_text": "Severe Sepsis and Septic Shock secondary to Klebsiella urosepsis",
        }],
        "Sepsis / Septic Shock",
        "Sepsis-3",
    )
    assert rows[0]["inference_backend"] == "vertex:gemini-2.5-flash"
    assert rows[0]["clinical_rationale"].startswith("Chart documents Klebsiella")
    assert rows[0]["confidence_score"] == 0.83
    assert engine.client.models.calls
    assert engine.client.models.calls[0]["model"] == "gemini-2.5-flash"


def test_engine_reports_keyword_backend_locally(monkeypatch):
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("ROSEGOLD_LLM_BACKEND", raising=False)
    from app.engine import AdjudicationEngine

    status = AdjudicationEngine().backend_status()
    assert status["backend"] == "keyword_rules"
    assert status["llm_real"] is False
