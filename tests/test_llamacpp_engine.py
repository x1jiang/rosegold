from app.llamacpp_engine import LlamaCppEngine
from app.model_store import ensure_llama_gguf


class _FakeLlama:
    def create_chat_completion(self, **kwargs):
        return {
            "choices": [{
                "message": {
                    "content": (
                        '{"clinical_rationale": "Notes document urosepsis with vasopressors.",'
                        '"primary_criteria_met": ["infection", "shock"],'
                        '"key_evidence": [{"evidence_quote": "Severe Sepsis and Septic Shock",'
                        '"interpretation": "discharge diagnosis"}],'
                        '"phenotype_status": "CONFIRMED_POSITIVE",'
                        '"condition_present": true,'
                        '"confidence_score": 0.77}'
                    )
                }
            }]
        }


def test_llamacpp_engine_tags_real_llama_backend():
    engine = LlamaCppEngine(model_path="injected", llm=_FakeLlama())
    rows = engine.adjudicate_batch(
        [{
            "person_id": 1001,
            "visit_occurrence_id": 20001,
            "visit_start_date": "2026-03-01",
            "visit_end_date": "2026-03-08",
            "notes_formatted_text": "Severe Sepsis and Septic Shock",
        }],
        "Sepsis / Septic Shock",
        "Sepsis-3",
    )
    assert rows[0]["inference_backend"].startswith("llamacpp:")
    assert rows[0]["phenotype_status"] == "CONFIRMED_POSITIVE"
    assert "urosepsis" in rows[0]["clinical_rationale"]


def test_ensure_gguf_uses_local_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ROSEGOLD_LOCAL_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("ROSEGOLD_LLAMA_GGUF", "tiny.gguf")
    import app.model_store as store

    monkeypatch.setattr(store, "MIN_BYTES", 4)
    (tmp_path / "tiny.gguf").write_bytes(b"gguf")
    assert ensure_llama_gguf().endswith("tiny.gguf")
