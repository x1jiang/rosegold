import os


def test_normalize_audit_aligns_ui_and_api_fields():
    from app.storage import normalize_audit

    api_style = normalize_audit({
        "visit_occurrence_id": 20001,
        "person_id": 1001,
        "reviewer_id": "md_01",
        "adjudication_status": "CONFIRMED_POSITIVE",
        "reviewer_agreement": True,
        "comments": "agree",
    })
    assert api_style["llm_status"] == "CONFIRMED_POSITIVE"
    assert api_style["llm_positive"] is True
    assert api_style["human_positive"] is True

    ui_style = normalize_audit({
        "visit_occurrence_id": 20002,
        "person_id": 1002,
        "llm_status": "CONFIRMED_NEGATIVE",
        "human_decision": "Override: Mark Positive",
        "reviewer_agreement": False,
    })
    assert ui_style["human_positive"] is True
    assert ui_style["llm_positive"] is False


def test_append_and_read_audit(tmp_path, monkeypatch):
    monkeypatch.setenv("ROSEGOLD_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("ROSEGOLD_AUDIT_LOG", str(tmp_path / "human_audit_log.jsonl"))
    from app.storage import append_audit, read_audit, storage_status

    append_audit({
        "visit_occurrence_id": 20001,
        "person_id": 1001,
        "adjudication_status": "CONFIRMED_POSITIVE",
        "reviewer_agreement": True,
        "comments": "persisted",
    })
    rows = read_audit()
    assert len(rows) == 1
    assert rows[0]["comments"] == "persisted"
    assert rows[0]["llm_positive"] is True
    status = storage_status()
    assert status["writable"] is True
    assert status["audit_exists"] is True


def test_save_and_load_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("ROSEGOLD_OUTPUT_DIR", str(tmp_path))
    from app.storage import load_batch_results, save_batch_results

    results = [{
        "person_id": 1001,
        "visit_occurrence_id": 20001,
        "phenotype_status": "CONFIRMED_POSITIVE",
        "condition_present": True,
        "confidence_score": 0.9,
        "clinical_rationale": "test",
        "primary_criteria_met": [],
        "key_evidence": [],
    }]
    paths = save_batch_results(results, "Sepsis / Septic Shock")
    assert os.path.exists(paths["csv"])
    assert os.path.exists(paths["jsonl"])
    assert os.path.exists(paths["omop"])
    loaded = load_batch_results()
    assert loaded[0]["visit_occurrence_id"] == 20001


def test_criteria_persist_and_resolve_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("ROSEGOLD_OUTPUT_DIR", str(tmp_path))
    from app.config_loader import resolve_criteria
    from app.storage import load_criteria, save_criteria

    assert load_criteria() is None
    path = save_criteria("Site-specific qSOFA plus confirmed bacteremia")
    assert os.path.exists(path)
    assert load_criteria() == "Site-specific qSOFA plus confirmed bacteremia"
    assert resolve_criteria("Sepsis / Septic Shock") == "Site-specific qSOFA plus confirmed bacteremia"
    assert resolve_criteria("Sepsis / Septic Shock", "Use this override") == "Use this override"
