import os
import json
import tempfile
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.concordance import calculate_concordance_metrics
from app.omop_loader import load_omop_data, load_visit_index
from app.api import app


@pytest.fixture
def api_client():
    return TestClient(app)


def test_concordance_empty_and_missing():
    # Empty DataFrame
    res = calculate_concordance_metrics(pd.DataFrame())
    assert res["total_evaluated"] == 0
    assert res["cohens_kappa"] == 0.0

    # Missing expected columns
    df_wrong = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
    res = calculate_concordance_metrics(df_wrong)
    assert res["total_evaluated"] == 0
    assert res["cohens_kappa"] == 0.0


def test_concordance_homogeneous_and_nans():
    # All true positive (pe == 1.0, po == 1.0) -> Kappa should be 1.0
    df_all_pos = pd.DataFrame({
        "human_positive": [True, True, True, True],
        "llm_positive": [True, True, True, True]
    })
    res = calculate_concordance_metrics(df_all_pos)
    assert res["total_evaluated"] == 4
    assert res["cohens_kappa"] == 1.0
    assert res["overall_agreement_pct"] == 100.0

    # Contains NaNs
    df_nans = pd.DataFrame({
        "human_positive": [True, False, None, True],
        "llm_positive": [True, False, True, None]
    })
    res = calculate_concordance_metrics(df_nans)
    assert res["total_evaluated"] == 2
    assert res["overall_agreement_pct"] == 100.0


def test_omop_loader_dirty_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        notes_csv = os.path.join(tmpdir, "dirty_notes.csv")
        visits_csv = os.path.join(tmpdir, "dirty_visits.csv")

        # Dirty notes with NaN visit IDs, empty text, whitespace text
        df_notes = pd.DataFrame({
            "visit_occurrence_id": [101, None, 102, 103, "invalid"],
            "note_text": ["Patient with sepsis", "Some text", "   ", "Stable post-op", "Notes"],
            "note_date": ["2026-03-01", "2026-03-01", "2026-03-02", "2026-03-02", "2026-03-03"],
            "note_id": [1, 2, 3, 4, 5],
        })
        df_notes.to_csv(notes_csv, index=False)

        # Dirty visits with NaN IDs and trailing blanks
        df_visits = pd.DataFrame({
            "visit_occurrence_id": [101, 102, None, 103],
            "person_id": [1, 2, 3, 4],
            "visit_start_date": ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04"],
            "visit_end_date": ["2026-03-05", "2026-03-06", "2026-03-07", "2026-03-08"],
        })
        df_visits.to_csv(visits_csv, index=False)

        # load_visit_index should not crash
        index = load_visit_index(notes_csv, visits_csv)
        assert len(index) == 3
        assert {v["visit_occurrence_id"] for v in index} == {101, 102, 103}

        # load_omop_data should not crash
        records = load_omop_data(notes_csv, visits_csv)
        assert len(records) == 3
        # Visit 101 should have 1 note
        v101 = next(r for r in records if r["visit_occurrence_id"] == 101)
        assert v101["note_count"] == 1
        assert "Patient with sepsis" in v101["notes_formatted_text"]


def test_adjudicator_cli_and_checkpoint_resumption():
    import subprocess
    import sys

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = os.path.join(tmpdir, "test_out.csv")
        out_jsonl = os.path.join(tmpdir, "test_out.jsonl")

        cmd = [
            sys.executable, "-m", "app.adjudicator",
            "--notes_path", "data/synthetic_notes.csv",
            "--visits_path", "data/synthetic_visits.csv",
            "--output_path", out_csv,
            "--backend", "keyword_rules",
            "--clinical_criteria", "1. Documented infection.\n2. Organ dysfunction.",
            "--chunk_size", "2"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"CLI execution failed: {res.stderr}"
        assert os.path.exists(out_csv)
        assert os.path.exists(out_jsonl)

        df = pd.read_csv(out_csv)
        assert len(df) > 0
        assert "phenotype_status" in df.columns

        # Verify resumption: run again, it should resume and recognize done IDs
        res2 = subprocess.run(cmd, capture_output=True, text=True)
        assert res2.returncode == 0
        assert "Resuming from checkpoint" in res2.stdout


def test_api_hardening(api_client):
    # Health endpoint
    res = api_client.get("/health")
    assert res.status_code == 200

    # Single adjudication with raw text
    res = api_client.post("/api/adjudicate/single", json={
        "notes_formatted_text": "Patient admitted with fever 39C, severe sepsis secondary to pneumonia.",
        "target_condition": "Sepsis / Septic Shock"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["condition_present"] is True

    # Single adjudication missing both ID and text
    res = api_client.post("/api/adjudicate/single", json={
        "target_condition": "Sepsis / Septic Shock"
    })
    assert res.status_code == 400

    # Batch with nonexistent visit IDs
    res = api_client.post("/api/adjudicate/batch", json={
        "visit_occurrence_ids": [99999999],
        "target_condition": "Sepsis / Septic Shock"
    })
    assert res.status_code == 400


def test_singularity_definition_files():
    for def_file in ["singularity/rosegold.def", "singularity/rosegold_cpu.def"]:
        assert os.path.exists(def_file)
        with open(def_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Bootstrap:" in content
            assert "%files" in content
            assert "%environment" in content
            assert "%post" in content
            assert "%runscript" in content


def test_schema_coercion_and_normalization():
    from app.schemas import RoseGoldAdjudication, ClinicalEvidence

    ev = ClinicalEvidence(note_id="Note 40012", evidence_quote="Lactate elevated at 4.2 mmol/L")
    assert ev.note_id == 40012

    ev_none = ClinicalEvidence(note_id="N/A", evidence_quote="Hypotension requiring pressors")
    assert ev_none.note_id is None

    adj = RoseGoldAdjudication(
        clinical_rationale="Confirmed sepsis with organ dysfunction.",
        primary_criteria_met=["Infection", "Lactate > 2"],
        key_evidence=[
            {"note_id": "Note #101", "evidence_quote": "Valid verbatim clinical quote here"},
            {"note_id": 102, "evidence_quote": "no"},  # too short (<5 chars), should be filtered out
            "Direct string evidence quote from chart",
        ],
        phenotype_status="positive",
        condition_present=True,
        confidence_score="94%",
    )
    assert adj.phenotype_status == "CONFIRMED_POSITIVE"
    assert adj.confidence_score == 0.94
    assert len(adj.key_evidence) == 2
    assert adj.key_evidence[0].note_id == 101
    assert adj.key_evidence[1].evidence_quote == "Direct string evidence quote from chart"


def test_parse_model_json_resilience():
    from app.vertex_engine import _parse_model_json

    messy_output = """Here is my clinical chart review adjudication:
```json
{
  "reasoning": "Patient has documented Klebsiella bacteremia and septic shock.",
  "primary_criteria_met": ["Infection", "Vasopressor requirement",],
  "key_evidence": [
    {
      "note_id": "40001",
      "evidence_quote": "Klebsiella bacteremia and septic shock on norepinephrine",
    },
  ],
  "phenotype_status": "probable",
  "condition_present": true,
  "confidence_score": 88,
}
```
Hope this helps!"""
    parsed = _parse_model_json(messy_output)
    assert parsed.phenotype_status == "SUSPECTED_PROBABLE"
    assert parsed.confidence_score == 0.88
    assert "Klebsiella bacteremia" in parsed.clinical_rationale
    assert parsed.key_evidence[0].note_id == 40001


def test_clinical_rules_empty_notes_and_custom_phenotype():
    from app.clinical_rules import adjudicate_clinical_rules

    empty_rec = {
        "person_id": 1,
        "visit_occurrence_id": 100,
        "notes_formatted_text": "[No notes found for this visit encounter]",
    }
    res_empty = adjudicate_clinical_rules([empty_rec], "Sepsis / Septic Shock")[0]
    assert res_empty["phenotype_status"] == "INDETERMINATE_INSUFFICIENT_DATA"
    assert res_empty["condition_present"] is False
    assert res_empty["confidence_score"] == 0.0

    pe_pos_rec = {
        "person_id": 2,
        "visit_occurrence_id": 200,
        "notes_formatted_text": "[Note 1 | Date: 2026-03-01] CTA chest demonstrates acute segmental pulmonary embolism in the right lower lobe.",
    }
    res_pe_pos = adjudicate_clinical_rules(
        [pe_pos_rec],
        "Pulmonary Embolism",
        clinical_criteria="1. CTA chest with filling defect\n2. Acute pulmonary embolism",
    )[0]
    assert res_pe_pos["condition_present"] is True
    assert res_pe_pos["phenotype_status"] in ("CONFIRMED_POSITIVE", "SUSPECTED_PROBABLE")
    assert len(res_pe_pos["key_evidence"]) >= 1

    pe_neg_rec = {
        "person_id": 3,
        "visit_occurrence_id": 300,
        "notes_formatted_text": "[Note 2 | Date: 2026-03-02] CTA chest negative for pulmonary embolism. No acute cardiopulmonary process.",
    }
    res_pe_neg = adjudicate_clinical_rules(
        [pe_neg_rec],
        "Pulmonary Embolism",
        clinical_criteria="1. CTA chest with filling defect\n2. Acute pulmonary embolism",
    )[0]
    assert res_pe_neg["condition_present"] is False
    assert res_pe_neg["phenotype_status"] == "CONFIRMED_NEGATIVE"


def test_evidence_quote_grounding_filters_hallucinations():
    from app.clinical_rules import verify_and_ground_adjudication

    rec = {
        "person_id": 10,
        "visit_occurrence_id": 500,
        "notes_formatted_text": "[Note 9001 | Date: 2026-03-05] Patient in septic shock with lactate 4.5 mmol/L on norepinephrine.",
    }
    adjudication = {
        "person_id": 10,
        "visit_occurrence_id": 500,
        "condition_present": True,
        "phenotype_status": "CONFIRMED_POSITIVE",
        "confidence_score": 0.95,
        "primary_criteria_met": ["Septic shock"],
        "key_evidence": [
            {
                "note_id": 9001,
                "note_date": "2026-03-05",
                "evidence_quote": "Completely fabricated quote that does not exist anywhere in the chart",
                "interpretation": "Hallucinated evidence",
            },
            {
                "note_id": 9001,
                "note_date": "2026-03-05",
                "evidence_quote": "Patient in septic shock with lactate 4.5 mmol/L on norepinephrine.",
                "interpretation": "Grounded evidence",
            },
        ],
        "clinical_rationale": "Documented septic shock.",
    }
    grounded = verify_and_ground_adjudication(adjudication, rec, "Sepsis / Septic Shock")
    assert len(grounded["key_evidence"]) == 1
    assert "lactate 4.5" in grounded["key_evidence"][0]["evidence_quote"]

