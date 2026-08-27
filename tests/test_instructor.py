import pytest
from app.schemas import RoseGoldAdjudication, ClinicalEvidence
from app.instructor_engine import InstructorLlamaAdjudicator
from pydantic import ValidationError

def test_instructor_schema_validation():
    # Test valid model
    valid_data = {
        "clinical_rationale": "Patient presented with severe septic shock requiring vasopressors.",
        "primary_criteria_met": ["SIRS >= 2", "Lactate > 4.0"],
        "key_evidence": [
            {
                "note_id": 101,
                "note_date": "2026-03-01",
                "evidence_quote": "Patient admitted with fever, tachycardia, hypotension refractory to initial fluids.",
                "interpretation": "Shock criteria met"
            }
        ],
        "phenotype_status": "CONFIRMED_POSITIVE",
        "condition_present": True,
        "confidence_score": 0.95
    }
    model = RoseGoldAdjudication(**valid_data)
    assert model.condition_present is True
    assert model.confidence_score == 0.95

def test_instructor_schema_catches_invalid_status():
    invalid_data = {
        "clinical_rationale": "Test rationale",
        "phenotype_status": "MAYBE_POSITIVE", # Invalid enum
        "condition_present": True,
        "confidence_score": 0.95
    }
    with pytest.raises(ValidationError):
        RoseGoldAdjudication(**invalid_data)

def test_instructor_schema_syncs_condition_present():
    # If phenotype is CONFIRMED_NEGATIVE but condition_present was set True by mistake, validator fixes it
    corrected_data = {
        "clinical_rationale": "No findings of sepsis in chart.",
        "phenotype_status": "CONFIRMED_NEGATIVE",
        "condition_present": True, # Inconsistent
        "confidence_score": 0.98
    }
    model = RoseGoldAdjudication(**corrected_data)
    assert model.condition_present is False # Auto-corrected

def test_instructor_adjudicator_execution():
    adj = InstructorLlamaAdjudicator()
    dummy_record = {
        "person_id": 1001,
        "visit_occurrence_id": 20001,
        "visit_start_date": "2026-03-01",
        "visit_end_date": "2026-03-08",
        "notes_formatted_text": "Patient admitted with fever and lactate 4.2 in septic shock."
    }
    res = adj.adjudicate(dummy_record, "Sepsis", "Consensus Criteria")
    assert res['condition_present'] is True
    assert res['phenotype_status'] == "CONFIRMED_POSITIVE"
    assert len(res['key_evidence']) > 0
