import pytest
import os
import json
import pandas as pd
from app.omop_loader import load_omop_data
from app.engine import AdjudicationEngine
pytest.importorskip("instructor")
from app.instructor_engine import InstructorLlamaAdjudicator  # noqa: E402
from app.concordance import calculate_concordance_metrics
from app.omop_export import export_to_omop_observation

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
    "muse-glimmer-30b"
]

PHENOTYPES = [
    "Sepsis / Septic Shock",
    "Acute Ischemic Stroke",
    "Acute Respiratory Distress Syndrome (ARDS)",
    "Acute Kidney Injury (AKI)"
]

def test_full_matrix_models_and_phenotypes():
    records = load_omop_data("data/synthetic_notes.csv", "data/synthetic_visits.csv")
    assert len(records) == 20, f"Expected 20 records, got {len(records)}"

    print(f"\n[Matrix Test] Testing {len(MODELS)} models across {len(PHENOTYPES)} phenotypes on 20 patients...")

    for model in MODELS:
        engine = AdjudicationEngine(model_name=model)
        for pheno in PHENOTYPES:
            results = engine.adjudicate_batch(records, pheno, "Consensus Criteria")
            assert len(results) == 20
            
            # Verify required schema fields on every result
            for r in results:
                assert "person_id" in r
                assert "visit_occurrence_id" in r
                assert "condition_present" in r
                assert "phenotype_status" in r
                assert "confidence_score" in r
                assert "primary_criteria_met" in r
                assert "key_evidence" in r
                assert "clinical_rationale" in r
                assert 0.0 <= r["confidence_score"] <= 1.0

            # Verify OMOP CDM Observation Export for each phenotype
            df_obs = export_to_omop_observation(results, pheno)
            assert len(df_obs) == 20
            assert "observation_concept_id" in df_obs.columns
            assert "value_as_concept_id" in df_obs.columns
            assert df_obs["value_as_concept_id"].isin([4181412, 4188540, 4181413, 45877994]).all()

    print("[Matrix Test] Successfully validated all 12 model-phenotype combinations on 240 patient evaluations!")

def test_instructor_live_validation_across_phenotypes():
    adj = InstructorLlamaAdjudicator()
    records = load_omop_data("data/synthetic_notes.csv", "data/synthetic_visits.csv")

    for pheno in PHENOTYPES:
        for r in records[:5]:
            res = adj.adjudicate(r, pheno, "Standard Clinical Consensus")
            assert res["phenotype_status"] in [
                "CONFIRMED_POSITIVE", "SUSPECTED_PROBABLE", "CONFIRMED_NEGATIVE", "INDETERMINATE_INSUFFICIENT_DATA"
            ]
            assert isinstance(res["clinical_rationale"], str)
            assert len(res["clinical_rationale"]) > 10

def test_calibration_and_metrics_integrity():
    records = load_omop_data("data/synthetic_notes.csv", "data/synthetic_visits.csv")
    engine = AdjudicationEngine(model_name="muse-glimmer-30b")
    results = engine.adjudicate_batch(records, "Sepsis / Septic Shock", "Consensus Criteria")

    # Build evaluation DataFrame
    eval_rows = []
    for r, res in zip(records, results):
        # Known ground truth from synthetic cohort generator
        text = r['notes_formatted_text'].lower()
        true_pos = "septic shock" in text or "lactate 4.2" in text or "urosepsis" in text or "mssa bacteremia" in text
        eval_rows.append({
            "visit_occurrence_id": r['visit_occurrence_id'],
            "person_id": r['person_id'],
            "human_positive": true_pos,
            "llm_positive": res['condition_present']
        })

    df_eval = pd.DataFrame(eval_rows)
    metrics = calculate_concordance_metrics(df_eval)

    assert metrics["total_evaluated"] == 20
    assert metrics["cohens_kappa"] > 0.70 # Strong calibration concordance
    assert metrics["sensitivity"] >= 80.0
    assert metrics["specificity"] >= 80.0
    assert (metrics["tp"] + metrics["fp"] + metrics["fn"] + metrics["tn"]) == 20
