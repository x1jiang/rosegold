import pytest
import pandas as pd
from app.concordance import calculate_concordance_metrics
from app.omop_export import export_to_omop_observation

def test_concordance_metrics_perfect_agreement():
    df = pd.DataFrame({
        'human_positive': [True, True, False, False],
        'llm_positive': [True, True, False, False]
    })
    metrics = calculate_concordance_metrics(df)
    assert metrics['overall_agreement_pct'] == 100.0
    assert metrics['cohens_kappa'] == 1.0
    assert metrics['sensitivity'] == 100.0
    assert metrics['specificity'] == 100.0

def test_concordance_metrics_with_disagreement():
    df = pd.DataFrame({
        'human_positive': [True, True, False, False],
        'llm_positive': [True, False, False, True] # 1 FN, 1 FP
    })
    metrics = calculate_concordance_metrics(df)
    assert metrics['overall_agreement_pct'] == 50.0
    assert metrics['tp'] == 1
    assert metrics['fn'] == 1
    assert metrics['fp'] == 1
    assert metrics['tn'] == 1

def test_omop_observation_export():
    adjudications = [
        {
            "person_id": 1001,
            "visit_occurrence_id": 20001,
            "phenotype_status": "CONFIRMED_POSITIVE",
            "confidence_score": 0.96,
            "clinical_rationale": "Sepsis secondary to Klebsiella urosepsis."
        },
        {
            "person_id": 1002,
            "visit_occurrence_id": 20002,
            "phenotype_status": "CONFIRMED_NEGATIVE",
            "confidence_score": 0.98,
            "clinical_rationale": "Elective surgery without infection."
        }
    ]
    df_obs = export_to_omop_observation(adjudications, "Sepsis / Septic Shock")
    assert len(df_obs) == 2
    assert "observation_concept_id" in df_obs.columns
    assert "observation_type_concept_id" in df_obs.columns
    assert df_obs.iloc[0]["value_as_concept_id"] == 4181412 # Present
    assert df_obs.iloc[1]["value_as_concept_id"] == 4188540 # Absent
