from pathlib import Path

import pandas as pd

from app.concordance import calculate_concordance_metrics
from app.engine import AdjudicationEngine
from app.mimic_ext_notes import phenotype_gold
from app.omop_loader import load_omop_data
from app.mimic_ext_simulate import simulate_from_omop

MIMIC_DIR = Path("data/synthetic_mimic_ext_notes")


def test_omop_synthetic_pipeline_adjudicates_sepsis():
    records = load_omop_data("data/synthetic_notes.csv", "data/synthetic_visits.csv")
    assert len(records) == 20
    engine = AdjudicationEngine(model_name="auto")
    results = engine.adjudicate_batch(records, "Sepsis / Septic Shock", "infection plus organ dysfunction")
    by_visit = {int(item["visit_occurrence_id"]): item for item in results}
    assert by_visit[20001]["condition_present"] is True
    assert by_visit[20002]["condition_present"] is False
    assert by_visit[20012]["condition_present"] is True


def test_simulated_mimic_ext_notes_roundtrip(tmp_path):
    paths = simulate_from_omop("data/synthetic_notes.csv", str(tmp_path))
    notes = pd.read_csv(tmp_path / "notes.csv")
    labels = pd.read_csv(paths["labels"])
    records = load_omop_data(str(tmp_path / "notes.csv"), str(tmp_path / "missing.csv"))
    assert len(notes) >= 40
    assert len(records) == 20
    shock = next(item for item in records if item["visit_occurrence_id"] == 20001)
    assert "Septic Shock" in shock["notes_formatted_text"] or "septic shock" in shock["notes_formatted_text"].lower()

    gold = phenotype_gold(notes, labels, "Sepsis / Septic Shock")
    engine = AdjudicationEngine(model_name="auto")
    preds = engine.adjudicate_batch(records, "Sepsis / Septic Shock", "infection plus organ dysfunction")
    merged = gold.merge(
        pd.DataFrame(
            {
                "visit_occurrence_id": [int(item["visit_occurrence_id"]) for item in preds],
                "llm_positive": [bool(item["condition_present"]) for item in preds],
            }
        ),
        on="visit_occurrence_id",
    )
    metrics = calculate_concordance_metrics(merged)
    assert metrics["total_evaluated"] == 20
    assert metrics["tp"] >= 1
    assert metrics["tn"] >= 1
    assert metrics["overall_agreement_pct"] >= 70.0


def test_committed_mimic_simulation_is_loadable():
    assert (MIMIC_DIR / "notes.csv").is_file()
    records = load_omop_data(str(MIMIC_DIR / "notes.csv"), str(MIMIC_DIR / "omop_visits.csv"))
    assert len(records) == 20
    assert all(item["note_count"] >= 1 for item in records)
