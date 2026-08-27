from app.config_loader import criteria_for, pipeline_settings, resolve_criteria
from app.omop_loader import load_omop_data, load_visit_index
from app.prompts import build_chat_prompt


def test_pipeline_settings_and_phenotype_criteria(tmp_path, monkeypatch):
    monkeypatch.setenv("ROSEGOLD_OUTPUT_DIR", str(tmp_path))
    settings = pipeline_settings()
    assert settings["max_notes_per_visit"] >= 1
    assert settings["enable_prefix_caching"] in {True, False, "true", "false"}
    sepsis = criteria_for("Sepsis / Septic Shock")
    assert "infection" in sepsis.lower() or "organ" in sepsis.lower()
    stroke = criteria_for("Acute Ischemic Stroke")
    assert "neurolog" in stroke.lower() or "infarct" in stroke.lower()
    assert resolve_criteria("Sepsis / Septic Shock", "Consensus Criteria") == sepsis
    assert resolve_criteria("Sepsis / Septic Shock", "Custom site rule") == "Custom site rule"


def test_chat_templates_match_model_family():
    llama = build_chat_prompt("hello", "meta-llama/Llama-3.1-8B-Instruct")
    gemma = build_chat_prompt("hello", "google/gemma-2-9b-it")
    assert "<|start_header_id|>" in llama
    assert "<start_of_turn>" in gemma
    assert "<|start_header_id|>" not in gemma


def test_omop_loader_filters_and_caches():
    one = load_omop_data(
        "data/synthetic_notes.csv",
        "data/synthetic_visits.csv",
        target_visits=[20001],
    )
    assert len(one) == 1
    assert one[0]["visit_occurrence_id"] == 20001
    assert one[0]["note_count"] >= 1

    again = load_omop_data(
        "data/synthetic_notes.csv",
        "data/synthetic_visits.csv",
        target_visits=[20001],
    )
    assert again[0]["notes_formatted_text"] == one[0]["notes_formatted_text"]

    index = load_visit_index("data/synthetic_notes.csv", "data/synthetic_visits.csv")
    assert len(index) >= 5
    match = next(item for item in index if item["visit_occurrence_id"] == 20001)
    assert match["note_count"] == one[0]["note_count"]
