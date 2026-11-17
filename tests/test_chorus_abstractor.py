"""Synthetic unit and CLI tests for app.chorus_abstractor."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.chorus_abstractor import (
    DischargeGateOutput,
    FieldEvidenceItem,
    SepsisChartBundleOutput,
    StructuredStageCaller,
    UncertainGateOutput,
    abstract_sepsis_cohort,
    abstract_sepsis_encounter,
    audit_prompt_dictionary,
    compute_sepsis_structured_facts,
    load_chorus_encounters,
    load_chorus_instrument,
    main as abstractor_main,
    normalize_surg_confounders,
    scrub_calendar_dates,
)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None, encoding: str = "utf-8") -> Path:
    cols = fieldnames or list(rows[0].keys())
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _synthetic_instrument(tmp_path: Path) -> Path:
    cols = [
        "Variable / Field Name",
        "Form Name",
        "Section Header",
        "Field Type",
        "Field Label",
        "Choices, Calculations, OR Slider Labels",
        "Branching Logic (Show field only if...)",
    ]
    rows = [
        {"Variable / Field Name": "record_id", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "Person ID", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": ""},
        {"Variable / Field Name": "institution_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "dropdown", "Field Label": "Site", "Choices, Calculations, OR Slider Labels": "99, SYNTHETIC-SITE", "Branching Logic (Show field only if...)": ""},
        {"Variable / Field Name": "reviewer_id", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "Reviewer", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": ""},
        {"Variable / Field Name": "vid", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "Visit ID", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": ""},
        {"Variable / Field Name": "t0", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "ICU admission time", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": ""},
        {"Variable / Field Name": "concept_1_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "checkbox", "Field Label": "Discharge concepts", "Choices, Calculations, OR Slider Labels": "1, Sepsis | 10, None of the above", "Branching Logic (Show field only if...)": ""},
        {"Variable / Field Name": "sepsis_ync_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "radio", "Field Label": "Supports sepsis?", "Choices, Calculations, OR Slider Labels": "1, YES | 2, NO | 3, I am not sure", "Branching Logic (Show field only if...)": ""},
        {"Variable / Field Name": "sepsis_uncertain_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "radio", "Field Label": "Evidence in H&P/Labs/Micro/Meds?", "Choices, Calculations, OR Slider Labels": "1, YES | 2, NO", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 3"},
        {"Variable / Field Name": "sep1_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "checkbox", "Field Label": "Systemic signs", "Choices, Calculations, OR Slider Labels": "1, Temperature: >38C or <36C | 2, Heart Rate >90 | 3, Respiratory Rate >20 or PaCO2 <32 | 4, WBC >12,000 or <4,000 or >10% bands", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
        {"Variable / Field Name": "sep3_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "Culture", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
        {"Variable / Field Name": "sep4_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "yesno", "Field Label": "Antibiotics >24h", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
        {"Variable / Field Name": "sep5_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "Other unstructured evidence", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
        {"Variable / Field Name": "sep6_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "Highest lactate", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
        {"Variable / Field Name": "sep_surg_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "checkbox", "Field Label": "Surgery / trauma confounders", "Choices, Calculations, OR Slider Labels": "1, Yes - major surgery and the patient is <48 hours from the surgery | 2, Yes - there is a comment on vasoplegia | 3, Yes - active burn care / trauma care with <72 hours from the admission | 4, No | 9, Unclear to me given notes", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
        {"Variable / Field Name": "sep7_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "Daily SOFA scores", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
        {"Variable / Field Name": "sep_day_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "SOFA rise >2 day", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
        {"Variable / Field Name": "sep_dt_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "Sepsis time", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
        {"Variable / Field Name": "sep_days", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "Sepsis days", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
        {"Variable / Field Name": "sep_time", "Form Name": "chorus", "Section Header": "", "Field Type": "calc", "Field Label": "Interval", "Choices, Calculations, OR Slider Labels": 'datediff([t0],[sep_dt_v3],"m","mdy",true)', "Branching Logic (Show field only if...)": ""},
        {"Variable / Field Name": "conf1", "Form Name": "chorus", "Section Header": "", "Field Type": "radio", "Field Label": "Confidence", "Choices, Calculations, OR Slider Labels": "1, 1 | 5, 5", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
    ]
    return _write_csv(tmp_path / "instrument.csv", rows, fieldnames=cols)


def test_audit_prompt_dictionary_catches_code_and_gate_defects(tmp_path: Path):
    inst = _synthetic_instrument(tmp_path)
    prompt_csv = _write_csv(
        tmp_path / "draft_prompts.csv",
        [
            {"QID": "Q1", "Code": "sepsis_ync_v3", "Type": "Concept gate", "UI": "radio", "Question": "Supports sepsis?", "Prompt": 'Do you think "[DATA]" supports SEPSIS, respond with a YES or NO'},
            {"QID": "Q3.1", "Code": "sep1_v3___1", "Type": "Criterion", "UI": "checkbox", "Question": "Temp > 38?C (100.4?F)", "Prompt": 'In "[DATA]" was temp >100.4F, respond with a YES or NO'},
            {"QID": "Q3.4", "Code": "sep1_v3___4", "Type": "Criterion", "UI": "checkbox", "Question": "WBC > 12,000/‘L", "Prompt": 'In "[DATA]" was WBC > 12,000/‘L, respond with a YES or NO'},
            {"QID": "Q8.4", "Code": "sep_surg_v3___4", "Type": "Criterion", "UI": "checkbox", "Question": "major surgery related burn care", "Prompt": ""},
            {"QID": "Q8.9", "Code": "sep_surg_v3___9", "Type": "Criterion", "UI": "checkbox", "Question": "major surgery related trauma care", "Prompt": ""},
            {"QID": "Q9", "Code": "sep7_v3", "Type": "Criterion", "UI": "text", "Question": "SOFA scores", "Prompt": ""},
            {"QID": "Q10", "Code": "sep_day_v3", "Type": "Timing", "UI": "text", "Question": "SOFA rise day", "Prompt": ""},
            {"QID": "Q11", "Code": "sep_dt_v3", "Type": "Timing", "UI": "text", "Question": "Time", "Prompt": ""},
            {"QID": "Q12", "Code": "sep_days", "Type": "Timing", "UI": "text", "Question": "Days", "Prompt": ""},
            {"QID": "Q14", "Code": "conf1", "Type": "Confidence", "UI": "radio", "Question": "Confidence 1?5", "Prompt": ""},
        ],
        encoding="cp1252",
    )
    findings = audit_prompt_dictionary(prompt_csv, inst)
    issue_map = {(f["code"], f["issue_type"]) for f in findings}
    assert ("sepsis_ync_v3", "forced_binary_on_three_way_gate") in issue_map
    assert ("sep1_v3___1", "missing_unknown_option") in issue_map
    assert ("sep_surg_v3___4", "choice_code_mismatch") in issue_map
    assert ("sep_surg_v3___9", "choice_code_mismatch") in issue_map
    assert ("sep1_v3___4", "encoding_mojibake") in issue_map
    assert ("sep7_v3", "unhandled_quantitative_or_temporal_field") in issue_map
    assert ("sep_days", "unhandled_quantitative_or_temporal_field") in issue_map
    assert ("conf1", "human_only_field") in issue_map


def test_structured_calculators_sirs_cultures_abx_sofa_and_dates(tmp_path: Path):
    notes = _write_csv(tmp_path / "notes.csv", [
        {"note_id": 1, "visit_occurrence_id": 501, "note_datetime": "2026-03-01 23:00:00", "note_title": "ED Note", "note_text": "Feb 15th BP 118/74. Fever may indicate infection."},
        {"note_id": 2, "visit_occurrence_id": 501, "note_datetime": "2026-03-03 10:00:00", "note_title": "Discharge Summary", "note_text": "Severe sepsis resolved."},
    ])
    visits = _write_csv(tmp_path / "visits.csv", [
        {"visit_occurrence_id": 501, "person_id": 101, "record_id": "SYN-01", "institution_v3": "99", "icu_admit_datetime": "2026-03-01 23:00:00"},
    ])
    meas = _write_csv(tmp_path / "meas.csv", [
        {"visit_occurrence_id": 501, "measurement_datetime": "2026-03-02 02:00:00", "measurement_name": "temperature", "value_as_number": 101.5, "unit": "F"},
        {"visit_occurrence_id": 501, "measurement_datetime": "2026-03-02 02:15:00", "measurement_name": "heart_rate", "value_as_number": 84, "unit": "bpm"},
        {"visit_occurrence_id": 501, "measurement_datetime": "2026-03-02 02:15:00", "measurement_name": "respiratory_rate", "value_as_number": 16, "unit": "bpm"},
        {"visit_occurrence_id": 501, "measurement_datetime": "2026-03-02 02:15:00", "measurement_name": "wbc", "value_as_number": 14800, "unit": "/uL"},
        {"visit_occurrence_id": 501, "measurement_datetime": "2026-03-01 23:30:00", "measurement_name": "platelets", "value_as_number": 180, "unit": "10^3/uL"},
        {"visit_occurrence_id": 501, "measurement_datetime": "2026-03-01 23:30:00", "measurement_name": "creatinine", "value_as_number": 1.0, "unit": "mg/dL"},
        {"visit_occurrence_id": 501, "measurement_datetime": "2026-03-02 08:00:00", "measurement_name": "platelets", "value_as_number": 90, "unit": "10^3/uL"},
        {"visit_occurrence_id": 501, "measurement_datetime": "2026-03-02 08:00:00", "measurement_name": "creatinine", "value_as_number": 2.1, "unit": "mg/dL"},
        {"visit_occurrence_id": 501, "measurement_datetime": "2026-03-02 08:00:00", "measurement_name": "lactate", "value_as_number": 3.7, "unit": "mmol/L"},
    ])
    meds = _write_csv(tmp_path / "meds.csv", [
        {"visit_occurrence_id": 501, "drug_start_datetime": "2026-03-02 03:00:00", "drug_end_datetime": "2026-03-02 07:00:00", "drug_name": "Vancomycin", "route": "IV", "dose_rate": ""},
        {"visit_occurrence_id": 501, "drug_start_datetime": "2026-03-03 19:00:00", "drug_end_datetime": "2026-03-04 12:00:00", "drug_name": "Cefepime", "route": "IV", "dose_rate": ""},
    ])
    cult = _write_csv(tmp_path / "cult.csv", [
        {"visit_occurrence_id": 501, "collection_datetime": "2026-03-02 02:30:00", "source": "Blood", "organism": "Escherichia coli", "result": "(+)"},
    ])
    enc = load_chorus_encounters(notes, visits, meas, meds, cult)[0]
    facts = compute_sepsis_structured_facts(enc)

    assert facts["t0"] == "23:00"
    assert facts["sep1_v3___1"] == "1"
    assert facts["sep1_v3___2"] == "0"
    assert facts["sep1_v3___3"] == "0"
    assert facts["sep1_v3___4"] == "1"
    assert facts["sep3_v3"] == "Blood, Escherichia coli, (+)"
    assert facts["sep4_v3"] == "1"  # 36h vancomycin trough pause bridged cleanly
    assert facts["sep6_v3"] == "3.7"
    assert facts["sep7_v3"] == "0, 4"
    assert facts["sep_day_v3"] == "1"
    assert facts["sep_dt_v3"] == "02:00"
    assert facts["sep_days"] == "1"
    assert facts["sep_time"] == "180"

    # Unmeasured RR and PaCO2 stay None (UNKNOWN) rather than 0
    enc_sparse = {"icu_admit_dt": enc["icu_admit_dt"], "measurements": [], "medications": [{"drug_start_datetime": "2026-03-02 01:00:00", "drug_end_datetime": "2026-03-02 03:00:00", "drug_name": "Cefazolin"}], "cultures": [{"collection_datetime": "2026-03-02 01:00:00", "source": "Blood", "organism": "None", "result": "No growth"}]}
    sparse_facts = compute_sepsis_structured_facts(enc_sparse)
    assert sparse_facts["sep1_v3___3"] is None
    assert sparse_facts["sep3_v3"] == "NO"
    assert sparse_facts["sep4_v3"] == "0"  # 2h single surgical prophylaxis <= 24h

    # Date scrubbing removes calendar dates while keeping clinical phrases intact
    scrubbed = scrub_calendar_dates("Admitted 2026-03-01 and Feb 15th, 03/05/2026; BP 118/74 may indicate sepsis.")
    assert "2026-03-01" not in scrubbed
    assert "Feb 15th" not in scrubbed
    assert "03/05/2026" not in scrubbed
    assert "BP 118/74" in scrubbed
    assert "may indicate sepsis" in scrubbed


def test_staged_abstraction_branching_grounding_and_fail_safe(tmp_path: Path):
    notes = _write_csv(tmp_path / "notes.csv", [
        {"note_id": 10, "visit_occurrence_id": 601, "note_datetime": "2026-04-01 08:00:00", "note_title": "ED Note", "note_text": "Post-op anastomotic leak, T 39.0C, HR 118, WBC 18.2."},
        {"note_id": 11, "visit_occurrence_id": 602, "note_datetime": "2026-04-02 09:00:00", "note_title": "Discharge Summary", "note_text": "Elective procedure, no sepsis."},
    ])
    visits = _write_csv(tmp_path / "visits.csv", [
        {"visit_occurrence_id": 601, "person_id": 201, "record_id": "SYN-A", "institution_v3": "99", "icu_admit_datetime": "2026-04-01 08:00:00"},
        {"visit_occurrence_id": 602, "person_id": 202, "record_id": "SYN-B", "institution_v3": "99", "icu_admit_datetime": "2026-04-02 09:00:00"},
    ])
    encounters = load_chorus_encounters(notes, visits)

    assert normalize_surg_confounders(["1", "4", "9"]) == ["1"]
    assert normalize_surg_confounders(["4", "9"]) == ["9"]

    def fake_caller(stage_name, sys_p, usr_p, schema_cls, enc):
        if enc["vid"] == "602" and stage_name == "stage1":
            return DischargeGateOutput(sepsis_ync_v3="2", concept_1_v3_sepsis=False, rationale="No sepsis.")
        if enc["vid"] == "601" and stage_name == "stage1":
            return DischargeGateOutput(sepsis_ync_v3="3", concept_1_v3_sepsis=False, rationale="Missing discharge summary.")
        if enc["vid"] == "601" and stage_name == "stage2":
            return UncertainGateOutput(sepsis_uncertain_v3="1", rationale="ED note shows leak and fever.")
        return SepsisChartBundleOutput(
            sep1_temp="YES",
            sep1_hr="YES",
            sep1_rr_paco2="UNKNOWN",
            sep1_wbc_bands="YES",
            sep5_unstructured_evidence="Post-op anastomotic leak documented on 2026-04-01",
            sep_surg_confounders=["1", "4"],
            overall_phenotype_status="CONFIRMED_POSITIVE",
            confidence_score=0.92,
            rationale="Confirmed post-op sepsis.",
            field_evidence=[
                FieldEvidenceItem(field_name="sep5_v3", note_id=10, evidence_quote="Post-op anastomotic leak, T 39.0C, HR 118, WBC 18.2."),
                FieldEvidenceItem(field_name="sep5_v3", note_id=10, evidence_quote="Completely fabricated non-existent clinical quote xyz"),
            ],
        )

    res_a = abstract_sepsis_encounter(encounters[0], StructuredStageCaller(stage_caller=fake_caller))
    row_a = res_a["redcap_row"]
    assert row_a["sepsis_ync_v3"] == "3"
    assert row_a["sepsis_uncertain_v3"] == "1"
    assert row_a["sep1_v3___1"] == "1"
    assert row_a["sep1_v3___3"] == ""  # UNKNOWN stays blank
    assert row_a["sep_surg_v3___1"] == "1"
    assert row_a["sep_surg_v3___4"] == "0"
    assert "2026-04-01" not in row_a["sep5_v3"]
    assert row_a["conf1"] == "" and row_a["reviewer_id"] == ""
    quotes_a = [e["evidence_quote"] for e in res_a["evidence_rows"]]
    assert "Completely fabricated non-existent clinical quote xyz" not in quotes_a

    res_b = abstract_sepsis_encounter(encounters[1], StructuredStageCaller(stage_caller=fake_caller))
    assert res_b["redcap_row"]["sepsis_ync_v3"] == "2"
    assert res_b["redcap_row"]["sepsis_uncertain_v3"] == ""
    assert res_b["redcap_row"]["sep1_v3___1"] == ""

    def failing_caller(*args):
        raise RuntimeError("simulated transport timeout")

    res_fail = abstract_sepsis_encounter(encounters[0], StructuredStageCaller(stage_caller=failing_caller))
    assert res_fail["adjudication"]["phenotype_status"] == "INDETERMINATE_INSUFFICIENT_DATA"
    assert res_fail["redcap_row"]["sepsis_ync_v3"] == ""


def test_cli_end_to_end_rules_and_simulated_hosted_json(tmp_path: Path, monkeypatch):
    inst = _synthetic_instrument(tmp_path)
    notes = _write_csv(tmp_path / "notes.csv", [
        {"note_id": 1, "visit_occurrence_id": 701, "note_datetime": "2026-05-01 10:00:00", "note_title": "Discharge Summary", "note_text": "Klebsiella urosepsis with septic shock. Lactate 4.1."},
        {"note_id": 2, "visit_occurrence_id": 702, "note_datetime": "2026-05-02 10:00:00", "note_title": "Discharge Summary", "note_text": "Elective cataract repair. No infection or sepsis."},
    ])
    visits = _write_csv(tmp_path / "visits.csv", [
        {"visit_occurrence_id": 701, "person_id": 301, "record_id": "SYN-701", "institution_v3": "99", "icu_admit_datetime": "2026-05-01 10:00:00"},
        {"visit_occurrence_id": 702, "person_id": 302, "record_id": "SYN-702", "institution_v3": "99", "icu_admit_datetime": "2026-05-02 10:00:00"},
    ])
    out_dir = tmp_path / "out_mock"
    monkeypatch.setenv("ROSEGOLD_LLM_BACKEND", "keyword_rules")
    monkeypatch.setenv("ROSEGOLD_ALLOW_MOCK", "1")
    rc = abstractor_main(["--notes", str(notes), "--visits", str(visits), "--dictionary", str(inst), "--output-dir", str(out_dir)])
    assert rc == 0
    assert (out_dir / "redcap_import.csv").exists()
    assert (out_dir / "field_evidence.csv").exists()
    manifest = json.loads((out_dir / "run_manifest.json").read_text())
    assert manifest["encounters_processed"] == 2

    # Simulated hosted OpenAI/Muse transport without network
    import requests

    monkeypatch.setenv("ROSEGOLD_LLM_BACKEND", "hybrid")
    monkeypatch.setenv("ROSEGOLD_MUSE_URL", "https://muse.internal/v1/chat/completions")
    monkeypatch.setenv("ROSEGOLD_MUSE_MODEL", "Muse-Synthetic")
    monkeypatch.setenv("ROSEGOLD_ALLOW_MOCK", "0")

    def fake_post(self, url, **kwargs):
        user_msg = kwargs["json"]["messages"][1]["content"]
        class Resp:
            status_code = 200
            def json(self_inner):
                if "--- Discharge Summary ---" in user_msg and "Klebsiella" in user_msg:
                    content = json.dumps({"sepsis_ync_v3": "1", "concept_1_v3_sepsis": True, "evidence_quotes": [{"field_name": "sepsis_ync_v3", "note_id": 1, "evidence_quote": "Klebsiella urosepsis with septic shock"}], "rationale": "Positive."})
                elif "--- Discharge Summary ---" in user_msg:
                    content = json.dumps({"sepsis_ync_v3": "2", "concept_1_v3_sepsis": False, "evidence_quotes": [], "rationale": "Negative."})
                else:
                    content = json.dumps({"sep1_temp": "UNKNOWN", "sep1_hr": "UNKNOWN", "sep1_rr_paco2": "UNKNOWN", "sep1_wbc_bands": "UNKNOWN", "sep5_unstructured_evidence": "Klebsiella urosepsis", "sep_surg_confounders": ["4"], "overall_phenotype_status": "CONFIRMED_POSITIVE", "confidence_score": 0.95, "rationale": "Positive.", "field_evidence": [{"field_name": "sep5_v3", "note_id": 1, "evidence_quote": "Klebsiella urosepsis with septic shock"}]})
                return {"choices": [{"message": {"content": content}}]}
        return Resp()

    monkeypatch.setattr(requests.Session, "post", fake_post)
    out_hosted = tmp_path / "out_hosted"
    summary = abstract_sepsis_cohort(notes, visits, inst, out_hosted, backend="hybrid", model_name="Muse-Synthetic")
    assert summary["manifest"]["status_counts"]["CONFIRMED_POSITIVE"] == 1
    assert summary["manifest"]["status_counts"]["CONFIRMED_NEGATIVE"] == 1
