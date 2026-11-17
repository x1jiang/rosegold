"""Synthetic tests for app.chorus_concordance."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.chorus_abstractor import load_chorus_instrument
from app.chorus_concordance import evaluate_chorus_concordance, main as concordance_main
from tests.test_chorus_abstractor import _synthetic_instrument


def _write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _sample_row(**overrides):
    base = {
        "institution_v3": "99",
        "record_id": "SYN-1",
        "vid": "101",
        "sepsis_ync_v3": "1",
        "sepsis_uncertain_v3": "",
        "sep1_v3___1": "1",
        "sep1_v3___2": "1",
        "sep1_v3___3": "0",
        "sep1_v3___4": "1",
        "sep3_v3": "Blood, Escherichia coli, (+)",
        "sep4_v3": "1",
        "sep6_v3": "4.2",
        "sep_surg_v3___1": "0",
        "sep_surg_v3___2": "0",
        "sep_surg_v3___3": "0",
        "sep_surg_v3___4": "1",
        "sep_surg_v3___9": "0",
        "sep7_v3": "2, 5, 3",
        "sep_day_v3": "1",
        "sep_dt_v3": "02:00",
        "sep_days": "1",
        "conf1": "5",
    }
    base.update(overrides)
    return base


def test_evaluate_chorus_concordance_gates_items_timing_and_confidence(tmp_path: Path):
    specs = load_chorus_instrument(_synthetic_instrument(tmp_path))

    human_rows = [
        _sample_row(record_id="SYN-1", vid="101", sepsis_ync_v3="1", conf1="5"),
        _sample_row(record_id="SYN-2", vid="102", sepsis_ync_v3="3", sepsis_uncertain_v3="1", sep3_v3="NO", sep6_v3="2.4", sep7_v3="1, 4", sep_day_v3="1", sep_dt_v3="04:00", sep_days="0", conf1="2"),
        _sample_row(record_id="SYN-3", vid="103", sepsis_ync_v3="2", sep1_v3___1="", sep1_v3___2="", sep1_v3___3="", sep1_v3___4="", sep3_v3="", sep4_v3="", sep6_v3="", sep_surg_v3___1="", sep_surg_v3___2="", sep_surg_v3___3="", sep_surg_v3___4="", sep_surg_v3___9="", sep7_v3="", sep_day_v3="", sep_dt_v3="", sep_days="", conf1=""),
        _sample_row(record_id="SYN-4", vid="104", sepsis_ync_v3="3", sepsis_uncertain_v3="2", sep1_v3___1="", sep1_v3___2="", sep1_v3___3="", sep1_v3___4="", sep3_v3="", sep4_v3="", sep6_v3="", sep_surg_v3___1="", sep_surg_v3___2="", sep_surg_v3___3="", sep_surg_v3___4="", sep_surg_v3___9="", sep7_v3="", sep_day_v3="", sep_dt_v3="", sep_days="", conf1="3"),
    ]

    llm_rows = [
        _sample_row(record_id="SYN-1", vid="101", sepsis_ync_v3="1", sep3_v3="Blood, Escherichia coli, (+)", sep6_v3="4.2", sep7_v3="2, 5, 3", sep_day_v3="1", sep_dt_v3="02:00", sep_days="1", conf1=""),
        _sample_row(record_id="SYN-2", vid="102", sepsis_ync_v3="3", sepsis_uncertain_v3="1", sep1_v3___3="", sep3_v3="NO", sep6_v3="2.6", sep7_v3="1, 5", sep_day_v3="1", sep_dt_v3="05:00", sep_days="0", conf1=""),
        _sample_row(record_id="SYN-3", vid="103", sepsis_ync_v3="2", sep1_v3___1="", sep1_v3___2="", sep1_v3___3="", sep1_v3___4="", sep3_v3="", sep4_v3="", sep6_v3="", sep_surg_v3___1="", sep_surg_v3___2="", sep_surg_v3___3="", sep_surg_v3___4="", sep_surg_v3___9="", sep7_v3="", sep_day_v3="", sep_dt_v3="", sep_days="", conf1=""),
        _sample_row(record_id="SYN-4", vid="104", sepsis_ync_v3="1", sepsis_uncertain_v3="", sep6_v3="1.9", conf1=""),
    ]
    llm_ev = [{"record_id": "SYN-4", "vid": "104", "field_name": "sepsis_ync_v3", "evidence_quote": "Ambiguous discharge wording", "source_ref": "note:44"}]

    summary = evaluate_chorus_concordance(human_rows, llm_rows, specs, llm_evidence_rows=llm_ev)

    assert summary["matched_encounters"] == 4
    assert summary["shared_open_branch_n"] == 2
    assert summary["gate_3way"]["paired_n"] == 4
    assert summary["gate_3way"]["matrix"]["1"]["1"] == 1
    assert summary["gate_3way"]["matrix"]["3"]["1"] == 1
    assert summary["gate_3way"]["linear_weighted_kappa"] >= summary["gate_3way"]["unweighted_kappa"]

    assert summary["gate_resolved"]["total_evaluated"] == 4
    assert summary["gate_resolved"]["tp"] == 2
    assert summary["gate_resolved"]["fp"] == 1
    assert summary["gate_resolved"]["tn"] == 1
    assert "high_confidence_4_5" in summary["gate_resolved"]["by_clinician_confidence"]
    assert "low_confidence_1_3" in summary["gate_resolved"]["by_clinician_confidence"]

    assert summary["item_metrics"]["sep1_v3___3"]["llm_missing_count"] == 1
    assert summary["item_metrics"]["sep3_v3_positive"]["sep3_organism_jaccard_mean"] == 1.0

    nt = summary["numeric_and_timing_metrics"]
    assert nt["sep6_v3"]["mae"] == 0.1
    assert nt["sep6_v3"]["within_0_2_mmol_pct"] == 100.0
    assert nt["sep7_v3"]["peak_sofa_mae"] == 0.5
    assert nt["sep7_v3"]["within_1_point_pct"] == 100.0
    assert nt["sep_day_v3"]["exact_match_pct"] == 100.0
    assert nt["sep_days"]["exact_day_match_pct"] == 100.0
    assert nt["sep_days"]["combined_onset_mae_hours"] == 0.5

    disc_fields = {(d["record_id"], d["field"]) for d in summary["discrepancies"]}
    assert ("SYN-4", "gate_resolved") in disc_fields
    syn4_disc = next(d for d in summary["discrepancies"] if d["record_id"] == "SYN-4" and d["field"] == "sepsis_ync_v3")
    assert syn4_disc["llm_evidence_quote"] == "Ambiguous discharge wording"


def test_concordance_cli_writes_markdown_json_and_csv(tmp_path: Path):
    inst = _synthetic_instrument(tmp_path)
    human = _write_csv(tmp_path / "human.csv", [_sample_row(record_id="SYN-1", vid="101")])
    llm = _write_csv(tmp_path / "llm.csv", [_sample_row(record_id="SYN-1", vid="101", conf1="")])
    out = tmp_path / "conc_out"
    rc = concordance_main(["--human-csv", str(human), "--llm-csv", str(llm), "--dictionary", str(inst), "--output-dir", str(out)])
    assert rc == 0
    assert (out / "concordance_summary.json").exists()
    assert (out / "concordance_report.md").exists()
    assert (out / "discrepancies.csv").exists()
    report = (out / "concordance_report.md").read_text()
    assert "Stage-1 Three-Way Gate" in report
    data = json.loads((out / "concordance_summary.json").read_text())
    assert data["gate_resolved"]["cohens_kappa"] == 1.0
