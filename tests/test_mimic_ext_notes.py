import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from app.mimic_ext_notes import (
    MimicExtNotesError,
    join_labels,
    looks_like_mimic_ext_notes_path,
    notes_to_omop,
    phenotype_gold,
    score_label_predictions,
    validate_labels,
    validate_notes,
)
from app.omop_loader import load_omop_data, load_visit_index

FIXTURE = Path("tests/fixtures/mimic_iii_ext_notes")


def _notes():
    return pd.read_csv(FIXTURE / "notes.csv")


def _labels():
    return pd.read_csv(FIXTURE / "labels.csv")


def test_schema_accepts_official_and_rejects_missing():
    notes = validate_notes(_notes())
    labels = validate_labels(_labels(), notes)
    assert list(notes.columns) == ["row_id", "hadm_id", "subject_id", "text"]
    assert "detection" in labels.columns
    assert looks_like_mimic_ext_notes_path(str(FIXTURE / "notes.csv"))

    broken = _notes().drop(columns=["hadm_id"])
    with pytest.raises(MimicExtNotesError, match="hadm_id"):
        validate_notes(broken)

    missing_label = _labels().drop(columns=["negation"])
    with pytest.raises(MimicExtNotesError, match="negation"):
        validate_labels(missing_label, notes)


def test_notes_convert_to_omop_ids():
    omop_notes, omop_visits = notes_to_omop(_notes())
    assert set(omop_notes["note_id"]) == {101, 102}
    assert set(omop_notes["visit_occurrence_id"]) == {9001, 9002}
    assert set(omop_notes["person_id"]) == {501, 502}
    shock = omop_notes.loc[omop_notes["note_id"] == 101, "note_text"].iloc[0]
    assert "septic shock" in shock
    assert set(omop_visits["visit_occurrence_id"]) == {9001, 9002}
    assert omop_visits.set_index("visit_occurrence_id").loc[9001, "person_id"] == 501

    shared = pd.DataFrame(
        {
            "row_id": [201, 202],
            "hadm_id": [9100, 9100],
            "subject_id": [601, 601],
            "text": [
                "SYNTHETIC-FIXTURE day 1 septic shock",
                "SYNTHETIC-FIXTURE day 2 improving",
            ],
        }
    )
    shared_notes, shared_visits = notes_to_omop(shared)
    assert len(shared_notes) == 2
    assert list(shared_visits["visit_occurrence_id"]) == [9100]


def test_converted_notes_load_through_chart_loader():
    records = load_omop_data(str(FIXTURE / "notes.csv"), "data/synthetic_visits.csv")
    assert {item["visit_occurrence_id"] for item in records} == {9001, 9002}
    shock = next(item for item in records if item["visit_occurrence_id"] == 9001)
    assert shock["person_id"] == 501
    assert shock["note_count"] == 1
    assert "SYNTHETIC-FIXTURE" in shock["notes_formatted_text"]
    assert "septic shock" in shock["notes_formatted_text"]
    assert "Nursing Note" in shock["notes_formatted_text"]

    index = load_visit_index(str(FIXTURE / "notes.csv"), "data/synthetic_visits.csv")
    assert {item["visit_occurrence_id"] for item in index} == {9001, 9002}
    assert next(item for item in index if item["visit_occurrence_id"] == 9001)["note_count"] == 1

    missing_visits = load_omop_data(str(FIXTURE / "notes.csv"), "data/does_not_exist_visits.csv")
    assert {item["visit_occurrence_id"] for item in missing_visits} == {9001, 9002}


def test_labels_join_dash_rules_and_gold_mismatch():
    notes = _notes()
    labels = _labels()
    merged = join_labels(notes, labels)
    assert len(merged) == 5
    assert set(merged["row_id"]) == {101, 102}

    bad = labels.copy()
    bad.loc[bad["detection"] == "no", "encounter"] = "yes"
    with pytest.raises(MimicExtNotesError, match="detection=no"):
        validate_labels(bad, validate_notes(notes))

    gold = phenotype_gold(notes, labels, "Sepsis / Septic Shock")
    gold_map = dict(zip(gold["visit_occurrence_id"], gold["human_positive"]))
    assert gold_map[9001] is True
    assert gold_map[9002] is False

    perfect = labels[["row_id", "concept", "detection", "encounter", "negation"]].copy()
    metrics = score_label_predictions(labels, perfect)
    assert metrics["tasks"]["detection"]["f1"] == 1.0
    assert metrics["tasks"]["encounter"]["f1"] == 1.0

    flipped = perfect.copy()
    flipped.loc[flipped["concept"] == "Septic Shock", "detection"] = "no"
    flipped.loc[flipped["concept"] == "Septic Shock", "encounter"] = "-"
    flipped.loc[flipped["concept"] == "Septic Shock", "negation"] = "-"
    mismatched = score_label_predictions(labels, flipped)
    assert mismatched["tasks"]["detection"]["f1"] < 1.0
    assert mismatched["tasks"]["detection"]["fn"] >= 1


def test_prepare_cli_writes_omop_csvs(tmp_path):
    output = tmp_path / "prepared"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_mimic_iii_ext_notes.py",
            "--source",
            str(FIXTURE),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    notes = pd.read_csv(output / "omop_notes.csv")
    visits = pd.read_csv(output / "omop_visits.csv")
    labels = pd.read_csv(output / "labels.csv")
    manifest = json.loads((output / "manifest.json").read_text())
    assert list(notes["note_id"]) == [101, 102]
    assert list(notes["visit_occurrence_id"]) == [9001, 9002]
    assert list(visits["visit_occurrence_id"]) == [9001, 9002]
    assert manifest["n_notes"] == len(notes)
    assert manifest["n_visits"] == len(visits)
    assert manifest["n_labels"] == len(labels)
    assert "text" not in manifest
    assert "septic shock" not in (output / "manifest.json").read_text()
