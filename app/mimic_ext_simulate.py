"""Build a synthetic MIMIC-III-Ext-Notes bundle from the OMOP demo cohort."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.mimic_ext_notes import validate_labels, validate_notes, write_omop_bundle

CONCEPTS = (
    ("septic shock", "Septic Shock", "dsyn"),
    ("severe sepsis", "Sepsis", "dsyn"),
    ("sepsis", "Sepsis", "dsyn"),
    ("urosepsis", "Urosepsis", "dsyn"),
    ("acute ischemic stroke", "Cerebral Infarction", "dsyn"),
    ("acute infarct", "Cerebral Infarction", "dsyn"),
    ("tia", "Transient Ischemic Attack", "dsyn"),
    ("ards", "Respiratory Distress Syndrome, Adult", "dsyn"),
    ("acute respiratory distress", "Respiratory Distress", "sosy"),
    ("acute kidney injury", "Acute Kidney Injury", "dsyn"),
    ("aki", "Acute Kidney Injury", "dsyn"),
    ("fever", "Fever", "sosy"),
    ("afebrile", "Fever", "sosy"),
    ("aphasia", "Aphasia", "sosy"),
    ("hemiplegia", "Hemiplegia", "dsyn"),
    ("bacteremia", "Bacteremia", "dsyn"),
    ("cellulitis", "Cellulitis", "dsyn"),
    ("appendicitis", "Appendicitis", "dsyn"),
    ("pancreatitis", "Pancreatitis", "dsyn"),
    ("heart failure", "Heart Failure", "dsyn"),
    ("cardiogenic shock", "Cardiogenic Shock", "dsyn"),
    ("diabetic ketoacidosis", "Diabetic Ketoacidosis", "dsyn"),
    ("hernia", "Hernia", "dsyn"),
    ("nursing home", "Nursing Home", "dsyn"),
)

FALSE_POSITIVE = {"nursing home"}
NEGATED_TRIGGERS = {"afebrile"}


def _bounded(text: str, start: int, end: int) -> bool:
    if start > 0 and text[start - 1].isalnum():
        return False
    if end < len(text) and text[end].isalnum():
        return False
    return True


def _find_spans(text: str, needle: str) -> list[tuple[int, int, str]]:
    hits = []
    lower = text.lower()
    cursor = 0
    token = needle.lower()
    while True:
        idx = lower.find(token, cursor)
        if idx < 0:
            break
        end = idx + len(token)
        if _bounded(lower, idx, end):
            hits.append((idx, end, text[idx:end]))
        cursor = end
    return hits


def simulate_from_omop(notes_path: str, output_dir: str) -> dict[str, str]:
    omop = pd.read_csv(notes_path)
    omop.columns = [str(col).lower() for col in omop.columns]
    notes = validate_notes(
        pd.DataFrame(
            {
                "row_id": omop["note_id"].astype("int64"),
                "hadm_id": omop["visit_occurrence_id"].astype("int64"),
                "subject_id": omop["person_id"].astype("int64"),
                "text": omop["note_text"].astype(str),
            }
        )
    )

    rows = []
    for note in notes.itertuples(index=False):
        text = note.text
        lowered = text.lower()
        seen = set()
        for needle, concept, semtypes in CONCEPTS:
            if needle in seen:
                continue
            spans = _find_spans(text, needle)
            if not spans:
                continue
            seen.add(needle)
            start, end, trigger = spans[0]
            window = lowered[max(0, start - 24) : min(len(lowered), end + 24)]
            if needle in FALSE_POSITIVE:
                detection, encounter, negation = "no", "-", "-"
            elif needle in NEGATED_TRIGGERS or "denies " + needle in window or "no " + needle in window:
                detection, encounter, negation = "yes", "yes", "yes"
            elif "resolved" in window:
                detection, encounter, negation = "yes", "yes", "yes"
            else:
                detection, encounter, negation = "yes", "yes", "no"
            rows.append(
                {
                    "row_id": int(note.row_id),
                    "trigger_word": trigger,
                    "concept": concept,
                    "semtypes": semtypes,
                    "start": start,
                    "end": end,
                    "detection": detection,
                    "encounter": encounter,
                    "negation": negation,
                }
            )

    labels = validate_labels(pd.DataFrame(rows), notes)
    paths = write_omop_bundle(notes, output_dir, labels)
    notes_out = Path(output_dir) / "notes.csv"
    notes.to_csv(notes_out, index=False)
    paths["source_notes"] = str(notes_out)
    return paths
