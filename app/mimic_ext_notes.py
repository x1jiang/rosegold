"""
MIMIC-III-Ext-Notes v1.0.0 adapter.

PhysioNet project: https://physionet.org/content/mimic-iii-ext-notes/1.0.0/
Schema is taken from the published data description. Credentialed notes.csv /
labels.csv are never bundled. Callers must supply a local extract they are
authorized to use.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd

DATASET_ID = "mimic-iii-ext-notes"
DATASET_VERSION = "1.0.0"
DATASET_DOI = "10.13026/9tfx-yx07"

NOTE_COLUMNS = ("row_id", "hadm_id", "subject_id", "text")
LABEL_COLUMNS = (
    "row_id",
    "trigger_word",
    "concept",
    "semtypes",
    "start",
    "end",
    "detection",
    "encounter",
    "negation",
)
SEMTYPES = {"sosy", "dsyn", "mobd"}
DETECTION_VALUES = {"yes", "no"}
ENCOUNTER_VALUES = {"yes", "no", "-"}
NEGATION_VALUES = {"yes", "no", "unsure", "-"}

PHENOTYPE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Sepsis / Septic Shock": ("sepsis", "septic shock", "septic", "septicemia"),
    "Acute Ischemic Stroke": (
        "stroke",
        "cerebral infarction",
        "cerebrovascular accident",
        "ischemic stroke",
        "transient cerebral ischemia",
    ),
    "Acute Respiratory Distress Syndrome (ARDS)": (
        "ards",
        "respiratory distress",
        "acute respiratory distress",
        "respiratory failure",
    ),
    "Acute Kidney Injury (AKI)": (
        "acute kidney injury",
        "aki",
        "acute renal failure",
        "pre-renal acute kidney injury",
        "kidney failure",
    ),
}


class MimicExtNotesError(ValueError):
    """Invalid MIMIC-III-Ext-Notes table or label contract."""


def _norm_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    return out


def _as_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def looks_like_mimic_ext_notes(frame: pd.DataFrame) -> bool:
    cols = {str(col).strip().lower() for col in frame.columns}
    return set(NOTE_COLUMNS).issubset(cols) and "note_text" not in cols


def looks_like_mimic_ext_notes_path(path: str) -> bool:
    if not path or not os.path.isfile(path):
        return False
    header = pd.read_csv(path, nrows=0)
    return looks_like_mimic_ext_notes(header)


def _require_columns(frame: pd.DataFrame, required: Sequence[str], table: str) -> None:
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise MimicExtNotesError(f"{table} missing required columns: {', '.join(missing)}")


def validate_notes(frame: pd.DataFrame) -> pd.DataFrame:
    notes = _norm_columns(frame)
    _require_columns(notes, NOTE_COLUMNS, "notes.csv")
    notes = notes.loc[:, list(NOTE_COLUMNS)].copy()
    notes["row_id"] = pd.to_numeric(notes["row_id"], errors="raise").astype("int64")
    notes["hadm_id"] = pd.to_numeric(notes["hadm_id"], errors="raise").astype("int64")
    notes["subject_id"] = pd.to_numeric(notes["subject_id"], errors="raise").astype("int64")
    notes["text"] = notes["text"].map(_as_str)
    if notes["row_id"].duplicated().any():
        raise MimicExtNotesError("notes.csv row_id values must be unique")
    if (notes["text"] == "").any():
        raise MimicExtNotesError("notes.csv text must be non-empty")
    return notes.reset_index(drop=True)


def validate_labels(
    frame: pd.DataFrame,
    notes: Optional[pd.DataFrame] = None,
    normalize_undetected: bool = False,
) -> pd.DataFrame:
    labels = _norm_columns(frame)
    _require_columns(labels, LABEL_COLUMNS, "labels.csv")
    labels = labels.loc[:, list(LABEL_COLUMNS)].copy()
    labels["row_id"] = pd.to_numeric(labels["row_id"], errors="raise").astype("int64")
    labels["start"] = pd.to_numeric(labels["start"], errors="raise").astype("int64")
    labels["end"] = pd.to_numeric(labels["end"], errors="raise").astype("int64")
    labels["trigger_word"] = labels["trigger_word"].map(_as_str)
    labels["concept"] = labels["concept"].map(_as_str)
    labels["semtypes"] = labels["semtypes"].map(lambda value: _as_str(value).lower() or None)
    labels["detection"] = labels["detection"].map(lambda value: _as_str(value).lower())
    labels["encounter"] = labels["encounter"].map(lambda value: _as_str(value).lower())
    labels["negation"] = labels["negation"].map(lambda value: _as_str(value).lower())

    bad_sem = labels["semtypes"].notna() & ~labels["semtypes"].isin(SEMTYPES)
    if bad_sem.any():
        raise MimicExtNotesError("labels.csv semtypes must be sosy, dsyn, mobd, or empty")
    if (~labels["detection"].isin(DETECTION_VALUES)).any():
        raise MimicExtNotesError("labels.csv detection must be yes or no")
    if (~labels["encounter"].isin(ENCOUNTER_VALUES)).any():
        raise MimicExtNotesError("labels.csv encounter must be yes, no, or -")
    if (~labels["negation"].isin(NEGATION_VALUES)).any():
        raise MimicExtNotesError("labels.csv negation must be yes, no, unsure, or -")

    undetected = labels["detection"] == "no"
    if normalize_undetected:
        labels.loc[undetected, "encounter"] = "-"
        labels.loc[undetected, "negation"] = "-"
    else:
        if ((labels.loc[undetected, "encounter"] != "-") | (labels.loc[undetected, "negation"] != "-")).any():
            raise MimicExtNotesError("labels.csv detection=no requires encounter='-' and negation='-'")

    if notes is not None:
        known = set(notes["row_id"].tolist())
        unknown = sorted(set(labels["row_id"]) - known)
        if unknown:
            raise MimicExtNotesError(f"labels.csv row_id values missing from notes.csv: {unknown}")
        notes_by_id = notes.set_index("row_id")["text"]
        for row in labels.itertuples(index=False):
            text = notes_by_id.loc[int(row.row_id)]
            start, end = int(row.start), int(row.end)
            if start < 0 or end < start or end > len(text):
                raise MimicExtNotesError(
                    f"label span [{start}:{end}] is outside note {row.row_id} (len={len(text)})"
                )
            span = text[start:end]
            if span != row.trigger_word:
                raise MimicExtNotesError(
                    f"label span for note {row.row_id} is {span!r}, expected trigger {row.trigger_word!r}"
                )
    return labels.reset_index(drop=True)


def notes_to_omop(notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    valid = validate_notes(notes)
    omop_notes = pd.DataFrame(
        {
            "note_id": valid["row_id"],
            "person_id": valid["subject_id"],
            "visit_occurrence_id": valid["hadm_id"],
            "note_date": "",
            "note_datetime": "",
            "note_type_concept_id": 44814645,
            "note_title": "Nursing Note",
            "note_text": valid["text"],
        }
    )
    visits = (
        valid.groupby("hadm_id", as_index=False)
        .agg(person_id=("subject_id", "first"))
        .rename(columns={"hadm_id": "visit_occurrence_id"})
    )
    visits["visit_concept_id"] = 9201
    visits["visit_start_date"] = ""
    visits["visit_end_date"] = ""
    visits["visit_type_concept_id"] = 44818518
    visits["care_site_id"] = None
    return omop_notes, visits


def join_labels(
    notes: pd.DataFrame,
    labels: pd.DataFrame,
    normalize_undetected: bool = False,
) -> pd.DataFrame:
    valid_notes = validate_notes(notes)
    valid_labels = validate_labels(labels, valid_notes, normalize_undetected=normalize_undetected)
    return valid_labels.merge(valid_notes, on="row_id", how="left")


def phenotype_gold(
    notes: pd.DataFrame,
    labels: pd.DataFrame,
    target_condition: str,
    normalize_undetected: bool = True,
) -> pd.DataFrame:
    merged = join_labels(notes, labels, normalize_undetected=normalize_undetected)
    aliases = PHENOTYPE_ALIASES.get(target_condition, (target_condition.lower(),))
    concept = merged["concept"].str.lower()
    hit = pd.Series(False, index=merged.index)
    for alias in aliases:
        hit = hit | concept.str.contains(alias, regex=False, na=False)
    active = (
        hit
        & (merged["detection"] == "yes")
        & (merged["encounter"] == "yes")
        & (merged["negation"] == "no")
    )
    positive_visits = set(merged.loc[active, "hadm_id"].astype(int))
    visits = (
        validate_notes(notes)
        .groupby("hadm_id", as_index=False)
        .agg(person_id=("subject_id", "first"))
    )
    visits["visit_occurrence_id"] = visits["hadm_id"].astype(int)
    visits["human_positive"] = visits["visit_occurrence_id"].isin(positive_visits)
    visits["target_condition"] = target_condition
    return visits[["person_id", "visit_occurrence_id", "human_positive", "target_condition"]]


def score_label_predictions(gold: pd.DataFrame, predictions: pd.DataFrame) -> Dict[str, Any]:
    gold_labels = validate_labels(gold)
    preds = _norm_columns(predictions)
    required = ("row_id", "concept", "detection", "encounter", "negation")
    _require_columns(preds, required, "predictions")
    preds["row_id"] = pd.to_numeric(preds["row_id"], errors="raise").astype("int64")
    preds["concept"] = preds["concept"].map(_as_str)
    for col in ("detection", "encounter", "negation"):
        preds[col] = preds[col].map(lambda value: _as_str(value).lower())

    key = ["row_id", "concept"]
    merged = gold_labels.merge(preds, on=key, how="left", suffixes=("_gold", "_pred"))
    if merged["detection_pred"].isna().any():
        raise MimicExtNotesError("predictions are missing one or more gold row_id+concept keys")

    tasks = {}
    for task in ("detection", "encounter", "negation"):
        y_true = merged[f"{task}_gold"]
        y_pred = merged[f"{task}_pred"]
        comparable = y_true != "-"
        true_yes = (y_true == "yes") & comparable
        pred_yes = (y_pred == "yes") & comparable
        tp = int((true_yes & pred_yes).sum())
        fp = int((~true_yes & pred_yes).sum())
        fn = int((true_yes & ~pred_yes).sum())
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        tasks[task] = {
            "n": int(comparable.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return {"n_labels": int(len(gold_labels)), "tasks": tasks}


def build_concept_eval_prompt(note_text: str, concept: str, trigger_word: str) -> str:
    return (
        "You are evaluating one MetaMap-extracted clinical concept inside a single nursing note.\n"
        "Answer only the three annotation questions used by MIMIC-III-Ext-Notes.\n\n"
        f"Concept: {concept}\n"
        f"Trigger span: {trigger_word}\n\n"
        "Note text:\n"
        f"{note_text}\n\n"
        "Questions:\n"
        "1. detection: Is the concept correctly detected in this note? yes or no.\n"
        "2. encounter: If detection is yes, is the concept being dealt with in the current encounter? yes or no. If detection is no, answer -.\n"
        "3. negation: If detection is yes, should the condition be treated as negated or already resolved? yes, no, or unsure. If detection is no, answer -.\n\n"
        'Return JSON only: {"detection":"...","encounter":"...","negation":"..."}'
    )


def write_omop_bundle(
    notes: pd.DataFrame,
    output_dir: str,
    labels: Optional[pd.DataFrame] = None,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    omop_notes, omop_visits = notes_to_omop(notes)
    paths = {
        "notes": os.path.join(output_dir, "omop_notes.csv"),
        "visits": os.path.join(output_dir, "omop_visits.csv"),
        "manifest": os.path.join(output_dir, "manifest.json"),
    }
    omop_notes.to_csv(paths["notes"], index=False)
    omop_visits.to_csv(paths["visits"], index=False)
    manifest: Dict[str, Any] = {
        "dataset": DATASET_ID,
        "version": DATASET_VERSION,
        "doi": DATASET_DOI,
        "n_notes": int(len(omop_notes)),
        "n_visits": int(len(omop_visits)),
        "n_labels": 0,
        "note_id_source": "row_id",
        "visit_occurrence_id_source": "hadm_id",
        "person_id_source": "subject_id",
    }
    if labels is not None:
        valid_labels = validate_labels(labels, validate_notes(notes), normalize_undetected=True)
        paths["labels"] = os.path.join(output_dir, "labels.csv")
        valid_labels.to_csv(paths["labels"], index=False)
        manifest["n_labels"] = int(len(valid_labels))
    with open(paths["manifest"], "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return paths


def load_source_bundle(
    source_dir: str,
    normalize_undetected: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    notes_path = os.path.join(source_dir, "notes.csv")
    labels_path = os.path.join(source_dir, "labels.csv")
    if not os.path.isfile(notes_path):
        raise MimicExtNotesError(f"notes.csv not found in {source_dir}")
    notes = validate_notes(pd.read_csv(notes_path))
    labels = None
    if os.path.isfile(labels_path):
        labels = validate_labels(
            pd.read_csv(labels_path),
            notes,
            normalize_undetected=normalize_undetected,
        )
    return notes, labels
