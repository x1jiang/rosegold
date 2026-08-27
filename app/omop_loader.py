import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from app.config_loader import pipeline_settings

_TABLE_CACHE: Dict[Tuple[Any, ...], pd.DataFrame] = {}
_RECORD_CACHE: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
_INDEX_CACHE: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
_CACHE_LIMIT = 24


def _cache_put(store: Dict, key: Tuple[Any, ...], value: Any, limit: int = _CACHE_LIMIT) -> None:
    store[key] = value
    while len(store) > limit:
        store.pop(next(iter(store)))


def _file_stamp(path: str) -> Tuple[str, int, int]:
    abspath = os.path.abspath(path)
    stat = os.stat(abspath)
    return abspath, int(stat.st_mtime_ns), int(stat.st_size)


def _read_table(path: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    stamp = _file_stamp(path)
    key = (stamp, tuple(c.lower() for c in columns) if columns else None)
    cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    if path.endswith(".parquet"):
        frame = pd.read_parquet(path, columns=list(columns) if columns else None)
    else:
        frame = pd.read_csv(path, usecols=list(columns) if columns else None)
    frame.columns = [str(col).lower() for col in frame.columns]
    _cache_put(_TABLE_CACHE, key, frame)
    return frame


def load_visit_index(notes_path: str, visits_path: str) -> List[Dict[str, Any]]:
    """Visit metadata plus note counts, without materializing note text."""
    key = (_file_stamp(notes_path), _file_stamp(visits_path))
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    df_visits = _read_table(visits_path)
    try:
        df_notes = _read_table(notes_path, columns=["visit_occurrence_id"])
    except (ValueError, KeyError, OSError):
        df_notes = _read_table(notes_path)

    counts = (
        df_notes.groupby("visit_occurrence_id").size()
        if "visit_occurrence_id" in df_notes.columns
        else pd.Series(dtype="int64")
    )

    index: List[Dict[str, Any]] = []
    for visit in df_visits.itertuples(index=False):
        visit_id = int(visit.visit_occurrence_id)
        index.append(
            {
                "visit_occurrence_id": visit_id,
                "person_id": int(visit.person_id),
                "visit_start_date": str(getattr(visit, "visit_start_date", "Unknown")),
                "visit_end_date": str(getattr(visit, "visit_end_date", "Unknown")),
                "note_count": int(counts.get(visit_id, 0)),
            }
        )
    _cache_put(_INDEX_CACHE, key, index)
    return index


def load_omop_data(
    notes_path: str,
    visits_path: str,
    target_visits: Optional[List[int]] = None,
    max_notes_per_visit: Optional[int] = None,
    max_chars_per_note: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Loads OMOP NOTE and VISIT_OCCURRENCE tables, joins and orders notes chronologically per visit.
    Supports CSV and Parquet files. Caches parsed tables and formatted records by file stamp.
    """
    settings = pipeline_settings()
    max_notes = int(max_notes_per_visit if max_notes_per_visit is not None else settings.get("max_notes_per_visit", 50))
    max_chars = int(max_chars_per_note if max_chars_per_note is not None else settings.get("max_chars_per_note", 4000))
    visit_key = tuple(int(v) for v in target_visits) if target_visits is not None else None
    cache_key = (_file_stamp(notes_path), _file_stamp(visits_path), visit_key, max_notes, max_chars)
    cached = _RECORD_CACHE.get(cache_key)
    if cached is not None:
        return cached

    df_visits = _read_table(visits_path)
    df_notes = _read_table(notes_path)

    if target_visits is not None:
        wanted = set(int(v) for v in target_visits)
        df_visits = df_visits[df_visits["visit_occurrence_id"].isin(wanted)]
        if "visit_occurrence_id" in df_notes.columns:
            df_notes = df_notes[df_notes["visit_occurrence_id"].isin(wanted)]

    if "note_text" in df_notes.columns:
        df_notes = df_notes[df_notes["note_text"].notna() & (df_notes["note_text"].astype(str).str.strip() != "")]

    date_col = "note_datetime" if "note_datetime" in df_notes.columns else "note_date"
    if date_col in df_notes.columns:
        df_notes = df_notes.copy()
        df_notes[date_col] = pd.to_datetime(df_notes[date_col], errors="coerce")
        df_notes = df_notes.sort_values(by=["visit_occurrence_id", date_col])

    notes_by_visit: Dict[int, pd.DataFrame] = {}
    if not df_notes.empty and "visit_occurrence_id" in df_notes.columns:
        for visit_id, group in df_notes.groupby("visit_occurrence_id", sort=False):
            notes_by_visit[int(visit_id)] = group.head(max_notes)

    prepared_records: List[Dict[str, Any]] = []
    for visit in df_visits.itertuples(index=False):
        visit_id = int(visit.visit_occurrence_id)
        visit_notes = notes_by_visit.get(visit_id)
        if visit_notes is not None and not visit_notes.empty:
            chunks = []
            for note in visit_notes.itertuples(index=False):
                note_id = getattr(note, "note_id", "N/A")
                note_date = str(getattr(note, date_col, getattr(note, "note_date", "Unknown")))
                note_title = str(getattr(note, "note_title", getattr(note, "note_type_concept_id", "Clinical Note")))
                note_text = str(getattr(note, "note_text", ""))[:max_chars]
                chunks.append(
                    f"--- [Note ID: {note_id} | Date: {note_date} | Type/Title: {note_title}] ---\n{note_text.strip()}\n"
                )
            chart_text = "\n".join(chunks)
            note_count = len(visit_notes)
        else:
            chart_text = "[No notes found for this visit encounter]"
            note_count = 0

        prepared_records.append(
            {
                "person_id": int(visit.person_id),
                "visit_occurrence_id": visit_id,
                "visit_start_date": str(getattr(visit, "visit_start_date", "Unknown")),
                "visit_end_date": str(getattr(visit, "visit_end_date", "Unknown")),
                "note_count": note_count,
                "notes_formatted_text": chart_text,
            }
        )

    _cache_put(_RECORD_CACHE, cache_key, prepared_records)
    return prepared_records
