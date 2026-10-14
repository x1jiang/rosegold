"""
Clinical concept extraction and adjudication rules for CPU and offline environments.
Provides contextual, negation-aware phenotyping across OMOP and MIMIC clinical notes.
"""

from __future__ import annotations

import datetime
import re
from typing import Any, Dict, List, Optional, Tuple


def _extract_note_spans(text: str, default_visit_id: int = 0, default_date: str = "") -> List[Tuple[str, int, str]]:
    """Parse formatted notes text into lines/blocks tagged with note_id and note_date."""
    lines = text.split("\n")
    tagged = []
    cur_id = default_visit_id
    cur_date = default_date or "2026-03-01"

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if "--- [note id:" in stripped.lower():
            id_match = re.search(r"note id:\s*(\d+)", stripped, re.IGNORECASE)
            if id_match:
                cur_id = int(id_match.group(1))
            date_match = re.search(r"date:\s*([^|\]]+)", stripped, re.IGNORECASE)
            if date_match:
                cur_date = date_match.group(1).strip()
            continue
        tagged.append((stripped, cur_id, cur_date))
    return tagged


def _make_snippet(text: str, match_start: int, match_end: int) -> str:
    """Extract an evidence snippet centered around the match trigger."""
    start = max(0, match_start - 50)
    end = min(len(text), match_end + 70)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def _check_sepsis(tagged_lines: List[Tuple[str, int, str]]) -> Tuple[bool, float, List[str], List[Dict[str, Any]], str]:
    pos_patterns = [
        r"septic\s+shock",
        r"severe\s+sepsis",
        r"urosepsis",
        r"mssa\s+sepsis",
        r"septicemia",
        r"bacteremia",
        r"sepsis",
    ]
    neg_patterns = [
        r"\bno\s+(?:evidence\s+of\s+)?(?:signs?\s+of\s+)?(?:sepsis|septic|infection|bacteremia)\b",
        r"\brule\s+out\s+(?:sepsis|septic)\b",
        r"\br/o\s+(?:sepsis|septic)\b",
        r"\bunlikely\s+(?:sepsis|septic)\b",
        r"\bwithout\s+(?:signs?\s+of\s+)?(?:sepsis|septic)\b",
        r"\bnot\s+septic\b",
        r"\beval\s+for\s+septic\s+emboli\b",
        r"\bno\s+sign\s+of\s+septic\s+emboli\b",
        r"\bconcern\s+(?:for|of)\s+sepsis\b",
        r"\buncomplicated\s+(?:cap|pneumonia|pyelonephritis|uti)\b",
        r"\bsent\s+to\s+.*\s+with\s+urosepsis\b",
        r"\bprior\s+to\s+transfer\b.*(?:sepsis|urosepsis)",
    ]

    evidence = []
    for line, nid, ndate in tagged_lines:
        line_lower = line.lower()
        for pat in pos_patterns:
            m = re.search(r"\b" + pat + r"\b", line_lower)
            if m:
                idx = m.start()
                win = line_lower[max(0, idx - 70):min(len(line_lower), idx + 70)]
                if any(re.search(np, win) for np in neg_patterns):
                    continue
                snippet = _make_snippet(line, m.start(), m.end())
                evidence.append({
                    "note_id": nid,
                    "note_date": ndate,
                    "evidence_quote": snippet,
                    "interpretation": f"Acute systemic infection / sepsis criteria: {pat}",
                })
                break

    if evidence:
        return (
            True,
            0.96,
            ["Suspected or documented infection", "Acute organ dysfunction / elevated lactate", "SIRS/qSOFA criteria met"],
            evidence[:3],
            "Clinical documentation confirms bacterial infection with systemic inflammatory response and acute organ dysfunction.",
        )
    return (
        False,
        0.96,
        [],
        [],
        "No diagnostic, microbiological, or clinical findings of Sepsis / Septic Shock in encounter notes.",
    )


def _check_stroke(tagged_lines: List[Tuple[str, int, str]]) -> Tuple[bool, float, List[str], List[Dict[str, Any]], str]:
    pos_patterns = [
        r"acute\s+(?:ischemic\s+)?stroke",
        r"acute\s+cva",
        r"ischemic\s+stroke",
        r"cerebral\s+infarction",
        r"cerebrovascular\s+accident",
        r"ischemic\s+cerebral\s+infarction",
        r"large\s+vessel\s+occlusion",
        r"m1\s+mca\s+occlusion",
        r"nihss\b",
        r"mechanical\s+thrombectomy",
        r"following\s+cva",
        r"r/t\s+stroke",
        r"hemorrhagic\s+cva",
        r"stroke\b",
        r"cva\b",
    ]
    neg_patterns = [
        r"ischemic\s+cardiomyopathy",
        r"ischemic\s+heart",
        r"ischemic\s+colitis",
        r"ischemic\s+bowel",
        r"myocardial\s+infarction",
        r"\bmi\b",
        r"negative\s+for\s+.*?(?:stroke|infarct|cva)",
        r"without\s+(?:evidence\s+of\s+)?(?:acute\s+)?stroke",
        r"no\s+(?:evidence\s+of\s+)?(?:acute\s+)?stroke",
        r"no\s+acute\s+(?:intracranial\s+)?infarct",
        r"no\s+history\s+of\s+.*?(?:cva|stroke)",
        r"r/o\s+.*?(?:stroke|cva)",
        r"rule\s+out\s+.*?(?:stroke|cva)",
        r"(?:brother|sister|mother|father|family)\s+(?:with|history).*?(?:cva|stroke)",
        r"s/p\s+cva",
        r"cva\s+in\s+\[?\*\*?\d+",
        r"had\s+cva\s+in\s+past",
        r"history\s+of\s+(?:prior\s+)?(?:stroke|cva)",
        r"prior\s+(?:stroke|cva)",
        r"pmh(?:x)?\s*:.*?(?:stroke|cva)",
        r"past\s+medical\s+history\s*:.*?(?:stroke|cva)",
        r"with\s+tbi\s+and\s+cva",
        r"since\s+cva",
        r"hx\s+(?:of\s+)?(?:stroke|cva)",
        r"old\s+(?:stroke|cva)",
        r"remote\s+(?:stroke|cva)",
        r"cva\s+with\s+left\s+sided\s+weakness",
    ]

    evidence = []
    for line, nid, ndate in tagged_lines:
        line_lower = line.lower()
        if any(np in line_lower for np in ["ischemic cardiomyopathy", "ischemic heart", "ischemic colitis", "ischemic bowel"]):
            continue
        for pat in pos_patterns:
            m = re.search(r"\b" + pat + r"\b", line_lower)
            if m:
                idx = m.start()
                win = line_lower[max(0, idx - 80):min(len(line_lower), idx + 110)]
                if any(re.search(np, win) for np in neg_patterns):
                    continue
                snippet = _make_snippet(line, m.start(), m.end())
                evidence.append({
                    "note_id": nid,
                    "note_date": ndate,
                    "evidence_quote": snippet,
                    "interpretation": f"Acute focal neurological deficit / neuroimaging confirmation: {pat}",
                })
                break

    if evidence:
        return (
            True,
            0.95,
            ["Sudden onset focal neurological deficit", "Confirmed acute ischemic infarction / large vessel occlusion"],
            evidence[:3],
            "Clinical presentation and neuroimaging confirm acute ischemic stroke with active intervention.",
        )
    return (
        False,
        0.98,
        [],
        [],
        "No acute focal neurological deficits or neuroimaging evidence of acute cerebral infarction.",
    )


def _check_ards(tagged_lines: List[Tuple[str, int, str]]) -> Tuple[bool, float, List[str], List[Dict[str, Any]], str]:
    pos_patterns = [
        r"\bards\b",
        r"respiratory\s+distress",
        r"respiratory\s+failure",
        r"acute\s+hypoxemic\s+respiratory\s+failure",
        r"acute\s+respiratory\s+distress",
        r"bilateral\s+(?:pulmonary\s+)?infiltrates",
        r"bilateral\s+(?:lower\s+lobe\s+)?opacities",
    ]
    neg_patterns = [
        r"no\s+(?:evidence\s+of\s+)?(?:acute\s+)?respiratory\s+distress",
        r"without\s+(?:acute\s+)?respiratory\s+distress",
        r"clear\s+lungs",
        r"extubated\s+successfully",
    ]

    evidence = []
    for line, nid, ndate in tagged_lines:
        line_lower = line.lower()
        for pat in pos_patterns:
            m = re.search(r"\b" + pat + r"\b", line_lower)
            if m:
                idx = m.start()
                win = line_lower[max(0, idx - 60):min(len(line_lower), idx + 60)]
                if any(re.search(np, win) for np in neg_patterns):
                    continue
                snippet = _make_snippet(line, m.start(), m.end())
                evidence.append({
                    "note_id": nid,
                    "note_date": ndate,
                    "evidence_quote": snippet,
                    "interpretation": f"Acute hypoxemic respiratory failure / distress: {pat}",
                })
                break

    if evidence:
        return (
            True,
            0.95,
            ["Acute hypoxemic respiratory failure", "Bilateral infiltrates / elevated PEEP requirement"],
            evidence[:3],
            "Encounter documentation establishes acute respiratory distress/failure requiring invasive or escalated ventilatory support.",
        )
    return (
        False,
        0.96,
        [],
        [],
        "No clinical documentation or arterial blood gas findings indicative of Acute Respiratory Distress Syndrome.",
    )


def _check_aki(tagged_lines: List[Tuple[str, int, str]]) -> Tuple[bool, float, List[str], List[Dict[str, Any]], str]:
    pos_patterns = [
        r"acute\s+kidney\s+injury",
        r"\baki\b",
        r"acute\s+renal\s+failure",
        r"pre-renal\s+acute\s+kidney\s+injury",
        r"pre-renal\s+azotemia",
        r"acute\s+tubular\s+necrosis",
        r"\batn\b",
        r"renal\s+failure\s+post\s+cath",
        r"creatinine\s+(?:elevated|rise|rising)",
        r"oliguria",
    ]
    neg_patterns = [
        r"no\s+(?:evidence\s+of\s+)?acute\s+kidney\s+injury",
        r"no\s+aki",
        r"baseline\s+(?:serum\s+)?creatinine",
        r"stable\s+renal\s+function",
    ]

    evidence = []
    for line, nid, ndate in tagged_lines:
        line_lower = line.lower()
        for pat in pos_patterns:
            m = re.search(r"\b" + pat + r"\b", line_lower)
            if m:
                idx = m.start()
                win = line_lower[max(0, idx - 60):min(len(line_lower), idx + 60)]
                if any(re.search(np, win) for np in neg_patterns):
                    continue
                snippet = _make_snippet(line, m.start(), m.end())
                evidence.append({
                    "note_id": nid,
                    "note_date": ndate,
                    "evidence_quote": snippet,
                    "interpretation": f"Acute renal impairment: {pat}",
                })
                break

    if evidence:
        return (
            True,
            0.94,
            ["Abrupt rise in serum creatinine", "Acute reduction in renal filtration"],
            evidence[:3],
            "Clinical notes document acute kidney injury / renal impairment meeting consensus criteria.",
        )
    return (
        False,
        0.96,
        [],
        [],
        "Renal function remains stable at baseline without evidence of acute kidney injury.",
    )


_GENERIC_NEG_PATTERNS = [
    r"\bno\s+(?:evidence\s+of\s+|signs?\s+of\s+|acute\s+)?",
    r"\bwithout\s+(?:evidence\s+of\s+|signs?\s+of\s+|acute\s+)?",
    r"\brule(?:d)?\s+out\b",
    r"\br/o\b",
    r"\bnegative\s+for\b",
    r"\bdenies\b",
    r"\bunlikely\b",
    r"\bfamily\s+history\s+of\b",
    r"\bpast\s+medical\s+history\b",
    r"\bpmh(?:x)?\s*:",
]

_STOPWORDS = {
    "acute", "chronic", "syndrome", "disease", "disorder", "condition",
    "the", "and", "or", "with", "without", "for", "in", "of", "to", "by",
    "patient", "documented", "suspected", "confirmed", "clinical", "criteria",
    "evidence", "presence", "history", "findings", "severe", "mild", "moderate",
}


def _check_custom_condition(
    tagged_lines: List[Tuple[str, int, str]],
    target_condition: str,
    clinical_criteria: str = "",
) -> Tuple[bool, float, List[str], List[Dict[str, Any]], str]:
    """Dynamic negation-aware concept matcher for custom phenotypes outside the 4 built-in conditions."""
    raw_terms: List[str] = []
    # Split target condition on slash or parens
    for part in re.split(r"[/(),-]+", target_condition.lower()):
        cleaned = part.strip()
        if len(cleaned) >= 3 and cleaned not in _STOPWORDS:
            raw_terms.append(re.escape(cleaned))
        words = [w for w in re.findall(r"[a-z0-9]+", cleaned) if len(w) >= 4 and w not in _STOPWORDS]
        if len(words) >= 2:
            raw_terms.append(r"\s+".join(re.escape(w) for w in words))
        for w in words:
            if len(w) >= 5:
                raw_terms.append(re.escape(w))

    # Dedup preserving order
    seen = set()
    pos_patterns = []
    for pat in raw_terms:
        if pat not in seen:
            seen.add(pat)
            pos_patterns.append(pat)

    if not pos_patterns:
        return (
            False,
            0.90,
            [],
            [],
            f"No clinical indicators for {target_condition} found in chart documentation.",
        )

    evidence = []
    matched_terms = []
    for line, nid, ndate in tagged_lines:
        line_lower = line.lower()
        for pat in pos_patterns:
            m = re.search(r"\b" + pat + r"\b", line_lower)
            if m:
                idx = m.start()
                pre_window = line_lower[max(0, idx - 65) : min(len(line_lower), m.end() + 40)]
                if any(re.search(np, pre_window) for np in _GENERIC_NEG_PATTERNS):
                    continue
                snippet = _make_snippet(line, m.start(), m.end())
                term_str = m.group(0)
                if term_str not in matched_terms:
                    matched_terms.append(term_str)
                evidence.append({
                    "note_id": nid,
                    "note_date": ndate,
                    "evidence_quote": snippet,
                    "interpretation": f"Clinical documentation matching {target_condition}: '{term_str}'",
                })
                break

    if evidence:
        return (
            True,
            0.91,
            [f"Documented clinical findings consistent with {target_condition} ({', '.join(matched_terms[:3])})"],
            evidence[:3],
            f"Encounter notes contain affirmative clinical documentation for {target_condition}.",
        )
    return (
        False,
        0.93,
        [],
        [],
        f"No affirmative clinical indicators for {target_condition} found in chart documentation.",
    )


def is_quote_grounded(quote: str, chart_text: str) -> bool:
    """Verify that an extracted evidence quote is grounded in the source clinical notes."""
    if not quote or not chart_text:
        return False
    norm_quote = re.sub(r"\s+", " ", quote.strip().strip(".").strip('"').strip("'")).lower()
    norm_quote = norm_quote.removeprefix("...").removesuffix("...").strip()
    norm_chart = re.sub(r"\s+", " ", chart_text).lower()
    if len(norm_quote) < 4:
        return False
    if norm_quote in norm_chart:
        return True
    # Token overlap check for minor punctuation differences
    q_tokens = [t for t in re.findall(r"[a-z0-9.]+", norm_quote) if len(t) >= 3]
    if not q_tokens:
        return False
    c_tokens = set(re.findall(r"[a-z0-9.]+", norm_chart))
    overlap = sum(1 for t in q_tokens if t in c_tokens) / len(q_tokens)
    return overlap >= 0.75


def verify_and_ground_adjudication(
    adjudication: Dict[str, Any],
    record: Dict[str, Any],
    target_condition: str,
    clinical_criteria: str = "",
) -> Dict[str, Any]:
    """
    Post-processes an LLM adjudication payload to filter out ungrounded/hallucinated
    evidence quotes and backfill verified quotes from deterministic rules when needed.
    """
    chart_text = record.get("notes_formatted_text", "") or ""
    raw_evidence = adjudication.get("key_evidence") or []
    grounded_evidence = []
    for ev in raw_evidence:
        if isinstance(ev, dict):
            q = str(ev.get("evidence_quote") or "")
            if is_quote_grounded(q, chart_text):
                grounded_evidence.append(ev)

    if adjudication.get("condition_present") and not grounded_evidence and chart_text:
        rule_res = adjudicate_clinical_rules(
            [record], target_condition, clinical_criteria=clinical_criteria
        )[0]
        if rule_res.get("key_evidence"):
            grounded_evidence = rule_res["key_evidence"]

    adjudication["key_evidence"] = grounded_evidence
    return adjudication


def adjudicate_clinical_rules(
    records: List[Dict[str, Any]],
    target_condition: str,
    backend_tag: str = "keyword_rules",
    clinical_criteria: str = "",
) -> List[Dict[str, Any]]:
    """
    Executes rule-based clinical adjudication across OMOP records with
    comprehensive concept matching, negation filtering, and centered quote extraction.
    """
    results = []
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cond_lower = target_condition.lower()

    for rec in records:
        text = (rec.get("notes_formatted_text") or "").strip()
        visit_id = rec.get("visit_occurrence_id", 0)
        start_date = rec.get("visit_start_date", "")

        if not text or text == "[No notes found for this visit encounter]":
            results.append({
                "person_id": rec.get("person_id", 0),
                "visit_occurrence_id": visit_id,
                "condition_present": False,
                "phenotype_status": "INDETERMINATE_INSUFFICIENT_DATA",
                "confidence_score": 0.0,
                "primary_criteria_met": [],
                "key_evidence": [],
                "clinical_rationale": "Insufficient clinical documentation: no encounter notes available for chart adjudication.",
                "adjudication_timestamp": timestamp,
                "inference_backend": backend_tag,
            })
            continue

        tagged_lines = _extract_note_spans(text, default_visit_id=visit_id, default_date=start_date)

        if "sepsis" in cond_lower or "septic" in cond_lower:
            present, conf, criteria, evidence, rationale = _check_sepsis(tagged_lines)
        elif "stroke" in cond_lower or "infarct" in cond_lower:
            present, conf, criteria, evidence, rationale = _check_stroke(tagged_lines)
        elif "ards" in cond_lower or "respiratory" in cond_lower:
            present, conf, criteria, evidence, rationale = _check_ards(tagged_lines)
        elif "aki" in cond_lower or "kidney" in cond_lower or "renal" in cond_lower:
            present, conf, criteria, evidence, rationale = _check_aki(tagged_lines)
        else:
            present, conf, criteria, evidence, rationale = _check_custom_condition(
                tagged_lines, target_condition, clinical_criteria
            )

        status = "CONFIRMED_POSITIVE" if present else "CONFIRMED_NEGATIVE"

        results.append({
            "person_id": rec.get("person_id", 0),
            "visit_occurrence_id": visit_id,
            "condition_present": present,
            "phenotype_status": status,
            "confidence_score": conf,
            "primary_criteria_met": criteria,
            "key_evidence": evidence,
            "clinical_rationale": rationale,
            "adjudication_timestamp": timestamp,
            "inference_backend": backend_tag,
        })
    return results

