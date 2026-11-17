"""Multi-level concordance evaluator comparing human CHoRUS REDCap reviews vs. RoseGold LLM outputs."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from app.chorus_abstractor import _read_csv_flexible, _write_safe_csv, load_chorus_instrument
from app.concordance import calculate_concordance_metrics

_BINARY_ITEMS = [
    "sep1_v3___1",
    "sep1_v3___2",
    "sep1_v3___3",
    "sep1_v3___4",
    "sep4_v3",
    "sep_surg_v3___1",
    "sep_surg_v3___2",
    "sep_surg_v3___3",
    "sep_surg_v3___4",
    "sep_surg_v3___9",
]

_STOP_MICRO_TOKENS = {
    "blood", "urine", "sputum", "bal", "csf", "fluid", "wound", "pleural", "peritoneal",
    "culture", "positive", "pos", "negative", "neg", "source", "other", "growth",
    "organism", "pathogen", "yes", "no", "none",
}


def _norm_gate(val: Any) -> Optional[str]:
    text = str(val or "").strip()
    if not text:
        return None
    low = text.lower()
    if low in {"1", "yes", "y", "true"}:
        return "1"
    if low in {"2", "no", "n", "false"}:
        return "2"
    if low in {"3", "i am not sure", "not sure", "uncertain"}:
        return "3"
    return None


def _norm_binary(val: Any) -> Optional[bool]:
    text = str(val or "").strip()
    if not text:
        return None
    low = text.lower()
    if low in {"1", "yes", "y", "true", "checked", "confirmed"}:
        return True
    if low in {"0", "no", "n", "false", "unchecked"}:
        return False
    return None


def _resolved_sepsis(row: Dict[str, Any]) -> Optional[bool]:
    ync = _norm_gate(row.get("sepsis_ync_v3"))
    if ync == "1":
        return True
    if ync == "2":
        return False
    if ync == "3":
        unc = _norm_gate(row.get("sepsis_uncertain_v3"))
        if unc == "1":
            return True
        if unc == "2":
            return False
    return None


def _parse_culture_pos(val: Any) -> Optional[bool]:
    text = str(val or "").strip()
    if not text:
        return None
    low = text.lower()
    if low in {"no", "none", "negative", "neg", "n/a", "na", "no growth"}:
        return False
    return True


def _organism_tokens(val: str) -> set[str]:
    cleaned = re.sub(r"\(\+\)|\(\-\)|\+", " ", str(val or "").lower())
    tokens = {t for t in re.findall(r"[a-z0-9]+", cleaned) if len(t) >= 2 and t not in _STOP_MICRO_TOKENS}
    return tokens


def _parse_first_float(val: Any) -> Optional[float]:
    text = str(val or "").strip()
    if not text:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _parse_int_list(val: Any) -> List[float]:
    text = str(val or "").strip()
    if not text:
        return []
    nums: List[float] = []
    for part in re.split(r"[,;]+", text):
        v = _parse_first_float(part)
        if v is not None:
            nums.append(v)
    return nums


def _parse_hhmm_minutes(val: Any) -> Optional[int]:
    text = str(val or "").strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::\d{2})?", text)
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    if hh > 23 or mm > 59:
        return None
    return hh * 60 + mm


def _cohens_kappa_multiclass(
    pairs: List[Tuple[str, str]],
    labels: Sequence[str],
    linear_weighted: bool = False,
) -> float:
    n = len(pairs)
    if n == 0:
        return 0.0
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    obs = np.zeros((k, k), dtype=float)
    for h, l in pairs:
        if h in idx and l in idx:
            obs[idx[h], idx[l]] += 1.0
    total = obs.sum()
    if total == 0:
        return 0.0
    row_marg = obs.sum(axis=1)
    col_marg = obs.sum(axis=0)
    exp = np.outer(row_marg, col_marg) / total

    if not linear_weighted:
        po = np.trace(obs) / total
        pe = np.trace(exp) / total
        denom = 1.0 - pe
        if abs(denom) <= 1e-9:
            return 1.0 if abs(po - 1.0) <= 1e-9 else 0.0
        return round(float((po - pe) / denom), 3)

    w = np.zeros((k, k), dtype=float)
    span = max(k - 1, 1)
    for i in range(k):
        for j in range(k):
            w[i, j] = abs(i - j) / span
    num = float(np.sum(w * obs))
    den = float(np.sum(w * exp))
    if abs(den) <= 1e-9:
        return 1.0 if abs(num) <= 1e-9 else 0.0
    return round(float(1.0 - (num / den)), 3)


def _index_rows(rows: List[Dict[str, Any]], use_inst: bool, use_rec: bool, side: str) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    indexed: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for row in rows:
        vid = str(row.get("vid") or "").strip()
        if not vid:
            raise ValueError(f"{side} CSV contains row without 'vid'")
        parts: List[str] = []
        if use_inst:
            parts.append(str(row.get("institution_v3") or "").strip())
        if use_rec:
            parts.append(str(row.get("record_id") or "").strip())
        parts.append(vid)
        key = tuple(parts)
        if key in indexed:
            raise ValueError(f"Duplicate {side} encounter join key: {key}")
        indexed[key] = row
    return indexed


def _discrepancy_category(field: str, h_val: Any, l_val: Any) -> str:
    if field in {"sepsis_ync_v3", "sepsis_uncertain_v3", "gate_resolved"}:
        return "gate_classification_mismatch"
    if field.startswith("sep1_v3"):
        return "sirs_criterion_mismatch"
    if field.startswith("sep_surg_v3"):
        return "surgical_trauma_confounder_mismatch"
    if field in {"sep3_v3", "sep3_v3_positive", "sep4_v3"}:
        return "infection_or_antibiotic_mismatch"
    if field in {"sep6_v3", "sep7_v3"}:
        return "organ_dysfunction_or_lactate_mismatch"
    if field in {"sep_day_v3", "sep_days", "sep_dt_v3"}:
        return "onset_timing_mismatch"
    return "field_mismatch"


def evaluate_chorus_concordance(
    human_rows: List[Dict[str, Any]],
    llm_rows: List[Dict[str, Any]],
    specs: Dict[str, Dict[str, Any]],
    llm_evidence_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute gate, item, numeric, and timing concordance between human and LLM REDCap rows."""
    del specs  # Validated instrument specs available for downstream extensions.
    use_inst = all(str(r.get("institution_v3") or "").strip() for r in human_rows) and all(
        str(r.get("institution_v3") or "").strip() for r in llm_rows
    )
    use_rec = all(str(r.get("record_id") or "").strip() for r in human_rows) and all(
        str(r.get("record_id") or "").strip() for r in llm_rows
    )

    human_map = _index_rows(human_rows, use_inst, use_rec, "human")
    llm_map = _index_rows(llm_rows, use_inst, use_rec, "llm")
    matched_keys = [k for k in human_map if k in llm_map]

    ev_by_visit_field: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    ev_by_visit: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for ev in llm_evidence_rows or []:
        rec = str(ev.get("record_id") or "").strip()
        vid = str(ev.get("vid") or "").strip()
        fld = str(ev.get("field_name") or "").strip()
        if (rec, vid, fld) not in ev_by_visit_field:
            ev_by_visit_field[(rec, vid, fld)] = ev
        if (rec, vid) not in ev_by_visit:
            ev_by_visit[(rec, vid)] = ev

    def _find_ev(row_l: Dict[str, Any], field: str) -> Tuple[str, str]:
        rec = str(row_l.get("record_id") or "").strip()
        vid = str(row_l.get("vid") or "").strip()
        base = field.split("___")[0]
        ev = (
            ev_by_visit_field.get((rec, vid, field))
            or ev_by_visit_field.get((rec, vid, base))
            or ev_by_visit.get((rec, vid))
        )
        if not ev:
            return "", ""
        return str(ev.get("evidence_quote") or ""), str(ev.get("source_ref") or "")

    # 1. 3-way Gate (sepsis_ync_v3)
    labels_3way = ("1", "2", "3")
    matrix_3way = {h: {l: 0 for l in labels_3way} for h in labels_3way}
    gate_pairs: List[Tuple[str, str]] = []
    resolved_rows: List[Dict[str, Any]] = []
    unresolved_h = 0
    unresolved_l = 0
    discrepancies: List[Dict[str, Any]] = []

    def _add_disc(h_row: Dict[str, Any], l_row: Dict[str, Any], field: str, h_val: Any, l_val: Any) -> None:
        quote, ref = _find_ev(l_row, field)
        discrepancies.append({
            "institution_v3": str(h_row.get("institution_v3") or l_row.get("institution_v3") or ""),
            "record_id": str(h_row.get("record_id") or l_row.get("record_id") or ""),
            "vid": str(h_row.get("vid") or l_row.get("vid") or ""),
            "field": field,
            "human_value": "" if h_val is None else str(h_val),
            "llm_value": "" if l_val is None else str(l_val),
            "clinician_conf1": str(h_row.get("conf1") or ""),
            "discrepancy_category": _discrepancy_category(field, h_val, l_val),
            "llm_evidence_quote": quote,
            "llm_source_ref": ref,
        })

    shared_open_keys: List[Tuple[str, ...]] = []
    for k in matched_keys:
        h_row, l_row = human_map[k], llm_map[k]
        h_ync, l_ync = _norm_gate(h_row.get("sepsis_ync_v3")), _norm_gate(l_row.get("sepsis_ync_v3"))
        if h_ync in labels_3way and l_ync in labels_3way:
            matrix_3way[h_ync][l_ync] += 1
            gate_pairs.append((h_ync, l_ync))
            if h_ync != l_ync:
                _add_disc(h_row, l_row, "sepsis_ync_v3", h_ync, l_ync)
        if h_ync == "3" and l_ync == "3":
            h_unc, l_unc = _norm_gate(h_row.get("sepsis_uncertain_v3")), _norm_gate(l_row.get("sepsis_uncertain_v3"))
            if h_unc != l_unc:
                _add_disc(h_row, l_row, "sepsis_uncertain_v3", h_unc or "", l_unc or "")

        h_res = _resolved_sepsis(h_row)
        l_res = _resolved_sepsis(l_row)
        if h_res is None:
            unresolved_h += 1
        if l_res is None:
            unresolved_l += 1
        if h_res is not None and l_res is not None:
            resolved_rows.append({
                "human_positive": h_res,
                "llm_positive": l_res,
                "conf1": str(h_row.get("conf1") or "").strip(),
            })
            if h_res != l_res:
                _add_disc(h_row, l_row, "gate_resolved", "1" if h_res else "0", "1" if l_res else "0")
        if h_res is True and l_res is True:
            shared_open_keys.append(k)

    agree_3 = sum(1 for h, l in gate_pairs if h == l)
    gate_3way = {
        "paired_n": len(gate_pairs),
        "matrix": matrix_3way,
        "overall_agreement_pct": round((agree_3 / len(gate_pairs)) * 100.0, 1) if gate_pairs else 0.0,
        "unweighted_kappa": _cohens_kappa_multiclass(gate_pairs, labels_3way, linear_weighted=False),
        "linear_weighted_kappa": _cohens_kappa_multiclass(gate_pairs, ("2", "3", "1"), linear_weighted=True),
    }

    df_res = pd.DataFrame(resolved_rows) if resolved_rows else pd.DataFrame(columns=["human_positive", "llm_positive", "conf1"])
    gate_resolved = calculate_concordance_metrics(df_res)
    gate_resolved["unresolved_human_n"] = unresolved_h
    gate_resolved["unresolved_llm_n"] = unresolved_l

    by_conf: Dict[str, Any] = {}
    if not df_res.empty and df_res["conf1"].astype(str).str.strip().ne("").any():
        high_df = df_res[df_res["conf1"].isin(["4", "5"])]
        low_df = df_res[df_res["conf1"].isin(["1", "2", "3"])]
        if not high_df.empty:
            by_conf["high_confidence_4_5"] = calculate_concordance_metrics(high_df)
        if not low_df.empty:
            by_conf["low_confidence_1_3"] = calculate_concordance_metrics(low_df)
    gate_resolved["by_clinician_confidence"] = by_conf

    # 2. Item-level concordance inside shared open Sepsis branches
    item_metrics: Dict[str, Any] = {}
    for col in _BINARY_ITEMS:
        pairs_list: List[Dict[str, bool]] = []
        llm_missing = 0
        human_missing = 0
        for k in shared_open_keys:
            h_row, l_row = human_map[k], llm_map[k]
            hb = _norm_binary(h_row.get(col))
            lb = _norm_binary(l_row.get(col))
            if lb is None:
                llm_missing += 1
            if hb is None:
                human_missing += 1
            if hb is not None and lb is not None:
                pairs_list.append({"human_positive": hb, "llm_positive": lb})
                if hb != lb:
                    _add_disc(h_row, l_row, col, "1" if hb else "0", "1" if lb else "0")
        m = calculate_concordance_metrics(pd.DataFrame(pairs_list) if pairs_list else pd.DataFrame())
        m["shared_branch_n"] = len(shared_open_keys)
        m["llm_missing_count"] = llm_missing
        m["human_missing_count"] = human_missing
        item_metrics[col] = m

    # Culture positivity + organism Jaccard
    cult_pairs: List[Dict[str, bool]] = []
    cult_llm_missing = 0
    cult_human_missing = 0
    jaccards: List[float] = []
    for k in shared_open_keys:
        h_row, l_row = human_map[k], llm_map[k]
        hp = _parse_culture_pos(h_row.get("sep3_v3"))
        lp = _parse_culture_pos(l_row.get("sep3_v3"))
        if lp is None:
            cult_llm_missing += 1
        if hp is None:
            cult_human_missing += 1
        if hp is not None and lp is not None:
            cult_pairs.append({"human_positive": hp, "llm_positive": lp})
            if hp != lp:
                _add_disc(h_row, l_row, "sep3_v3_positive", h_row.get("sep3_v3"), l_row.get("sep3_v3"))
            elif hp is True and lp is True:
                htok = _organism_tokens(str(h_row.get("sep3_v3") or ""))
                ltok = _organism_tokens(str(l_row.get("sep3_v3") or ""))
                if htok or ltok:
                    jaccards.append(len(htok & ltok) / len(htok | ltok))
    cult_m = calculate_concordance_metrics(pd.DataFrame(cult_pairs) if cult_pairs else pd.DataFrame())
    cult_m["shared_branch_n"] = len(shared_open_keys)
    cult_m["llm_missing_count"] = cult_llm_missing
    cult_m["human_missing_count"] = cult_human_missing
    cult_m["sep3_organism_jaccard_mean"] = round(float(np.mean(jaccards)), 3) if jaccards else None
    item_metrics["sep3_v3_positive"] = cult_m

    # 3. Numeric & timing metrics
    lac_errs: List[float] = []
    sofa_day0_errs: List[float] = []
    sofa_peak_errs: List[float] = []
    sep_day_errs: List[float] = []
    sep_days_errs: List[float] = []
    combined_hr_errs: List[float] = []

    for k in shared_open_keys:
        h_row, l_row = human_map[k], llm_map[k]

        hl, ll = _parse_first_float(h_row.get("sep6_v3")), _parse_first_float(l_row.get("sep6_v3"))
        if hl is not None and ll is not None:
            err = abs(hl - ll)
            lac_errs.append(err)
            if err > 0.2:
                _add_disc(h_row, l_row, "sep6_v3", h_row.get("sep6_v3"), l_row.get("sep6_v3"))

        hs, ls = _parse_int_list(h_row.get("sep7_v3")), _parse_int_list(l_row.get("sep7_v3"))
        if hs and ls:
            sofa_day0_errs.append(abs(hs[0] - ls[0]))
            peak_err = abs(max(hs) - max(ls))
            sofa_peak_errs.append(peak_err)
            if peak_err > 1.0:
                _add_disc(h_row, l_row, "sep7_v3", h_row.get("sep7_v3"), l_row.get("sep7_v3"))

        hrise, lrise = _parse_first_float(h_row.get("sep_day_v3")), _parse_first_float(l_row.get("sep_day_v3"))
        if hrise is not None and lrise is not None:
            d_err = abs(hrise - lrise)
            sep_day_errs.append(d_err)
            if d_err > 0:
                _add_disc(h_row, l_row, "sep_day_v3", h_row.get("sep_day_v3"), l_row.get("sep_day_v3"))

        h_days, l_days = _parse_first_float(h_row.get("sep_days")), _parse_first_float(l_row.get("sep_days"))
        if h_days is not None and l_days is not None:
            days_err = abs(h_days - l_days)
            sep_days_errs.append(days_err)
            if days_err > 0:
                _add_disc(h_row, l_row, "sep_days", h_row.get("sep_days"), l_row.get("sep_days"))

        h_min, l_min = _parse_hhmm_minutes(h_row.get("sep_dt_v3")), _parse_hhmm_minutes(l_row.get("sep_dt_v3"))
        if h_days is not None and l_days is not None and h_min is not None and l_min is not None:
            h_total_h = h_days * 24.0 + h_min / 60.0
            l_total_h = l_days * 24.0 + l_min / 60.0
            combined_hr_errs.append(abs(h_total_h - l_total_h))

    numeric_and_timing = {
        "sep6_v3": {
            "paired_n": len(lac_errs),
            "mae": round(float(np.mean(lac_errs)), 3) if lac_errs else None,
            "median_ae": round(float(np.median(lac_errs)), 3) if lac_errs else None,
            "within_0_2_mmol_pct": round(100.0 * sum(1 for e in lac_errs if e <= 0.2 + 1e-9) / len(lac_errs), 1) if lac_errs else None,
        },
        "sep7_v3": {
            "paired_n": len(sofa_peak_errs),
            "day0_sofa_mae": round(float(np.mean(sofa_day0_errs)), 3) if sofa_day0_errs else None,
            "peak_sofa_mae": round(float(np.mean(sofa_peak_errs)), 3) if sofa_peak_errs else None,
            "within_1_point_pct": round(100.0 * sum(1 for e in sofa_peak_errs if e <= 1.0 + 1e-9) / len(sofa_peak_errs), 1) if sofa_peak_errs else None,
        },
        "sep_day_v3": {
            "paired_n": len(sep_day_errs),
            "exact_match_pct": round(100.0 * sum(1 for e in sep_day_errs if e == 0.0) / len(sep_day_errs), 1) if sep_day_errs else None,
            "within_1_day_pct": round(100.0 * sum(1 for e in sep_day_errs if e <= 1.0 + 1e-9) / len(sep_day_errs), 1) if sep_day_errs else None,
            "mae_days": round(float(np.mean(sep_day_errs)), 3) if sep_day_errs else None,
        },
        "sep_days": {
            "paired_n": len(sep_days_errs),
            "exact_day_match_pct": round(100.0 * sum(1 for e in sep_days_errs if e == 0.0) / len(sep_days_errs), 1) if sep_days_errs else None,
            "within_1_day_pct": round(100.0 * sum(1 for e in sep_days_errs if e <= 1.0 + 1e-9) / len(sep_days_errs), 1) if sep_days_errs else None,
            "mae_days": round(float(np.mean(sep_days_errs)), 3) if sep_days_errs else None,
            "combined_paired_n": len(combined_hr_errs),
            "combined_onset_mae_hours": round(float(np.mean(combined_hr_errs)), 3) if combined_hr_errs else None,
        },
    }

    return {
        "matched_encounters": len(matched_keys),
        "human_only_encounters": len(human_map) - len(matched_keys),
        "llm_only_encounters": len(llm_map) - len(matched_keys),
        "shared_open_branch_n": len(shared_open_keys),
        "gate_3way": gate_3way,
        "gate_resolved": gate_resolved,
        "item_metrics": item_metrics,
        "numeric_and_timing_metrics": numeric_and_timing,
        "discrepancies": discrepancies,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def format_concordance_markdown(summary: Dict[str, Any]) -> str:
    """Format a human-readable Markdown concordance report."""
    g3 = summary["gate_3way"]
    gr = summary["gate_resolved"]
    nt = summary["numeric_and_timing_metrics"]
    lines = [
        "# CHoRUS Sepsis Concordance Report (Clinician Review vs. RoseGold LLM)",
        "",
        f"- **Matched Encounters:** {summary['matched_encounters']}",
        f"- **Shared Open Sepsis Branches:** {summary['shared_open_branch_n']}",
        "",
        "## 1. Stage-1 Three-Way Gate (`sepsis_ync_v3`: 1=YES, 2=NO, 3=Uncertain)",
        "",
        f"- **Paired N:** {g3['paired_n']}",
        f"- **Exact Agreement:** {g3['overall_agreement_pct']}%",
        f"- **Unweighted Kappa:** {g3['unweighted_kappa']}",
        f"- **Linear-Weighted Kappa (2 < 3 < 1):** {g3['linear_weighted_kappa']}",
        "",
        "| Human \\ LLM | 1 (YES) | 2 (NO) | 3 (Uncertain) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for h_lab, label_txt in (("1", "1 (YES)"), ("2", "2 (NO)"), ("3", "3 (Uncertain)")):
        row = g3["matrix"][h_lab]
        lines.append(f"| **{label_txt}** | {row['1']} | {row['2']} | {row['3']} |")

    lines.extend([
        "",
        "## 2. Resolved Sepsis Gate (After `sepsis_uncertain_v3` Follow-Up)",
        "",
        f"- **Evaluated N:** {gr['total_evaluated']} (TP={gr['tp']}, FP={gr['fp']}, FN={gr['fn']}, TN={gr['tn']})",
        f"- **Agreement:** {gr['overall_agreement_pct']}% | **Cohen's Kappa:** {gr['cohens_kappa']}",
        f"- **Sensitivity:** {gr['sensitivity']}% | **Specificity:** {gr['specificity']}% | **PPV:** {gr['ppv']}% | **NPV:** {gr['npv']}%",
    ])
    if gr.get("by_clinician_confidence"):
        lines.append("")
        lines.append("### Stratified by Clinician Confidence (`conf1`)")
        for stratum, metrics in gr["by_clinician_confidence"].items():
            lines.append(
                f"- **{stratum}**: N={metrics['total_evaluated']}, Agreement={metrics['overall_agreement_pct']}%, Kappa={metrics['cohens_kappa']}"
            )

    lines.extend([
        "",
        "## 3. Item-Level Concordance (Shared Open Sepsis Branches)",
        "",
        "| Field | Paired N | Agreement (%) | Kappa | Sensitivity (%) | Specificity (%) | LLM Missing |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for field, m in summary["item_metrics"].items():
        lines.append(
            f"| `{field}` | {m['total_evaluated']} | {m['overall_agreement_pct']} | {m['cohens_kappa']} | "
            f"{m['sensitivity']} | {m['specificity']} | {m['llm_missing_count']} |"
        )
    if summary["item_metrics"].get("sep3_v3_positive", {}).get("sep3_organism_jaccard_mean") is not None:
        lines.append("")
        lines.append(
            f"- **Mean Positive Culture Organism Token Jaccard (`sep3_v3`):** "
            f"{summary['item_metrics']['sep3_v3_positive']['sep3_organism_jaccard_mean']}"
        )

    lines.extend([
        "",
        "## 4. Quantitative & Timing Accuracy",
        "",
        "| Metric | Paired N | Error / Accuracy Summary |",
        "| --- | ---: | --- |",
        f"| `sep6_v3` (Highest Lactate) | {nt['sep6_v3']['paired_n']} | MAE={nt['sep6_v3']['mae']} mmol/L; Within 0.2 mmol/L={nt['sep6_v3']['within_0_2_mmol_pct']}% |",
        f"| `sep7_v3` (Daily SOFA) | {nt['sep7_v3']['paired_n']} | Day-0 MAE={nt['sep7_v3']['day0_sofa_mae']}; Peak MAE={nt['sep7_v3']['peak_sofa_mae']}; Within 1 pt={nt['sep7_v3']['within_1_point_pct']}% |",
        f"| `sep_day_v3` (SOFA Rise Day) | {nt['sep_day_v3']['paired_n']} | Exact={nt['sep_day_v3']['exact_match_pct']}%; Within 1d={nt['sep_day_v3']['within_1_day_pct']}%; MAE={nt['sep_day_v3']['mae_days']} d |",
        f"| `sep_days` & `sep_dt_v3` (Onset) | {nt['sep_days']['paired_n']} | Exact day={nt['sep_days']['exact_day_match_pct']}%; Within 1d={nt['sep_days']['within_1_day_pct']}%; Combined MAE={nt['sep_days']['combined_onset_mae_hours']} h |",
        "",
        f"## 5. Field Discrepancies ({len(summary['discrepancies'])} total)",
        "",
    ])
    for d in summary["discrepancies"][:20]:
        lines.append(
            f"- `record_id={d['record_id']}`, `vid={d['vid']}`, `{d['field']}`: "
            f"Human=`{d['human_value']}` vs LLM=`{d['llm_value']}` ({d['discrepancy_category']})"
        )
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-csv", type=Path, required=True, help="Clinician raw-coded REDCap CSV")
    parser.add_argument("--llm-csv", type=Path, required=True, help="RoseGold REDCap CSV output")
    parser.add_argument("--dictionary", type=Path, required=True, help="Canonical CHoRUS REDCap dictionary CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for concordance outputs")
    parser.add_argument("--llm-evidence", type=Path, default=None, help="Optional field_evidence.csv from chorus_abstractor")
    args = parser.parse_args(argv)

    specs = load_chorus_instrument(args.dictionary)
    _, human_rows = _read_csv_flexible(args.human_csv)
    _, llm_rows = _read_csv_flexible(args.llm_csv)
    ev_rows: List[Dict[str, str]] = []
    if args.llm_evidence and args.llm_evidence.exists():
        _, ev_rows = _read_csv_flexible(args.llm_evidence)

    summary = evaluate_chorus_concordance(human_rows, llm_rows, specs, llm_evidence_rows=ev_rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "concordance_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "concordance_report.md").write_text(format_concordance_markdown(summary), encoding="utf-8")

    disc_cols = [
        "institution_v3", "record_id", "vid", "field", "human_value", "llm_value",
        "clinician_conf1", "discrepancy_category", "llm_evidence_quote", "llm_source_ref",
    ]
    _write_safe_csv(out / "discrepancies.csv", summary["discrepancies"], disc_cols)
    print(json.dumps({
        "matched_encounters": summary["matched_encounters"],
        "gate_3way_agreement_pct": summary["gate_3way"]["overall_agreement_pct"],
        "gate_resolved_kappa": summary["gate_resolved"]["cohens_kappa"],
        "discrepancies": len(summary["discrepancies"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
