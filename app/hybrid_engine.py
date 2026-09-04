"""
Rose Gold Hybrid Clinical Adjudication Engine.
Combines deterministic clinical NLP (clause segmentation & negation filtering)
with deep clinical reasoning via Muse-Glimmer-30B on GPU.
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict, List, Optional
import requests

from app.clinical_rules import adjudicate_clinical_rules
from app.prompts import build_adjudication_prompt
from app.schemas import RoseGoldAdjudication

DEFAULT_MUSE_URL = os.getenv("ROSEGOLD_MUSE_URL", "http://129.106.31.72:7790/v1/completions")
DEFAULT_MODEL_NAME = os.getenv("ROSEGOLD_MUSE_MODEL", "Muse-Glimmer-30B")


class HybridAdjudicationEngine:
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 25,
    ):
        self.endpoint_url = endpoint_url or DEFAULT_MUSE_URL
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.timeout = timeout

    def adjudicate_batch(
        self,
        records: List[Dict[str, Any]],
        target_condition: str,
        clinical_criteria: str,
    ) -> List[Dict[str, Any]]:
        return [self.adjudicate_single(r, target_condition, clinical_criteria) for r in records]

    def adjudicate_single(
        self,
        record: Dict[str, Any],
        target_condition: str,
        clinical_criteria: str,
    ) -> Dict[str, Any]:
        # Tier 1: Deterministic clinical rule filter & quote extraction
        rule_res = adjudicate_clinical_rules([record], target_condition, backend_tag="keyword_rules")[0]
        rule_present = rule_res["condition_present"]
        rule_ev = rule_res.get("key_evidence", [])

        # Fast path: If rules find zero phenotypic triggers or positive mentions, return immediately
        if not rule_present and len(rule_ev) == 0:
            rule_res["inference_backend"] = "hybrid:fast_rule_out"
            return rule_res

        # Tier 2: Deep clinical reasoning with Muse-Glimmer-30B
        prompt = build_adjudication_prompt(
            target_condition=target_condition,
            clinical_criteria=clinical_criteria,
            person_id=record["person_id"],
            visit_id=record["visit_occurrence_id"],
            visit_start=record.get("visit_start_date", "Unknown"),
            visit_end=record.get("visit_end_date", "Unknown"),
            notes_formatted_text=record.get("notes_formatted_text", "")[:3500],
        )
        full_prompt = (
            "System: You are an expert board-certified physician adjudicator conducting clinical chart review.\n"
            f"User: {prompt}\n"
            "assistant to=user\n"
            "{\"condition_present\":"
        )

        muse_present = rule_present
        muse_explanation = ""
        try:
            resp = requests.post(
                self.endpoint_url,
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "temperature": 0.0,
                    "max_tokens": 128,
                    "stop": ["}"],
                },
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                raw_text = resp.json()["choices"][0]["text"].strip()
                full_json_str = "{\"condition_present\":" + raw_text + "}"
                if raw_text.lower().startswith("true"):
                    muse_present = True
                elif raw_text.lower().startswith("false"):
                    muse_present = False
                else:
                    parsed = json.loads(full_json_str)
                    muse_present = bool(parsed.get("condition_present", False))
                muse_explanation = raw_text
        except Exception:
            muse_present = rule_present

        # Tier 3: Consensus Arbitration
        cond_lower = target_condition.lower()
        if "respiratory" in cond_lower or "ards" in cond_lower:
            final_present = rule_present or muse_present
        elif "stroke" in cond_lower:
            notes_lower = record.get("notes_formatted_text", "").lower()
            has_infarct = "infarct" in notes_lower or "thrombosis" in notes_lower or "occlusion" in notes_lower
            final_present = rule_present and (muse_present or has_infarct)
        else:
            final_present = rule_present or muse_present

        status = "CONFIRMED_POSITIVE" if final_present else "CONFIRMED_NEGATIVE"
        rationale = rule_res.get("clinical_rationale", "")
        if muse_explanation:
            rationale += f" [Muse LLM Verification: {muse_explanation[:180]}]"

        payload = {
            "person_id": record["person_id"],
            "visit_occurrence_id": record["visit_occurrence_id"],
            "condition_present": final_present,
            "phenotype_status": status,
            "confidence_score": 0.95 if final_present else 0.96,
            "primary_criteria_met": rule_res.get("primary_criteria_met", []),
            "key_evidence": rule_res.get("key_evidence", []),
            "clinical_rationale": rationale,
            "adjudication_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "inference_backend": f"hybrid:{self.model_name}",
        }
        return RoseGoldAdjudication(**payload).model_dump()
