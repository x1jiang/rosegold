"""Vertex Gemini adjudication for Cloud Run / GCP (real hosted LLM).

One malformed or failed model response yields an INDETERMINATE record for that
visit; it never aborts the rest of the batch.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional

from app.prompts import SYSTEM_PROMPT, build_adjudication_prompt
from app.schemas import RoseGoldAdjudication

logger = logging.getLogger("rosegold.vertex")


def _default_model() -> str:
    return os.getenv("ROSEGOLD_VERTEX_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"


def _project_id() -> str:
    return (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("ROSEGOLD_GCP_PROJECT")
        or os.getenv("GCP_PROJECT")
        or ""
    )


def _location() -> str:
    return os.getenv("ROSEGOLD_VERTEX_LOCATION", "us-central1").strip() or "us-central1"


def _extract_balanced_json(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
    end = text.rfind("}")
    return text[start : end + 1] if end > start else text[start:]


def _parse_model_json(text: str) -> RoseGoldAdjudication:
    import re as _re

    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    if not raw.startswith("{") or not raw.endswith("}"):
        raw = _extract_balanced_json(raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Remove trailing commas before } or ]
        repaired = _re.sub(r",\s*([}\]])", r"\1", raw)
        payload = json.loads(repaired)
    if isinstance(payload, dict) and "clinical_rationale" not in payload:
        for value in payload.values():
            if isinstance(value, dict) and (
                "clinical_rationale" in value or "phenotype_status" in value
            ):
                payload = value
                break
    if isinstance(payload, dict) and not payload.get("clinical_rationale"):
        payload["clinical_rationale"] = str(
            payload.get("rationale")
            or payload.get("reasoning")
            or payload.get("explanation")
            or "Clinical chart review completed."
        )
    return RoseGoldAdjudication.model_validate(payload)



class VertexGeminiEngine:
    """Structured JSON adjudication via Vertex AI Gemini."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        client: Any = None,
    ):
        self.model_name = model_name or _default_model()
        self.project = project or _project_id()
        self.location = location or _location()
        if client is not None:
            self.client = client
        else:
            from google import genai

            self.client = genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
            )

    def adjudicate_batch(
        self,
        records: List[Dict[str, Any]],
        target_condition: str,
        clinical_criteria: str,
    ) -> List[Dict[str, Any]]:
        from google.genai import types

        results: List[Dict[str, Any]] = []
        for rec in records:
            user_prompt = build_adjudication_prompt(
                target_condition=target_condition,
                clinical_criteria=clinical_criteria,
                person_id=rec["person_id"],
                visit_id=rec["visit_occurrence_id"],
                visit_start=rec.get("visit_start_date", "Unknown"),
                visit_end=rec.get("visit_end_date", "Unknown"),
                notes_formatted_text=rec["notes_formatted_text"],
            )
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=RoseGoldAdjudication,
                    ),
                )
                parsed = _parse_model_json(response.text)
                from app.clinical_rules import verify_and_ground_adjudication
                payload = verify_and_ground_adjudication(
                    parsed.model_dump(), rec, target_condition, clinical_criteria
                )
            except Exception as exc:
                # Log the failure class only; the prompt contains clinical text.
                logger.warning(
                    "Vertex adjudication failed for visit %s: %s", rec.get("visit_occurrence_id"), type(exc).__name__
                )
                payload = {
                    "clinical_rationale": f"Vertex response unavailable or unparsable ({type(exc).__name__}).",
                    "primary_criteria_met": [],
                    "key_evidence": [],
                    "phenotype_status": "INDETERMINATE_INSUFFICIENT_DATA",
                    "condition_present": False,
                    "confidence_score": 0.0,
                }
            payload["person_id"] = rec["person_id"]
            payload["visit_occurrence_id"] = rec["visit_occurrence_id"]
            payload["adjudication_timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            payload["inference_backend"] = f"vertex:{self.model_name}"
            results.append(payload)
        return results
