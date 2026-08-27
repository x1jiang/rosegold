"""Vertex Gemini adjudication for Cloud Run / GCP (real hosted LLM)."""

from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict, List, Optional

from app.prompts import SYSTEM_PROMPT, build_adjudication_prompt
from app.schemas import RoseGoldAdjudication


def _default_model() -> str:
    return os.getenv("ROSEGOLD_VERTEX_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"


def _project_id() -> str:
    return (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("ROSEGOLD_GCP_PROJECT")
        or os.getenv("GCP_PROJECT")
        or "sbmi-jiang-ai-testing01"
    )


def _location() -> str:
    return os.getenv("ROSEGOLD_VERTEX_LOCATION", "us-central1").strip() or "us-central1"


def _parse_model_json(text: str) -> RoseGoldAdjudication:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[: -3]
        raw = raw.strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    payload = json.loads(raw)
    if isinstance(payload, dict) and "clinical_rationale" not in payload:
        for value in payload.values():
            if isinstance(value, dict) and "clinical_rationale" in value:
                payload = value
                break
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
            payload = parsed.model_dump()
            payload["person_id"] = rec["person_id"]
            payload["visit_occurrence_id"] = rec["visit_occurrence_id"]
            payload["adjudication_timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            payload["inference_backend"] = f"vertex:{self.model_name}"
            results.append(payload)
        return results
