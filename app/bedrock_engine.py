"""Native AWS Bedrock Converse adjudication.

Use this when the site-approved path is IAM + Bedrock Runtime rather than
an OpenAI-compatible URL.

If ``ROSEGOLD_OPENAI_BASE_URL`` is set with ``ROSEGOLD_LLM_BACKEND=bedrock``,
the OpenAI-compatible engine is used instead — see ``app.engine``.

``boto3`` is an optional extra (``pip install -r requirements-bedrock.txt``).
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import Any, Dict, List, Optional

from app.hosted_llm import failed_adjudication, first_env, json_schema_instruction
from app.prompts import SYSTEM_PROMPT, build_adjudication_prompt
from app.vertex_engine import _parse_model_json

logger = logging.getLogger("rosegold.bedrock")


def resolve_region() -> str:
    return first_env("ROSEGOLD_BEDROCK_REGION", "AWS_REGION", "AWS_DEFAULT_REGION") or "us-west-2"


def resolve_model_id(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit.strip()
    name = first_env("ROSEGOLD_BEDROCK_MODEL", "ROSEGOLD_MODEL_NAME")
    if name and name.lower() != "auto":
        return name
    raise ValueError(
        "ROSEGOLD_BEDROCK_MODEL is not set. Use the Bedrock model id or inference "
        "profile approved at your site (for example anthropic.claude-3-5-sonnet-*)."
    )


class BedrockEngine:
    """Structured JSON adjudication via Bedrock Runtime ``converse``."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        region: Optional[str] = None,
        client: Any = None,
    ):
        self.model_name = resolve_model_id(model_name)
        self.region = (region or resolve_region()).strip()
        if client is not None:
            self.client = client
        else:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError(
                    "boto3 is required for the native Bedrock backend. "
                    "Install with: pip install -r requirements-bedrock.txt"
                ) from exc
            kwargs: Dict[str, Any] = {"region_name": self.region}
            profile = os.getenv("AWS_PROFILE", "").strip() or os.getenv("ROSEGOLD_BEDROCK_PROFILE", "").strip()
            if profile:
                session = boto3.Session(profile_name=profile)
                self.client = session.client("bedrock-runtime", region_name=self.region)
            else:
                self.client = boto3.client("bedrock-runtime", **kwargs)

    def adjudicate_batch(
        self,
        records: List[Dict[str, Any]],
        target_condition: str,
        clinical_criteria: str,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        system = f"{SYSTEM_PROMPT}\n\n{json_schema_instruction()}"
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
                response = self.client.converse(
                    modelId=self.model_name,
                    system=[{"text": system}],
                    messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                    inferenceConfig={"temperature": 0.0, "maxTokens": 2048},
                )
                parts = response["output"]["message"]["content"]
                text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
                parsed = _parse_model_json(text)
                payload = parsed.model_dump()
            except Exception as exc:
                logger.warning(
                    "Bedrock adjudication failed for visit %s: %s",
                    rec.get("visit_occurrence_id"),
                    type(exc).__name__,
                )
                payload = failed_adjudication(exc)
            payload["person_id"] = rec["person_id"]
            payload["visit_occurrence_id"] = rec["visit_occurrence_id"]
            payload["adjudication_timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            payload["inference_backend"] = f"bedrock:{self.model_name}"
            results.append(payload)
        return results
