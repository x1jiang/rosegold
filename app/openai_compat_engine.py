"""OpenAI-compatible hosted LLM adjudication.

Covers Databricks Model Serving, AWS Bedrock's OpenAI-compatible endpoint,
Azure OpenAI, and any hospital gateway that speaks ``/v1/chat/completions``.

Notes are POSTed only to the operator-configured base URL. There is no default.
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import Any, Dict, List, Optional

from app.hosted_llm import (
    failed_adjudication,
    first_env,
    hosted_timeout_seconds,
    json_schema_instruction,
    mantle_base_url,
    validate_https_endpoint,
)
from app.prompts import SYSTEM_PROMPT, build_adjudication_prompt
from app.vertex_engine import _parse_model_json

logger = logging.getLogger("rosegold.openai_compat")

_TOKEN_REQUIRED = {"databricks", "mantle", "bedrock"}
_MANTLE_ALIASES = {"mantle", "bedrock_mantle", "bedrock"}
_BACKEND_TAGS = {
    "databricks": "databricks",
    "openai": "openai",
    "openai_compat": "openai",
    "hosted": "openai",
    "mantle": "mantle",
    "bedrock_mantle": "mantle",
    "bedrock": "mantle",
    "aws_bedrock": "mantle",
}


def _backend_tag() -> str:
    name = os.getenv("ROSEGOLD_LLM_BACKEND", "").lower().strip()
    return _BACKEND_TAGS.get(name, "openai")


def resolve_base_url(explicit: Optional[str] = None, backend_tag: Optional[str] = None) -> str:
    if explicit:
        return validate_https_endpoint(explicit, env_name="ROSEGOLD_OPENAI_BASE_URL")
    direct = os.getenv("ROSEGOLD_OPENAI_BASE_URL", "").strip()
    if direct:
        return validate_https_endpoint(direct, env_name="ROSEGOLD_OPENAI_BASE_URL")
    tag = backend_tag or _backend_tag()
    if tag in _MANTLE_ALIASES or os.getenv("ROSEGOLD_LLM_BACKEND", "").lower().strip() in {
        "mantle",
        "bedrock_mantle",
        "bedrock",
        "aws_bedrock",
    }:
        return mantle_base_url()
    host = os.getenv("DATABRICKS_HOST", "").strip()
    if host:
        if "://" not in host:
            host = f"https://{host}"
        host = host.rstrip("/")
        if not host.endswith("/serving-endpoints"):
            host = f"{host}/serving-endpoints"
        return validate_https_endpoint(host, env_name="DATABRICKS_HOST")
    raise ValueError(
        "ROSEGOLD_OPENAI_BASE_URL is not set. For Bedrock Mantle, set AWS_REGION "
        "and a Bedrock API key. For Databricks, set DATABRICKS_HOST."
    )


def resolve_api_key() -> str:
    return first_env(
        "ROSEGOLD_OPENAI_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "DATABRICKS_TOKEN",
        "DATABRICKS_API_TOKEN",
        "OPENAI_API_KEY",
    )


def resolve_model_name(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit.strip()
    name = first_env(
        "ROSEGOLD_OPENAI_MODEL",
        "ROSEGOLD_BEDROCK_MODEL",
        "DATABRICKS_SERVING_ENDPOINT",
        "ROSEGOLD_MODEL_NAME",
    )
    if name and name.lower() != "auto":
        return name
    raise ValueError(
        "ROSEGOLD_OPENAI_MODEL is not set. Use the site-approved Bedrock Mantle "
        "model id (or Databricks serving-endpoint name)."
    )


class OpenAICompatEngine:
    """Structured JSON adjudication via an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Any = None,
        timeout: Optional[float] = None,
        backend_tag: Optional[str] = None,
    ):
        self.backend_tag = backend_tag or _backend_tag()
        self.base_url = resolve_base_url(base_url, backend_tag=self.backend_tag)
        self.api_key = api_key if api_key is not None else resolve_api_key()
        self.model_name = resolve_model_name(model_name)
        self.timeout = float(timeout) if timeout is not None else hosted_timeout_seconds()
        if self.backend_tag in _TOKEN_REQUIRED and not self.api_key:
            if self.backend_tag in _MANTLE_ALIASES:
                raise ValueError(
                    "AWS_BEARER_TOKEN_BEDROCK (or ROSEGOLD_OPENAI_API_KEY) is required "
                    "for Bedrock Mantle."
                )
            raise ValueError(
                "DATABRICKS_TOKEN (or ROSEGOLD_OPENAI_API_KEY) is required for the "
                "Databricks backend."
            )
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI

            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "EMPTY",
                timeout=self.timeout,
            )

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
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                text = response.choices[0].message.content or ""
                parsed = _parse_model_json(text)
                payload = parsed.model_dump()
            except Exception as exc:
                logger.warning(
                    "Hosted LLM adjudication failed for visit %s: %s",
                    rec.get("visit_occurrence_id"),
                    type(exc).__name__,
                )
                payload = failed_adjudication(exc)
            payload["person_id"] = rec["person_id"]
            payload["visit_occurrence_id"] = rec["visit_occurrence_id"]
            payload["adjudication_timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            payload["inference_backend"] = f"{self.backend_tag}:{self.model_name}"
            results.append(payload)
        return results
