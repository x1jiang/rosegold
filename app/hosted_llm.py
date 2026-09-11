"""Shared helpers for hospital-approved hosted LLM backends.

Rose Gold is the chart-review layer (OMOP ingest, phenotypes, evidence quotes,
audit, OMOP export). It does not ship a model that a site must install.
These helpers send notes only to an operator-configured HTTPS endpoint —
Databricks Model Serving, AWS Bedrock, or any OpenAI-compatible hospital gateway.

There is no default endpoint. Clinical text must not leave the site's approved
boundary unless the operator points it there.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import urllib.parse
from typing import Any, Dict, Optional

_REGION_RE = re.compile(r"^[a-z]{2}(?:-[a-z0-9]+)+-\d+$")
_MANTLE_PATHS = {"v1", "openai/v1"}

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def allow_http(env_name: str = "ROSEGOLD_OPENAI_ALLOW_HTTP") -> bool:
    return os.getenv(env_name, "").lower() in {"1", "true", "yes"}


def is_loopback(host: Optional[str]) -> bool:
    if not host:
        return False
    if host.lower() in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_https_endpoint(
    url: Optional[str],
    *,
    env_name: str,
    allow_http_env: str = "ROSEGOLD_OPENAI_ALLOW_HTTP",
) -> str:
    """Return ``url`` if it is an acceptable place to send clinical text."""
    text = (url or "").strip()
    if not text:
        raise ValueError(
            f"{env_name} is not set. Hosted backends have no default endpoint; "
            "point Rose Gold at your site-approved Databricks, Bedrock, or "
            "OpenAI-compatible serving URL."
        )
    parsed = urllib.parse.urlparse(text)
    if not parsed.netloc or parsed.scheme not in {"http", "https"}:
        raise ValueError(f"{env_name} must be an absolute http(s) URL.")
    if parsed.scheme == "http" and not (is_loopback(parsed.hostname) or allow_http(allow_http_env)):
        raise ValueError(
            f"{env_name} uses plain http:// to a non-loopback host. Use https://, "
            f"or set {allow_http_env}=1 if the endpoint is on a trusted private network."
        )
    return text.rstrip("/")


def redact_secret(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if len(text) <= 8:
        return "****"
    return f"{text[:4]}…{text[-2:]} ({len(text)} chars)"


def first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def resolve_aws_region(explicit: str = "") -> str:
    raw = (explicit or first_env("ROSEGOLD_BEDROCK_REGION", "AWS_REGION", "AWS_DEFAULT_REGION") or "us-west-2").strip()
    if not _REGION_RE.fullmatch(raw):
        raise ValueError(
            f"Invalid AWS region {raw!r}. Set ROSEGOLD_BEDROCK_REGION or AWS_REGION "
            "(for example us-west-2)."
        )
    return raw


def mantle_base_url(region: str = "") -> str:
    """OpenAI-compatible Bedrock Mantle Chat Completions base.

    Default: ``https://bedrock-mantle.{region}.api.aws/v1``
    Some models use ``.../openai/v1`` — set ``ROSEGOLD_MANTLE_PATH=openai/v1``.
    """
    path = os.getenv("ROSEGOLD_MANTLE_PATH", "v1").strip().strip("/") or "v1"
    if path not in _MANTLE_PATHS:
        raise ValueError("ROSEGOLD_MANTLE_PATH must be 'v1' or 'openai/v1'.")
    return validate_https_endpoint(
        f"https://bedrock-mantle.{resolve_aws_region(region)}.api.aws/{path}",
        env_name="ROSEGOLD_OPENAI_BASE_URL",
    )


def hosted_timeout_seconds(default: float = 45.0) -> float:
    try:
        return max(1.0, float(os.getenv("ROSEGOLD_OPENAI_TIMEOUT", str(default))))
    except ValueError:
        return default


def failed_adjudication(exc: BaseException) -> Dict[str, Any]:
    """INDETERMINATE payload for one visit. Never include note text."""
    return {
        "clinical_rationale": (
            f"Hosted LLM response unavailable or unparsable ({type(exc).__name__})."
        ),
        "primary_criteria_met": [],
        "key_evidence": [],
        "phenotype_status": "INDETERMINATE_INSUFFICIENT_DATA",
        "condition_present": False,
        "confidence_score": 0.0,
    }


def json_schema_instruction() -> str:
    from app.schemas import RoseGoldAdjudication

    schema = json.dumps(RoseGoldAdjudication.model_json_schema())
    return (
        "Respond with a single JSON object matching this schema and no other text.\n"
        f"{schema}"
    )


def describe_hosted_config() -> Dict[str, Any]:
    """Operator check: what is configured, with secrets redacted. No network call."""
    backend = os.getenv("ROSEGOLD_LLM_BACKEND", "").strip() or "(unset)"
    token = first_env(
        "ROSEGOLD_OPENAI_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "DATABRICKS_TOKEN",
        "DATABRICKS_API_TOKEN",
        "OPENAI_API_KEY",
    )
    base = first_env("ROSEGOLD_OPENAI_BASE_URL", "DATABRICKS_HOST")
    region = first_env("ROSEGOLD_BEDROCK_REGION", "AWS_REGION", "AWS_DEFAULT_REGION")
    computed_mantle = ""
    try:
        computed_mantle = mantle_base_url()
    except ValueError:
        computed_mantle = ""
    return {
        "backend": backend,
        "base_url_or_host": base or computed_mantle,
        "mantle_base_url": computed_mantle,
        "model": first_env(
            "ROSEGOLD_OPENAI_MODEL",
            "ROSEGOLD_BEDROCK_MODEL",
            "ROSEGOLD_MODEL_NAME",
            "DATABRICKS_SERVING_ENDPOINT",
        ),
        "api_key": redact_secret(token),
        "api_key_present": bool(token),
        "aws_region": region or "us-west-2",
        "timeout_seconds": hosted_timeout_seconds(),
        "phi_stays_at_this_endpoint": True,
        "docker_required": False,
        "singularity_required": False,
    }


if __name__ == "__main__":
    print(json.dumps(describe_hosted_config(), indent=2))
