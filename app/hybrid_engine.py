"""
Rose Gold Hybrid Clinical Adjudication Engine.
Combines deterministic clinical NLP (clause segmentation & negation filtering)
with deep clinical reasoning via Muse-Glimmer-30B on GPU.

Hardening
---------
* There is **no default endpoint**. Clinical notes are sent to the Muse
  completions URL, so it must be set explicitly via ``ROSEGOLD_MUSE_URL``.
* The URL must be ``https://``. Plain ``http://`` is accepted only for loopback
  hosts, or when ``ROSEGOLD_MUSE_ALLOW_HTTP=1`` is set for a private network.
* ``ROSEGOLD_MUSE_API_KEY`` (optional) is sent as a bearer token.
* When the LLM call fails, the result is labelled ``hybrid:rules_only(...)`` with
  a reduced confidence instead of claiming an LLM verification that never ran.
"""

from __future__ import annotations

import datetime
import ipaddress
import json
import logging
import os
import urllib.parse
from typing import Any, Dict, List, Optional
import requests

from app.clinical_rules import adjudicate_clinical_rules
from app.prompts import build_adjudication_prompt
from app.schemas import RoseGoldAdjudication

logger = logging.getLogger("rosegold.hybrid")

DEFAULT_MODEL_NAME = "Muse-Glimmer-30B"
RULES_ONLY_CONFIDENCE = 0.70
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _allow_http() -> bool:
    return os.getenv("ROSEGOLD_MUSE_ALLOW_HTTP", "").lower() in {"1", "true", "yes"}


def _is_loopback(host: Optional[str]) -> bool:
    if not host:
        return False
    if host.lower() in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_muse_url(url: Optional[str]) -> str:
    """Return ``url`` if it is an acceptable place to send clinical text, else raise ``ValueError``."""
    text = (url or "").strip()
    if not text:
        raise ValueError(
            "ROSEGOLD_MUSE_URL is not set. The hybrid backend has no default endpoint; "
            "point it at your Muse/vLLM completions URL."
        )
    parsed = urllib.parse.urlparse(text)
    if not parsed.netloc or parsed.scheme not in {"http", "https"}:
        raise ValueError("ROSEGOLD_MUSE_URL must be an absolute http(s) URL.")
    if parsed.scheme == "http" and not (_is_loopback(parsed.hostname) or _allow_http()):
        raise ValueError(
            "ROSEGOLD_MUSE_URL uses plain http:// to a non-loopback host. Use https://, "
            "or set ROSEGOLD_MUSE_ALLOW_HTTP=1 if the endpoint is on a trusted private network."
        )
    return text


def muse_url() -> str:
    return validate_muse_url(os.getenv("ROSEGOLD_MUSE_URL"))


def muse_model_name() -> str:
    return os.getenv("ROSEGOLD_MUSE_MODEL", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME


def muse_timeout() -> float:
    try:
        return max(1.0, float(os.getenv("ROSEGOLD_MUSE_TIMEOUT", "25")))
    except ValueError:
        return 25.0


def _auth_headers() -> Dict[str, str]:
    key = os.getenv("ROSEGOLD_MUSE_API_KEY", "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


class HybridAdjudicationEngine:
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
        session: Optional[requests.Session] = None,
    ):
        self.endpoint_url = validate_muse_url(endpoint_url) if endpoint_url else muse_url()
        self.model_name = model_name or muse_model_name()
        self.timeout = float(timeout) if timeout is not None else muse_timeout()
        self._session = session or requests.Session()

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
        llm_verified = False
        llm_failure = ""
        try:
            resp = self._session.post(
                self.endpoint_url,
                headers=_auth_headers(),
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
                raw_text = str(resp.json()["choices"][0]["text"]).strip()
                full_json_str = "{\"condition_present\":" + raw_text + "}"
                if raw_text.lower().startswith("true"):
                    muse_present = True
                elif raw_text.lower().startswith("false"):
                    muse_present = False
                else:
                    parsed = json.loads(full_json_str)
                    muse_present = bool(parsed.get("condition_present", False))
                muse_explanation = raw_text
                llm_verified = True
            else:
                llm_failure = f"http_{resp.status_code}"
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            llm_failure = type(exc).__name__
        if not llm_verified:
            # Log at warning level but never include note text or the raw prompt.
            logger.warning(
                "Muse LLM verification unavailable for visit %s (%s); falling back to rules-only verdict",
                record.get("visit_occurrence_id"),
                llm_failure or "unknown",
            )
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
        if llm_verified:
            if muse_explanation:
                rationale += f" [Muse LLM Verification: {muse_explanation[:180]}]"
            confidence = 0.95 if final_present else 0.96
            backend_tag = f"hybrid:{self.model_name}"
        else:
            rationale += " [Muse LLM unavailable; verdict is from deterministic rules only and needs physician review]"
            confidence = RULES_ONLY_CONFIDENCE
            backend_tag = f"hybrid:rules_only({llm_failure or 'llm_unavailable'})"

        payload = {
            "person_id": record["person_id"],
            "visit_occurrence_id": record["visit_occurrence_id"],
            "condition_present": final_present,
            "phenotype_status": status,
            "confidence_score": confidence,
            "primary_criteria_met": rule_res.get("primary_criteria_met", []),
            "key_evidence": rule_res.get("key_evidence", []),
            "clinical_rationale": rationale,
            "adjudication_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "inference_backend": backend_tag,
        }
        return RoseGoldAdjudication(**payload).model_dump()
