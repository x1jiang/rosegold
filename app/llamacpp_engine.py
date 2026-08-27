"""Real Llama inference via llama.cpp on CPU."""

from __future__ import annotations

import datetime
import json
from typing import Any, Dict, List, Optional

from app.model_store import ensure_llama_gguf, model_display_name
from app.prompts import SYSTEM_PROMPT, build_adjudication_prompt
from app.schemas import RoseGoldAdjudication
from app.vertex_engine import _parse_model_json

_COMPACT_SCHEMA = """{
  "clinical_rationale": "step-by-step reasoning grounded in the notes",
  "primary_criteria_met": ["criterion"],
  "key_evidence": [{"note_id": 40001, "note_date": "2026-03-01", "evidence_quote": "verbatim excerpt", "interpretation": "why it matters"}],
  "phenotype_status": "CONFIRMED_POSITIVE",
  "condition_present": true,
  "confidence_score": 0.8
}
phenotype_status must be one of CONFIRMED_POSITIVE, SUSPECTED_PROBABLE, CONFIRMED_NEGATIVE, INDETERMINATE_INSUFFICIENT_DATA."""


class LlamaCppEngine:
    def __init__(
        self,
        model_path: Optional[str] = None,
        llm: Any = None,
        n_ctx: int = 8192,
        n_threads: Optional[int] = None,
    ):
        self.model_name = model_display_name()
        if llm is not None:
            self.llm = llm
            self.model_path = model_path or "injected"
            return

        path = model_path or ensure_llama_gguf()
        self.model_path = path
        from llama_cpp import Llama

        threads = n_threads or min(4, os_cpu_count())
        print(f"[Llama] Loading {self.model_name} from {path} (n_ctx={n_ctx}, threads={threads})", flush=True)
        self.llm = Llama(
            model_path=path,
            n_ctx=n_ctx,
            n_threads=threads,
            n_batch=256,
            logits_all=False,
            verbose=False,
        )
        print("[Llama] Weights loaded.", flush=True)

    def adjudicate_batch(
        self,
        records: List[Dict[str, Any]],
        target_condition: str,
        clinical_criteria: str,
    ) -> List[Dict[str, Any]]:
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
            kwargs: Dict[str, Any] = dict(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"{SYSTEM_PROMPT}\n"
                            "Return one flat JSON object only. Do not wrap it in another key.\n"
                            f"Shape:\n{_COMPACT_SCHEMA}"
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            try:
                response = self.llm.create_chat_completion(
                    **kwargs,
                    response_format={"type": "json_object"},
                )
            except TypeError:
                response = self.llm.create_chat_completion(**kwargs)
            text = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            try:
                parsed = _parse_model_json(text)
                payload = parsed.model_dump()
            except Exception as exc:
                payload = {
                    "clinical_rationale": f"Llama JSON parse failed: {exc}. Raw: {str(text)[:400]}",
                    "primary_criteria_met": [],
                    "key_evidence": [],
                    "phenotype_status": "INDETERMINATE_INSUFFICIENT_DATA",
                    "condition_present": False,
                    "confidence_score": 0.0,
                }
            payload["person_id"] = rec["person_id"]
            payload["visit_occurrence_id"] = rec["visit_occurrence_id"]
            payload["adjudication_timestamp"] = timestamp
            payload["inference_backend"] = f"llamacpp:{self.model_name}"
            results.append(payload)
        return results


def os_cpu_count() -> int:
    import os

    return os.cpu_count() or 2
