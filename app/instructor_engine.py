import os
import json
import datetime
from typing import Dict, Any, List, Optional
from pydantic import ValidationError

from app.schemas import RoseGoldAdjudication
from app.prompts import SYSTEM_PROMPT, build_adjudication_prompt

class InstructorLlamaAdjudicator:
    """
    Powered by the Instructor framework for LLaMA.
    Guarantees schema validation, Chain-of-Thought reasoning, and self-correcting retry loops.
    """
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        max_retries: int = 3
    ):
        self.model_name = model_name
        self.max_retries = max_retries
        self.base_url = base_url
        self.client = None
        self._init_instructor_client(base_url, api_key)

    def _init_instructor_client(self, base_url: str, api_key: str):
        try:
            import socket
            import urllib.parse
            parsed = urllib.parse.urlparse(base_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 8000
            
            # Quick 50ms socket check
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.05)
                res = sock.connect_ex((host, port))
                if res != 0:
                    self.client = None
                    return

            import instructor
            from openai import OpenAI
            raw_client = OpenAI(base_url=base_url, api_key=api_key, timeout=1.0)
            self.client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
        except Exception:
            self.client = None

    def adjudicate(
        self,
        record: Dict[str, Any],
        target_condition: str,
        clinical_criteria: str
    ) -> Dict[str, Any]:
        """
        Adjudicates a clinical visit record with Instructor self-correcting validation.
        """
        user_prompt = build_adjudication_prompt(
            target_condition=target_condition,
            clinical_criteria=clinical_criteria,
            person_id=record['person_id'],
            visit_id=record['visit_occurrence_id'],
            visit_start=record.get('visit_start_date', 'Unknown'),
            visit_end=record.get('visit_end_date', 'Unknown'),
            notes_formatted_text=record['notes_formatted_text']
        )

        # If connected to active vLLM / OpenAI server
        if self.client is not None:
            try:
                from app.clinical_rules import verify_and_ground_adjudication
                response: RoseGoldAdjudication = self.client.chat.completions.create(
                    model=self.model_name,
                    response_model=RoseGoldAdjudication,
                    max_retries=self.max_retries,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                res_dict = response.model_dump()
                res_dict['person_id'] = record['person_id']
                res_dict['visit_occurrence_id'] = record['visit_occurrence_id']
                res_dict['adjudication_timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                return verify_and_ground_adjudication(
                    res_dict, record, target_condition, clinical_criteria
                )
            except Exception as e:
                print(f"[Instructor] Fallback to engine due to connection: {e}")

        # Local validation & formatting fallback
        return self._local_validated_adjudication(record, target_condition, clinical_criteria)

    def _local_validated_adjudication(
        self,
        record: Dict[str, Any],
        target_condition: str,
        clinical_criteria: str = "",
    ) -> Dict[str, Any]:
        from app.clinical_rules import adjudicate_clinical_rules
        payload = adjudicate_clinical_rules(
            [record],
            target_condition,
            backend_tag="instructor_local",
            clinical_criteria=clinical_criteria,
        )[0]
        validated = RoseGoldAdjudication(**payload)
        res = validated.model_dump()
        res["inference_backend"] = "instructor_local"
        return res

