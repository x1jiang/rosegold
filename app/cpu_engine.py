import json
import datetime
from typing import Any, Dict, List, Optional

from app.schemas import RoseGoldAdjudication
from app.prompts import SYSTEM_PROMPT, build_adjudication_prompt

class CPULlamaGemmaEngine:
    """
    Dedicated CPU Inference Engine for LLaMA and Gemma architectures.
    Enables hospital sites without GPUs (or local laptops) to run clinical chart reviews.
    """
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        max_new_tokens: int = 512,
        load_in_memory: bool = False
    ):
        self.model_name = model_name
        self.device_name = "cpu"
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        self.is_hf_loaded = False

        if load_in_memory:
            self._load_hf_model()

    def _load_hf_model(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            if torch.cuda.is_available():
                device = torch.device("cuda")
                dtype = torch.float16
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                dtype = torch.float16
            else:
                device = torch.device("cpu")
                dtype = torch.float32
            self.device_name = str(device)
            print(f"[CPU Engine] Loading {self.model_name} on {device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(device)
            self.model.eval()
            self.is_hf_loaded = True
            print(f"[CPU Engine] Successfully loaded {self.model_name} on {device}.")
        except Exception as e:
            print(f"[CPU Engine] HuggingFace CPU weight loading not triggered ({e}). Using optimized CPU validation mode.")
            self.is_hf_loaded = False

    def adjudicate_single(
        self,
        record: Dict[str, Any],
        target_condition: str,
        clinical_criteria: str
    ) -> Dict[str, Any]:
        """Adjudicates a single record on CPU."""
        return self.adjudicate_batch([record], target_condition, clinical_criteria)[0]

    def adjudicate_batch(
        self,
        records: List[Dict[str, Any]],
        target_condition: str,
        clinical_criteria: str
    ) -> List[Dict[str, Any]]:
        """Executes CPU batch adjudication."""
        if self.is_hf_loaded and self.model is not None and self.tokenizer is not None:
            return self._run_hf_cpu_generation(records, target_condition, clinical_criteria)
        return self._cpu_rule_adjudicate(records, target_condition)

    def _run_hf_cpu_generation(
        self,
        records: List[Dict[str, Any]],
        target_condition: str,
        clinical_criteria: str
    ) -> List[Dict[str, Any]]:
        results = []
        schema_json = json.dumps(RoseGoldAdjudication.model_json_schema())
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for rec in records:
            user_prompt = build_adjudication_prompt(
                target_condition=target_condition,
                clinical_criteria=clinical_criteria,
                person_id=rec['person_id'],
                visit_id=rec['visit_occurrence_id'],
                visit_start=rec.get('visit_start_date', 'Unknown'),
                visit_end=rec.get('visit_end_date', 'Unknown'),
                notes_formatted_text=rec['notes_formatted_text']
            )

            messages = [
                {"role": "system", "content": f"{SYSTEM_PROMPT}\nOutput must conform strictly to JSON Schema:\n{schema_json}"},
                {"role": "user", "content": user_prompt}
            ]

            prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            import torch

            inputs = self.tokenizer(prompt_text, return_tensors="pt")
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            gen_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

            try:
                # Extract JSON block
                if "{" in gen_text and "}" in gen_text:
                    json_str = gen_text[gen_text.find("{"):gen_text.rfind("}")+1]
                    parsed = json.loads(json_str)
                else:
                    parsed = json.loads(gen_text)
                
                parsed['person_id'] = rec['person_id']
                parsed['visit_occurrence_id'] = rec['visit_occurrence_id']
                parsed['adjudication_timestamp'] = timestamp
                results.append(RoseGoldAdjudication(**parsed).model_dump())
            except Exception as e:
                results.append(self._fallback_record(rec, str(e), timestamp))

        return results

    def _cpu_rule_adjudicate(self, records: List[Dict[str, Any]], target_condition: str) -> List[Dict[str, Any]]:
        from app.clinical_rules import adjudicate_clinical_rules
        return adjudicate_clinical_rules(records, target_condition, backend_tag="cpu_rules")

    def _fallback_record(self, rec: Dict[str, Any], err_msg: str, timestamp: str) -> Dict[str, Any]:
        return {
            'person_id': rec['person_id'],
            'visit_occurrence_id': rec['visit_occurrence_id'],
            'condition_present': False,
            'phenotype_status': 'INDETERMINATE_INSUFFICIENT_DATA',
            'confidence_score': 0.0,
            'primary_criteria_met': [],
            'key_evidence': [],
            'clinical_rationale': f"CPU Generation error: {err_msg}",
            'adjudication_timestamp': timestamp
        }
