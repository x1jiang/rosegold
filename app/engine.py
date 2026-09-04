import os
import json
import datetime
from typing import List, Dict, Any, Optional

from app.schemas import RoseGoldAdjudication
from app.prompts import build_adjudication_prompt, build_chat_prompt
from app.model_selector import resolve_model_and_engine
from app.config_loader import pipeline_settings


def _allow_mock() -> bool:
    return os.getenv("ROSEGOLD_ALLOW_MOCK", "").lower() in {"1", "true", "yes"}


def _backend_name() -> str:
    return os.getenv("ROSEGOLD_LLM_BACKEND", "").lower().strip()


def _want_vertex() -> bool:
    return _backend_name() in {"vertex", "gemini"}


def _want_hybrid() -> bool:
    return _backend_name() in {"hybrid", "rosegold_hybrid", "muse_hybrid"}


def _want_llamacpp() -> bool:
    backend = _backend_name()
    if backend in {"mock", "keyword", "rules", "vertex", "gemini", "hybrid", "rosegold_hybrid", "muse_hybrid"}:
        return False
    if backend in {"llama", "llamacpp", "llama.cpp", "gguf"}:
        return True
    if os.getenv("K_SERVICE"):
        return not _allow_mock()
    return False


class AdjudicationEngine:
    def __init__(
        self,
        model_name: str = "auto",
        model_family: str = "llama",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 32768,
        quantization: Optional[str] = None,
        force_device: Optional[str] = None
    ):
        self.resolution = resolve_model_and_engine(
            requested_model=model_name,
            requested_family=model_family,
            force_device=force_device
        )
        self.model_name = self.resolution["selected_model"]
        self.is_gpu = self.resolution["is_gpu"]
        self.hardware_info = self.resolution["hardware_info"]
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.quantization = quantization
        self.llm = None
        self.cpu_engine = None
        self.vertex_engine = None
        self.llama_engine = None
        self.hybrid_engine = None
        self.is_vllm_available = False
        self._backend_initialized = False
        self.backend_error = None

    def backend_status(self, init: bool = False) -> Dict[str, Any]:
        if init:
            self._init_backend()
        if self.is_vllm_available and self.llm is not None:
            return {"backend": "vllm", "model_name": self.model_name, "llm_real": True}
        if self.cpu_engine is not None and getattr(self.cpu_engine, "is_hf_loaded", False):
            return {"backend": "hf_cpu", "model_name": self.model_name, "llm_real": True}
        if self.llama_engine is not None:
            return {
                "backend": "llamacpp",
                "model_name": self.llama_engine.model_name,
                "llm_real": True,
            }
        if self.hybrid_engine is not None:
            return {
                "backend": "hybrid",
                "model_name": f"Hybrid (Rules + {self.hybrid_engine.model_name})",
                "llm_real": True,
            }
        if self.vertex_engine is not None:
            return {
                "backend": "vertex",
                "model_name": self.vertex_engine.model_name,
                "llm_real": True,
            }
        if _want_llamacpp() and not self._backend_initialized:
            return {
                "backend": "loading",
                "model_name": "Llama-3.2-3B-Instruct (loading)",
                "llm_real": False,
            }
        return {
            "backend": "keyword_rules",
            "model_name": "keyword_rules",
            "llm_real": False,
            "error": self.backend_error,
        }

    def _init_backend(self):
        """Load vLLM, HF CPU weights, or Vertex Gemini on first use."""
        if self._backend_initialized:
            return
        self._backend_initialized = True
        settings = pipeline_settings()
        prefix_cache = bool(settings.get("enable_prefix_caching", True))

        if self.is_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    from vllm import LLM

                    extra_kwargs = {}
                    if self.quantization:
                        extra_kwargs["quantization"] = self.quantization
                    print(f"[Engine] Loading vLLM engine: {self.model_name}")
                    llm_kwargs = dict(
                        model=self.model_name,
                        tensor_parallel_size=self.tensor_parallel_size,
                        gpu_memory_utilization=self.gpu_memory_utilization,
                        max_model_len=self.max_model_len,
                        trust_remote_code=True,
                        **extra_kwargs,
                    )
                    try:
                        self.llm = LLM(enable_prefix_caching=prefix_cache, **llm_kwargs)
                    except TypeError:
                        self.llm = LLM(**llm_kwargs)
                    self.is_vllm_available = True
                    print("[Engine] vLLM engine successfully initialized.")
            except Exception as e:
                print(f"[Engine] Could not initialize vLLM ({e}).")
                self.is_vllm_available = False
                self.backend_error = str(e)

        load_cpu_weights = os.getenv("ROSEGOLD_LOAD_CPU_WEIGHTS", "").lower() in {"1", "true", "yes"}
        if not self.is_vllm_available and load_cpu_weights:
            try:
                from app.cpu_engine import CPULlamaGemmaEngine

                candidate = CPULlamaGemmaEngine(
                    model_name=self.model_name,
                    load_in_memory=True,
                )
                if candidate.is_hf_loaded:
                    self.cpu_engine = candidate
                else:
                    self.backend_error = "CPU weights requested but HuggingFace model did not load."
            except Exception as e:
                print(f"[Engine] CPU weight load skipped ({e}).")
                self.backend_error = str(e)

        if not self.is_vllm_available and self.cpu_engine is None and _want_llamacpp():
            try:
                from app.llamacpp_engine import LlamaCppEngine

                self.llama_engine = LlamaCppEngine()
                self.model_name = self.llama_engine.model_name
                print(f"[Engine] Llama.cpp ready: {self.model_name}")
            except Exception as e:
                print(f"[Engine] Llama.cpp init failed ({e}).")
                self.backend_error = str(e)

        if (
            not self.is_vllm_available
            and self.cpu_engine is None
            and self.llama_engine is None
            and self.vertex_engine is None
            and _want_hybrid()
        ):
            try:
                from app.hybrid_engine import HybridAdjudicationEngine

                self.hybrid_engine = HybridAdjudicationEngine()
                self.model_name = f"hybrid:{self.hybrid_engine.model_name}"
                print(f"[Engine] Hybrid Engine ready: {self.model_name}")
            except Exception as e:
                print(f"[Engine] Hybrid Engine init failed ({e}).")
                self.backend_error = str(e)

        if (
            not self.is_vllm_available
            and self.cpu_engine is None
            and self.llama_engine is None
            and _want_vertex()
        ):
            try:
                from app.vertex_engine import VertexGeminiEngine

                self.vertex_engine = VertexGeminiEngine()
                self.model_name = self.vertex_engine.model_name
                print(f"[Engine] Vertex Gemini ready: {self.model_name}")
            except Exception as e:
                print(f"[Engine] Vertex Gemini init failed ({e}).")
                self.backend_error = str(e)

    def adjudicate_single(
        self,
        record: Dict[str, Any],
        target_condition: str,
        clinical_criteria: str
    ) -> Dict[str, Any]:
        """Adjudicates a single patient visit record."""
        results = self.adjudicate_batch([record], target_condition, clinical_criteria)
        return results[0]

    def adjudicate_batch(
        self,
        records: List[Dict[str, Any]],
        target_condition: str,
        clinical_criteria: str
    ) -> List[Dict[str, Any]]:
        """Adjudicates a batch of patient visit records."""
        self._init_backend()
        if self.is_vllm_available and self.llm is not None:
            return self._vllm_adjudicate(records, target_condition, clinical_criteria)
        if self.hybrid_engine is not None:
            return self.hybrid_engine.adjudicate_batch(records, target_condition, clinical_criteria)
        if self.cpu_engine is not None:
            return self.cpu_engine.adjudicate_batch(records, target_condition, clinical_criteria)
        if self.llama_engine is not None:
            return self.llama_engine.adjudicate_batch(records, target_condition, clinical_criteria)
        if self.vertex_engine is not None:
            return self.vertex_engine.adjudicate_batch(records, target_condition, clinical_criteria)
        if (_want_llamacpp() or _want_vertex()) and not _allow_mock():
            raise RuntimeError(
                self.backend_error
                or "No real LLM backend is available (Llama weights did not load)."
            )
        return self._mock_adjudicate(records, target_condition)

    def _vllm_adjudicate(
        self,
        records: List[Dict[str, Any]],
        target_condition: str,
        clinical_criteria: str,
    ) -> List[Dict[str, Any]]:

        from vllm import SamplingParams
        try:
            from vllm.sampling_params import GuidedDecodingParams
            guided_decoding = GuidedDecodingParams(json=json.dumps(RoseGoldAdjudication.model_json_schema()))
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=2048,
                guided_decoding=guided_decoding
            )
        except ImportError:
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=2048,
                extra_body={"guided_json": json.dumps(RoseGoldAdjudication.model_json_schema())}
            )

        prompts = []
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
            prompts.append(build_chat_prompt(user_prompt, self.model_name))

        outputs = self.llm.generate(prompts, sampling_params)
        results = []
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for rec, out in zip(records, outputs):
            text = out.outputs[0].text.strip()
            try:
                parsed = json.loads(text)
                parsed['person_id'] = rec['person_id']
                parsed['visit_occurrence_id'] = rec['visit_occurrence_id']
                parsed['adjudication_timestamp'] = timestamp
                parsed['inference_backend'] = f"vllm:{self.model_name}"
                results.append(RoseGoldAdjudication(**parsed).model_dump())
            except Exception as e:
                results.append({
                    'person_id': rec['person_id'],
                    'visit_occurrence_id': rec['visit_occurrence_id'],
                    'condition_present': False,
                    'phenotype_status': 'INDETERMINATE_INSUFFICIENT_DATA',
                    'confidence_score': 0.0,
                    'primary_criteria_met': [],
                    'key_evidence': [],
                    'clinical_rationale': f"Parsing error: {e}",
                    'adjudication_timestamp': timestamp
                })
        return results

    def _mock_adjudicate(self, records: List[Dict[str, Any]], target_condition: str) -> List[Dict[str, Any]]:
        from app.clinical_rules import adjudicate_clinical_rules
        return adjudicate_clinical_rules(records, target_condition, backend_tag="keyword_rules")

