import os
import json
import datetime
import logging
import threading
import time
from typing import List, Dict, Any, Optional

from app.schemas import RoseGoldAdjudication
from app.prompts import build_adjudication_prompt, build_chat_prompt
from app.model_selector import resolve_model_and_engine
from app.config_loader import pipeline_settings

logger = logging.getLogger("rosegold.engine")


def _allow_mock() -> bool:
    return os.getenv("ROSEGOLD_ALLOW_MOCK", "").lower() in {"1", "true", "yes"}


def _backend_name() -> str:
    return os.getenv("ROSEGOLD_LLM_BACKEND", "").lower().strip()


def _retry_cooldown_seconds() -> float:
    """How long to wait before re-attempting a failed backend load (default 60s).

    A transient failure at cold start (weight download timeout, Vertex IAM not yet
    propagated) must not leave the instance answering 503 until someone restarts it.
    """
    try:
        return max(5.0, float(os.getenv("ROSEGOLD_BACKEND_RETRY_SECONDS", "60")))
    except ValueError:
        return 60.0


def _trust_remote_code() -> bool:
    """vLLM ``trust_remote_code`` executes Python shipped inside a model repo.

    Off by default (Llama / Gemma do not need it). Opt in with
    ``ROSEGOLD_TRUST_REMOTE_CODE=1`` only for repos you have reviewed.
    """
    return os.getenv("ROSEGOLD_TRUST_REMOTE_CODE", "").lower() in {"1", "true", "yes"}


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
        self._backend_ready = False
        self._initializing = False
        self._next_retry_at: Optional[float] = None
        self._init_lock = threading.Lock()
        # llama.cpp / vLLM / HF handles are not safe for concurrent generate()
        # calls; uvicorn runs sync endpoints on a threadpool, so serialize.
        self._infer_lock = threading.Lock()
        self.backend_error = None

    @property
    def is_initializing(self) -> bool:
        return self._initializing

    def wants_real_backend(self) -> bool:
        """True when configuration demands a real LLM and forbids the rules fallback."""
        return (_want_llamacpp() or _want_vertex() or _want_hybrid()) and not _allow_mock()

    def has_real_backend(self) -> bool:
        return bool(
            (self.is_vllm_available and self.llm is not None)
            or self.cpu_engine is not None
            or self.llama_engine is not None
            or self.hybrid_engine is not None
            or self.vertex_engine is not None
        )

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
        if self.wants_real_backend() and not self._backend_ready:
            return {
                "backend": "loading",
                "model_name": f"{_backend_name() or 'llamacpp'} (loading)",
                "llm_real": False,
            }
        return {
            "backend": "keyword_rules",
            "model_name": "keyword_rules",
            "llm_real": False,
            "error": self.backend_error,
        }

    def _retry_due(self) -> bool:
        return (
            self._next_retry_at is not None
            and time.monotonic() >= self._next_retry_at
            and not self.has_real_backend()
        )

    def _init_backend(self):
        """Load vLLM, HF CPU weights, llama.cpp, hybrid, or Vertex Gemini on first use.

        Guarded by a lock so concurrent first requests wait for one load instead
        of racing a half-initialized engine. If a required backend failed to load,
        the attempt is repeated after a cooldown instead of pinning the instance
        in a permanently degraded state.
        """
        if self._backend_ready and not self._retry_due():
            return
        with self._init_lock:
            if self._backend_initialized and not self._retry_due():
                return
            self._backend_initialized = True
            self._initializing = True
            try:
                self._init_backend_locked()
            finally:
                self._initializing = False
                self._backend_ready = True
                if self.wants_real_backend() and not self.has_real_backend():
                    self._next_retry_at = time.monotonic() + _retry_cooldown_seconds()
                    logger.warning(
                        "Required LLM backend '%s' is unavailable (%s); will retry in %.0fs",
                        _backend_name(),
                        self.backend_error or "unknown error",
                        _retry_cooldown_seconds(),
                    )
                else:
                    self._next_retry_at = None

    def _init_backend_locked(self):
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
                    logger.info("Loading vLLM engine: %s", self.model_name)
                    llm_kwargs = dict(
                        model=self.model_name,
                        tensor_parallel_size=self.tensor_parallel_size,
                        gpu_memory_utilization=self.gpu_memory_utilization,
                        max_model_len=self.max_model_len,
                        trust_remote_code=_trust_remote_code(),
                        **extra_kwargs,
                    )
                    try:
                        self.llm = LLM(enable_prefix_caching=prefix_cache, **llm_kwargs)
                    except TypeError:
                        self.llm = LLM(**llm_kwargs)
                    self.is_vllm_available = True
                    logger.info("vLLM engine initialized.")
            except Exception as e:
                logger.warning("Could not initialize vLLM: %s", e)
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
                logger.warning("CPU weight load skipped: %s", e)
                self.backend_error = str(e)

        if not self.is_vllm_available and self.cpu_engine is None and _want_llamacpp():
            try:
                from app.llamacpp_engine import LlamaCppEngine

                self.llama_engine = LlamaCppEngine()
                self.model_name = self.llama_engine.model_name
                logger.info("llama.cpp ready: %s", self.model_name)
            except Exception as e:
                logger.error("llama.cpp init failed: %s", e)
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
                logger.info("Hybrid engine ready: %s", self.model_name)
            except Exception as e:
                logger.error("Hybrid engine init failed: %s", e)
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
                logger.info("Vertex Gemini ready: %s", self.model_name)
            except Exception as e:
                logger.error("Vertex Gemini init failed: %s", e)
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
        if not records:
            return []
        if self.is_vllm_available and self.llm is not None:
            with self._infer_lock:
                return self._vllm_adjudicate(records, target_condition, clinical_criteria)
        if self.hybrid_engine is not None:
            return self.hybrid_engine.adjudicate_batch(records, target_condition, clinical_criteria)
        if self.cpu_engine is not None:
            with self._infer_lock:
                return self.cpu_engine.adjudicate_batch(records, target_condition, clinical_criteria)
        if self.llama_engine is not None:
            with self._infer_lock:
                return self.llama_engine.adjudicate_batch(records, target_condition, clinical_criteria)
        if self.vertex_engine is not None:
            return self.vertex_engine.adjudicate_batch(records, target_condition, clinical_criteria)
        if self.wants_real_backend():
            # Never hand out keyword-rule labels when the operator asked for an LLM.
            raise RuntimeError(
                self.backend_error
                or f"Requested LLM backend '{_backend_name()}' is not available."
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

