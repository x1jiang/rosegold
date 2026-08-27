import os
from functools import lru_cache
from typing import Any, Dict, Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_CONFIG_PATH = os.path.join(_ROOT, "configs", "config.yaml")

_DEFAULT_PIPELINE = {
    "default_condition": "Sepsis / Septic Shock",
    "max_notes_per_visit": 50,
    "max_chars_per_note": 4000,
    "gpu_memory_utilization": 0.90,
    "max_model_len": 32768,
    "infer_chunk_size": 32,
    "enable_prefix_caching": True,
}

_PLACEHOLDER_CRITERIA = {
    "",
    "standard consensus definition",
    "consensus criteria",
    "standard clinical consensus",
}

_PHENOTYPE_ALIASES = (
    ("septic", "sepsis"),
    ("sepsis", "sepsis"),
    ("stroke", "stroke"),
    ("infarct", "stroke"),
    ("ards", "ards"),
    ("respiratory", "ards"),
    ("aki", "aki"),
    ("kidney", "aki"),
)


@lru_cache(maxsize=4)
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = path or os.getenv("ROSEGOLD_CONFIG", _DEFAULT_CONFIG_PATH)
    if not os.path.isfile(cfg_path):
        return {"pipeline": dict(_DEFAULT_PIPELINE), "phenotypes": {}}
    try:
        import yaml

        with open(cfg_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            return {"pipeline": dict(_DEFAULT_PIPELINE), "phenotypes": {}}
        return data
    except Exception:
        return {"pipeline": dict(_DEFAULT_PIPELINE), "phenotypes": {}}


def pipeline_settings() -> Dict[str, Any]:
    merged = dict(_DEFAULT_PIPELINE)
    merged.update(load_config().get("pipeline") or {})
    return merged


def _phenotype_key(target_condition: str) -> Optional[str]:
    lowered = (target_condition or "").lower()
    for needle, key in _PHENOTYPE_ALIASES:
        if needle in lowered:
            return key
    return None


def criteria_for(target_condition: str) -> str:
    phenotypes = load_config().get("phenotypes") or {}
    key = _phenotype_key(target_condition)
    if key and isinstance(phenotypes.get(key), dict):
        criteria = (phenotypes[key].get("criteria") or "").strip()
        if criteria:
            return criteria
    return "Standard Consensus Definition"


def resolve_criteria(target_condition: str, override: Optional[str] = None) -> str:
    cleaned = (override or "").strip()
    if cleaned and cleaned.lower() not in _PLACEHOLDER_CRITERIA:
        return cleaned
    try:
        from app.storage import load_criteria

        saved = load_criteria()
        if saved:
            return saved
    except Exception:
        pass
    return criteria_for(target_condition)
