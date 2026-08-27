import os
from typing import Any, Dict, Optional

# Standard Model Matrix for Hardware Profiles
MODEL_CATALOG = {
    "gpu": {
        "llama": "meta-llama/Llama-3.1-8B-Instruct",
        "gemma": "google/gemma-2-9b-it",
        "muse": "muse-glimmer-30b"
    },
    "cpu": {
        "llama": "meta-llama/Llama-3.2-3B-Instruct",
        "gemma": "google/gemma-2-2b-it",
        "muse": "meta-llama/Llama-3.2-3B-Instruct" # Muse requires GPU, auto-fallback for CPU
    }
}

def _cpu_info() -> Dict[str, Any]:
    import multiprocessing

    return {
        "device": "cpu",
        "is_gpu": False,
        "device_name": f"Host CPU ({multiprocessing.cpu_count()} cores)",
        "device_count": 1,
        "vram_gb": 0.0,
        "tier": "Commodity CPU Cluster / VM",
    }


def detect_hardware() -> Dict[str, Any]:
    """Detects CUDA, Apple MPS, or CPU. Torch is imported only when installed."""
    try:
        import torch
    except ImportError:
        return _cpu_info()

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return {
            "device": "cuda",
            "is_gpu": True,
            "device_name": gpu_name,
            "device_count": gpu_count,
            "vram_gb": round(vram_gb, 1),
            "tier": "H100/A100 High-End" if vram_gb >= 70 else "Standard GPU (16-48GB)",
        }
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return {
            "device": "mps",
            "is_gpu": False,
            "device_name": "Apple Silicon (MPS)",
            "device_count": 1,
            "vram_gb": 0.0,
            "tier": "Apple Silicon Unified Memory",
        }
    return _cpu_info()

def resolve_model_and_engine(
    requested_model: str = "auto",
    requested_family: str = "llama",
    force_device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Automatically selects the best model and execution backend based on detected hardware.
    """
    hw = detect_hardware()
    target_device = force_device if force_device in ["cuda", "cpu"] else hw["device"]
    
    device_mode = "gpu" if target_device == "cuda" else "cpu"
    family = requested_family.lower()
    if "gemma" in family or "gemma" in requested_model.lower():
        family_key = "gemma"
    elif "muse" in family or "muse" in requested_model.lower():
        family_key = "muse"
    else:
        family_key = "llama"

    if requested_model and requested_model != "auto" and "/" in requested_model:
        selected_model = requested_model
    else:
        selected_model = MODEL_CATALOG[device_mode][family_key]

    return {
        "selected_model": selected_model,
        "model_family": family_key,
        "target_device": target_device,
        "is_gpu": (target_device == "cuda"),
        "hardware_info": hw,
        "engine_type": "vLLM Engine" if target_device == "cuda" else "CPU Lightweight Engine",
        "description": f"Auto-selected {selected_model} for {hw['device_name']} ({device_mode.upper()} mode)"
    }
