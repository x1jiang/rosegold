import pytest
from app.model_selector import detect_hardware, resolve_model_and_engine, MODEL_CATALOG

def test_detect_hardware():
    hw = detect_hardware()
    assert "device" in hw
    assert "device_name" in hw
    assert "is_gpu" in hw
    assert hw["device"] in ["cuda", "mps", "cpu"]

def test_resolve_model_for_gpu():
    res = resolve_model_and_engine(requested_family="llama", force_device="cuda")
    assert res["is_gpu"] is True
    assert res["target_device"] == "cuda"
    assert res["selected_model"] == "meta-llama/Llama-3.1-8B-Instruct"

    res_gemma = resolve_model_and_engine(requested_family="gemma", force_device="cuda")
    assert res_gemma["selected_model"] == "google/gemma-2-9b-it"

    res_muse = resolve_model_and_engine(requested_family="muse", force_device="cuda")
    assert res_muse["selected_model"] == "muse-glimmer-30b"

def test_resolve_model_for_cpu():
    res = resolve_model_and_engine(requested_family="llama", force_device="cpu")
    assert res["is_gpu"] is False
    assert res["target_device"] == "cpu"
    # Verifies that CPU auto-selects lightweight 3B model
    assert res["selected_model"] == "meta-llama/Llama-3.2-3B-Instruct"

    # Verifies that CPU auto-selects lightweight 2B Gemma model
    res_gemma = resolve_model_and_engine(requested_family="gemma", force_device="cpu")
    assert res_gemma["selected_model"] == "google/gemma-2-2b-it"

    # Verifies that Muse on CPU gracefully falls back to lightweight model
    res_muse = resolve_model_and_engine(requested_family="muse", force_device="cpu")
    assert res_muse["selected_model"] == "meta-llama/Llama-3.2-3B-Instruct"
