import pytest
import os
import pandas as pd
from app.cpu_engine import CPULlamaGemmaEngine
from app.omop_loader import load_omop_data

def test_cpu_engine_llama():
    engine = CPULlamaGemmaEngine(model_name="meta-llama/Llama-3.2-3B-Instruct")
    records = load_omop_data("data/synthetic_notes.csv", "data/synthetic_visits.csv", target_visits=[20001, 20002])
    
    rec_20001 = next(r for r in records if r["visit_occurrence_id"] == 20001)
    rec_20002 = next(r for r in records if r["visit_occurrence_id"] == 20002)

    # Sepsis positive test on CPU
    res_pos = engine.adjudicate_single(rec_20001, "Sepsis / Septic Shock", "Consensus Criteria")
    assert res_pos["visit_occurrence_id"] == 20001
    assert res_pos["condition_present"] is True
    assert res_pos["phenotype_status"] == "CONFIRMED_POSITIVE"
    assert res_pos["confidence_score"] >= 0.90

    # Negative control test on CPU
    res_neg = engine.adjudicate_single(rec_20002, "Sepsis / Septic Shock", "Consensus Criteria")
    assert res_neg["visit_occurrence_id"] == 20002
    assert res_neg["condition_present"] is False
    assert res_neg["phenotype_status"] == "CONFIRMED_NEGATIVE"

def test_cpu_engine_gemma():
    engine = CPULlamaGemmaEngine(model_name="google/gemma-2-2b-it")
    records = load_omop_data("data/synthetic_notes.csv", "data/synthetic_visits.csv", target_visits=[20003, 20002])
    rec_20003 = next(r for r in records if r["visit_occurrence_id"] == 20003)
    
    # Stroke test on CPU
    res_stroke = engine.adjudicate_single(rec_20003, "Acute Ischemic Stroke", "Consensus Criteria")
    assert res_stroke["visit_occurrence_id"] == 20003
    assert res_stroke["condition_present"] is True
    assert res_stroke["phenotype_status"] == "CONFIRMED_POSITIVE"

def test_cpu_batch_processing():
    engine = CPULlamaGemmaEngine(model_name="meta-llama/Llama-3.1-8B-Instruct")
    records = load_omop_data("data/synthetic_notes.csv", "data/synthetic_visits.csv")
    
    batch_results = engine.adjudicate_batch(records, "Sepsis / Septic Shock", "Consensus Criteria")
    assert len(batch_results) == 20
    for r in batch_results:
        assert "confidence_score" in r
        assert "clinical_rationale" in r
        assert "phenotype_status" in r
