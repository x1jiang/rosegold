import os
import json
import argparse
import datetime
import pandas as pd
from typing import List, Dict, Any

from app.engine import AdjudicationEngine
from app.omop_loader import load_omop_data
from app.config_loader import pipeline_settings, resolve_criteria
from app.storage import default_batch_csv, output_dir, save_batch_results

def run_vllm_batch_adjudication(
    records: List[Dict[str, Any]],
    model_name: str,
    target_condition: str,
    clinical_criteria: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 32768,
    temperature: float = 0.0,
    quantization: str = None
) -> List[Dict[str, Any]]:
    """
    Executes offline high-throughput batch inference with vLLM and strict JSON schema decoding.
    """
    del temperature  # Engine uses greedy decoding for reproducible labels.
    engine = AdjudicationEngine(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        quantization=quantization,
    )
    print(f"Initializing engine with model: {engine.model_name} (TP={tensor_parallel_size}, max_len={max_model_len})")
    return engine.adjudicate_batch(records, target_condition, clinical_criteria)

def run_mock_adjudication(records: List[Dict[str, Any]], target_condition: str) -> List[Dict[str, Any]]:
    """Mock rule-based demonstrator for CPU/local environments without GPUs."""
    results = []
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for rec in records:
        text = rec['notes_formatted_text'].lower()
        if "septic shock" in text or "sepsis" in text or "icu transfer" in text or "lactate" in text:
            status = "CONFIRMED_POSITIVE"
            present = True
            conf = 0.95
            criteria = ["SIRS/qSOFA criteria met", "Documented infection source", "Lactate > 2.0 mmol/L"]
            evidence = [{
                "note_id": 101,
                "note_date": rec['visit_start_date'],
                "evidence_quote": "Patient admitted with fever, tachycardia, hypotension refractory to initial fluids, elevated lactate 3.4.",
                "interpretation": "Meets consensus definition for Sepsis-3"
            }]
            rationale = f"Encounter notes document acute organ dysfunction in setting of confirmed bacterial infection meeting Sepsis criteria."
        elif "stroke" in text or "nihss" in text or "infarct" in text:
            status = "CONFIRMED_POSITIVE"
            present = True
            conf = 0.92
            criteria = ["Acute focal neurologic deficit", "MRI/CT confirmed infarct", "Elevated NIHSS"]
            evidence = [{
                "note_id": 104,
                "note_date": rec['visit_start_date'],
                "evidence_quote": "CT Head demonstrated acute ischemic stroke in left MCA territory.",
                "interpretation": "Radiological and clinical confirmation of acute ischemic stroke"
            }]
            rationale = "Clinical and neuroimaging findings confirm acute ischemic stroke."
        else:
            status = "CONFIRMED_NEGATIVE"
            present = False
            conf = 0.98
            criteria = []
            evidence = []
            rationale = f"Thorough chart review reveals no clinical documentation, lab findings, or treatment indicating {target_condition}."

        results.append({
            'person_id': rec['person_id'],
            'visit_occurrence_id': rec['visit_occurrence_id'],
            'condition_present': present,
            'phenotype_status': status,
            'confidence_score': conf,
            'primary_criteria_met': criteria,
            'key_evidence': evidence,
            'clinical_rationale': rationale,
            'adjudication_timestamp': timestamp
        })
    return results

def main():
    parser = argparse.ArgumentParser(description="Rose Gold Clinical Note Batch Adjudication Pipeline")
    parser.add_argument("--notes_path", type=str, default="data/synthetic_notes.csv", help="Path to OMOP NOTE CSV/Parquet")
    parser.add_argument("--visits_path", type=str, default="data/synthetic_visits.csv", help="Path to OMOP VISIT_OCCURRENCE CSV/Parquet")
    parser.add_argument("--output_path", type=str, default=default_batch_csv(), help="Path to save adjudication outputs")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="HuggingFace model ID or local weight path")
    parser.add_argument("--target_condition", type=str, default="Sepsis / Septic Shock", help="Condition to adjudicate")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to shard model across")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization scheme (e.g. 'fp8', 'awq', 'bitsandbytes')")
    parser.add_argument("--max_model_len", type=int, default=32768, help="Max context window length in tokens")

    args = parser.parse_args()
    settings = pipeline_settings()
    criteria = resolve_criteria(args.target_condition)
    chunk_size = max(1, int(settings.get("infer_chunk_size", 32)))

    print("=" * 70)
    print("ROSE GOLD CLINICAL PHENOTYPING & ADJUDICATION PIPELINE")
    print(f"Condition: {args.target_condition}")
    print(f"Notes File: {args.notes_path}")
    print(f"Visits File: {args.visits_path}")
    print(f"Model: {args.model_name}")
    print("=" * 70)

    records = load_omop_data(notes_path=args.notes_path, visits_path=args.visits_path)
    print(f"Loaded {len(records)} visit encounters ready for adjudication.")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    parquet_path = args.output_path.rsplit(".", 1)[0] + ".parquet"
    jsonl_path = args.output_path.rsplit(".", 1)[0] + ".jsonl"

    done_ids = set()
    adjudications = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                adjudications.append(item)
                if item.get("visit_occurrence_id") is not None:
                    done_ids.add(int(item["visit_occurrence_id"]))
        print(f"Resuming from checkpoint: {len(done_ids)} visits already written.")

    pending = [rec for rec in records if int(rec["visit_occurrence_id"]) not in done_ids]
    engine = AdjudicationEngine(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
    )
    print(f"Pending visits: {len(pending)} (chunk size {chunk_size})")

    with open(jsonl_path, "a", encoding="utf-8") as handle:
        for start in range(0, len(pending), chunk_size):
            chunk = pending[start:start + chunk_size]
            chunk_results = engine.adjudicate_batch(chunk, args.target_condition, criteria)
            for item in chunk_results:
                handle.write(json.dumps(item) + "\n")
                adjudications.append(item)
            handle.flush()
            print(f"  wrote {min(start + chunk_size, len(pending))}/{len(pending)} new visits")

    df_out = pd.DataFrame(adjudications)
    df_out.to_csv(args.output_path, index=False)
    try:
        df_out.to_parquet(parquet_path, index=False)
    except Exception:
        parquet_path = None

    persisted = None
    if os.path.abspath(args.output_path).startswith(output_dir() + os.sep) or os.path.abspath(args.output_path) == default_batch_csv():
        persisted = save_batch_results(adjudications, args.target_condition)

    print(f"\n[SUCCESS] Adjudication complete!")
    print(f" - CSV saved to: {args.output_path}")
    if parquet_path:
        print(f" - Parquet saved to: {parquet_path}")
    print(f" - JSON Lines saved to: {jsonl_path}")
    if persisted:
        print(f" - OMOP OBSERVATION saved to: {persisted['omop']}")
    print("\nSummary of Rose Gold Labels:")
    print(df_out["phenotype_status"].value_counts())

if __name__ == "__main__":
    main()
