import os
import json
import argparse
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

def main():
    parser = argparse.ArgumentParser(description="Rose Gold Clinical Note Batch Adjudication Pipeline")
    parser.add_argument("--notes_path", type=str, default="data/synthetic_notes.csv", help="Path to OMOP NOTE or MIMIC-III-Ext-Notes notes.csv")
    parser.add_argument("--visits_path", type=str, default="data/synthetic_visits.csv", help="Path to OMOP VISIT_OCCURRENCE CSV/Parquet (optional for MIMIC-III-Ext-Notes)")
    parser.add_argument("--output_path", type=str, default=default_batch_csv(), help="Path to save adjudication outputs")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="HuggingFace model ID or local weight path")
    parser.add_argument("--target_condition", type=str, default="Sepsis / Septic Shock", help="Condition to adjudicate")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to shard model across")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization scheme (e.g. 'fp8', 'awq', 'bitsandbytes')")
    parser.add_argument("--max_model_len", type=int, default=32768, help="Max context window length in tokens")
    parser.add_argument("--clinical_criteria", type=str, default=None, help="Custom clinical criteria text (overrides config.yaml default)")
    parser.add_argument("--backend", type=str, default=None, help="Explicit backend to use (vllm, hybrid, llamacpp, vertex, keyword_rules)")
    parser.add_argument("--chunk_size", type=int, default=None, help="Override batch chunk size")

    args = parser.parse_args()
    if args.backend:
        os.environ["ROSEGOLD_LLM_BACKEND"] = args.backend.strip()

    settings = pipeline_settings()
    criteria = resolve_criteria(args.target_condition, args.clinical_criteria)
    chunk_size = args.chunk_size if args.chunk_size and args.chunk_size > 0 else max(1, int(settings.get("infer_chunk_size", 32)))

    print("=" * 70)
    print("ROSE GOLD CLINICAL PHENOTYPING & ADJUDICATION PIPELINE")
    print(f"Condition: {args.target_condition}")
    print(f"Notes File: {args.notes_path}")
    print(f"Visits File: {args.visits_path}")
    print(f"Model: {args.model_name}")
    if args.backend:
        print(f"Backend: {args.backend}")
    print("=" * 70)

    records = load_omop_data(notes_path=args.notes_path, visits_path=args.visits_path)
    print(f"Loaded {len(records)} visit encounters ready for adjudication.")
    if not records:
        print("[WARNING] No encounter records found to adjudicate. Exiting.")
        return

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
                try:
                    item = json.loads(line)
                    adjudications.append(item)
                    if item.get("visit_occurrence_id") is not None:
                        done_ids.add(int(item["visit_occurrence_id"]))
                except Exception:
                    continue
        print(f"Resuming from checkpoint: {len(done_ids)} visits already written.")

    pending = [rec for rec in records if int(rec.get("visit_occurrence_id", 0)) not in done_ids]
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
