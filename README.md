# Rose Gold: On-Premises LLM Clinical Chart Adjudication

Turnkey OMOP `NOTE` + `VISIT_OCCURRENCE` phenotyping for multi-center consortia. Outputs structured Rose Gold labels with evidence quotes and confidence scores. The default Cloud Run image stays CPU-light so the UI and API start in seconds; GPU sites can still run vLLM locally or via the GPU Dockerfile.

## What is fast in this tree

- **Lazy engine init** — health checks and page load do not download or load model weights
- **Cached OMOP reads** — parsed tables and formatted visits are reused by file stamp
- **Visit index** — `/api/visits` counts notes without materializing chart text
- **Single-visit note filter** — `/api/notes/{id}` does not scan the rest of the NOTE table
- **vLLM prefix caching** — identical system prompts share KV cache on GPU
- **Family-correct chat templates** — Gemma no longer receives Llama-3 tokens
- **Checkpointed CLI writes** — JSONL is appended in chunks so a crash can resume
- **One in-process engine** — Streamlit reuses a cached engine or the local FastAPI service

## Quick start (local, no GPU)

```bash
pip install -r requirements.txt
./start_services.sh
```

- API: http://127.0.0.1:8000/docs
- UI: http://127.0.0.1:8501

```bash
python -m app.adjudicator \
  --notes_path data/synthetic_notes.csv \
  --visits_path data/synthetic_visits.csv \
  --output_path outputs/rose_gold_adjudications.csv \
  --target_condition "Sepsis / Septic Shock"
```

## GPU / hospital firewall

Use `Dockerfile` (vLLM base) and `docker-compose.yml`. Optional CPU weights:

```bash
export ROSEGOLD_LOAD_CPU_WEIGHTS=1
```

## Cloud Run (CPU, quick load)

Same pattern as `note_extraction` / `cdw_copilot`: account `xjiang2@uth.edu`, project `sbmi-jiang-ai-testing01`.

```bash
gcloud auth login xjiang2@uth.edu
./deploy_to_gcp.sh
```

The script runs tests, builds `Dockerfile.cpu` via Cloud Build, creates `gs://sbmi-jiang-ai-testing01-rosegold-data`, mounts it at `/mnt/gcs`, deploys Cloud Run, grants you admin, and smoke-tests Streamlit. Annotations, criteria, and batch exports persist under `gs://…/outputs`. Override with `PROJECT_ID=... REGION=... SERVICE_NAME=... GCS_BUCKET=...` if needed.

Synthetic OMOP data only is bundled. Do not mount real PHI to a public service.

## MIMIC-III-Ext-Notes v1.0.0

[MIMIC-III-Ext-Notes](https://physionet.org/content/mimic-iii-ext-notes/1.0.0/) is a credentialed 150-note nursing-note sample with 2,288 clinician-adjudicated concepts (`detection`, `encounter`, `negation`). Rose Gold does not ship the files. After you complete PhysioNet credentialing, CITI training, and the project DUA:

```bash
python scripts/prepare_mimic_iii_ext_notes.py \
  --source /path/to/mimic-iii-ext-notes-1.0.0 \
  --output data/mimic-iii-ext-notes

python -m app.adjudicator \
  --notes_path data/mimic-iii-ext-notes/omop_notes.csv \
  --visits_path data/mimic-iii-ext-notes/omop_visits.csv \
  --target_condition "Sepsis / Septic Shock"
```

The UI and loader also accept the raw `notes.csv` and derive visits from `hadm_id`. Keep the extract under `data/mimic-iii-ext-notes/` or `data/physionet/` (gitignored). Do not upload it to the public Cloud Run demo.

A synthetic stand-in of that schema is generated from the OMOP demo cohort and shipped in the image:

```bash
python scripts/simulate_mimic_ext_notes.py \
  --notes_path data/synthetic_notes.csv \
  --output data/synthetic_mimic_ext_notes
```

The UI Cohort selector can switch between `Synthetic OMOP (20 visits)` and `Synthetic MIMIC-III-Ext-Notes`.

Concept-level scoring against `labels.csv`:

```bash
python scripts/eval_mimic_iii_ext_notes.py \
  --source data/mimic-iii-ext-notes \
  --write-prompts outputs/mimic_ext_prompts.jsonl \
  --phenotype-gold outputs/mimic_ext_sepsis_gold.csv \
  --predictions outputs/mimic_ext_predictions.csv
```

## Tests

```bash
python -m pytest tests/ -q
```
