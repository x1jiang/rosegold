# Rose Gold: On-Premises LLM Clinical Chart Adjudication

Turnkey OMOP `NOTE` + `VISIT_OCCURRENCE` phenotyping for multi-center consortia. Outputs structured Rose Gold labels with evidence quotes and confidence scores. The default Cloud Run image stays CPU-light so the UI and API start in seconds; GPU sites can still run vLLM locally or via the GPU Dockerfile.

## Clinical Benchmark: RoseGold Hybrid vs. Frontier LLMs

Empirical evaluation on clinical inpatient encounters across 4 consensus phenotypes (Acute Ischemic Stroke, Acute Respiratory Distress Syndrome, Acute Kidney Injury, and Sepsis / Septic Shock):

| Model Architecture & Tier | Adjudication Paradigm | Overall Accuracy | Cohen's $\kappa$ | Sensitivity (Recall) | Specificity | False Positives | False Negatives | Avg Latency / Visit | Deployment Boundary |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **RoseGold Hybrid (Rules + Muse-30B)** | **Two-Tier Hybrid Cascade** | **93.8%** | **0.875** | **87.5%** | **100.0%** | **0** | **1** | **0.66s** | **100% On-Premises (Internal GPU)** |
| **Barebone GPT-5.6-Luna** | Standalone Frontier Cloud API | **81.2%** | **0.625** | **62.5%** | **100.0%** | **0** | **3** | **2.37s** | External Cloud API (OpenAI) |

### Why the Hybrid Architecture Wins:
- **Tier 1 (Deterministic NLP Pre-Filter)**: High-speed clause segmentation and bidirectional negation screening (±70 characters) instantly eliminate 70% of negative encounters (< 0.01s), preventing wasteful LLM compute.
- **Tier 2 (Muse-Glimmer-30B Clinical Reasoner)**: On-premises 30B MoE/Dense reasoning on GPU (`sglang`) adjudicates complex candidate evidence, verifies clinical plausibility, rules out mimics, and generates step-by-step chain-of-thought rationale.
- **Tier 3 (Consensus Arbitration & Provenance)**: Reconciles syndrome equivalents, extracts verbatim evidence quotes with Note ID and timestamp, and records physician review in durable audit logs.
- **100% On-Premises & Zero PHI Risk**: Complies with hospital IRB, MIMIC Data Use Agreements, and HIPAA regulations without external cloud data exfiltration.

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

## Docker & Container Deployment

### Rootless Docker (Recommended for Hospital Firewalls & Multi-User Servers)

Rootless Docker runs both the Docker daemon and containers inside an unprivileged user namespace, complying with clinical and enterprise security restrictions without requiring `root` or `sudo` access.

#### 1. Setup Rootless Environment
Ensure the rootless daemon is active and your shell points to the user socket:
```bash
# Start user daemon (systemd)
systemctl --user start docker
systemctl --user enable docker

# Point Docker CLI to the rootless socket
export DOCKER_HOST="unix://${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock"
```

#### 2. Run CPU Container (Zero-Install, Fast Cold-Start)
```bash
# Create local outputs folder with host user write permissions
mkdir -p outputs

# Build the CPU-optimized image
docker build -f Dockerfile.cpu -t rosegold:cpu .

# Run with rootless volume mounts and unprivileged port 8080
docker run -d \
  --name rosegold \
  -p 8080:8080 \
  -v "$(pwd)/data:/workspace/data:ro" \
  -v "$(pwd)/outputs:/workspace/outputs:rw" \
  rosegold:cpu
```
Open **http://localhost:8080** in your browser.

#### 3. Run with Docker Compose (Rootless)
```bash
mkdir -p outputs
docker compose up --build
```
- API Docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

#### 4. Rootless GPU (vLLM on NVIDIA Hardware)
If running on an unprivileged GPU node with the NVIDIA Container Toolkit (v1.14+), generate the Container Device Interface (CDI) spec:
```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# Run container with CDI GPU passthrough
docker run --gpus all -p 8000:8000 -p 8501:8501 rosegold:latest
```

#### 5. Rootless Podman (Drop-in Alternative)
For Red Hat Enterprise Linux, Rocky Linux, or Fedora clusters:
```bash
mkdir -p outputs
podman build -f Dockerfile.cpu -t rosegold:cpu .
podman run -d \
  --name rosegold \
  -p 8080:8080 \
  -v ./data:/workspace/data:ro,Z \
  -v ./outputs:/workspace/outputs:rw,Z \
  rosegold:cpu
```
*(The `:Z` flag configures SELinux volume relabeling automatically).*

### Standard GPU Deployment
For dedicated root or server environments with CUDA:
Use `Dockerfile` (vLLM base) and `docker-compose.yml`. Optional CPU weights:

```bash
export ROSEGOLD_LOAD_CPU_WEIGHTS=1
```

## Cloud Run (CPU, quick load)

Deploy the turnkey container to Google Cloud Run:

```bash
# Authenticate and select your target GCP project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Run the automated deployment script
./deploy_to_gcp.sh
```

The script runs tests, builds `Dockerfile.cpu` via Cloud Build, provisions `gs://${PROJECT_ID}-rosegold-data`, mounts it at `/mnt/gcs`, deploys Cloud Run, configures IAM roles, and smoke-tests Streamlit. Annotations, criteria, and batch exports persist under `gs://${PROJECT_ID}-rosegold-data/outputs`. Override defaults with:
```bash
PROJECT_ID=my-project REGION=us-central1 SERVICE_NAME=rosegold GCS_BUCKET=my-bucket ./deploy_to_gcp.sh
```

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
