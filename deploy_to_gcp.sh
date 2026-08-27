#!/usr/bin/env bash
# Rose Gold — Cloud Run deploy
# Pattern matches note_extraction / cdw_copilot under xjiang2@uth.edu.
#
# Prerequisites:
#   gcloud authenticated as xjiang2@uth.edu
#   project sbmi-jiang-ai-testing01 (override with PROJECT_ID=...)
#
# Usage:
#   gcloud auth login xjiang2@uth.edu
#   ./deploy_to_gcp.sh

set -euo pipefail
cd "$(dirname "$0")"

PROJECT_ID="${PROJECT_ID:-sbmi-jiang-ai-testing01}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-rosegold}"
ACCOUNT="${ACCOUNT:-xjiang2@uth.edu}"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"
GCS_BUCKET="${GCS_BUCKET:-${PROJECT_ID}-rosegold-data}"
PYTHON="${PYTHON:-/Users/xiaoqianjiang/anaconda3/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

echo "🚀 Deploying Rose Gold to Google Cloud Run"
echo "   Account: $ACCOUNT"
echo "   Project: $PROJECT_ID"
echo "   Region:  $REGION"
echo "   Service: $SERVICE_NAME"
echo "   Image:   $IMAGE"
echo "   Bucket:  gs://$GCS_BUCKET"
echo "   Image is CPU-light (Dockerfile.cpu) for a fast cold start."
echo ""

if ! command -v gcloud >/dev/null 2>&1; then
  echo "❌ gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
  exit 1
fi

if [[ ! -f Dockerfile.cpu || ! -f cloudbuild.yaml || ! -f app/api.py ]]; then
  echo "❌ Run this script from the rosegold repo root."
  exit 1
fi

if [[ "${DEPLOY_YES:-}" == "1" ]]; then
  echo "DEPLOY_YES=1 — continuing without prompt."
else
  read -r -p "Deploy now as ${ACCOUNT} to ${PROJECT_ID}? [y/N] " reply
  if [[ ! "${reply}" =~ ^[Yy]$ ]]; then
    echo "Cancelled. When ready:"
    echo "  gcloud auth login ${ACCOUNT}"
    echo "  ./deploy_to_gcp.sh"
    exit 0
  fi
fi

echo ""
echo "👤 Setting gcloud account to ${ACCOUNT}"
gcloud config set account "$ACCOUNT"
gcloud config set project "$PROJECT_ID"

ACTIVE="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -1 || true)"
if [[ "$ACTIVE" != "$ACCOUNT" ]]; then
  echo "❌ Active account is '${ACTIVE:-none}', expected '${ACCOUNT}'."
  echo "   Reauthentication is required (UTH tokens expire). Run in your terminal:"
  echo "     gcloud auth login ${ACCOUNT}"
  echo "     gcloud config set account ${ACCOUNT}"
  echo "     gcloud config set project ${PROJECT_ID}"
  echo "   Then re-run: ./deploy_to_gcp.sh"
  exit 1
fi
echo "✓ Authenticated as: $ACTIVE"
echo ""

echo "🧪 Test gate..."
"$PYTHON" -m pytest tests/ -q
echo ""

echo "🔧 Enabling required APIs (best effort)..."
for svc in cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com artifactregistry.googleapis.com storage.googleapis.com aiplatform.googleapis.com; do
  gcloud services enable "$svc" --project="$PROJECT_ID" --quiet 2>/dev/null \
    || echo "⚠️  Could not enable ${svc} (may already be enabled)"
done
echo ""

echo "🪣 Ensuring GCS bucket gs://${GCS_BUCKET} for durable annotations..."
if ! gcloud storage buckets describe "gs://${GCS_BUCKET}" --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${GCS_BUCKET}" \
    --project="$PROJECT_ID" \
    --location="$REGION" \
    --uniform-bucket-level-access
fi
echo "" | gcloud storage cp - "gs://${GCS_BUCKET}/outputs/.keep" 2>/dev/null || true
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
RUNTIME_SA="$(gcloud run services describe "$SERVICE_NAME" --project="$PROJECT_ID" --region="$REGION" --format='value(spec.template.spec.serviceAccountName)' 2>/dev/null || true)"
RUNTIME_SA="${RUNTIME_SA:-${PROJECT_NUMBER}-compute@developer.gserviceaccount.com}"
gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/storage.objectAdmin" \
  --quiet >/dev/null \
  || echo "⚠️  Could not grant bucket IAM (may already be set)"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/aiplatform.user" \
  --quiet >/dev/null \
  || echo "⚠️  Could not grant Vertex AI IAM (may already be set)"
echo "✓ Bucket ready: gs://${GCS_BUCKET} (runtime SA ${RUNTIME_SA})"
echo ""

echo "📦 Building CPU image via Cloud Build..."
gcloud builds submit --config cloudbuild.yaml --project "$PROJECT_ID"

echo ""
echo "☁️  Deploying Cloud Run service..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 8Gi \
  --cpu 4 \
  --cpu-boost \
  --no-cpu-throttling \
  --timeout 900 \
  --min-instances 0 \
  --max-instances 2 \
  --execution-environment gen2 \
  --add-volume="name=rosegold-data,type=cloud-storage,bucket=${GCS_BUCKET}" \
  --add-volume-mount="volume=rosegold-data,mount-path=/mnt/gcs" \
  --set-env-vars "ROSEGOLD_DATA_DIR=/workspace/data,ROSEGOLD_MODEL_NAME=auto,ROSEGOLD_API_URL=http://127.0.0.1:8000,ROSEGOLD_OUTPUT_DIR=/mnt/gcs/outputs,ROSEGOLD_AUDIT_LOG=/mnt/gcs/outputs/human_audit_log.jsonl,ROSEGOLD_GCS_BUCKET=${GCS_BUCKET},ROSEGOLD_LLM_BACKEND=llamacpp,ROSEGOLD_LLAMA_REPO=bartowski/Llama-3.2-3B-Instruct-GGUF,ROSEGOLD_LLAMA_GGUF=Llama-3.2-3B-Instruct-Q4_K_M.gguf,ROSEGOLD_MODEL_DIR=/mnt/gcs/models,ROSEGOLD_LOCAL_MODEL_DIR=/tmp/rosegold-models,GOOGLE_CLOUD_PROJECT=${PROJECT_ID}"

echo ""
echo "🔑 Granting ${ACCOUNT} Cloud Run admin on this service..."
gcloud run services add-iam-policy-binding "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --member="user:${ACCOUNT}" \
  --role="roles/run.admin" \
  --quiet >/dev/null \
  || echo "⚠️  Could not add IAM binding (may already exist)"

SERVICE_URL="$(gcloud run services describe "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format='value(status.url)')"

echo ""
echo "🧪 Smoke testing ${SERVICE_URL}/_stcore/health ..."
ready=0
for i in $(seq 1 18); do
  CODE="$(curl -sS -o /tmp/rosegold_health.txt -w '%{http_code}' "${SERVICE_URL}/_stcore/health" 2>/dev/null || echo 000)"
  BODY="$(cat /tmp/rosegold_health.txt 2>/dev/null || true)"
  if [[ "$CODE" == "200" ]] && echo "$BODY" | grep -qi 'ok'; then
    ready=1
    echo "✅ Streamlit health OK (attempt ${i})"
    break
  fi
  sleep 5
done

if [[ "$ready" -eq 0 ]]; then
  echo "⚠️  Health check not ready yet. Logs:"
  echo "    gcloud run services logs read ${SERVICE_NAME} --region=${REGION} --project=${PROJECT_ID}"
fi

echo ""
echo "✅ Deployment complete"
echo "🌐 UI:     ${SERVICE_URL}"
echo "🩺 Health: ${SERVICE_URL}/_stcore/health"
echo "💾 GCS:    gs://${GCS_BUCKET}/outputs"
echo "   FastAPI stays on 127.0.0.1:8000 inside the container; the UI calls it locally."
echo ""
echo "Useful commands:"
echo "  gcloud run services logs read ${SERVICE_NAME} --region=${REGION} --project=${PROJECT_ID}"
echo "  gcloud run services describe ${SERVICE_NAME} --region=${REGION} --project=${PROJECT_ID}"
echo "  ./deploy_to_gcp.sh"
