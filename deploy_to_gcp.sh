#!/usr/bin/env bash
# Rose Gold — Cloud Run deploy
#
# Prerequisites:
#   gcloud CLI installed and authenticated
#   GCP project with billing enabled (override with PROJECT_ID=...)
#
# Usage:
#   gcloud auth login
#   PROJECT_ID="your-project-id" ./deploy_to_gcp.sh
#
# Knobs (env vars):
#   REGION, SERVICE_NAME, GCS_BUCKET, ACCOUNT, PYTHON   as before
#   RUNTIME_SA_NAME   dedicated runtime service account (default: rosegold-runtime).
#                     Falls back to the default compute SA if it cannot be created.
#   REQUIRE_AUTH=1    deploy with --no-allow-unauthenticated (IAM-gated UI).
#   LLM_BACKEND       llamacpp (default) | vertex | hybrid. Vertex IAM is only
#                     granted when the backend actually needs it.
#   ALLOW_DIRTY=1     deploy with uncommitted changes (image tag gets -dirty).
#   DEPLOY_YES=1      skip the confirmation prompt.
#   ROSEGOLD_API_KEY  optional shared secret for the loopback API (passed through).
#   ROSEGOLD_LLAMA_SHA256  optional pin for the downloaded GGUF (passed through).

set -euo pipefail
cd "$(dirname "$0")"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" ]]; then
  echo "❌ No GCP project specified. Please set PROJECT_ID=your-project-id or run 'gcloud config set project <PROJECT_ID>'."
  exit 1
fi

REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-rosegold}"
ACCOUNT="${ACCOUNT:-$(gcloud config get-value account 2>/dev/null || true)}"
IMAGE_BASE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
GCS_BUCKET="${GCS_BUCKET:-${PROJECT_ID}-rosegold-data}"
PYTHON="${PYTHON:-python3}"
RUNTIME_SA_NAME="${RUNTIME_SA_NAME:-rosegold-runtime}"
LLM_BACKEND="${LLM_BACKEND:-llamacpp}"
CONTAINER_UID="${CONTAINER_UID:-1000}"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "❌ gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
  exit 1
fi

if [[ ! -f Dockerfile.cpu || ! -f cloudbuild.yaml || ! -f app/api.py ]]; then
  echo "❌ Run this script from the rosegold repo root."
  exit 1
fi

# Immutable image tag from the commit being deployed.
GIT_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || true)"
if [[ -z "$GIT_SHA" ]]; then
  IMAGE_TAG="manual-$(date -u +%Y%m%d%H%M%S)"
elif [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
  if [[ "${ALLOW_DIRTY:-}" != "1" ]]; then
    echo "❌ Working tree has uncommitted changes. Commit first so the deployed image maps to a commit,"
    echo "   or re-run with ALLOW_DIRTY=1."
    exit 1
  fi
  IMAGE_TAG="${GIT_SHA}-dirty-$(date -u +%Y%m%d%H%M%S)"
else
  IMAGE_TAG="$GIT_SHA"
fi
IMAGE="${IMAGE_BASE}:${IMAGE_TAG}"

if [[ "${REQUIRE_AUTH:-}" == "1" ]]; then
  AUTH_FLAG="--no-allow-unauthenticated"
else
  AUTH_FLAG="--allow-unauthenticated"
fi

echo "🚀 Deploying Rose Gold to Google Cloud Run"
if [[ -n "$ACCOUNT" && "$ACCOUNT" != "(unset)" ]]; then
  echo "   Account: $ACCOUNT"
fi
echo "   Project: $PROJECT_ID"
echo "   Region:  $REGION"
echo "   Service: $SERVICE_NAME"
echo "   Image:   $IMAGE"
echo "   Bucket:  gs://$GCS_BUCKET"
echo "   Backend: $LLM_BACKEND"
echo "   Access:  ${AUTH_FLAG#--}"
echo "   Image is CPU-light (Dockerfile.cpu), runs as uid ${CONTAINER_UID}."
echo ""

if [[ "${DEPLOY_YES:-}" == "1" ]]; then
  echo "DEPLOY_YES=1 — continuing without prompt."
else
  read -r -p "Deploy now to ${PROJECT_ID}? [y/N] " reply
  if [[ ! "${reply}" =~ ^[Yy]$ ]]; then
    echo "Cancelled. When ready:"
    echo "  ./deploy_to_gcp.sh"
    exit 0
  fi
fi

echo ""
if [[ -n "$ACCOUNT" && "$ACCOUNT" != "(unset)" ]]; then
  echo "👤 Setting gcloud account to ${ACCOUNT}"
  gcloud config set account "$ACCOUNT" --quiet 2>/dev/null || true
fi
gcloud config set project "$PROJECT_ID" --quiet

ACTIVE="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -1 || true)"
if [[ -n "$ACCOUNT" && "$ACCOUNT" != "(unset)" && "$ACTIVE" != "$ACCOUNT" ]]; then
  echo "❌ Active account is '${ACTIVE:-none}', expected '${ACCOUNT}'."
  echo "   Reauthentication may be required. Run in your terminal:"
  echo "     gcloud auth login ${ACCOUNT}"
  echo "     gcloud config set account ${ACCOUNT}"
  echo "     gcloud config set project ${PROJECT_ID}"
  echo "   Then re-run: ./deploy_to_gcp.sh"
  exit 1
fi
echo "✓ Authenticated as: ${ACTIVE:-default}"
echo ""

echo "🧪 Test gate..."
"$PYTHON" -m pytest -p no:logfire tests/ -q
echo ""

echo "🔧 Enabling required APIs (best effort)..."
APIS=(cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com artifactregistry.googleapis.com storage.googleapis.com iam.googleapis.com)
if [[ "$LLM_BACKEND" == "vertex" || "$LLM_BACKEND" == "gemini" ]]; then
  APIS+=(aiplatform.googleapis.com)
fi
for svc in "${APIS[@]}"; do
  gcloud services enable "$svc" --project="$PROJECT_ID" --quiet 2>/dev/null \
    || echo "⚠️  Could not enable ${svc} (may already be enabled)"
done
echo ""

echo "🪣 Ensuring GCS bucket gs://${GCS_BUCKET} for durable annotations..."
if ! gcloud storage buckets describe "gs://${GCS_BUCKET}" --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${GCS_BUCKET}" \
    --project="$PROJECT_ID" \
    --location="$REGION" \
    --uniform-bucket-level-access \
    --public-access-prevention
fi
gcloud storage buckets update "gs://${GCS_BUCKET}" --public-access-prevention --quiet >/dev/null 2>&1 || true
echo "" | gcloud storage cp - "gs://${GCS_BUCKET}/outputs/.keep" 2>/dev/null || true

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
DEFAULT_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
RUNTIME_SA="${RUNTIME_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "🔐 Ensuring dedicated runtime service account ${RUNTIME_SA}..."
if ! gcloud iam service-accounts describe "$RUNTIME_SA" --project="$PROJECT_ID" >/dev/null 2>&1; then
  if gcloud iam service-accounts create "$RUNTIME_SA_NAME" \
      --project="$PROJECT_ID" \
      --display-name="Rose Gold Cloud Run runtime" \
      --description="Least-privilege identity for the rosegold Cloud Run service" \
      --quiet >/dev/null 2>&1; then
    echo "✓ Created ${RUNTIME_SA}"
    # IAM eventual consistency: give the new principal a moment to exist everywhere.
    sleep 10
  else
    echo "⚠️  Could not create ${RUNTIME_SA}; falling back to default compute SA ${DEFAULT_SA}"
    RUNTIME_SA="$DEFAULT_SA"
  fi
fi

gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/storage.objectAdmin" \
  --quiet >/dev/null \
  || echo "⚠️  Could not grant bucket IAM (may already be set)"
if [[ "$LLM_BACKEND" == "vertex" || "$LLM_BACKEND" == "gemini" ]]; then
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${RUNTIME_SA}" \
    --role="roles/aiplatform.user" \
    --quiet >/dev/null \
    || echo "⚠️  Could not grant Vertex AI IAM (may already be set)"
fi
echo "✓ Bucket ready: gs://${GCS_BUCKET} (runtime SA ${RUNTIME_SA})"
echo ""

echo "📦 Building CPU image ${IMAGE} via Cloud Build..."
gcloud builds submit \
  --config cloudbuild.yaml \
  --project "$PROJECT_ID" \
  --substitutions "_TAG=${IMAGE_TAG},_IMAGE=${IMAGE_BASE}"

ENV_VARS="ROSEGOLD_DATA_DIR=/workspace/data"
ENV_VARS+=",ROSEGOLD_MODEL_NAME=auto"
ENV_VARS+=",ROSEGOLD_API_URL=http://127.0.0.1:8000"
ENV_VARS+=",ROSEGOLD_OUTPUT_DIR=/mnt/gcs/outputs"
ENV_VARS+=",ROSEGOLD_AUDIT_LOG=/mnt/gcs/outputs/human_audit_log.jsonl"
ENV_VARS+=",ROSEGOLD_GCS_BUCKET=${GCS_BUCKET}"
ENV_VARS+=",ROSEGOLD_LLM_BACKEND=${LLM_BACKEND}"
ENV_VARS+=",ROSEGOLD_LLAMA_REPO=bartowski/Llama-3.2-3B-Instruct-GGUF"
ENV_VARS+=",ROSEGOLD_LLAMA_GGUF=Llama-3.2-3B-Instruct-Q4_K_M.gguf"
ENV_VARS+=",ROSEGOLD_MODEL_DIR=/mnt/gcs/models"
ENV_VARS+=",ROSEGOLD_LOCAL_MODEL_DIR=/tmp/rosegold-models"
ENV_VARS+=",GOOGLE_CLOUD_PROJECT=${PROJECT_ID}"
ENV_VARS+=",ROSEGOLD_GIT_SHA=${GIT_SHA:-unknown}"
if [[ -n "${ROSEGOLD_API_KEY:-}" ]]; then
  ENV_VARS+=",ROSEGOLD_API_KEY=${ROSEGOLD_API_KEY}"
fi
if [[ -n "${ROSEGOLD_LLAMA_SHA256:-}" ]]; then
  ENV_VARS+=",ROSEGOLD_LLAMA_SHA256=${ROSEGOLD_LLAMA_SHA256}"
fi

echo ""
echo "☁️  Deploying Cloud Run service..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --platform managed \
  "$AUTH_FLAG" \
  --service-account "$RUNTIME_SA" \
  --port 8080 \
  --memory 8Gi \
  --cpu 4 \
  --cpu-boost \
  --no-cpu-throttling \
  --timeout 900 \
  --min-instances 0 \
  --max-instances 2 \
  --execution-environment gen2 \
  --ingress all \
  --add-volume="name=rosegold-data,type=cloud-storage,bucket=${GCS_BUCKET},mount-options=uid=${CONTAINER_UID};gid=${CONTAINER_UID}" \
  --add-volume-mount="volume=rosegold-data,mount-path=/mnt/gcs" \
  --set-env-vars "$ENV_VARS"

if [[ -n "$ACCOUNT" && "$ACCOUNT" != "(unset)" ]]; then
  echo ""
  echo "🔑 Granting ${ACCOUNT} Cloud Run admin on this service..."
  gcloud run services add-iam-policy-binding "$SERVICE_NAME" \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --member="user:${ACCOUNT}" \
    --role="roles/run.admin" \
    --quiet >/dev/null \
    || echo "⚠️  Could not add IAM binding (may already exist)"
fi

SERVICE_URL="$(gcloud run services describe "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format='value(status.url)')"

echo ""
echo "🧪 Smoke testing ${SERVICE_URL}/_stcore/health ..."
HEALTH_BODY="$(mktemp)"
trap 'rm -f "$HEALTH_BODY"' EXIT
ready=0
for i in $(seq 1 18); do
  CODE="$(curl -sS -o "$HEALTH_BODY" -w '%{http_code}' "${SERVICE_URL}/_stcore/health" 2>/dev/null || echo 000)"
  BODY="$(cat "$HEALTH_BODY" 2>/dev/null || true)"
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
echo "🌐 UI:      ${SERVICE_URL}"
echo "🩺 Health:  ${SERVICE_URL}/_stcore/health"
echo "💾 GCS:     gs://${GCS_BUCKET}/outputs"
echo "🏷  Image:   ${IMAGE}"
echo "🔐 Runtime: ${RUNTIME_SA}"
echo "   FastAPI stays on 127.0.0.1:8000 inside the container; the UI calls it locally."
echo ""
echo "Useful commands:"
echo "  gcloud run services logs read ${SERVICE_NAME} --region=${REGION} --project=${PROJECT_ID}"
echo "  gcloud run services describe ${SERVICE_NAME} --region=${REGION} --project=${PROJECT_ID}"
echo "  python3 scripts/verify_gcs_persist.py ${SERVICE_URL}/   # end-to-end persistence check (needs playwright)"
echo "  ./deploy_to_gcp.sh"
