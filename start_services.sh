#!/bin/bash
# Full-stack entrypoint for docker-compose / Singularity / bare hosts.
# FastAPI binds to ROSEGOLD_API_BIND (default 0.0.0.0 so other containers can
# reach it). Set ROSEGOLD_API_KEY to require a shared secret on /api/*.
set -euo pipefail

API_BIND="${ROSEGOLD_API_BIND:-0.0.0.0}"
API_PORT="${ROSEGOLD_API_PORT:-8000}"
UI_PORT="${ROSEGOLD_UI_PORT:-8501}"
mkdir -p "${ROSEGOLD_OUTPUT_DIR:-/workspace/outputs}" 2>/dev/null || true

echo "========================================================"
echo "Starting Rose Gold Full-Stack System"
echo " - FastAPI Service: http://${API_BIND}:${API_PORT}"
echo " - Streamlit Web UI: http://0.0.0.0:${UI_PORT}"
if [[ -n "${ROSEGOLD_API_KEY:-}" ]]; then
  echo " - API auth: X-API-Key required on /api/*"
else
  echo " - API auth: none (set ROSEGOLD_API_KEY to enable)"
fi
echo "========================================================"

FASTAPI_PID=""
UI_PID=""
cleanup() {
  for pid in "$FASTAPI_PID" "$UI_PID"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

uvicorn app.api:app \
  --host "${API_BIND}" \
  --port "${API_PORT}" \
  --workers 1 \
  --no-server-header \
  --timeout-keep-alive 5 \
  --limit-concurrency "${ROSEGOLD_API_CONCURRENCY:-64}" &
FASTAPI_PID=$!

python - "$API_PORT" <<'PY'
import sys
import time
import urllib.request
port = sys.argv[1]
for _ in range(100):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.4)
        break
    except Exception:
        time.sleep(0.1)
PY

export ROSEGOLD_API_URL="${ROSEGOLD_API_URL:-http://127.0.0.1:${API_PORT}}"
streamlit run app/ui.py \
  --server.port "${UI_PORT}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.fileWatcherType none \
  --server.enableXsrfProtection true \
  --browser.gatherUsageStats false &
UI_PID=$!

wait -n "$FASTAPI_PID" "$UI_PID"
status=$?
echo "A Rose Gold process exited (status ${status}); shutting down." >&2
exit "$status"
