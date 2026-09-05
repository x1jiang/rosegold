#!/bin/bash
# Cloud Run entrypoint: FastAPI on loopback, Streamlit on $PORT.
#
# Both children are supervised. If either exits, the container exits non-zero so
# Cloud Run replaces the instance instead of serving a UI whose API is gone
# (which would silently fall back to loading a second model in-process).
set -euo pipefail

PUBLIC_PORT="${PORT:-8080}"
API_PORT="${ROSEGOLD_API_PORT:-8000}"
mkdir -p "${ROSEGOLD_OUTPUT_DIR:-/workspace/outputs}"

API_PID=""
UI_PID=""
cleanup() {
  for pid in "$API_PID" "$UI_PID"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

echo "Starting Rose Gold API on 127.0.0.1:${API_PORT} and UI on 0.0.0.0:${PUBLIC_PORT}"

uvicorn app.api:app \
  --host 127.0.0.1 \
  --port "${API_PORT}" \
  --workers 1 \
  --no-server-header \
  --timeout-keep-alive 5 \
  --limit-concurrency "${ROSEGOLD_API_CONCURRENCY:-64}" &
API_PID=$!

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

export ROSEGOLD_API_URL="http://127.0.0.1:${API_PORT}"
streamlit run app/ui.py \
  --server.port "${PUBLIC_PORT}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.fileWatcherType none \
  --server.enableXsrfProtection true \
  --browser.gatherUsageStats false &
UI_PID=$!

# Exit as soon as either child does; Cloud Run restarts the instance.
wait -n "$API_PID" "$UI_PID"
status=$?
echo "A Rose Gold process exited (status ${status}); shutting down container." >&2
exit "$status"
