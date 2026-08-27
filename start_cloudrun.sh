#!/bin/bash
set -euo pipefail

PUBLIC_PORT="${PORT:-8080}"
mkdir -p "${ROSEGOLD_OUTPUT_DIR:-/workspace/outputs}"

echo "Starting Rose Gold API on 127.0.0.1:8000 and UI on 0.0.0.0:${PUBLIC_PORT}"

uvicorn app.api:app --host 127.0.0.1 --port 8000 --workers 1 &
API_PID=$!

python - <<'PY'
import time
import urllib.request
for _ in range(50):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=0.4)
        break
    except Exception:
        time.sleep(0.1)
PY

export ROSEGOLD_API_URL="http://127.0.0.1:8000"
exec streamlit run app/ui.py \
  --server.port "${PUBLIC_PORT}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
