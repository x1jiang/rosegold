#!/bin/bash
set -euo pipefail

echo "========================================================"
echo "Starting Rose Gold Full-Stack System"
echo " - FastAPI Service: http://0.0.0.0:8000"
echo " - Streamlit Web UI: http://0.0.0.0:8501"
echo "========================================================"

uvicorn app.api:app --host 0.0.0.0 --port 8000 --workers 1 &
FASTAPI_PID=$!

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

export ROSEGOLD_API_URL="${ROSEGOLD_API_URL:-http://127.0.0.1:8000}"
streamlit run app/ui.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

kill "${FASTAPI_PID}" 2>/dev/null || true
