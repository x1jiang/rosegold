# Dockerfile for Rose Gold Clinical Adjudication Pipeline (GPU / vLLM)
# Packages FastAPI Backend, Streamlit Web Interface, and vLLM Engine.
#
# Pin the base image for reproducible builds:
#   docker build --build-arg VLLM_TAG=v0.6.3 -t rosegold-suite:0.6.3 .
# The vLLM image runs as root (it needs the CUDA driver hooks); do not expose
# this container directly to untrusted networks. Put it behind your ingress and
# set ROSEGOLD_API_KEY.
ARG VLLM_TAG=latest
FROM vllm/vllm-openai:${VLLM_TAG}

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    ROSEGOLD_DATA_DIR=/workspace/data \
    ROSEGOLD_OUTPUT_DIR=/workspace/outputs \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

# Install data processing, API, and web UI dependencies
RUN pip install --no-cache-dir \
    "fastapi>=0.110.0,<1" \
    "uvicorn>=0.28.0,<1" \
    "streamlit>=1.41.0,<2" \
    "pydantic>=2.7.0,<3" \
    "pandas>=2.2.0,<4" \
    "pyarrow>=15.0.0,<25" \
    "pyyaml>=6.0.1,<7" \
    "requests>=2.31.0,<3" \
    "outlines>=0.0.46,<1" \
    "xgrammar<1"

# Copy source code and scripts
COPY app/ /workspace/app/
COPY configs/ /workspace/configs/
# Only the synthetic cohort ships in the image; mount real OMOP extracts at runtime.
COPY data/synthetic_notes.csv data/synthetic_visits.csv /workspace/data/
COPY data/synthetic_mimic_ext_notes/ /workspace/data/synthetic_mimic_ext_notes/
COPY start_services.sh /workspace/start_services.sh

RUN chmod 0755 /workspace/start_services.sh && mkdir -p /workspace/outputs

EXPOSE 8000 8501

# Liveness for docker/compose/k8s: the API must answer; Streamlit is supervised
# by start_services.sh, which exits the container if either child dies.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Default launches both FastAPI and Streamlit Web Interface
CMD ["/workspace/start_services.sh"]
