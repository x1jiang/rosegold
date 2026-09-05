# Dockerfile for Rose Gold Clinical Adjudication Pipeline
# Packages FastAPI Backend, Streamlit Web Interface, and vLLM Engine
FROM vllm/vllm-openai:latest

WORKDIR /workspace

# Install data processing, API, and web UI dependencies
RUN pip install --no-cache-dir \
    "fastapi>=0.110.0,<1" \
    "uvicorn>=0.28.0,<1" \
    "streamlit>=1.35.0,<2" \
    "pydantic>=2.7.0,<3" \
    "pandas>=2.2.0,<4" \
    "pyarrow>=15.0.0,<25" \
    "pyyaml>=6.0.1,<7" \
    "requests>=2.31.0,<3" \
    "outlines>=0.0.46" \
    xgrammar

# Copy source code and scripts
COPY app/ /workspace/app/
COPY configs/ /workspace/configs/
# Only the synthetic cohort ships in the image; mount real OMOP extracts at runtime.
COPY data/synthetic_notes.csv data/synthetic_visits.csv /workspace/data/
COPY data/synthetic_mimic_ext_notes/ /workspace/data/synthetic_mimic_ext_notes/
COPY start_services.sh /workspace/start_services.sh

RUN chmod +x /workspace/start_services.sh

EXPOSE 8000 8501

# Default launches both FastAPI and Streamlit Web Interface
CMD ["/workspace/start_services.sh"]
