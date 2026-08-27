# Dockerfile for Rose Gold Clinical Adjudication Pipeline
# Packages FastAPI Backend, Streamlit Web Interface, and vLLM Engine
FROM vllm/vllm-openai:latest

WORKDIR /workspace

# Install data processing, API, and web UI dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    uvicorn>=0.28.0 \
    streamlit>=1.35.0 \
    pydantic>=2.7.0 \
    pandas>=2.2.0 \
    pyarrow>=15.0.0 \
    pyyaml>=6.0.1 \
    requests>=2.31.0 \
    outlines>=0.0.46 \
    xgrammar

# Copy source code and scripts
COPY app/ /workspace/app/
COPY configs/ /workspace/configs/
COPY data/ /workspace/data/
COPY start_services.sh /workspace/start_services.sh

RUN chmod +x /workspace/start_services.sh

EXPOSE 8000 8501

# Default launches both FastAPI and Streamlit Web Interface
CMD ["/workspace/start_services.sh"]
