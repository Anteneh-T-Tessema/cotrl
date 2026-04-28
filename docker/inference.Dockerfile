FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (CPU-only inference; no CUDA needed for generation sampling)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    transformers accelerate datasets pydantic loguru tqdm sympy

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/

ENV PYTHONPATH=/app
ENV HF_HOME=/app/.cache/huggingface

CMD ["python", "scripts/run_baseline.py"]
