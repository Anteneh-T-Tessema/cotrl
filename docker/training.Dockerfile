FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# PyTorch + CUDA 12.1
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# HuggingFace stack for GRPO training
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    trl>=0.8.6 \
    accelerate>=0.29.0 \
    peft>=0.10.0 \
    bitsandbytes>=0.43.0 \
    datasets>=2.18.0 \
    loguru sympy tqdm pydantic

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/

ENV PYTHONPATH=/app
ENV HF_HOME=/app/.cache/huggingface

CMD ["python", "scripts/train_rl.py"]
