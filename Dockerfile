# syntax=docker/dockerfile:1

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

# Shared runtime base.

FROM ${PYTHON_IMAGE} AS runtime-base

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRAPHWISE_CONFIG=/app/config.properties \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_cpu.txt requirements_gpu.txt ./
COPY . .

RUN useradd -m -u 10001 appuser \
    && mkdir -p "${HF_HOME}" \
    && chown -R appuser:appuser /app /home/appuser


# CPU image

FROM runtime-base AS cpu

# The target is authoritative: CPU images never use CUDA workers.

ENV GRAPHWISE_RUNTIME_TARGET=cpu

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements_cpu.txt \
    && python setup.py build \
    && python -c 'import torch, transformers, sentence_transformers; assert torch.version.cuda is None; print("torch", torch.__version__, "cpu"); print("transformers", transformers.__version__); print("sentence-transformers", sentence_transformers.__version__)'

RUN chown -R appuser:appuser /app "${HF_HOME}"
USER appuser

EXPOSE 5050
ENTRYPOINT ["python", "-m", "graphwise_transformer.server"]


# GPU image
FROM runtime-base AS gpu

# GPU images never create CPU inference workers. Overriding
# GRAPHWISE_CPU_WORKERS to a non-zero value is rejected by configuration
# validation at startup.
ENV GRAPHWISE_RUNTIME_TARGET=gpu

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements_gpu.txt \
    && python setup.py build \
    && python -c 'import torch, transformers, sentence_transformers, flash_attn; assert torch.version.cuda and torch.version.cuda.startswith("12."); assert torch.compiled_with_cxx11_abi(); print("torch", torch.__version__, "cuda", torch.version.cuda); print("transformers", transformers.__version__); print("sentence-transformers", sentence_transformers.__version__); print("flash-attn", flash_attn.__version__)'

RUN chown -R appuser:appuser /app "${HF_HOME}"
USER appuser

EXPOSE 5050
ENTRYPOINT ["python", "-m", "graphwise_transformer.server"]
