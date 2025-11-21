# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps (git for sentence-transformers when resolving models, and build essentials for some wheels)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirement files first to leverage layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Generate gRPC stubs at build time
RUN python setup.py build

# Create non-root user
RUN useradd -m -u 10001 appuser
USER appuser

EXPOSE 5050

ENV GRAPHWISE_CONFIG=/app/config.properties

ENTRYPOINT ["python", "-m", "graphwise_transformer.server"]
