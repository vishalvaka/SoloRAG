# docker/backend.Dockerfile
# ──────────────────────────────────────────────────────────────
# Build a slim image that serves FastAPI + your FAISS artefacts
#
#  CPU build (default):
#   docker build -t solorag-backend -f docker/backend.Dockerfile .
#
#  GPU build:
#   docker buildx build \
#     -t solorag-backend:gpu \
#     -f docker/backend.Dockerfile \
#     --build-arg BUILD_MODE=gpu \
#     --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
#     .
# ──────────────────────────────────────────────────────────────
ARG BUILD_MODE=cpu
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# Redeclare the build-time argument to make it available in the rest of the file
ARG BUILD_MODE

# 1. Install OS libs
# For nvidia/cuda base images, we need to install python first.
# For python:slim base images, these packages are mostly present.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl gnupg ca-certificates python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Ollama (works for both CPU and GPU)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 2. App code & Python dependencies
# Use a cache mount to speed up subsequent builds
WORKDIR /app
COPY ./requirements /app/requirements

# First, install common deps
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir -r requirements/common.txt

# Then, install the target-specific packages (cpu or gpu)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir -r requirements/${BUILD_MODE}.txt

# Finally, install sentence-transformers, which depends on torch
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir -r requirements.txt

# Copy app code and artifacts
COPY ./app ./app
COPY ./artifacts ./artifacts

ENV OLLAMA_URL=http://localhost:11434

# 3. Expose and launch
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
