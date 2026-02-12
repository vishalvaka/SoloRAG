# docker/fargate.Dockerfile
# ──────────────────────────────────────────────────────────────
# Slim production image for AWS ECS Fargate.
#
# - No Ollama (LLM via Bedrock or external Ollama service)
# - No Streamlit (runs as a separate ECS service)
# - Non-root user
# - Custom CA cert support for corporate proxy/DPI
#
#  Build:
#   docker buildx build -t solorag-fargate -f docker/fargate.Dockerfile .
#
#  With custom CA cert:
#   docker buildx build -t solorag-fargate \
#     --build-arg CUSTOM_CA_CERT=certs/corporate-ca.pem \
#     -f docker/fargate.Dockerfile .
# ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

ARG CUSTOM_CA_CERT=""

# 1. OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 2. Optional: install corporate CA certificate
#    Only copies if CUSTOM_CA_CERT build-arg points to a real file.
RUN --mount=type=bind,target=/build-context,source=. \
    if [ -n "${CUSTOM_CA_CERT}" ] && [ -f "/build-context/${CUSTOM_CA_CERT}" ]; then \
        cp "/build-context/${CUSTOM_CA_CERT}" /usr/local/share/ca-certificates/corporate-ca.crt && \
        update-ca-certificates; \
    fi

# 3. Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# 4. Python dependencies
WORKDIR /app
COPY ./requirements /app/requirements
RUN pip install --no-cache-dir -r requirements/common.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. App code and artifacts
COPY ./app ./app
COPY ./artifacts ./artifacts
COPY ./alembic ./alembic
COPY ./alembic.ini ./alembic.ini

# 6. Pre-warm HuggingFace models
RUN python3 - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer('intfloat/e5-base-v2')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print('Prewarmed models')
PY

# 7. Entrypoint
COPY docker/entrypoint.fargate.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 8. Switch to non-root
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

ENTRYPOINT ["/entrypoint.sh"]
