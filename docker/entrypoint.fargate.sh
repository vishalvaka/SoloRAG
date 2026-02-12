#!/usr/bin/env bash
set -e

echo "SoloRAG Fargate entrypoint starting ..."

# Run Alembic migrations (production uses this instead of create_all)
if [ "${APP_ENV}" = "production" ]; then
    echo "Running Alembic migrations ..."
    python -m alembic upgrade head || echo "Migration warning (may already be up to date)"
fi

# Start FastAPI (single process -- Streamlit is a separate ECS service)
echo "Starting FastAPI on port 8000 ..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2

