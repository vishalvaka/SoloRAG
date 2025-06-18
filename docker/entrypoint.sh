#!/usr/bin/env bash
set -e

# Default model if not provided
MODEL="${OLLAMA_MODEL:-llama3:8b-instruct-q5_K_M}"

# Start Ollama daemon
ollama serve &
PID=$!

# Wait until Ollama responds
until curl -sS http://localhost:11434/api/tags > /dev/null; do
  echo "Waiting for Ollama to be ready …"
  sleep 2
done

# Ensure the requested model is pulled (non-fatal if already present)
ollama pull "$MODEL" || true

# Launch FastAPI backend (will use OLLAMA_URL=localhost)
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 