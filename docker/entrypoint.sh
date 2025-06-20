#!/usr/bin/env bash
set -e

# Default model if not provided
MODEL="${OLLAMA_MODEL:-llama3:8b-instruct-q5_K_M}"

# Start Ollama daemon
ollama serve &
OLLAMA_PID=$!

# Wait until Ollama responds
until curl -sS http://localhost:11434/api/tags > /dev/null; do
  echo "Waiting for Ollama to be ready …"
  sleep 2
done

# Ensure the requested model is pulled (non-fatal if already present)
ollama pull "$MODEL" || true

# Start FastAPI backend in background
echo "Starting FastAPI backend..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start Streamlit server in background
echo "Starting Streamlit server..."
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
STREAMLIT_PID=$!

# Function to handle shutdown
cleanup() {
    echo "Shutting down services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $STREAMLIT_PID 2>/dev/null || true
    kill $OLLAMA_PID 2>/dev/null || true
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Wait for both services to be ready
echo "Waiting for services to be ready..."
sleep 5

echo "SoloRAG is ready!"
echo "- FastAPI backend: http://localhost:8000"
echo "- Streamlit UI: http://localhost:8501"
echo "- API docs: http://localhost:8000/docs"

# Keep the container running
wait 