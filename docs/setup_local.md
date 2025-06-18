# Setting up SoloRAG Locally

This guide walks you through running the full stack on your laptop in **≤ 10 minutes**.

---

## 1. Prerequisites
| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11 (>= 3.10 works) | Used for dev / tests. |
| Docker | 24+ | Required for running the LLM & backend in containers. |
| Git | latest | Clone the repo. |
| (Optional) GPU drivers | CUDA 12.2 | Only needed for GPU compose variant. |

---

## 2. Clone & Create a Virtual Environment
```bash
# 1. Clone
 git clone https://github.com/<your-fork>/SoloRAG.git
 cd SoloRAG

# 2. Create venv
 python -m venv .venv
 source .venv/bin/activate

# 3. Install dependencies
 pip install -r requirements-dev.txt
```
Running `pip install` takes ~1 min on a typical connection.

---

## 3. Build the Knowledge Index
```bash
python scripts/build_index.py \
       --input data/raw/stripe_faqs_full.jsonl \
       --output_dir artifacts/
```
This script creates `artifacts/faiss.idx` and `artifacts/meta.npy` (~15 MB total).

---

## 4. Run Unit Tests (optional but recommended)
```bash
./scripts/run_tests.sh -q   # -q for concise output
```
You should see all 17 tests pass.

---

## 5. Starting the Stack
### Option A: Docker Compose (recommended)
```bash
cd docker
# CPU version
docker compose up --build -d
# GPU users: docker compose -f compose.gpu.yaml up --build -d
```
Services:
* `backend` – FastAPI app on http://localhost:8000
* `ollama` – LLM runtime on http://localhost:11434
* `prometheus` – Metrics UI on http://localhost:9090

Logs: `docker compose logs -f backend`.

### Option B: Pure Python Dev Server
```bash
# Still inside project root & venv
uvicorn app.main:app --reload --port 8000
```
You **must** have an Ollama daemon running in the background for this mode. Start it manually:
```bash
ollama serve &   # or `ollama run llama3` to pull model first
```

---

## 6. Interacting with the API
1. Open Swagger UI: http://localhost:8000/docs
2. Try the `/answer` endpoint with a question like **"How can I refund a payment?"**
3. Watch the streaming response in real time.

---

## 7. Updating the Model
Pass `-e OLLAMA_MODEL=<model>` to the backend container or set it in `.env` when running locally. The entrypoint automatically pulls the model on first use.

Example:
```bash
docker compose up --build -d --pull always \
  --no-deps backend \
  -e OLLAMA_MODEL=mistral:7b-instruct-q5_K_M
```

---

## 8. Troubleshooting
| Symptom | Fix |
|---------|-----|
| Port 11434 already in use | `docker ps | grep 11434` then `docker rm -f <ID>` |
| Tests fail randomly | Ensure `.venv` active; run `pip install -r requirements-dev.txt` |
| Slow responses | Use GPU compose file or smaller quantised model (`q4`). |

---

## 9. Next Steps
* `scripts/evaluate.py` – measure answer quality.
* `docs/architecture.md` – deep dive into how the pieces fit.
* Fork & star the repo if you find it useful ✨
