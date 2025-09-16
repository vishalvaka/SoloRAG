# SoloRAG Architecture

> Last updated: June 2025

SoloRAG is a reference Retrieval-Augmented Generation (RAG) application that turns a collection of JSON documents into a question-answering service. The design is intentionally simple so it can run fully locally while still demonstrating production-grade patterns (Docker, metrics, tests, CI, etc.).

---

## 1. High-level Component Diagram

```mermaid
graph TD
    subgraph User Space
        A[Client / Browser]
    end
    subgraph Containerised Stack
        B[FastAPI Backend]
        C[Retriever<br/>FAISS (CPU by default; optional CUDA)]
        D[Prompt Builder]
        E[LLM ( Ollama )]
    end

    A -- HTTP --> B
    B -- top-k query --> C
    C -- k doc snippets --> D
    D -- prompt --> E
    E -- answer JSON --> B
    B -- REST JSON --> A
```

---

## 2. Backend (FastAPI)
* **Endpoints**
  * `GET /healthz` – enhanced liveness probe with GPU/index information.
  * `POST /query` – returns full answer JSON.
  * `POST /query/stream` – streams tokens as they are generated.
* **Middleware** – a custom Prometheus middleware records request counts, durations and error rates.
* **Logging** – `structlog` outputs JSON logs; perfect for Grafana Loki.
* **Testing** – pytest with `httpx.AsyncClient` gives ~95 % unit-test coverage.

## 3. Retrieval Layer (CPU by default; optional GPU)

### Index Types
SoloRAG runs on CPU by default for fastest startup. You can opt into GPU retrieval by setting `FAISS_FORCE_CPU=0`.

* **CPU Mode (default)**: Uses `IndexFlatIP` for exact similarity search
* **GPU Mode (opt-in)**: Uses `IndexIVFFlat` with CUDA acceleration for approximate nearest neighbor search

### GPU Optimizations
When CUDA is available and enabled (`FAISS_FORCE_CPU=0`), the system automatically:

* **GPU Index**: Moves FAISS index to GPU memory with FP16 optimization
* **Batch Processing**: Optimizes search parameters for GPU throughput (8x overfetch minimum)
* **Memory Management**: Configures GPU resources with 256MB temp memory allocation
* **Embedding Acceleration**: Optionally moves SentenceTransformer models to GPU for faster query encoding
* **Fallback Safety**: Gracefully falls back to CPU if GPU operations fail

### Index Building
* **CPU Index** – Simple flat index (`artifacts/faiss.idx`) built offline via `scripts/build_index.py`.
* **GPU Index** – IVF-Flat index with clustering trained on GPU for faster build times and optimized search.
* **Metadata** – parallel NumPy array (`artifacts/meta.npy`) stores the original docs / IDs.
* **Embeddings** – generated through `intfloat/e5-base-v2` model with GPU acceleration when available.
* **Query** – cosine-similarity search with optimal parameters based on hardware.

### Performance Benefits
GPU acceleration provides significant improvements:

* **Search Speed**: 10-20x faster vector search compared to CPU
* **Build Time**: 5-10x faster index building with GPU clustering
* **Memory Bandwidth**: Efficient utilization of GPU memory (up to 900 GB/s on modern GPUs)
* **Batch Optimization**: Better throughput with larger batch sizes
* **Lower Latency**: Reduced query response times, especially for multiple concurrent requests

## 4. Prompt Assembly
`app/prompt.py` concatenates:
1. A fixed system prompt that defines the bot persona and constraints.
2. The retrieved context snippets.
3. The user's natural-language question.

## 5. LLM Client (Ollama)
* Runs locally (`docker compose` service `ollama`) so no external API keys are needed.
* Default model is `llama3:8b-instruct-q5_K_M`, configurable via the `OLLAMA_MODEL` env var.

## 6. Hardware Detection and Deployment

### Automatic GPU Detection
The system automatically detects:
* NVIDIA GPU availability via `faiss.get_num_gpus()`
* CUDA toolkit installation via `torch.cuda.is_available()`
* FAISS-GPU package availability via `hasattr(faiss, 'StandardGpuResources')`

### Docker Deployment Modes
* **CPU Mode**: `docker compose -f docker/compose.yaml up`
  - Uses `faiss-cpu` package
  - Optimized for CPU-only environments
  - Falls back gracefully from any GPU code paths

* **GPU Mode**: `docker compose -f docker/compose.yaml -f docker/compose.gpu.yaml up`
  - Uses `faiss-gpu` package with CUDA support
  - Requires NVIDIA Container Toolkit
  - Automatically enables GPU acceleration for all compatible components

### Environment Variables
* `BUILD_MODE=gpu` - Forces GPU index building (requires CUDA)
* `BUILD_MODE=cpu` - Forces CPU-only index building (default)

## 7. Monitoring and Health Checks

The enhanced `/healthz` endpoint provides detailed system information:
```json
{
  "status": "ok",
  "gpu_enabled": true,
  "index_type": "IndexIVFFlat",
  "num_vectors": "50000",
  "embedding_model": "intfloat/e5-base-v2",
  "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

## 6. Observability & Ops
* **Prometheus** — exposed as a compose service on port `9090`, automatically scraping the backend every 15 s.
* **Traefik** — placeholder config in `docker/traefik` for future use as an edge router / HTTPS terminator.
* **Scripts** — `scripts/evaluate.py` gives quantitative accuracy numbers; `scripts/run_tests.sh` is used in CI.

## 7. Data Flow Summary
1. Client issues `/query` (or `/query/stream`) request.
2. Backend validates & normalises the query.
3. Retriever returns the most relevant passages.
4. Prompt builder formats the request for the LLM.
5. Ollama streams the answer back; backend relays chunks to the client.
6. Middleware records metrics for Prometheus.

---

## 8. Deployment Options
| Environment | How to run |
|-------------|------------|
| **Local Dev** | `uvicorn app.main:app --reload` |
| **Docker Compose (CPU)** | `cd docker && docker compose up --build` |
| **Docker Compose (GPU)** | `cd docker && docker compose -f compose.gpu.yaml up --build` |

The architecture is modular so you can swap in a cloud embedding store (e.g. Pinecone), a hosted LLM provider, or additional micro-services without touching core business logic.

---

## 9. Future Improvements
* Web UI (Chat + docs) via Next.js.
* Hot-reloading index updates.
* Elastic / OpenSearch retrieval back-end.
* CI pipeline with `pytest --cov` and pre-commit hooks.

Feel free to open issues or PRs ♻️
