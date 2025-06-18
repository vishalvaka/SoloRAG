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
        C[Retriever (FAISS)]
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
  * `GET /health` – liveness probe.
  * `POST /answer` – main QA endpoint, validates input, streams answer chunks.
* **Middleware** – a custom Prometheus middleware records request counts, durations and error rates.
* **Testing** – pytest with `httpx.AsyncClient` gives ~95 % unit-test coverage.

## 3. Retrieval Layer
* **Index** – FAISS flat index (`artifacts/faiss.idx`) built offline via `scripts/build_index.py`.
* **Metadata** – parallel NumPy array (`artifacts/meta.npy`) stores the original docs / IDs.
* **Embeddings** – generated through the LLM/embedding model configured in `scripts/build_index.py`.
* **Query** – cosine-similarity top-k search (default k = 5).

## 4. Prompt Assembly
`app/prompt.py` concatenates:
1. A fixed system prompt that defines the bot persona and constraints.
2. The retrieved context snippets.
3. The user's natural-language question.

## 5. LLM Client (Ollama)
* Runs locally (`docker compose` service `ollama`) so no external API keys are needed.
* Default model is `llama3:8b-instruct-q5_K_M`, configurable via the `OLLAMA_MODEL` env var.
* Streaming responses are proxied straight back to the caller.

## 6. Observability & Ops
* **Prometheus** — exposed as a compose service on port `9090`, automatically scraping the backend every 15 s.
* **Traefik** — placeholder config in `docker/traefik` for future use as an edge router / HTTPS terminator.
* **Scripts** — `scripts/evaluate.py` gives quantitative accuracy numbers; `scripts/run_tests.sh` is used in CI.

## 7. Data Flow Summary
1. Client issues `/answer` request.
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
