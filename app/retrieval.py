# app/retrieval.py
"""
Vector search  ➜  rerank  ➜  build prompt  ➜  call Ollama
Exposes a single async function:  get_answer(question:str) -> (markdown, sources)
"""

import os, pathlib, asyncio, json, textwrap
import requests, faiss, numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from .ollama_client import generate as call_ollama, stream_generate as call_ollama_stream
from .prompt import build_prompt
from .logger import logger

# ─── artefact paths ───────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).resolve().parent.parent
ART_DIR    = BASE_DIR / "artifacts"
INDEX_FILE = ART_DIR / "faiss.idx"
META_FILE  = ART_DIR / "meta.npy"

# ─── load index & models once at import time ──────────────────────────────
logger.info("loading_index", details="Loading FAISS index & embeddings …")
INDEX   = faiss.read_index(str(INDEX_FILE))
TEXTS   = np.load(META_FILE, allow_pickle=True)
EMBED   = SentenceTransformer("intfloat/e5-base-v2")
RERANK  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ─── Ollama config ────────────────────────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b-instruct-q5_K_M")

# ─── helpers ──────────────────────────────────────────────────────────────
def _search(query: str, k: int = 4, overfetch: int = 5):
    """Vector search + cross-encoder rerank → top-k paragraphs."""
    q_vec_np = np.asarray(EMBED.encode([query], normalize_embeddings=True), dtype="float32")
    _, idx = INDEX.search(q_vec_np, k * overfetch)

    passages = [TEXTS[i] for i in idx[0]]
    scores   = RERANK.predict([[query, p] for p in passages])
    ranked   = sorted(
        zip(passages, [float(s) for s in scores]),
        key=lambda x: x[1],
        reverse=True,
    )[:k]
    return [{"text": p, "score": float(s)} for p, s in ranked]

# ─── public API ───────────────────────────────────────────────────────────
async def get_answer(question: str):
    """
    Returns (markdown_answer, source_snippets)
    source_snippets: List[{"text": str, "score": float}]
    """
    ctx = _search(question)
    prompt = build_prompt(question, ctx)
    answer = await call_ollama(prompt)
    return answer, ctx

# ─── streaming variant ───────────────────────────────────────────────────
async def stream_answer(question: str):
    """Async generator yielding answer chunks; yields sources at end as JSON string."""
    ctx = _search(question)
    prompt = build_prompt(question, ctx)

    async for chunk in call_ollama_stream(prompt):
        yield chunk
    # After streaming answer, append newline and JSON sources
    yield "\n\n[SOURCES] " + json.dumps(ctx)
