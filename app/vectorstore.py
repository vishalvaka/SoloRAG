# app/vectorstore.py
"""Pluggable vector store: FAISS (local) or OpenSearch (AWS).

Both implementations expose:
  - ``async initialize()``
  - ``search(query, k) -> list[dict]``  (vector search + rerank)
  - ``get_info() -> dict``
"""

from __future__ import annotations

import asyncio
import os
import pathlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from .config import get_settings
from .logger import logger

# ─── Shared models (loaded once, used by both backends) ──────────────────
_EMBED: SentenceTransformer | None = None
_RERANK: CrossEncoder | None = None
_models_lock = asyncio.Lock()

EMBEDDING_MODEL = "intfloat/e5-base-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


async def _ensure_models() -> tuple[SentenceTransformer, CrossEncoder]:
    """Lazy-load embedding and reranking models (shared across backends)."""
    global _EMBED, _RERANK
    if _EMBED is not None and _RERANK is not None:
        return _EMBED, _RERANK
    async with _models_lock:
        if _EMBED is not None and _RERANK is not None:
            return _EMBED, _RERANK
        logger.info("loading_models", details="Loading embedding + rerank models ...")
        _EMBED = SentenceTransformer(EMBEDDING_MODEL)
        _RERANK = CrossEncoder(RERANK_MODEL)
        logger.info("models_loaded")
    return _EMBED, _RERANK


def _rerank(query: str, passages: list[str], reranker: CrossEncoder, k: int) -> list[dict]:
    """Cross-encoder rerank and return top-k."""
    pairs = [[query, p] for p in passages]
    scores = reranker.predict(pairs)
    ranked = sorted(
        zip(passages, [float(s) for s in scores]),
        key=lambda x: x[1],
        reverse=True,
    )[:k]
    return [{"text": p, "score": s} for p, s in ranked]


# ═══════════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════════

class VectorStore(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int = 4) -> list[dict]: ...

    @abstractmethod
    def get_info(self) -> dict[str, Any]: ...


# ═══════════════════════════════════════════════════════════════════════════
# FAISS implementation (local dev / standalone)
# ═══════════════════════════════════════════════════════════════════════════

class FaissVectorStore(VectorStore):
    """FAISS flat-IP index loaded from local artifacts."""

    def __init__(self) -> None:
        self._index = None
        self._texts: np.ndarray | None = None
        self._embed: SentenceTransformer | None = None
        self._rerank: CrossEncoder | None = None
        self._initialized = False
        self._lock = asyncio.Lock()

        base_dir = pathlib.Path(__file__).resolve().parent.parent
        art_dir = base_dir / "artifacts"
        self._index_file = art_dir / "faiss.idx"
        self._meta_file = art_dir / "meta.npy"

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            import faiss  # type: ignore[import-untyped]

            logger.info("faiss_loading", details="Loading FAISS index ...")
            try:
                io_flags = getattr(faiss, "IO_FLAG_MMAP", 0)
                self._index = (
                    faiss.read_index(str(self._index_file), io_flags)
                    if io_flags
                    else faiss.read_index(str(self._index_file))
                )
            except Exception:
                self._index = faiss.read_index(str(self._index_file))

            self._texts = np.load(self._meta_file, allow_pickle=True)
            self._embed, self._rerank = await _ensure_models()
            self._initialized = True
            logger.info("faiss_loaded", num_vectors=self._index.ntotal)

    def search(self, query: str, k: int = 4, overfetch: int = 5) -> list[dict]:
        assert self._index is not None and self._texts is not None
        q_vec = np.asarray(
            self._embed.encode([query], normalize_embeddings=True), dtype="float32"  # type: ignore[union-attr]
        )
        _, idx = self._index.search(q_vec, k * overfetch)
        passages = [self._texts[i] for i in idx[0]]
        return _rerank(query, passages, self._rerank, k)  # type: ignore[arg-type]

    def get_info(self) -> dict[str, Any]:
        settings = get_settings()
        return {
            "gpu_enabled": settings.FAISS_FORCE_CPU != "1",
            "index_type": type(self._index).__name__ if self._index else "uninitialized",
            "num_vectors": self._index.ntotal if self._index else "unknown",
            "embedding_model": EMBEDDING_MODEL,
            "rerank_model": RERANK_MODEL,
        }


# ═══════════════════════════════════════════════════════════════════════════
# OpenSearch implementation (AWS production)
# ═══════════════════════════════════════════════════════════════════════════

class OpenSearchVectorStore(VectorStore):
    """k-NN vector search backed by Amazon OpenSearch Service."""

    def __init__(self) -> None:
        self._embed: SentenceTransformer | None = None
        self._rerank: CrossEncoder | None = None
        self._client = None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            from opensearchpy import OpenSearch  # type: ignore[import-untyped]

            settings = get_settings()
            self._index_name = settings.OPENSEARCH_INDEX

            # Parse URL for host/port
            url = settings.OPENSEARCH_URL
            use_ssl = url.startswith("https")
            host = url.replace("https://", "").replace("http://", "").rstrip("/")
            port = 443 if use_ssl else 9200
            if ":" in host:
                host, port_str = host.rsplit(":", 1)
                port = int(port_str)

            self._client = OpenSearch(
                hosts=[{"host": host, "port": port}],
                use_ssl=use_ssl,
                verify_certs=use_ssl,
                ssl_show_warn=False,
            )

            self._embed, self._rerank = await _ensure_models()
            self._initialized = True
            logger.info("opensearch_connected", url=settings.OPENSEARCH_URL, index=self._index_name)

    def search(self, query: str, k: int = 4, overfetch: int = 5) -> list[dict]:
        assert self._client is not None and self._embed is not None
        q_vec = self._embed.encode([query], normalize_embeddings=True)[0].tolist()

        body = {
            "size": k * overfetch,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": q_vec,
                        "k": k * overfetch,
                    }
                }
            },
            "_source": ["text"],
        }

        resp = self._client.search(index=self._index_name, body=body)
        passages = [hit["_source"]["text"] for hit in resp["hits"]["hits"]]

        if not passages:
            return []

        return _rerank(query, passages, self._rerank, k)  # type: ignore[arg-type]

    def get_info(self) -> dict[str, Any]:
        settings = get_settings()
        count = "unknown"
        if self._client:
            try:
                count = str(self._client.count(index=settings.OPENSEARCH_INDEX)["count"])
            except Exception:
                pass
        return {
            "gpu_enabled": False,
            "index_type": "OpenSearch k-NN",
            "num_vectors": count,
            "embedding_model": EMBEDDING_MODEL,
            "rerank_model": RERANK_MODEL,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

_store_instance: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Return a singleton vector store based on ``VECTOR_BACKEND``."""
    global _store_instance
    if _store_instance is not None:
        return _store_instance

    settings = get_settings()
    backend = settings.VECTOR_BACKEND.lower()

    if backend == "faiss":
        _store_instance = FaissVectorStore()
    elif backend == "opensearch":
        _store_instance = OpenSearchVectorStore()
    else:
        raise ValueError(f"Unknown VECTOR_BACKEND: {backend!r}. Must be 'faiss' or 'opensearch'.")

    return _store_instance
