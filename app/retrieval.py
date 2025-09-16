# app/retrieval.py
"""
Vector search  ➜  rerank  ➜  build prompt  ➜  call Ollama
Exposes a single async function:  get_answer(question:str) -> (markdown, sources)
"""

import os, pathlib, asyncio, json, textwrap
from typing import AsyncGenerator
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

# ─── GPU detection and configuration ──────────────────────────────────────
def _detect_gpu_capability() -> bool:
    """Detect if GPU FAISS is available and functional."""
    try:
        import torch
        
        # Check if FAISS has GPU functions available
        gpu_functions_available = hasattr(faiss, 'StandardGpuResources')
        
        # Check if PyTorch has CUDA support
        torch_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
        
        # Check FAISS GPU count (might be 0 in virtualized environments)
        ngpus = faiss.get_num_gpus()
        
        logger.info("gpu_detection", details=f"FAISS GPUs={ngpus}, PyTorch CUDA={torch_cuda}, GPU functions={gpu_functions_available}")
        
        # Use GPU if we have both PyTorch CUDA and FAISS GPU functions
        # (even if faiss.get_num_gpus() returns 0 in virtualized environments)
        if torch_cuda and gpu_functions_available:
            logger.info("gpu_detection", details="GPU acceleration enabled")
            return True
        
        logger.info("gpu_detection", details="GPU acceleration disabled - using CPU")
        return False
    except Exception as e:
        logger.warning("gpu_detection", details=f"GPU detection failed: {e}, falling back to CPU")
        return False

GPU_AVAILABLE = _detect_gpu_capability()

# Force CPU indexing by default for faster startup and simpler behavior
# Set FAISS_FORCE_CPU=0 to allow GPU usage again.
try:
    if os.getenv("FAISS_FORCE_CPU", "1") == "1":
        GPU_AVAILABLE = False
        logger.info("gpu_forced_off", details="FAISS forced to CPU mode (FAISS_FORCE_CPU=1)")
except Exception:
    GPU_AVAILABLE = False

# ─── lazy initialization (non-blocking import) ────────────────────────────
INDEX = None
TEXTS = None
EMBED = None
RERANK = None
_initialized = False
_init_lock: asyncio.Lock = asyncio.Lock()
_gpu_upgrade_task: asyncio.Task | None = None

async def _ensure_initialized() -> None:
    """Initialize heavy resources on first use.
    Safe to call multiple times; concurrent callers will serialize on the lock.
    """
    global INDEX, TEXTS, EMBED, RERANK, _initialized, GPU_AVAILABLE
    if _initialized:
        return
    async with _init_lock:
        if _initialized:
            return
        logger.info("loading_index", details="Loading FAISS index & embeddings …")
        # Load FAISS artefacts (prefer memory-mapped for fast startup)
        try:
            io_flags = getattr(faiss, "IO_FLAG_MMAP", 0)
            cpu_index = faiss.read_index(str(INDEX_FILE), io_flags) if io_flags else faiss.read_index(str(INDEX_FILE))
        except Exception:
            cpu_index = faiss.read_index(str(INDEX_FILE))
        TEXTS = np.load(META_FILE, allow_pickle=True)

        # Always use CPU index (fast startup)
        INDEX = cpu_index

        # Load embedding and reranking models
        EMBED = SentenceTransformer("intfloat/e5-base-v2")
        RERANK = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Move embedding model to GPU if available
        if GPU_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available():
                    EMBED = EMBED.to('cuda')
                    logger.info("embedding_gpu", details="Moved embedding model to GPU")
            except Exception as e:
                logger.warning("embedding_gpu", details=f"Failed to move embedding model to GPU: {e}")

        _initialized = True
        logger.info("init_complete", details="Retrieval stack initialized")

# ─── Ollama config ────────────────────────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b-instruct-q5_K_M")

# ─── GPU-optimized search parameters ──────────────────────────────────────
def _get_optimal_search_params(k: int, overfetch: int) -> dict:
    """Get optimal search parameters based on GPU availability and index type."""
    if GPU_AVAILABLE:
        # For GPU, we can afford larger overfetch for better accuracy
        # since GPU memory bandwidth is much higher
        return {
            'k': k,
            'overfetch': max(overfetch, 8),  # Minimum 8x overfetch on GPU
            'batch_size': 32,  # Optimal batch size for GPU
        }
    else:
        return {
            'k': k,
            'overfetch': overfetch,
            'batch_size': 16,  # Smaller batch for CPU
        }

# ─── helpers ──────────────────────────────────────────────────────────────
def _search(query: str, k: int = 4, overfetch: int = 5) -> list:
    """Vector search + cross-encoder rerank → top-k paragraphs.
    
    Optimized for GPU when available with larger overfetch ratios
    and efficient batch processing.
    """
    params = _get_optimal_search_params(k, overfetch)
    
    # Encode query with GPU acceleration if available
    q_vec_np = np.asarray(EMBED.encode([query], normalize_embeddings=True), dtype="float32")
    
    # Perform vector search with optimized parameters
    search_k = k * params['overfetch']
    
    # CPU search
    _, idx = INDEX.search(q_vec_np, search_k)
    
    # Extract passages and rerank
    passages = [TEXTS[i] for i in idx[0]]
    
    # Batch reranking for efficiency
    query_passage_pairs = [[query, p] for p in passages]
    scores = RERANK.predict(query_passage_pairs)
    
    # Sort by reranking scores and return top-k
    ranked = sorted(
        zip(passages, [float(s) for s in scores]),
        key=lambda x: x[1],
        reverse=True,
    )[:k]
    
    return [{"text": p, "score": float(s)} for p, s in ranked]

# ─── public API ───────────────────────────────────────────────────────────
async def get_answer(question: str) -> tuple:
    """
    Returns (markdown_answer, source_snippets)
    source_snippets: List[{"text": str, "score": float}]
    """
    await _ensure_initialized()
    ctx = _search(question)
    prompt = build_prompt(question, ctx)
    answer = await call_ollama(prompt)
    return answer, ctx

# ─── streaming variant ───────────────────────────────────────────────────
async def stream_answer(question: str) -> AsyncGenerator[str, None]:
    """Async generator yielding answer chunks; yields sources at end as JSON string."""
    await _ensure_initialized()
    ctx = _search(question)
    prompt = build_prompt(question, ctx)

    async for chunk in call_ollama_stream(prompt):
        yield chunk
    # After streaming answer, append newline and JSON sources
    yield "\n\n[SOURCES] " + json.dumps(ctx)

# ─── utility functions ───────────────────────────────────────────────────
def get_index_info() -> dict:
    """Return information about the current index configuration."""
    info = {
        "gpu_enabled": GPU_AVAILABLE,
        "index_type": type(INDEX).__name__ if INDEX is not None else "uninitialized",
        "num_vectors": INDEX.ntotal if hasattr(INDEX, 'ntotal') else "unknown",
        "embedding_model": "intfloat/e5-base-v2",
        "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }
    return info
