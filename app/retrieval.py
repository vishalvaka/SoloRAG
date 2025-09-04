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

# ─── GPU detection and configuration ──────────────────────────────────────
def _detect_gpu_capability():
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

# ─── load index & models once at import time ──────────────────────────────
logger.info("loading_index", details="Loading FAISS index & embeddings …")

# Load the index and check if we should move it to GPU
cpu_index = faiss.read_index(str(INDEX_FILE))
TEXTS = np.load(META_FILE, allow_pickle=True)

if GPU_AVAILABLE:
    try:
        # Move index to GPU with optimizations
        gpu_resources = faiss.StandardGpuResources()
        
        # Configure GPU memory and options for stability
        gpu_resources.setTempMemory(128 * 1024 * 1024)  # 128MB temp memory (more conservative)
        gpu_options = faiss.GpuClonerOptions()
        gpu_options.useFloat16 = False  # Use FP32 for stability (can enable FP16 later if needed)
        gpu_options.usePrecomputed = True  # Use precomputed tables when available
        
        # Clone index to GPU
        INDEX = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index, gpu_options)
        logger.info("index_gpu", details=f"Successfully moved FAISS index to GPU")
        
        # Clean up CPU index to save memory
        del cpu_index
        
    except Exception as e:
        # Check if this is a CUBLAS error (common with certain FAISS GPU builds)
        if "cublas" in str(e).lower() or "CUBLAS_STATUS" in str(e):
            logger.warning("index_gpu", details=f"CUBLAS error detected: {e}. This is a known compatibility issue with certain FAISS GPU builds. Using CPU index.")
        else:
            logger.warning("index_gpu", details=f"Failed to move index to GPU: {e}, using CPU index")
        INDEX = cpu_index
        GPU_AVAILABLE = False
else:
    INDEX = cpu_index

# Load embedding and reranking models
EMBED   = SentenceTransformer("intfloat/e5-base-v2")
RERANK  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Move embedding model to GPU if available
if GPU_AVAILABLE:
    try:
        import torch
        if torch.cuda.is_available():
            EMBED = EMBED.to('cuda')
            logger.info("embedding_gpu", details="Moved embedding model to GPU")
    except Exception as e:
        logger.warning("embedding_gpu", details=f"Failed to move embedding model to GPU: {e}")

# ─── Ollama config ────────────────────────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b-instruct-q5_K_M")

# ─── GPU-optimized search parameters ──────────────────────────────────────
def _get_optimal_search_params(k: int, overfetch: int):
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
def _search(query: str, k: int = 4, overfetch: int = 5):
    """Vector search + cross-encoder rerank → top-k paragraphs.
    
    Optimized for GPU when available with larger overfetch ratios
    and efficient batch processing.
    """
    params = _get_optimal_search_params(k, overfetch)
    
    # Encode query with GPU acceleration if available
    if GPU_AVAILABLE:
        import torch
        with torch.cuda.device(0):
            q_vec_np = np.asarray(EMBED.encode([query], normalize_embeddings=True), dtype="float32")
    else:
        q_vec_np = np.asarray(EMBED.encode([query], normalize_embeddings=True), dtype="float32")
    
    # Perform vector search with optimized parameters
    search_k = k * params['overfetch']
    
    if GPU_AVAILABLE:
        # GPU search can handle larger batches efficiently
        _, idx = INDEX.search(q_vec_np, search_k)
    else:
        # CPU search with smaller batches
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

# ─── utility functions ───────────────────────────────────────────────────
def get_index_info():
    """Return information about the current index configuration."""
    return {
        "gpu_enabled": GPU_AVAILABLE,
        "index_type": type(INDEX).__name__,
        "num_vectors": INDEX.ntotal if hasattr(INDEX, 'ntotal') else "unknown",
        "embedding_model": "intfloat/e5-base-v2",
        "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    }
