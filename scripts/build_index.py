import json, pathlib, numpy as np, faiss, warnings, time, os
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util

# -------- paths -----------------------------------------------------------
DATA_PATH = pathlib.Path("data/raw/stripe_faqs_full.jsonl")
ART_DIR   = pathlib.Path("artifacts"); ART_DIR.mkdir(exist_ok=True)
IDX_FILE  = ART_DIR / "faiss.idx"
META_FILE = ART_DIR / "meta.npy"

# -------- GPU detection ---------------------------------------------------
def detect_gpu_capability():
    """Detect if GPU is available for index building."""
    try:
        import torch
        ngpus = faiss.get_num_gpus()
        # Check if GPU functions are available (faiss-gpu vs faiss-cpu)
        gpu_functions_available = hasattr(faiss, 'StandardGpuResources')
        return ngpus > 0 and torch.cuda.is_available() and gpu_functions_available
    except:
        return False

GPU_AVAILABLE = detect_gpu_capability()
BUILD_MODE = os.getenv("BUILD_MODE", "cpu")
USE_GPU = GPU_AVAILABLE and BUILD_MODE == "gpu"

print(f"🔧 Build mode: {BUILD_MODE}")
print(f"🎮 GPU available: {GPU_AVAILABLE}")
print(f"⚡ Using GPU for index: {USE_GPU}")

# -------- choose an available embedding model ----------------------------
CANDIDATES = [
    "nomic-ai/nomic-embed-text-v1",
    "intfloat/e5-base-v2",
    "thenlper/gte-base",
]

model = None
for name in CANDIDATES:
    try:
        t0 = time.time()
        model = SentenceTransformer(name)
        
        # Move model to GPU if available for faster embedding
        if USE_GPU:
            import torch
            model = model.to('cuda')
            print(f"✅  Loaded '{name}' on GPU in {time.time()-t0:.1f}s")
        else:
            print(f"✅  Loaded '{name}' on CPU in {time.time()-t0:.1f}s")
        break
    except Exception as e:
        warnings.warn(f"Model '{name}' failed: {e}")

if model is None:
    raise RuntimeError("No embedding model could be loaded. Check internet / HF token.")

# -------- load texts ------------------------------------------------------
texts = []
with DATA_PATH.open() as f:
    for line in f:
        texts.append(json.loads(line)["text"])

print(f"Embedding {len(texts):,} paragraphs …")

# -------- embed in batches -----------------------------------------------
vecs = []
batch_size = 128 if USE_GPU else 64  # Larger batches for GPU

for start in tqdm(range(0, len(texts), 256)):
    batch = texts[start:start+256]
    vec   = model.encode(batch,
                         batch_size=batch_size,
                         normalize_embeddings=True,
                         show_progress_bar=False)
    # Convert to numpy array and ensure float32 dtype
    try:
        import torch
        if torch.is_tensor(vec):  # Handle torch tensors
            vec = vec.cpu().numpy()
    except ImportError:
        pass  # torch not available
    
    vec = np.asarray(vec, dtype="float32")
    vecs.append(vec)

vecs = np.vstack(vecs)
print("Vector matrix:", vecs.shape)

# -------- build optimized FAISS index -----------------------------------
def build_cpu_index(vectors):
    """Build a simple flat index for CPU."""
    print("🔨 Building CPU IndexFlatIP...")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

def build_gpu_optimized_index(vectors):
    """Build GPU-optimized index with IVF clustering."""
    dim = vectors.shape[1]
    nlist = min(int(np.sqrt(len(vectors))), 16384)  # Optimal cluster count
    
    print(f"🚀 Building GPU-optimized IVF index with {nlist} clusters...")
    
    # Create IVF-Flat index for better performance than basic flat
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    
    if USE_GPU and hasattr(faiss, 'StandardGpuResources'):
        try:
            # Move to GPU for training and adding
            gpu_resources = faiss.StandardGpuResources()  # type: ignore
            gpu_resources.setTempMemory(512 * 1024 * 1024)  # 512MB for large datasets
            
            gpu_options = faiss.GpuClonerOptions()  # type: ignore
            gpu_options.useFloat16 = False  # Keep FP32 for training accuracy
            
            # Train on GPU
            gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, index, gpu_options)  # type: ignore
            
            print("🎯 Training IVF centroids on GPU...")
            gpu_index.train(vectors)
            
            print("📝 Adding vectors to GPU index...")
            gpu_index.add(vectors)
            
            # Copy back to CPU for saving (more compatible)
            index = faiss.index_gpu_to_cpu(gpu_index)  # type: ignore
            
            print("✅ GPU index training completed, copied back to CPU for storage")
            
        except Exception as e:
            print(f"⚠️  GPU training failed: {e}, falling back to CPU")
            # Fallback to CPU training
            print("🎯 Training IVF centroids on CPU...")
            index.train(vectors)
            print("📝 Adding vectors to CPU index...")
            index.add(vectors)
    else:
        # Fallback to CPU training
        print("🎯 Training IVF centroids on CPU...")
        index.train(vectors)
        print("📝 Adding vectors to CPU index...")
        index.add(vectors)
    
    # Set optimal search parameters
    index.nprobe = max(1, nlist // 8)  # Search ~12.5% of clusters by default
    print(f"🔍 Set default nprobe to {index.nprobe} (searching {index.nprobe/nlist*100:.1f}% of clusters)")
    
    return index

# Choose index type based on GPU availability and dataset size
if USE_GPU and len(vecs) > 10000:  # Use advanced index for larger datasets
    index = build_gpu_optimized_index(vecs)
    print(f"🎉 GPU-optimized IVF index built: {index.ntotal:,} vectors, {getattr(index, 'nlist', 'N/A')} clusters")
else:
    index = build_cpu_index(vecs)
    print(f"🎉 CPU flat index built: {index.ntotal:,} vectors")

# -------- save index and metadata ----------------------------------------
print(f"💾 Saving index to {IDX_FILE}")
faiss.write_index(index, str(IDX_FILE))

print(f"💾 Saving metadata to {META_FILE}")
np.save(META_FILE, np.array(texts, dtype=object))

print(f"\n🎉 Index building completed!")
print(f"   📁 Index: {IDX_FILE} ({IDX_FILE.stat().st_size / 1024 / 1024:.1f} MB)")
print(f"   📁 Metadata: {META_FILE} ({META_FILE.stat().st_size / 1024 / 1024:.1f} MB)")
print(f"   🎯 Index type: {type(index).__name__}")
if hasattr(index, 'nlist'):
    print(f"   🔍 Clusters: {index.nlist}")
if hasattr(index, 'nprobe'):
    print(f"   🎯 Default nprobe: {index.nprobe}")
