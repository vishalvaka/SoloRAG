import json, pathlib, numpy as np, faiss, warnings, time
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util

# -------- paths -----------------------------------------------------------
DATA_PATH = pathlib.Path("data/raw/stripe_faqs_full.jsonl")
ART_DIR   = pathlib.Path("artifacts"); ART_DIR.mkdir(exist_ok=True)
IDX_FILE  = ART_DIR / "faiss.idx"
META_FILE = ART_DIR / "meta.npy"

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
        print(f"✅  Loaded '{name}' in {time.time()-t0:.1f}s")
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
for start in tqdm(range(0, len(texts), 256)):
    batch = texts[start:start+256]
    vec   = model.encode(batch,
                         batch_size=64,
                         normalize_embeddings=True,
                         show_progress_bar=False)
    vecs.append(vec.astype("float32"))

vecs = np.vstack(vecs)
print("Vector matrix:", vecs.shape)

# -------- build & save FAISS ---------------------------------------------
index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)
faiss.write_index(index, str(IDX_FILE))
np.save(META_FILE, np.array(texts, dtype=object))

print("🎉  FAISS index saved →", IDX_FILE, "| meta →", META_FILE)
