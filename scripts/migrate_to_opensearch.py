#!/usr/bin/env python3
"""Migrate FAISS artifacts (meta.npy) into an OpenSearch k-NN index.

Usage:
    export OPENSEARCH_URL=https://your-opensearch-domain:443
    export OPENSEARCH_INDEX=solorag-vectors
    python scripts/migrate_to_opensearch.py

The script:
  1. Loads ``artifacts/meta.npy`` (text passages)
  2. Encodes them with the same embedding model used at query time
  3. Creates the OpenSearch index with a k-NN mapping
  4. Bulk-indexes all documents
"""

from __future__ import annotations

import os
import sys
import pathlib

import numpy as np
from opensearchpy import OpenSearch, helpers  # type: ignore[import-untyped]
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "https://localhost:9200")
INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "solorag-vectors")
EMBEDDING_MODEL = "intfloat/e5-base-v2"
BATCH_SIZE = 64

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
META_FILE = BASE_DIR / "artifacts" / "meta.npy"


def main() -> None:
    if not META_FILE.exists():
        print(f"ERROR: {META_FILE} not found. Build the FAISS index first.")
        sys.exit(1)

    # 1. Load texts
    texts: np.ndarray = np.load(META_FILE, allow_pickle=True)
    print(f"Loaded {len(texts)} passages from {META_FILE}")

    # 2. Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dim}")

    # 3. Connect to OpenSearch
    url = OPENSEARCH_URL
    use_ssl = url.startswith("https")
    host = url.replace("https://", "").replace("http://", "").rstrip("/")
    port = 443 if use_ssl else 9200
    if ":" in host:
        host, port_str = host.rsplit(":", 1)
        port = int(port_str)

    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        use_ssl=use_ssl,
        verify_certs=use_ssl,
        ssl_show_warn=False,
    )

    # 4. Create index with k-NN mapping
    if client.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' already exists -- deleting ...")
        client.indices.delete(index=INDEX_NAME)

    mapping = {
        "settings": {
            "index.knn": True,
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                    },
                },
            }
        },
    }
    client.indices.create(index=INDEX_NAME, body=mapping)
    print(f"Created index '{INDEX_NAME}' (dim={dim})")

    # 5. Encode and bulk-index
    total = len(texts)
    actions = []
    for i in tqdm(range(0, total, BATCH_SIZE), desc="Encoding & indexing"):
        batch_texts = texts[i : i + BATCH_SIZE].tolist()
        embeddings = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
        for text, emb in zip(batch_texts, embeddings):
            actions.append({
                "_index": INDEX_NAME,
                "_source": {
                    "text": text,
                    "embedding": emb.tolist(),
                },
            })
        if len(actions) >= 500:
            helpers.bulk(client, actions)
            actions = []

    if actions:
        helpers.bulk(client, actions)

    client.indices.refresh(index=INDEX_NAME)
    count = client.count(index=INDEX_NAME)["count"]
    print(f"Done! Indexed {count} documents into '{INDEX_NAME}'.")


if __name__ == "__main__":
    main()
