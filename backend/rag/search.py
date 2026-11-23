"""
Vector search over the exported index (backend/data), matching the behaviour of
modules/retrieval/retriever.py: cosine distance, optional domain filter, top-k,
then drop anything above DISTANCE_THRESHOLD.
"""

import gzip
import json
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from . import config
from .embedder import embed_query

_lock = threading.Lock()
_matrix: Optional[np.ndarray] = None
_chunks: Optional[List[Dict[str, Any]]] = None
_domains: Optional[np.ndarray] = None


def _load_index():
    global _matrix, _chunks, _domains
    if _matrix is not None:
        return _matrix, _chunks, _domains
    with _lock:
        if _matrix is None:
            if not config.EMBEDDINGS_PATH.exists() or not config.CHUNKS_PATH.exists():
                raise FileNotFoundError(
                    "Search index not found in backend/data. Run: python backend/scripts/build_index.py"
                )
            matrix = np.load(config.EMBEDDINGS_PATH).astype(np.float32)
            with gzip.open(config.CHUNKS_PATH, "rt", encoding="utf-8") as f:
                chunks = json.load(f)
            if len(chunks) != matrix.shape[0]:
                raise ValueError("Index files are out of sync; rebuild with backend/scripts/build_index.py")
            _domains = np.asarray([c.get("domain", "") for c in chunks])
            _chunks = chunks
            _matrix = matrix
    return _matrix, _chunks, _domains


def index_size() -> int:
    matrix, _, _ = _load_index()
    return int(matrix.shape[0])


def retrieve_chunks(
    query: str,
    domain_filter: Optional[str] = None,
    n_results: int = config.N_RESULTS,
    distance_threshold: float = config.DISTANCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        return []

    matrix, chunks, domains = _load_index()
    query_vector = embed_query(query)

    candidates = np.arange(matrix.shape[0])
    clean_domain = (domain_filter or "").lower().strip()
    if clean_domain and clean_domain not in ("both", "all", "none"):
        candidates = np.flatnonzero(domains == clean_domain)
        if candidates.size == 0:
            return []

    distances = 1.0 - matrix[candidates] @ query_vector  # cosine distance (vectors are normalised)
    k = min(n_results, candidates.size)
    top = np.argpartition(distances, k - 1)[:k]
    top = top[np.argsort(distances[top])]

    results = []
    for i in top:
        dist = float(distances[i])
        if dist <= distance_threshold:
            results.append({"chunk": chunks[int(candidates[i])], "distance": dist})
    return results
