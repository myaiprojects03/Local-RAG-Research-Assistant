"""
build_index.py

Exports the local ChromaDB collection ("research_papers") into lightweight files
that the Vercel FastAPI backend can search with plain NumPy:

    backend/data/embeddings.npy   float16 matrix (n_chunks x 768), L2-normalised
    backend/data/chunks.json.gz   chunk text + metadata (+ resolved arXiv URL)
    backend/data/manifest.json    counts, model name, build info

ChromaDB, sentence-transformers and torch are far too large for a Vercel Python
function (250 MB limit), so the deployed backend never imports them. Run this script
locally (where chromadb is installed) whenever you re-run modules/ingestion/ingest.py:

    python backend/scripts/build_index.py
"""

import gzip
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
METADATA_JSON = PROJECT_ROOT / "Papers" / "metadata.json"
DATA_DIR = BACKEND_DIR / "data"
COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
PAGE_SIZE = 2000
# Chunks shorter than this (after stripping) are page headers / title-only pages such as
# "A PREPRINT - JANUARY 2, 2023". They carry no answerable content but embed close to
# title-style queries, crowding real passages out of the top-k, so they are dropped.
MIN_CHUNK_CHARS = 150


def load_url_map() -> dict:
    """pdf_filename -> source_url from Papers/metadata.json (if present)."""
    if not METADATA_JSON.exists():
        print(f"Note: {METADATA_JSON} not found; falling back to URLs derived from filenames.")
        return {}
    with open(METADATA_JSON, "r", encoding="utf-8") as f:
        return {
            r["pdf_filename"]: r["source_url"]
            for r in json.load(f)
            if r.get("pdf_filename") and r.get("source_url")
        }


def fallback_url(source_file: str) -> str:
    """Same filename -> arXiv URL fallback used in modules/generation/rag_pipeline.py."""
    match = re.match(r"^([a-zA-Z0-9\.\-]+?)_", source_file or "")
    if not match:
        return ""
    raw_id = match.group(1)
    formatted_id = f"{raw_id[:4]}.{raw_id[4:]}" if len(raw_id) >= 9 and raw_id[:4].isdigit() else raw_id
    return f"https://arxiv.org/abs/{formatted_id}"


def main() -> None:
    # chromadb is a local-only dependency: it is deliberately NOT in backend/requirements.txt
    # (too large for Vercel), so the backend venv cannot resolve it. Run this script with the
    # project's main Python environment, where chromadb is installed via the root requirements.txt.
    try:
        import chromadb  # pyright: ignore[reportMissingImports]
    except ImportError:
        sys.exit(
            "chromadb is not installed in this Python environment.\n"
            "Run this script with the main project environment (pip install -r requirements.txt at the "
            "project root), not backend/.venv:\n    python backend/scripts/build_index.py"
        )

    if not CHROMA_DB_DIR.exists():
        sys.exit(f"ChromaDB directory not found at {CHROMA_DB_DIR}. Run modules/ingestion/ingest.py first.")

    collection = chromadb.PersistentClient(path=str(CHROMA_DB_DIR)).get_collection(COLLECTION_NAME)
    total = collection.count()
    print(f"Exporting {total} chunks from '{COLLECTION_NAME}'...")

    url_map = load_url_map()
    ids, docs, metas, embs = [], [], [], []
    for offset in range(0, total, PAGE_SIZE):
        page = collection.get(
            limit=PAGE_SIZE,
            offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )
        ids.extend(page["ids"])
        docs.extend(page["documents"])
        metas.extend(page["metadatas"])
        embs.append(np.asarray(page["embeddings"], dtype=np.float32))

    matrix = np.vstack(embs)
    # Re-normalise defensively so a dot product equals cosine similarity.
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)

    keep = [i for i, doc in enumerate(docs) if len((doc or "").strip()) >= MIN_CHUNK_CHARS]
    dropped = len(docs) - len(keep)
    print(f"Dropping {dropped} near-empty chunks (< {MIN_CHUNK_CHARS} chars); keeping {len(keep)}.")
    matrix = matrix[keep]
    ids = [ids[i] for i in keep]
    docs = [docs[i] for i in keep]
    metas = [metas[i] for i in keep]

    chunks = []
    for chunk_id, doc, meta in zip(ids, docs, metas):
        source_file = meta.get("source_file", "")
        chunks.append({
            "id": chunk_id,
            "text": doc,
            "title": meta.get("title", "Untitled Paper"),
            "page": meta.get("page", "?"),
            "domain": meta.get("domain", ""),
            "subtopic": meta.get("subtopic", ""),
            "source_file": source_file,
            "source_url": url_map.get(source_file) or fallback_url(source_file),
        })

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "embeddings.npy", matrix.astype(np.float16))
    with gzip.open(DATA_DIR / "chunks.json.gz", "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(chunks, f, ensure_ascii=False, separators=(",", ":"))

    domains = {}
    for c in chunks:
        domains[c["domain"]] = domains.get(c["domain"], 0) + 1
    manifest = {
        "collection": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "chunks": len(chunks),
        "source_chunks": total,
        "dropped_near_empty_chunks": dropped,
        "min_chunk_chars": MIN_CHUNK_CHARS,
        "dimensions": int(matrix.shape[1]),
        "papers": len({c["source_file"] for c in chunks}),
        "domains": domains,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(DATA_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    for name in ("embeddings.npy", "chunks.json.gz", "manifest.json"):
        print(f"  {name}: {(DATA_DIR / name).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
