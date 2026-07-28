"""
retriever.py

Phase 3: RAG Retrieval Module.
Loads existing local persistent ChromaDB collection ("research_papers") and
uses BAAI/bge-base-en-v1.5 to perform semantic vector search with optional domain filtering.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Distance threshold: cosine distance scores above 0.40 are filtered out as irrelevant
DISTANCE_THRESHOLD = 0.40

_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None
_embedder: Optional[SentenceTransformer] = None


def _init_retriever() -> Tuple[chromadb.Collection, SentenceTransformer]:
    """Initialize ChromaDB client and SentenceTransformer embedder lazily."""
    global _chroma_client, _collection, _embedder
    
    if _collection is None:
        if not CHROMA_DB_DIR.exists():
            raise FileNotFoundError(f"ChromaDB directory not found at '{CHROMA_DB_DIR.resolve()}'. Please run ingest.py first.")
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        _collection = _chroma_client.get_collection(name=COLLECTION_NAME)
        
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        
    return _collection, _embedder


def retrieve_chunks(
    query: str,
    domain_filter: Optional[str] = None,
    n_results: int = 5,
    distance_threshold: float = DISTANCE_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Retrieve semantic chunks matching user query, filtering out chunks exceeding distance_threshold.
    
    Args:
        query: User search string or question.
        domain_filter: Optional domain filter ('cybersecurity', 'ai_ml', 'both', or None).
        n_results: Number of chunks to retrieve (default: 5).
        distance_threshold: Maximum allowed cosine distance (default: 0.65).
        
    Returns:
        List of dicts containing 'text', 'metadata', and 'distance' for accepted chunks.
    """
    if not query or not query.strip():
        return []
        
    collection, embedder = _init_retriever()
    
    query_vector = embedder.encode(query, normalize_embeddings=True).tolist()
    
    where_clause = None
    if domain_filter:
        clean_domain = domain_filter.lower().strip()
        if clean_domain not in ("both", "all", "none"):
            where_clause = {"domain": clean_domain}
            
    query_kwargs = {
        "query_embeddings": [query_vector],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"]
    }
    if where_clause:
        query_kwargs["where"] = where_clause
        
    res = collection.query(**query_kwargs)
    
    retrieved_items = []
    if res and res.get("documents") and len(res["documents"]) > 0:
        docs = res["documents"][0]
        metas = res["metadatas"][0] if res.get("metadatas") else [{}] * len(docs)
        dists = res["distances"][0] if res.get("distances") else [0.0] * len(docs)
        
        for doc, meta, dist in zip(docs, metas, dists):
            # Filter out any chunk whose distance exceeds distance_threshold
            if dist <= distance_threshold:
                retrieved_items.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                })
            
    return retrieved_items


if __name__ == "__main__":
    import json
    print("Testing modules.retrieval.retriever standalone...")
    results = retrieve_chunks("zero trust security architecture", domain_filter="cybersecurity", n_results=2)
    print(f"Retrieved {len(results)} chunks.")
    for idx, item in enumerate(results, 1):
        print(f"\n--- Chunk #{idx} ---")
        print("Metadata:", json.dumps(item["metadata"], indent=2))
        print("Text:", item["text"][:200] + "...")
