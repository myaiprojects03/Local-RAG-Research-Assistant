"""
ingest.py

Phase 2: Local RAG Vector Ingestion Pipeline Module.
Processes research PDFs from /Papers/ domain and subtopic subfolders:
1. Extracts text page-by-page using PyMuPDF (fitz).
2. Cleans text and strips reference/bibliography sections.
3. Chunks text using RecursiveCharacterTextSplitter with page preservation.
4. Tags chunks with rich metadata (domain, subtopic, title, source_file, page).
5. Generates embeddings using sentence-transformers (BAAI/bge-base-en-v1.5).
6. Stores chunks and metadata in a single persistent local ChromaDB collection.
7. Supports incremental ingestion via chroma_db/ingested_files.json.
"""

import os
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
PAPERS_DIR = PROJECT_ROOT / "Papers"
METADATA_MANIFEST = PAPERS_DIR / "metadata.json"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
INGESTED_MANIFEST = CHROMA_DB_DIR / "ingested_files.json"
COLLECTION_NAME = "research_papers"

CHUNK_SIZE = 2000     # ~500-600 tokens
CHUNK_OVERLAP = 300   # ~15% overlap
BATCH_SIZE = 100      # Vector batch insertion size for ChromaDB

REF_HEADER_PATTERN = re.compile(
    r'^(?:references|bibliography|works cited|literature cited)\b',
    re.IGNORECASE | re.MULTILINE
)


def load_title_mapping(metadata_file: Path) -> Dict[str, str]:
    """Load metadata.json to map pdf_filename -> paper title."""
    mapping = {}
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                records = json.load(f)
                for r in records:
                    pdf_filename = r.get("pdf_filename")
                    title = r.get("title")
                    if pdf_filename and title:
                        mapping[pdf_filename] = title
        except Exception as e:
            print(f"Warning: Could not load metadata manifest ({e}). Falling back to filenames.")
    return mapping


def load_ingested_files(manifest_path: Path) -> Set[str]:
    """Load set of already ingested PDF filenames."""
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()


def save_ingested_files(manifest_path: Path, ingested_set: Set[str]) -> None:
    """Save updated set of ingested PDF filenames to manifest."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(ingested_set)), f, indent=2, ensure_ascii=False)
    temp_path.replace(manifest_path)


def clean_page_text(text: str) -> str:
    """Clean page text: strip excessive whitespace and common PDF header/footer artifacts."""
    if not text:
        return ""
    
    text = text.replace('\x00', '')
    lines = [line.strip() for line in text.splitlines()]
    
    filtered_lines = []
    for line in lines:
        if re.match(r'^(?:page\s+\d+(?:\s+of\s+\d+)?|\d+)$', line, re.IGNORECASE):
            continue
        filtered_lines.append(line)
        
    cleaned = "\n".join(filtered_lines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


def extract_pages_from_pdf(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Extract text page-by-page from a PDF using PyMuPDF (fitz).
    Strips the references/bibliography section if detected near the end.
    Returns list of tuples: (page_number_1_based, page_text).
    """
    doc = fitz.open(pdf_path)
    extracted_pages = []
    ref_found = False

    for page_idx in range(len(doc)):
        if ref_found:
            break
            
        page = doc[page_idx]
        raw_text = page.get_text("text") or ""
        cleaned_text = clean_page_text(raw_text)
        
        if not cleaned_text:
            continue
            
        if page_idx >= 3:
            match = REF_HEADER_PATTERN.search(cleaned_text)
            if match:
                cleaned_text = cleaned_text[:match.start()].strip()
                ref_found = True
                
        if cleaned_text:
            extracted_pages.append((page_idx + 1, cleaned_text))
            
    doc.close()
    return extracted_pages


def discover_pdf_files(papers_dir: Path) -> List[Tuple[Path, str, str]]:
    """
    Walk /Papers directory and return list of tuples: (pdf_path, domain, subtopic).
    """
    pdf_entries = []
    if not papers_dir.exists():
        return pdf_entries

    for root, _, files in os.walk(papers_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                full_path = Path(root) / file
                rel_parts = full_path.relative_to(papers_dir).parts
                if len(rel_parts) >= 2:
                    domain = rel_parts[0].lower()
                    subtopic = rel_parts[1].lower()
                    pdf_entries.append((full_path, domain, subtopic))
    return pdf_entries


def run_ingestion() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    print("=" * 70)
    print("        Local RAG Research Assistant - Vector Ingestion Pipeline")
    print("=" * 70)
    print(f"Embedding Model      : {EMBEDDING_MODEL}")
    print(f"ChromaDB Storage Dir : {CHROMA_DB_DIR.resolve()}")
    print(f"Target Collection    : {COLLECTION_NAME}")
    print("=" * 70)

    # 1. Discover PDFs
    pdf_entries = discover_pdf_files(PAPERS_DIR)
    total_files = len(pdf_entries)
    print(f"\nDiscovered {total_files} PDF papers under '{PAPERS_DIR.resolve()}'.")
    if total_files == 0:
        print("No PDF files found to ingest. Exiting.")
        return

    # 2. Load Manifests & Mappings
    title_map = load_title_mapping(METADATA_MANIFEST)
    ingested_set = load_ingested_files(INGESTED_MANIFEST)
    print(f"Loaded title mapping for {len(title_map)} papers.")
    print(f"Already ingested files in manifest: {len(ingested_set)}")

    # 3. Initialize ChromaDB Client & Collection
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 4. Load SentenceTransformer Embedding Model
    print(f"\nLoading embedding model '{EMBEDDING_MODEL}' into memory...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model loaded successfully.")

    # 5. Initialize Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # 6. Process PDF Papers
    processed_count = 0
    skipped_count = 0
    total_chunks_created = 0
    failed_files: List[Tuple[str, str]] = []
    
    start_time = time.time()

    for idx, (pdf_path, domain, subtopic) in enumerate(pdf_entries, 1):
        filename = pdf_path.name

        if filename in ingested_set:
            skipped_count += 1
            print(f"[{idx}/{total_files}] SKIPPED (already ingested): {filename}")
            continue

        paper_title = title_map.get(filename, pdf_path.stem)

        try:
            pages = extract_pages_from_pdf(pdf_path)
            if not pages:
                print(f"[{idx}/{total_files}] Warning: No text extracted from {filename}")
                continue

            file_chunks: List[str] = []
            file_metadatas: List[Dict[str, Any]] = []
            file_ids: List[str] = []

            for page_num, page_text in pages:
                page_chunks = text_splitter.split_text(page_text)
                for chunk_idx, chunk in enumerate(page_chunks):
                    if not chunk.strip():
                        continue
                    
                    chunk_id = f"{filename}_p{page_num}_c{chunk_idx}_{hash(chunk) & 0xFFFFFFFF}"
                    file_chunks.append(chunk)
                    file_metadatas.append({
                        "domain": domain,
                        "subtopic": subtopic,
                        "title": paper_title,
                        "source_file": filename,
                        "page": page_num
                    })
                    file_ids.append(chunk_id)

            if not file_chunks:
                print(f"[{idx}/{total_files}] Warning: 0 chunks created for {filename}")
                continue

            embeddings = embedder.encode(file_chunks, show_progress_bar=False, normalize_embeddings=True)
            
            for b_start in range(0, len(file_chunks), BATCH_SIZE):
                b_end = b_start + BATCH_SIZE
                collection.upsert(
                    ids=file_ids[b_start:b_end],
                    embeddings=embeddings[b_start:b_end].tolist(),
                    documents=file_chunks[b_start:b_end],
                    metadatas=file_metadatas[b_start:b_end]
                )

            ingested_set.add(filename)
            save_ingested_files(INGESTED_MANIFEST, ingested_set)

            processed_count += 1
            total_chunks_created += len(file_chunks)

            print(f"[{idx}/{total_files}] Ingested: {filename} -> {len(file_chunks)} chunks ({domain}/{subtopic})")

        except Exception as e:
            print(f"[{idx}/{total_files}] ERROR processing {filename}: {e}")
            failed_files.append((filename, str(e)))

    elapsed = time.time() - start_time
    final_collection_count = collection.count()

    print("\n" + "=" * 70)
    print("                    INGESTION SUMMARY")
    print("=" * 70)
    print(f"Total PDFs Discovered   : {total_files}")
    print(f"Files Processed (New)   : {processed_count}")
    print(f"Files Skipped           : {skipped_count}")
    print(f"Files Failed            : {len(failed_files)}")
    print(f"Chunks Created This Run : {total_chunks_created}")
    print(f"Total Chunks in Chroma  : {final_collection_count}")
    print(f"ChromaDB Location       : {CHROMA_DB_DIR.resolve()}")
    print(f"Ingestion Time Elapsed  : {elapsed:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    run_ingestion()
