"""
paper_collector.py

Phase 1: Automated Paper Collector Module for Local RAG Research Assistant.
Fetches research papers from arXiv API across cybersecurity and AI/ML topics,
organizes PDFs into designated topic subfolders, maintains a metadata manifest
(metadata.json), deduplicates downloads, respects API rate limits, and handles errors.
"""

import os
import re
import json
import time
import urllib.request
import urllib.parse
import urllib.error
import socket
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional

# Define namespaces used in arXiv Atom XML response
NAMESPACES = {
    'atom': 'http://www.w3.org/2005/Atom',
    'arxiv': 'http://arxiv.org/schemas/atom'
}

# Subtopic definitions with domain, target query, and relative folder path
TOPIC_CONFIGS = [
    {
        "domain": "cybersecurity",
        "subtopic": "malware_analysis",
        "query": "cat:cs.CR AND all:(malware detection analysis)",
        "rel_path": os.path.join("CyberSecurity", "malware_analysis")
    },
    {
        "domain": "cybersecurity",
        "subtopic": "zero_trust",
        "query": "cat:cs.CR AND all:(zero trust architecture)",
        "rel_path": os.path.join("CyberSecurity", "zero_trust")
    },
    {
        "domain": "cybersecurity",
        "subtopic": "cloud_security",
        "query": "cat:cs.CR AND all:(cloud security)",
        "rel_path": os.path.join("CyberSecurity", "cloud_security")
    },
    {
        "domain": "cybersecurity",
        "subtopic": "threat_intel",
        "query": "cat:cs.CR AND all:(cyber threat intelligence)",
        "rel_path": os.path.join("CyberSecurity", "threat_intel")
    },
    {
        "domain": "ai_ml",
        "subtopic": "llms",
        "query": "cat:cs.CL AND all:(large language models)",
        "rel_path": os.path.join("AI_ML", "llms")
    },
    {
        "domain": "ai_ml",
        "subtopic": "adversarial_ml",
        "query": "(cat:cs.LG OR cat:cs.CR) AND all:(adversarial machine learning)",
        "rel_path": os.path.join("AI_ML", "adversarial_ml")
    },
    {
        "domain": "ai_ml",
        "subtopic": "deep_learning",
        "query": "cat:cs.LG AND all:(deep learning architecture)",
        "rel_path": os.path.join("AI_ML", "deep_learning")
    },
    {
        "domain": "ai_ml",
        "subtopic": "ai_safety",
        "query": "(cat:cs.AI OR cat:cs.CL) AND all:(AI safety alignment)",
        "rel_path": os.path.join("AI_ML", "ai_safety")
    }
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PAPERS_DIR = PROJECT_ROOT / "Papers"
METADATA_FILE = DEFAULT_PAPERS_DIR / "metadata.json"
REQUEST_DELAY = 3.0       # seconds delay between PDF downloads
SUBTOPIC_DELAY = 5.0      # seconds cooling pause between subtopic API searches


def sanitize_filename(name: str) -> str:
    """Sanitize string to be safe for filenames."""
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[-\s]+', '_', name).strip('_')
    return name[:100]  # Limit length for filesystem safety


def clean_arxiv_id(raw_id: str) -> str:
    """Extract clean arXiv ID from entry ID URL."""
    parts = raw_id.strip().split('/abs/')
    if len(parts) > 1:
        return parts[-1]
    return raw_id.strip().split('/')[-1]


def load_metadata(metadata_path: Path) -> List[Dict[str, Any]]:
    """Load existing metadata manifest if it exists."""
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load existing metadata file ({e}). Starting fresh.")
    return []


def save_metadata(metadata_path: Path, records: List[Dict[str, Any]]) -> None:
    """Save metadata records atomically to metadata.json."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = metadata_path.with_suffix('.tmp')
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    temp_path.replace(metadata_path)


def fetch_arxiv_papers(query: str, max_results: int = 50, max_retries: int = 4) -> List[Dict[str, Any]]:
    """
    Fetch papers matching query from arXiv API with robust retry and backoff handling.
    Returns list of paper dictionaries parsed from Atom XML.
    """
    params = {
        'search_query': query,
        'start': 0,
        'max_results': max_results,
        'sortBy': 'relevance',
        'sortOrder': 'descending'
    }
    url = f"https://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
    headers = {'User-Agent': 'LocalRAGResearchAssistant/1.0 (academic paper collector; mailto:research@example.com)'}
    
    xml_data = None
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=40) as response:
                xml_data = response.read()
                break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_time = attempt * 15
                print(f"    [Rate Limit 429] arXiv API requested cooling down. Waiting {wait_time}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"    [HTTP Error {e.code}] Fetching arXiv query '{query}': {e}")
                if attempt < max_retries:
                    time.sleep(attempt * 5)
        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            wait_time = attempt * 10
            print(f"    [Network Timeout] Fetching arXiv query '{query}': {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"    [Unexpected Error] Fetching arXiv query '{query}': {e}")
            if attempt < max_retries:
                time.sleep(5)
            
    if xml_data is None:
        print(f"    [Failed] Could not retrieve search results for query '{query}'.")
        return []
    
    root = ET.fromstring(xml_data)
    papers = []
    
    for entry in root.findall('atom:entry', NAMESPACES):
        id_elem = entry.find('atom:id', NAMESPACES)
        title_elem = entry.find('atom:title', NAMESPACES)
        published_elem = entry.find('atom:published', NAMESPACES)
        
        if id_elem is None or title_elem is None:
            continue
            
        raw_id = id_elem.text or ""
        arxiv_id = clean_arxiv_id(raw_id)
        
        title = " ".join((title_elem.text or "").split())  # Normalize whitespace
        published = (published_elem.text or "")[:10] if published_elem is not None else ""
        
        # Authors
        authors = []
        for author in entry.findall('atom:author', NAMESPACES):
            name_elem = author.find('atom:name', NAMESPACES)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())
                
        # Links (PDF & Abstract)
        pdf_url = ""
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        for link in entry.findall('atom:link', NAMESPACES):
            rel = link.attrib.get('rel')
            title_attr = link.attrib.get('title')
            href = link.attrib.get('href', '')
            if title_attr == 'pdf' or 'pdf' in href:
                pdf_url = href.replace('http://', 'https://')
                
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "published": published,
            "abs_url": abs_url,
            "pdf_url": pdf_url
        })
        
    return papers


def download_pdf(pdf_url: str, dest_path: Path, max_retries: int = 3) -> bool:
    """Download PDF file to target path with rate limit retry handling."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {'User-Agent': 'LocalRAGResearchAssistant/1.0'}
    
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(pdf_url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=45) as resp, open(dest_path, 'wb') as out_f:
                out_f.write(resp.read())
            return True
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries:
                wait_time = attempt * 15
                print(f"    [PDF Rate Limit 429] Cooling down {wait_time}s before retrying PDF download...")
                time.sleep(wait_time)
            else:
                print(f"    [Error] Failed to download {pdf_url}: {e}")
                break
        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            if attempt < max_retries:
                time.sleep(attempt * 5)
            else:
                print(f"    [Error] Timeout downloading {pdf_url}: {e}")
                break
        except Exception as e:
            print(f"    [Error] Failed to download {pdf_url}: {e}")
            break
            
    if dest_path.exists():
        try:
            dest_path.unlink()
        except OSError:
            pass
    return False


def run_paper_collector(
    target_per_subtopic: int = 40,
    papers_dir: Path = DEFAULT_PAPERS_DIR,
    metadata_path: Path = METADATA_FILE
) -> None:
    """Run full collection workflow across all topics."""
    print("=" * 70)
    print("        Local RAG Research Assistant - Paper Collector")
    print("=" * 70)
    print(f"Target count per subtopic : {target_per_subtopic}")
    print(f"Root Papers Directory     : {papers_dir.resolve()}")
    print(f"Metadata File             : {metadata_path.resolve()}")
    print("=" * 70)
    
    # Load metadata and build deduplication lookup sets
    metadata_records = load_metadata(metadata_path)
    existing_arxiv_ids = {r.get("arxiv_id") for r in metadata_records if r.get("arxiv_id")}
    existing_titles = {r.get("title").lower().strip() for r in metadata_records if r.get("title")}
    
    total_new_downloaded = 0

    for idx, topic_cfg in enumerate(TOPIC_CONFIGS, 1):
        domain = topic_cfg["domain"]
        subtopic = topic_cfg["subtopic"]
        query = topic_cfg["query"]
        target_dir = papers_dir / topic_cfg["rel_path"]
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Count existing papers already in this subtopic manifest
        existing_subtopic_count = sum(1 for r in metadata_records if r.get("subtopic") == subtopic)
        print(f"\n[Topic {idx}/{len(TOPIC_CONFIGS)}] {domain}/{subtopic} (Query: '{query}')")
        print(f" -> Existing in manifest: {existing_subtopic_count}/{target_per_subtopic}")
        
        needed = target_per_subtopic - existing_subtopic_count
        if needed <= 0:
            print(" -> Target already reached. Skipping download batch.")
            continue
            
        print(f" -> Fetching paper candidates from arXiv API...")
        time.sleep(SUBTOPIC_DELAY)  # Cooling pause before API call
        
        # Fetch candidate results
        candidates = fetch_arxiv_papers(query, max_results=needed + 25)
        print(f" -> Received {len(candidates)} candidate entries from arXiv API.")
        
        downloaded_in_topic = 0
        for paper in candidates:
            if downloaded_in_topic >= needed:
                break
                
            arxiv_id = paper["arxiv_id"]
            title = paper["title"]
            title_lower = title.lower().strip()
            
            # Skip duplicates
            if arxiv_id in existing_arxiv_ids or title_lower in existing_titles:
                continue
                
            # Create safe filename
            clean_title_part = sanitize_filename(title)
            safe_id_part = sanitize_filename(arxiv_id)
            pdf_filename = f"{safe_id_part}_{clean_title_part}.pdf"
            pdf_filepath = target_dir / pdf_filename
            
            print(f" -> [{existing_subtopic_count + downloaded_in_topic + 1}/{target_per_subtopic}] Downloading: '{title[:60]}...'")
            
            # Download PDF
            success = download_pdf(paper["pdf_url"], pdf_filepath)
            time.sleep(REQUEST_DELAY)  # Respect arXiv rate limit
            
            if success:
                record = {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "authors": paper["authors"],
                    "domain": domain,
                    "subtopic": subtopic,
                    "pdf_filename": pdf_filename,
                    "pdf_filepath": str(pdf_filepath.relative_to(papers_dir.parent)),
                    "source_url": paper["abs_url"],
                    "published_date": paper["published"]
                }
                metadata_records.append(record)
                existing_arxiv_ids.add(arxiv_id)
                existing_titles.add(title_lower)
                save_metadata(metadata_path, metadata_records)
                
                downloaded_in_topic += 1
                total_new_downloaded += 1
                
        print(f" -> Finished subtopic '{subtopic}'. Downloaded {downloaded_in_topic} new papers.")
        
    print("\n" + "=" * 70)
    print(f"Collection Complete! Total new papers downloaded: {total_new_downloaded}")
    print(f"Total entries in manifest ({metadata_path}): {len(metadata_records)}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automated arXiv Paper Collector for Local RAG")
    parser.add_argument("--target", type=int, default=40, help="Target papers per subtopic (default: 40)")
    parser.add_argument("--papers-dir", type=str, default=str(DEFAULT_PAPERS_DIR), help="Root folder for papers")
    args = parser.parse_args()

    run_paper_collector(
        target_per_subtopic=args.target,
        papers_dir=Path(args.papers_dir)
    )
