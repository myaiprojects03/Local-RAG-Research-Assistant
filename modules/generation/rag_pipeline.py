"""
rag_pipeline.py

Phase 3: RAG & Generation Module.
Combines vector retrieval with Groq LLaMA-3.3-70B API generation to answer
research questions strictly grounded in retrieved paper context with citations.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import groq
from modules.retrieval.retriever import retrieve_chunks

# Load environment variables from .env
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# System instructions enforcing strict grounding and explicit refusal rules
SYSTEM_PROMPT = (
    "You are an expert scientific research assistant specializing in Cybersecurity and AI/ML.\n\n"
    "STRICT INSTRUCTIONS:\n"
    "1. Answer the user's question using ONLY the factual information in the provided context snippets below.\n"
    "2. If the provided context is insufficient, vague, or irrelevant to answer the question, state simply:\n"
    "   'I couldn't find relevant information in the research paper corpus to answer this question.'\n"
    "3. NEVER use outside or general knowledge not contained within the provided context snippets.\n"
    "4. NEVER use hedging phrases such as 'I believe', 'generally known', 'typically', 'in general', "
    "   or 'commonly understood'. State ONLY facts explicitly present in the provided context.\n"
    "5. Provide direct, objective, and concise answers."
)


def _get_groq_api_key() -> str:
    """Retrieve GROQ_API_KEY from Streamlit secrets (Cloud) or environment (Local)."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            val = st.secrets["GROQ_API_KEY"]
            if val:
                return str(val).strip()
    except Exception:
        pass
        
    return (os.getenv("GROQ_API_KEY") or "").strip()


def _init_groq_client() -> groq.Groq:
    """Initialize Groq client using secrets or environment API key."""
    api_key = _get_groq_api_key()
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY is not set or contains default placeholder. "
            "Please update your Streamlit Secrets or .env file with a valid Groq API key."
        )
    return groq.Groq(api_key=api_key)


def _load_url_map() -> Dict[str, str]:
    """Load pdf_filename -> source_url mapping from Papers/metadata.json."""
    url_map = {}
    meta_path = PROJECT_ROOT / "Papers" / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                records = json.load(f)
                for r in records:
                    pdf_filename = r.get("pdf_filename")
                    url = r.get("source_url")
                    if pdf_filename and url:
                        url_map[pdf_filename] = url
        except Exception:
            pass
    return url_map


def generate_answer(
    user_question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    domain_filter: Optional[str] = None,
    n_results: int = 5
) -> Dict[str, Any]:
    """
    Execute full RAG workflow: Retrieve relevant chunks -> Construct prompt -> Call Groq API -> Return answer & citations.
    """
    if chat_history is None:
        chat_history = []

    url_map = _load_url_map()

    # 1. Retrieve relevant paper chunks (filtered by distance threshold)
    retrieved_chunks = retrieve_chunks(
        query=user_question,
        domain_filter=domain_filter,
        n_results=n_results
    )

    # 2. Skip Groq API call entirely if retrieve_chunks returns no chunks
    if not retrieved_chunks:
        return {
            "answer": "I couldn't find relevant information in the research paper corpus to answer this question.",
            "sources": [],
            "chunks": []
        }

    # 2. Build context string from retrieved chunks
    context_blocks = []
    sources = []
    seen_sources = set()

    for idx, item in enumerate(retrieved_chunks, 1):
        text = item.get("text", "").strip()
        meta = item.get("metadata", {})
        title = meta.get("title", "Untitled Paper")
        page = meta.get("page", "?")
        source_file = meta.get("source_file", "")
        domain = meta.get("domain", "")
        subtopic = meta.get("subtopic", "")

        # Lookup arXiv URL or construct fallback link
        source_url = url_map.get(source_file, "")
        if not source_url and source_file:
            arxiv_id_match = re.match(r'^([a-zA-Z0-9\.\-]+?)_', source_file)
            if arxiv_id_match:
                raw_id = arxiv_id_match.group(1)
                if len(raw_id) >= 9 and raw_id[:4].isdigit():
                    formatted_id = f"{raw_id[:4]}.{raw_id[4:]}"
                else:
                    formatted_id = raw_id
                source_url = f"https://arxiv.org/abs/{formatted_id}"

        context_blocks.append(
            f"--- Context Snippet #{idx} (Paper: '{title}', Page {page}) ---\n{text}"
        )

        source_key = (title, page, source_file)
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "title": title,
                "page": page,
                "source_file": source_file,
                "source_url": source_url,
                "domain": domain,
                "subtopic": subtopic
            })

    context_str = "\n\n".join(context_blocks)

    # 3. Construct messages payload for Groq API
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
    for msg in recent_history:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    user_prompt = (
        f"Context from Research Papers:\n{context_str}\n\n"
        f"User Question: {user_question}"
    )
    messages.append({"role": "user", "content": user_prompt})

    # 4. Invoke Groq API
    client = _init_groq_client()
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1024
        )
        answer_text = response.choices[0].message.content or "No response returned."
    except Exception as e:
        answer_text = f"An error occurred while calling the Groq API: {str(e)}"

    return {
        "answer": answer_text,
        "sources": sources,
        "chunks": retrieved_chunks
    }


if __name__ == "__main__":
    print("Testing modules.generation.rag_pipeline standalone...")
    print("Groq Model configured:", GROQ_MODEL)
