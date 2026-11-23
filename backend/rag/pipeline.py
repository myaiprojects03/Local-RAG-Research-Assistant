"""
RAG answer generation for the API. Mirrors modules/generation/rag_pipeline.py
(same system prompt, history window, model and parameters) without importing it,
because that module depends on ChromaDB/sentence-transformers.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from . import config
from .search import retrieve_chunks

logger = logging.getLogger("research-api")

NO_RESULTS_ANSWER ="I couldn't find relevant information in the research paper corpus to answer this question."

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

# The React frontend renders answers as plain text, and shows citations in its own
# Sources panel, so ask for unformatted prose and clean up anything that slips through.
FORMAT_INSTRUCTIONS = (
    "\n\nOUTPUT FORMAT:\n"
    "- Write in plain prose paragraphs. Do NOT use Markdown (no **bold**, headings, bullet or numbered lists, tables).\n"
    "- Do NOT mention or cite snippet numbers (e.g. 'Snippet 3'); sources are displayed separately to the user."
)

_CITATION_MARKERS = re.compile(
    r"\s*(?:【[^】]*】|\[(?:Context\s+)?Snippet[^\]]*\]|\((?:Source:\s*)?(?:Context\s+)?Snippets?\s*#?[\d,\s#and]+\))",
    re.IGNORECASE,
)


def _clean_answer(text: str) -> str:
    text = _CITATION_MARKERS.sub("", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)          # bold
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)      # headings
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)           # bullets
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class ConfigurationError(RuntimeError):
    """Server is missing required configuration (e.g. GROQ_API_KEY)."""


class UpstreamError(RuntimeError):
    """The LLM provider call failed."""


class RateLimitedError(UpstreamError):
    """Every configured Groq model is currently rate limited."""


_groq_client = None


def _client():
    global _groq_client
    if _groq_client is None:
        api_key = config.groq_api_key()
        if not api_key:
            raise ConfigurationError(
                "GROQ_API_KEY is not configured on the server. Add it to the backend's environment variables."
            )
        import groq

        # No SDK auto-retries: on a 429 the SDK would silently sleep for the retry-after period
        # (often 10-40 s). Falling back to another model is faster; see _complete_with_fallback.
        _groq_client = groq.Groq(api_key=api_key, timeout=config.GROQ_TIMEOUT_SECONDS, max_retries=0)
    return _groq_client


REWRITE_PROMPT = (
    "You rewrite the user's latest message into a standalone search query for a search engine over "
    "Cybersecurity and AI/ML research papers. Use the conversation to resolve references such as "
    "'it', 'this approach' or 'those attacks' into the concrete topic. If the message is already "
    "self-contained, return it unchanged. Do not add filler such as 'research papers' or 'definition'. "
    "Do not answer the question. Output only the query text."
)


def _search_query(user_question: str, chat_history: List[Dict[str, str]]) -> str:
    """Turn a follow-up into a standalone retrieval query; falls back to the raw question."""
    if not chat_history:
        return user_question
    turns = []
    for msg in chat_history[-4:]:
        content = msg["content"] if msg["role"] == "user" else msg["content"][:600]
        turns.append(f"{msg['role'].upper()}: {content}")
    params = {
        "model": config.GROQ_REWRITE_MODEL,
        "messages": [
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": "Conversation:\n" + "\n".join(turns) + f"\n\nLatest message: {user_question}"},
        ],
        "temperature": 0,
        "max_completion_tokens": 400,
        "timeout": config.GROQ_REWRITE_TIMEOUT_SECONDS,
    }
    if config.is_reasoning_model(config.GROQ_REWRITE_MODEL):
        params["reasoning_effort"] = "low"
        params["include_reasoning"] = False
    try:
        response = _client().chat.completions.create(**params)
        query = (response.choices[0].message.content or "").strip().strip('"').strip()
    except ConfigurationError:
        raise
    except Exception:
        return user_question  # retrieval still works on the raw question
    return query if 0 < len(query) <= 500 else user_question


def generate_answer(
    user_question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    domain_filter: Optional[str] = None,
) -> Dict[str, Any]:
    chat_history = [
        m for m in (chat_history or []) if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    retrieved = retrieve_chunks(_search_query(user_question, chat_history), domain_filter=domain_filter)
    if not retrieved:
        return {"answer": NO_RESULTS_ANSWER, "sources": []}

    context_blocks, sources, seen = [], [], set()
    for idx, item in enumerate(retrieved, 1):
        c = item["chunk"]
        context_blocks.append(
            f"--- Context Snippet #{idx} (Paper: '{c['title']}', Page {c['page']}) ---\n{c['text'].strip()}"
        )
        key = (c["title"], c["page"], c["source_file"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "title": c["title"],
                "page": c["page"],
                "source_file": c["source_file"],
                "source_url": c["source_url"],
                "domain": c["domain"],
                "subtopic": c["subtopic"],
            })

    messages = [{"role": "system", "content": SYSTEM_PROMPT + FORMAT_INSTRUCTIONS}]
    for msg in (chat_history or [])[-config.HISTORY_TURNS:]:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    context_str = "\n\n".join(context_blocks)
    messages.append({
        "role": "user",
        "content": f"Context from Research Papers:\n{context_str}\n\nUser Question: {user_question}",
    })

    response = _complete_with_fallback(messages)
    answer = _clean_answer(response.choices[0].message.content or "") or "No response returned."
    return {"answer": answer, "sources": sources}


def _complete_with_fallback(messages: List[Dict[str, str]]):
    """
    Try GROQ_MODEL, then each GROQ_FALLBACK_MODELS entry. Each Groq model has its own
    rate-limit bucket (free tier: 8,000 tokens/minute), so falling back on a 429 or a
    server/connection error keeps the app answering instead of failing.
    """
    import groq

    rate_limited = False
    last_error: Optional[Exception] = None
    for model in [config.GROQ_MODEL, *config.GROQ_FALLBACK_MODELS]:
        params = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_completion_tokens": config.MAX_COMPLETION_TOKENS,
        }
        if config.is_reasoning_model(model):
            params["reasoning_effort"] = config.GROQ_REASONING_EFFORT
            params["include_reasoning"] = False  # return only the final answer text
        try:
            return _client().chat.completions.create(**params)
        except groq.RateLimitError as exc:
            logger.warning("Groq model %s rate limited; trying fallback", model)
            rate_limited, last_error = True, exc
        except (groq.APIConnectionError, groq.InternalServerError, groq.NotFoundError) as exc:
            # NotFoundError: model retired/renamed (as happened to llama-3.3-70b-versatile).
            logger.warning("Groq model %s unavailable (%s); trying fallback", model, type(exc).__name__)
            last_error = exc
        except groq.APIStatusError as exc:  # 4xx other than 429: a real request/config problem
            raise UpstreamError(f"The language model rejected the request (HTTP {exc.status_code}).") from exc

    if rate_limited:
        raise RateLimitedError(
            "The AI service is busy right now (rate limit reached). Please wait a few seconds and try again."
        ) from last_error
    raise UpstreamError("The language model service could not be reached. Please try again shortly.") from last_error
