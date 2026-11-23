"""
FastAPI backend for the Scientific Paper Research Assistant frontend.

Entrypoint for Vercel (api/index.py exposes `app`) and for local development:

    cd backend
    uvicorn api.index:app --reload --port 8000

Endpoints
    POST /api/chat     {question, domain_filter, chat_history} -> {answer, sources}
    GET  /api/health   readiness/status info
"""

import logging
import sys
from pathlib import Path
from typing import List, Literal, Optional

# Make the sibling `rag` package importable regardless of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from rag import config, embedder, search  # noqa: E402
from rag.pipeline import ConfigurationError, RateLimitedError, UpstreamError, generate_answer  # noqa: E402

logger = logging.getLogger("research-api")

app = FastAPI(
    title="Scientific Paper Research Assistant API",
    version="1.0.0",
    description="RAG over 300+ arXiv papers in Cybersecurity and AI/ML.",
)

allow_all = "*" in config.ALLOWED_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all else config.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError):
    # The frontend shows `detail` as text, so flatten FastAPI's default list-of-errors into one string.
    messages = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err.get("loc", []) if p != "body")
        messages.append(f"{field}: {err.get('msg')}" if field else str(err.get("msg")))
    return JSONResponse(status_code=422, content={"detail": "Invalid request - " + "; ".join(messages)})


class ChatTurn(BaseModel):
    role: str
    content: str = ""


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    domain_filter: Literal["both", "cybersecurity", "ai_ml"] = "both"
    chat_history: List[ChatTurn] = Field(default_factory=list)


class Source(BaseModel):
    title: str
    page: Optional[int | str] = None
    source_file: str = ""
    source_url: str = ""
    domain: str = ""
    subtopic: str = ""


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]


@app.get("/")
@app.get("/api")
def root():
    return {"service": app.title, "chat_endpoint": "/api/chat", "health": "/api/health", "docs": "/docs"}


@app.get("/api/health")
def health():
    try:
        chunks = search.index_size()
        index_ok = True
    except Exception as exc:  # pragma: no cover - surfaced to operators
        chunks, index_ok = 0, False
        index_error = str(exc)
    body = {
        "status": "ok" if index_ok and config.groq_api_key() else "degraded",
        "index_loaded": index_ok,
        "chunks": chunks,
        "embedding_model_loaded": embedder.is_loaded(),
        "groq_configured": bool(config.groq_api_key()),
        "groq_model": config.GROQ_MODEL,
    }
    if not index_ok:
        body["index_error"] = index_error
    return body


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Please enter a question.")

    history = [t.model_dump() for t in request.chat_history if t.role in ("user", "assistant") and t.content]
    try:
        # CPU-bound embedding + blocking Groq call: keep the event loop free.
        return await run_in_threadpool(generate_answer, question, history, request.domain_filter)
    except ConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except RateLimitedError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "10"})
    except UpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception:
        logger.exception("Unhandled error in /api/chat")
        raise HTTPException(status_code=500, detail="Unexpected server error. Please try again.")
