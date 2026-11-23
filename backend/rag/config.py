"""Runtime configuration for the FastAPI backend (env vars with sensible defaults)."""

import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

# Local development: read backend/.env first, then the project-level .env used by Streamlit.
# On Vercel neither file exists and values come from the project's Environment Variables.
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")

IS_VERCEL = bool(os.getenv("VERCEL"))

DATA_DIR = BACKEND_DIR / "data"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
CHUNKS_PATH = DATA_DIR / "chunks.json.gz"
MANIFEST_PATH = DATA_DIR / "manifest.json"

# --- Embeddings (must match the model used at ingestion: BAAI/bge-base-en-v1.5) ---
# ONNX export of bge-base-en-v1.5 (fp16 weights, identical outputs to sentence-transformers).
# Pinned to a commit so deployments are reproducible.
EMBED_MODEL_REPO = os.getenv("EMBED_MODEL_REPO", "Xenova/bge-base-en-v1.5")
EMBED_MODEL_REVISION = os.getenv("EMBED_MODEL_REVISION", "4d6cd88e18e51a5e020c2c305726d76ada9c03cf")
EMBED_MODEL_FILE = os.getenv("EMBED_MODEL_FILE", "onnx/model_fp16.onnx")
EMBED_MAX_TOKENS = 512
# Vercel functions can only write to /tmp; locally we cache inside backend/.model_cache.
MODEL_CACHE_DIR = Path(
    os.getenv("MODEL_CACHE_DIR") or ("/tmp/bge-base-en-v1.5" if IS_VERCEL else BACKEND_DIR / ".model_cache")
)

# --- Retrieval (same values as modules/retrieval/retriever.py) ---
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.40"))
N_RESULTS = int(os.getenv("N_RESULTS", "5"))

# --- Generation (mirrors modules/generation/rag_pipeline.py) ---
# Groq has retired llama-3.3-70b-versatile (the model the Streamlit app used);
# openai/gpt-oss-120b is Groq's current large general-purpose model. Override with GROQ_MODEL.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Used in order when GROQ_MODEL is rate limited or unavailable (comma-separated; empty to disable).
GROQ_FALLBACK_MODELS = [
    m.strip() for m in os.getenv("GROQ_FALLBACK_MODELS", "openai/gpt-oss-20b").split(",") if m.strip()
]
# Reasoning models (gpt-oss, qwen3) spend tokens thinking before answering, so they get a
# larger budget and low reasoning effort to keep latency close to the original setup.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "2048"))
HISTORY_TURNS = 6
# Follow-ups like "what are its limitations?" are rewritten into a standalone search query
# (using the conversation) before retrieval; a small fast model is enough for that.
GROQ_REWRITE_MODEL = os.getenv("GROQ_REWRITE_MODEL", "openai/gpt-oss-20b")
# Keep the worst case under the Vercel function limit (maxDuration 60s in vercel.json) so a slow
# Groq response becomes a clean JSON error instead of a platform timeout:
# rewrite (8s) + primary model (20s) + one fallback model (20s) + cold start (~5-10s) < 60s.
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "20"))
GROQ_REWRITE_TIMEOUT_SECONDS = float(os.getenv("GROQ_REWRITE_TIMEOUT_SECONDS", "8"))


def is_reasoning_model(model: str) -> bool:
    return model.startswith(("openai/gpt-oss", "qwen/"))

# --- CORS: comma-separated list of allowed frontend origins, or "*" ---
ALLOWED_ORIGINS = [o.strip().rstrip("/") for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]


def groq_api_key() -> str:
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    return "" if key == "your_groq_api_key_here" else key
