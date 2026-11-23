# FastAPI Backend (Vercel)

REST API that connects the React frontend (`scientific-research-assistant-main`) to the RAG pipeline.
It answers from the same ChromaDB chunks with the same embedding model (BAAI/bge-base-en-v1.5),
the same top-5 retrieval and the same 0.40 cosine-distance threshold as the Streamlit app.
The export keeps 9,797 of the 9,956 chunks: 159 near-empty chunks (under 150 characters, e.g. page headers
or title-only pages) are dropped, because they crowded real passages out of the top 5 without adding content.

## Why it is different from the Streamlit app

Vercel Python functions must be **≤ 250 MB unzipped**. ChromaDB + sentence-transformers + torch are well over 1 GB,
so the backend is built to be light:

| Streamlit app | This backend |
|---|---|
| ChromaDB persistent client | Index exported to `data/embeddings.npy` (float16) + `data/chunks.json.gz`, searched with NumPy |
| sentence-transformers + torch | ONNX Runtime + `tokenizers`, using the fp16 ONNX export of bge-base-en-v1.5 (identical embeddings) |
| ~1.5 GB of dependencies | ~196 MB total bundle (deps + data) |

Retrieval was verified against the original Chroma retriever before the near-empty chunks were dropped:
identical top-5 results for every test query (distance difference < 0.0001).

The ONNX model (~218 MB) is too large to bundle, so it downloads from Hugging Face (pinned revision) into `/tmp`
on the first request to a new instance. Loading takes ~1.4 s plus the download (a few seconds on Vercel's network;
~50 s on a 4 MB/s home connection). After that, embedding + search takes ~35 ms; total response time is mostly Groq (1–5 s).

**LLM model:** Groq has retired `llama-3.3-70b-versatile`, so the backend defaults to `openai/gpt-oss-120b`
(set `GROQ_MODEL` to change it). The Streamlit app still references the retired model and will fail until it is updated.

**Follow-up questions:** when there is chat history, a fast model (`openai/gpt-oss-20b`) first rewrites the latest
message into a standalone search query (e.g. "What attacks is it vulnerable to?" becomes
"What attacks is federated learning vulnerable to?"), so retrieval finds the topic of the conversation. If that
step fails, the raw question is used.

**Groq rate limits and fallback:** on Groq's free tier each model allows **8,000 tokens per minute**, and one
question uses about 2,500 tokens, i.e. roughly 3 questions per minute per model, shared by all users. When the main model
is rate limited, retired or unreachable, the backend retries once with each model in `GROQ_FALLBACK_MODELS`
(default `openai/gpt-oss-20b`, which has its own quota). If every model is limited, the API returns HTTP 429 with
a short "busy, try again" message. For real traffic, upgrade the Groq account (Dev tier) for higher limits.

## API

`POST /api/chat`

```json
{ "question": "Explain federated learning", "domain_filter": "both", "chat_history": [{ "role": "user", "content": "..." }] }
```

`domain_filter` is `both`, `cybersecurity` or `ai_ml`. Response:

```json
{ "answer": "...", "sources": [{ "title": "...", "page": 3, "source_file": "...", "source_url": "https://arxiv.org/abs/...", "domain": "ai_ml", "subtopic": "llms" }] }
```

Errors always return `{ "detail": "<readable string>" }`, which the frontend shows in its error banner:
422 = invalid request, 429 = all Groq models rate limited, 500 = server misconfiguration, 502 = Groq API failure.

`GET /api/health` shows index/model/Groq status. `GET /docs` is the interactive Swagger UI.

## Run locally

```powershell
cd backend
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m uvicorn api.index:app --reload --port 8000
```

The key is read from `backend/.env` or the project's root `.env` (`GROQ_API_KEY=...`).
The frontend's default `VITE_API_URL` is `http://localhost:8000`, so `npm run dev` in the frontend folder works with no extra setup.

## Deploy to Vercel

### 1. Backend

1. Push this repository to GitHub (the `backend/data/` files are regular files, not Git LFS).
2. In Vercel: **Add New → Project**, import the repo, and set **Root Directory** to `backend`.
   Framework Preset: **Other** (or FastAPI if Vercel detects it). Leave build/output settings empty.
3. **Environment Variables**:
   - `GROQ_API_KEY`: your Groq key (required)
   - `ALLOWED_ORIGINS`: your frontend URL, e.g. `https://your-frontend.vercel.app` (optional; defaults to `*`)
4. Deploy, then open `https://<backend>.vercel.app/api/health` and check `"status": "ok"`.
5. Run the end-to-end checks against the live backend (25 checks covering the exact contract the frontend uses):
   ```powershell
   python backend/scripts/smoke_test.py https://<backend>.vercel.app --origin https://<frontend>.vercel.app
   ```
   The first chat request after a deploy downloads the embedding model (cold start), so it is slower than the rest.

### 2. Frontend

1. Push the frontend folder to its own GitHub repository (it is not a Git repo yet), then import it into Vercel
   (Vite is detected automatically; build `npm run build`, output `dist`). No frontend code changes are needed.
2. Add the environment variable `VITE_API_URL=https://<backend>.vercel.app` (no trailing slash).
   Vite inlines it **at build time**, so redeploy the frontend after changing it.
3. Deploy. If you set `ALLOWED_ORIGINS` on the backend, it must include this frontend URL.

## Updating the paper corpus

After adding papers and re-running `modules/ingestion/ingest.py`, regenerate the backend index (needs `chromadb` installed locally):

```powershell
python backend/scripts/build_index.py
```

Then commit `backend/data/` and redeploy the backend.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | (required) | Groq API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq chat model |
| `GROQ_FALLBACK_MODELS` | `openai/gpt-oss-20b` | Comma-separated models tried when the main one is rate limited/unavailable |
| `GROQ_REWRITE_MODEL` | `openai/gpt-oss-20b` | Model that rewrites follow-ups into standalone search queries |
| `GROQ_TIMEOUT_SECONDS` | `20` | Per-call Groq timeout (keeps worst case under Vercel's 60 s limit) |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `DISTANCE_THRESHOLD` | `0.40` | Max cosine distance for a chunk to count as relevant |
| `N_RESULTS` | `5` | Chunks retrieved per question |
| `GROQ_REASONING_EFFORT` | `low` | Reasoning effort for gpt-oss/qwen models |
| `MAX_COMPLETION_TOKENS` | `2048` | Completion token budget |
