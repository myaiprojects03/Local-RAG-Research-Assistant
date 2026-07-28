# 📚 Local RAG Research Assistant

An automated, local Retrieval-Augmented Generation (RAG) scientific research assistant built in Python. It indexes and searches over 300+ arXiv research papers across **Cybersecurity** and **AI/ML**, using **ChromaDB**, **BAAI/bge-base-en-v1.5** embeddings, and Groq's **LLaMA-3.3-70B** for precise, grounded answers with page-level citations.

---

## 🚀 Live Demo & Deployment

- **Live Application URL**: `[Insert your Streamlit Cloud URL here after deploying]`
- **Deployment Platform**: [Streamlit Community Cloud](https://share.streamlit.io/)

---

## 💡 Key Features

- **Automated arXiv Collection**: Scrapes and catalogs academic papers across 8 domain subtopics (Malware Analysis, Zero Trust, Cloud Security, Threat Intelligence, LLMs, Adversarial ML, Deep Learning, AI Safety).
- **Page-Level Vector Ingestion**: Extracts text page-by-page using PyMuPDF (`fitz`), strips noisy references, chunks text (~600 tokens), and attaches rich metadata (`domain`, `subtopic`, `title`, `source_file`, `page`).
- **Domain-Filtered Semantic Retrieval**: Performs cosine similarity search using `BAAI/bge-base-en-v1.5` embeddings with a strict **distance threshold (0.40)** to filter out irrelevant matches.
- **Hallucination-Free Generation**: Powered by Groq's `llama-3.3-70b-versatile` model. Skips LLM calls entirely if context distance exceeds threshold, and provides direct factual answers without meta-talk.
- **Direct arXiv Paper Verification**: Generates clickable `https://arxiv.org/abs/...` links for every cited paper so users can verify sources online.

---

## 🏗️ Project Architecture

```text
Local RAG Research Assistant/
├── .env                              # Local secrets (not committed to Git)
├── .env.example                      # Environment variables template
├── .gitignore                        # Git exclusion rules
├── .gitattributes                    # Git LFS tracking rules for chroma.sqlite3
├── requirements.txt                  # Pinned Python dependencies
├── app.py                            # Streamlit web UI entrypoint
├── README.md                         # Project documentation
├── .streamlit/
│   ├── config.toml                   # Streamlit UI & server configuration
│   └── secrets.toml.example          # Cloud secrets configuration template
├── modules/
│   ├── collector/                    # arXiv paper scraping & cataloging
│   ├── ingestion/                    # PDF parsing, text cleaning & vector persistence
│   ├── retrieval/                    # Semantic vector search & distance thresholding
│   └── generation/                   # Prompt construction, Groq LLM API & citations
├── Papers/                           # Raw research PDFs dataset (excluded from Git)
└── chroma_db/                        # Persistent local ChromaDB vector store (9,956 chunks)
```

---

## ⚙️ Local Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd "Local RAG Research Assistant"
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Your Groq API Key**:
   Copy `.env.example` to `.env` and add your API key:
   ```env
   GROQ_API_KEY=gsk_your_groq_api_key_here
   ```

4. **Run the Streamlit Web Application**:
   ```bash
   python -m streamlit run app.py
   ```

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

To deploy this app on **share.streamlit.io**:

1. **Push your repository to GitHub** (make sure Git LFS is enabled for `chroma_db/chroma.sqlite3`).
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/) and click **New App**.
3. Select your repository, set **Main file path** to `app.py`, and click **Advanced settings**.
4. Under **Secrets**, paste your Groq API key in TOML format:
   ```toml
   GROQ_API_KEY = "gsk_your_groq_api_key_here"
   ```
5. Click **Deploy!**
