# Secure Hybrid RAG Agent

A functional prototype of a RAG (Retrieval-Augmented Generation) system with a ReAct agent, deployable on **Streamlit Community Cloud** (free tier). Designed for single-user experimentation and small-scale demos, not production workloads.

## Features

- **Hybrid Search** - Combines BM25 keyword search with FAISS semantic vector search using weighted score fusion.
- **ReAct Agent** - Manual ReAct loop (no langchain.agents) that reasons about when to search the knowledge base before answering.
- **Image Analysis** - Attach images to questions; analyzed via Gemini's native vision API alongside retrieved text context. Note: image support bypasses the ReAct agent loop — tools are not available alongside visual queries.
- **Domain Configuration** - Switch the agent's persona between Financial, Healthcare, Legal, Technology, or General.
- **Document Upload** - Upload PDF or TXT files to build a custom knowledge base in real time.
- **Security Guardrails** - Dual-layer defense: pre-execution input sanitization + post-generation output validation.
- **API Key Protection** - Key loaded exclusively from Streamlit secrets (never hardcoded or exposed).

## Architecture

```
User Query
    |
    v
[Security Layer 1: Input Sanitization]
    |
    v
[Hybrid Search: BM25 + FAISS] --> Context
    |
    v
[ReAct Agent / Multimodal Gemini] --> Raw Response
    |
    v
[Security Layer 2: Output Validation] --> Final Answer
```

## Production Readiness

This codebase is a **functional prototype**, not a production system. The following gaps must be addressed before real-world use:

| Gap | Details |
|-----|---------|
| **Error hardening** | API failures return user-facing strings, not structured exceptions. Downstream callers cannot distinguish quota errors from network failures. |
| **Testing** | No unit, integration, or end-to-end tests exist. No CI/CD pipeline. |
| **Observability** | No metrics, no tracing, no health endpoint. Debugging relies entirely on Streamlit Cloud's raw stdout logs. |
| **Multimodal agent loop** | Image queries bypass the ReAct agent entirely — the agent cannot use tools (web search, KB search) alongside visual analysis in a single turn. |
| **Scalability** | All state is in-process memory (FAISS index, BM25 corpus, chat history). No database, no queue, no horizontal scaling. Designed for exactly one concurrent user. |
| **Secrets management** | Keys are loaded via `st.secrets` only. No fallback to environment variables or vaults for non-Streamlit deployments. |
| **Dependency locking** | `requirements.txt` uses loose upper bounds. No lockfile — builds are not reproducible. |

### Can Langfuse be added for observability within 1 GB RAM?

**Yes, with caveats.** Langfuse's Python SDK v3 (OpenTelemetry-based, GA since June 2025) batches traces in the background and can run within 1 GB if:

- **Reuse the client as a singleton** — creating a new Langfuse client per request creates background threads that accumulate and cause OOM (see [langfuse#3456](https://github.com/langfuse/langfuse/issues/3456))
- **Set `capture_input=False`** — the SDK famously leaked memory when storing large prompt payloads; see [langfuse#5032](https://github.com/langfuse/langfuse/issues/5032) (Django dev server: 600 MB → 4 GB in 1 minute)
- **Set `LANGFUSE_FLUSH_AT=32`** — lower batch size prevents in-memory span queue from growing too large
- **Use the cloud API** — self-hosting the Langfuse backend is not feasible on 1 GB

A simpler alternative with zero memory risk: **structured JSON logging to stdout** paired with Streamlit Cloud's built-in log viewer or a log shipping service.

## Quick Start (Local Testing)

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

Create a file at `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your_google_api_key_here"
```

> **Note:** This file is in `.gitignore` and will NOT be committed to your repo.

### 4. Run the app

```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Secure Hybrid RAG Agent"
git remote add origin https://github.com/YOUR_USERNAME/Secure-Hybrid-RAG.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Click **New App**.
3. Select your repository, branch (`main`), and main file (`app.py`).
4. Click **Deploy**.

### 3. Add your API key

1. In the Streamlit Cloud dashboard, go to your app.
2. Navigate to **Settings** > **Secrets**.
3. Add a new secret:
   ```
   GOOGLE_API_KEY = your_google_api_key_here
   ```
4. Click **Save**. The app will automatically redeploy.

> **Important:** Do NOT use Google Colab's `userdata.get()`. Streamlit Cloud uses its own secrets management system. The API key from Colab will NOT work here. You need to add it separately in Streamlit Cloud's Settings.

## Project Structure

```
Secure-Hybrid-RAG/
  app.py              Streamlit UI, session state, chat interface
  engine.py           HybridKnowledgeBase, ReAct agent, multimodal query, document processing
  security.py         Input sanitization, system prompts, output validation
  config.py           All tunable parameters (models, chunking, limits)
  requirements.txt    Python dependencies
  .gitignore          Files to exclude from version control
  .streamlit/
    config.toml       Streamlit server configuration
```

## How Hybrid Search Works

1. **BM25 (Keyword)** - Tokenizes the query and scores documents by term frequency/inverse document frequency. Good for exact term matches (product codes, names, acronyms).

2. **FAISS (Semantic)** - Embeds the query using `models/gemini-embedding-001` and finds nearest neighbors by L2 distance. Good for conceptual/paraphrased queries.

3. **Score Fusion** - Both score sets are min-max normalized to [0, 1], then combined with a weighted sum:
   ```
   final_score = alpha * bm25_score + (1 - alpha) * vector_score
   ```
   The `alpha` slider in the sidebar controls this balance.

## Memory Constraints (Streamlit Cloud Free Tier)

This app is designed to run within the ~1 GB RAM limit:

- **No local models** - All embeddings and LLM inference use Google's remote APIs.
- **No CLIP/sentence-transformers** - Multimodal queries use Gemini's native vision API.
- **In-memory only** - FAISS index and BM25 corpus are stored in Streamlit session state.
- **Warning:** The knowledge base resets when the app restarts or the session expires. You will need to re-upload your documents. This is a known limitation of Streamlit Cloud's free tier.

## Security

| Layer | Mechanism | Purpose |
|-------|-----------|---------|
| Input | Regex-based pattern matching | Block prompt injection and harmful queries |
| System Prompt | Hardcoded guardrails | Prevent system prompt disclosure and hallucination |
| Output | Leakage detection + formatting cleanup | Catch edge cases where guardrails are bypassed |

## License

This project is provided as-is for educational and research purposes.