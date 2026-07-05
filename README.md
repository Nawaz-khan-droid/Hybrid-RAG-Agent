# Secure Hybrid RAG Agent

A production-ready, multimodal RAG (Retrieval-Augmented Generation) system with a ReAct agent, deployable on **Streamlit Community Cloud** (free tier).

## Features

- **Hybrid Search** - Combines BM25 keyword search with FAISS semantic vector search using weighted score fusion.
- **ReAct Agent** - Manual ReAct loop (no langchain.agents) that reasons about when to search the knowledge base before answering.
- **Multimodal Queries** - Attach images to questions; analyzed via Gemini's native vision API alongside retrieved text context.
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