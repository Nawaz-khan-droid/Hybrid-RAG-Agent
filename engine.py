"""
Core RAG engine implementing:

  1. HybridKnowledgeBase  - Dual-index (FAISS vector + BM25 keyword)
                            with weighted score fusion and domain filtering.

  2. query_agent          - Manual ReAct loop with a tool registry and
                            dispatch table. Supports N tools (KB search,
                            web search, future tools) without importing
                            langchain.agents or langchain.chains.

  3. web_search           - Web search via Tavily (primary) or DDG
                            (fallback). Returns RAG-ready text snippets.

  4. fetch_url_content    - Fetches and extracts text content from URLs
                            via Tavily's extract API.

  5. multimodal_query     - Native Gemini vision API for image+text queries.

  6. Document processing  - PDF/TXT extraction and sentence-boundary-aware
                            chunking for knowledge base ingestion.

Design decisions for Streamlit Community Cloud (1 GB RAM):
  - All embeddings and LLM calls use remote Google APIs (zero local models).
  - FAISS index and BM25 corpus live in process memory only.
  - No local CLIP or sentence-transformers; multimodal inference is
    delegated entirely to Gemini's native vision capabilities.
  - The HybridKnowledgeBase is designed to be stored in st.session_state
    so it persists across Streamlit reruns and can be mutated in-place.
  - langchain.agents and langchain.chains are intentionally avoided to
    prevent pydantic crashes on Python 3.14+.
"""

import re
import json
import time
import logging
from typing import Optional, Callable

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from google.api_core.exceptions import ResourceExhausted

from security import get_system_prompt, validate_output, sanitize_search_query
from exceptions import APIError
from config import (
    EMBEDDING_MODEL,
    PRIMARY_LLM_MODEL,
    FALLBACK_LLM_MODEL,
    FALLBACK_EMBEDDING_MODEL,
    LLM_TEMPERATURE,
    MAX_AGENT_ITERATIONS,
    DEFAULT_TOP_K,
    DEFAULT_ALPHA,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_CHARS_PER_FILE,
    WEB_SEARCH_MAX_RESULTS,
    DEFAULT_WEB_PROVIDER,
    MAX_SEARCH_QUERY_LENGTH,
    URL_FETCH_TIMEOUT,
    URL_MAX_RESPONSE_BYTES,
    URL_MAX_REDIRECTS,
    MODE_KB_ONLY,
    MODE_WEB_ONLY,
    MODE_HYBRID,
    MODE_DIRECT,
    TEXT_MODEL_CANDIDATES,
    VISION_MODEL_CANDIDATES,
    OCR_DPI,
    is_vision_model,
)

logger = logging.getLogger(__name__)


# ── Structured JSON Logging ────────────────────────────────

class _JSONFormatter(logging.Formatter):
    """Formats log records as newline-delimited JSON for machine parsing."""
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "t": self.formatTime(record),
            "lvl": record.levelname,
            "mod": record.module,
            "msg": record.getMessage(),
        }
        for key in ("duration_ms", "api_service", "error_code", "is_rate_limit", "query_len", "chunk_count"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        return json.dumps(entry, default=str)

_handler = logging.StreamHandler()
_handler.setFormatter(_JSONFormatter())
logger.root.handlers.clear()
logger.root.addHandler(_handler)
logger.root.setLevel(logging.INFO)


# ── Lightweight In-Memory Metrics ──────────────────────────

_metrics: dict[str, list[float]] = {}

def _record_metric(name: str, value: float) -> None:
    _metrics.setdefault(name, []).append(value)
    # Keep only last 500 to bound memory
    if len(_metrics[name]) > 500:
        _metrics[name] = _metrics[name][-250:]

def get_metrics_snapshot() -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name, vals in _metrics.items():
        if vals:
            s_vals = sorted(vals)
            n = len(s_vals)
            summary[name] = {
                "count": n,
                "min": s_vals[0],
                "p50": s_vals[n // 2],
                "p95": s_vals[int(n * 0.95)],
                "max": s_vals[-1],
                "avg": sum(vals) / n,
            }
    return summary


# ── Tokenizer for BM25 ──────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """
    Simple word-level tokenizer for BM25 indexing.
    Splits on non-alphanumeric characters and lowercases.
    Chosen over NLTK/spaCy to keep the dependency footprint minimal.
    """
    return re.findall(r"\w+", text.lower())


# ── Retry helper for Gemini API calls ──────────────────────

def _is_rate_limit(exc: Exception) -> bool:
    """Check if an exception indicates a rate limit / quota error."""
    msg = str(exc).lower()
    if any(kw in msg for kw in ("429", "ratelimit", "rate limit", "resource exhausted", "quota")):
        return True
    if hasattr(exc, "status_code") and exc.status_code == 429:  # type: ignore[union-attr]
        return True
    if hasattr(exc, "code") and exc.code == 429:  # type: ignore[union-attr]
        return True
    return False


def _invoke_with_retry(callable_fn, max_attempts=3, service="gemini"):
    """
    Invokes a callable with exponential backoff retry on failure.

    Rate-limit errors (429 / ResourceExhausted) get longer backoff:
      5s, 10s, 20s — giving the quota window time to recover.
    All other transient errors get shorter backoff: 1s, 2s, 4s.

    After exhausting retries raises APIError so the caller can
    distinguish infrastructure failures from expected states
    (empty KB, blocked content).

    Usage:
        result = _invoke_with_retry(lambda: llm.invoke(prompt))
    """
    start = time.perf_counter()
    last_exc = None
    is_rate = False
    for attempt in range(max_attempts):
        try:
            result = callable_fn()
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "%s call succeeded in %.0fms (attempt %d)",
                service, duration_ms, attempt + 1,
                extra={"duration_ms": duration_ms, "api_service": service},
            )
            _record_metric(f"{service}_latency_ms", duration_ms)
            return result
        except Exception as e:
            last_exc = e
            is_rate = _is_rate_limit(e)
            delay = (2 ** attempt) * 5 if is_rate else (2 ** attempt)  # rate: 5,10,20s else 1,2,4s
            logger.warning(
                "%s call failed (attempt %d/%d, retry in %ds): %s",
                service, attempt + 1, max_attempts, delay, e,
                extra={"api_service": service, "is_rate_limit": is_rate},
            )
            if attempt < max_attempts - 1:
                time.sleep(delay)
    duration_ms = (time.perf_counter() - start) * 1000
    _record_metric(f"{service}_error_count", 1)
    _record_metric(f"{service}_rate_limited" if is_rate else f"{service}_failed", 1)
    raise APIError(
        message=f"{service} API unreachable after {max_attempts} attempts: {last_exc}",
        service=service,
        status_code=429 if is_rate else 0,
    ) from last_exc


# ═══════════════════════════════════════════════════════════════
#  Model Registry — test + cache model availability
# ═══════════════════════════════════════════════════════════════

# Cache: model_name -> status ("ok" | "rate_limited" | "unavailable")
_model_status: dict[str, str] = {}

def _test_model(model: str, api_key: str) -> str:
    """
    Make a single minimal API call to check if a model is usable.
    Caches the result so the model is only tested once per session.

    Returns one of: "ok", "rate_limited", "unavailable".
    """
    if model in _model_status:
        return _model_status[model]

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        _model_status[model] = "unavailable"
        return "unavailable"

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents="Reply OK",
            config=types.GenerateContentConfig(
                max_output_tokens=1,
                temperature=0.0,
            ),
        )
        _model_status[model] = "ok"
        return "ok"
    except Exception as exc:
        msg = str(exc).lower()
        if _is_rate_limit(exc):
            _model_status[model] = "rate_limited"
            logger.info("Model %s is rate-limited, caching as unavailable", model)
        else:
            _model_status[model] = "unavailable"
            logger.info("Model %s unavailable: %s", model, msg[:120])
        return _model_status[model]


def _first_working_model(
    candidates: list[str],
    api_key: str,
    needs_vision: bool = False,
) -> str | None:
    """
    Return the first model from *candidates* that:
      - (if needs_vision) is in VISION_CAPABLE_MODELS
      - passes the lightweight ping test (or was already proven ok)

    Returns None when all models are exhausted.
    """
    if needs_vision:
        candidates = [m for m in candidates if is_vision_model(m)]

    for model in candidates:
        status = _test_model(model, api_key)
        if status == "ok":
            return model
    return None

class HybridKnowledgeBase:
    """
    Manages a dual-index knowledge base that combines:
      - FAISS IndexFlatL2 for semantic (vector) similarity search
      - BM25Okapi for lexical (keyword) search

    Both indices are updated in-place when new documents are added,
    making this class safe for Streamlit session state storage.

    Typical lifecycle inside a Streamlit app:
        kb = HybridKnowledgeBase(api_key=API_KEY)
        st.session_state["kb"] = kb          # persist across reruns
        kb.add_documents(chunks, metas)       # mutate in-place
        results = kb.hybrid_search(query)     # query both indices
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key,
        )
        self.vector_index: Optional[faiss.Index] = None
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: list[str] = []
        self.metadata: list[dict] = []

    # ── Properties ───────────────────────────────────────────

    @property
    def is_empty(self) -> bool:
        """Returns True when no documents have been ingested."""
        return len(self.corpus) == 0

    @property
    def chunk_count(self) -> int:
        """Total number of text chunks currently indexed."""
        return len(self.corpus)

    @property
    def source_count(self) -> int:
        """Number of unique source files in the knowledge base."""
        sources = {m.get("source", "unknown") for m in self.metadata}
        return len(sources)

    @property
    def source_names(self) -> list[str]:
        """List of unique source file names."""
        return sorted({m.get("source", "unknown") for m in self.metadata})

    # ── Document Ingestion ───────────────────────────────────

    def add_documents(self, chunks: list[str], metas: list[dict]) -> int:
        """
        Ingests text chunks into both FAISS and BM25 indices.

        Args:
            chunks: List of text strings to index.
            metas: Parallel list of metadata dicts (one per chunk).

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        self.corpus.extend(chunks)
        self.metadata.extend(metas)

        # FAISS: embed and index directly (no langchain wrapper)
        try:
            vectors = np.array(
                self.embeddings.embed_documents(chunks), dtype=np.float32
            )
        except ResourceExhausted:
            logger.warning(
                "Embedding model %s quota exhausted, falling back to %s",
                EMBEDDING_MODEL, FALLBACK_EMBEDDING_MODEL,
            )
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=FALLBACK_EMBEDDING_MODEL,
                google_api_key=self.api_key,
            )
            vectors = np.array(
                self.embeddings.embed_documents(chunks), dtype=np.float32
            )
        if self.vector_index is None:
            self.vector_index = faiss.IndexFlatL2(vectors.shape[1])
        self.vector_index.add(vectors)

        # BM25: full rebuild (BM25Okapi lacks incremental update)
        tokenized_corpus = [_tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info("Added %d chunks. Total chunks: %d", len(chunks), self.chunk_count)
        return len(chunks)

    # ── Hybrid Search ────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        alpha: float = DEFAULT_ALPHA,
        domain: Optional[str] = None,
    ) -> str:
        """
        Combines BM25 keyword scores and FAISS vector similarity scores
        using weighted linear fusion, then returns the top-K results.

        Score normalization:
          - BM25 raw scores are min-max normalized to [0, 1].
          - FAISS L2 distances are converted to similarity via 1/(1+d)
            and then min-max normalized to [0, 1].

        Domain filtering:
          - If a domain is specified and it is not "Custom/General",
            documents whose metadata domain does not match receive
            a score of 0 in both indices.

        Args:
            query: The user's search query.
            top_k: Number of results to return.
            alpha: Weight for BM25. Vector weight is (1 - alpha).
            domain: Optional domain filter from metadata.

        Returns:
            Formatted string of retrieved context passages, or a
            message indicating the KB is empty or no results were found.
        """
        if self.is_empty:
            return "Knowledge base is empty. Please upload documents first."

        effective_k = min(top_k, len(self.corpus))
        n = len(self.corpus)

        # ── 1. BM25 Keyword Scores ───────────────────────────
        tokenized_query = _tokenize(query)
        bm25_raw = np.array(
            self.bm25.get_scores(tokenized_query), dtype=np.float64
        )
        bm25_max = bm25_raw.max()
        bm25_norm = bm25_raw / bm25_max if bm25_max > 0 else np.zeros(n)

        # ── 2. FAISS Vector Scores ───────────────────────────
        vector_norm = np.zeros(n, dtype=np.float64)

        if self.vector_index is not None:
            query_vec = np.array(
                self.embeddings.embed_query(query), dtype=np.float32
            ).reshape(1, -1)
            distances, indices = self.vector_index.search(query_vec, n)
            for i in range(n):
                idx = indices[0][i]
                if idx != -1:
                    similarity = 1.0 / (1.0 + float(distances[0][i]))
                    vector_norm[idx] = similarity

            v_max = vector_norm.max()
            if v_max > 0:
                vector_norm = vector_norm / v_max

        # ── 3. Optional Domain Filtering ─────────────────────
        if domain and domain != "Custom/General":
            for i, meta in enumerate(self.metadata):
                if meta.get("domain") != domain:
                    bm25_norm[i] = 0.0
                    vector_norm[i] = 0.0

        # ── 4. Weighted Fusion ───────────────────────────────
        combined = alpha * bm25_norm + (1.0 - alpha) * vector_norm

        # ── 5. Select Top-K (skip zero-score results) ────────
        top_indices = combined.argsort()[-effective_k:][::-1]

        results = []
        for i in top_indices:
            if combined[i] > 1e-9:
                source = self.metadata[i].get("source", "Unknown")
                results.append(f"[Source: {source}]\n{self.corpus[i]}")

        if not results:
            # Fallback: return most recent chunks so the agent has context
            fallback = []
            for i in range(min(3, len(self.corpus))):
                idx = len(self.corpus) - 1 - i
                source = self.metadata[idx].get("source", "Unknown")
                fallback.append(f"[Source: {source}]\n{self.corpus[idx]}")
            return "\n\n---\n\n".join(fallback)

        return "\n\n---\n\n".join(results)


# ═══════════════════════════════════════════════════════════════
#  Web Search (Tavily primary, DDG fallback)
# ═══════════════════════════════════════════════════════════════

def web_search(
    query: str,
    provider: str = DEFAULT_WEB_PROVIDER,
    tavily_api_key: Optional[str] = None,
    max_results: int = WEB_SEARCH_MAX_RESULTS,
) -> str:
    """
    Performs a web search and returns formatted results for RAG.

    Provider selection logic:
      - "tavily": Always use Tavily (requires API key).
      - "ddg": Always use DuckDuckGo (no API key, free, but rate-limited).
      - "auto": Try Tavily first (if API key is available),
                fall back to DDG on failure.

    Args:
        query: The search query string.
        provider: One of "tavily", "ddg", "auto".
        tavily_api_key: Tavily API key (from Streamlit secrets).
        max_results: Maximum number of results to return.

    Returns:
        Formatted string of search results, or an error message.
    """
    query = sanitize_search_query(query)
    if not query:
        return "Search query is empty after sanitization."

    if provider == "ddg":
        return _web_search_ddg(query, max_results)

    if provider == "tavily":
        if not tavily_api_key:
            return (
                "Tavily web search requires an API key. "
                "Add TAVILY_API_KEY to your Streamlit secrets, "
                "or switch to DuckDuckGo provider."
            )
        return _web_search_tavily(query, tavily_api_key, max_results)

    # "auto" mode: try Tavily first, fall back to DDG
    if tavily_api_key:
        result = _web_search_tavily(query, tavily_api_key, max_results)
        if not result.startswith("Web search failed"):
            return result
        logger.warning("Tavily search failed, falling back to DDG.")

    return _web_search_ddg(query, max_results)


def _web_search_tavily(
    query: str, api_key: str, max_results: int
) -> str:
    """
    Web search via Tavily API. Returns AI-extracted relevant content
    that is ready for direct injection into LLM context.

    Tavily's 'content' field is purpose-built for RAG: it uses
    proprietary extraction to return the most relevant portions
    of each page, not raw HTML or short snippets.
    """
    try:
        from tavily import TavilyClient
    except ImportError:
        return (
            "Web search failed: 'tavily-python' package is not installed. "
            "Falling back to DuckDuckGo is not available either."
        )

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )

        results = response.get("results", [])
        if not results:
            return f"No web search results found for: {query}"

        formatted = []
        for item in results:
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            content = item.get("content", "")
            score = item.get("score", 0)
            formatted.append(
                f"[Web Source: {title}]({url}) "
                f"[Relevance: {score:.0%}]\n{content}"
            )

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        logger.error("Tavily search failed: %s", e)
        return f"Web search failed: {e}"


def _web_search_ddg(query: str, max_results: int) -> str:
    """
    Web search via DuckDuckGo (ddgs package). Returns title,
    URL, and snippet for each result.

    DDG returns only short snippets (~150 chars). For full page
    content, use fetch_url_content() with each result URL.

    Note: DDG may rate-limit requests from cloud datacenter IPs.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return (
            "Web search failed: 'ddgs' package is not installed."
        )

    try:
        ddgs_client = DDGS()
        results = ddgs_client.text(query, max_results=max_results)

        if not results:
            return f"No web search results found for: {query}"

        formatted = []
        for item in results:
            title = item.get("title", "Untitled")
            href = item.get("href", "")
            body = item.get("body", "")
            formatted.append(
                f"[Web Source: {title}]({href})\n{body}"
            )

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        logger.error("DDG search failed: %s", e)
        return f"Web search failed: {e}"


# ═══════════════════════════════════════════════════════════════
#  URL Content Fetching (via Tavily extract)
# ═══════════════════════════════════════════════════════════════

def fetch_url_content(
    urls: list[str],
    tavily_api_key: Optional[str] = None,
) -> str:
    """
    Fetches and extracts text content from one or more URLs.

    Tries providers in order:
      1. Tavily Extract (server-side, best quality, needs API key)
      2. Jina Reader (free 1000 req/month, clean markdown, no key needed)
      3. Local HTTP fetcher with regex HTML stripping (last resort)

    Args:
        urls: List of URL strings to fetch.
        tavily_api_key: Tavily API key for the extract endpoint.

    Returns:
        Concatenated text content from all successfully fetched URLs,
        or an error message.
    """
    if not urls:
        return "No URLs provided for content fetching."

    if tavily_api_key:
        result = _fetch_urls_tavily(urls, tavily_api_key)
        if result:
            return result
        logger.warning("Tavily extract failed, trying Jina Reader.")

    result = _fetch_urls_jina(urls)
    if result:
        return result
    logger.warning("Jina Reader failed, falling back to local fetch.")

    return _fetch_urls_local(urls)


def _fetch_urls_tavily(urls: list[str], api_key: str) -> str:
    """
    Fetches URL content via Tavily's extract API.
    Tavily handles the HTTP request server-side (no SSRF risk to us).
    Returns empty string on failure (caller should fall back).
    """
    try:
        from tavily import TavilyClient
    except ImportError:
        return ""

    try:
        client = TavilyClient(api_key=api_key)
        response = client.extract(
            urls=urls,
            extract_depth="basic",
            format="text",
        )

        results = response.get("results", [])
        if not results:
            return ""

        formatted = []
        for item in results:
            url = item.get("url", "Unknown URL")
            content = item.get("raw_content", "")
            if content:
                # Truncate to prevent RAM exhaustion
                content = content[:MAX_CHARS_PER_FILE]
                formatted.append(f"[Fetched: {url}]\n{content}")

        if not formatted:
            return ""

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        logger.error("Tavily extract failed: %s", e)
        return ""


def _fetch_urls_jina(urls: list[str]) -> str:
    """
    Fetches URL content via Jina Reader (r.jina.ai).

    Free tier: 1000 requests/month, no API key required.
    Returns clean markdown content extracted by Jina's AI parser.
    Returns empty string on failure (caller should fall back).
    """
    import requests as _requests

    formatted = []
    for url in urls:
        try:
            resp = _requests.get(
                f"https://r.jina.ai/{url}",
                headers={
                    "Accept": "text/markdown",
                    "X-No-Cache": "true",
                },
                timeout=URL_FETCH_TIMEOUT,
            )
            if resp.status_code == 200 and resp.text.strip():
                content = resp.text.strip()[:MAX_CHARS_PER_FILE]
                formatted.append(f"[Fetched: {url}]\n{content}")
            else:
                logger.warning(
                    "Jina Reader returned status %d for %s",
                    resp.status_code, url,
                )
        except Exception as e:
            logger.error("Jina Reader failed for %s: %s", url, e)

    return "\n\n---\n\n".join(formatted) if formatted else ""


def _fetch_urls_local(urls: list[str]) -> str:
    """
    Fetches URL content using local HTTP requests (requests library).
    The caller MUST have already validated each URL via validate_url()
    to prevent SSRF attacks.

    This is a fallback when Tavily is unavailable.
    """
    import requests
    from html import unescape
    from requests.adapters import HTTPAdapter

    formatted = []
    session = requests.Session()
    _adapter = HTTPAdapter()
    _adapter.max_redirects = URL_MAX_REDIRECTS
    session.mount('http://', _adapter)
    session.mount('https://', _adapter)

    for url in urls:
        try:
            resp = session.get(
                url,
                timeout=URL_FETCH_TIMEOUT,
                headers={"User-Agent": "SecureRAG-Bot/1.0"},
                allow_redirects=True,
                stream=True,
            )

            # Check response size before reading fully
            try:
                cl = int(resp.headers.get("Content-Length", "0"))
            except (ValueError, TypeError):
                cl = 0
            if cl > URL_MAX_RESPONSE_BYTES:
                formatted.append(
                    f"[Fetched: {url}]\n"
                    "(Content too large, truncated.)"
                )
                continue

            # Only process text-like content types
            content_type = resp.headers.get("Content-Type", "")
            if "text" not in content_type and "html" not in content_type:
                formatted.append(
                    f"[Fetched: {url}]\n"
                    "(Non-text content type, skipped.)"
                )
                continue

            # Read with size limit via streaming to avoid buffering huge payloads
            raw = b""
            for chunk in resp.iter_content(chunk_size=65536):
                raw += chunk
                if len(raw) >= URL_MAX_RESPONSE_BYTES:
                    break
            raw = raw[:URL_MAX_RESPONSE_BYTES]

            # Strip HTML tags for basic text extraction
            text = raw.decode("utf-8", errors="replace")
            # Remove script and style blocks
            text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            # Remove remaining HTML tags
            text = re.sub(r"<[^>]+>", " ", text)
            # Decode HTML entities
            text = unescape(text)
            # Normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()

            if text:
                formatted.append(f"[Fetched: {url}]\n{text}")

        except Exception as e:
            logger.error("Failed to fetch URL %s: %s", url, e)
            formatted.append(f"[Fetched: {url}]\n(Error: Could not fetch this URL: {e})")

    if not formatted:
        return "Failed to fetch content from the provided URLs."

    return "\n\n---\n\n".join(formatted)


# ═══════════════════════════════════════════════════════════════
#  ReAct Agent (Manual Loop - Tool Registry + Dispatch)
# ═══════════════════════════════════════════════════════════════

def _build_tool_prompt(tools: dict[str, str]) -> str:
    """
    Builds the tool listing section of the ReAct prompt from a
    tool registry dict. Each entry maps a tool name to a
    human-readable description.

    Args:
        tools: Dict of {tool_name: tool_description}.

    Returns:
        Formatted string listing all available tools.
    """
    lines = []
    for name, desc in tools.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def _build_react_conversation(
    system_prompt: str,
    tools: dict[str, str],
    query: str,
    mode: str = MODE_KB_ONLY,
) -> str:
    """
    Builds the initial ReAct conversation string with the system
    prompt, tool listing, and the user's question.

    Args:
        system_prompt: The mode- and model-aware system prompt.
        tools: Dict of {tool_name: tool_description}.
        query: The user's question.
        mode: Search mode for mode-specific format instructions.

    Returns:
        The complete initial conversation string.
    """
    tool_listing = _build_tool_prompt(tools)

    # Mode-specific format rules to avoid prompt conflicts
    mode_rules = {
        MODE_KB_ONLY: [
            "You MUST search the knowledge base before providing a final answer.",
            "Do not answer questions about uploaded documents from general knowledge alone.",
        ],
        MODE_WEB_ONLY: [
            "You MUST search the web before providing a final answer.",
            "Always cite source URLs for web-derived information.",
        ],
        MODE_HYBRID: [
            "You MUST search the knowledge base before providing a final answer.",
            "You may also search the web for supplementary information.",
            "When information conflicts, prefer the knowledge base.",
        ],
    }
    extra_rules = "\n".join(
        f"- {r}" for r in mode_rules.get(mode, mode_rules[MODE_KB_ONLY])
    )

    return f"""{system_prompt}

You have access to the following tools:
{tool_listing}

Use the following format strictly:

Question: the input question you must answer
Thought: you should always think about what to do
Action: <tool_name>
Action Input: a concise search query
Observation: <the search results will appear here>
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer based on the observation(s).
Final Answer: the final answer to the original input question

IMPORTANT RULES:
- You MUST use one of the listed tool names exactly as written.
- Provide ONLY the tool name after "Action:", nothing else.
- Provide a concise search query after "Action Input:".
- After receiving the Observation, decide if you need more information
  or if you can provide the Final Answer.
{extra_rules}

Begin!

Question: {query}

Thought: I need to search first."""


def _parse_action(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parses the LLM's output to extract the tool name and action input.

    Looks for the pattern:
        Action: <tool_name>
        Action Input: <query>

    Args:
        text: The LLM's response text.

    Returns:
        A tuple of (tool_name, action_input). Either may be None
        if the pattern is not found.
    """
    # Extract tool name from "Action: ..."
    action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
    if not action_match:
        return None, None
    tool_name = action_match.group(1).strip()

    # Extract action input from "Action Input: ..."
    input_match = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)
    if not input_match:
        return tool_name, None
    action_input = input_match.group(1).strip()
    # Take only the first line (ignore anything after a newline)
    action_input = action_input.split("\n")[0].strip()
    # Remove any trailing "Observation:" if the LLM generated it
    action_input = action_input.split("Observation:")[0].strip()

    return tool_name, action_input


def query_agent(
    kb: "HybridKnowledgeBase",
    api_key: str,
    domain: str,
    query: str,
    mode: str = MODE_KB_ONLY,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = DEFAULT_ALPHA,
    tavily_api_key: Optional[str] = None,
) -> str:
    """
    Executes a ReAct-style Think-Act-Observe loop using a tool
    registry and dispatch table. Supports any number of tools
    without importing langchain.agents or langchain.chains.

    Tool registry:
      A dict mapping tool names to (description, handler_function).
      The handler function receives the action input string and
      returns an observation string.

    Available tools per mode:
      - kb_only:  Hybrid_Knowledge_Search
      - web_only: Web_Search
      - hybrid:   Hybrid_Knowledge_Search + Web_Search

    Error recovery (F5 fix):
      If a tool handler throws an exception, the error message is
      returned as the Observation instead of crashing the loop.
      The LLM can then decide to retry or provide a fallback answer.

    Args:
        kb: The HybridKnowledgeBase instance (from session state).
        api_key: Google API key for LLM initialization.
        domain: Current domain for system prompt selection.
        query: The user's question.
        mode: Search mode (MODE_KB_ONLY, MODE_WEB_ONLY, MODE_HYBRID).
        top_k: Number of documents to retrieve per KB search.
        alpha: BM25 weight for hybrid search.
        tavily_api_key: Tavily API key for web search (optional).

    Returns:
        Validated response string.
    """
    # ── Build tool registry based on mode ──────────────────
    tools: dict[str, tuple[str, Callable]] = {}

    if mode in (MODE_KB_ONLY, MODE_HYBRID):
        tools["Hybrid_Knowledge_Search"] = (
            "Searches the uploaded knowledge base using hybrid retrieval "
            "(BM25 keyword + FAISS semantic). Use this for questions about "
            "the user's uploaded documents.",
            lambda q: kb.hybrid_search(
                q, top_k=top_k, alpha=alpha, domain=domain
            ),
        )

    if mode in (MODE_WEB_ONLY, MODE_HYBRID):
        tools["Web_Search"] = (
            "Searches the web for current information. Use this for "
            "questions about recent events, general knowledge, or topics "
            "not covered by the uploaded documents.",
            lambda q: web_search(
                q, provider=DEFAULT_WEB_PROVIDER,
                tavily_api_key=tavily_api_key,
            ),
        )

    if not tools:
        return validate_output(
            "No search tools are available in the current mode."
        )

    # ── Find a working text model ─────────────────────────
    text_model = _first_working_model(
        TEXT_MODEL_CANDIDATES,
        api_key,
    )
    if not text_model:
        return validate_output(
            "No AI model is currently available. All models are "
            "rate-limited or unreachable. Please wait and try again."
        )

    def _try_llm(model: str) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=model, google_api_key=api_key, temperature=LLM_TEMPERATURE,
        )

    llm = _try_llm(text_model)
    system_prompt = get_system_prompt(domain, mode=mode, model=text_model)

    # Build tool descriptions dict for prompt generation
    tool_descriptions = {name: desc for name, (desc, _) in tools.items()}
    conversation = _build_react_conversation(
        system_prompt, tool_descriptions, query, mode=mode
    )

    last_text = ""

    for iteration in range(MAX_AGENT_ITERATIONS + 1):
        try:
            response = _invoke_with_retry(lambda: llm.invoke(conversation), service="gemini-llm")
            text = response.content if hasattr(response, "content") else str(response)
        except APIError as e:
            if _is_rate_limit(e):
                logger.warning("Rate limited on %s, searching for next model", llm.model)
                _record_metric("primary_rate_limited", 1)
                _model_status[llm.model] = "rate_limited"
                next_model = _first_working_model(
                    TEXT_MODEL_CANDIDATES,
                    api_key,
                )
                if next_model:
                    llm = _try_llm(next_model)
                    system_prompt = get_system_prompt(domain, mode=mode, model=next_model)
                    conversation = _build_react_conversation(
                        system_prompt, tool_descriptions, query, mode=mode
                    )
                    try:
                        response = _invoke_with_retry(lambda: llm.invoke(conversation), service="gemini-llm-fallback")
                        text = response.content if hasattr(response, "content") else str(response)
                    except APIError as e2:
                        logger.error("Fallback LLM also failed: %s", e2)
                        return "AI model is temporarily unavailable. Please wait a moment and try again."
                    except Exception as e2:
                        logger.error("Fallback LLM unexpected error: %s", e2, exc_info=True)
                        return "An unexpected error occurred. Please try again."
                else:
                    return "All AI models are rate-limited. Please wait and try again."
            else:
                logger.error("LLM failed: %s", e, extra={"api_service": e.service, "error_code": e.error_code})
                return f"The AI model is temporarily unavailable. Please try again in a few seconds."
        except Exception as e:
            logger.error("LLM invocation failed: %s", e, exc_info=True)
            return f"An unexpected error occurred while contacting the AI model."

        last_text = text

        # Check for Final Answer
        if "Final Answer:" in text:
            answer = text.split("Final Answer:", 1)[1].strip()
            return validate_output(answer)

        # Parse Action and Action Input
        tool_name, action_input = _parse_action(text)

        if tool_name and action_input:
            # Look up the tool in the registry
            if tool_name not in tools:
                # Unknown tool — feed error back as observation
                valid_tools = ", ".join(tools.keys())
                observation = (
                    f"Error: Unknown tool '{tool_name}'. "
                    f"Available tools are: {valid_tools}. "
                    "Please use one of the listed tools."
                )
            else:
                # Execute the tool with error recovery
                _, handler = tools[tool_name]
                try:
                    observation = handler(action_input)
                except Exception as e:
                    logger.error(
                        "Tool '%s' failed on iteration %d: %s",
                        tool_name, iteration, e,
                    )
                    observation = (
                        f"Error: The tool '{tool_name}' encountered "
                        f"an error: {e}. Try rephrasing your search "
                        "query or try a different approach."
                    )

            # Append reasoning + observation, prompt for next step
            conversation += f"""{text}
Observation: {observation}

Thought:"""
        else:
            # No structured output detected; return as-is
            break

    # Fallback: return whatever the LLM produced in the last iteration
    return validate_output(last_text)


# ═══════════════════════════════════════════════════════════════
#  Direct Chat (No Retrieval)
# ═══════════════════════════════════════════════════════════════

def direct_chat(
    api_key: str,
    domain: str,
    query: str,
    chat_history: list[dict],
) -> str:
    """
    Handles queries in Direct Chat mode (no retrieval tools).
    The LLM answers using its built-in knowledge, with the
    domain-specific system prompt and security guardrails applied.

    Args:
        api_key: Google API key.
        domain: Current domain for system prompt.
        query: The user's question.
        chat_history: Recent chat messages for context.

    Returns:
        Validated response string.
    """
    def _try_llm(model: str) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=model, google_api_key=api_key, temperature=LLM_TEMPERATURE,
        )

    text_model = _first_working_model(
        TEXT_MODEL_CANDIDATES,
        api_key,
    )
    if not text_model:
        return validate_output(
            "No AI model is currently available. All models are "
            "rate-limited or unreachable. Please wait and try again."
        )

    llm = _try_llm(text_model)
    system_prompt = get_system_prompt(domain, mode=MODE_DIRECT, model=text_model)

    # Build message list from recent chat history
    messages = [{"role": "system", "content": system_prompt}]

    for msg in chat_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": query})

    try:
        response = _invoke_with_retry(lambda: llm.invoke(messages), service="gemini-chat")
        text = response.content if hasattr(response, "content") else str(response)
        return validate_output(text)
    except APIError as e:
        if _is_rate_limit(e):
            logger.warning("Rate limited on %s, searching for next model", llm.model)
            _record_metric("primary_rate_limited", 1)
            _model_status[llm.model] = "rate_limited"
            next_model = _first_working_model(
                TEXT_MODEL_CANDIDATES,
                api_key,
            )
            if next_model:
                llm = _try_llm(next_model)
                system_prompt = get_system_prompt(domain, mode=MODE_DIRECT, model=next_model)
                messages[0] = {"role": "system", "content": system_prompt}
                try:
                    response = _invoke_with_retry(lambda: llm.invoke(messages), service="gemini-chat-fallback")
                    text = response.content if hasattr(response, "content") else str(response)
                    return validate_output(text)
                except APIError as e2:
                    logger.error("Fallback LLM also failed: %s", e2)
                    return "AI model is temporarily unavailable. Please wait a moment and try again."
                except Exception as e2:
                    logger.error("Fallback LLM unexpected error: %s", e2, exc_info=True)
                    return "An unexpected error occurred. Please try again."
            else:
                return "All AI models are rate-limited. Please wait and try again."
        else:
            logger.error("Direct chat failed: %s", e, extra={"api_service": e.service, "error_code": e.error_code})
            return f"The AI model is temporarily unavailable. Please try again."
    except Exception as e:
        logger.error("Direct chat failed: %s", e, exc_info=True)
        return f"An unexpected error occurred while contacting the AI model."


# ═══════════════════════════════════════════════════════════════
#  Multimodal Query (via google-genai SDK)
# ═══════════════════════════════════════════════════════════════

def multimodal_query(
    api_key: str,
    query: str,
    image_bytes: bytes,
    mime_type: str,
    context: str,
    domain: str,
    mode: str = MODE_KB_ONLY,
) -> str:
    """
    Handles queries with attached images using Gemini's native
    multimodal capabilities via the google-genai SDK.

    Instead of running a local CLIP model (which would exceed
    Streamlit Cloud's 1 GB RAM), the image bytes are sent
    directly to the Gemini model alongside the retrieved text
    context from the hybrid search.

    This bypasses the ReAct agent loop because multimodal reasoning
    requires a single unified call that combines image and text.

    Args:
        api_key: Google API key.
        query: The user's text question.
        image_bytes: Raw image file bytes (JPEG/PNG).
        mime_type: MIME type (e.g. "image/jpeg").
        context: Retrieved text context from hybrid search or web.
        domain: Current domain for system prompt.
        mode: Search mode (affects system prompt grounding).

    Returns:
        Validated response string from the model.
    """
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError:
        return (
            "Multimodal query requires the 'google-genai' package. "
            "Please ensure it is installed."
        )

    client = genai.Client(api_key=api_key)

    # Build context section based on what was provided
    if context and "empty" not in context.lower() and "no relevant" not in context.lower():
        context_section = (
            f"RETRIEVED CONTEXT:\n{context}\n\n"
            "Use BOTH the visual content of the image and the "
            "text context to answer the question.\n\n"
        )
    else:
        context_section = (
            "No relevant context was retrieved. Answer based "
            "primarily on the image content.\n\n"
        )

    # Determine the first model to try — must support vision
    vision_model = _first_working_model(
        VISION_MODEL_CANDIDATES,
        api_key,
        needs_vision=True,
    )
    if not vision_model:
        return validate_output(
            "No vision-capable AI model is currently available. "
            "All models are rate-limited or unreachable."
        )

    system_prompt = get_system_prompt(domain, mode=mode, model=vision_model)

    full_prompt = (
        f"{system_prompt}\n\n"
        f"You are analyzing an image alongside available context. "
        f"{context_section}"
        f"USER QUESTION: {query}\n\n"
        "Provide a clear, factual answer based on the image and "
        "context above. If the context or image does not contain "
        "enough information, state that clearly."
    )

    try:
        response = _invoke_with_retry(
            lambda: client.models.generate_content(
                model=vision_model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    full_prompt,
                ],
            ),
            service="gemini-vision",
        )
        return validate_output(response.text)
    except APIError as e:
        if _is_rate_limit(e):
            logger.warning("Rate limited on %s, searching for next vision model", vision_model)
            _record_metric("primary_rate_limited", 1)
            _model_status[vision_model] = "rate_limited"
            next_vision = _first_working_model(
                VISION_MODEL_CANDIDATES,
                api_key,
                needs_vision=True,
            )
            if next_vision:
                try:
                    response = _invoke_with_retry(
                        lambda: client.models.generate_content(
                            model=next_vision,
                            contents=[
                                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                                full_prompt,
                            ],
                        ),
                        service="gemini-vision-fallback",
                    )
                    return validate_output(response.text)
                except APIError as e2:
                    logger.error("Fallback vision model also failed: %s", e2)
                    return "AI model is temporarily unavailable. Please wait a moment and try again."
                except Exception as e2:
                    logger.error("Fallback vision unexpected error: %s", e2, exc_info=True)
                    return "An unexpected error occurred. Please try again."
            else:
                return "All vision AI models are rate-limited. Please wait and try again."
        else:
            logger.error(
                "Multimodal query failed on %s: %s",
                vision_model, e,
                extra={"api_service": e.service, "error_code": e.error_code},
            )
            return f"The AI model is temporarily unavailable. Please try again."
    except Exception as e:
        logger.error("Multimodal query failed on %s: %s", vision_model, e, exc_info=True)
        return f"An unexpected error occurred while processing your image query."


# ═══════════════════════════════════════════════════════════════
#  Document Processing Utilities
# ═══════════════════════════════════════════════════════════════

# Module-level OCR diagnostic (cleared before each extraction)
_ocr_diagnostics: list[str] = []

def get_ocr_diagnostics() -> list[str]:
    return list(_ocr_diagnostics)

def extract_text_from_file(
    file_obj,
    max_chars: int = MAX_CHARS_PER_FILE,
    api_key: str | None = None,
) -> str:
    """
    Extracts plain text from an uploaded file object (PDF or TXT).

    Strategy:
      1. pypdf extracts text from text-based PDFs (fast, zero API calls).
      2. If no text is found (scanned/image PDF), pymupdf renders pages
         to images and Gemini Vision API performs OCR — zero local models.
      3. For TXT files, decodes as UTF-8 with fallback for encoding errors.

    Args:
        file_obj: A Streamlit UploadedFile object.
        max_chars: Maximum characters to extract (safety limit).
        api_key: Google API key for Gemini Vision OCR fallback.

    Returns:
        Extracted text string, or empty string on failure.
    """
    _ocr_diagnostics.clear()
    name = file_obj.name.lower()

    try:
        if name.endswith(".pdf"):
            # Stage 1: Try text extraction with pypdf
            try:
                from pypdf import PdfReader
            except ImportError:
                from PyPDF2 import PdfReader

            reader = PdfReader(file_obj)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    if len(text) > max_chars:
                        break

            # Stage 2: If no text (scanned PDF), use Gemini Vision OCR
            if not text.strip() and api_key:
                logger.info(
                    "PDF '%s' has no extractable text (%d pages). "
                    "Attempting Gemini Vision OCR...",
                    file_obj.name, len(reader.pages),
                )
                text = _ocr_pdf_with_gemini(file_obj, api_key, max_chars)

            return text[:max_chars].strip()

        elif name.endswith(".txt"):
            raw = file_obj.getvalue()
            return raw.decode("utf-8", errors="replace")[:max_chars].strip()

        else:
            logger.warning("Unsupported file type: %s", name)
            return ""

    except Exception as e:
        logger.error("File extraction failed for %s: %s", name, e)
        return ""


def _ocr_pdf_with_gemini(
    file_obj, api_key: str, max_chars: int
) -> str:
    """
    Renders each PDF page to an image at high DPI, then sends each page
    individually to Gemini Vision API for text extraction.

    Page-by-page processing (instead of bulk) means:
      - If one page fails, the rest still produce chunks
      - Each API call has a smaller payload (faster, less token usage)
      - The model can focus on one page at a time for better accuracy

    Uses the model registry to test + cache model availability,
    and falls back to the next candidate if the current model fails.
    """
    try:
        import pymupdf
    except ImportError as e:
        msg = f"pymupdf not installed: {e}"
        logger.error(msg)
        _ocr_diagnostics.append(msg)
        return ""

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        msg = f"google-genai not installed: {e}"
        logger.error(msg)
        _ocr_diagnostics.append(msg)
        return ""

    try:
        file_obj.seek(0)
        doc = pymupdf.open(stream=file_obj.read(), filetype="pdf")
        total_pages = len(doc)
        pages_to_process = min(total_pages, 10)

        logger.info(
            "Rendering %d/%d pages at %d DPI for Gemini OCR...",
            pages_to_process, total_pages, OCR_DPI,
        )

        # Render all pages to PNG images at high DPI
        page_images: list[bytes] = []
        for i in range(pages_to_process):
            page = doc[i]
            pix = page.get_pixmap(dpi=OCR_DPI)
            page_images.append(pix.tobytes("png"))
        doc.close()

        client = genai.Client(api_key=api_key)

        # Find a working vision model
        ocr_model = _first_working_model(
            VISION_MODEL_CANDIDATES,
            api_key,
            needs_vision=True,
        )
        if not ocr_model:
            msg = (
                "No vision-capable model available for OCR. "
                "Enable a vision model (e.g. gemini-2.0-flash) in "
                "your Google API console."
            )
            logger.error(msg)
            _ocr_diagnostics.append(msg)
            return ""

        all_text_parts: list[str] = []
        model_tried = ocr_model
        candidate_pool = list(VISION_MODEL_CANDIDATES)

        for page_idx, img_bytes in enumerate(page_images):
            # If current model was rate-limited, find next
            if _model_status.get(model_tried) in ("rate_limited", "unavailable"):
                remaining = [m for m in candidate_pool if _model_status.get(m) != "rate_limited" and _model_status.get(m) != "unavailable"]
                if remaining:
                    model_tried = remaining[0]
                else:
                    logger.warning("All models exhausted after page %d", page_idx)
                    break

            prompt = types.Part.from_text(
                text=(
                    f"Transcribe ALL visible text from page {page_idx+1} of {pages_to_process} "
                    "verbatim — every word, heading, list item, code line, and punctuation mark. "
                    "Do NOT summarize, paraphrase, or omit anything. "
                    "Preserve paragraphs and line breaks exactly as they appear. "
                    "Return ONLY the raw transcribed text, no commentary."
                )
            )
            image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")

            try:
                response = _invoke_with_retry(
                    lambda m=model_tried: client.models.generate_content(
                        model=m, contents=[prompt, image_part],
                        config=types.GenerateContentConfig(
                            max_output_tokens=8192,
                            temperature=0.0,
                        ),
                    ),
                    service=f"gemini-ocr-{model_tried}",
                )
                page_text = response.text if hasattr(response, "text") else str(response)
                if page_text.strip():
                    all_text_parts.append(page_text.strip())
                    logger.info(
                        "Page %d: extracted %d chars using %s",
                        page_idx + 1, len(page_text), model_tried,
                    )
                else:
                    logger.warning("Page %d: empty text from %s", page_idx + 1, model_tried)
            except APIError as e:
                logger.warning("Page %d failed on %s: %s", page_idx + 1, model_tried, e)
                _ocr_diagnostics.append(f"Page {page_idx+1} on {model_tried}: {e}")
                if _is_rate_limit(e):
                    _model_status[model_tried] = "rate_limited"
                else:
                    _model_status[model_tried] = "unavailable"
            except Exception as e:
                logger.warning("Page %d unexpected error on %s: %s", page_idx + 1, model_tried, e)
                _ocr_diagnostics.append(f"Page {page_idx+1} error: {type(e).__name__}")

        combined = "\n\n".join(all_text_parts)
        if combined:
            logger.info(
                "OCR complete: %d chars from %d/%d pages using %s",
                len(combined), len(all_text_parts), pages_to_process, ocr_model,
            )
        else:
            logger.warning("OCR produced no text from any page")
            _ocr_diagnostics.append("OCR returned empty for all pages")

        return combined[:max_chars]

    except Exception as e:
        msg = f"Gemini Vision OCR setup failed: {type(e).__name__}: {e}"
        logger.error(msg)
        _ocr_diagnostics.append(msg)
        return ""


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Splits text into overlapping chunks using a sliding window
    with sentence-boundary awareness.

    Instead of blindly cutting at a fixed character offset, this
    function looks for the nearest sentence-ending punctuation
    (period, exclamation, question mark) within a 100-character
    window of the target cut point. This prevents mid-sentence
    splits that would degrade retrieval quality.

    Args:
        text: The full document text.
        chunk_size: Target size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunk strings.
    """
    if not text or not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        # Try to break at a sentence boundary near the target end
        if end < text_len:
            search_start = max(start, end - 100)
            boundary = text.rfind(".", search_start, end + 50)
            if boundary == -1:
                boundary = text.rfind("!", search_start, end + 50)
            if boundary == -1:
                boundary = text.rfind("?", search_start, end + 50)
            if boundary != -1 and boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk and len(chunk) > 20:  # Skip very short fragments
            chunks.append(chunk)

        # Advance with overlap, prevent infinite loop
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks