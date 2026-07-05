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
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from security import get_system_prompt, validate_output, sanitize_search_query
from exceptions import APIError
from config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
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
        for key in ("duration_ms", "api_service", "error_code", "query_len", "chunk_count"):
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

def _invoke_with_retry(callable_fn, max_attempts=3, service="gemini"):
    """
    Invokes a callable with exponential backoff retry on failure.

    Targets transient API errors (rate limits, server hiccups) by
    sleeping 1s, then 2s, then 4s between attempts. After exhausting
    retries raises APIError so the caller can distinguish infrastructure
    failures from expected states (empty KB, blocked content).

    Usage:
        result = _invoke_with_retry(lambda: llm.invoke(prompt))
    """
    start = time.perf_counter()
    last_exc = None
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
            logger.warning(
                "%s call failed (attempt %d/%d): %s",
                service, attempt + 1, max_attempts, e,
                extra={"api_service": service},
            )
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
    duration_ms = (time.perf_counter() - start) * 1000
    _record_metric(f"{service}_error_count", 1)
    raise APIError(
        message=f"{service} API unreachable after {max_attempts} attempts: {last_exc}",
        service=service,
    ) from last_exc


# ═══════════════════════════════════════════════════════════════
#  Hybrid Knowledge Base
# ═══════════════════════════════════════════════════════════════

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
        self.vectorstore: Optional[FAISS] = None
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

        # FAISS: create new or append to existing
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_texts(
                chunks, self.embeddings, metadatas=metas
            )
        else:
            self.vectorstore.add_texts(chunks, metadatas=metas)

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

        if self.vectorstore is not None:
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query, k=n
            )
            for doc, score in docs_with_scores:
                # Convert L2 distance to similarity: closer = higher score
                similarity = 1.0 / (1.0 + float(score))
                try:
                    idx = self.corpus.index(doc.page_content)
                    vector_norm[idx] = similarity
                except ValueError:
                    # Edge case: content not in corpus (should not happen)
                    pass

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
            return (
                "No relevant documents found for the given query "
                "and domain filter. Try adjusting your search "
                "parameters or uploading relevant documents."
            )

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

    Uses Tavily's extract API which fetches and parses pages
    server-side, avoiding SSRF risk on the Streamlit instance.
    If Tavily is unavailable, falls back to a local HTTP fetcher
    (with SSRF protections applied by the caller via validate_url).

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
        logger.warning("Tavily extract failed, falling back to local fetch.")

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
) -> str:
    """
    Builds the initial ReAct conversation string with the system
    prompt, tool listing, and the user's question.

    Args:
        system_prompt: The mode-aware system prompt.
        tools: Dict of {tool_name: tool_description}.
        query: The user's question.

    Returns:
        The complete initial conversation string.
    """
    tool_listing = _build_tool_prompt(tools)

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

Begin!

Question: {query}

Thought:"""


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

    # ── Initialize LLM ──────────────────────────────────────
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=LLM_TEMPERATURE,
    )

    system_prompt = get_system_prompt(domain, mode=mode)

    # Build tool descriptions dict for prompt generation
    tool_descriptions = {name: desc for name, (desc, _) in tools.items()}
    conversation = _build_react_conversation(
        system_prompt, tool_descriptions, query
    )

    last_text = ""

    for iteration in range(MAX_AGENT_ITERATIONS + 1):
        try:
            response = _invoke_with_retry(lambda: llm.invoke(conversation), service="gemini-llm")
            text = response.content if hasattr(response, "content") else str(response)
        except APIError as e:
            logger.error("LLM failed: %s", e, extra={"api_service": e.service, "error_code": e.error_code})
            return f"The AI model is temporarily unavailable ({e.service}). Please try again in a few seconds."
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
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=LLM_TEMPERATURE,
    )

    system_prompt = get_system_prompt(domain, mode=MODE_DIRECT)

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
        logger.error("Direct chat failed: %s", e, extra={"api_service": e.service, "error_code": e.error_code})
        return f"The AI model is temporarily unavailable ({e.service}). Please try again."
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
    system_prompt = get_system_prompt(domain, mode=mode)

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
                model=LLM_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    full_prompt,
                ],
            ),
            service="gemini-vision",
        )
        return validate_output(response.text)
    except APIError as e:
        logger.error("Multimodal query failed: %s", e, extra={"api_service": e.service, "error_code": e.error_code})
        return f"The AI model is temporarily unavailable ({e.service}). Please try again."
    except Exception as e:
        logger.error("Multimodal query failed: %s", e, exc_info=True)
        return f"An unexpected error occurred while processing your image query."


# ═══════════════════════════════════════════════════════════════
#  Document Processing Utilities
# ═══════════════════════════════════════════════════════════════

def extract_text_from_file(file_obj, max_chars: int = MAX_CHARS_PER_FILE) -> str:
    """
    Extracts plain text from an uploaded file object (PDF or TXT).

    For PDFs, uses PyPDF2 to iterate pages and concatenate extracted text.
    For TXT files, decodes as UTF-8 with fallback for encoding errors.

    Args:
        file_obj: A Streamlit UploadedFile object.
        max_chars: Maximum characters to extract (safety limit).

    Returns:
        Extracted text string, or empty string on failure.
    """
    name = file_obj.name.lower()

    try:
        if name.endswith(".pdf"):
            from PyPDF2 import PdfReader

            reader = PdfReader(file_obj)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    if len(text) > max_chars:
                        break
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