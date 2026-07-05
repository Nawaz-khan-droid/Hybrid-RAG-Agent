"""
Configuration constants for the Secure Hybrid RAG system.

All tunable parameters are centralized here for easy maintenance
and deployment consistency. Modify these values to adjust system
behavior without touching core logic.
"""


# ── Model Configuration ──────────────────────────────────────
# Embedding model for vectorization (via Google Gemini API)
EMBEDDING_MODEL: str = "models/gemini-embedding-001"

# LLM model for generation and ReAct reasoning
LLM_MODEL: str = "gemini-2.5-flash-lite"

# Low temperature for deterministic, factual responses
LLM_TEMPERATURE: float = 0.1

# Maximum ReAct agent thought-action-observation loops
MAX_AGENT_ITERATIONS: int = 4


# ── Search Mode Configuration ─────────────────────────────────
# Available search modes for the agent
SEARCH_MODES: list[str] = [
    "Knowledge Base Only",
    "Web Search Only",
    "Hybrid (KB + Web)",
    "Direct Chat (No Retrieval)",
]

# Internal mode identifiers (used in logic, not shown to users)
MODE_KB_ONLY: str = "kb_only"
MODE_WEB_ONLY: str = "web_only"
MODE_HYBRID: str = "hybrid"
MODE_DIRECT: str = "direct"

# Map from display names to internal identifiers
MODE_MAP: dict[str, str] = {
    "Knowledge Base Only": MODE_KB_ONLY,
    "Web Search Only": MODE_WEB_ONLY,
    "Hybrid (KB + Web)": MODE_HYBRID,
    "Direct Chat (No Retrieval)": MODE_DIRECT,
}


# ── Retrieval Configuration ──────────────────────────────────
# Number of documents to retrieve per query
DEFAULT_TOP_K: int = 5

# Hybrid search weight: 0.0 = pure vector, 1.0 = pure BM25
DEFAULT_ALPHA: float = 0.5

# Text chunking parameters for document ingestion
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

# Maximum characters to extract from a single file
MAX_CHARS_PER_FILE: int = 500_000


# ── Web Search Configuration ──────────────────────────────────
# Web search provider: "tavily", "ddg", or "auto" (tavily first, ddg fallback)
DEFAULT_WEB_PROVIDER: str = "auto"

# Number of web search results to fetch
WEB_SEARCH_MAX_RESULTS: int = 5

# Maximum length for a search query sent to the web search API
MAX_SEARCH_QUERY_LENGTH: int = 200

# Streamlit secrets key name for the Tavily API key
TAVILY_API_KEY_NAME: str = "TAVILY_API_KEY"


# ── URL Fetching Configuration ────────────────────────────────
# Maximum time (seconds) to wait for a URL fetch
URL_FETCH_TIMEOUT: int = 10

# Maximum response size (bytes) for a fetched URL
URL_MAX_RESPONSE_BYTES: int = 512_000

# Maximum redirect hops to follow
URL_MAX_REDIRECTS: int = 3

# Maximum URLs a user can fetch at once
MAX_URLS_PER_FETCH: int = 5

# Only these URL schemes are allowed
ALLOWED_URL_SCHEMES: list[str] = ["http", "https"]

# IP ranges that are blocked (private, loopback, link-local)
BLOCKED_CIDR_RANGES: list[str] = [
    "127.0.0.0/8",       # loopback
    "10.0.0.0/8",        # private class A
    "172.16.0.0/12",     # private class B
    "192.168.0.0/16",    # private class C
    "169.254.0.0/16",    # link-local (AWS metadata endpoint)
    "0.0.0.0/8",         # current network
    "224.0.0.0/4",       # multicast
    "240.0.0.0/4",       # reserved
    "255.255.255.255/32", # broadcast
    "::1/128",           # IPv6 loopback
    "fc00::/7",          # IPv6 unique local
    "fe80::/10",         # IPv6 link-local
]


# ── Domain Configuration ─────────────────────────────────────
AVAILABLE_DOMAINS: list[str] = [
    "Custom/General",
    "Financial",
    "Healthcare",
    "Legal",
    "Technology",
]


# ── File Upload Constraints ──────────────────────────────────
# These limits keep the app within Streamlit Cloud free tier RAM
MAX_FILE_SIZE_MB: int = 10
MAX_FILES_PER_UPLOAD: int = 10
ALLOWED_EXTENSIONS: list[str] = ["pdf", "txt"]


# ── Security Configuration ───────────────────────────────────
MAX_INPUT_LENGTH: int = 2000

# Magic byte signatures for file validation
FILE_MAGIC_BYTES: dict[str, bytes] = {
    "pdf": b"%PDF-",
}


# ── Chat History Configuration ───────────────────────────────
# Maximum number of message pairs (user+assistant) to send to the LLM
# Older messages are still displayed but not sent as context
MAX_CONTEXT_MESSAGES: int = 10