"""
Configuration constants for the Secure Hybrid RAG system.

All tunable parameters are centralized here for easy maintenance
and deployment consistency. Modify these values to adjust system
behavior without touching core logic.
"""


# ── Model Configuration ──────────────────────────────────────
# Embedding model for vectorization (via Google Gemini API)
# Converts text to vector embeddings for semantic search in FAISS.
EMBEDDING_MODEL: str = "models/gemini-embedding-2"

# Fallback embedding model (separate quota pool)
FALLBACK_EMBEDDING_MODEL: str = "models/gemini-embedding-2-preview"

# Primary LLM for generation and ReAct reasoning
# Uses the fastest available flash-lite model for low latency.
PRIMARY_LLM_MODEL: str = "gemini-2.0-flash-lite"

# Fallback LLM when primary is rate-limited or unavailable
FALLBACK_LLM_MODEL: str = "gemini-2.0-flash"

# Alias for backward-compatible access
LLM_MODEL: str = PRIMARY_LLM_MODEL

# Current date for web search recency context
# Injected into system prompts so the agent searches for fresh results.
from datetime import date
CURRENT_DATE: str = date.today().isoformat()  # e.g. "2026-07-06"

# Models known to support image/vision input
# These are tried first when the query includes an image.
VISION_CAPABLE_MODELS: list[str] = [
    # Flash family
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-flash-latest",
    "gemini-2.5-flash-image",
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
    "gemini-omni-flash-preview",
    "gemini-3.1-flash-live-preview",
    "gemini-2.5-flash-native-audio-latest",
    # Pro family
    "gemini-2.5-pro",
    "gemini-pro-latest",
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools",
    # 3.x flash variants (likely vision)
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash-image-preview",
    "gemini-3.1-flash-image",
    # Image-specific
    "gemini-3.1-flash-lite-image",
    # Image-specific
    "gemini-3-pro-image-preview",
    "gemini-3-pro-image",
    # Fallback vision models (slower but reliable)
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Models that are text-only (non-vision)
# Trying to send images to these will cause API errors.
NON_VISION_MODELS: list[str] = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash-lite",
    "gemini-flash-lite-latest",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
]

def is_vision_model(model: str) -> bool:
    """Return True if the model supports image input."""
    if model in VISION_CAPABLE_MODELS:
        return True
    if model in NON_VISION_MODELS:
        return False
    # Unknown model — assume non-vision to avoid hard errors
    return False

# Curated model candidates for the Registry — tested lazily, one at a time
# MAX 3 per category to avoid spamming the API with test calls.
# Ordered by preference (first = fastest / most available).
TEXT_MODEL_CANDIDATES: list[str] = [
    "gemini-3.1-flash-lite",       # fastest text-only (replaced quota-exhausted gemini-2.0-flash-lite)
    "gemini-2.5-flash-lite",      # newer, still fast
    "gemini-2.0-flash",           # fast, vision-capable fallback
]

VISION_MODEL_CANDIDATES: list[str] = [
    "gemini-2.0-flash",           # fastest vision model
    "gemini-2.5-flash",           # newer, good quality
    "gemini-3.1-flash-lite-image", # fresh quota pool, good OCR
    "gemini-2.0-flash-001",       # alternate variant
]

# OCR rendering resolution (DPI). Higher = better text recognition
# but larger image payloads. 200 is a good balance for scanned docs.
OCR_DPI: int = 200

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
DEFAULT_TOP_K: int = 8

# Hybrid search weight: 0.0 = pure vector, 1.0 = pure BM25
DEFAULT_ALPHA: float = 0.5

# Text chunking parameters for document ingestion
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 100

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