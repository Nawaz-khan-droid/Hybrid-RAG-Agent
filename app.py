"""
Secure Multimodal RAG Agent - Streamlit Application

A deployment-ready Streamlit app for a Hybrid RAG system with
ReAct agent capabilities, designed for Streamlit Community Cloud.

Architecture overview:
  - Hybrid Search: BM25 (keyword) + FAISS (semantic) with weighted fusion
  - Agent: Manual ReAct loop with tool registry (KB search, web search)
  - Web Search: Tavily (primary) or DuckDuckGo (fallback)
  - Multimodal: Native Gemini vision for image + text queries
  - URL Fetching: Tavily extract API with SSRF protection
  - Security: Three-layer guardrails (input sanitization, output validation,
    URL/file perimeter checks)

API Key handling:
  - On Streamlit Cloud: Set GOOGLE_API_KEY (and optionally TAVILY_API_KEY)
    in the Secrets panel.
  - For local testing: Create .streamlit/secrets.toml with the keys.
  - Google Colab userdata.get() will NOT work here.

Session state design:
  - "kb" (HybridKnowledgeBase): Survives reruns, mutable in-place.
  - "messages" (list[dict]): Full chat history for display.
  - "domain" (str): Current domain selection.
  - "search_mode" (str): Current search mode selection.
"""

import logging
import os

import streamlit as st

from config import (
    LLM_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_ALPHA,
    DEFAULT_WEB_PROVIDER,
    MAX_FILE_SIZE_MB,
    MAX_FILES_PER_UPLOAD,
    MAX_URLS_PER_FETCH,
    ALLOWED_EXTENSIONS,
    AVAILABLE_DOMAINS,
    SEARCH_MODES,
    MODE_MAP,
    MODE_DIRECT,
    MODE_KB_ONLY,
    MODE_WEB_ONLY,
    MODE_HYBRID,
    MAX_CONTEXT_MESSAGES,
    TAVILY_API_KEY_NAME,
)
from engine import (
    HybridKnowledgeBase,
    query_agent,
    direct_chat,
    multimodal_query,
    fetch_url_content,
    extract_text_from_file,
    chunk_text,
    get_metrics_snapshot,
)
from security import (
    sanitize_input,
    validate_url,
    validate_file_magic,
)


# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Page Configuration ──────────────────────────────────────
st.set_page_config(
    page_title="Secure Multimodal RAG Agent",
    page_icon="\U0001f6e1\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════
#  1. API KEY LOADING
# ═══════════════════════════════════════════════════════════════

def _load_api_key(secret_name: str, optional: bool = False) -> str | None:
    """
    Loads an API key from Streamlit secrets or environment variables.

    Checks st.secrets first (Streamlit Cloud / local), then falls back
    to os.environ (HuggingFace Spaces, Docker, etc.).

    Args:
        secret_name: The key name.
        optional: If True, returns None instead of stopping the app
                  when the key is missing.

    Returns:
        The API key string, or None if optional and missing.
    """
    key = None
    try:
        key = st.secrets[secret_name]
    except (KeyError, FileNotFoundError, Exception):
        key = os.environ.get(secret_name)

    if key and key.strip():
        return key.strip()

    if optional:
        return None
    st.error(
        f"**{secret_name} not found.**\n\n"
        "Set it in one of:\n"
        "- **Streamlit Cloud:** Settings > Secrets\n"
        "- **HuggingFace Spaces:** Settings > Variables and secrets > Secrets\n"
        "- **Local:** `.streamlit/secrets.toml` or environment variable\n"
        f"Add: `{secret_name} = your_key_here`"
    )
    st.stop()


API_KEY = _load_api_key("GOOGLE_API_KEY")
TAVILY_API_KEY = _load_api_key(TAVILY_API_KEY_NAME, optional=True)


# ═══════════════════════════════════════════════════════════════
#  2. SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════

def _init_session_state() -> None:
    """
    Ensures all required session state variables exist.
    Called once at the top of every script execution.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "kb" not in st.session_state:
        st.session_state.kb = HybridKnowledgeBase(api_key=API_KEY)
        logger.info("Initialized new HybridKnowledgeBase in session state.")

    if "domain" not in st.session_state:
        st.session_state.domain = AVAILABLE_DOMAINS[0]

    if "search_mode" not in st.session_state:
        st.session_state.search_mode = SEARCH_MODES[0]


_init_session_state()


# ═══════════════════════════════════════════════════════════════
#  3. HELPER: Get Recent Chat History (F2 Fix)
# ═══════════════════════════════════════════════════════════════

def _get_recent_messages() -> list[dict]:
    """
    Returns the most recent chat messages for LLM context.

    Full history is kept in session_state for display, but only
    the last MAX_CONTEXT_MESSAGES messages are sent to the LLM
    to avoid exceeding context windows and wasting tokens.

    Returns:
        List of message dicts with "role" and "content" keys.
    """
    messages = st.session_state.messages
    if len(messages) <= MAX_CONTEXT_MESSAGES:
        return messages
    return messages[-MAX_CONTEXT_MESSAGES:]


# ═══════════════════════════════════════════════════════════════
#  4. SIDEBAR: CONFIGURATION & KNOWLEDGE BASE MANAGEMENT
# ═══════════════════════════════════════════════════════════════

with st.sidebar:

    # ── Domain Selection ────────────────────────────────────
    st.header("Agent Configuration")

    domain = st.selectbox(
        label="Domain Focus",
        options=AVAILABLE_DOMAINS,
        index=AVAILABLE_DOMAINS.index(st.session_state.domain),
        help=(
            "Selects the system prompt persona. The agent will "
            "specialize its responses for the chosen domain."
        ),
    )
    st.session_state.domain = domain

    # ── Search Mode Toggle ──────────────────────────────────
    mode_display = st.selectbox(
        label="Search Mode",
        options=SEARCH_MODES,
        index=SEARCH_MODES.index(st.session_state.search_mode),
        help=(
            "Controls where the agent retrieves information from:\n"
            "- Knowledge Base Only: Searches uploaded documents.\n"
            "- Web Search Only: Searches the internet (needs Tavily key or uses DDG).\n"
            "- Hybrid (KB + Web): Searches both and merges results.\n"
            "- Direct Chat (No Retrieval): Uses Gemini's built-in knowledge."
        ),
    )
    st.session_state.search_mode = mode_display
    mode_internal = MODE_MAP.get(mode_display, MODE_KB_ONLY)

    # Show web provider info when web is involved
    if mode_internal in (MODE_WEB_ONLY, MODE_HYBRID):
        web_status = (
            "Tavily (configured)" if TAVILY_API_KEY
            else "DuckDuckGo (free, may be rate-limited)"
        )
        st.caption(f"Web provider: {web_status}")

    st.divider()

    # ── Search Parameters ───────────────────────────────────
    st.subheader("Search Parameters")

    alpha = st.slider(
        label="BM25 Weight (alpha)",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_ALPHA,
        step=0.05,
        help=(
            "Controls the balance between keyword and semantic search.\n"
            "0.0 = pure vector (FAISS) search\n"
            "0.5 = balanced hybrid\n"
            "1.0 = pure keyword (BM25) search"
        ),
        disabled=(mode_internal in (MODE_WEB_ONLY, MODE_DIRECT)),
    )

    top_k = st.slider(
        label="Top-K Results",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
        help="Number of context passages to retrieve per query.",
        disabled=(mode_internal in (MODE_WEB_ONLY, MODE_DIRECT)),
    )

    st.divider()

    # ── Knowledge Base Management ───────────────────────────
    st.subheader("Knowledge Base")

    kb: HybridKnowledgeBase = st.session_state.kb

    # Display KB statistics
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric(label="Chunks", value=kb.chunk_count)
    with col_stat2:
        st.metric(label="Sources", value=kb.source_count)

    # Show source files if any exist
    if kb.source_names:
        with st.expander("Indexed Sources", expanded=False):
            for src in kb.source_names:
                st.caption(f"- {src}")

    # File uploader
    uploaded_files = st.file_uploader(
        label="Upload PDF or TXT files",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=True,
        key="kb_uploader",
        help=(
            f"Maximum {MAX_FILE_SIZE_MB}MB per file. "
            f"Up to {MAX_FILES_PER_UPLOAD} files at once."
        ),
    )

    if st.button(
        label="Process Documents",
        type="primary",
        use_container_width=True,
    ):
        if not uploaded_files:
            st.warning("Please upload at least one file first.")
        else:
            progress_bar = st.progress(0, text="Processing...")
            total_chunks = 0
            total_files = len(uploaded_files)

            for idx, file in enumerate(uploaded_files):
                progress_bar.progress(
                    idx / total_files,
                    text=f"Processing {file.name} ({idx+1}/{total_files})...",
                )

                # Magic byte validation
                ext = file.name.rsplit(".", 1)[-1].lower() if "." in file.name else ""
                is_valid, reason = validate_file_magic(file, ext)
                if not is_valid:
                    st.warning(f"Validation failed for {file.name}: {reason}")
                    continue

                text = extract_text_from_file(file, api_key=API_KEY)
                if not text:
                    st.warning(f"Could not extract text from: {file.name}")
                    continue

                chunks = chunk_text(text)
                if not chunks:
                    st.warning(f"No usable text chunks from: {file.name}")
                    continue

                metas = [
                    {"source": file.name, "domain": domain}
                    for _ in chunks
                ]

                count = kb.add_documents(chunks, metas)
                total_chunks += count
                logger.info("Processed %s: %d chunks", file.name, count)

            progress_bar.progress(
                1.0, text="Processing complete."
            )

            if total_chunks > 0:
                st.success(
                    f"Added {total_chunks} chunks from "
                    f"{total_files} file(s) to the knowledge base."
                )
            else:
                st.error(
                    "No text could be extracted from the uploaded files. "
                    "Ensure the files contain readable text."
                )

    st.divider()

    # ── URL Fetching ────────────────────────────────────────
    st.subheader("Fetch from URL")

    url_input = st.text_area(
        label="Enter URLs (one per line)",
        height=80,
        placeholder="https://example.com/article1\nhttps://example.com/article2",
        help=(
            "Paste one or more URLs to fetch their content and add it "
            "to the knowledge base. Each URL is validated for security "
            f"(SSRF protection). Maximum {MAX_URLS_PER_FETCH} URLs at once."
        ),
        key="url_input_area",
    )

    if st.button(
        label="Fetch & Add to Knowledge Base",
        use_container_width=True,
    ):
        if not url_input or not url_input.strip():
            st.warning("Please enter at least one URL.")
        else:
            urls = [
                u.strip() for u in url_input.strip().splitlines() if u.strip()
            ][:MAX_URLS_PER_FETCH]

            # Validate all URLs before fetching any
            valid_urls = []
            for url in urls:
                is_safe, reason = validate_url(url)
                if not is_safe:
                    st.error(f"Blocked: {url}\n{reason}")
                else:
                    valid_urls.append(url)

            if valid_urls:
                with st.spinner("Fetching content from URLs..."):
                    fetched_text = fetch_url_content(
                        valid_urls, tavily_api_key=TAVILY_API_KEY
                    )

                if fetched_text and "Failed" not in fetched_text and "Error" not in fetched_text.split("\n")[0]:
                    chunks = chunk_text(fetched_text)
                    if chunks:
                        metas = [
                            {"source": "web:" + u.split("//")[-1].split("/")[0], "domain": domain}
                            for u in valid_urls
                            for _ in range(max(1, len(chunks) // max(1, len(valid_urls))))
                        ]
                        # Ensure metas matches chunks length
                        while len(metas) < len(chunks):
                            metas.append(metas[-1] if metas else {"source": "web:unknown", "domain": domain})
                        metas = metas[:len(chunks)]

                        count = kb.add_documents(chunks, metas)
                        st.success(
                            f"Fetched and indexed {count} chunks from "
                            f"{len(valid_urls)} URL(s)."
                        )
                    else:
                        st.warning(
                            "URL content was fetched but produced no "
                            "usable text chunks."
                        )
                else:
                    st.error(
                        f"Failed to fetch URL content. "
                        f"{'Tavily API key may be missing. ' if not TAVILY_API_KEY else ''}"
                        f"Details: {fetched_text[:200]}"
                    )

    st.divider()

    # ── Clear Actions ───────────────────────────────────────
    col_clear1, col_clear2 = st.columns(2)

    with col_clear1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col_clear2:
        if st.button("Clear Knowledge Base", use_container_width=True):
            st.session_state.kb = HybridKnowledgeBase(api_key=API_KEY)
            st.success("Knowledge base cleared.")
            st.rerun()

    st.divider()
    with st.expander("Telemetry (latency ms)", expanded=False):
        snap = get_metrics_snapshot()
        if snap:
            for name, stats in snap.items():
                st.caption(
                    f"{name}: avg={stats['avg']:.0f} "
                    f"p95={stats['p95']:.0f} "
                    f"n={stats['count']}"
                )
        else:
            st.caption("No data yet — send a query")
    st.caption(
        "Secure Hybrid RAG v2.0\n"
        "Powered by Gemini + FAISS + BM25 + Web Search"
    )


# ═══════════════════════════════════════════════════════════════
#  5. MAIN AREA: CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════

mode_label = st.session_state.search_mode
st.title("Secure Multimodal RAG Agent")
st.caption(
    f"Mode: **{mode_label}** | "
    f"Domain: **{domain}** | "
    f"Retrieval: Hybrid (BM25 + FAISS) | "
    f"Model: {LLM_MODEL}"
)

# ── Render Chat History ─────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("image_bytes"):
            st.image(msg["image_bytes"], width=300, caption="Uploaded image")


# ── Image Attachment (for Multimodal Queries) ───────────────
with st.expander("Attach an image (optional)", expanded=False):
    img_file = st.file_uploader(
        label="Upload an image for visual analysis",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )
    if img_file:
        st.image(img_file, width=200, caption="This image will be attached to your next query")


# ── Chat Input & Query Processing ───────────────────────────
if user_query := st.chat_input("Ask the agent a question..."):

    # ── Layer 1 Security: Input Sanitization ───────────────
    is_safe, block_reason = sanitize_input(user_query)

    if not is_safe:
        # Log and display the security block
        security_msg = f"**Security Alert:** {block_reason}"
        st.session_state.messages.append({
            "role": "assistant",
            "content": security_msg,
        })
        with st.chat_message("assistant"):
            st.error(block_reason)
        logger.warning("Input blocked: %s", block_reason)

    else:
        # ── Store User Message ─────────────────────────────
        user_msg: dict = {"role": "user", "content": user_query}
        if img_file:
            user_msg["image_bytes"] = img_file.getvalue()

        st.session_state.messages.append(user_msg)

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
            if img_file:
                st.image(img_file, width=300, caption="Attached image")

        # ── Generate Response ──────────────────────────────
        with st.chat_message("assistant"):
            status_placeholder = st.empty()

            try:
                # ── Multimodal Path (image attached) ───────
                if img_file:
                    status_placeholder.info(
                        "Retrieving context and analyzing image..."
                    )

                    image_bytes = img_file.getvalue()
                    ext = img_file.name.split(".")[-1].lower()
                    mime_map = {
                        "jpg": "image/jpeg",
                        "jpeg": "image/jpeg",
                        "png": "image/png",
                    }
                    mime_type = mime_map.get(ext, "image/jpeg")

                    # Retrieve text context based on mode
                    context = ""
                    if mode_internal in (MODE_KB_ONLY, MODE_HYBRID):
                        context = kb.hybrid_search(
                            user_query,
                            top_k=top_k,
                            alpha=alpha,
                            domain=domain,
                        )
                    if mode_internal in (MODE_WEB_ONLY, MODE_HYBRID) and context:
                        # For hybrid with image, note that web search
                        # context is not fetched here to keep latency low.
                        # The image path prioritizes the image analysis.
                        pass

                    response = multimodal_query(
                        API_KEY,
                        user_query,
                        image_bytes,
                        mime_type,
                        context,
                        domain,
                        mode=mode_internal,
                    )

                # ── Direct Chat Path ────────────────────────
                elif mode_internal == MODE_DIRECT:
                    status_placeholder.info("Thinking...")
                    recent_messages = _get_recent_messages()
                    response = direct_chat(
                        API_KEY, domain, user_query, recent_messages
                    )

                # ── Web Only (KB empty is OK here) ─────────
                elif mode_internal == MODE_WEB_ONLY:
                    status_placeholder.info("Searching the web...")
                    response = query_agent(
                        kb, API_KEY, domain, user_query,
                        mode=mode_internal,
                        top_k=top_k, alpha=alpha,
                        tavily_api_key=TAVILY_API_KEY,
                    )

                # ── KB Only or Hybrid ──────────────────────
                elif mode_internal in (MODE_KB_ONLY, MODE_HYBRID):
                    if kb.is_empty and mode_internal == MODE_KB_ONLY:
                        response = (
                            "Knowledge base is empty. Please upload PDF or TXT "
                            "documents using the sidebar, or fetch content from "
                            "URLs, before asking questions in this mode. "
                            "Alternatively, switch to Web Search or Direct Chat mode."
                        )
                        status_placeholder.warning(response)
                    else:
                        status_placeholder.info("Agent is reasoning...")
                        response = query_agent(
                            kb, API_KEY, domain, user_query,
                            mode=mode_internal,
                            top_k=top_k, alpha=alpha,
                            tavily_api_key=TAVILY_API_KEY,
                        )

                else:
                    response = "Unknown search mode. Please select a valid mode in the sidebar."

                # Display and store the response
                status_placeholder.empty()
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                })

            except Exception as e:
                error_msg = (
                    f"An error occurred while processing your query: {e}"
                )
                logger.error("Agent execution failed: %s", e, exc_info=True)
                status_placeholder.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })