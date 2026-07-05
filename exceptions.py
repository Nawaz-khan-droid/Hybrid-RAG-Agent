"""
Typed exception hierarchy for the RAG application.

Allows the UI layer (app.py) to distinguish recoverable errors
(empty KB, blocked input) from infrastructure failures (API down,
network error) and display appropriate messages.

Exception tree:
  RAGError
   +-- APIError          (Gemini, Tavily, etc. unreachable or throttled)
   +-- InputBlockedError (user query blocked by security filters)
   +-- ConfigurationError (missing API key, invalid config)
"""


class RAGError(Exception):
    """Base for all application-level errors."""

    def __init__(self, message: str, error_code: str = "UNKNOWN", details: str = ""):
        self.error_code = error_code
        self.details = details
        super().__init__(message)


class APIError(RAGError):
    """External API call failed after retries (Gemini, Tavily, etc.)."""

    def __init__(self, message: str, service: str = "unknown", status_code: int = 0):
        self.service = service
        self.status_code = status_code
        super().__init__(message, error_code="API_ERROR")


class InputBlockedError(RAGError):
    """User input was blocked by security guardrails."""

    def __init__(self, message: str, reason: str = ""):
        super().__init__(message, error_code="INPUT_BLOCKED", details=reason)


class ConfigurationError(RAGError):
    """Missing or invalid configuration (API keys, settings)."""

    def __init__(self, message: str, key_name: str = ""):
        self.key_name = key_name
        super().__init__(message, error_code="CONFIG_ERROR")
