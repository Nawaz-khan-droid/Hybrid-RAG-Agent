"""Tests for the custom exception hierarchy."""
import pytest
from exceptions import RAGError, APIError, InputBlockedError, ConfigurationError


class TestRAGError:
    def test_base_exception(self):
        err = RAGError("something went wrong")
        assert str(err) == "something went wrong"
        assert err.error_code == "UNKNOWN"
        assert err.details == ""

    def test_base_with_code(self):
        err = RAGError("custom", error_code="CUSTOM", details="detail text")
        assert err.error_code == "CUSTOM"
        assert err.details == "detail text"


class TestAPIError:
    def test_default_service(self):
        err = APIError("API failed")
        assert err.service == "unknown"
        assert err.status_code == 0
        assert err.error_code == "API_ERROR"

    def test_with_service_and_status(self):
        err = APIError("Gemini down", service="gemini", status_code=429)
        assert err.service == "gemini"
        assert err.status_code == 429

    def test_is_rag_error(self):
        err = APIError("fail")
        assert isinstance(err, RAGError)
        assert isinstance(err, Exception)


class TestInputBlockedError:
    def test_default_reason(self):
        err = InputBlockedError("Blocked")
        assert err.error_code == "INPUT_BLOCKED"
        assert err.details == ""

    def test_with_reason(self):
        err = InputBlockedError("Blocked", reason="prompt injection detected")
        assert err.details == "prompt injection detected"

    def test_is_rag_error(self):
        assert isinstance(InputBlockedError("x"), RAGError)


class TestConfigurationError:
    def test_default_key(self):
        err = ConfigurationError("Missing key")
        assert err.error_code == "CONFIG_ERROR"
        assert err.key_name == ""

    def test_with_key(self):
        err = ConfigurationError("Missing", key_name="GOOGLE_API_KEY")
        assert err.key_name == "GOOGLE_API_KEY"

    def test_is_rag_error(self):
        assert isinstance(ConfigurationError("x"), RAGError)
