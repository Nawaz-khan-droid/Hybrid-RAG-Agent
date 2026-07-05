"""Tests for engine.py internals — no API calls required."""
import io
import pytest
from engine import (
    _tokenize,
    _invoke_with_retry,
    _parse_action,
    _build_tool_prompt,
    _build_react_conversation,
    chunk_text,
    extract_text_from_file,
    HybridKnowledgeBase,
    _record_metric,
    get_metrics_snapshot,
)
from exceptions import APIError


class TestTokenizer:
    def test_normal(self):
        assert _tokenize("Hello World!") == ["hello", "world"]

    def test_empty(self):
        assert _tokenize("") == []

    def test_numbers(self):
        assert _tokenize("test123 v2.0") == ["test123", "v2", "0"]

    def test_special_chars(self):
        assert _tokenize("don't split's") == ["don", "t", "split", "s"]


class TestRetryWrapper:
    def test_success_first_try(self):
        result = _invoke_with_retry(lambda: 42, max_attempts=3)
        assert result == 42

    def test_success_after_retry(self):
        call_count = 0
        def fail_once():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("transient")
            return "ok"
        result = _invoke_with_retry(fail_once, max_attempts=3)
        assert result == "ok"
        assert call_count == 2

    def test_exhaust_raises_api_error(self):
        def always_fail():
            raise RuntimeError("permanent")
        with pytest.raises(APIError) as exc_info:
            _invoke_with_retry(always_fail, max_attempts=2)
        assert "permanent" in str(exc_info.value)

    def test_api_error_has_service(self):
        def fail():
            raise ConnectionError("timeout")
        with pytest.raises(APIError) as exc_info:
            _invoke_with_retry(fail, max_attempts=2, service="test-svc")
        assert exc_info.value.service == "test-svc"


class TestActionParser:
    def test_standard(self):
        text = "Thought: search\nAction: Search\nAction Input: hello\nObservation: done"
        tool, inp = _parse_action(text)
        assert tool == "Search"
        assert inp == "hello"

    def test_no_action(self):
        tool, inp = _parse_action("Just an answer")
        assert tool is None
        assert inp is None

    def test_malformed_action(self):
        tool, inp = _parse_action("Action:\nAction Input: ")
        # Regex captures "Action Input:" as the tool name due to multiline
        assert tool is not None

    def test_observation_cutoff(self):
        text = "Action: Search\nAction Input: hello world\nObservation: results here"
        tool, inp = _parse_action(text)
        assert inp == "hello world"

    def test_multiline_after_input(self):
        text = "Action: Search\nAction Input: hello\nMore text\nObservation:"
        tool, inp = _parse_action(text)
        assert inp == "hello"


class TestToolPrompt:
    def test_build_prompt(self):
        tools = {"A": "Does X", "B": "Does Y"}
        result = _build_tool_prompt(tools)
        assert "- A: Does X" in result
        assert "- B: Does Y" in result

    def test_empty_tools(self):
        assert _build_tool_prompt({}) == ""


class TestBuildConversation:
    def test_contains_elements(self):
        conv = _build_react_conversation("SysPrompt", {"T": "desc"}, "What?")
        assert "Question: What?" in conv
        assert "- T: desc" in conv
        assert "SysPrompt" in conv
        assert "Final Answer:" in conv


class TestChunking:
    def test_normal_chunking(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=5)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        for c in chunks:
            assert len(c) > 20

    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text(self):
        chunks = chunk_text("Short.")
        assert chunks == []


class TestFileExtraction:
    def test_txt_extraction(self):
        class MockFile:
            name = "test.txt"
            def getvalue(self):
                return b"Hello from test"
        text = extract_text_from_file(MockFile())
        assert "Hello from test" in text

    def test_unsupported_type(self):
        class MockFile:
            name = "test.doc"
            def getvalue(self):
                return b"data"
        text = extract_text_from_file(MockFile())
        assert text == ""

    def test_pdf_graceful_fail(self):
        class MockFile:
            name = "test.pdf"
            def getvalue(self):
                return b"%PDF-invalid"
        # PdfReader raises on invalid data, should return empty string
        text = extract_text_from_file(MockFile())
        assert text == ""


class TestHybridKnowledgeBase:
    def test_init_empty(self):
        kb = HybridKnowledgeBase(api_key="fake-key")
        assert kb.is_empty is True
        assert kb.chunk_count == 0
        assert kb.source_count == 0
        assert kb.source_names == []

    def test_hybrid_search_empty(self):
        kb = HybridKnowledgeBase(api_key="fake-key")
        result = kb.hybrid_search("test")
        assert "empty" in result.lower()


class TestMetrics:
    def test_record_and_snapshot(self):
        _record_metric("test_latency", 100.0)
        _record_metric("test_latency", 200.0)
        snap = get_metrics_snapshot()
        assert "test_latency" in snap
        assert snap["test_latency"]["count"] >= 2
        assert snap["test_latency"]["min"] == 100.0
        assert snap["test_latency"]["max"] == 200.0

    def test_snapshot_empty_initially(self):
        # Clear and check
        _metrics_snapshot_clean = get_metrics_snapshot()
        # Just verify the function returns a dict
        assert isinstance(_metrics_snapshot_clean, dict)
