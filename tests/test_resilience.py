"""Tests for the unified resilience classifier."""
from __future__ import annotations

import http.client

import pytest
import requests

from factory.core.resilience import (
    classify_llm_error,
    is_permanent_error,
    is_transient,
)


class TestClassifyLlmError:
    def test_context_overflow_keyword(self):
        retryable, ctx = classify_llm_error(Exception("prompt is too long: 200000 tokens"))
        assert retryable is True
        assert ctx is True

    def test_context_overflow_alt_keyword(self):
        retryable, ctx = classify_llm_error(Exception("context_length_exceeded"))
        assert retryable is True
        assert ctx is True

    def test_rate_limit(self):
        retryable, ctx = classify_llm_error(Exception("429 rate limit reached"))
        assert retryable is True
        assert ctx is False

    @pytest.mark.parametrize("msg", [
        "500 internal server error",
        "502 bad gateway",
        "503 service unavailable",
        "504 gateway timeout",
        "Connection error: refused",
        "Remote end closed connection without response",
        "RemoteDisconnected('Remote end closed connection')",
        "ChunkedEncodingError",
        "broken pipe",
        "request timed out",
    ])
    def test_server_keywords(self, msg):
        retryable, ctx = classify_llm_error(Exception(msg))
        assert retryable is True
        assert ctx is False

    @pytest.mark.parametrize("msg", [
        "401 Unauthorized",
        "403 Forbidden",
        "invalid api key",
        "404 not found",
        "Invalid model: gpt-9000",
    ])
    def test_permanent_keywords(self, msg):
        retryable, ctx = classify_llm_error(Exception(msg))
        assert retryable is False
        assert ctx is False

    def test_unknown_error_not_retryable(self):
        retryable, ctx = classify_llm_error(Exception("Some weird new failure"))
        assert retryable is False
        assert ctx is False

    def test_requests_connection_error_isinstance(self):
        retryable, ctx = classify_llm_error(requests.exceptions.ConnectionError())
        assert retryable is True
        assert ctx is False

    def test_requests_timeout_isinstance(self):
        retryable, ctx = classify_llm_error(requests.exceptions.Timeout())
        assert retryable is True
        assert ctx is False

    def test_remote_disconnected_isinstance(self):
        retryable, ctx = classify_llm_error(http.client.RemoteDisconnected())
        assert retryable is True
        assert ctx is False

    def test_permanent_takes_precedence_over_server_substring(self):
        # "400 Bad Request" must not be matched by the "400" substring as
        # transient; permanent classification wins.
        retryable, ctx = classify_llm_error(Exception("400 Bad Request: malformed payload"))
        assert retryable is False
        assert ctx is False


class TestIsTransient:
    def test_transient_true(self):
        assert is_transient(Exception("503 service unavailable")) is True

    def test_transient_false_for_permanent(self):
        assert is_transient(Exception("401 unauthorized")) is False

    def test_transient_false_for_unknown(self):
        assert is_transient(Exception("weird")) is False


class TestIsPermanentError:
    def test_permanent_true(self):
        assert is_permanent_error(Exception("403 Forbidden")) is True

    def test_permanent_false_for_transient(self):
        assert is_permanent_error(Exception("503")) is False

    def test_permanent_false_for_unknown(self):
        assert is_permanent_error(Exception("weird")) is False
