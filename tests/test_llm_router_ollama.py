# Copyright 2024 SoftwareTeamFabrik contributors
# SPDX-License-Identifier: MIT
"""Tests for Ollama provider in LlmRouter."""

import json
import os
from unittest.mock import Mock, patch

import pytest

from factory.adapters.llm_router import (
    LlmRouter,
    LLMPermanentError,
    LLMTransientError,
)
from factory.core.competence import AgentProfile
from factory.core.token_budget import BudgetExceededError


class TestOllamaAdapter:
    """Tests for the _ollama method in LlmRouter."""

    @pytest.fixture
    def router(self):
        # Ensure OLLAMA_TIMEOUT is set to a valid value for tests.
        # Note: OLLAMA_API_KEY and OLLAMA_BASE_URL are intentionally NOT set here.
        # Tests that need specific values for these must set them explicitly via
        # their own patch.dict. This avoids fragile layering where an inner patch
        # might override the fixture's value unexpectedly.
        with patch.dict("os.environ", {"OLLAMA_TIMEOUT": "120"}, clear=False):
            router = LlmRouter()
            yield router

    @pytest.fixture
    def agent(self):
        return AgentProfile(
            name="test_agent",
            display_name="Test Agent",
            model="llama3.2",
            provider="ollama",
        )

    def test_successful_response(self, router, agent):
        """Test successful Ollama response with valid JSON."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": "Hello, world!"},
            "prompt_eval_count": 10,
            "eval_count": 20,
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            result = router._ollama(agent, "You are a bot", "Hello", "llama3.2")

        assert result.content == "Hello, world!"
        assert result.model == "llama3.2"
        assert result.provider == "ollama"
        assert result.input_tokens == 10
        assert result.output_tokens == 20

    def test_error_in_response_body(self, router, agent):
        """Test Ollama response with 'error' key (HTTP 200 but error).
        
        'model not found' is a permanent error (invalid model), so should
        raise LLMPermanentError without retrying.
        """
        mock_response = Mock()
        mock_response.json.return_value = {"error": "model not found"}
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(LLMPermanentError) as exc_info:
                router._ollama(agent, "You are a bot", "Hello", "llama3.2")

        # Should only be called once (no retries for permanent errors)
        assert mock_post.call_count == 1
        assert "permanent" in str(exc_info.value).lower()

    def test_missing_message_key(self, router, agent):
        """Test Ollama response missing 'message' key.
        
        This is treated as a transient error and will be retried.
        After max retries, should raise LLMTransientError.
        """
        mock_response = Mock()
        mock_response.json.return_value = {"some_other_key": "value"}
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(LLMTransientError) as exc_info:
                router._ollama(agent, "You are a bot", "Hello", "llama3.2")

        # Should be called 3 times (max retries)
        assert mock_post.call_count == 3
        assert "failed after" in str(exc_info.value).lower()

    def test_missing_content_key(self, router, agent):
        """Test Ollama response with 'message' but missing 'content'.
        
        This is treated as a transient error and will be retried.
        After max retries, should raise LLMTransientError.
        """
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"role": "assistant"}}
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(LLMTransientError) as exc_info:
                router._ollama(agent, "You are a bot", "Hello", "llama3.2")

        # Should be called 3 times (max retries)
        assert mock_post.call_count == 3
        assert "failed after" in str(exc_info.value).lower()

    def test_non_json_response(self, router, agent):
        """Test Ollama response with non-JSON body.
        
        This is treated as a transient error and will be retried.
        After max retries, should raise LLMTransientError.
        """
        import requests.exceptions
        
        mock_response = Mock()
        mock_response.text = "Not JSON"
        mock_response.raise_for_status = Mock()
        mock_response.json.side_effect = requests.exceptions.JSONDecodeError("Expecting value", "", 0)

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(LLMTransientError) as exc_info:
                router._ollama(agent, "You are a bot", "Hello", "llama3.2")

        # Should be called 3 times (max retries)
        assert mock_post.call_count == 3
        assert "failed after" in str(exc_info.value).lower()

    def test_http_error(self, router, agent):
        """Test HTTP error from Ollama.
        
        HTTP errors are caught and classified based on status code.
        A generic HTTPError without status info is treated as transient.
        """
        import requests
        
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("HTTP 500")

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            # HTTPError is caught and re-raised as LLMTransientError or LLMPermanentError
            # depending on classification. Generic HTTPError is transient.
            with pytest.raises((LLMTransientError, LLMPermanentError)):
                router._ollama(agent, "You are a bot", "Hello", "llama3.2")

    def test_missing_token_counts(self, router, agent):
        """Test Ollama response missing token count fields."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": "Hello"},
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            result = router._ollama(agent, "You are a bot", "Hello", "llama3.2")

        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_api_key_in_header(self, router, agent):
        """Test that OLLAMA_API_KEY is used in Authorization header."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": "Hello"},
            "prompt_eval_count": 5,
            "eval_count": 10,
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            with patch.dict("os.environ", {"OLLAMA_API_KEY": "test-key"}):
                mock_post.return_value = mock_response
                router._ollama(agent, "You are a bot", "Hello", "llama3.2")

        # Check the headers passed to requests.post
        call_args = mock_post.call_args
        headers = call_args.kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer test-key"

    def test_no_api_key_no_header(self, router, agent):
        """Test that no Authorization header when OLLAMA_API_KEY not set."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": "Hello"},
            "prompt_eval_count": 5,
            "eval_count": 10,
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            with patch.dict("os.environ", {"OLLAMA_API_KEY": ""}):
                mock_post.return_value = mock_response
                router._ollama(agent, "You are a bot", "Hello", "llama3.2")
                call_args = mock_post.call_args
                headers = call_args.kwargs.get("headers", {})
                assert "Authorization" not in headers

    def test_timeout_validation_invalid(self, agent):
        """Test that invalid OLLAMA_TIMEOUT raises RuntimeError on first Ollama call."""
        with patch.dict("os.environ", {"OLLAMA_TIMEOUT": "not-a-number"}):
            router = LlmRouter()
            with pytest.raises(RuntimeError) as exc_info:
                router._ollama(agent, "You are a bot", "Hello", "llama3.2")

        assert "must be an integer number of seconds" in str(exc_info.value)

    def test_timeout_validation_zero(self, agent):
        """Test that OLLAMA_TIMEOUT <= 0 raises RuntimeError on first Ollama call."""
        with patch.dict("os.environ", {"OLLAMA_TIMEOUT": "0"}):
            router = LlmRouter()
            with pytest.raises(RuntimeError) as exc_info:
                router._ollama(agent, "You are a bot", "Hello", "llama3.2")

        assert "must be positive" in str(exc_info.value)

    def test_timeout_validation_negative(self, agent):
        """Test that negative OLLAMA_TIMEOUT raises RuntimeError on first Ollama call."""
        with patch.dict("os.environ", {"OLLAMA_TIMEOUT": "-5"}):
            router = LlmRouter()
            with pytest.raises(RuntimeError) as exc_info:
                router._ollama(agent, "You are a bot", "Hello", "llama3.2")

        assert "must be positive" in str(exc_info.value)

    def test_timeout_validation_too_large(self, agent):
        """Test that OLLAMA_TIMEOUT > 600 raises RuntimeError on first Ollama call."""
        with patch.dict("os.environ", {"OLLAMA_TIMEOUT": "999999"}):
            router = LlmRouter()
            with pytest.raises(RuntimeError) as exc_info:
                router._ollama(agent, "You are a bot", "Hello", "llama3.2")

        assert "must not exceed 600 seconds" in str(exc_info.value)

    def test_custom_base_url(self, agent):
        """Test that OLLAMA_BASE_URL is used."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": "Hello"},
            "prompt_eval_count": 5,
            "eval_count": 10,
        }
        mock_response.raise_for_status = Mock()

        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://custom:8080", "OLLAMA_TIMEOUT": "120"}):
            router = LlmRouter()
            with patch("requests.post") as mock_post:
                mock_post.return_value = mock_response
                router._ollama(agent, "You are a bot", "Hello", "llama3.2")

            call_args = mock_post.call_args
            url = call_args.args[0] if call_args.args else call_args.kwargs.get("url")
            assert "http://custom:8080/api/chat" in str(url)

    def test_temperature_in_payload(self, router, agent):
        """Test that temperature is included in payload when set."""
        agent_with_temp = AgentProfile(
            name="test_agent",
            display_name="Test Agent",
            model="llama3.2",
            provider="ollama",
            temperature=0.7,
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": "Hello"},
            "prompt_eval_count": 5,
            "eval_count": 10,
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            router._ollama(agent_with_temp, "You are a bot", "Hello", "llama3.2")

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload.get("temperature") == 0.7

    def test_invalid_base_url_scheme(self, agent):
        """Test that invalid OLLAMA_BASE_URL scheme raises RuntimeError on first Ollama call."""
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "ftp://evil.example.com", "OLLAMA_TIMEOUT": "120"}):
            router = LlmRouter()
            with pytest.raises(RuntimeError) as exc_info:
                router._ollama(agent, "You are a bot", "Hello", "llama3.2")

        assert "must use http:// or https:// scheme" in str(exc_info.value)

    def test_invalid_base_url_missing_scheme(self, agent):
        """Test that OLLAMA_BASE_URL without scheme (e.g., 'localhost:11434') raises RuntimeError on first Ollama call."""
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "localhost:11434", "OLLAMA_TIMEOUT": "120"}):
            router = LlmRouter()
            with pytest.raises(RuntimeError) as exc_info:
                router._ollama(agent, "You are a bot", "Hello", "llama3.2")

        assert "must use http:// or https:// scheme" in str(exc_info.value)
        assert "localhost" in str(exc_info.value)

    def test_invalid_base_url_with_credentials(self, agent):
        """Test that OLLAMA_BASE_URL with credentials raises RuntimeError on first Ollama call."""
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://user:pass@localhost:11434", "OLLAMA_TIMEOUT": "120"}):
            router = LlmRouter()
            with pytest.raises(RuntimeError) as exc_info:
                router._ollama(agent, "You are a bot", "Hello", "llama3.2")

        assert "must not contain credentials" in str(exc_info.value)


class TestOllamaComplete:
    """Tests for the ollama_complete method (tool-calling path)."""

    @pytest.fixture
    def router(self):
        with patch.dict("os.environ", {"OLLAMA_TIMEOUT": "120"}, clear=False):
            router = LlmRouter()
            yield router

    @pytest.fixture
    def agent(self):
        return AgentProfile(
            name="test_agent",
            display_name="Test Agent",
            model="llama3.2",
            provider="ollama",
        )

    def test_ollama_complete_successful_response(self, router, agent):
        """Test successful ollama_complete response with valid JSON."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "Hello, world!",
                "tool_calls": [
                    {
                        "name": "test_tool",
                        "arguments": {"param": "value"}
                    }
                ]
            },
            "prompt_eval_count": 10,
            "eval_count": 20,
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            result = router.ollama_complete(
                agent, "You are a bot", "Hello",
                tools=[{"name": "test_tool", "description": "Test tool"}]
            )

        # Content should be JSON-encoded message
        content_data = json.loads(result.content)
        assert content_data["content"] == "Hello, world!"
        assert content_data["tool_calls"] == [
            {"name": "test_tool", "arguments": {"param": "value"}}
        ]
        assert result.model == "llama3.2"
        assert result.provider == "ollama"
        assert result.input_tokens == 10
        assert result.output_tokens == 20

    def test_ollama_complete_with_history(self, router, agent):
        """Test ollama_complete with conversation history."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Response with history"},
            "prompt_eval_count": 15,
            "eval_count": 25,
        }
        mock_response.raise_for_status = Mock()

        history = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
        ]

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            result = router.ollama_complete(
                agent, "You are a bot", "Next message", history=history
            )

        # Verify history was included in request
        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json", {})
        assert len(payload["messages"]) == 4  # system + history (2) + user
        assert payload["messages"][1] == history[0]
        assert payload["messages"][2] == history[1]

        assert result.content == json.dumps({"role": "assistant", "content": "Response with history"})

    def test_ollama_complete_tool_calls(self, router, agent):
        """Test ollama_complete with tool definitions."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "I need to use a tool",
                "tool_calls": [
                    {
                        "name": "search_web",
                        "arguments": {"query": "test query"}
                    }
                ]
            },
            "prompt_eval_count": 12,
            "eval_count": 18,
        }
        mock_response.raise_for_status = Mock()

        tools = [
            {"name": "search_web", "description": "Search the web"},
            {"name": "calculate", "description": "Perform calculations"}
        ]

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            result = router.ollama_complete(
                agent, "You are a bot", "Hello", tools=tools
            )

        # Verify tools were included in request
        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload["tools"] == tools

        # Verify response contains tool calls
        content_data = json.loads(result.content)
        assert content_data["tool_calls"] == [
            {"name": "search_web", "arguments": {"query": "test query"}}
        ]

    def test_ollama_complete_temperature(self, router, agent):
        """Test that temperature is passed through in ollama_complete."""
        agent_with_temp = AgentProfile(
            name="test_agent",
            display_name="Test Agent",
            model="llama3.2",
            provider="ollama",
            temperature=0.5,
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Response"},
            "prompt_eval_count": 8,
            "eval_count": 12,
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            router.ollama_complete(agent_with_temp, "You are a bot", "Hello")

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload.get("temperature") == 0.5

    def test_ollama_complete_error_response(self, router, agent):
        """Test ollama_complete with error response containing 'model not found'.

        The error message 'model not found' matches permanent-error keywords,
        so this should raise LLMPermanentError without retrying.
        """
        mock_response = Mock()
        mock_response.json.return_value = {"error": "model not found"}
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(LLMPermanentError) as exc_info:
                router.ollama_complete(agent, "You are a bot", "Hello")

        assert "Ollama error" in str(exc_info.value)

    def test_ollama_complete_empty_error_message(self, router, agent):
        """Test that empty/falsy error message is sanitized to '(unknown)'.

        An empty string error does not match permanent-error keywords, so it is
        classified as transient and retried. After max retries, raises
        LLMTransientError with '(unknown)' in the message.
        """
        mock_response = Mock()
        mock_response.json.return_value = {"error": ""}
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(LLMTransientError) as exc_info:
                router.ollama_complete(agent, "You are a bot", "Hello")

        assert "(unknown)" in str(exc_info.value)

    def test_ollama_complete_none_error_message(self, router, agent):
        """Test that None error message is sanitized to '(unknown)'.

        A None error does not match permanent-error keywords, so it is
        classified as transient and retried. After max retries, raises
        LLMTransientError with '(unknown)' in the message.
        """
        mock_response = Mock()
        mock_response.json.return_value = {"error": None}
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(LLMTransientError) as exc_info:
                router.ollama_complete(agent, "You are a bot", "Hello")

        assert "(unknown)" in str(exc_info.value)

    def test_ollama_complete_long_error_message_truncated(self, router, agent):
        """Test that long error message is truncated to 200 chars.

        A long string of 'x' characters does not match permanent-error keywords,
        so it is classified as transient and retried. After max retries, raises
        LLMTransientError. The key assertion is that the original 300-char
        message is not present in full in the exception.
        """
        long_error = "x" * 300
        mock_response = Mock()
        mock_response.json.return_value = {"error": long_error}
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(LLMTransientError) as exc_info:
                router.ollama_complete(agent, "You are a bot", "Hello")

        # The exception message should contain "Ollama error" but NOT the
        # full 300-char string (it should be truncated to 200 chars).
        error_str = str(exc_info.value)
        assert "Ollama error" in error_str
        assert long_error not in error_str

    def test_ollama_complete_missing_message(self, router, agent):
        """Test ollama_complete with missing message in response."""
        mock_response = Mock()
        mock_response.json.return_value = {"some_other_key": "value"}
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(LLMTransientError) as exc_info:
                router.ollama_complete(agent, "You are a bot", "Hello")

        assert "Unexpected Ollama response structure" in str(exc_info.value)

    def test_ollama_complete_message_is_string(self, router, agent):
        """Test ollama_complete with non-dict message (string) raises RuntimeError.

        The response structure validation requires message to be a dict,
        not a string. This ensures we catch malformed responses early.
        """
        mock_response = Mock()
        mock_response.json.return_value = {"message": "not a dict"}
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(LLMTransientError) as exc_info:
                router.ollama_complete(agent, "You are a bot", "Hello")

        assert "Unexpected Ollama response structure" in str(exc_info.value)

    def test_ollama_complete_message_is_none(self, router, agent):
        """Test ollama_complete with None message raises RuntimeError."""
        mock_response = Mock()
        mock_response.json.return_value = {"message": None}
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(LLMTransientError) as exc_info:
                router.ollama_complete(agent, "You are a bot", "Hello")

        assert "Unexpected Ollama response structure" in str(exc_info.value)

    def test_ollama_complete_message_is_list(self, router, agent):
        """Test ollama_complete with list message raises RuntimeError."""
        mock_response = Mock()
        mock_response.json.return_value = {"message": [{"role": "assistant"}]}
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(LLMTransientError) as exc_info:
                router.ollama_complete(agent, "You are a bot", "Hello")

        assert "Unexpected Ollama response structure" in str(exc_info.value)

    def test_ollama_complete_http_error(self, router, agent):
        """Test ollama_complete with non-200 HTTP response.

        When Ollama returns a non-200 status code, raw.ok is False,
        and the method raises RuntimeError with the status and body.
        The error message includes the status code which determines
        retry classification (e.g. 404 → permanent, 500 → transient).
        """
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            # 500 status code matches "500" server keyword → transient
            with pytest.raises(LLMTransientError):
                router.ollama_complete(agent, "You are a bot", "Hello")

    def test_ollama_complete_http_error_404(self, router, agent):
        """Test ollama_complete with 404 HTTP response is permanent."""
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            # 404 status code matches "404" permanent keyword
            with pytest.raises(LLMPermanentError):
                router.ollama_complete(agent, "You are a bot", "Hello")

    def test_ollama_complete_budget_exceeded(self, agent):
        """Test that ollama_complete raises BudgetExceededError when budget is exceeded."""
        from factory.core.token_budget import TokenBudgetManager

        mock_budget = Mock(spec=TokenBudgetManager)
        mock_budget.can_consume.return_value = (False, "Daily budget exceeded")

        with patch.dict("os.environ", {"OLLAMA_TIMEOUT": "120"}, clear=False):
            router = LlmRouter()
        router._budget_manager = mock_budget

        with pytest.raises(BudgetExceededError) as exc_info:
            router.ollama_complete(agent, "You are a bot", "Hello")

        assert "budget check failed" in str(exc_info.value)

    def test_ollama_complete_retry_on_transient_error(self, router, agent):
        """Test that ollama_complete retries on transient errors."""
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"role": "assistant", "content": "Success"}}
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            # First two calls fail, third succeeds
            mock_post.side_effect = [
                RuntimeError("Temporary failure"),
                RuntimeError("Temporary failure"),
                mock_response
            ]
            result = router.ollama_complete(agent, "You are a bot", "Hello")

        assert mock_post.call_count == 3
        assert result.content == json.dumps({"role": "assistant", "content": "Success"})

    def test_ollama_complete_timeout(self, router, agent):
        """Test that ollama_complete uses configured timeout."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Response"},
            "prompt_eval_count": 5,
            "eval_count": 10,
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            router.ollama_complete(agent, "You are a bot", "Hello")

        # Verify timeout was passed
        call_args = mock_post.call_args
        timeout = call_args.kwargs.get("timeout")
        assert timeout == 120  # From OLLAMA_TIMEOUT env var

    def test_ollama_complete_api_key_header(self, router, agent):
        """Test that ollama_complete includes API key in headers when set."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Response"},
            "prompt_eval_count": 5,
            "eval_count": 10,
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            with patch.dict("os.environ", {"OLLAMA_API_KEY": "secret-key"}):
                mock_post.return_value = mock_response
                router.ollama_complete(agent, "You are a bot", "Hello")

        call_args = mock_post.call_args
        headers = call_args.kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer secret-key"

    def test_ollama_complete_no_tools(self, router, agent):
        """Test ollama_complete without tools parameter."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Simple response"},
            "prompt_eval_count": 7,
            "eval_count": 11,
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            result = router.ollama_complete(agent, "You are a bot", "Hello")

        # Verify no tools in payload
        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json", {})
        assert "tools" not in payload

        content_data = json.loads(result.content)
        assert content_data["content"] == "Simple response"

    def test_ollama_complete_empty_history(self, router, agent):
        """Test ollama_complete with empty history list."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Response"},
            "prompt_eval_count": 6,
            "eval_count": 9,
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            result = router.ollama_complete(
                agent, "You are a bot", "Hello", history=[]
            )

        # Should still work with empty history
        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json", {})
        assert len(payload["messages"]) == 2  # system + user only

        content_data = json.loads(result.content)
        assert content_data["content"] == "Response"

    def test_ollama_complete_stream_false(self, router, agent):
        """Test that ollama_complete sets stream to False."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Response"},
            "prompt_eval_count": 5,
            "eval_count": 8,
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            router.ollama_complete(agent, "You are a bot", "Hello")

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload.get("stream") == False

    def test_ollama_complete_missing_token_counts(self, router, agent):
        """Test ollama_complete with missing token count fields."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Response"},
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            result = router.ollama_complete(agent, "You are a bot", "Hello")

        assert result.input_tokens == 0
        assert result.output_tokens == 0