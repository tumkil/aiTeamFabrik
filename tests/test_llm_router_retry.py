"""Unit tests for LlmRouter retry mechanism and error handling."""
import pytest
from unittest.mock import Mock, patch
import requests

from factory.adapters.llm_router import (
    LlmRouter,
    LLMTimeoutError,
    LLMTransientError,
    LLMPermanentError,
)
from factory.core.competence import AgentProfile


def _stub_agent(provider: str = "stub", model: str = "stub") -> AgentProfile:
    return AgentProfile(
        name="test-agent",
        display_name="Test Agent",
        model=model,
        provider=provider,
        system_prompt="Test system prompt",
        tools=[],
    )


class TestRetryConfiguration:
    """Test retry configuration defaults and customization."""

    def test_default_retry_settings(self):
        """Router should have sensible default retry settings."""
        router = LlmRouter()
        assert router._max_retries == 3
        assert router._retry_base_delay == 1.0
        assert router._retry_max_delay == 60.0
        assert router._retry_jitter == 0.1

    def test_backoff_calculation(self):
        """Backoff should increase exponentially with jitter."""
        router = LlmRouter()
        
        # First attempt: base_delay * 2^0 = 1.0
        delay0 = router._calculate_backoff(0)
        assert 1.0 <= delay0 <= 1.1  # base + up to 10% jitter
        
        # Second attempt: base_delay * 2^1 = 2.0
        delay1 = router._calculate_backoff(1)
        assert 2.0 <= delay1 <= 2.2
        
        # Third attempt: base_delay * 2^2 = 4.0
        delay2 = router._calculate_backoff(2)
        assert 4.0 <= delay2 <= 4.4
        
        # Should cap at max_delay
        delay10 = router._calculate_backoff(10)
        assert delay10 <= 60.0 + 6.0  # max_delay + 10% jitter


class TestIsRetryableError:
    """Test error classification for retry logic."""

    def test_timeout_is_retryable(self):
        """Timeout errors should be retryable."""
        router = LlmRouter()
        timeout_exc = requests.exceptions.Timeout("Connection timed out")
        assert router._is_retryable_error(timeout_exc) is True
        
        llm_timeout = LLMTimeoutError("API timeout")
        assert router._is_retryable_error(llm_timeout) is True

    def test_connection_error_is_retryable(self):
        """Connection errors should be retryable."""
        router = LlmRouter()
        conn_exc = requests.exceptions.ConnectionError("Connection refused")
        assert router._is_retryable_error(conn_exc) is True

    def test_503_is_retryable(self):
        """503 Service Unavailable should be retryable."""
        router = LlmRouter()
        exc = Exception("503 Service Unavailable")
        assert router._is_retryable_error(exc) is True

    def test_502_is_retryable(self):
        """502 Bad Gateway should be retryable."""
        router = LlmRouter()
        exc = Exception("502 Bad Gateway")
        assert router._is_retryable_error(exc) is True

    def test_504_is_retryable(self):
        """504 Gateway Timeout should be retryable."""
        router = LlmRouter()
        exc = Exception("504 Gateway Timeout")
        assert router._is_retryable_error(exc) is True

    def test_429_is_retryable(self):
        """429 Rate Limit should be retryable."""
        router = LlmRouter()
        exc = Exception("429 Too Many Requests")
        assert router._is_retryable_error(exc) is True

    def test_401_is_not_retryable(self):
        """401 Unauthorized should NOT be retryable."""
        router = LlmRouter()
        exc = Exception("401 Unauthorized: Invalid API key")
        assert router._is_retryable_error(exc) is False

    def test_403_is_not_retryable(self):
        """403 Forbidden should NOT be retryable."""
        router = LlmRouter()
        exc = Exception("403 Forbidden: Access denied")
        assert router._is_retryable_error(exc) is False

    def test_invalid_model_is_not_retryable(self):
        """Invalid model errors should NOT be retryable."""
        router = LlmRouter()
        exc = Exception("Invalid model: model-not-found")
        assert router._is_retryable_error(exc) is False

    def test_unknown_error_is_retryable_by_default(self):
        """Unknown errors should be retryable by default (fail-safe)."""
        router = LlmRouter()
        exc = Exception("Some unknown error")
        assert router._is_retryable_error(exc) is True


class TestRetryBehavior:
    """Test actual retry behavior with mocked providers."""

    @patch('factory.adapters.llm_router.requests.post')
    def test_ollama_retries_on_timeout(self, mock_post):
        """Ollama should retry on timeout and succeed on second attempt."""
        mock_post.side_effect = [
            requests.exceptions.Timeout("First attempt timeout"),
            requests.exceptions.Timeout("Second attempt timeout"),
            Mock(ok=True, json=lambda: {
                "message": {"content": "Success!"},
                "prompt_eval_count": 10,
                "eval_count": 20,
            }),
        ]
        
        router = LlmRouter()
        agent = _stub_agent(provider="ollama", model="llama3")
        
        with patch.object(router, '_init_ollama_config'):
            router._ollama_initialized = True
            response = router._ollama(agent, "system", "prompt", "llama3")
        
        assert response.content == "Success!"
        assert mock_post.call_count == 3

    @patch('factory.adapters.llm_router.requests.post')
    def test_ollama_raises_after_max_retries(self, mock_post):
        """Ollama should raise LLMTimeoutError after max retries exhausted."""
        mock_post.side_effect = requests.exceptions.Timeout("Always times out")
        
        router = LlmRouter()
        agent = _stub_agent(provider="ollama", model="llama3")
        
        with patch.object(router, '_init_ollama_config'):
            router._ollama_initialized = True
            with pytest.raises(LLMTimeoutError) as exc_info:
                router._ollama(agent, "system", "prompt", "llama3")
        
        assert "timeout" in str(exc_info.value).lower()
        assert mock_post.call_count == 3

    @patch('factory.adapters.llm_router.requests.post')
    def test_ollama_does_not_retry_on_permanent_error(self, mock_post):
        """Ollama should NOT retry on permanent errors like 401."""
        mock_post.side_effect = Exception("401 Unauthorized: Invalid API key")
        
        router = LlmRouter()
        agent = _stub_agent(provider="ollama", model="llama3")
        
        with patch.object(router, '_init_ollama_config'):
            router._ollama_initialized = True
            with pytest.raises(LLMPermanentError) as exc_info:
                router._ollama(agent, "system", "prompt", "llama3")
        
        # Should only be called once (no retries for permanent errors)
        assert mock_post.call_count == 1
        assert "permanent" in str(exc_info.value).lower()

    @patch('factory.adapters.llm_router.requests.post')
    def test_ollama_retries_on_503(self, mock_post):
        """Ollama should retry on 503 errors."""
        mock_post.side_effect = [
            Exception("503 Service Unavailable"),
            Mock(ok=True, json=lambda: {
                "message": {"content": "Success after retry!"},
                "prompt_eval_count": 10,
                "eval_count": 20,
            }),
        ]
        
        router = LlmRouter()
        agent = _stub_agent(provider="ollama", model="llama3")
        
        with patch.object(router, '_init_ollama_config'):
            router._ollama_initialized = True
            response = router._ollama(agent, "system", "prompt", "llama3")
        
        assert response.content == "Success after retry!"
        assert mock_post.call_count == 2

    @patch('factory.adapters.llm_router.requests.post')
    def test_ollama_no_retry_on_401(self, mock_post):
        """Ollama should NOT retry on 401 Unauthorized."""
        # Create a mock response that raises HTTPError on raise_for_status()
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 401
        mock_response.text = "401 Unauthorized"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Client Error")
        mock_post.return_value = mock_response
        
        router = LlmRouter()
        agent = _stub_agent(provider="ollama", model="llama3")
        
        with patch.object(router, '_init_ollama_config'):
            router._ollama_initialized = True
            with pytest.raises(LLMPermanentError):
                router._ollama(agent, "system", "prompt", "llama3")
        
        # Should only be called once (no retries for permanent errors)
        assert mock_post.call_count == 1

    @patch('factory.adapters.llm_router.requests.post')
    def test_ollama_success_no_retry(self, mock_post):
        """Ollama should succeed without retry on first attempt."""
        mock_post.return_value = Mock(
            ok=True,
            json=lambda: {
                "message": {"content": "Success!"},
                "prompt_eval_count": 10,
                "eval_count": 20,
            }
        )
        
        router = LlmRouter()
        agent = _stub_agent(provider="ollama", model="llama3")
        
        with patch.object(router, '_init_ollama_config'):
            router._ollama_initialized = True
            response = router._ollama(agent, "system", "prompt", "llama3")
        
        assert response.content == "Success!"
        assert mock_post.call_count == 1


class TestErrorClasses:
    """Test custom error classes."""

    def test_llm_timeout_error(self):
        """LLMTimeoutError should be raised for timeout scenarios."""
        with pytest.raises(LLMTimeoutError) as exc_info:
            raise LLMTimeoutError("Request timed out after 30s")
        assert "timed out" in str(exc_info.value).lower()

    def test_llm_transient_error(self):
        """LLMTransientError should be raised for transient failures."""
        with pytest.raises(LLMTransientError) as exc_info:
            raise LLMTransientError("Service temporarily unavailable")
        assert "unavailable" in str(exc_info.value).lower()

    def test_llm_permanent_error(self):
        """LLMPermanentError should be raised for permanent failures."""
        with pytest.raises(LLMPermanentError) as exc_info:
            raise LLMPermanentError("Invalid API key")
        assert "api key" in str(exc_info.value).lower()

    def test_error_classes_are_distinct(self):
        """Error classes should be distinct types."""
        assert LLMTimeoutError is not LLMTransientError
        assert LLMTransientError is not LLMPermanentError
        assert LLMTimeoutError is not LLMPermanentError
        
        # All should be Exception subclasses
        assert issubclass(LLMTimeoutError, Exception)
        assert issubclass(LLMTransientError, Exception)
        assert issubclass(LLMPermanentError, Exception)
