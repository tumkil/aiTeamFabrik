# Copyright 2024 SoftwareTeamFabrik contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

from factory.core.competence import AgentProfile
from factory.core.token_budget import TokenBudgetManager, BudgetExceededError

logger = logging.getLogger(__name__)

# Fallback used when no system_prompt is set in the agent YAML
_DEFAULT_SYSTEM = (
    "You are a {display_name} agent in the SoftwareTeamFabrik autonomous development factory. "
    "Analyse the given task and respond with a clear, actionable plan."
)

# Anthropic tool definition for web search (built-in, no extra API key needed)
_WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
}

# Retry configuration
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds
_RETRY_MAX_DELAY = 60.0  # seconds
_RETRY_JITTER = 0.1  # random jitter factor (10%)


class LLMTimeoutError(Exception):
    """Raised when an LLM API call times out."""
    pass


class LLMTransientError(Exception):
    """Raised for transient LLM errors that may succeed on retry."""
    pass


class LLMPermanentError(Exception):
    """Raised for permanent LLM errors that should not be retried."""
    pass


@dataclass
class LlmResponse:
    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0


class LlmRouter:
    """Dispatches prompts to the correct LLM provider based on the agent profile.
    
    Integrates with TokenBudgetManager for budget enforcement and tracking.
    """
    
    def __init__(self, config_path: Path = Path("config/factory.yml"),
                 usage_path: Path = Path("config/token_usage.yml")):
        self._budget_manager = None
        self._ollama_initialized = False
        self._ollama_timeout: int = 120
        self._ollama_base_url: str = "http://localhost:11434"
        self._max_retries: int = _MAX_RETRIES
        self._retry_base_delay: float = _RETRY_BASE_DELAY
        self._retry_max_delay: float = _RETRY_MAX_DELAY
        self._retry_jitter: float = _RETRY_JITTER
        # Instance-level lock for thread-safe lazy initialization of Ollama config
        self._ollama_init_lock = threading.Lock()

        if os.environ.get("FACTORY_IGNORE_BUDGET", "").lower() in ("1", "true", "yes"):
            logger.debug("FACTORY_IGNORE_BUDGET set — budget enforcement disabled.")
            return
        try:
            self._budget_manager = TokenBudgetManager(config_path, usage_path)
        except FileNotFoundError:
            logger.debug("No budget config found; running in legacy (unbudgeted) mode.")
        except (ValueError, KeyError, OSError) as exc:
            logger.warning(
                "Budget configuration error (%s): %s. Running without budget enforcement.",
                type(exc).__name__, exc,
            )

    def _estimate_tokens(self, text: str) -> int:
        """Rough pre-flight token estimate: 1 token ≈ 3 characters.

        Estimates input tokens only (system + prompt). Output tokens are not
        predicted; a large response (e.g. a full code review) can add 2–8k
        tokens on top of this estimate. Budget limits should include headroom
        for output: e.g. a reviewer with an 80k daily limit can handle roughly
        8–10 large reviews before the actual post-call consume() tips it over.
        """
        if not text:
            return 0
        return len(text) // 3

    def _is_retryable_error(self, exc: Exception) -> bool:
        """Determine if an error is retryable (transient) or permanent.

        Defers classification to :mod:`factory.core.resilience` for shared
        keyword/SDK-type matching across the codebase, but preserves this
        module's historical fail-safe default: unknown errors are treated as
        retryable so a transient bug never becomes a hard failure on first
        encounter.
        """
        from factory.core.resilience import classify_llm_error, is_permanent_error

        # Timeout errors are always retryable (LLMTimeoutError is internal).
        if isinstance(exc, (requests.exceptions.Timeout, LLMTimeoutError)):
            return True

        retryable, _ = classify_llm_error(exc)
        if retryable:
            return True
        if is_permanent_error(exc):
            return False

        # Unknown error → retry once rather than fail permanently.
        return True

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        import random
        delay = self._retry_base_delay * (2 ** attempt)
        delay = min(delay, self._retry_max_delay)
        jitter = delay * self._retry_jitter * random.random()
        return delay + jitter

    def complete(self, agent: AgentProfile, system: str, prompt: str) -> LlmResponse:
        """Complete a prompt using the appropriate LLM provider.

        Budget enforcement is two-phase:
        1. Pre-check (read-only): estimates input tokens via can_consume().
           can_consume() never deducts from the budget — it is a gate only.
           Raises BudgetExceededError in strict mode when over budget.
           Logs a warning in warn mode and allows the call.
        2. Post-call (write): records actual input+output tokens via consume().
           Called in a finally block. If the provider raises before returning,
           the estimated input tokens are recorded as a fallback so budget
           accuracy is maintained even on failed calls.
        """
        resolved_system = (
            agent.system_prompt.strip()
            or system
            or _DEFAULT_SYSTEM.format(display_name=agent.display_name)
        )

        # Pre-flight budget check (read-only — does not deduct tokens).
        # estimated_tokens stays 0 when there is no budget manager; the
        # finally block's elif branch is then dead code by design.
        estimated_tokens = 0
        budget_exceeded = False
        if self._budget_manager is not None:
            estimated_tokens = self._estimate_tokens(resolved_system + prompt)
            allowed, reason = self._budget_manager.can_consume(agent.name, estimated_tokens)
            if not allowed:
                logger.warning("Budget exceeded for agent '%s': %s", agent.name, reason)
                budget_exceeded = True
                raise BudgetExceededError(
                    f"Agent '{agent.name}' budget check failed: {reason}"
                )
            elif allowed and reason:  # warn mode: allowed but over threshold
                logger.warning("Budget warning for agent '%s': %s", agent.name, reason)

        provider = agent.provider.lower()
        # Resolve model in case it's a dict with provider-specific models
        model = agent.resolve_model()
        response: LlmResponse | None = None
        try:
            if provider == "anthropic":
                response = self._anthropic(agent, resolved_system, prompt, model)
            elif provider == "mistral":
                response = self._mistral(agent, resolved_system, prompt, model)
            elif provider == "google":
                response = self._google(agent, resolved_system, prompt, model)
            elif provider == "stub":
                response = self._stub(agent, prompt, model)
            elif provider == "ollama":
                response = self._ollama(agent, resolved_system, prompt, model)
            else:
                raise ValueError(f"Unknown provider '{provider}' for agent '{agent.name}'")
        finally:
            if self._budget_manager is not None and not budget_exceeded:
                # Always record when we got a response (even 0-token stub calls).
                # When the provider raised before returning, fall back to the
                # pre-flight estimate so the failed request is still accounted for.
                # Note: recorded_model uses the resolved model (from agent.resolve_model())
                # rather than agent.model, since that's what was actually used/attempted.
                in_tok, out_tok, recorded_model = 0, 0, model  # safe defaults
                if response is not None:
                    in_tok, out_tok, recorded_model = (
                        response.input_tokens, response.output_tokens, response.model
                    )
                elif estimated_tokens > 0:
                    in_tok, out_tok, recorded_model = estimated_tokens, 0, model
                if response is not None or estimated_tokens > 0:
                    try:
                        self._budget_manager.consume(
                            agent=agent.name,
                            input_tokens=in_tok,
                            output_tokens=out_tok,
                            model=recorded_model,
                        )
                    except Exception as exc:
                        logger.warning("Failed to record token consumption: %s", exc)

        return response

    # ------------------------------------------------------------------
    def _anthropic(self, agent: AgentProfile, system: str, prompt: str, model: str) -> LlmResponse:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")

        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        tools = []
        if "web_search" in (agent.tools or []):
            tools.append(_WEB_SEARCH_TOOL)

        kwargs: dict = dict(
            model=model,
            max_tokens=4096,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
        if agent.temperature is not None:
            temperature = agent.temperature
            if temperature > 1.0:
                logger.warning(
                    "Agent '%s' temperature %.2f exceeds Anthropic maximum (1.0); clamping.",
                    agent.name, temperature,
                )
                temperature = 1.0
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = tools

        # Retry logic for transient errors
        last_exception = None
        for attempt in range(self._max_retries):
            try:
                message = client.messages.create(**kwargs)

                # Collect all text blocks (tool use may interleave them)
                content = "\n\n".join(
                    block.text for block in message.content if hasattr(block, "text")
                )

                return LlmResponse(
                    content=content,
                    model=model,
                    provider="anthropic",
                    input_tokens=message.usage.input_tokens,
                    output_tokens=message.usage.output_tokens,
                )
            except Exception as exc:
                last_exception = exc
                if self._is_retryable_error(exc) and attempt < self._max_retries - 1:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        "Anthropic API error (attempt %d/%d): %s. Retrying in %.1f seconds...",
                        attempt + 1, self._max_retries, exc, delay
                    )
                    time.sleep(delay)
                else:
                    # Non-retryable error or exhausted retries
                    if isinstance(exc, (requests.exceptions.Timeout,)):
                        raise LLMTimeoutError(f"Anthropic API timeout after {self._max_retries} attempts: {exc}") from exc
                    elif self._is_retryable_error(exc):
                        raise LLMTransientError(f"Anthropic API failed after {self._max_retries} attempts: {exc}") from exc
                    else:
                        raise LLMPermanentError(f"Anthropic API permanent error: {exc}") from exc
        
        # Should not reach here, but just in case
        raise LLMTransientError(f"Anthropic API failed after {self._max_retries} attempts: {last_exception}")

    def _mistral(self, agent: AgentProfile, system: str, prompt: str, model: str) -> LlmResponse:
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set.")

        from mistralai import Mistral

        client = Mistral(api_key=api_key)
        kwargs: dict = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        if agent.temperature is not None:
            temperature = agent.temperature
            if temperature > 1.0:
                logger.warning(
                    "Agent '%s' temperature %.2f exceeds Mistral maximum (1.0); clamping.",
                    agent.name, temperature,
                )
                temperature = 1.0
            kwargs["temperature"] = temperature
        

        # Retry mechanism for transient errors (expanded beyond just 503)
        last_exception = None
        
        for attempt in range(self._max_retries):
            try:
                response = client.chat.complete(**kwargs)
                return LlmResponse(
                    content=response.choices[0].message.content,
                    model=model,
                    provider="mistral",
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                )
            except Exception as exc:
                last_exception = exc
                if self._is_retryable_error(exc) and attempt < self._max_retries - 1:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        "Mistral API error (attempt %d/%d): %s. Retrying in %.1f seconds...",
                        attempt + 1, self._max_retries, exc, delay
                    )
                    time.sleep(delay)
                else:
                    # Non-retryable error or exhausted retries
                    if isinstance(exc, (requests.exceptions.Timeout,)):
                        raise LLMTimeoutError(f"Mistral API timeout after {self._max_retries} attempts: {exc}") from exc
                    elif self._is_retryable_error(exc):
                        raise LLMTransientError(f"Mistral API failed after {self._max_retries} attempts: {exc}") from exc
                    else:
                        raise LLMPermanentError(f"Mistral API permanent error: {exc}") from exc
        
        # Should not reach here, but just in case
        raise LLMTransientError(f"Mistral API failed after {self._max_retries} attempts: {last_exception}")

    def _google(self, agent: AgentProfile, system: str, prompt: str, model: str) -> LlmResponse:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set.")

        import google.generativeai as genai

        genai.configure(api_key=api_key)

        # Map model name to the correct generation config
        generation_config = {
            "temperature": agent.temperature if agent.temperature is not None else 0.9,
            "max_output_tokens": 4096,
        }
        if generation_config["temperature"] > 1.0:
            logger.warning(
                "Agent '%s' temperature %.2f exceeds Google maximum (1.0); clamping.",
                agent.name, generation_config["temperature"],
            )
            generation_config["temperature"] = 1.0

        # Initialize the model
        try:
            gemini_model = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
            )
        except Exception as exc:
            raise LLMPermanentError(f"Failed to initialize Google Gemini model '{model}': {exc}")

        # Combine system and prompt into a single message
        full_prompt = f"{system}\n\n{prompt}"

        # Retry logic for transient errors
        last_exception = None
        for attempt in range(self._max_retries):
            try:
                response = gemini_model.generate_content([full_prompt])
                
                # Extract content and token usage
                content = response.text if hasattr(response, 'text') else str(response)
                
                # Google's response object may not always have usage_metadata
                input_tokens = 0
                output_tokens = 0
                if hasattr(response, 'usage_metadata'):
                    input_tokens = response.usage_metadata.prompt_token_count
                    output_tokens = response.usage_metadata.candidates_token_count

                return LlmResponse(
                    content=content,
                    model=model,
                    provider="google",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            except Exception as exc:
                last_exception = exc
                if self._is_retryable_error(exc) and attempt < self._max_retries - 1:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        "Google API error (attempt %d/%d): %s. Retrying in %.1f seconds...",
                        attempt + 1, self._max_retries, exc, delay
                    )
                    time.sleep(delay)
                else:
                    # Non-retryable error or exhausted retries
                    if isinstance(exc, (requests.exceptions.Timeout,)):
                        raise LLMTimeoutError(f"Google API timeout after {self._max_retries} attempts: {exc}") from exc
                    elif self._is_retryable_error(exc):
                        raise LLMTransientError(f"Google API failed after {self._max_retries} attempts: {exc}") from exc
                    else:
                        raise LLMPermanentError(f"Google API permanent error: {exc}") from exc
        
        # Should not reach here, but just in case
        raise LLMTransientError(f"Google API failed after {self._max_retries} attempts: {last_exception}")

    def _stub(self, agent: AgentProfile, prompt: str, model: str) -> LlmResponse:
        tools_note = f" (tools: {', '.join(agent.tools)})" if agent.tools else ""
        return LlmResponse(
            content=(
                f"[STUB] Agent '{agent.name}'{tools_note} would respond to: {prompt[:80]}..."
            ),
            model=model,
            provider="stub",
        )

    def ollama_complete(
        self,
        agent: AgentProfile,
        system: str,
        prompt: str,
        tools: list | None = None,
        history: list | None = None,
    ) -> "LlmResponse":
        """Call Ollama with tool-calling support and conversation history.

        Returns LlmResponse where content is the JSON-encoded assistant message,
        so callers can extract both text content and tool_calls from it.
        """
        import json
        import requests as _requests

        self._init_ollama_config()

        model = agent.resolve_model()

        # Pre-flight budget check (read-only — does not deduct tokens).
        estimated_tokens = 0
        budget_exceeded = False
        if self._budget_manager is not None:
            estimated_tokens = self._estimate_tokens(system + prompt)
            allowed, reason = self._budget_manager.can_consume(agent.name, estimated_tokens)
            if not allowed:
                logger.warning("Budget exceeded for agent '%s': %s", agent.name, reason)
                budget_exceeded = True
                raise BudgetExceededError(
                    f"Agent '{agent.name}' budget check failed: {reason}"
                )
            elif allowed and reason:  # warn mode: allowed but over threshold
                logger.warning("Budget warning for agent '%s': %s", agent.name, reason)

        api_key = os.environ.get("OLLAMA_API_KEY", "")
        url = f"{self._ollama_base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        messages: list = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        payload: dict = {"model": model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
        if agent.temperature is not None:
            payload["temperature"] = min(agent.temperature, 1.0)

        # Retry logic for transient errors
        last_exception = None
        response = None
        try:
            for attempt in range(self._max_retries):
                try:
                    raw = _requests.post(url, headers=headers, json=payload, timeout=self._ollama_timeout)
                    if not raw.ok:
                        raise RuntimeError(
                            f"Ollama {raw.status_code}: {raw.text[:400]}"
                        )
                    data = raw.json()
                    if "error" in data:
                        # Sanitize error message to avoid exposing sensitive data
                        error_msg = str(data["error"])[:200] if data["error"] else "(unknown)"
                        raise RuntimeError(f"Ollama error: {error_msg}")

                    # Validate response structure
                    message = data.get("message")
                    if not isinstance(message, dict):
                        data_str = str(data)[:200] if data else "(empty)"
                        logger.debug("Unexpected Ollama response structure: %s", data_str)
                        raise RuntimeError("Unexpected Ollama response structure. Check debug logs for details.")

                    response = LlmResponse(
                        content=json.dumps(message),
                        model=model,
                        provider="ollama",
                        input_tokens=data.get("prompt_eval_count") or 0,
                        output_tokens=data.get("eval_count") or 0,
                    )
                    return response
                except Exception as exc:
                    last_exception = exc
                    if self._is_retryable_error(exc) and attempt < self._max_retries - 1:
                        delay = self._calculate_backoff(attempt)
                        logger.warning(
                            "Ollama API error (attempt %d/%d): %s. Retrying in %.1f seconds...",
                            attempt + 1, self._max_retries, exc, delay
                        )
                        time.sleep(delay)
                    else:
                        # Non-retryable error or exhausted retries
                        if isinstance(exc, (requests.exceptions.Timeout,)):
                            raise LLMTimeoutError(f"Ollama API timeout after {self._max_retries} attempts: {exc}") from exc
                        elif self._is_retryable_error(exc):
                            raise LLMTransientError(f"Ollama API failed after {self._max_retries} attempts: {exc}") from exc
                        else:
                            raise LLMPermanentError(f"Ollama API permanent error: {exc}") from exc
            
            # Should not reach here, but just in case
            raise LLMTransientError(f"Ollama API failed after {self._max_retries} attempts: {last_exception}")
        finally:
            # Record token consumption after the call (success or failure),
            # but only when the budget pre-flight check passed.
            if self._budget_manager is not None and not budget_exceeded:
                in_tok, out_tok, recorded_model = 0, 0, model
                if response is not None:
                    in_tok, out_tok, recorded_model = (
                        response.input_tokens, response.output_tokens, response.model
                    )
                elif estimated_tokens > 0:
                    in_tok, out_tok, recorded_model = estimated_tokens, 0, model
                if response is not None or estimated_tokens > 0:
                    try:
                        self._budget_manager.consume(
                            agent=agent.name,
                            input_tokens=in_tok,
                            output_tokens=out_tok,
                            model=recorded_model,
                        )
                    except Exception as exc:
                        logger.warning("Failed to record token consumption: %s", exc)

    def anthropic_complete(
        self,
        agent: AgentProfile,
        system: str,
        messages: list,
        tools: list | None = None,
        max_tokens: int = 8192,
    ):
        """Call Anthropic with tool-calling support and conversation history.

        Returns the raw anthropic Message object so callers can inspect the
        content blocks (text + tool_use) directly. Use this for agent loops;
        use complete() for one-shot text completions.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")

        import anthropic

        model = agent.resolve_model()

        # Pre-flight budget check (read-only — does not deduct tokens).
        estimated_tokens = 0
        budget_exceeded = False
        if self._budget_manager is not None:
            estimated_tokens = self._estimate_tokens(
                system + "".join(
                    m.get("content", "") if isinstance(m.get("content"), str) else ""
                    for m in messages
                )
            )
            allowed, reason = self._budget_manager.can_consume(agent.name, estimated_tokens)
            if not allowed:
                logger.warning("Budget exceeded for agent '%s': %s", agent.name, reason)
                budget_exceeded = True
                raise BudgetExceededError(
                    f"Agent '{agent.name}' budget check failed: {reason}"
                )
            elif allowed and reason:  # warn mode: allowed but over threshold
                logger.warning("Budget warning for agent '%s': %s", agent.name, reason)

        client = anthropic.Anthropic(api_key=api_key)

        kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        )
        if tools:
            kwargs["tools"] = tools
        if agent.temperature is not None:
            kwargs["temperature"] = min(agent.temperature, 1.0)

        # Retry logic for transient errors
        last_exception = None
        message = None
        try:
            for attempt in range(self._max_retries):
                try:
                    message = client.messages.create(**kwargs)
                    return message
                except Exception as exc:
                    last_exception = exc
                    if self._is_retryable_error(exc) and attempt < self._max_retries - 1:
                        delay = self._calculate_backoff(attempt)
                        logger.warning(
                            "Anthropic API error (attempt %d/%d): %s. Retrying in %.1f seconds...",
                            attempt + 1, self._max_retries, exc, delay
                        )
                        time.sleep(delay)
                    else:
                        # Non-retryable error or exhausted retries
                        if isinstance(exc, (requests.exceptions.Timeout,)):
                            raise LLMTimeoutError(f"Anthropic API timeout after {self._max_retries} attempts: {exc}") from exc
                        elif self._is_retryable_error(exc):
                            raise LLMTransientError(f"Anthropic API failed after {self._max_retries} attempts: {exc}") from exc
                        else:
                            raise LLMPermanentError(f"Anthropic API permanent error: {exc}") from exc
            
            # Should not reach here, but just in case
            raise LLMTransientError(f"Anthropic API failed after {self._max_retries} attempts: {last_exception}")
        finally:
            # Record token consumption after the call (success or failure),
            # but only when the budget pre-flight check passed.
            if self._budget_manager is not None and not budget_exceeded:
                in_tok, out_tok = 0, 0
                if message is not None:
                    in_tok = message.usage.input_tokens
                    out_tok = message.usage.output_tokens
                elif estimated_tokens > 0:
                    in_tok, out_tok = estimated_tokens, 0
                if message is not None or estimated_tokens > 0:
                    try:
                        self._budget_manager.consume(
                            agent=agent.name,
                            input_tokens=in_tok,
                            output_tokens=out_tok,
                            model=model,
                        )
                    except Exception as exc:
                        logger.warning("Failed to record token consumption: %s", exc)

    def _init_ollama_config(self) -> None:
        """Lazy initialization of Ollama configuration. Called on first Ollama call.
        
        Thread-safe via instance-level _ollama_init_lock. Uses _ollama_initialized
        flag to ensure exactly-one initialization per instance.
        """
        if self._ollama_initialized:
            return
        
        with self._ollama_init_lock:
            # Double-check after acquiring lock
            if self._ollama_initialized:
                return
            
            # Parse timeout from env, using instance defaults as fallback
            _raw_timeout = os.environ.get("OLLAMA_TIMEOUT", str(self._ollama_timeout))
            try:
                self._ollama_timeout = int(_raw_timeout)
            except ValueError:
                raise RuntimeError(
                    f"OLLAMA_TIMEOUT must be an integer number of seconds, got: {_raw_timeout!r}"
                )
            if self._ollama_timeout <= 0:
                raise RuntimeError(
                    f"OLLAMA_TIMEOUT must be positive, got: {self._ollama_timeout}"
                )
            if self._ollama_timeout > 600:
                raise RuntimeError(
                    f"OLLAMA_TIMEOUT must not exceed 600 seconds (10 minutes), got: {self._ollama_timeout}"
                )
            
            # Parse and validate base URL from env, using instance defaults as fallback
            raw_base_url = os.environ.get("OLLAMA_BASE_URL", self._ollama_base_url)
            parsed = urlparse(raw_base_url)
            # Validate scheme - block file://, gopher://, ftp:// and other non-HTTP schemes
            # Only http:// and https:// are allowed to prevent SSRF
            if parsed.scheme not in ("http", "https"):
                scheme_display = parsed.scheme if parsed.scheme else "(none)"
                raise RuntimeError(
                    f"OLLAMA_BASE_URL must use http:// or https:// scheme, got: {scheme_display}"
                )
            # Enforce no credentials in URL to prevent accidental exposure
            if parsed.username or parsed.password:
                raise RuntimeError(
                    "OLLAMA_BASE_URL must not contain credentials. "
                    "Use OLLAMA_API_KEY for authentication instead."
                )
            # Normalize: store base URL without trailing slash for consistent concatenation
            self._ollama_base_url = raw_base_url.rstrip('/')
            
            self._ollama_initialized = True

    def _ollama(self, agent: AgentProfile, system: str, prompt: str, model: str) -> LlmResponse:
        # Lazy initialize Ollama config on first use (thread-safe)
        self._init_ollama_config()
        
        api_key = os.environ.get("OLLAMA_API_KEY", "")
        if not api_key:
            logger.debug("OLLAMA_API_KEY not set; sending unauthenticated request to Ollama.")
        
        # Use validated and normalized base_url
        url = f"{self._ollama_base_url}/api/chat"

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }

        if agent.temperature is not None:
            temperature = agent.temperature
            if temperature > 1.0:
                logger.warning(
                    "Agent '%s' temperature %.2f exceeds Ollama maximum (1.0); clamping.",
                    agent.name, temperature,
                )
                temperature = 1.0
            payload["temperature"] = temperature

        # Retry logic for transient errors
        last_exception = None
        for attempt in range(self._max_retries):
            try:
                # Use instance-level timeout to avoid repeated env var lookups and parsing.
                # Runtime changes require router reconstruction.
                timeout = self._ollama_timeout
                
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()

                try:
                    data = response.json()
                except (requests.exceptions.JSONDecodeError, ValueError) as exc:
                    # Truncate response text to avoid exposing sensitive data in exceptions
                    truncated_text = response.text[:200] if response.text else "(empty)"
                    logger.debug("Ollama returned non-JSON response: %s", truncated_text)
                    raise RuntimeError(
                        "Ollama returned non-JSON response. Check debug logs for details."
                    ) from exc

                # Handle Ollama error responses (HTTP 200 with "error" key)
                if "error" in data:
                    # Sanitize error message to avoid exposing sensitive data
                    error_msg = str(data["error"])[:200] if data["error"] else "(unknown)"
                    raise RuntimeError(f"Ollama error: {error_msg}")

                # Ensure message and content exist
                message = data.get("message")
                if not isinstance(message, dict):
                    # Truncate data to avoid exposing sensitive content in exceptions
                    data_str = str(data)[:200] if data else "(empty)"
                    logger.debug("Unexpected Ollama response structure: %s", data_str)
                    raise RuntimeError("Unexpected Ollama response structure. Check debug logs for details.")
                
                content = message.get("content")
                if not isinstance(content, str):
                    # Truncate data to avoid exposing sensitive content in exceptions
                    data_str = str(data)[:200] if data else "(empty)"
                    logger.debug("Missing or non-string 'content' in Ollama response: %s", data_str)
                    raise RuntimeError("Missing or non-string 'content' in Ollama response. Check debug logs for details.")
                
                # Empty string content is valid (e.g., after stop sequence)
                if not content:
                    logger.debug("Ollama returned empty content in response")

                # Ollama only populates these fields when stream: false (which we set above)
                input_tokens = data.get("prompt_eval_count")
                output_tokens = data.get("eval_count")
                if input_tokens is None or output_tokens is None:
                    logger.debug(
                        "Ollama response missing one or more token count fields: "
                        "prompt_eval_count=%s, eval_count=%s",
                        input_tokens, output_tokens,
                    )
                return LlmResponse(
                    content=content,
                    model=model,
                    provider="ollama",
                    input_tokens=input_tokens or 0,
                    output_tokens=output_tokens or 0,
                )
            except Exception as exc:
                last_exception = exc
                if self._is_retryable_error(exc) and attempt < self._max_retries - 1:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        "Ollama API error (attempt %d/%d): %s. Retrying in %.1f seconds...",
                        attempt + 1, self._max_retries, exc, delay
                    )
                    time.sleep(delay)
                else:
                    # Non-retryable error or exhausted retries
                    if isinstance(exc, (requests.exceptions.Timeout,)):
                        raise LLMTimeoutError(f"Ollama API timeout after {self._max_retries} attempts: {exc}") from exc
                    elif self._is_retryable_error(exc):
                        raise LLMTransientError(f"Ollama API failed after {self._max_retries} attempts: {exc}") from exc
                    else:
                        raise LLMPermanentError(f"Ollama API permanent error: {exc}") from exc
        
        # Should not reach here, but just in case
        raise LLMTransientError(f"Ollama API failed after {self._max_retries} attempts: {last_exception}")