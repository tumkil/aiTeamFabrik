"""Unified transient-error classification for SoftwareTeamFabrik.

This module is the single source of truth for deciding whether an exception
raised by an LLM provider, an HTTP client, or the GitLab SDK is *transient*
(safe to retry) or *permanent* (must surface to the caller).

Three call sites historically duplicated this logic:

- ``factory.core.execution_engine._classify_llm_error``
- ``factory.adapters.llm_router._is_retryable_error``
- ``factory.core.monitor._is_transient_error``

They have been collapsed onto :func:`classify_llm_error`.

Note: ``factory.adapters.gitlab_client.retry_with_backoff`` is a *separate*
abstraction (exponential backoff that retries any exception type passed to
it). It is not affected by this module.
"""

from __future__ import annotations

from typing import Tuple

# Substrings (case-insensitive) that indicate the LLM context window /
# input token limit was exceeded. These are recoverable: trim history and
# retry without delay.
_CTX_KEYWORDS: tuple[str, ...] = (
    "context_length_exceeded",
    "context window",
    "too many tokens",
    "prompt is too long",
    "maximum context length",
    "input is too long",
    "tokens exceed",
    "reduce the length",
)

# Rate-limit signals — wait, then retry.
_RATE_KEYWORDS: tuple[str, ...] = (
    "429",
    "rate limit",
    "ratelimit",
    "too many requests",
    "quota exceeded",
)

# Transient server / network errors — wait, then retry.
_SERVER_KEYWORDS: tuple[str, ...] = (
    "500", "502", "503", "504",
    "internal server error",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    "connection error",
    "connection refused",
    "connection reset",
    "connection aborted",
    "broken pipe",
    "remote end closed connection",
    "remotedisconnected",
    "chunkedencodingerror",
    "incompleteread",
    "timeout",
    "timed out",
    "network error",
    "temporary failure",
    "transient",
)

# Permanent failure signals — never retry.
_PERMANENT_KEYWORDS: tuple[str, ...] = (
    "401", "unauthorized", "invalid api key", "authentication",
    "403", "forbidden", "access denied",
    "400", "bad request", "invalid model", "model not found",
    "404", "not found",
)


def classify_llm_error(exc: Exception) -> Tuple[bool, bool]:
    """Classify an exception for retry policy.

    Returns ``(retryable, context_overflow)``:

    - ``retryable=True``         — the call may succeed if retried.
    - ``context_overflow=True``  — history must be trimmed before retrying
      (no delay needed); always implies ``retryable=True``.

    Order of precedence:

    1. Context-overflow keywords (most specific recovery action).
    2. Permanent-failure keywords (auth/not-found/bad-request — never retry).
    3. Rate-limit keywords.
    4. Server/network keywords.
    5. SDK-specific exception types (``requests``, ``http.client``,
       ``anthropic``).
    6. Default: ``(False, False)`` — surface the error.
    """
    msg = str(exc).lower()

    # 1. Context-overflow — recoverable via history trim.
    if any(k in msg for k in _CTX_KEYWORDS):
        return True, True

    # 2. Permanent failures (auth/4xx) — surface immediately.
    #    Checked BEFORE rate-limit/server keywords so that "400 Bad Request"
    #    is not accidentally treated as retryable by a "400" → server match.
    if any(k in msg for k in _PERMANENT_KEYWORDS):
        return False, False

    # 3. Rate limit.
    if any(k in msg for k in _RATE_KEYWORDS):
        return True, False

    # 4. Generic server / network keywords.
    if any(k in msg for k in _SERVER_KEYWORDS):
        return True, False

    # 5. SDK-specific exception types whose ``str()`` may not contain a
    #    recognisable keyword.
    try:
        from requests.exceptions import (
            ConnectionError as _ReqConnErr,
            ChunkedEncodingError,
            Timeout as _ReqTimeout,
        )
        if isinstance(exc, (_ReqConnErr, ChunkedEncodingError, _ReqTimeout)):
            return True, False
    except ImportError:
        pass

    try:
        from http.client import RemoteDisconnected
        if isinstance(exc, RemoteDisconnected):
            return True, False
    except ImportError:
        pass

    try:
        import anthropic as _anthropic
        if isinstance(
            exc,
            (
                _anthropic.RateLimitError,
                _anthropic.InternalServerError,
                _anthropic.APIConnectionError,
                _anthropic.APITimeoutError,
            ),
        ):
            return True, False
        if isinstance(exc, _anthropic.BadRequestError) and any(k in msg for k in _CTX_KEYWORDS):
            return True, True
    except ImportError:
        pass

    return False, False


def is_transient(exc: Exception) -> bool:
    """Convenience wrapper: True iff the exception is safe to retry.

    Equivalent to ``classify_llm_error(exc)[0]``. Use this when the caller
    does not care about the context-overflow distinction (e.g. the monitor
    daemon retrying an MR review).
    """
    return classify_llm_error(exc)[0]


def is_permanent_error(exc: Exception) -> bool:
    """Return True iff the exception matches a known permanent-failure pattern
    (auth/forbidden/not-found/invalid-model/bad-request).

    Note: this is *not* the negation of :func:`is_transient`. An unknown error
    is neither transient nor permanent — callers decide the default policy:

    - LLM router: defaults unknown → retryable (fail-safe; one extra call is
      cheaper than a wrongly-permanent failure).
    - Execution engine / monitor: defaults unknown → not retryable (avoid
      spinning on an unrecognised bug).
    """
    msg = str(exc).lower()
    return any(k in msg for k in _PERMANENT_KEYWORDS)
