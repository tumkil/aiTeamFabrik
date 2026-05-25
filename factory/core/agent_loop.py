# Copyright 2024 SoftwareTeamFabrik contributors
# SPDX-License-Identifier: MIT

"""Shared agent loop skeleton with provider-specific adapters.

Extracts the common iteration/retry/tool-execution logic from
_agent_loop_anthropic and _agent_loop_ollama into a single
parameterized loop, with provider-specific behavior delegated
to adapter classes.

ADR-012: Shared Agent Loop Core with Provider-Specific Hooks
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (shared across all provider loops)
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 100
MAX_TOOL_HISTORY_BYTES = 200_000
MAX_LLM_RETRIES = 3          # attempts per iteration (1 original + 2 retries)
RETRY_DELAYS = (5, 15, 30)   # seconds between successive attempts


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """Normalized tool call extracted from an LLM response."""

    id: str
    name: str
    args: dict


@dataclass
class LoopResponse:
    """Normalized LLM response for the agent loop."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class LoopResult:
    """Result of the agent loop (mirrors ExecutionResult fields used by the loop)."""

    response: str = ""
    iterations: int = 0
    needs_continuation: bool = True
    files_changed: list[str] = field(default_factory=list)
    continuation_context: str = ""


# ---------------------------------------------------------------------------
# Provider adapter interface
# ---------------------------------------------------------------------------

class ProviderAdapter(ABC):
    """Abstract base class for provider-specific agent loop behavior.

    Each provider (Anthropic, Ollama, etc.) implements these hooks to
    customize how conversation history is managed, how LLM calls are made,
    and how responses are parsed and added to history.
    """

    @abstractmethod
    def build_initial_history(self, engine: Any) -> list:
        """Build the initial conversation history for the first iteration."""

    @abstractmethod
    def truncate_history(self, history: list, engine: Any) -> list:
        """Truncate conversation history to stay within context limits.

        Called at the start of each iteration.  Returns potentially modified
        history.  Implementations may also clear the engine's read cache.
        """

    @abstractmethod
    def call_llm(
        self,
        router: Any,
        engine: Any,
        system: str,
        history: list,
        tools: list,
        prompt: str,
    ) -> Any:
        """Make an LLM API call.  Returns the raw provider response object."""

    @abstractmethod
    def parse_response(self, response: Any, engine: Any) -> LoopResponse:
        """Parse a provider response into a normalized LoopResponse."""

    @abstractmethod
    def add_assistant_turn(
        self,
        history: list,
        raw_response: Any,
        loop_response: LoopResponse,
        engine: Any,
    ) -> list:
        """Add the assistant's response turn to conversation history.

        Called after parsing the response, before tool execution.
        """

    @abstractmethod
    def add_tool_result(
        self,
        history: list,
        tool_call: ToolCall,
        tool_result: str,
        engine: Any,
    ) -> list:
        """Add a single tool execution result to conversation history.

        Called for each tool call result.  Some providers (Ollama) add
        results immediately; others (Anthropic) buffer them for a
        later :meth:`finalize_tool_results` call.
        """

    @abstractmethod
    def finalize_tool_results(self, history: list, engine: Any) -> list:
        """Flush any buffered tool results into conversation history.

        Called after *all* tool calls in an iteration have been executed.
        Providers that add results immediately in :meth:`add_tool_result`
        (Ollama) should make this a no-op.
        """

    @abstractmethod
    def trim_history_on_overflow(self, history: list, engine: Any) -> list:
        """Aggressively trim history when a context-overflow error occurs."""

    @abstractmethod
    def build_iteration_prompt(self, engine: Any, iteration: int) -> str:
        """Build the prompt text for this iteration.

        Providers that embed the task description in the initial messages
        (Anthropic) should return an empty string.  Providers that rebuild
        the prompt each iteration (Ollama) should return the full prompt.
        """

    def stop_on_task_complete(self) -> bool:
        """Whether to stop executing remaining tools after ``task_complete``.

        Anthropic requires all tool results in a single user message, so it
        returns *False*.  Ollama can stop after ``task_complete`` and returns
        *True*.
        """
        return False


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------

class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic provider's agent loop behavior."""

    def __init__(self) -> None:
        self._tool_result_buffer: list[dict] = []

    # -- history management --------------------------------------------------

    def build_initial_history(self, engine: Any) -> list:
        return [
            {
                "role": "user",
                "content": f"Issue: {engine.issue_title}\n\n{engine.issue_description}",
            }
        ]

    def truncate_history(self, history: list, engine: Any) -> list:
        history_bytes = sum(len(json.dumps(m, default=str)) for m in history)
        if history_bytes > MAX_TOOL_HISTORY_BYTES:
            head = history[:1]  # keep original task description
            tail = history[-6:]
            # If tail starts with a tool_result user message, drop it (orphan).
            while (
                tail
                and tail[0].get("role") == "user"
                and isinstance(tail[0].get("content"), list)
                and any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in tail[0]["content"]
                )
            ):
                tail.pop(0)
            history = head + tail
            engine._read_cache.clear()
            engine._read_call_count = 0
        return history

    def trim_history_on_overflow(self, history: list, engine: Any) -> list:
        # Trim aggressively: keep only the first user message + last 2 turns.
        head = history[:1]
        tail = history[-4:] if len(history) > 5 else history[1:]
        history = head + tail
        engine._read_cache.clear()
        engine._read_call_count = 0
        return history

    # -- LLM call -----------------------------------------------------------

    def call_llm(
        self,
        router: Any,
        engine: Any,
        system: str,
        history: list,
        tools: list,
        prompt: str,
    ) -> Any:
        # Anthropic uses messages, not a rebuilt prompt
        return router.anthropic_complete(
            agent=engine.agent,
            system=system,
            messages=history,
            tools=tools,
        )

    # -- response parsing ---------------------------------------------------

    def parse_response(self, response: Any, engine: Any) -> LoopResponse:
        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        args=block.input if isinstance(block.input, dict) else {},
                    )
                )
        return LoopResponse(
            text="\n".join(text_parts) if text_parts else "",
            tool_calls=tool_calls,
        )

    # -- history updates ----------------------------------------------------

    def add_assistant_turn(
        self,
        history: list,
        raw_response: Any,
        loop_response: LoopResponse,
        engine: Any,
    ) -> list:
        assistant_blocks: list[dict] = []
        for block in raw_response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                assistant_blocks.append({"type": "text", "text": block.text})
            elif btype == "tool_use":
                assistant_blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
        history.append({"role": "assistant", "content": assistant_blocks})
        return history

    def add_tool_result(
        self,
        history: list,
        tool_call: ToolCall,
        tool_result: str,
        engine: Any,
    ) -> list:
        # Buffer tool results; they are flushed as a single user message
        # in finalize_tool_results.
        self._tool_result_buffer.append(
            {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": tool_result,
            }
        )
        return history

    def finalize_tool_results(self, history: list, engine: Any) -> list:
        if self._tool_result_buffer:
            history.append({"role": "user", "content": list(self._tool_result_buffer)})
            self._tool_result_buffer = []
        return history

    # -- prompt (Anthropic uses messages, not a rebuilt prompt) -------------

    def build_iteration_prompt(self, engine: Any, iteration: int) -> str:
        return ""


# ---------------------------------------------------------------------------
# Ollama adapter
# ---------------------------------------------------------------------------

class OllamaAdapter(ProviderAdapter):
    """Adapter for Ollama provider's agent loop behavior."""

    def __init__(self) -> None:
        self._history_truncated = False
        self._openai_tools: list | None = None  # cached conversion

    # -- history management --------------------------------------------------

    def build_initial_history(self, engine: Any) -> list:
        return []

    def truncate_history(self, history: list, engine: Any) -> list:
        history_bytes = sum(len(json.dumps(msg)) for msg in history)
        if history_bytes > MAX_TOOL_HISTORY_BYTES:
            history = history[-6:]
            # Drop orphaned tool messages at the head.
            while history and history[0].get("role") == "tool":
                history.pop(0)
            # Drop assistant entries at the head whose tool_calls were sliced off.
            while (
                history
                and history[0].get("role") == "assistant"
                and history[0].get("tool_calls")
                and (
                    len(history) < 2
                    or history[1].get("role") != "tool"
                )
            ):
                history.pop(0)
            # Drop empty/null assistant entries at the head.
            while (
                history
                and history[0].get("role") == "assistant"
                and not history[0].get("tool_calls")
                and not history[0].get("content")
            ):
                history.pop(0)
            # Drop orphaned assistant messages at the tail.
            while (
                history
                and history[-1].get("role") == "assistant"
                and history[-1].get("tool_calls")
            ):
                history.pop()
            engine._read_cache.clear()
            engine._read_call_count = 0
            self._history_truncated = True
        return history

    def trim_history_on_overflow(self, history: list, engine: Any) -> list:
        history = history[-4:] if len(history) > 4 else []
        while history and history[0].get("role") == "tool":
            history.pop(0)
        engine._read_cache.clear()
        engine._read_call_count = 0
        # Note: _history_truncated is NOT set here — it is only set by
        # truncate_history during regular truncation, not during overflow
        # handling.  The next iteration's build_iteration_prompt will not
        # add the truncation notice because the flag was not set.
        return history

    # -- LLM call -----------------------------------------------------------

    def call_llm(
        self,
        router: Any,
        engine: Any,
        system: str,
        history: list,
        tools: list,
        prompt: str,
    ) -> Any:
        if self._openai_tools is None:
            self._openai_tools = engine._to_openai_tools(tools)
        return router.ollama_complete(
            agent=engine.agent,
            system=system,
            prompt=prompt,
            tools=self._openai_tools,
            history=history,
        )

    # -- response parsing ---------------------------------------------------

    def parse_response(self, response: Any, engine: Any) -> LoopResponse:
        response_data = engine._parse_ollama_response(response)
        tool_calls: list[ToolCall] = []
        for tc in response_data.get("tool_calls", []):
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=tc["function"]["name"],
                    args=tc["function"]["arguments"],
                )
            )
        content = response_data.get("content") or ""
        return LoopResponse(text=content, tool_calls=tool_calls)

    # -- history updates ----------------------------------------------------

    def add_assistant_turn(
        self,
        history: list,
        raw_response: Any,
        loop_response: LoopResponse,
        engine: Any,
    ) -> list:
        # Ollama adds the assistant turn per-tool-call in add_tool_result,
        # not as a separate step.
        return history

    def add_tool_result(
        self,
        history: list,
        tool_call: ToolCall,
        tool_result: str,
        engine: Any,
    ) -> list:
        tool_call_dict = {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.name,
                "arguments": tool_call.args,
            },
        }
        history.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call_dict],
            }
        )
        history.append(
            {
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_call.id,
            }
        )
        return history

    def finalize_tool_results(self, history: list, engine: Any) -> list:
        # Ollama adds results immediately in add_tool_result; nothing to flush.
        return history

    # -- prompt (Ollama rebuilds the prompt each iteration) -----------------

    def build_iteration_prompt(self, engine: Any, iteration: int) -> str:
        if iteration == 1:
            prompt = f"Issue: {engine.issue_title}\n\n{engine.issue_description}"
        else:
            short_desc = engine.issue_description[:500]
            prompt = (
                f"Issue: {engine.issue_title}\n\n{short_desc}\n\n"
                "[Full spec was in iteration 1 — do NOT re-read files, "
                "START writing code now.]"
            )

        if self._history_truncated:
            changed = (
                ", ".join(engine._files_changed) if engine._files_changed else "none"
            )
            prompt += (
                f"\n\n[Prior conversation history was truncated. "
                f"Files changed so far: {changed}]"
            )
            self._history_truncated = False

        if engine._read_cache:
            already = ", ".join(engine._read_cache.keys())
            prompt += f"\n\n[Already read: {already}. Do NOT read these again.]"

        if iteration > 8 and not engine._files_changed:
            prompt += (
                f"\n\n[REMINDER: {iteration} iterations elapsed and no files "
                "have been written yet. You have enough context — please write "
                "your first file now using write_file.]"
            )

        return prompt

    def stop_on_task_complete(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Shared agent loop
# ---------------------------------------------------------------------------

def run_agent_loop(
    engine: Any,
    router: Any,
    system: str,
    adapter: ProviderAdapter,
    tools: list,
    max_iterations: int = MAX_ITERATIONS,
) -> LoopResult:
    """Run the shared agent loop skeleton.

    This function implements the common iteration / retry / tool-execution
    logic shared by all provider-specific agent loops.  Provider-specific
    behavior is delegated to *adapter*.

    Args:
        engine: The :class:`CodeExecutionEngine` instance (for accessing
            shared state such as ``_read_cache``, ``_files_changed``, etc.).
        router: The :class:`LlmRouter` instance for making API calls.
        system: The system prompt.
        adapter: The provider-specific adapter.
        tools: The tool definitions for the LLM.
        max_iterations: Maximum number of iterations.

    Returns:
        A :class:`LoopResult` with response, iterations, and continuation
        info.  The caller is responsible for converting it to an
        :class:`ExecutionResult` if needed.
    """
    result = LoopResult()
    history = adapter.build_initial_history(engine)

    for iteration in range(1, max_iterations + 1):
        engine._progress(f"[dim]Iteration {iteration}[/dim]")

        # Truncate history to stay within context window
        history = adapter.truncate_history(history, engine)

        # Build iteration-specific prompt (empty string for providers
        # that embed the task in initial messages).
        prompt = adapter.build_iteration_prompt(engine, iteration)

        # LLM call with retry for transient errors
        llm_error: Exception | None = None
        raw_response = None
        for attempt in range(MAX_LLM_RETRIES):
            try:
                raw_response = adapter.call_llm(
                    router, engine, system, history, tools, prompt
                )
                break  # success
            except Exception as exc:
                retryable, ctx_overflow = engine._classify_llm_error(exc)
                if not retryable or attempt == MAX_LLM_RETRIES - 1:
                    llm_error = exc
                    break
                if ctx_overflow:
                    history = adapter.trim_history_on_overflow(history, engine)
                    engine._progress(
                        "[yellow]⚠ Context too long, trimmed history — retrying…[/yellow]"
                    )
                else:
                    delay = RETRY_DELAYS[attempt]
                    safe_exc = engine._sanitize_log(str(exc))
                    engine._progress(
                        f"[yellow]⚠ LLM error (attempt {attempt + 1}): "
                        f"{safe_exc[:120]} — retrying in {delay}s…[/yellow]"
                    )
                    logger.warning(
                        "LLM transient error iteration %d attempt %d: %s",
                        iteration,
                        attempt + 1,
                        safe_exc,
                    )
                    time.sleep(delay)

        if llm_error is not None:
            safe_exc = engine._sanitize_log(str(llm_error))
            logger.error("Iteration %d failed: %s", iteration, safe_exc)
            result.response += f"\nError in iteration {iteration}: {safe_exc}"
            result.iterations = iteration
            result.needs_continuation = True
            result.continuation_context = f"Error at iteration {iteration}: {safe_exc}"
            result.files_changed = list(engine._files_changed)
            break

        # Parse response and process tool calls.
        # Wrapped in try/except so that provider-specific parse errors or
        # unexpected response structures surface as iteration errors rather
        # than crashing the entire loop.
        try:
            loop_response = adapter.parse_response(raw_response, engine)

            # Handle text content
            if loop_response.text:
                result.response += loop_response.text
                if loop_response.text.strip():
                    engine._progress(
                        f"[dim]  (text response, {len(loop_response.text)} chars)[/dim]"
                    )

            # Add assistant's response turn to history
            history = adapter.add_assistant_turn(
                history, raw_response, loop_response, engine
            )

            # No tool calls → model is done talking
            if not loop_response.tool_calls:
                result.iterations = iteration
                result.needs_continuation = False
                result.files_changed = list(engine._files_changed)
                break

            # Execute tools
            task_completed = False
            completion_summary = ""
            for tc in loop_response.tool_calls:
                engine._progress(engine._format_tool_call_start(tc.name, tc.args))
                tool_result = engine._execute_tool(tc.name, tc.args)
                engine._progress(engine._format_tool_call_end(tc.name, tool_result))

                # Add tool result to history
                history = adapter.add_tool_result(history, tc, tool_result, engine)

                if tc.name == "task_complete":
                    completion_summary = tc.args.get("summary", tool_result)
                    task_completed = True
                    if adapter.stop_on_task_complete():
                        break

            # Finalize any buffered tool results
            history = adapter.finalize_tool_results(history, engine)

            if task_completed:
                result.response = completion_summary
                result.iterations = iteration
                result.files_changed = list(engine._files_changed)
                result.needs_continuation = False
                break

            if iteration >= max_iterations:
                result.iterations = iteration
                result.needs_continuation = True
                result.continuation_context = (
                    f"Iteration limit reached at {iteration}/{max_iterations}"
                )
                result.files_changed = list(engine._files_changed)
                break

        except Exception as exc:
            safe_exc = engine._sanitize_log(str(exc))
            logger.error("Iteration %d tool/parse error: %s", iteration, safe_exc)
            result.response += f"\nError in iteration {iteration}: {safe_exc}"
            result.iterations = iteration
            result.needs_continuation = True
            result.continuation_context = f"Error at iteration {iteration}: {safe_exc}"
            result.files_changed = list(engine._files_changed)
            break

    return result