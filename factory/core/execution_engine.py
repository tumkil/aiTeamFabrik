from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from filelock import FileLock, Timeout as FileLockTimeout

from factory.adapters.llm_router import LlmResponse
from factory.core.agent_loop import (
    AnthropicAdapter,
    LoopResult,
    OllamaAdapter,
    run_agent_loop,
)
from factory.core.competence import AgentProfile
from factory.core.resilience import classify_llm_error as _classify_llm_error_impl

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 100
MAX_FILE_READ_BYTES = 200_000
MAX_COMMAND_OUTPUT_BYTES = 64_000
MAX_COMMAND_TIMEOUT_SECONDS = 300
MAX_SEARCH_TIMEOUT_SECONDS = 30  # lower cap for grep -r to avoid blocking the loop
MAX_TOOL_HISTORY_BYTES = 200_000
MAX_READ_CALLS = 8

# Retry policy for transient LLM errors (rate-limit / server errors).
_MAX_LLM_RETRIES = 3          # attempts per iteration (1 original + 2 retries)
_RETRY_DELAYS = (5, 15, 30)   # seconds between successive attempts

# Exact command names the agent is allowed to run.
# Token-based matching (first shlex token) rather than prefix matching
# to prevent partial-word bypasses such as "python-evil" matching "python".
_ALLOWED_COMMANDS: frozenset[str] = frozenset({
    "git", "python", "python3", "pytest", "pip", "pip3",
    "ls", "find", "grep", "cat", "mkdir", "touch", "echo",
    # cp/mv/rm are intentionally excluded: they accept absolute paths and can
    # reach files outside the repo root. Use write_file/read_file tools instead.
})

# Commands whose non-flag arguments can be filesystem paths. Absolute paths
# and parent-directory traversal (../outside-repo) are rejected on these to
# reduce the risk of reading files outside the repository root.
# This is a best-effort heuristic — it does not fully replace _read_file's
# is_relative_to guard, and it does not cover python/pytest/pip args.
_PATH_RESTRICTED_COMMANDS: frozenset[str] = frozenset({"cat", "grep", "find", "ls"})

# Filename substrings that suggest a file may contain secrets.
# Used in _task_complete to warn the agent before it stages anything sensitive.
_SENSITIVE_FILE_PATTERNS: tuple[str, ...] = (
    ".env", "secret", "password", "credential", "private", "token",
    ".pem", ".key", ".p12", ".pfx", ".crt", "id_rsa", "id_ed25519", "id_dsa",
)

# git subcommands that are too destructive or could conflict with the engine's
# own git operations (branching, staging, committing, pushing).
# Note: internal methods (e.g. _git_commit_and_push) call git switch and
# git push directly via subprocess — this list only gates agent-invoked
# run_command calls, not the engine's own git operations.
_GIT_BLOCKED_SUBCOMMANDS: frozenset[str] = frozenset({
    "push", "config", "reset", "clean", "checkout", "switch",
    "merge", "rebase", "stash", "bisect", "cherry-pick", "revert",
    # remote/fetch can expose token-embedded URLs or mutate FETCH_HEAD;
    # clone/submodule/worktree/tag can interfere with engine's own git state.
    "remote", "fetch", "clone", "submodule", "worktree", "tag",
    # branch -D and branch -f can delete or force-reset branches including
    # the engine's own target branch.
    "branch",
})

_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file from the repository. Returns its contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to repo root."}
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file (creates or fully overwrites). Use for all code changes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to repo root."},
                "content": {"type": "string", "description": "Full file content."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files and sub-directories at a path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to repo root (default: '.')."},
            },
            "required": [],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a shell command inside the repository. "
            "Allowed commands (exact name): git, python, python3, pytest, pip, pip3, ls, find, grep, cat, mkdir, touch, echo."
            " Use write_file/read_file tools instead of cp/mv/rm (those are excluded to prevent access outside the repo)."
            " Note: git push, git config, git reset, git checkout and other state-modifying subcommands are blocked."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
            },
            "required": ["command"],
        },
    },
    {
        "name": "search_files",
        "description": "Search for a text pattern across files in the repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Grep pattern."},
                "path": {"type": "string", "description": "Directory to search (default: '.')."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "task_complete",
        "description": (
            "Signal that the implementation is finished. "
            "Call this when all code is written and tests pass."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Short description of what was implemented.",
                },
            },
            "required": ["summary"],
        },
    },
]

_EXECUTION_SYSTEM = """\
You are the Developer agent of SoftwareTeamFabrik running inside a git repository.
You have tools to read files, write files, list directories, run shell commands, and search code.

Your job is to implement the given GitLab issue completely — writing real, working code.

Workflow:
1. Explore the repo (max 5 read/list calls). Read only the files directly relevant to the issue.
2. Check if the issue is already implemented. If it is, call task_complete immediately with
   "Already implemented: <brief explanation of what exists>". Do NOT write any files.
3. If not implemented: write_file for every file you create or change.
4. Run the test suite: run_command('python3 -m pytest tests/ -v') and fix any failures.
5. When tests pass, call task_complete with a concise summary of what was implemented.

CRITICAL RULES:
- STOP reading after 5 files. You have enough context — start writing code.
- Do NOT re-read a file you have already read. The cache will block it anyway.
- Write complete file contents — never partial patches or snippets.
- No placeholders, no TODOs, no "implement this later".
- Follow the existing code style exactly (spacing, imports, naming).
- If you hit a genuine blocker, call task_complete describing it.
"""


def _strip_rich_markup(text: str) -> str:
    """Strip Rich markup tags from text for plain console output."""
    return re.sub(r'\[/?[a-zA-Z]+\]', '', text)


@dataclass
class ExecutionResult:
    """Result of an agent execution loop."""
    response: str
    commit_sha: str = ""
    # True when a commit was made but the remote push did not complete — either
    # the push failed, was skipped (no origin, SSH remote with token), or the
    # post-commit SHA retrieval failed. commit_sha may or may not be set.
    push_failed: bool = False
    files_changed: list[str] = field(default_factory=list)
    iterations: int = 0
    mode: str = "execute"  # plan | execute
    comment_url: str = ""
    mr_url: str = ""
    needs_continuation: bool = False
    continuation_context: str = ""


def _loop_result_to_execution_result(
    loop_result: LoopResult,
    mode: str,
) -> ExecutionResult:
    """Convert a :class:`LoopResult` from the shared agent loop into an
    :class:`ExecutionResult` consumed by the rest of the engine.
    """
    return ExecutionResult(
        response=loop_result.response,
        mode=mode,
        iterations=loop_result.iterations,
        needs_continuation=loop_result.needs_continuation,
        continuation_context=loop_result.continuation_context,
        files_changed=loop_result.files_changed,
    )


class CodeExecutionEngine:
    """Executes agent loops with tool-calling support for Anthropic, Mistral, and Ollama providers.

    Requires Python ≥ 3.11 (enforced by pyproject.toml requires-python).
    Path-traversal guards use Path.is_relative_to(), available since 3.9.
    """

    # Cached git version — populated once per process, shared across all instances.
    _cached_git_version: tuple = ()

    # Allowlist of env var names forwarded to the git push subprocess.
    # Explicit opt-in rather than opt-out from GIT_*: broad GIT_* forwarding
    # would leak GIT_SSH_COMMAND (embedded key paths), GIT_ASKPASS (arbitrary
    # script), and GIT_EXEC_PATH into the child process alongside our token.
    _PUSH_ENV_ALLOWLIST: frozenset[str] = frozenset({
        "HOME", "PATH", "USER", "LOGNAME", "USERNAME",
        "TMPDIR", "TEMP", "TMP",
        "LANG", "LC_ALL", "LC_MESSAGES",
        # Included for the no-token case where push falls back to SSH key auth.
        # When a token is set and the remote is non-HTTPS, push is skipped early;
        # SSH_AUTH_SOCK is forwarded only for SSH pushes without a token.
        "SSH_AUTH_SOCK",
    })
    # Only prefix-match GIT_SSL_* (self-signed cert CAs in CI), not all GIT_*.
    _PUSH_ENV_PREFIXES: tuple[str, ...] = ("GIT_SSL_", "SSL_", "KRB5_")

    @classmethod
    def _git_version_ok(cls) -> bool:
        """Return True if the installed git supports GIT_CONFIG_* env injection (≥ 2.31).

        Not protected by a lock: two threads can both observe an empty cache and
        both call `git --version`, racing on the check-then-act sequence. Each
        writes the same result; the last write wins harmlessly and no partial
        state is ever visible (CPython's GIL makes the class-attribute
        assignment non-interruptible). The cost is at most one extra subprocess
        call per race — accepted in exchange for avoiding a lock on every call.
        """
        if not cls._cached_git_version:
            try:
                r = subprocess.run(["git", "--version"], capture_output=True, text=True)
                m = re.search(r"(\d+)\.(\d+)", r.stdout or "")
                cls._cached_git_version = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
            except (FileNotFoundError, OSError):
                cls._cached_git_version = (0, 0)
        return cls._cached_git_version >= (2, 31)

    @staticmethod
    def _truncate_subject(text: str, max_len: int = 72) -> str:
        """Truncate text to max_len chars, appending '...' (3 ASCII chars) if cut.

        Uses ASCII '...' rather than the Unicode ellipsis U+2026 to keep the
        result within max_len bytes on systems that measure byte length.
        """
        if len(text) > max_len:
            return text[:max_len - 3] + "..."
        return text

    def __init__(
        self,
        agent: AgentProfile,
        repo_path: Path,
        issue_title: str,
        issue_description: str,
        gitlab_token: str,
        gitlab_project_id: str | int,
        gitlab_url: str,
        branch_name: str,
        mr_iid: int | None = None,
        clone_url: str = "",
        progress=None,
    ):
        self.agent = agent
        self.repo_path = Path(repo_path)
        self.issue_title = issue_title
        self.issue_description = issue_description
        _stripped_token = (gitlab_token or "").strip()
        if gitlab_token and not _stripped_token:
            logger.warning("gitlab_token is whitespace-only; treating as no token (push will be unauthenticated)")
        self.gitlab_token = _stripped_token or None
        self.gitlab_project_id = str(gitlab_project_id)
        self.gitlab_url = gitlab_url
        self.branch_name = branch_name
        self.mr_iid = mr_iid
        self.clone_url = clone_url
        self._history: list[dict] = []
        self._files_changed: list[str] = []
        self._task_summary: str = ""
        self._read_cache: dict[str, str] = {}
        self._push_failed: bool = False  # set True when commit succeeded but push failed
        self._history_truncated: bool = False  # set True when history was trimmed mid-run
        self._read_call_count: int = 0
        self._progress = progress or (lambda msg: print(_strip_rich_markup(msg)))

    def _to_openai_tools(self, tools: list[dict]) -> list[dict]:
        """Convert internal tool schema to OpenAI-compatible format for Ollama/Mistral.
        
        Normalizes tool definitions to the OpenAI tools API format that both
        Ollama (in OpenAI-compatible mode) and Mistral understand.
        """
        openai_tools = []
        for tool in tools:
            name = tool["name"]
            description = tool["description"]
            properties = tool["input_schema"].get("properties", {})
            required = tool["input_schema"].get("required", [])
            
            # Build parameters object
            parameters = {"type": "object", "properties": {}}
            for param_name, param_spec in properties.items():
                parameters["properties"][param_name] = param_spec
            
            if required:
                parameters["required"] = required
            
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            })
        
        return openai_tools

    # ------------------------------------------------------------------
    # Agent loops — thin wrappers that delegate to the shared loop in
    # factory.core.agent_loop.  The provider-specific logic (history
    # management, response parsing, prompt building) lives in the
    # AnthropicAdapter / OllamaAdapter classes respectively.  See
    # ADR-012 for the rationale behind this extraction.
    # ------------------------------------------------------------------

    def _agent_loop_anthropic(
        self,
        router: Any,
        system: str,
        max_iterations: int = MAX_ITERATIONS,
    ) -> ExecutionResult:
        """Anthropic agent loop — delegates to the shared agent loop core."""
        adapter = AnthropicAdapter()
        loop_result = run_agent_loop(
            engine=self,
            router=router,
            system=system,
            adapter=adapter,
            tools=_TOOLS,
            max_iterations=max_iterations,
        )
        return _loop_result_to_execution_result(loop_result, mode=self.agent.execution_mode)

    def _agent_loop_ollama(
        self,
        router: Any,
        system: str,
        max_iterations: int = MAX_ITERATIONS,
    ) -> ExecutionResult:
        """Ollama agent loop — delegates to the shared agent loop core."""
        adapter = OllamaAdapter()
        loop_result = run_agent_loop(
            engine=self,
            router=router,
            system=system,
            adapter=adapter,
            tools=_TOOLS,
            max_iterations=max_iterations,
        )
        return _loop_result_to_execution_result(loop_result, mode=self.agent.execution_mode)

    def _parse_ollama_response(self, llm_response: LlmResponse) -> dict:
        """Parse Ollama response, handling both native and OpenAI-compatible formats."""
        try:
            # Try to parse as JSON (OpenAI-compatible format from cloud-hosted Ollama).
            # Guard: require `tool_calls` to be a list and `content` to be a str/None
            # so an LLM response that happens to be valid JSON (e.g. a code snippet
            # containing `"content": ...`) is not silently treated as a structured
            # API response — which would drop tool calls and mangle the content.
            response_data = json.loads(llm_response.content)
            if isinstance(response_data, dict):
                tool_calls = response_data.get("tool_calls")
                content = response_data.get("content")
                has_tool_calls = isinstance(tool_calls, list)
                has_content = content is None or isinstance(content, str)
                # Only treat as a structured API response when the "tool_calls"
                # key is explicitly present. Responses that just have {"content":
                # null} without "tool_calls" could be LLM output that happens to
                # be valid JSON — treat those as plain text instead.
                if "tool_calls" in response_data and has_tool_calls and has_content:
                    # Validate that every tool call has a function.name string so
                    # malformed entries don't reach _execute_tool with missing keys.
                    if tool_calls:
                        if not all(
                            isinstance(tc, dict)
                            and isinstance(tc.get("function", {}).get("name"), str)
                            for tc in tool_calls
                        ):
                            # Malformed tool call structure; treat as plain text
                            return {"content": llm_response.content, "tool_calls": []}
                    return response_data
        except (json.JSONDecodeError, TypeError):
            pass

        # Return as plain-text response — no structured tool calls detected.
        return {
            "content": llm_response.content,
            "tool_calls": [],
        }

    def _format_tool_call_start(self, tool_name: str, tool_args: dict) -> str:
        """Format a tool call start message for progress display."""
        icons = {
            "read_file": "📖",
            "write_file": "✎",
            "list_directory": "📂",
            "run_command": "▶",
            "search_files": "🔍",
            "task_complete": "✓",
        }
        icon = icons.get(tool_name, "→")
        
        # Build args display
        args_display = self._format_tool_args(tool_name, tool_args)
        
        if tool_name == "write_file":
            return f"  [cyan]{icon} Calling write_file[/cyan] → {args_display}"
        if tool_name == "read_file":
            return f"  [blue]{icon} Calling read_file[/blue] → {args_display}"
        if tool_name == "run_command":
            return f"  [yellow]{icon} Calling run_command[/yellow] → {args_display}"
        if tool_name == "list_directory":
            return f"  [magenta]{icon} Calling list_directory[/magenta] → {args_display}"
        if tool_name == "search_files":
            return f"  [cyan]{icon} Calling search_files[/cyan] → {args_display}"
        if tool_name == "task_complete":
            return f"  [green]{icon} Calling task_complete[/green] → {args_display}"
        return f"  {icon} Calling {tool_name} → {args_display}"

    def _format_tool_call_end(self, tool_name: str, tool_result: str) -> str:
        """Format a tool call completion message for progress display."""
        # Determine success/failure based on result content
        is_error = tool_result.startswith("Error") or tool_result.startswith("Access denied") or tool_result.startswith("Unknown tool")
        
        if is_error:
            # Truncate error message for display
            error_preview = tool_result[:80] + "..." if len(tool_result) > 80 else tool_result
            return f"    [red]✗ Failed:[/red] {error_preview}"
        elif tool_name == "task_complete":
            return f"    [green]✓ Task completed[/green]"
        elif tool_name == "write_file":
            return f"    [green]✓ File written[/green]"
        elif tool_name == "read_file":
            # Show content length for read operations
            return f"    [green]✓ Read {len(tool_result)} bytes[/green]"
        elif tool_name == "run_command":
            # Show output length for command execution
            lines = tool_result.count('\n') + 1
            return f"    [green]✓ Command completed[/green] ({lines} lines output)"
        elif tool_name == "list_directory":
            # Parse and show item count for directory listing
            try:
                data = json.loads(tool_result)
                files = len(data.get("files", []))
                dirs = len(data.get("directories", []))
                return f"    [green]✓ Directory listed[/green] ({files} files, {dirs} dirs)"
            except (json.JSONDecodeError, TypeError):
                return f"    [green]✓ Directory listed[/green]"
        elif tool_name == "search_files":
            # Show number of matches
            matches = len([l for l in tool_result.split('\n') if l.strip()])
            return f"    [green]✓ Search completed[/green] ({matches} matches)"
        else:
            return f"    [green]✓ Completed[/green]"

    def _format_tool_args(self, tool_name: str, tool_args: dict) -> str:
        """Format tool arguments for display."""
        if not isinstance(tool_args, dict):
            return str(tool_args)[:60]
        
        if tool_name == "write_file":
            path = tool_args.get("path", "")
            content_len = len(tool_args.get("content", ""))
            return f"path={path} ({content_len} bytes)"
        if tool_name == "read_file":
            return f"path={tool_args.get('path', '')}"
        if tool_name == "run_command":
            cmd = tool_args.get("command", "")
            # Truncate long commands
            if len(cmd) > 50:
                return f"command={cmd[:50]}..."
            return f"command={cmd}"
        if tool_name == "list_directory":
            return f"path={tool_args.get('path', '.')}"
        if tool_name == "search_files":
            pattern = tool_args.get("pattern", "")
            path = tool_args.get("path", ".")
            return f"pattern={pattern}, path={path}"
        if tool_name == "task_complete":
            summary = tool_args.get("summary", "")
            if len(summary) > 50:
                return f"summary={summary[:50]}..."
            return f"summary={summary}"
        
        # Generic fallback: show first few key-value pairs
        items = list(tool_args.items())[:3]
        parts = [f"{k}={str(v)[:30]}" for k, v in items]
        return ", ".join(parts)

    def _format_tool_call(self, tool_name: str, tool_args: dict) -> str:
        """Format a tool call for progress display (legacy method, kept for compatibility)."""
        return self._format_tool_call_start(tool_name, tool_args)

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """Execute a tool call and return the result.

        Uses .get() for all required arguments so that a malformed LLM response
        (missing a required key) returns a clean error string instead of raising
        KeyError, which would surface as a generic iteration failure.
        """
        if not isinstance(tool_args, dict):
            # Some providers (especially Ollama) serialize arguments as a JSON
            # string instead of a pre-parsed dict — try to recover gracefully.
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                    if not isinstance(tool_args, dict):
                        return f"Error: tool arguments JSON did not decode to a dict (got {type(tool_args).__name__})"
                except json.JSONDecodeError as exc:
                    return f"Error: tool arguments could not be parsed as JSON: {exc}"
            else:
                return f"Error: tool arguments must be a dict, got {type(tool_args).__name__}"
        if tool_name == "read_file":
            path = tool_args.get("path")
            if not path:
                return "Error: 'path' argument is required for read_file"
            return self._read_file(path)
        elif tool_name == "write_file":
            path = tool_args.get("path")
            content = tool_args.get("content")
            if not path:
                return "Error: 'path' argument is required for write_file"
            if content is None:
                return "Error: 'content' argument is required for write_file"
            return self._write_file(path, content)
        elif tool_name == "list_directory":
            return self._list_directory(tool_args.get("path", "."))
        elif tool_name == "run_command":
            command = tool_args.get("command")
            if not command:
                return "Error: 'command' argument is required for run_command"
            return self._run_command(command)
        elif tool_name == "search_files":
            pattern = tool_args.get("pattern")
            if not pattern:
                return "Error: 'pattern' argument is required for search_files"
            return self._search_files(pattern, tool_args.get("path", "."))
        elif tool_name == "task_complete":
            return self._task_complete(tool_args.get("summary", ""))
        else:
            return f"Unknown tool: {tool_name}"

    def _read_file(self, path: str) -> str:
        """Read a file from the repository."""
        if path in self._read_cache:
            return "[cached — content already in conversation history]"
        repo = self.repo_path.resolve()
        safe = (self.repo_path / path).resolve()
        if not safe.is_relative_to(repo):
            return f"Access denied: {path} is outside the repository root"
        if not safe.exists():
            return f"File not found: {path}"
        try:
            with open(safe, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(MAX_FILE_READ_BYTES)
            if len(content) == MAX_FILE_READ_BYTES:
                content += f"\n[truncated at {MAX_FILE_READ_BYTES} bytes]"
            self._read_cache[path] = content
            return content
        except Exception as exc:
            return f"Error reading {path}: {exc}"

    def _write_file(self, path: str, content: str) -> str:
        """Write content to a file."""
        repo = self.repo_path.resolve()
        safe = (self.repo_path / path).resolve()
        if not safe.is_relative_to(repo):
            return f"Access denied: {path} is outside the repository root"
        try:
            safe.parent.mkdir(parents=True, exist_ok=True)
            with open(safe, "w", encoding="utf-8") as f:
                f.write(content)
            if path not in self._files_changed:
                self._files_changed.append(path)
            # Invalidate cache rather than backfilling with written content.
            # Backfilling would return stale data if the file is subsequently
            # overwritten (by another write_file call or an external process).
            # Clearing the entry lets the next read_file re-read from disk.
            self._read_cache.pop(path, None)
            return f"File written: {path}"
        except Exception as exc:
            return f"Error writing {path}: {exc}"

    def _list_directory(self, path: str = ".") -> str:
        """List files and sub-directories at a path."""
        repo = self.repo_path.resolve()
        dir_path = (self.repo_path / path).resolve()
        if not dir_path.is_relative_to(repo):
            return f"Access denied: {path} is outside the repository root"
        if not dir_path.exists():
            return f"Directory not found: {path}"
        try:
            items = list(dir_path.iterdir())
            files = [f.name for f in items if f.is_file()]
            dirs = [f.name for f in items if f.is_dir()]
            return json.dumps({"files": files, "directories": dirs})
        except Exception as exc:
            return f"Error listing {path}: {exc}"

    def _run_command(self, command: str) -> str:
        """Run a shell command."""
        try:
            args = shlex.split(command)
        except ValueError as exc:
            return f"Invalid command syntax: {exc}"
        # Validate the first token against the exact-name allowlist to prevent
        # partial-word matches (e.g. "python-evil" starts with "python") and
        # shell injection via metacharacters (shell=False handles the rest).
        if not args or args[0] not in _ALLOWED_COMMANDS:
            return f"Command not allowed: {command}"
        # For filesystem commands, reject absolute paths and parent-directory
        # traversal in non-flag arguments. This is a heuristic — use the
        # search_files and read_file tools for fully path-safe access.
        # Also reject grep's -f/--file flags: they accept an arbitrary path and
        # would bypass the positional-argument check (the path appears as a flag
        # value, which starts with "-" and would be skipped by the guard below).
        if args[0] == "grep" and any(
            a == "-f" or a.startswith("--file") for a in args[1:]
        ):
            return "grep -f / --file is not allowed (use search_files tool instead)"
        if args[0] in _PATH_RESTRICTED_COMMANDS:
            for arg in args[1:]:
                if arg.startswith("-"):
                    continue
                if arg.startswith("/") or arg == ".." or arg.startswith("../") or "/.." in arg:
                    return f"Absolute paths and parent-directory references are not allowed: {arg}"
        # Block destructive git subcommands that could corrupt repo state or
        # conflict with the engine's own branch/stage/commit/push operations.
        # Check ALL non-flag tokens so that `git -c key=val push` (where the
        # value "key=val" appears before "push") cannot bypass a check that only
        # inspects args[1]. Using next() on the first non-flag token would pick
        # up "key=val" instead of "push" and silently pass the command through.
        if args[0] == "git" and len(args) > 1:
            non_flag_tokens = [t for t in args[1:] if not t.startswith("-")]
            blocked = next((t for t in non_flag_tokens if t in _GIT_BLOCKED_SUBCOMMANDS), None)
            if blocked:
                return f"git {blocked} is not available in this context"
        try:
            # shell=False with shlex.split prevents shell injection: operators like
            # `;`, `&&`, and `|` become literal arguments rather than shell metacharacters.
            result = subprocess.run(
                args,
                shell=False,
                capture_output=True,
                text=True,
                timeout=MAX_COMMAND_TIMEOUT_SECONDS,
                cwd=self.repo_path,
            )
            output = result.stdout[:MAX_COMMAND_OUTPUT_BYTES]
            if result.returncode != 0:
                stderr = result.stderr[:MAX_COMMAND_OUTPUT_BYTES]
                output += f"\n[exit {result.returncode}]\n{stderr}"
            return output
        except subprocess.TimeoutExpired:
            return f"Command timed out: {command}"
        except Exception as exc:
            return f"Error running command: {exc}"

    def _search_files(self, pattern: str, path: str = ".") -> str:
        """Search for a text pattern across files.

        Note: this method calls grep directly via subprocess.run rather than
        routing through _run_command. The design is intentional — _run_command
        enforces an agent-facing command allowlist, while _search_files is an
        internal tool method with its own is_relative_to() path guard. The two
        grep invocation paths have different security postures by design: agent
        commands go through _run_command, internal tool calls go through here.
        """
        repo = self.repo_path.resolve()
        search_path = (self.repo_path / path).resolve()
        if not search_path.is_relative_to(repo):
            return f"Access denied: {path} is outside the repository root"
        if not search_path.exists():
            return f"Path not found: {path}"
        try:
            result = subprocess.run(
                ["grep", "-r", "-n", "--", pattern, str(search_path)],
                capture_output=True,
                text=True,
                timeout=MAX_SEARCH_TIMEOUT_SECONDS,
                cwd=self.repo_path,  # consistent with _run_command
            )
            return result.stdout[:MAX_COMMAND_OUTPUT_BYTES]
        except Exception as exc:
            return f"Error searching files: {exc}"

    def _task_complete(self, summary: str) -> str:
        """Signal task completion."""
        # If no files were written via write_file, the agent may have produced
        # changes through shell tools (patch, code generators, etc.) without
        # going through the engine's tracking mechanism. Check git status and
        # prompt the agent to review and stage those files rather than silently
        # completing with nothing pushed.
        if not self._files_changed:
            try:
                r = subprocess.run(
                    ["git", "status", "--short"],
                    capture_output=True, text=True,
                    cwd=self.repo_path, timeout=10,
                )
                if r.returncode == 0 and r.stdout.strip():
                    status_lines = r.stdout.strip().splitlines()
                    flagged = [
                        line[3:].strip() for line in status_lines
                        if any(pat in line[3:].lower() for pat in _SENSITIVE_FILE_PATTERNS)
                    ]
                    files_display = "\n".join(f"  {line}" for line in status_lines)
                    warning = ""
                    if flagged:
                        warning = (
                            "\n\n⚠️  WARNING: the following files may contain secrets"
                            " — do NOT stage them:\n"
                            + "\n".join(f"  {f}" for f in flagged)
                        )
                    return (
                        "No files were tracked via write_file, so nothing will be"
                        " committed yet.\ngit status shows the following changes:\n"
                        f"{files_display}{warning}\n\n"
                        "If these files are part of your implementation and do not"
                        " contain secrets, stage each one with"
                        " run_command('git add <file>') and then call task_complete again."
                    )
            except Exception:
                pass  # git not available or repo not initialised — fall through
        else:
            # Files were tracked via write_file. Verify at least one differs from HEAD.
            # If git status shows none of them as modified/new, git add will silently
            # stage nothing and the commit will be skipped — match the real failure mode
            # seen when an agent re-writes a file with identical content.
            try:
                r = subprocess.run(
                    ["git", "status", "--short", "--"] + list(self._files_changed),
                    capture_output=True, text=True,
                    cwd=self.repo_path, timeout=10,
                )
                if r.returncode == 0 and not r.stdout.strip():
                    files_display = "\n".join(f"  {f}" for f in self._files_changed)
                    return (
                        "The following files were written via write_file but appear"
                        " identical to the already-committed version — git add will"
                        " stage nothing and no commit will be created:\n"
                        f"{files_display}\n\n"
                        "This usually means the implementation you wrote matches the"
                        " existing code. Check whether the task is already complete,"
                        " or review the files to confirm your changes are actually"
                        " different from what is currently committed."
                    )
            except Exception:
                pass  # git not available — fall through to normal completion
        self._task_summary = summary
        return f"Task completed: {summary}"

    def _sanitize_log(self, text: str) -> str:
        """Replace the GitLab token with *** in text before writing to logs."""
        if self.gitlab_token:
            return text.replace(self.gitlab_token, "***")
        return text

    @staticmethod
    def _classify_llm_error(exc: Exception) -> tuple[bool, bool]:
        """Classify a provider exception for the retry policy.

        Thin wrapper around :func:`factory.core.resilience.classify_llm_error`
        kept as a static method so existing call sites and tests that patch
        ``CodeExecutionEngine._classify_llm_error`` continue to work.
        """
        return _classify_llm_error_impl(exc)

    # Conventional commit prefixes — used to avoid double-prefixing summaries
    # that already carry one (e.g. an agent that writes "fix: ..." as its summary).
    _CONV_PREFIXES = (
        "feat:", "fix:", "docs:", "chore:", "refactor:",
        "test:", "style:", "ci:", "perf:", "build:",
    )
    # Agent summaries that start with these phrases also skip the "feat:" prefix
    # for readability, but they are not conventional-commit types — they are
    # system-prompt outputs ("Already implemented: …") that would read oddly as
    # "feat: Already implemented: …". Kept separate from _CONV_PREFIXES to keep
    # the prefix list semantically homogeneous.
    _SKIP_FEAT_PHRASES = ("already implemented:",)

    def _git_commit_and_push(self, summary: str) -> str:
        """Commit all written files and push the branch to GitLab.

        Returns the commit SHA on success, empty string on failure.
        The GitLab token is injected via GIT_CONFIG_* environment variables
        (git ≥ 2.31) so it never appears in subprocess argv or .git/config.
        Note: GIT_CONFIG_VALUE_0 is visible in the *child* process's
        /proc/<pid>/environ for the lifetime of the push — an accepted
        trade-off vs. argv/config exposure. The exposure window is bounded
        by the push subprocess timeout (120 s). The parent process's
        os.environ is never mutated; push_env is a filtered dict passed only
        to the child subprocess.
        A per-repo file lock prevents concurrent runs from stomping each other's
        git index.
        """
        if not self._files_changed:
            # The agent may have staged files directly via run_command('git add ...')
            # after being prompted by _task_complete. Check for pre-staged changes
            # before bailing — if any exist, fall through and commit them.
            try:
                chk = subprocess.run(
                    ["git", "diff", "--cached", "--quiet"],
                    capture_output=True, cwd=self.repo_path,
                )
                if chk.returncode == 0:  # 0 = index is clean, nothing to commit
                    return ""
                # returncode != 0 means staged changes exist — fall through.
            except Exception:
                return ""

        # Reject branch names starting with '-' — git would interpret them as flags.
        if self.branch_name.startswith("-"):
            logger.error("Invalid branch name %r starts with '-'; skipping commit", self.branch_name)
            return ""

        # Allowlist branch names to safe characters before passing to any subprocess.
        # This catches null bytes, whitespace, and shell metacharacters that could
        # cause unexpected behaviour in git commands even with shell=False, and
        # provides a second layer of defence beyond check-ref-format.
        if not re.fullmatch(r"[a-zA-Z0-9._/\-]+", self.branch_name):
            logger.error("Branch name %r contains invalid characters; skipping commit", self.branch_name)
            return ""

        repo = Path(self.repo_path).resolve()

        def _run(args: list[str], timeout: int = 60, env=None) -> subprocess.CompletedProcess:
            return subprocess.run(
                args, capture_output=True, text=True,
                cwd=repo, timeout=timeout, env=env,
            )

        # Validate against git's full branch naming rules (catches '..', null bytes, etc.).
        # Order matters: the '-' prefix guard must run BEFORE check-ref-format because
        # check-ref-format parses its argument via getopt — a branch name starting with
        # '-' would be interpreted as an unknown flag and cause an error rather than a
        # clean "invalid name" rejection. The '-' guard handles that case, then
        # check-ref-format handles all remaining naming rules.
        r = _run(["git", "check-ref-format", "--branch", self.branch_name])  # validate name
        if r.returncode != 0:
            logger.error("Invalid branch name %r rejected by git; skipping commit", self.branch_name)
            return ""

        # Verify git supports GIT_CONFIG_* env injection — cached across all instances.
        if not self._git_version_ok():
            logger.error(
                "git %s installed; GIT_CONFIG_* env vars require git ≥ 2.31 — push skipped",
                ".".join(str(x) for x in self._cached_git_version),
            )
            return ""

        # Serialize all git operations per repo to avoid index stomping when
        # multiple engine instances run concurrently against the same working tree.
        # Timeout is the lock *acquisition* wait — 60 s is long enough that a
        # concurrent run finishing normally won't cause a spurious failure, but
        # short enough that a crashed prior run doesn't block the next one for
        # minutes. The push itself has a separate timeout of 120 s (see _run call).
        git_dir = repo / ".git"
        if git_dir.is_dir():
            lock_path = git_dir / "factory-push.lock"
        else:
            # git worktree: .git is a file, not a directory. Fall back to a
            # process-wide temp lock keyed by the repo name so concurrent
            # worktree-based engines still serialise against each other.
            lock_path = Path(tempfile.gettempdir()) / f"factory-push-{repo.name}.lock"
        try:
            with FileLock(str(lock_path), timeout=60):
                committed = False  # tracks whether `git commit` has completed
                # Committer identity injected via env vars rather than `git config --local`
                # to avoid permanently mutating .git/config (which persists after the run
                # and would affect any subsequent manual commits in the repo).
                # Use the same allowlist-filtered base as push_env so the GitLab token
                # (which git commit does not need) is not forwarded unnecessarily.
                # Minimal env for git commit: HOME/PATH (for git binary and config
                # lookup), TMPDIR/TEMP/TMP (git writes COMMIT_EDITMSG to a temp dir
                # and macOS uses a session-scoped TMPDIR that differs from /tmp),
                # plus the four identity vars. All other GIT_* vars — including
                # GIT_CONFIG_* that a CI system might inject for GPG signing or hook
                # config — are intentionally excluded. Factory commits are unsigned
                # and do not rely on external hook infrastructure.
                identity_env = {
                    k: os.environ[k]
                    for k in ("HOME", "PATH", "TMPDIR", "TEMP", "TMP",
                              "LANG", "LC_ALL", "LC_MESSAGES")
                    if k in os.environ
                }
                identity_env.update({
                    "GIT_AUTHOR_NAME": "SoftwareTeamFabrik",
                    "GIT_AUTHOR_EMAIL": "factory@softwareteamfabrik.invalid",
                    "GIT_COMMITTER_NAME": "SoftwareTeamFabrik",
                    "GIT_COMMITTER_EMAIL": "factory@softwareteamfabrik.invalid",
                })
                # Ensure a temp dir is always present — git needs it for COMMIT_EDITMSG.
                # Linux CI often has none of TMPDIR/TEMP/TMP set; fall back to /tmp.
                if not any(k in identity_env for k in ("TMPDIR", "TEMP", "TMP")):
                    identity_env["TMPDIR"] = tempfile.gettempdir()
                # Initialized here (before try:) so _unstage_this_run is always callable
                # in except/finally even when an exception fires before the staging loop.
                staged_rel_paths: list[str] = []

                def _unstage_this_run() -> None:
                    # Captures staged_rel_paths by reference — intentional: the list
                    # is populated in the staging loop after this definition, and the
                    # closure must see the populated state at call time (finally/except).
                    # Uses identity_env so CI secrets are not forwarded to child process.
                    if staged_rel_paths:
                        try:
                            _run(["git", "restore", "--staged", "--"] + staged_rel_paths, timeout=10, env=identity_env)
                        except Exception as exc:
                            # Log rather than silently swallow: if restore fails (e.g.
                            # the index is locked by another process), the staged
                            # changes survive and may be accidentally included in a
                            # future commit by a subsequent engine run on the same tree.
                            logger.warning("git restore --staged failed during cleanup: %s", exc)

                try:
                    # Switch to the target branch, or create it. Skip the switch when
                    # already on it to avoid disturbing staged state from a prior step.
                    # Only fall through to plain `checkout` on the specific "already exists"
                    # error — any other failure (detached HEAD, invalid name) is fatal.
                    r_cur = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], env=identity_env)  # current branch
                    current_branch = r_cur.stdout.strip() if r_cur.returncode == 0 else ""
                    if current_branch != self.branch_name:
                        # git switch is available since 2.23 (well within the 2.31
                        # floor we already require) and is distinct from the checkout
                        # subcommand blocked for agents in _GIT_BLOCKED_SUBCOMMANDS.
                        r = _run(["git", "switch", "-c", self.branch_name], env=identity_env)  # create + switch
                        if r.returncode != 0:
                            if "already exists" in r.stderr:
                                r2 = _run(["git", "switch", self.branch_name], env=identity_env)  # switch only
                                if r2.returncode != 0:
                                    logger.warning("git switch %s failed: %s", self.branch_name, self._sanitize_log(r2.stderr)[:200])
                                    return ""  # pre-staging: staged_rel_paths empty; finally _unstage_this_run is no-op
                            else:
                                logger.warning("git switch -c %s failed: %s", self.branch_name, self._sanitize_log(r.stderr)[:200])
                                return ""  # pre-staging: staged_rel_paths empty; finally _unstage_this_run is no-op

                    # All returns above this point are pre-staging: staged_rel_paths is
                    # still empty, so the finally block's _unstage_this_run() is a no-op.
                    # Any return added *below* this line that fires after git add calls
                    # must go through the finally block (not bare return) for cleanup.

                    # Stage files. `--` prevents filenames starting with `-` from being
                    # treated as flags. is_relative_to() guards against path traversal
                    # (str.startswith is unsafe: /app/repo-evil starts with /app/repo).
                    # This is a defense-in-depth second line — the primary check should
                    # be in write_file before the path is added to self._files_changed.
                    # Use repo-relative paths (not absolute) to avoid git add behaving
                    # unexpectedly with some git versions when cwd is the repo root.
                    rejected_paths: list[str] = []   # security-rejected (never passed to git)
                    add_failed_paths: list[str] = [] # git add returned non-zero
                    for path in self._files_changed:
                        # Reject empty string and "." — (repo / "").resolve() == repo,
                        # so `git add -- .` would stage the entire working tree.
                        if not path or path == ".":
                            logger.warning("Rejecting invalid path %r in _files_changed (security)", path)
                            rejected_paths.append(path)
                            continue
                        # Absolute paths silently replace `repo` in the `/` join,
                        # making the is_relative_to check meaningless.
                        if Path(path).is_absolute():
                            logger.warning("Rejecting absolute path in _files_changed (security): %s", path)
                            rejected_paths.append(path)
                            continue
                        safe = (repo / path).resolve()
                        if not safe.is_relative_to(repo):
                            logger.warning("Rejecting path outside repo root (security): %s", path)
                            rejected_paths.append(path)
                            continue
                        rel = str(safe.relative_to(repo))
                        r = _run(["git", "add", "--", rel], env=identity_env)
                        if r.returncode != 0:
                            logger.warning("git add failed for %s: %s", path, self._sanitize_log(r.stderr)[:200])
                            add_failed_paths.append(path)
                        else:
                            staged_rel_paths.append(rel)

                    if rejected_paths:
                        logger.error(
                            "Aborting commit: %d path(s) rejected for security reasons: %s",
                            len(rejected_paths), ", ".join(rejected_paths),
                        )
                        _unstage_this_run()
                        return ""
                    if add_failed_paths:
                        # Abort rather than commit a partial set — a commit missing files
                        # would be harder to debug and recover from than no commit at all.
                        logger.error(
                            "Aborting commit: %d file(s) failed to stage: %s",
                            len(add_failed_paths), ", ".join(add_failed_paths),
                        )
                        _unstage_this_run()
                        return ""

                    # Abort if nothing was actually staged (all adds failed or skipped).
                    r = _run(["git", "diff", "--cached", "--quiet"], env=identity_env)  # check staged changes
                    if r.returncode == 0:  # exit 0 means no staged changes
                        logger.warning("Nothing staged to commit — all git add calls may have failed")
                        return ""

                    # Build commit subject. Strip \r before splitting so Windows-style
                    # CRLF in agent responses doesn't leave a trailing \r in the subject.
                    # Take only the first line to prevent injecting a commit body.
                    # Avoid prepending `feat:` when the summary already has a prefix.
                    # _truncate_subject appends ASCII '...' (3 chars) when the line is cut,
                    # keeping the result within 72 chars on both character and byte counts.
                    first_line = summary.replace("\r", "").split("\n", 1)[0].strip()
                    if not first_line:
                        # Guard against empty or whitespace-only summaries — fall back
                        # to the issue title so the commit subject is always meaningful.
                        first_line = (self.issue_title or "automated changes").strip()
                    _fl = first_line.lower()
                    if (any(_fl.startswith(p) for p in self._CONV_PREFIXES)
                            or any(_fl.startswith(p) for p in self._SKIP_FEAT_PHRASES)):
                        commit_msg = self._truncate_subject(first_line)
                    else:
                        commit_msg = self._truncate_subject(f"feat: {first_line}")

                    r = _run(["git", "commit", "-m", commit_msg], env=identity_env)  # create commit
                    if r.returncode != 0:
                        logger.warning("git commit failed: %s", self._sanitize_log(r.stderr)[:200])
                        # Unstage so the next run's `git diff --cached` check starts clean.
                        _unstage_this_run()
                        return ""
                    committed = True

                    # stdout is the commit SHA — not sensitive, no sanitization needed.
                    r = _run(["git", "rev-parse", "HEAD"], env=identity_env)  # retrieve commit SHA
                    if r.returncode != 0:
                        # The commit exists in git history but the SHA cannot be
                        # retrieved. Set _push_failed so callers see push_failed=True
                        # even though commit_sha="" — this signals "something went wrong
                        # after committing" rather than "nothing happened".
                        logger.error("git rev-parse HEAD failed after commit; commit exists but SHA unknown: %s",
                                     self._sanitize_log(r.stderr)[:200])
                        self._push_failed = True
                        return ""
                    sha = r.stdout.strip()
                    # Accept 40-char SHA-1 or 64-char SHA-256 (git ≥ 2.29 object format).
                    if not re.fullmatch(r"[0-9a-f]{40,64}", sha):
                        logger.error("Unexpected rev-parse output %r; expected 40- or 64-char SHA", sha[:80])
                        return ""

                    # Validate origin exists and is HTTPS.
                    r_origin = _run(["git", "remote", "get-url", "origin"], env=identity_env)
                    if r_origin.returncode != 0:
                        logger.error("No 'origin' remote configured; commit succeeded but push skipped")
                        self._push_failed = True
                        return sha
                    origin_url = r_origin.stdout.strip()

                    # Auth strategy: embed the token in the remote URL using the
                    # oauth2 basic-auth format — the same approach used for clone.
                    # This is universally supported by GitLab's git HTTPS endpoint,
                    # unlike http.extraHeader (Bearer) which GitLab does not support
                    # for the git smart-HTTP protocol.
                    push_env = None
                    if self.gitlab_token and (origin_url.startswith("https://") or origin_url.startswith("http://")):
                        from urllib.parse import urlparse, urlunparse
                        parsed = urlparse(origin_url)
                        # Only embed if credentials are not already present.
                        if not parsed.username:
                            netloc = f"oauth2:{self.gitlab_token}@{parsed.hostname}"
                            if parsed.port:
                                netloc += f":{parsed.port}"
                            # Force https: http→https redirect strips embedded credentials.
                            auth_url = urlunparse(parsed._replace(scheme="https", netloc=netloc))
                            _run(["git", "remote", "set-url", "origin", "--", auth_url],
                                 env=identity_env)
                        push_env = {
                            k: v for k, v in os.environ.items()
                            if k in self._PUSH_ENV_ALLOWLIST
                            or k.startswith(self._PUSH_ENV_PREFIXES)
                        }
                        push_env["GIT_TERMINAL_PROMPT"] = "0"
                    elif self.gitlab_token and not (origin_url.startswith("https://") or origin_url.startswith("http://")):
                        safe_origin = re.sub(r"://[^@]+@", "://<redacted>@", origin_url)
                        logger.warning(
                            "origin remote is not HTTPS (%r); token auth requires HTTPS"
                            " — skipping push",
                            self._sanitize_log(safe_origin)[:80],
                        )
                        self._push_failed = True
                        return sha

                    # `--` terminates options so a branch name starting with `-` isn't
                    # parsed as a flag (already rejected above, but defence-in-depth).
                    r = _run(["git", "push", "-u", "origin", "--", self.branch_name], timeout=120, env=push_env)

                    if r.returncode != 0:
                        # Sanitize the full stderr before truncating so a token near the
                        # boundary isn't split across the cut point.
                        logger.warning("git push failed: %s", self._sanitize_log(r.stderr)[:200])
                        self._progress(f"  [yellow]⚠ git push failed:[/yellow] {self._sanitize_log(r.stderr)[:200]}")
                        # The commit is local; the remote branch does NOT exist yet.
                        # _push_failed lets callers warn that commit_sha being set
                        # does not mean the remote branch is available for MR creation.
                        self._push_failed = True
                        return sha

                    self._progress(f"  [green]✓ pushed[/green] {self.branch_name} ({sha[:8]})")
                    return sha

                except subprocess.TimeoutExpired as exc:
                    logger.error("git command timed out: %s", exc.cmd)
                    self._progress("  [red]✗ git timed out[/red]")
                    return ""
                except Exception as exc:
                    safe_exc = self._sanitize_log(str(exc))
                    logger.error("git commit/push error: %s", safe_exc)
                    self._progress(f"  [red]✗ git error:[/red] {safe_exc}")
                    return ""
                finally:
                    # Guarantee index cleanup on every non-committed exit path —
                    # covers exceptions that fire between staging and `git commit`.
                    # Once committed, staged_rel_paths is empty so this is a no-op.
                    if not committed:
                        _unstage_this_run()

        except FileLockTimeout:
            logger.error("Could not acquire git lock within 60 s; another push may be running")
            return ""

    def _ensure_git_repo(self) -> None:
        """Clone the GitLab project into a temp dir if repo_path is not a git repo.

        Auth strategy: GIT_CONFIG_* header injection for git ≥ 2.31; token-in-URL
        fallback for older git (common in container base images). repo_path is
        updated in-place on success.
        """
        if (self.repo_path / ".git").exists():
            return

        if not self.clone_url:
            logger.warning(
                "repo_path %s has no .git directory and no clone_url was provided — "
                "git operations will fail",
                self.repo_path,
            )
            return

        # Deterministic temp dir per project so repeated runs reuse the clone.
        safe_branch = re.sub(r"[^a-zA-Z0-9._-]", "_", self.branch_name)[:60]
        clone_dir = Path(tempfile.gettempdir()) / f"factory_{self.gitlab_project_id}_{safe_branch}"

        # GIT_TERMINAL_PROMPT=0: exit immediately on auth failure, never block.
        base_env = {
            k: v for k, v in os.environ.items()
            if k in self._PUSH_ENV_ALLOWLIST
            or k.startswith(self._PUSH_ENV_PREFIXES)
        }
        base_env["GIT_TERMINAL_PROMPT"] = "0"

        # Auth strategy: embed token in the HTTPS clone URL using the oauth2
        # basic-auth format that GitLab's git HTTP smart protocol accepts.
        # The GIT_CONFIG_* Authorization: Bearer approach is NOT supported by
        # GitLab's git endpoint (only the REST API accepts it), so we use
        # oauth2:TOKEN@host unconditionally — this works with all git versions.
        clone_url = self.clone_url
        clone_env = base_env
        if self.gitlab_token:
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(self.clone_url)
            if parsed.scheme in ("http", "https"):
                netloc = f"oauth2:{self.gitlab_token}@{parsed.hostname}"
                if parsed.port:
                    netloc += f":{parsed.port}"
                # Force https: http→https redirect strips embedded credentials.
                clone_url = urlunparse(parsed._replace(scheme="https", netloc=netloc))

        if not (clone_dir / ".git").exists():
            self._progress(f"[dim]Cloning repository to {clone_dir}…[/dim]")
            r = subprocess.run(
                ["git", "clone", "--depth", "1", "--no-single-branch",
                 "--", clone_url, str(clone_dir)],
                capture_output=True, text=True,
                stdin=subprocess.DEVNULL,
                env=clone_env,
                timeout=120,
            )
            if r.returncode != 0:
                logger.error("git clone failed: %s", self._sanitize_log(r.stderr)[:300])
                self._progress(f"[red]✗ git clone failed: {self._sanitize_log(r.stderr)[:200]}[/red]")
                return
            self._progress(f"[green]✓[/green] Repository cloned")

        # Fetch and checkout the working branch.
        r_fetch = subprocess.run(
            ["git", "fetch", "origin", "--", self.branch_name],
            capture_output=True, text=True,
            stdin=subprocess.DEVNULL,
            env=clone_env, cwd=clone_dir, timeout=60,
        )
        if r_fetch.returncode != 0:
            logger.warning("git fetch %s failed: %s", self.branch_name,
                           self._sanitize_log(r_fetch.stderr)[:200])
        r_co = subprocess.run(
            ["git", "checkout", self.branch_name],
            capture_output=True, text=True,
            stdin=subprocess.DEVNULL,
            cwd=clone_dir, timeout=30,
        )
        if r_co.returncode != 0:
            # Branch doesn't exist locally — create it tracking origin.
            subprocess.run(
                ["git", "checkout", "-b", self.branch_name,
                 f"origin/{self.branch_name}"],
                capture_output=True, text=True,
                stdin=subprocess.DEVNULL,
                cwd=clone_dir, timeout=30,
            )

        self.repo_path = clone_dir

    def run(self, router: Any, system: str = _EXECUTION_SYSTEM) -> ExecutionResult:
        """Run the appropriate agent loop based on provider."""
        self._ensure_git_repo()
        if self.agent.provider == "ollama":
            result = self._agent_loop_ollama(router, system)
        elif self.agent.provider == "anthropic":
            result = self._agent_loop_anthropic(router, system)
        else:
            # Stub — implement the provider's agent loop here when adding support.
            # INVARIANT: any provider loop (including this stub) MUST set
            # needs_continuation=True whenever it fails or cannot complete the task.
            # The commit/push block below is guarded by `not needs_continuation`,
            # so a provider that forgets this flag on failure will trigger an
            # unintended commit of whatever files were written up to that point.
            logger.warning(
                "Provider %r has no agent loop implementation; task will not be executed",
                self.agent.provider,
            )
            result = ExecutionResult(
                response=f"Provider '{self.agent.provider}' is not yet implemented",
                mode=self.agent.execution_mode,
                needs_continuation=True,
            )

        # Commit and push for any provider that completes with file changes.
        # Placed outside the if/else so new providers get this step automatically.
        # Use self._files_changed (live list) — result.files_changed is a mid-run snapshot.
        if self._files_changed and not result.needs_continuation:
            # Prefer the explicit task_complete summary over the full response text.
            summary = self._task_summary or result.response
            # Snapshot before the commit so result.files_changed reflects what was
            # written regardless of whether _git_commit_and_push succeeds.
            result.files_changed = list(self._files_changed)
            result.commit_sha = self._git_commit_and_push(summary)
            if result.commit_sha:
                # Clear the change list after a successful commit so that calling
                # run() again on the same instance does not re-commit already-committed
                # files. Engine instances are intended for single use, but clearing here
                # is a safe guard against accidental reuse.
                self._files_changed.clear()
            if self._push_failed:
                # commit_sha is set but the remote branch does not exist yet.
                # Callers must not attempt to open an MR against this branch.
                result.push_failed = True
                logger.warning(
                    "Push failed; commit %s is local only — remote branch '%s' may not exist",
                    result.commit_sha[:8] if result.commit_sha else "(none)",
                    self.branch_name,
                )

        if result.needs_continuation:
            # Partial run: reflect any files written so far so callers know
            # what changed even though the task was not completed.
            result.files_changed = list(self._files_changed)

        return result