"""Unit tests for CodeExecutionEngine._git_commit_and_push."""
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from factory.core.competence import AgentProfile
from factory.core.execution_engine import CodeExecutionEngine
from filelock import Timeout as FileLockTimeout


@pytest.fixture(autouse=True)
def reset_git_version_cache():
    """Reset the cached git version between tests for isolation."""
    CodeExecutionEngine._cached_git_version = ()
    yield
    CodeExecutionEngine._cached_git_version = ()


def _make_engine(tmp_path: Path, token: str = "tok123") -> CodeExecutionEngine:
    agent = AgentProfile(
        name="developer",
        display_name="Developer",
        model="qwen2.5-coder:7b",
        provider="ollama",
        execution_mode="execute",
        system_prompt="",
        task_labels=[],
    )
    engine = CodeExecutionEngine(
        agent=agent,
        repo_path=tmp_path,
        issue_title="Add widget",
        issue_description="Build it.",
        gitlab_token=token,
        gitlab_project_id="42",
        gitlab_url="https://gitlab.example.com",
        branch_name="issue-1",
    )
    return engine


def _completed_proc(returncode=0, stdout="", stderr="") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class TestGitVersionOk:
    def test_cached_across_instances(self):
        CodeExecutionEngine._cached_git_version = ()
        with patch("subprocess.run", return_value=_completed_proc(stdout="git version 2.40.1")) as mock:
            r1 = CodeExecutionEngine._git_version_ok()
            r2 = CodeExecutionEngine._git_version_ok()
        assert r1 is True
        assert r2 is True
        assert mock.call_count == 1  # second call hits cache

    def test_returns_false_for_old_git(self):
        CodeExecutionEngine._cached_git_version = (2, 30)
        assert CodeExecutionEngine._git_version_ok() is False

    def test_returns_false_when_git_not_found(self):
        CodeExecutionEngine._cached_git_version = ()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = CodeExecutionEngine._git_version_ok()
        assert result is False
        assert CodeExecutionEngine._cached_git_version == (0, 0)


class TestTruncateSubject:
    def test_short_string_unchanged(self):
        assert CodeExecutionEngine._truncate_subject("hello") == "hello"

    def test_exactly_max_len_unchanged(self):
        s = "x" * 72
        assert CodeExecutionEngine._truncate_subject(s) == s

    def test_long_string_truncated_with_ellipsis(self):
        s = "x" * 80
        result = CodeExecutionEngine._truncate_subject(s)
        assert len(result) == 72
        assert result.endswith("...")


class TestGitCommitAndPush:
    def _run_side_effects(self, *, ref_format_ok=True,
                          checkout_ok=True,
                          add_ok=True, commit_ok=True, rev_parse_sha="a" * 40,
                          origin_url="https://gitlab.example.com/org/repo.git",
                          push_ok=True, num_files: int = 1):
        """Build a side_effect list for subprocess.run assuming git version is cached.

        Tests must set ``CodeExecutionEngine._cached_git_version = (2, 40)`` before
        calling this so that ``_git_version_ok`` does not consume a slot.

        ``num_files`` controls how many ``git add`` slots are inserted.  Each
        file in ``_files_changed`` triggers one subprocess call, so tests with
        multiple changed files must pass the matching ``num_files`` value or use
        a custom dispatch-style side_effect instead.
        """
        results = [
            _completed_proc(returncode=0 if ref_format_ok else 1),             # check-ref-format
            _completed_proc(stdout="main"),                                     # rev-parse abbrev-ref
            _completed_proc(returncode=0 if checkout_ok else 1, stderr=""),    # switch -c
        ]
        for _ in range(num_files):
            results.append(_completed_proc(returncode=0 if add_ok else 1, stderr=""))  # git add
        results += [
            _completed_proc(returncode=1),                                      # diff --cached (changes → rc=1)
            _completed_proc(returncode=0 if commit_ok else 1, stderr=""),      # git commit
            _completed_proc(stdout=rev_parse_sha + "\n"),                      # rev-parse HEAD
            _completed_proc(stdout=origin_url + "\n"),                         # remote get-url
            _completed_proc(),                                                  # remote set-url (embed creds)
            _completed_proc(                                                    # git push
                returncode=0 if push_ok else 1,
                stderr="" if push_ok else "push failed msg",
            ),
        ]
        return results

    def test_happy_path_returns_sha(self, tmp_path):
        CodeExecutionEngine._cached_git_version = (2, 40)
        engine = _make_engine(tmp_path)
        engine._files_changed = ["foo.py"]

        side_effects = self._run_side_effects()
        with patch("subprocess.run", side_effect=side_effects):
            sha = engine._git_commit_and_push("feat: add widget")

        assert sha == "a" * 40

    def test_no_files_changed_returns_empty(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine._files_changed = []
        sha = engine._git_commit_and_push("summary")
        assert sha == ""

    def test_branch_starting_with_dash_rejected(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine._files_changed = ["foo.py"]
        engine.branch_name = "-bad-branch"
        assert engine._git_commit_and_push("summary") == ""

    def test_old_git_version_skips_push(self, tmp_path):
        CodeExecutionEngine._cached_git_version = (2, 30)
        engine = _make_engine(tmp_path)
        engine._files_changed = ["foo.py"]
        with patch("subprocess.run", return_value=_completed_proc(returncode=1)) as mock:
            sha = engine._git_commit_and_push("summary")
        assert sha == ""

    def test_empty_path_rejected_as_security_violation(self, tmp_path):
        CodeExecutionEngine._cached_git_version = (2, 40)
        engine = _make_engine(tmp_path)
        engine._files_changed = [""]

        with patch("subprocess.run", side_effect=self._run_side_effects()):
            sha = engine._git_commit_and_push("summary")

        # empty path is a security rejection → abort → return ""
        assert sha == ""

    def test_dot_path_rejected_as_security_violation(self, tmp_path):
        CodeExecutionEngine._cached_git_version = (2, 40)
        engine = _make_engine(tmp_path)
        engine._files_changed = ["."]

        with patch("subprocess.run", side_effect=self._run_side_effects()):
            sha = engine._git_commit_and_push("summary")

        assert sha == ""

    def test_absolute_path_rejected_as_security_violation(self, tmp_path):
        CodeExecutionEngine._cached_git_version = (2, 40)
        engine = _make_engine(tmp_path)
        engine._files_changed = ["/etc/passwd"]

        with patch("subprocess.run", side_effect=self._run_side_effects()):
            sha = engine._git_commit_and_push("summary")

        assert sha == ""

    def test_git_add_failure_aborts_commit(self, tmp_path):
        CodeExecutionEngine._cached_git_version = (2, 40)
        engine = _make_engine(tmp_path)
        engine._files_changed = ["foo.py"]

        side_effects = self._run_side_effects(add_ok=False)
        with patch("subprocess.run", side_effect=side_effects):
            sha = engine._git_commit_and_push("summary")

        assert sha == ""

    def test_commit_failure_unstages_and_returns_empty(self, tmp_path):
        CodeExecutionEngine._cached_git_version = (2, 40)
        engine = _make_engine(tmp_path)
        engine._files_changed = ["foo.py"]

        side_effects = self._run_side_effects(commit_ok=False)
        side_effects.append(_completed_proc())  # restore --staged cleanup
        with patch("subprocess.run", side_effect=side_effects) as mock:
            sha = engine._git_commit_and_push("summary")

        assert sha == ""
        restore_calls = [c for c in mock.call_args_list if "restore" in str(c)]
        assert restore_calls

    def test_push_failure_returns_sha(self, tmp_path):
        """Push failure is non-fatal — the commit SHA is still returned."""
        CodeExecutionEngine._cached_git_version = (2, 40)
        engine = _make_engine(tmp_path)
        engine._files_changed = ["foo.py"]

        side_effects = self._run_side_effects(push_ok=False)
        with patch("subprocess.run", side_effect=side_effects):
            sha = engine._git_commit_and_push("summary")

        assert sha == "a" * 40

    def test_checkout_branch_already_exists_falls_through(self, tmp_path):
        """When checkout -b fails with 'already exists', a plain checkout is tried."""
        CodeExecutionEngine._cached_git_version = (2, 40)
        engine = _make_engine(tmp_path)
        engine._files_changed = ["foo.py"]

        rev_parse_sha = "b" * 40
        calls: list[list[str]] = []

        def dispatch(args, **kwargs):
            calls.append(list(args))
            if args[:2] == ["git", "check-ref-format"]:
                return _completed_proc()
            if args[:3] == ["git", "rev-parse", "--abbrev-ref"]:
                return _completed_proc(stdout="main")
            if args[:3] == ["git", "switch", "-c"]:
                # Simulate "branch already exists"
                return _completed_proc(returncode=1, stderr="fatal: A branch named 'issue-1' already exists.")
            if args[:2] == ["git", "switch"] and args[2:3] == ["issue-1"]:
                return _completed_proc()  # plain switch succeeds
            if args[:2] == ["git", "add"]:
                return _completed_proc()
            if args[:3] == ["git", "diff", "--cached"]:
                return _completed_proc(returncode=1)  # staged changes
            if args[:2] == ["git", "commit"]:
                return _completed_proc()
            if args[:3] == ["git", "rev-parse", "HEAD"]:
                return _completed_proc(stdout=rev_parse_sha + "\n")
            if args[:4] == ["git", "remote", "get-url", "origin"]:
                return _completed_proc(stdout="https://gitlab.example.com/org/repo.git\n")
            if args[:2] == ["git", "push"]:
                return _completed_proc()
            return _completed_proc()

        with patch("subprocess.run", side_effect=dispatch):
            sha = engine._git_commit_and_push("summary")

        assert sha == rev_parse_sha
        # Verify both switch -c and the fallback plain switch were called
        switch_c_calls = [c for c in calls if c[:3] == ["git", "switch", "-c"]]
        plain_switch_calls = [c for c in calls if c[:2] == ["git", "switch"] and "-c" not in c]
        assert switch_c_calls, "switch -c was not attempted"
        assert plain_switch_calls, "fallback plain switch was not attempted"

    def test_already_on_target_branch_skips_checkout(self, tmp_path):
        """When already on the target branch, git checkout must not be called."""
        CodeExecutionEngine._cached_git_version = (2, 40)
        engine = _make_engine(tmp_path)
        engine._files_changed = ["foo.py"]

        rev_parse_sha = "c" * 40
        calls: list[list[str]] = []

        def dispatch(args, **kwargs):
            calls.append(list(args))
            if args[:2] == ["git", "check-ref-format"]:
                return _completed_proc()
            if args[:3] == ["git", "rev-parse", "--abbrev-ref"]:
                return _completed_proc(stdout="issue-1")  # already on target branch
            if args[:2] == ["git", "add"]:
                return _completed_proc()
            if args[:3] == ["git", "diff", "--cached"]:
                return _completed_proc(returncode=1)
            if args[:2] == ["git", "commit"]:
                return _completed_proc()
            if args[:3] == ["git", "rev-parse", "HEAD"]:
                return _completed_proc(stdout=rev_parse_sha + "\n")
            if args[:4] == ["git", "remote", "get-url", "origin"]:
                return _completed_proc(stdout="https://gitlab.example.com/org/repo.git\n")
            if args[:2] == ["git", "push"]:
                return _completed_proc()
            return _completed_proc()

        with patch("subprocess.run", side_effect=dispatch):
            sha = engine._git_commit_and_push("summary")

        assert sha == rev_parse_sha
        branch_switch_calls = [c for c in calls if "switch" in c or "checkout" in c]
        assert not branch_switch_calls, "branch switch was called despite already being on target branch"

    def test_empty_summary_falls_back_to_issue_title(self, tmp_path):
        CodeExecutionEngine._cached_git_version = (2, 40)
        engine = _make_engine(tmp_path)
        engine._files_changed = ["foo.py"]

        commit_subjects: list[str] = []

        original_side_effects = self._run_side_effects()
        call_idx = 0

        def capture_commit(args, **kwargs):
            nonlocal call_idx
            result = original_side_effects[call_idx]
            call_idx += 1
            if args[:2] == ["git", "commit"]:
                commit_subjects.append(args[args.index("-m") + 1])
            return result

        with patch("subprocess.run", side_effect=capture_commit):
            sha = engine._git_commit_and_push("")  # empty summary

        assert sha == "a" * 40
        assert commit_subjects, "git commit was not called"
        # Fallback to issue_title ("Add widget") when summary is empty
        assert "Add widget" in commit_subjects[0]

    def test_git_trace2_excluded_from_push_env(self, tmp_path):
        """GIT_TRACE2* vars must be excluded from push_env (they log the bearer token).

        The filter `not k.startswith("GIT_TRACE")` already covers GIT_TRACE2,
        GIT_TRACE2_EVENT, and GIT_TRACE2_PERF because "GIT_TRACE" is a prefix of
        each. This test proves that invariant so a future refactor cannot break it.
        """
        CodeExecutionEngine._cached_git_version = (2, 40)
        engine = _make_engine(tmp_path)
        engine._files_changed = ["foo.py"]

        captured_push_envs: list = []
        side_effects = iter(self._run_side_effects())

        def capture_env(args, **kwargs):
            if args[:2] == ["git", "push"]:
                captured_push_envs.append(dict(kwargs.get("env") or {}))
            return next(side_effects)

        trace_vars = {
            "GIT_TRACE": "/tmp/trace.log",
            "GIT_TRACE2": "/tmp/trace2.log",
            "GIT_TRACE2_EVENT": "/tmp/trace2_event.log",
            "GIT_TRACE2_PERF": "/tmp/trace2_perf.log",
            "GIT_TRACE_PACK_ACCESS": "/tmp/pack.log",
        }
        with patch("subprocess.run", side_effect=capture_env), \
             patch.dict(os.environ, trace_vars):
            engine._git_commit_and_push("summary")

        assert captured_push_envs, "git push was never called"
        env = captured_push_envs[0]
        for k in trace_vars:
            assert k not in env, f"{k} must be excluded from push_env"

    def test_no_token_skips_credential_injection(self, tmp_path):
        CodeExecutionEngine._cached_git_version = (2, 40)
        engine = _make_engine(tmp_path, token="")
        assert engine.gitlab_token is None
        engine._files_changed = ["foo.py"]

        captured_push_envs: list = []
        side_effects = iter(self._run_side_effects())

        def capture_env(args, **kwargs):
            if args[:2] == ["git", "push"]:
                captured_push_envs.append(kwargs.get("env"))
            return next(side_effects)

        with patch("subprocess.run", side_effect=capture_env):
            engine._git_commit_and_push("summary")

        assert captured_push_envs, "git push was never called"
        # With no token push_env is None (no credential injection)
        assert captured_push_envs[0] is None
        # GIT_CONFIG_COUNT must not appear when no token is set
        assert not any(
            "GIT_CONFIG_COUNT" in (e or {}) for e in captured_push_envs
        )

    def test_ssh_remote_with_token_skips_push(self, tmp_path):
        """When the remote is SSH and a token is set, push should be skipped."""
        CodeExecutionEngine._cached_git_version = (2, 40)
        engine = _make_engine(tmp_path)
        engine._files_changed = ["foo.py"]

        rev_parse_sha = "d" * 40
        calls: list[list[str]] = []

        def dispatch(args, **kwargs):
            calls.append(list(args))
            if args[:2] == ["git", "check-ref-format"]:
                return _completed_proc()
            if args[:3] == ["git", "rev-parse", "--abbrev-ref"]:
                return _completed_proc(stdout="main")
            if args[:3] == ["git", "switch", "-c"]:
                return _completed_proc()
            if args[:2] == ["git", "add"]:
                return _completed_proc()
            if args[:3] == ["git", "diff", "--cached"]:
                return _completed_proc(returncode=1)
            if args[:2] == ["git", "commit"]:
                return _completed_proc()
            if args[:3] == ["git", "rev-parse", "HEAD"]:
                return _completed_proc(stdout=rev_parse_sha + "\n")
            if args[:4] == ["git", "remote", "get-url", "origin"]:
                # Return an SSH remote URL
                return _completed_proc(stdout="git@gitlab.example.com:org/repo.git\n")
            return _completed_proc()

        with patch("subprocess.run", side_effect=dispatch):
            sha = engine._git_commit_and_push("summary")

        assert sha == rev_parse_sha
        # Verify that git push was NOT called
        push_calls = [c for c in calls if c[:2] == ["git", "push"]]
        assert not push_calls, "git push was called despite SSH remote with token"


class TestConvPrefixes:
    """Test that _CONV_PREFIXES match is case-insensitive and prevents double-prefixing."""

    def _capture_commit_subject(self, engine, summary: str, side_effects: list) -> str:
        subjects: list[str] = []
        call_idx = 0

        def capture(args, **kwargs):
            nonlocal call_idx
            result = side_effects[call_idx]
            call_idx += 1
            if args[:2] == ["git", "commit"]:
                subjects.append(args[args.index("-m") + 1])
            return result

        with patch("subprocess.run", side_effect=capture):
            engine._git_commit_and_push(summary)
        return subjects[0] if subjects else ""

    def test_already_implemented_lowercase_not_prefixed(self, tmp_path):
        """'already implemented:' summaries must not get a 'feat:' prefix."""
        CodeExecutionEngine._cached_git_version = (2, 40)
        engine = _make_engine(tmp_path)
        engine._files_changed = ["foo.py"]
        subject = self._capture_commit_subject(
            engine, "already implemented: widget already exists",
            TestGitCommitAndPush()._run_side_effects(),
        )
        assert subject.startswith("already implemented:"), f"unexpected: {subject!r}"
        assert not subject.startswith("feat:"), f"unexpected 'feat:' prefix: {subject!r}"

    def test_already_implemented_titlecase_not_prefixed(self, tmp_path):
        """'Already implemented:' (title-case, as the system prompt produces) must not get 'feat:'."""
        CodeExecutionEngine._cached_git_version = (2, 40)
        engine = _make_engine(tmp_path)
        engine._files_changed = ["foo.py"]
        subject = self._capture_commit_subject(
            engine, "Already implemented: widget already exists",
            TestGitCommitAndPush()._run_side_effects(),
        )
        assert "Already implemented:" in subject, f"unexpected: {subject!r}"
        assert not subject.startswith("feat:"), f"unexpected 'feat:' prefix: {subject!r}"

    def test_regular_summary_gets_feat_prefix(self, tmp_path):
        """Summaries without a known prefix must have 'feat:' prepended."""
        CodeExecutionEngine._cached_git_version = (2, 40)
        engine = _make_engine(tmp_path)
        engine._files_changed = ["foo.py"]
        subject = self._capture_commit_subject(
            engine, "add widget to the sidebar",
            TestGitCommitAndPush()._run_side_effects(),
        )
        assert subject.startswith("feat:"), f"expected 'feat:' prefix, got: {subject!r}"


class TestRunCommand:
    def test_git_c_flag_push_bypass_blocked(self, tmp_path):
        """git -c key=val push must be blocked even though args[1] is '-c' not 'push'."""
        engine = _make_engine(tmp_path)
        result = engine._run_command("git -c http.proxy=evil push origin")
        assert "not available" in result

    def test_git_allowed_subcommand_passes(self, tmp_path):
        """git status is not in the blocklist and should reach subprocess."""
        engine = _make_engine(tmp_path)
        with patch("subprocess.run", return_value=_completed_proc(stdout="nothing to commit")):
            result = engine._run_command("git status")
        assert "nothing to commit" in result

    def test_disallowed_command_rejected(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._run_command("rm -rf /")
        assert "not allowed" in result

    def test_grep_relative_path_passes(self, tmp_path):
        """grep with a relative path should reach subprocess."""
        engine = _make_engine(tmp_path)
        with patch("subprocess.run", return_value=_completed_proc(stdout="ok")):
            result = engine._run_command("grep foo src/main.py")
        assert result == "ok"

    def test_grep_absolute_path_rejected(self, tmp_path):
        """grep with an absolute path must be rejected."""
        engine = _make_engine(tmp_path)
        result = engine._run_command("grep foo /etc/passwd")
        assert "Absolute paths and parent-directory references are not allowed" in result

    def test_cat_absolute_path_rejected(self, tmp_path):
        """cat with an absolute path must be rejected."""
        engine = _make_engine(tmp_path)
        result = engine._run_command("cat /etc/passwd")
        assert "Absolute paths and parent-directory references are not allowed" in result

    def test_find_absolute_path_rejected(self, tmp_path):
        """find with an absolute path must be rejected."""
        engine = _make_engine(tmp_path)
        result = engine._run_command("find / -name foo")
        assert "Absolute paths and parent-directory references are not allowed" in result

    def test_ls_absolute_path_rejected(self, tmp_path):
        """ls with an absolute path must be rejected."""
        engine = _make_engine(tmp_path)
        result = engine._run_command("ls /etc")
        assert "Absolute paths and parent-directory references are not allowed" in result

    def test_grep_parent_traversal_rejected(self, tmp_path):
        """grep with a parent-directory traversal must be rejected."""
        engine = _make_engine(tmp_path)
        result = engine._run_command("grep foo ../secret.py")
        assert "Absolute paths and parent-directory references are not allowed" in result

    def test_grep_double_dot_rejected(self, tmp_path):
        """grep with a bare '..' must be rejected."""
        engine = _make_engine(tmp_path)
        result = engine._run_command("grep foo ..")
        assert "Absolute paths and parent-directory references are not allowed" in result


class TestListDirectory:
    """Tests for _list_directory path-traversal guard (SEC-02)."""

    def test_happy_path_lists_contents(self, tmp_path):
        engine = _make_engine(tmp_path)
        (tmp_path / "file1.txt").write_text("hello")
        (tmp_path / "subdir").mkdir()
        result = engine._list_directory(".")
        data = __import__("json").loads(result)
        assert "file1.txt" in data["files"]
        assert "subdir" in data["directories"]

    def test_subdir_path(self, tmp_path):
        engine = _make_engine(tmp_path)
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested")
        result = engine._list_directory("subdir")
        data = __import__("json").loads(result)
        assert "nested.txt" in data["files"]

    def test_traversal_outside_repo_denied(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._list_directory("../")
        assert result.startswith("Access denied")
        assert "outside the repository root" in result

    def test_traversal_with_file_component_denied(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._list_directory("../../etc")
        assert result.startswith("Access denied")

    def test_nonexistent_dir_returns_not_found(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._list_directory("does_not_exist")
        assert result.startswith("Directory not found")

    def test_absolute_path_outside_repo_denied(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._list_directory("/etc")
        assert result.startswith("Access denied")


class TestGitlabTokenNormalization:
    def test_empty_string_normalized_to_none(self, tmp_path):
        engine = _make_engine(tmp_path, token="")
        assert engine.gitlab_token is None

    def test_whitespace_only_normalized_to_none(self, tmp_path):
        engine = _make_engine(tmp_path, token="   ")
        assert engine.gitlab_token is None

    def test_valid_token_preserved(self, tmp_path):
        engine = _make_engine(tmp_path, token="glpat-abc123")
        assert engine.gitlab_token == "glpat-abc123"

    def test_token_with_surrounding_whitespace_stripped(self, tmp_path):
        engine = _make_engine(tmp_path, token="  glpat-abc123  ")
        assert engine.gitlab_token == "glpat-abc123"


class TestFileLockTimeout:
    def test_file_lock_timeout_returns_empty_sha(self, tmp_path):
        """When FileLock acquisition times out, _git_commit_and_push must return empty string."""
        CodeExecutionEngine._cached_git_version = (2, 40)
        engine = _make_engine(tmp_path)
        engine._files_changed = ["foo.py"]

        # subprocess.run must be mocked so that `git check-ref-format` (called
        # before FileLock) does not fail because git is absent in the CI container.
        with patch("subprocess.run", return_value=_completed_proc(returncode=0)):
            # Patch the module-level FileLock reference that _git_commit_and_push
            # actually uses. Patching filelock.FileLock would not affect the
            # already-imported binding in execution_engine (Python's name-binding
            # rules mean each module's import is an independent reference).
            with patch("factory.core.execution_engine.FileLock", side_effect=FileLockTimeout("lock held by another process")):
                sha = engine._git_commit_and_push("summary")

        assert sha == ""

    def test_pre_staged_changes_fall_through_to_commit(self, tmp_path):
        """When _files_changed is empty but git diff --cached shows staged content, the commit proceeds.

        This covers the case where the agent called 'git add' via run_command after
        being prompted by _task_complete, then called task_complete a second time.
        The engine detects pre-staged changes via 'git diff --cached --quiet' and
        falls through to commit instead of returning early with an empty SHA.
        """
        CodeExecutionEngine._cached_git_version = (2, 40)
        engine = _make_engine(tmp_path)
        engine._files_changed = []  # no files tracked through write_file

        rev_parse_sha = "e" * 40
        calls: list[list[str]] = []

        def dispatch(args, **kwargs):
            calls.append(list(args))
            if args[:3] == ["git", "diff", "--cached"]:
                # Both the early-return check and the post-staging check must see staged changes.
                return _completed_proc(returncode=1)
            if args[:2] == ["git", "check-ref-format"]:
                return _completed_proc()
            if args[:3] == ["git", "rev-parse", "--abbrev-ref"]:
                return _completed_proc(stdout="main")
            if args[:3] == ["git", "switch", "-c"]:
                return _completed_proc()
            if args[:2] == ["git", "commit"]:
                return _completed_proc()
            if args[:3] == ["git", "rev-parse", "HEAD"]:
                return _completed_proc(stdout=rev_parse_sha + "\n")
            if args[:4] == ["git", "remote", "get-url", "origin"]:
                return _completed_proc(stdout="https://gitlab.example.com/org/repo.git\n")
            if args[:2] == ["git", "push"]:
                return _completed_proc()
            return _completed_proc()

        with patch("subprocess.run", side_effect=dispatch):
            sha = engine._git_commit_and_push("summary")

        assert sha == rev_parse_sha, "pre-staged changes must produce a commit SHA"
        diff_calls = [c for c in calls if c[:3] == ["git", "diff", "--cached"]]
        assert diff_calls, "git diff --cached was not called to detect pre-staged changes"
        commit_calls = [c for c in calls if c[:2] == ["git", "commit"]]
        assert commit_calls, "git commit was not called despite staged changes"

    def test_clean_index_with_empty_files_returns_empty(self, tmp_path):
        """When _files_changed is empty and the git index is clean, return empty string without committing."""
        engine = _make_engine(tmp_path)
        engine._files_changed = []

        with patch("subprocess.run", return_value=_completed_proc(returncode=0)) as mock:
            sha = engine._git_commit_and_push("summary")

        assert sha == ""
        diff_calls = [c for c in mock.call_args_list if c.args[0][:3] == ["git", "diff", "--cached"]]
        assert diff_calls, "git diff --cached must be called to verify a clean index"


class TestTaskComplete:
    """Regression tests for the _task_complete fix: BUG where agent writes files via shell
    tools (not write_file), calls task_complete, and nothing gets committed.

    The fix: when _files_changed is empty, run 'git status --short'. If the working tree
    has changes, return a prompt that tells the agent to stage them. If clean, complete normally.
    Sensitive-named files also trigger a warning so secrets are not accidentally staged.
    """

    def test_unstaged_changes_prompts_agent(self, tmp_path):
        """When _files_changed is empty and git status shows changes, return a staging prompt."""
        engine = _make_engine(tmp_path)
        engine._files_changed = []

        status_output = " M factory/core/monitor.py\n?? newfile.py\n"
        with patch("subprocess.run", return_value=_completed_proc(returncode=0, stdout=status_output)):
            result = engine._task_complete("Issue #1: FIXED")

        assert "No files were tracked via write_file" in result
        assert "factory/core/monitor.py" in result
        assert "newfile.py" in result
        assert "run_command('git add" in result

    def test_clean_status_completes_directly(self, tmp_path):
        """When _files_changed is empty and git status is clean, return 'Task completed:' immediately."""
        engine = _make_engine(tmp_path)
        engine._files_changed = []

        with patch("subprocess.run", return_value=_completed_proc(returncode=0, stdout="")):
            result = engine._task_complete("Issue #1: FIXED")

        assert result == "Task completed: Issue #1: FIXED"

    def test_sensitive_file_triggers_warning(self, tmp_path):
        """A .env file in git status must produce a ⚠️ WARNING in the prompt."""
        engine = _make_engine(tmp_path)
        engine._files_changed = []

        status_output = "?? .env\n M src/app.py\n"
        with patch("subprocess.run", return_value=_completed_proc(returncode=0, stdout=status_output)):
            result = engine._task_complete("done")

        assert "⚠️" in result
        assert "WARNING" in result
        assert ".env" in result
        assert "do NOT stage" in result

    def test_secret_patterns_all_warned(self, tmp_path):
        """Files matching each sensitive pattern must trigger the warning."""
        engine = _make_engine(tmp_path)
        engine._files_changed = []

        for filename in ("id_rsa", "credentials.json", "my.pem", "private_key.py"):
            status_output = f"?? {filename}\n"
            with patch("subprocess.run", return_value=_completed_proc(returncode=0, stdout=status_output)):
                result = engine._task_complete("done")
            assert "WARNING" in result, f"Expected warning for sensitive file: {filename!r}"

    def test_files_identical_to_head_prompts_agent(self, tmp_path):
        """When write_file was called but the files are identical to HEAD, return a prompt.

        This is the real failure mode from the logs: the agent wrote monitor.py via
        write_file, git add succeeded but staged nothing because the content matched HEAD.
        _task_complete must catch this before the loop exits so the agent can act.
        """
        engine = _make_engine(tmp_path)
        engine._files_changed = ["factory/core/monitor.py"]

        # git status --short -- factory/core/monitor.py returns empty → file is clean
        with patch("subprocess.run", return_value=_completed_proc(returncode=0, stdout="")):
            result = engine._task_complete("Issue #1: FIXED")

        assert "identical to the already-committed version" in result
        assert "factory/core/monitor.py" in result
        assert "Task completed" not in result

    def test_files_with_real_changes_complete_normally(self, tmp_path):
        """When write_file was called and the files differ from HEAD, complete normally."""
        engine = _make_engine(tmp_path)
        engine._files_changed = ["factory/core/monitor.py"]

        # git status --short -- factory/core/monitor.py returns " M ..." → real change
        with patch("subprocess.run", return_value=_completed_proc(returncode=0, stdout=" M factory/core/monitor.py\n")):
            result = engine._task_complete("Issue #1: FIXED")

        assert result == "Task completed: Issue #1: FIXED"

    def test_multiple_files_all_identical_prompts_agent(self, tmp_path):
        """All-clean case with multiple files still returns the prompt listing all of them."""
        engine = _make_engine(tmp_path)
        engine._files_changed = ["src/foo.py", "src/bar.py"]

        with patch("subprocess.run", return_value=_completed_proc(returncode=0, stdout="")):
            result = engine._task_complete("done")

        assert "src/foo.py" in result
        assert "src/bar.py" in result
        assert "Task completed" not in result

    def test_files_changed_git_status_error_falls_through(self, tmp_path):
        """When git status raises for non-empty _files_changed, fall through to completion."""
        engine = _make_engine(tmp_path)
        engine._files_changed = ["foo.py"]

        with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
            result = engine._task_complete("done")

        assert result == "Task completed: done"

    def test_git_status_exception_falls_through_to_completion(self, tmp_path):
        """When git is unavailable or raises, _task_complete must fall through and complete."""
        engine = _make_engine(tmp_path)
        engine._files_changed = []

        with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
            result = engine._task_complete("done")

        assert result == "Task completed: done"

    def test_git_status_nonzero_falls_through_to_completion(self, tmp_path):
        """When git status returns non-zero (e.g., not a git repo), fall through to completion."""
        engine = _make_engine(tmp_path)
        engine._files_changed = []

        with patch("subprocess.run", return_value=_completed_proc(returncode=128, stdout="", stderr="fatal: not a git repository")):
            result = engine._task_complete("done")

        assert result == "Task completed: done"

    def test_summary_stored_on_completion(self, tmp_path):
        """_task_complete must store the summary in _task_summary when completing normally."""
        engine = _make_engine(tmp_path)
        engine._files_changed = ["foo.py"]

        # Return a non-empty status line so the file appears changed and we complete normally.
        with patch("subprocess.run", return_value=_completed_proc(returncode=0, stdout=" M foo.py\n")):
            engine._task_complete("my summary")

        assert engine._task_summary == "my summary"