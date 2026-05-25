"""Unit tests for CodeExecutionEngine._ensure_git_repo."""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from factory.core.competence import AgentProfile
from factory.core.execution_engine import CodeExecutionEngine


def _make_engine(tmp_path: Path, token: str = "tok123", clone_url: str = "https://gitlab.example.com/org/repo.git") -> CodeExecutionEngine:
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
        clone_url=clone_url,
    )
    return engine


def _completed_proc(returncode=0, stdout="", stderr="") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class TestEnsureGitRepo:
    @pytest.fixture(autouse=True)
    def cleanup_clone_dir(self):
        """Remove the deterministic clone dir before and after every test so tests don't leak state."""
        clone_dir = Path(tempfile.gettempdir()) / "factory_42_issue-1"
        shutil.rmtree(clone_dir, ignore_errors=True)
        yield
        shutil.rmtree(clone_dir, ignore_errors=True)

    def test_existing_git_repo_no_op(self, tmp_path):
        """When .git exists, _ensure_git_repo should do nothing."""
        (tmp_path / ".git").mkdir()
        engine = _make_engine(tmp_path)
        
        with patch("subprocess.run") as mock_run:
            engine._ensure_git_repo()
        
        # subprocess.run should not be called
        mock_run.assert_not_called()

    def test_no_clone_url_without_git_repo_logs_warning(self, tmp_path, caplog):
        """When no .git and no clone_url, log a warning and do nothing."""
        # Ensure no .git directory exists
        assert not (tmp_path / ".git").exists()
        engine = _make_engine(tmp_path, clone_url="")
        
        with patch("subprocess.run") as mock_run:
            engine._ensure_git_repo()
        
        mock_run.assert_not_called()
        assert "no .git directory and no clone_url was provided" in caplog.text

    def test_clone_with_token_embeds_credentials(self, tmp_path):
        """When a token is provided, it should be embedded in the clone URL."""
        # Ensure no .git directory exists so clone actually happens
        assert not (tmp_path / ".git").exists()
        engine = _make_engine(tmp_path, token="glpat-abc123")
        
        captured_clone_urls = []
        
        def capture_clone(args, **kwargs):
            if args[:2] == ["git", "clone"]:
                # The URL is the argument after "--"
                url_idx = args.index("--") + 1 if "--" in args else 2
                captured_clone_urls.append(args[url_idx])
            return _completed_proc()
        
        with patch("subprocess.run", side_effect=capture_clone):
            engine._ensure_git_repo()
        
        assert captured_clone_urls
        clone_url = captured_clone_urls[0]
        assert "oauth2:glpat-abc123@" in clone_url

    def test_clone_without_token_uses_plain_url(self, tmp_path):
        """When no token is provided, the clone URL should not have embedded credentials."""
        # Ensure no .git directory exists
        assert not (tmp_path / ".git").exists()
        engine = _make_engine(tmp_path, token="")
        
        captured_clone_urls = []
        
        def capture_clone(args, **kwargs):
            if args[:2] == ["git", "clone"]:
                url_idx = args.index("--") + 1 if "--" in args else 2
                captured_clone_urls.append(args[url_idx])
            return _completed_proc()
        
        with patch("subprocess.run", side_effect=capture_clone):
            engine._ensure_git_repo()
        
        assert captured_clone_urls
        clone_url = captured_clone_urls[0]
        assert "oauth2:" not in clone_url
        assert "@" not in clone_url or "gitlab.example.com" in clone_url

    def test_clone_uses_deterministic_temp_dir(self, tmp_path):
        """Clone should use a deterministic temp dir based on project ID and branch."""
        # Ensure no .git directory exists
        assert not (tmp_path / ".git").exists()
        engine = _make_engine(tmp_path)
        
        captured_clone_dirs = []
        
        def capture_clone(args, **kwargs):
            if args[:2] == ["git", "clone"]:
                captured_clone_dirs.append(args[-1])
            return _completed_proc()
        
        with patch("subprocess.run", side_effect=capture_clone):
            engine._ensure_git_repo()
        
        assert captured_clone_dirs
        clone_dir = Path(captured_clone_dirs[0])
        assert "factory_42_issue-1" in str(clone_dir)

    def test_clone_failure_logs_error(self, tmp_path, caplog):
        """When git clone fails, log an error."""
        # Ensure no .git directory exists
        assert not (tmp_path / ".git").exists()
        engine = _make_engine(tmp_path)
        
        with patch("subprocess.run", return_value=_completed_proc(returncode=1, stderr="clone failed")):
            engine._ensure_git_repo()
        
        assert "git clone failed" in caplog.text

    def test_fetch_and_checkout_after_clone(self, tmp_path):
        """After successful clone, fetch and checkout the target branch."""
        # Ensure no .git directory exists
        assert not (tmp_path / ".git").exists()
        engine = _make_engine(tmp_path)
        
        calls = []
        
        def capture_calls(args, **kwargs):
            calls.append((args, kwargs.get("cwd")))
            if args[:2] == ["git", "clone"]:
                # Simulate successful clone by creating the directory
                clone_dir = Path(args[-1])
                clone_dir.mkdir(parents=True, exist_ok=True)
                (clone_dir / ".git").mkdir()
            return _completed_proc()
        
        with patch("subprocess.run", side_effect=capture_calls):
            engine._ensure_git_repo()
        
        # Verify fetch and checkout were called
        fetch_calls = [c for c in calls if c[0][:3] == ["git", "fetch", "origin"]]
        checkout_calls = [c for c in calls if c[0][:2] == ["git", "checkout"]]
        
        assert fetch_calls, "git fetch was not called after clone"
        assert checkout_calls, "git checkout was not called after clone"

    def test_checkout_fallback_when_branch_not_local(self, tmp_path):
        """When checkout fails because branch doesn't exist locally, create it tracking origin."""
        # Ensure no .git directory exists
        assert not (tmp_path / ".git").exists()
        engine = _make_engine(tmp_path)
        
        calls = []
        
        def capture_calls(args, **kwargs):
            calls.append((args, kwargs.get("cwd")))
            if args[:2] == ["git", "clone"]:
                clone_dir = Path(args[-1])
                clone_dir.mkdir(parents=True, exist_ok=True)
                (clone_dir / ".git").mkdir()
            elif args[:3] == ["git", "checkout", "issue-1"]:
                # Simulate branch not existing locally
                return _completed_proc(returncode=1, stderr="branch not found")
            return _completed_proc()
        
        with patch("subprocess.run", side_effect=capture_calls):
            engine._ensure_git_repo()
        
        # Verify fallback checkout -b was called
        fallback_calls = [c for c in calls if c[0][:4] == ["git", "checkout", "-b", "issue-1"]]
        assert fallback_calls, "fallback git checkout -b was not called"

    def test_repo_path_updated_after_clone(self, tmp_path):
        """After successful clone, repo_path should be updated to the clone directory."""
        # Ensure no .git directory exists
        assert not (tmp_path / ".git").exists()
        engine = _make_engine(tmp_path)
        
        original_repo_path = engine.repo_path
        
        def capture_clone(args, **kwargs):
            if args[:2] == ["git", "clone"]:
                clone_dir = Path(args[-1])
                clone_dir.mkdir(parents=True, exist_ok=True)
                (clone_dir / ".git").mkdir()
            return _completed_proc()
        
        with patch("subprocess.run", side_effect=capture_clone):
            engine._ensure_git_repo()
        
        assert engine.repo_path != original_repo_path
        assert (engine.repo_path / ".git").exists()

    def test_existing_clone_dir_reused(self, tmp_path):
        """When clone dir already exists with .git, reuse it instead of cloning again."""
        # Create a fake clone directory in temp dir (not in tmp_path)
        import tempfile
        clone_dir = Path(tempfile.gettempdir()) / "factory_42_issue-1"
        clone_dir.mkdir()
        (clone_dir / ".git").mkdir()
        
        # Ensure tmp_path has no .git
        assert not (tmp_path / ".git").exists()
        engine = _make_engine(tmp_path)
        
        clone_calls = []
        
        def capture_clone(args, **kwargs):
            if args[:2] == ["git", "clone"]:
                clone_calls.append(args)
            return _completed_proc()
        
        with patch("subprocess.run", side_effect=capture_clone):
            engine._ensure_git_repo()
        
        # git clone should not be called when directory already exists
        assert not clone_calls

    def test_git_terminal_prompt_set_in_env(self, tmp_path):
        """GIT_TERMINAL_PROMPT should be set to 0 in the environment."""
        # Ensure no .git directory exists
        assert not (tmp_path / ".git").exists()
        engine = _make_engine(tmp_path)
        
        captured_envs = []
        
        def capture_env(args, **kwargs):
            captured_envs.append(kwargs.get("env") or {})
            return _completed_proc()
        
        with patch("subprocess.run", side_effect=capture_env):
            engine._ensure_git_repo()
        
        # Check that GIT_TERMINAL_PROMPT=0 was in the environment for git operations
        git_envs = [env for env in captured_envs if "GIT_TERMINAL_PROMPT" in env]
        assert git_envs, "GIT_TERMINAL_PROMPT was not set in environment"
        for env in git_envs:
            assert env["GIT_TERMINAL_PROMPT"] == "0"

    def test_clone_uses_shallow_clone(self, tmp_path):
        """Clone should use --depth 1 and --no-single-branch for efficiency."""
        # Ensure no .git directory exists
        assert not (tmp_path / ".git").exists()
        engine = _make_engine(tmp_path)
        
        captured_clone_args = []
        
        def capture_clone(args, **kwargs):
            if args[:2] == ["git", "clone"]:
                captured_clone_args.append(args)
            return _completed_proc()
        
        with patch("subprocess.run", side_effect=capture_clone):
            engine._ensure_git_repo()
        
        assert captured_clone_args
        clone_args = captured_clone_args[0]
        assert "--depth" in clone_args
        assert "1" in clone_args[clone_args.index("--depth") + 1]
        assert "--no-single-branch" in clone_args

    def test_http_url_converted_to_https_with_token(self, tmp_path):
        """When token is provided and URL is http, it should be converted to https."""
        # Ensure no .git directory exists
        assert not (tmp_path / ".git").exists()
        engine = _make_engine(tmp_path, token="glpat-abc123", clone_url="http://gitlab.example.com/org/repo.git")
        
        captured_clone_urls = []
        
        def capture_clone(args, **kwargs):
            if args[:2] == ["git", "clone"]:
                url_idx = args.index("--") + 1 if "--" in args else 2
                captured_clone_urls.append(args[url_idx])
            return _completed_proc()
        
        with patch("subprocess.run", side_effect=capture_clone):
            engine._ensure_git_repo()
        
        assert captured_clone_urls
        clone_url = captured_clone_urls[0]
        assert clone_url.startswith("https://")
        assert "oauth2:glpat-abc123@" in clone_url

    def test_non_http_url_not_modified(self, tmp_path):
        """When URL is not http/https (e.g., ssh), it should not be modified."""
        # Ensure no .git directory exists
        assert not (tmp_path / ".git").exists()
        engine = _make_engine(tmp_path, token="glpat-abc123", clone_url="git@gitlab.example.com:org/repo.git")
        
        captured_clone_urls = []
        
        def capture_clone(args, **kwargs):
            if args[:2] == ["git", "clone"]:
                url_idx = args.index("--") + 1 if "--" in args else 2
                captured_clone_urls.append(args[url_idx])
            return _completed_proc()
        
        with patch("subprocess.run", side_effect=capture_clone):
            engine._ensure_git_repo()
        
        assert captured_clone_urls
        clone_url = captured_clone_urls[0]
        # SSH URL should not be modified
        assert clone_url == "git@gitlab.example.com:org/repo.git"
        assert "oauth2:" not in clone_url

    def test_branch_name_sanitized_in_clone_dir(self, tmp_path):
        """Branch name should be sanitized for filesystem safety in clone dir name."""
        # Ensure no .git directory exists
        assert not (tmp_path / ".git").exists()
        engine = _make_engine(tmp_path)
        engine.branch_name = "issue/1: special <chars>"
        
        captured_clone_dirs = []
        
        def capture_clone(args, **kwargs):
            if args[:2] == ["git", "clone"]:
                captured_clone_dirs.append(args[-1])
            return _completed_proc()
        
        with patch("subprocess.run", side_effect=capture_clone):
            engine._ensure_git_repo()
        
        assert captured_clone_dirs
        clone_dir = Path(captured_clone_dirs[0])
        # Branch name should be sanitized (special chars replaced with _)
        assert "issue_1__special__chars_" in str(clone_dir)
