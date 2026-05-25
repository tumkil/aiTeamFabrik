"""
Test suite for the factory run command.

This module tests the CLI command `factory run` which spawns an AI agent to analyse
an issue and open a draft MR, as well as the ``_git_config_is_set`` and
``_configure_git`` helpers.
"""

import subprocess
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from typer.testing import CliRunner

from factory.main import app
from factory.commands.run import cmd_run, _git_config_is_set, _configure_git


@pytest.fixture
def mock_gl_client():
    """Fixture providing a mocked GitLab client."""
    with patch("factory.commands.run.GitLabClient") as mock_cls:
        mock_instance = Mock()
        mock_instance.connect.return_value = (True, "Connected")
        mock_instance.project.issues.get.return_value = Mock(
            title="Test Issue",
            description="Test description",
            iid=1,
            labels=[],
            web_url="http://example.com/issue/1"
        )
        mock_cls.return_value = mock_instance
        yield mock_cls


@pytest.fixture
def mock_competence_manager():
    """Fixture providing a mocked CompetenceManager."""
    with patch("factory.commands.run.CompetenceManager") as mock_cls:
        mock_instance = Mock()
        mock_agent = Mock()
        mock_agent.display_name = "TestAgent"
        mock_agent.model = "test-model"
        mock_agent.execution_mode = "execute"
        mock_agent.provider = "anthropic"
        mock_instance.agents = [mock_agent]
        mock_cls.return_value = mock_instance
        yield mock_cls


@pytest.fixture
def mock_scrum_engine():
    """Fixture providing a mocked ScrumEngine."""
    with patch("factory.commands.run.ScrumEngine") as mock_cls:
        mock_instance = Mock()
        mock_cls.return_value = mock_instance
        yield mock_cls


@pytest.fixture
def mock_llm_router():
    """Fixture providing a mocked LlmRouter."""
    with patch("factory.commands.run.LlmRouter") as mock_cls:
        mock_instance = Mock()
        mock_cls.return_value = mock_instance
        yield mock_cls


@pytest.fixture
def mock_orchestrator():
    """Fixture providing a mocked Orchestrator."""
    with patch("factory.commands.run.Orchestrator") as mock_cls:
        mock_instance = Mock()
        mock_result = Mock()
        mock_result.mode = "execute"
        mock_result.iterations = 2
        mock_result.files_changed = ["file1.py", "file2.py"]
        mock_result.commit_sha = "abc123def456"
        mock_result.mr_url = "http://example.com/mr/1"
        mock_result.response = "Test response"
        mock_result.comment_url = None
        mock_result.needs_continuation = False
        mock_instance.run.return_value = mock_result
        
        # Mock the _resolve_agent method to return the mock_agent
        mock_agent = Mock()
        mock_agent.display_name = "TestAgent"
        mock_agent.model = "test-model"
        mock_agent.execution_mode = "execute"
        mock_agent.provider = "anthropic"
        mock_instance._resolve_agent.return_value = mock_agent
        
        mock_cls.return_value = mock_instance
        yield mock_cls


@pytest.fixture
def mock_default_models():
    """Fixture providing mocked DEFAULT_MODELS."""
    with patch("factory.commands.run.DEFAULT_MODELS", {
        "anthropic": "claude-3-opus-20240229",
        "mistral": "mistral-large-latest",
        "ollama": "llama3",
        "stub": "stub"
    }):
        yield


@pytest.fixture
def mock_configure_git():
    """Fixture that patches _configure_git so existing integration tests
    don't invoke real git subprocess calls."""
    with patch("factory.commands.run._configure_git"):
        yield


def test_run_command_success(
    mock_gl_client,
    mock_competence_manager,
    mock_scrum_engine,
    mock_llm_router,
    mock_orchestrator,
    mock_default_models,
    mock_configure_git,
):
    """Test successful execution of factory run command."""
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--issue", "1"])
    
    assert result.exit_code == 0
    assert "SoftwareTeamFabrik — Running Issue #1" in result.output
    assert "TestAgent" in result.output
    assert "execute — will write & commit code" in result.output
    assert "Test Issue" in result.output
    assert "Iterations" in result.output
    assert "2" in result.output
    assert "file1.py" in result.output
    assert "file2.py" in result.output
    assert "abc123de" in result.output
    assert "http://example.com/mr/1" in result.output
    assert "Implementation completed" in result.output


def test_run_command_with_stub_provider(
    mock_gl_client,
    mock_competence_manager,
    mock_scrum_engine,
    mock_llm_router,
    mock_orchestrator,
    mock_default_models,
    mock_configure_git,
):
    """Test factory run command with stub provider."""
    # Override the agent to use stub provider
    mock_agent = Mock()
    mock_agent.display_name = "TestAgent"
    mock_agent.model = "test-model"
    mock_agent.execution_mode = "plan"
    mock_agent.provider = "stub"
    mock_competence_manager.return_value.agents = [mock_agent]
    
    mock_result = Mock()
    mock_result.mode = "plan"
    mock_result.response = "Test analysis"
    mock_orchestrator.return_value.run.return_value = mock_result
    
    # Mock _resolve_agent to return the stub agent
    mock_orchestrator.return_value._resolve_agent.return_value = mock_agent
    
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--issue", "1", "--provider", "stub"])
    
    assert result.exit_code == 0
    assert "plan — will post analysis" in result.output


def test_run_command_with_unknown_provider(
    mock_gl_client,
    mock_competence_manager,
    mock_scrum_engine,
    mock_llm_router,
    mock_orchestrator,
    mock_default_models,
    mock_configure_git,
):
    """Test factory run command with unknown provider."""
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--issue", "1", "--provider", "unknown"])
    
    assert result.exit_code == 1
    assert "Unknown provider: unknown" in result.output


def test_run_command_gitlab_connection_failure(
    mock_gl_client,
    mock_competence_manager,
    mock_scrum_engine,
    mock_llm_router,
    mock_orchestrator,
    mock_default_models,
    mock_configure_git,
):
    """Test factory run command when GitLab connection fails."""
    mock_gl_client.return_value.connect.return_value = (False, "Connection failed")
    
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--issue", "1"])
    
    assert result.exit_code == 1
    assert "GitLab connection failed: Connection failed" in result.output


def test_run_command_agent_failure(
    mock_gl_client,
    mock_competence_manager,
    mock_scrum_engine,
    mock_llm_router,
    mock_orchestrator,
    mock_default_models,
    mock_configure_git,
):
    """Test factory run command when agent execution fails."""
    mock_orchestrator.return_value.run.side_effect = Exception("Agent failed")
    
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--issue", "1"])
    
    assert result.exit_code == 1
    assert "Agent failed: Agent failed" in result.output


def test_run_command_execution_engine_failure(
    mock_gl_client,
    mock_competence_manager,
    mock_scrum_engine,
    mock_llm_router,
    mock_orchestrator,
    mock_default_models,
    mock_configure_git,
):
    """Test factory run command when execution engine fails to start."""
    mock_result = Mock()
    mock_result.mode = "execute"
    mock_result.iterations = 0
    mock_result.files_changed = []
    mock_result.commit_sha = None
    mock_result.mr_url = None
    mock_result.response = ""
    mock_result.comment_url = "http://example.com/comment/1"
    mock_result.needs_continuation = False
    mock_orchestrator.return_value.run.return_value = mock_result
    
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--issue", "1"])
    
    assert result.exit_code == 1
    assert "Execution engine failed to start" in result.output
    assert "ANTHROPIC_API_KEY" in result.output
    assert "http://example.com/comment/1" in result.output


def test_run_command_continuation_needed(
    mock_gl_client,
    mock_competence_manager,
    mock_scrum_engine,
    mock_llm_router,
    mock_orchestrator,
    mock_default_models,
    mock_configure_git,
):
    """Test factory run command when implementation needs continuation."""
    mock_result = Mock()
    mock_result.mode = "execute"
    mock_result.iterations = 5
    mock_result.files_changed = ["file1.py"]
    mock_result.commit_sha = "abc123def456"
    mock_result.mr_url = "http://example.com/mr/1"
    mock_result.response = "Test response"
    mock_result.comment_url = None
    mock_result.needs_continuation = True
    mock_orchestrator.return_value.run.return_value = mock_result
    
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--issue", "1"])
    
    assert result.exit_code == 0
    assert "Implementation incomplete" in result.output
    assert "factory continue-mr --mr 1" in result.output


def test_run_command_plan_mode(
    mock_gl_client,
    mock_competence_manager,
    mock_scrum_engine,
    mock_llm_router,
    mock_orchestrator,
    mock_default_models,
    mock_configure_git,
):
    """Test factory run command in plan mode."""
    mock_agent = Mock()
    mock_agent.display_name = "TestAgent"
    mock_agent.model = "test-model"
    mock_agent.execution_mode = "plan"
    mock_agent.provider = "anthropic"
    mock_competence_manager.return_value.agents = [mock_agent]
    
    mock_result = Mock()
    mock_result.mode = "plan"
    mock_result.response = "This is a test analysis"
    mock_orchestrator.return_value.run.return_value = mock_result
    
    # Mock _resolve_agent to return the plan mode agent
    mock_orchestrator.return_value._resolve_agent.return_value = mock_agent
    
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--issue", "1"])
    
    assert result.exit_code == 0
    assert "plan — will post analysis" in result.output
    assert "This is a test analysis" in result.output


def test_run_command_custom_config_and_agents_dir(
    mock_gl_client,
    mock_competence_manager,
    mock_scrum_engine,
    mock_llm_router,
    mock_orchestrator,
    mock_default_models,
    mock_configure_git,
):
    """Test factory run command with custom config and agents directory."""
    runner = CliRunner()
    result = runner.invoke(
        app, 
        ["run", "--issue", "1", "--config", "custom/config.yml", "--agents", "custom/agents"]
    )
    
    assert result.exit_code == 0
    # Verify that the custom paths were used
    mock_gl_client.assert_called_once_with(config_path="custom/config.yml")
    mock_competence_manager.assert_called_once_with(agents_dir="custom/agents")


# ---------------------------------------------------------------------------
# Tests for _git_config_is_set
# ---------------------------------------------------------------------------


class TestGitConfigIsSet:
    """Tests for the ``_git_config_is_set`` helper."""

    @patch("factory.commands.run.subprocess.run")
    def test_returns_true_when_key_is_set(self, mock_run):
        """If ``git config --get`` exits with 0, the key is considered set."""
        mock_run.return_value = Mock(returncode=0)
        assert _git_config_is_set("user.name") is True
        mock_run.assert_called_once_with(
            ["git", "config", "--get", "user.name"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @patch("factory.commands.run.subprocess.run")
    def test_returns_false_when_key_is_unset(self, mock_run):
        """If ``git config --get`` exits with 1, the key is not set."""
        mock_run.return_value = Mock(returncode=1)
        assert _git_config_is_set("user.name") is False

    @patch("factory.commands.run.subprocess.run")
    def test_returns_false_on_other_nonzero_exit(self, mock_run):
        """Any non-zero exit code means the key is not set."""
        mock_run.return_value = Mock(returncode=128)
        assert _git_config_is_set("user.email") is False


# ---------------------------------------------------------------------------
# Tests for _configure_git
# ---------------------------------------------------------------------------


class TestConfigureGit:
    """Tests for the ``_configure_git`` helper.

    The function must **only** set ``user.name`` / ``user.email`` when they
    are not already configured at any scope, so it never silently overwrites
    a developer's existing git identity.  All writes use ``--local`` to
    scope changes to the current repo only.
    """

    @patch("factory.commands.run._git_config_is_set", return_value=False)
    @patch("factory.commands.run.subprocess.run")
    def test_sets_both_when_neither_configured(self, mock_run, mock_is_set):
        """When neither user.name nor user.email is set, both are configured."""
        _configure_git()
        # _git_config_is_set is called for both keys
        assert mock_is_set.call_count == 2
        mock_is_set.assert_any_call("user.name")
        mock_is_set.assert_any_call("user.email")
        # subprocess.run is called twice — once for name, once for email
        assert mock_run.call_count == 2
        mock_run.assert_any_call(
            ["git", "config", "--local", "user.name", "SoftwareTeamFabrik"],
            check=True,
        )
        mock_run.assert_any_call(
            ["git", "config", "--local", "user.email", "factory@example.com"],
            check=True,
        )

    @patch("factory.commands.run._git_config_is_set", side_effect=lambda k: k == "user.name")
    @patch("factory.commands.run.subprocess.run")
    def test_only_sets_email_when_name_already_set(self, mock_run, mock_is_set):
        """When user.name is already set, only user.email is configured."""
        _configure_git()
        mock_is_set.assert_any_call("user.name")
        mock_is_set.assert_any_call("user.email")
        # Only the email subprocess call should have been made
        assert mock_run.call_count == 1
        mock_run.assert_called_once_with(
            ["git", "config", "--local", "user.email", "factory@example.com"],
            check=True,
        )

    @patch("factory.commands.run._git_config_is_set", side_effect=lambda k: k == "user.email")
    @patch("factory.commands.run.subprocess.run")
    def test_only_sets_name_when_email_already_set(self, mock_run, mock_is_set):
        """When user.email is already set, only user.name is configured."""
        _configure_git()
        mock_is_set.assert_any_call("user.name")
        mock_is_set.assert_any_call("user.email")
        # Only the name subprocess call should have been made
        assert mock_run.call_count == 1
        mock_run.assert_called_once_with(
            ["git", "config", "--local", "user.name", "SoftwareTeamFabrik"],
            check=True,
        )

    @patch("factory.commands.run._git_config_is_set", return_value=True)
    @patch("factory.commands.run.subprocess.run")
    def test_does_not_overwrite_existing_config(self, mock_run, mock_is_set):
        """When both values are already set, nothing is written."""
        _configure_git()
        # subprocess.run should NOT be called at all
        mock_run.assert_not_called()

    @patch("factory.commands.run._git_config_is_set", return_value=False)
    @patch("factory.commands.run.subprocess.run")
    def test_sets_local_scope(self, mock_run, mock_is_set):
        """Values are set via ``git config --local`` so they never leak to
        global config, even when a global identity exists."""
        _configure_git()
        # Verify the exact commands that were run
        calls = mock_run.call_args_list
        assert calls[0] == call(
            ["git", "config", "--local", "user.name", "SoftwareTeamFabrik"],
            check=True,
        )
        assert calls[1] == call(
            ["git", "config", "--local", "user.email", "factory@example.com"],
            check=True,
        )