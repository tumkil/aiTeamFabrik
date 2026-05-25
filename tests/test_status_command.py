"""Test cases for the factory status command."""

import pytest
from typer.testing import CliRunner
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import yaml
from datetime import date

from factory.main import app


runner = CliRunner()


@pytest.fixture
def mock_config_file(tmp_path):
    """Create a mock factory.yml config file."""
    config = {
        "gitlab": {
            "host": "https://gitlab.example.com",
            "project": "test/project",
            "token": "fake-token"
        },
        "sprint": {
            "number": 1,
            "name": "Test Sprint",
            "start_date": "2023-01-01",
            "end_date": "2023-01-14"
        }
    }
    config_file = tmp_path / "factory.yml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return str(config_file)


@pytest.fixture
def mock_agents_dir(tmp_path):
    """Create a mock agents directory."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    return str(agents_dir)


@pytest.fixture
def mock_token_usage_file(tmp_path):
    """Create a mock token_usage.yml file."""
    usage = {
        "agents": {
            "test_agent": {
                "daily_used": 1000,
                "daily_limit": 10000,
                "sprint_used": 5000,
                "sprint_limit": 50000
            }
        }
    }
    usage_file = tmp_path / "token_usage.yml"
    with open(usage_file, "w") as f:
        yaml.dump(usage, f)
    return str(usage_file)


@pytest.fixture
def mock_sprint_info():
    """Create a mock sprint info object."""
    sprint = Mock()
    sprint.number = 1
    sprint.name = "Test Sprint"
    sprint.start_date = date(2023, 1, 1)
    sprint.end_date = date(2023, 1, 14)
    sprint.open_issues = 5
    sprint.in_progress_issues = 2
    sprint.closed_issues = 3
    sprint.completion_pct = 50.0
    return sprint


@pytest.fixture
def mock_agent():
    """Create a mock agent object."""
    agent = Mock()
    agent.display_name = "TestAgent"
    agent.model = "gpt-4"
    agent.status = "idle"
    agent.current_task = None
    return agent


@pytest.mark.integration
def test_status_command_success(
    mock_config_file, 
    mock_agents_dir, 
    mock_token_usage_file, 
    mock_sprint_info, 
    mock_agent
):
    """Test the status command with successful GitLab connection and valid data."""
    
    with patch("factory.commands.status.GitLabClient") as mock_gl_client_class:
        mock_gl = MagicMock()
        mock_gl.hostname = "gitlab.example.com"
        mock_gl.project_path = "test/project"
        mock_gl.connect.return_value = (True, "Connected")
        mock_gl.sprint_info.return_value = mock_sprint_info
        mock_gl.days_remaining.return_value = 7
        mock_gl_client_class.return_value = mock_gl
        
        with patch("factory.commands.status.CompetenceManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm.agents = [mock_agent]
            mock_cm_class.return_value = mock_cm
            
            with patch("factory.commands.status.TokenBudgetManager") as mock_budget_class:
                mock_budget = MagicMock()
                mock_budget.usage_report.return_value = {
                    "agents": {
                        "TestAgent": {
                            "daily_used": 1000,
                            "daily_limit": 10000,
                            "daily_pct": 10.0,
                            "sprint_used": 5000,
                            "sprint_limit": 50000,
                            "sprint_pct": 10.0
                        }
                    }
                }
                mock_budget.is_over_budget.return_value = False
                mock_budget_class.return_value = mock_budget
                
                result = runner.invoke(app, [
                    "status",
                    "--config", mock_config_file,
                    "--agents", mock_agents_dir,
                    "--usage", mock_token_usage_file
                ])
                
                assert result.exit_code == 0
                assert "SoftwareTeamFabrik — Status Report" in result.output
                assert "GitLab" in result.output
                assert "gitlab.example.com" in result.output
                assert "✓ connected" in result.output
                assert "Project" in result.output
                assert "test/project" in result.output
                assert "Sprint" in result.output
                assert "Test Sprint" in result.output
                assert "Token Budget Usage" in result.output
                assert "TestAgent" in result.output
                assert "Active Agents" in result.output
                assert "Open Issues" in result.output
                assert "Milestone" in result.output


@pytest.mark.integration
def test_status_command_gitlab_connection_failure(mock_config_file, mock_agents_dir):
    """Test the status command when GitLab connection fails."""
    
    with patch("factory.commands.status.GitLabClient") as mock_gl_client_class:
        mock_gl = MagicMock()
        mock_gl.hostname = "gitlab.example.com"
        mock_gl.connect.return_value = (False, "Connection failed")
        mock_gl_client_class.return_value = mock_gl
        
        result = runner.invoke(app, [
            "status",
            "--config", mock_config_file,
            "--agents", mock_agents_dir
        ])
        
        assert result.exit_code == 1
        assert "✗ Connection failed" in result.output


@pytest.mark.integration
def test_status_command_sprint_info_failure(
    mock_config_file, 
    mock_agents_dir, 
    mock_token_usage_file
):
    """Test the status command when sprint info cannot be loaded."""
    
    with patch("factory.commands.status.GitLabClient") as mock_gl_client_class:
        mock_gl = MagicMock()
        mock_gl.hostname = "gitlab.example.com"
        mock_gl.project_path = "test/project"
        mock_gl.connect.return_value = (True, "Connected")
        mock_gl.sprint_info.side_effect = Exception("Sprint info error")
        mock_gl_client_class.return_value = mock_gl
        
        with patch("factory.commands.status.CompetenceManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm.agents = []
            mock_cm_class.return_value = mock_cm
            
            with patch("factory.commands.status.TokenBudgetManager") as mock_budget_class:
                mock_budget = MagicMock()
                mock_budget.usage_report.return_value = {"agents": {}}
                mock_budget_class.return_value = mock_budget
                
                result = runner.invoke(app, [
                    "status",
                    "--config", mock_config_file,
                    "--agents", mock_agents_dir,
                    "--usage", mock_token_usage_file
                ])
                
                assert result.exit_code == 0
                assert "Could not load sprint info" in result.output


@pytest.mark.integration
def test_status_command_budget_tracking_unavailable(
    mock_config_file, 
    mock_agents_dir,
    mock_sprint_info
):
    """Test the status command when budget tracking is unavailable."""
    
    with patch("factory.commands.status.GitLabClient") as mock_gl_client_class:
        mock_gl = MagicMock()
        mock_gl.hostname = "gitlab.example.com"
        mock_gl.project_path = "test/project"
        mock_gl.connect.return_value = (True, "Connected")
        mock_gl.sprint_info.return_value = mock_sprint_info
        mock_gl.days_remaining.return_value = 7
        mock_gl_client_class.return_value = mock_gl
        
        with patch("factory.commands.status.CompetenceManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm.agents = []
            mock_cm_class.return_value = mock_cm
            
            with patch("factory.commands.status.TokenBudgetManager") as mock_budget_class:
                mock_budget_class.side_effect = Exception("Budget error")
                
                result = runner.invoke(app, [
                    "status",
                    "--config", mock_config_file,
                    "--agents", mock_agents_dir
                ])
                
                assert result.exit_code == 0
                assert "Budget tracking unavailable" in result.output


@pytest.mark.integration
def test_status_command_working_agent(
    mock_config_file, 
    mock_agents_dir, 
    mock_token_usage_file, 
    mock_sprint_info
):
    """Test the status command with a working agent."""
    
    with patch("factory.commands.status.GitLabClient") as mock_gl_client_class:
        mock_gl = MagicMock()
        mock_gl.hostname = "gitlab.example.com"
        mock_gl.project_path = "test/project"
        mock_gl.connect.return_value = (True, "Connected")
        mock_gl.sprint_info.return_value = mock_sprint_info
        mock_gl.days_remaining.return_value = 7
        mock_gl_client_class.return_value = mock_gl
        
        with patch("factory.commands.status.CompetenceManager") as mock_cm_class:
            mock_cm = MagicMock()
            
            working_agent = Mock()
            working_agent.display_name = "WorkingAgent"
            working_agent.model = "gpt-4"
            working_agent.status = "working"
            working_agent.current_task = "Implementing feature X"
            
            mock_cm.agents = [working_agent]
            mock_cm_class.return_value = mock_cm
            
            with patch("factory.commands.status.TokenBudgetManager") as mock_budget_class:
                mock_budget = MagicMock()
                mock_budget.usage_report.return_value = {"agents": {}}
                mock_budget_class.return_value = mock_budget
                
                result = runner.invoke(app, [
                    "status",
                    "--config", mock_config_file,
                    "--agents", mock_agents_dir,
                    "--usage", mock_token_usage_file
                ])
                
                assert result.exit_code == 0
                assert "working" in result.output
                assert "→" in result.output
                assert "Implementing feature X" in result.output


@pytest.mark.integration
def test_status_command_over_budget(
    mock_config_file, 
    mock_agents_dir, 
    mock_token_usage_file, 
    mock_sprint_info
):
    """Test the status command when an agent is over budget."""
    
    with patch("factory.commands.status.GitLabClient") as mock_gl_client_class:
        mock_gl = MagicMock()
        mock_gl.hostname = "gitlab.example.com"
        mock_gl.project_path = "test/project"
        mock_gl.connect.return_value = (True, "Connected")
        mock_gl.sprint_info.return_value = mock_sprint_info
        mock_gl.days_remaining.return_value = 7
        mock_gl_client_class.return_value = mock_gl
        
        with patch("factory.commands.status.CompetenceManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm.agents = []
            mock_cm_class.return_value = mock_cm
            
            with patch("factory.commands.status.TokenBudgetManager") as mock_budget_class:
                mock_budget = MagicMock()
                mock_budget.usage_report.return_value = {
                    "agents": {
                        "OverBudgetAgent": {
                            "daily_used": 15000,
                            "daily_limit": 10000,
                            "daily_pct": 150.0,
                            "sprint_used": 60000,
                            "sprint_limit": 50000,
                            "sprint_pct": 120.0
                        }
                    }
                }
                mock_budget.is_over_budget.return_value = True
                mock_budget_class.return_value = mock_budget
                
                result = runner.invoke(app, [
                    "status",
                    "--config", mock_config_file,
                    "--agents", mock_agents_dir,
                    "--usage", mock_token_usage_file
                ])
                
                assert result.exit_code == 0
                assert "❌ OVER BUDGET" in result.output
