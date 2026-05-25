"""Test cases for the factory monitor command."""

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
def test_monitor_command_help():
    """Test the monitor command help message."""
    result = runner.invoke(app, ["monitor", "--help"])
    assert result.exit_code == 0
    assert "Start monitoring service" in result.output
    assert "--config" in result.output
    assert "--agents-dir" in result.output
    assert "--interval" in result.output


@pytest.mark.integration
def test_monitor_command_success(
    mock_config_file, 
    mock_agents_dir, 
    mock_token_usage_file, 
    mock_sprint_info, 
    mock_agent
):
    """Test the monitor command with successful GitLab connection and valid data."""
    
    with patch("factory.core.monitor.GitLabClient") as mock_gl_client_class:
        mock_gl = MagicMock()
        mock_gl.hostname = "gitlab.example.com"
        mock_gl.project_path = "test/project"
        mock_gl.connect.return_value = (True, "Connected")
        mock_gl.sprint_info.return_value = mock_sprint_info
        mock_gl.days_remaining.return_value = 7
        mock_gl_client_class.return_value = mock_gl
        
        with patch("factory.core.monitor.CompetenceManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm.agents = [mock_agent]
            mock_cm_class.return_value = mock_cm
            
            with patch("factory.core.monitor.TokenBudgetManager") as mock_budget_class:
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
                
                # Mock the monitor's run method to avoid infinite loop
                with patch("factory.core.monitor.FactoryMonitor.run") as mock_run:
                    result = runner.invoke(app, [
                        "monitor",
                        "--config", mock_config_file,
                        "--agents-dir", mock_agents_dir
                    ])
                    
                    # The command should exit with code 0 on success
                    assert result.exit_code == 0
                    assert "✓ Monitor connected to gitlab.example.com" in result.output
                    mock_run.assert_called_once()


@pytest.mark.integration
def test_monitor_command_gitlab_connection_failure(mock_config_file, mock_agents_dir):
    """Test the monitor command when GitLab connection fails."""
    
    with patch("factory.core.monitor.GitLabClient") as mock_gl_client_class:
        mock_gl = MagicMock()
        mock_gl.hostname = "gitlab.example.com"
        mock_gl.connect.return_value = (False, "Connection failed")
        mock_gl_client_class.return_value = mock_gl
        
        result = runner.invoke(app, [
            "monitor",
            "--config", mock_config_file,
            "--agents-dir", mock_agents_dir
        ])
        
        # Should exit with code 1 on connection failure
        assert result.exit_code == 1
        assert "GitLab connection failed" in str(result.exception)


@pytest.mark.integration
def test_monitor_command_keyboard_interrupt(mock_config_file, mock_agents_dir):
    """Test the monitor command handles keyboard interrupt gracefully."""
    
    with patch("factory.core.monitor.GitLabClient") as mock_gl_client_class:
        mock_gl = MagicMock()
        mock_gl.hostname = "gitlab.example.com"
        mock_gl.project_path = "test/project"
        mock_gl.connect.return_value = (True, "Connected")
        mock_gl_client_class.return_value = mock_gl
        
        with patch("factory.core.monitor.CompetenceManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm.agents = []
            mock_cm_class.return_value = mock_cm
            
            with patch("factory.core.monitor.TokenBudgetManager") as mock_budget_class:
                mock_budget = MagicMock()
                mock_budget.usage_report.return_value = {"agents": {}}
                mock_budget_class.return_value = mock_budget
                
                # Mock the monitor's run method to simulate keyboard interrupt
                with patch("factory.core.monitor.FactoryMonitor.run") as mock_run:
                    mock_run.side_effect = KeyboardInterrupt
                    
                    result = runner.invoke(app, [
                        "monitor",
                        "--config", mock_config_file,
                        "--agents-dir", mock_agents_dir
                    ])
                    
                    # KeyboardInterrupt should be handled gracefully
                    # Exit code 130 is expected for KeyboardInterrupt
                    assert result.exit_code == 130


@pytest.mark.integration
def test_monitor_command_no_architect_agent(
    mock_config_file, 
    mock_agents_dir, 
    mock_token_usage_file, 
    mock_sprint_info
):
    """Test the monitor command when no architect agent is available."""
    
    with patch("factory.core.monitor.GitLabClient") as mock_gl_client_class:
        mock_gl = MagicMock()
        mock_gl.hostname = "gitlab.example.com"
        mock_gl.project_path = "test/project"
        mock_gl.connect.return_value = (True, "Connected")
        mock_gl.sprint_info.return_value = mock_sprint_info
        mock_gl.days_remaining.return_value = 7
        mock_gl_client_class.return_value = mock_gl
        
        with patch("factory.core.monitor.CompetenceManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm.get.return_value = None  # No architect agent
            mock_cm.agents = []
            mock_cm_class.return_value = mock_cm
            
            with patch("factory.core.monitor.TokenBudgetManager") as mock_budget_class:
                mock_budget = MagicMock()
                mock_budget.usage_report.return_value = {"agents": {}}
                mock_budget_class.return_value = mock_budget
                
                # Mock the monitor's run method to avoid infinite loop
                with patch("factory.core.monitor.FactoryMonitor.run") as mock_run:
                    result = runner.invoke(app, [
                        "monitor",
                        "--config", mock_config_file,
                        "--agents-dir", mock_agents_dir
                    ])
                    
                    # Should still succeed even without architect
                    assert result.exit_code == 0
                    assert "No 'architect' agent profile found" in result.output
                    mock_run.assert_called_once()
