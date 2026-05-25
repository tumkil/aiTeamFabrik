"""Integration tests for the factory plan CLI command."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from datetime import datetime, timezone
import yaml

from factory.main import app
from factory.core.planner import TokenAwarePlanner, SprintPlan, PlanEntry, PlanningAlgorithm


runner = CliRunner()


def test_plan_command_show_existing_plan():
    """Test that 'factory plan --show' displays an existing plan."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        plan_path = Path(tmpdir) / "sprint-1-plan.yml"
        
        # Write a minimal config file
        config_path.write_text("""
gitlab:
  url: https://gitlab.example.com
  token: test-token
  project: test/project
""")
        
        # Create a valid plan using the planner API
        planner = TokenAwarePlanner(plan_path)
        plan = SprintPlan(
            sprint_number="sprint-1",
            algorithm=PlanningAlgorithm.GREEDY,
            budget=1000000,
            entries=[
                PlanEntry(
                    issue_iid=1,
                    title="Test Issue",
                    agent="developer",
                    estimated_tokens=50000,
                    priority=5,
                    scheduled_at=datetime.now(timezone.utc),
                )
            ]
        )
        
        # Save the plan using the planner's serialization
        with open(plan_path, 'w') as f:
            yaml.safe_dump(plan.to_dict(), f, sort_keys=False)
        
        # Mock GitLabClient to avoid real API calls
        with patch('factory.commands.plan.GitLabClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.connect.return_value = (True, "Connected")
            mock_instance.project = MagicMock()
            mock_instance.project_path = "test/project"
            mock_client.return_value = mock_instance
            
            # Run the command with --show flag
            result = runner.invoke(app, [
                "plan",
                "--sprint", "1",
                "--config", str(config_path),
                "--show"
            ])
            
            # Should succeed
            assert result.exit_code == 0
            assert "sprint-1" in result.output
            assert "Test Issue" in result.output


def test_plan_command_create_new_plan():
    """Test that 'factory plan' creates a new plan when none exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        plan_path = Path(tmpdir) / "sprint-1-plan.yml"
        
        # Write a minimal config file
        config_path.write_text("""
gitlab:
  url: https://gitlab.example.com
  token: test-token
  project: test/project
""")
        
        # Mock GitLabClient to return sample issues
        with patch('factory.commands.plan.GitLabClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.connect.return_value = (True, "Connected")
            mock_instance.project = MagicMock()
            mock_instance.project_path = "test/project"
            
            # Mock issues list
            mock_issue1 = MagicMock()
            mock_issue1.iid = 1
            mock_issue1.title = "Test Feature"
            mock_issue1.labels = ["feature", "sprint-1"]
            mock_issue1.description = "Test description"
            
            mock_issue2 = MagicMock()
            mock_issue2.iid = 2
            mock_issue2.title = "Test Bug"
            mock_issue2.labels = ["bug", "sprint-1"]
            mock_issue2.description = "Bug description"
            
            mock_instance.project.issues.list.return_value = [mock_issue1, mock_issue2]
            mock_client.return_value = mock_instance
            
            # Run the command
            result = runner.invoke(app, [
                "plan",
                "--sprint", "1",
                "--config", str(config_path),
                "--algorithm", "greedy",
                "--budget", "1000000"
            ])
            
            # Should succeed
            assert result.exit_code == 0
            assert "Plan created successfully" in result.output
            assert plan_path.exists()
            
            # Verify plan file was created
            plan_content = plan_path.read_text()
            assert "sprint-1" in plan_content
            assert "Test Feature" in plan_content or "Test Bug" in plan_content


def test_plan_command_invalid_algorithm():
    """Test that invalid algorithm names are rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        
        # Write a minimal config file
        config_path.write_text("""
gitlab:
  url: https://gitlab.example.com
  token: test-token
  project: test/project
""")
        
        # Run with invalid algorithm
        result = runner.invoke(app, [
            "plan",
            "--sprint", "1",
            "--config", str(config_path),
            "--algorithm", "invalid-algorithm"
        ])
        
        # Should fail with appropriate error
        assert result.exit_code == 2  # Typer exit code for parameter error
        assert "Invalid algorithm" in result.output


def test_plan_command_invalid_budget():
    """Test that invalid budget values are rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        
        # Write a minimal config file
        config_path.write_text("""
gitlab:
  url: https://gitlab.example.com
  token: test-token
  project: test/project
""")
        
        # Test zero budget
        result = runner.invoke(app, [
            "plan",
            "--sprint", "1",
            "--config", str(config_path),
            "--budget", "0"
        ])
        
        assert result.exit_code == 2
        assert "Budget must be a positive integer" in result.output
        
        # Test negative budget
        result = runner.invoke(app, [
            "plan",
            "--sprint", "1",
            "--config", str(config_path),
            "--budget", "-100"
        ])
        
        assert result.exit_code == 2
        assert "Budget must be a positive integer" in result.output


def test_plan_command_gitlab_connection_failure():
    """Test behavior when GitLab connection fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        
        # Write a minimal config file
        config_path.write_text("""
gitlab:
  url: https://gitlab.example.com
  token: invalid-token
  project: test/project
""")
        
        # Mock GitLabClient to fail connection
        with patch('factory.commands.plan.GitLabClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.connect.return_value = (False, "Invalid token")
            mock_client.return_value = mock_instance
            
            result = runner.invoke(app, [
                "plan",
                "--sprint", "1",
                "--config", str(config_path)
            ])
            
            assert result.exit_code == 1
            assert "GitLab connection failed" in result.output


def test_plan_command_no_issues_found():
    """Test behavior when no issues are found for the sprint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        
        # Write a minimal config file
        config_path.write_text("""
gitlab:
  url: https://gitlab.example.com
  token: test-token
  project: test/project
""")
        
        # Mock GitLabClient with no issues
        with patch('factory.commands.plan.GitLabClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.connect.return_value = (True, "Connected")
            mock_instance.project = MagicMock()
            mock_instance.project_path = "test/project"
            mock_instance.project.issues.list.return_value = []
            mock_client.return_value = mock_instance
            
            result = runner.invoke(app, [
                "plan",
                "--sprint", "1",
                "--config", str(config_path)
            ])
            
            assert result.exit_code == 0
            assert "No issues found for sprint 1" in result.output
