#"""Unit tests for the label_issue command."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from factory.commands.label_issue import app
from typer.testing import CliRunner


runner = CliRunner()


@patch("factory.commands.label_issue.GitLabClient")
@patch("factory.commands.label_issue.ScrumEngine")
def test_label_issue_ready(mock_scrum, mock_gl):
    """Test labeling an issue as ready."""
    # Setup mocks
    mock_project = Mock()
    mock_issue = Mock()
    mock_issue.iid = 1
    mock_issue.title = "Test Issue"
    mock_issue.state = "opened"
    mock_issue.labels = ["bug"]
    mock_project.issues.get.return_value = mock_issue
    
    mock_gl_instance = Mock()
    mock_gl_instance.project = mock_project
    mock_gl_instance.connect.return_value = (True, "Connected")
    mock_gl.return_value = mock_gl_instance
    
    # Run command
    result = runner.invoke(app, ["--issue", "1", "--ready"])
    
    # Assertions
    assert result.exit_code == 0
    assert "Adding 'ready' label to issue" in result.output
    assert "Issue labels updated successfully!" in result.output
    mock_issue.save.assert_called_once()


@patch("factory.commands.label_issue.GitLabClient")
@patch("factory.commands.label_issue.ScrumEngine")
def test_label_issue_blocked(mock_scrum, mock_gl):
    """Test labeling an issue as blocked."""
    # Setup mocks
    mock_project = Mock()
    mock_issue = Mock()
    mock_issue.iid = 2
    mock_issue.title = "Blocked Issue"
    mock_issue.state = "opened"
    mock_issue.labels = ["ready"]
    mock_project.issues.get.return_value = mock_issue
    
    mock_gl_instance = Mock()
    mock_gl_instance.project = mock_project
    mock_gl_instance.connect.return_value = (True, "Connected")
    mock_gl.return_value = mock_gl_instance
    
    # Run command
    result = runner.invoke(app, ["--issue", "2", "--blocked"])
    
    # Assertions
    assert result.exit_code == 0
    assert "Adding 'blocked' label to issue" in result.output
    assert "Issue labels updated successfully!" in result.output
    mock_issue.save.assert_called_once()


@patch("factory.commands.label_issue.GitLabClient")
@patch("factory.commands.label_issue.ScrumEngine")
def test_label_issue_next_sprint(mock_scrum, mock_gl):
    """Test labeling an issue for next sprint."""
    # Setup mocks
    mock_project = Mock()
    mock_issue = Mock()
    mock_issue.iid = 3
    mock_issue.title = "Sprint Issue"
    mock_issue.state = "opened"
    mock_issue.labels = ["feature"]
    mock_project.issues.get.return_value = mock_issue
    
    mock_gl_instance = Mock()
    mock_gl_instance.project = mock_project
    mock_gl_instance.connect.return_value = (True, "Connected")
    mock_gl.return_value = mock_gl_instance
    
    # Run command
    result = runner.invoke(app, ["--issue", "3", "--next-sprint"])
    
    # Assertions
    assert result.exit_code == 0
    assert "Adding 'sprint-next' label to issue" in result.output
    assert "Issue labels updated successfully!" in result.output
    mock_issue.save.assert_called_once()


@patch("factory.commands.label_issue.GitLabClient")
@patch("factory.commands.label_issue.ScrumEngine")
def test_label_issue_both_flags(mock_scrum, mock_gl):
    """Test labeling an issue with both ready and next-sprint flags."""
    # Setup mocks
    mock_project = Mock()
    mock_issue = Mock()
    mock_issue.iid = 4
    mock_issue.title = "Ready Sprint Issue"
    mock_issue.state = "opened"
    mock_issue.labels = ["enhancement"]
    mock_project.issues.get.return_value = mock_issue
    
    mock_gl_instance = Mock()
    mock_gl_instance.project = mock_project
    mock_gl_instance.connect.return_value = (True, "Connected")
    mock_gl.return_value = mock_gl_instance
    
    # Run command
    result = runner.invoke(app, ["--issue", "4", "--ready", "--next-sprint"])
    
    # Assertions
    assert result.exit_code == 0
    assert "Adding 'ready' label to issue" in result.output
    assert "Adding 'sprint-next' label to issue" in result.output
    assert "Issue labels updated successfully!" in result.output
    mock_issue.save.assert_called_once()


@patch("factory.commands.label_issue.GitLabClient")
@patch("factory.commands.label_issue.ScrumEngine")
def test_label_issue_ready_and_blocked(mock_scrum, mock_gl):
    """Test that both ready and blocked flags cannot be specified together."""
    # Setup mocks
    mock_project = Mock()
    mock_issue = Mock()
    mock_issue.iid = 5
    mock_issue.title = "Ready Blocked Issue"
    mock_issue.state = "opened"
    mock_issue.labels = ["enhancement"]
    mock_project.issues.get.return_value = mock_issue
    
    mock_gl_instance = Mock()
    mock_gl_instance.project = mock_project
    mock_gl_instance.connect.return_value = (True, "Connected")
    mock_gl.return_value = mock_gl_instance
    
    # Run command
    result = runner.invoke(app, ["--issue", "5", "--ready", "--blocked"])
    
    # Assertions
    assert result.exit_code == 1
    assert "Cannot specify both --ready and --blocked" in result.output
    assert "An issue cannot be both ready and blocked" in result.output


@patch("factory.commands.label_issue.GitLabClient")
@patch("factory.commands.label_issue.ScrumEngine")
def test_label_issue_closed(mock_scrum, mock_gl):
    """Test that closed issues cannot be labeled."""
    # Setup mocks
    mock_project = Mock()
    mock_issue = Mock()
    mock_issue.iid = 6
    mock_issue.title = "Closed Issue"
    mock_issue.state = "closed"
    mock_project.issues.get.return_value = mock_issue
    
    mock_gl_instance = Mock()
    mock_gl_instance.project = mock_project
    mock_gl_instance.connect.return_value = (True, "Connected")
    mock_gl.return_value = mock_gl_instance
    
    # Run command
    result = runner.invoke(app, ["--issue", "6", "--ready"])
    
    # Assertions
    assert result.exit_code == 1
    assert "Issue #6 is not open" in result.output
    assert "Only open issues can be labeled" in result.output


@patch("factory.commands.label_issue.GitLabClient")
@patch("factory.commands.label_issue.ScrumEngine")
def test_label_issue_nonexistent(mock_scrum, mock_gl):
    """Test that non-existent issues cannot be labeled."""
    # Setup mocks
    mock_project = Mock()
    mock_project.issues.get.return_value = None
    
    mock_gl_instance = Mock()
    mock_gl_instance.project = mock_project
    mock_gl_instance.connect.return_value = (True, "Connected")
    mock_gl.return_value = mock_gl_instance
    
    # Run command
    result = runner.invoke(app, ["--issue", "999", "--ready"])
    
    # Assertions
    assert result.exit_code == 1
    assert "Issue #999 does not exist" in result.output


@patch("factory.commands.label_issue.GitLabClient")
@patch("factory.commands.label_issue.ScrumEngine")
def test_label_issue_no_flags(mock_scrum, mock_gl):
    """Test that command fails when no labeling flags are provided."""
    # Setup mocks
    mock_project = Mock()
    mock_issue = Mock()
    mock_issue.iid = 7
    mock_issue.title = "No Flags Issue"
    mock_issue.state = "opened"
    mock_issue.labels = ["task"]
    mock_project.issues.get.return_value = mock_issue
    
    mock_gl_instance = Mock()
    mock_gl_instance.project = mock_project
    mock_gl_instance.connect.return_value = (True, "Connected")
    mock_gl.return_value = mock_gl_instance
    
    # Run command
    result = runner.invoke(app, ["--issue", "7"])
    
    # Assertions
    assert result.exit_code == 1
    assert "No labeling action specified" in result.output


@patch("factory.commands.label_issue.GitLabClient")
@patch("factory.commands.label_issue.ScrumEngine")
def test_label_issue_connection_failure(mock_scrum, mock_gl):
    """Test behavior when GitLab connection fails."""
    # Setup mocks
    mock_gl_instance = Mock()
    mock_gl_instance.connect.return_value = (False, "Connection failed")
    mock_gl.return_value = mock_gl_instance
    
    # Run command
    result = runner.invoke(app, ["--issue", "8", "--ready"])
    
    # Assertions
    assert result.exit_code == 1
    assert "GitLab connection failed" in result.output


@patch("factory.commands.label_issue.GitLabClient")
@patch("factory.commands.label_issue.ScrumEngine")
def test_label_issue_update_failure(mock_scrum, mock_gl):
    """Test behavior when label update fails."""
    # Setup mocks
    mock_project = Mock()
    mock_issue = Mock()
    mock_issue.iid = 9
    mock_issue.title = "Update Failure Issue"
    mock_issue.state = "opened"
    mock_issue.labels = ["bug"]
    mock_issue.save.side_effect = Exception("Update failed")
    mock_project.issues.get.return_value = mock_issue
    
    mock_gl_instance = Mock()
    mock_gl_instance.project = mock_project
    mock_gl_instance.connect.return_value = (True, "Connected")
    mock_gl.return_value = mock_gl_instance
    
    # Run command
    result = runner.invoke(app, ["--issue", "9", "--ready"])
    
    # Assertions
    assert result.exit_code == 1
    assert "Failed to update labels" in result.output
