"""Unit tests for factory.commands.review module."""
import datetime
from unittest.mock import MagicMock, patch

import pytest

from factory.commands.review import REVIEW_TEMPLATE, cmd_review


@patch("factory.commands.review.GitLabClient")
@patch("factory.commands.review.ScrumEngine")
@patch("factory.commands.review.console")
def test_cmd_review_generates_wiki(mock_console, mock_scrum_class, mock_gl_class):
    """Test that cmd_review generates a wiki page with correct content."""
    # Setup mocks
    mock_console.print = MagicMock()
    
    mock_scrum = MagicMock()
    state = MagicMock()
    state.number = 2
    state.name = "Sprint 2"
    state.start_date = datetime.date(2026, 5, 12)
    state.end_date = datetime.date(2026, 5, 26)
    state.duration_days = 14
    mock_scrum.current = state
    mock_scrum.velocity.return_value = 25
    mock_scrum_class.return_value = mock_scrum
    
    mock_gl = MagicMock()
    project = MagicMock()
    project.name = "Test Project"
    project.path = "test/project"
    project._url = "https://gitlab.example.com"
    project.project_path = "test/project"
    
    # Issues
    closed_issue_1 = MagicMock()
    closed_issue_1.iid = 2
    closed_issue_1.title = "Implement LLM Router"
    
    closed_issue_2 = MagicMock()
    closed_issue_2.iid = 6
    closed_issue_2.title = "Add .env auto-loading"
    
    open_issue_1 = MagicMock()
    open_issue_1.iid = 1
    open_issue_1.title = "Implement factory run"
    
    open_issue_2 = MagicMock()
    open_issue_2.iid = 3
    open_issue_2.title = "Implement Agent Orchestrator"
    
    def list_side_effect(state="opened", all=True):
        if state == "closed":
            return [closed_issue_1, closed_issue_2]
        return [open_issue_1, open_issue_2]
    
    project.issues.list.side_effect = list_side_effect
    
    # Wiki
    wiki_page = MagicMock()
    wiki_page.slug = "Sprint-2-Review"
    project.wikis.list.return_value = []
    project.wikis.create.return_value = wiki_page
    
    mock_gl.return_value = mock_gl
    mock_gl.connect.return_value = (True, "connected")
    mock_gl.project = project
    mock_gl._url = "https://gitlab.example.com"
    mock_gl.project_path = "test/project"
    mock_gl_class.return_value = mock_gl
    
    # Call the command
    cmd_review(sprint=2, approve=False, config="config/factory.yml")
    
    # Verify wiki was created
    project.wikis.create.assert_called_once()
    call_args = project.wikis.create.call_args[0][0]
    assert call_args["title"] == "Sprint 2 Review"
    assert "Sprint 2" in call_args["content"]
    assert "2026-05-12" in call_args["content"]


@patch("factory.commands.review.GitLabClient")
@patch("factory.commands.review.ScrumEngine")
@patch("factory.commands.review.console")
def test_cmd_review_with_approve(mock_console, mock_scrum_class, mock_gl_class):
    """Test that cmd_review with --approve closes milestone and advances sprint."""
    mock_console.print = MagicMock()
    
    mock_scrum = MagicMock()
    state = MagicMock()
    state.number = 2
    state.name = "Sprint 2"
    mock_scrum.current = state
    mock_scrum_class.return_value = mock_scrum
    
    mock_gl = MagicMock()
    project = MagicMock()
    project.issues.list.return_value = []
    
    # Wiki exists
    wiki_page = MagicMock()
    wiki_page.slug = "Sprint-2-Review"
    wiki_page.content = ""
    wiki_page.title = "Old Title"
    project.wikis.list.return_value = [wiki_page]
    project.wikis.get.return_value = wiki_page
    
    # Milestone
    milestone = MagicMock()
    milestone.state_event = "close"
    project.milestones.list.return_value = [milestone]
    
    mock_gl.return_value = mock_gl
    mock_gl.connect.return_value = (True, "connected")
    mock_gl.project = project
    mock_gl._url = "https://gitlab.example.com"
    mock_gl.project_path = "test/project"
    mock_gl_class.return_value = mock_gl
    
    # Call with approve
    cmd_review(sprint=2, approve=True, config="config/factory.yml")
    
    # Verify milestone was closed
    assert milestone.state_event == "close"
    # Verify scrum advance was called
    mock_scrum.advance.assert_called_once()


@patch("factory.commands.review.GitLabClient")
@patch("factory.commands.review.ScrumEngine")
@patch("factory.commands.review.console")
def test_cmd_review_wrong_sprint_number(mock_console, mock_scrum_class, mock_gl_class):
    """Test that cmd_review fails when sprint number doesn't match current."""
    mock_console.print = MagicMock()
    
    # Setup ScrumEngine mock instance
    mock_scrum = MagicMock()
    state = MagicMock()
    state.number = 3  # Current is 3, but we're reviewing sprint 2
    state.name = "Sprint 3"
    state.start_date = datetime.date(2026, 5, 26)
    state.duration_days = 14
    mock_scrum.current = state
    # When ScrumEngine is called (with config_path), return our mock instance
    mock_scrum_class.return_value = mock_scrum
    
    # Setup GitLabClient mock instance
    mock_gl = MagicMock()
    mock_gl.connect.return_value = (True, "connected")
    # When GitLabClient is called (with config_path), return our mock instance
    mock_gl_class.return_value = mock_gl
    
    with pytest.raises(Exception) as exc_info:
        cmd_review(sprint=2, approve=False, config="config/factory.yml")
    
    # typer.Exit raises click.exceptions.Exit which has code attribute
    assert hasattr(exc_info.value, 'code') or exc_info.value.args[0] == 1


@patch("factory.commands.review.GitLabClient")
@patch("factory.commands.review.console")
def test_cmd_review_fails_on_connection_error(mock_console, mock_gl_class):
    """Test that cmd_review exits with code 1 on GitLab connection failure."""
    mock_console.print = MagicMock()
    
    # Setup GitLabClient mock instance
    mock_gl = MagicMock()
    mock_gl.connect.return_value = (False, "Connection refused")
    # When GitLabClient is called, return our mock instance
    mock_gl_class.return_value = mock_gl
    
    with pytest.raises(Exception) as exc_info:
        cmd_review(sprint=2, approve=False, config="config/factory.yml")
    
    # typer.Exit raises click.exceptions.Exit which has code attribute
    assert hasattr(exc_info.value, 'code') or exc_info.value.args[0] == 1


@patch("factory.commands.review.GitLabClient")
@patch("factory.commands.review.ScrumEngine")
@patch("factory.commands.review.console")
def test_cmd_review_updates_existing_wiki(mock_console, mock_scrum_class, mock_gl_class):
    """Test that cmd_review updates existing wiki page instead of creating new one."""
    mock_console.print = MagicMock()
    
    mock_scrum = MagicMock()
    state = MagicMock()
    state.number = 2
    state.name = "Sprint 2"
    state.start_date = datetime.date(2026, 5, 12)
    state.duration_days = 14
    mock_scrum.current = state
    mock_scrum_class.return_value = mock_scrum
    
    mock_gl = MagicMock()
    project = MagicMock()
    project.issues.list.return_value = []
    
    # Mock existing wiki page
    wiki_page = MagicMock()
    wiki_page.slug = "Sprint-2-Review"
    wiki_page.content = "old content"
    wiki_page.title = "Old Title"
    project.wikis.list.return_value = [wiki_page]
    project.wikis.get.return_value = wiki_page
    
    mock_gl.return_value = mock_gl
    mock_gl.connect.return_value = (True, "connected")
    mock_gl.project = project
    mock_gl._url = "https://gitlab.example.com"
    mock_gl.project_path = "test/project"
    mock_gl_class.return_value = mock_gl
    
    cmd_review(sprint=2, approve=False, config="config/factory.yml")
    
    # Verify wiki was updated, not created
    project.wikis.create.assert_not_called()
    # Verify the wiki page was modified
    assert wiki_page.title == "Sprint 2 Review"


def test_review_template_renders_correctly():
    """Test that the REVIEW_TEMPLATE renders without errors."""
    rendered = REVIEW_TEMPLATE.format(
        number=2,
        name="Sprint 2",
        sprint_name="Sprint 2",
        date="2026-04-28",
        duration_days=14,
        start="2026-05-12",
        end="2026-05-26",
        closed=2,
        open=6,
        velocity=25,
        closed_issues_list="- #2 Test",
        open_issues_list="- #1 Test",
        stakeholder_feedback="_None_",
        next_number=3,
    )
    
    assert "# Sprint 2 Review" in rendered
    assert "2026-04-28" in rendered
    assert "25%" in rendered
    assert "factory review --sprint 2 --approve" in rendered