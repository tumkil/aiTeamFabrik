"""Tests for the update_wiki command."""

import pytest
from unittest.mock import Mock, patch
from factory.commands.update_wiki import analyze_mr_changes, cmd_update_wiki


def test_analyze_mr_changes():
    """Test the analysis of merge request changes."""
    # Mock merge request changes
    mock_mr = Mock()
    mock_mr.changes.return_value = {
        "changes": [
            {
                "new_path": "src/file1.py",
                "old_path": "src/file1.py",
                "diff": "Some changes here"
            },
            {
                "new_path": "docs/README.md",
                "old_path": "docs/README.md",
                "diff": "Documentation update"
            },
            {
                "new_path": "src/file2.py",
                "old_path": "src/file2.py",
                "diff": "BREAKING CHANGE: This is a breaking change"
            }
        ]
    }
    
    changes_summary, files_changed, documentation_updates, breaking_changes = analyze_mr_changes(mock_mr)
    
    assert "3 file(s) changed" in changes_summary
    assert "src/file1.py" in files_changed
    assert "docs/README.md" in documentation_updates
    assert "src/file2.py" in breaking_changes


def test_cmd_update_wiki():
    """Test the update_wiki command."""
    with patch("factory.commands.update_wiki.GitLabClient") as mock_gl_client, \
         patch("factory.commands.update_wiki.Console") as mock_console:
        
        # Setup mock GitLab client
        mock_gl = Mock()
        mock_gl.connect.return_value = (True, "Connected")
        mock_gl.project = Mock()
        mock_gl.project.mergerequests.get.return_value = Mock(
            title="Test MR",
            author={"name": "Test User"},
            source_branch="test-branch",
            target_branch="main",
            state="opened",
            changes=lambda: {
                "changes": [
                    {
                        "new_path": "src/file1.py",
                        "old_path": "src/file1.py",
                        "diff": "Some changes here"
                    }
                ]
            }
        )
        mock_gl.project.wikis.list.return_value = []
        mock_gl.project.wikis.create.return_value = None
        mock_gl._url = "https://gitlab.example.com"
        mock_gl.project_path = "test/project"
        mock_gl_client.return_value = mock_gl
        
        # Run the command
        cmd_update_wiki(mr_id=1, config="config/factory.yml")
        
        # Verify wiki creation was called
        mock_gl.project.wikis.create.assert_called_once()
        args, _ = mock_gl.project.wikis.create.call_args
        assert args[0]["title"] == "MR !1 Update"
        assert "Merge Request Update" in args[0]["content"]


def test_cmd_update_wiki_existing():
    """Test updating an existing wiki page."""
    with patch("factory.commands.update_wiki.GitLabClient") as mock_gl_client, \
         patch("factory.commands.update_wiki.Console") as mock_console:
        
        # Setup mock GitLab client
        mock_gl = Mock()
        mock_gl.connect.return_value = (True, "Connected")
        mock_gl.project = Mock()
        mock_gl.project.mergerequests.get.return_value = Mock(
            title="Test MR",
            author={"name": "Test User"},
            source_branch="test-branch",
            target_branch="main",
            state="opened",
            changes=lambda: {
                "changes": [
                    {
                        "new_path": "src/file1.py",
                        "old_path": "src/file1.py",
                        "diff": "Some changes here"
                    }
                ]
            }
        )
        
        mock_wiki = Mock()
        mock_wiki.slug = "MR-1-Update"
        mock_gl.project.wikis.list.return_value = [mock_wiki]
        mock_gl.project.wikis.get.return_value = mock_wiki
        mock_gl._url = "https://gitlab.example.com"
        mock_gl.project_path = "test/project"
        mock_gl_client.return_value = mock_gl
        
        # Run the command
        cmd_update_wiki(mr_id=1, config="config/factory.yml")
        
        # Verify wiki update was called
        mock_wiki.save.assert_called_once()
