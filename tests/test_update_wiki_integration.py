"""Integration tests for update_wiki command with realistic scenarios."""

import pytest
from unittest.mock import Mock, patch
from factory.commands.update_wiki import analyze_mr_changes, cmd_update_wiki


def test_analyze_mr_changes_comprehensive():
    """Test analyze_mr_changes with a comprehensive set of changes."""
    # Mock merge request with various types of changes
    mock_mr = Mock()
    mock_mr.changes.return_value = {
        "changes": [
            {
                "new_path": "src/features/authentication.py",
                "old_path": "src/features/authentication.py",
                "diff": "+def login_user(username, password):\n    # New authentication function"
            },
            {
                "new_path": "src/features/payment.py",
                "old_path": "src/features/payment.py",
                "diff": "BREAKING CHANGE: Changed payment API to v2"
            },
            {
                "new_path": "docs/API.md",
                "old_path": "docs/API.md",
                "diff": "+# New Authentication API\n\nAdded login_user endpoint"
            },
            {
                "new_path": "docs/CHANGELOG.md",
                "old_path": "docs/CHANGELOG.md",
                "diff": "+## v1.2.0\n\n- Added authentication feature"
            },
            {
                "new_path": "src/utils/helpers.py",
                "old_path": "src/utils/helpers.py",
                "diff": "+def format_currency(amount):\n    # Helper function for currency formatting"
            },
            {
                "new_path": "tests/test_authentication.py",
                "old_path": "tests/test_authentication.py",
                "diff": "+def test_login_user():\n    # Test for new login function"
            },
        ]
    }
    
    changes_summary, files_changed, documentation_updates, breaking_changes = analyze_mr_changes(mock_mr)
    
    # Verify summary includes correct count
    assert "6 file(s) changed" in changes_summary
    
    # Verify files are categorized correctly
    assert "src/features/authentication.py" in files_changed
    assert "src/features/payment.py" in files_changed
    assert "src/utils/helpers.py" in files_changed
    assert "tests/test_authentication.py" in files_changed
    
    # Verify documentation updates
    assert "docs/API.md" in documentation_updates
    assert "docs/CHANGELOG.md" in documentation_updates
    
    # Verify breaking changes
    assert "src/features/payment.py" in breaking_changes


def test_cmd_update_wiki_comprehensive():
    """Test cmd_update_wiki with a comprehensive merge request."""
    with patch("factory.commands.update_wiki.GitLabClient") as mock_gl_client, \
         patch("factory.commands.update_wiki.Console") as mock_console:
        
        # Setup mock GitLab client
        mock_gl = Mock()
        mock_gl.connect.return_value = (True, "Connected")
        mock_gl.project = Mock()
        
        # Mock merge request with comprehensive data
        mock_mr = Mock()
        mock_mr.title = "Add user authentication feature"
        mock_mr.author = {"name": "John Developer", "username": "john.dev"}
        mock_mr.source_branch = "feature/authentication"
        mock_mr.target_branch = "main"
        mock_mr.state = "opened"
        mock_mr.web_url = "https://gitlab.example.com/test/project/-/merge_requests/42"
        mock_mr.description = "This MR adds user authentication with JWT tokens."
        mock_mr.changes.return_value = {
            "changes": [
                {
                    "new_path": "src/auth/auth.py",
                    "old_path": "src/auth/auth.py",
                    "diff": "+def authenticate_user(token):\n    # New authentication function"
                },
                {
                    "new_path": "src/auth/token.py",
                    "old_path": "src/auth/token.py",
                    "diff": "+def generate_jwt(user_id):\n    # JWT token generation"
                },
                {
                    "new_path": "docs/authentication.md",
                    "old_path": "docs/authentication.md",
                    "diff": "+# Authentication Guide\n\nHow to use the new auth system"
                },
            ]
        }
        
        mock_gl.project.mergerequests.get.return_value = mock_mr
        mock_gl.project.wikis.list.return_value = []
        mock_gl.project.wikis.create.return_value = None
        mock_gl._url = "https://gitlab.example.com"
        mock_gl.project_path = "test/project"
        mock_gl_client.return_value = mock_gl
        
        # Run the command
        cmd_update_wiki(mr_id=42, config="config/factory.yml")
        
        # Verify wiki creation was called with correct parameters
        mock_gl.project.wikis.create.assert_called_once()
        args, _ = mock_gl.project.wikis.create.call_args
        wiki_data = args[0]
        
        # Verify wiki title
        assert wiki_data["title"] == "MR !42 Update"
        
        # Verify wiki content contains key information
        content = wiki_data["content"]
        assert "Add user authentication feature" in content
        assert "John Developer" in content
        assert "feature/authentication" in content
        assert "main" in content
        assert "3 file(s) changed" in content
        assert "src/auth/auth.py" in content
        assert "src/auth/token.py" in content
        assert "docs/authentication.md" in content


def test_cmd_update_wiki_existing_page_update():
    """Test updating an existing wiki page with new content."""
    with patch("factory.commands.update_wiki.GitLabClient") as mock_gl_client, \
         patch("factory.commands.update_wiki.Console") as mock_console:
        
        # Setup mock GitLab client
        mock_gl = Mock()
        mock_gl.connect.return_value = (True, "Connected")
        mock_gl.project = Mock()
        
        # Mock merge request
        mock_mr = Mock()
        mock_mr.title = "Update authentication feature"
        mock_mr.author = {"name": "Jane Developer", "username": "jane.dev"}
        mock_mr.source_branch = "feature/auth-update"
        mock_mr.target_branch = "main"
        mock_mr.state = "merged"
        mock_mr.changes.return_value = {
            "changes": [
                {
                    "new_path": "src/auth/auth.py",
                    "old_path": "src/auth/auth.py",
                    "diff": "-def old_auth():\n+def new_auth():\n    # Updated authentication"
                },
            ]
        }
        
        mock_gl.project.mergerequests.get.return_value = mock_mr
        
        # Mock existing wiki page
        mock_wiki = Mock()
        mock_wiki.slug = "MR-42-Update"
        mock_wiki.content = "Old content"
        mock_gl.project.wikis.list.return_value = [mock_wiki]
        mock_gl.project.wikis.get.return_value = mock_wiki
        mock_gl._url = "https://gitlab.example.com"
        mock_gl.project_path = "test/project"
        mock_gl_client.return_value = mock_gl
        
        # Run the command
        cmd_update_wiki(mr_id=42, config="config/factory.yml")
        
        # Verify wiki update was called
        mock_wiki.save.assert_called_once()
        
        # Verify the content was updated
        assert mock_wiki.content != "Old content"
        assert "Update authentication feature" in mock_wiki.content
        assert "Jane Developer" in mock_wiki.content


def test_cmd_update_wiki_error_handling():
    """Test error handling in cmd_update_wiki."""
    with patch("factory.commands.update_wiki.GitLabClient") as mock_gl_client, \
         patch("factory.commands.update_wiki.Console") as mock_console:
        
        # Setup mock GitLab client to raise an exception
        mock_gl = Mock()
        mock_gl.connect.return_value = (True, "Connected")
        mock_gl.project = Mock()
        mock_gl.project.mergerequests.get.side_effect = Exception("MR not found")
        mock_gl_client.return_value = mock_gl
        
        # Run the command and verify it handles the error gracefully
        # The command should exit with code 1 when MR is not found
        with pytest.raises(Exception) as exc_info:
            cmd_update_wiki(mr_id=999, config="config/factory.yml")
        
        # Verify the exception is Exit with code 1
        assert str(exc_info.value) == "1"


def test_analyze_mr_changes_empty():
    """Test analyze_mr_changes with empty changes."""
    mock_mr = Mock()
    mock_mr.changes.return_value = {"changes": []}
    
    changes_summary, files_changed, documentation_updates, breaking_changes = analyze_mr_changes(mock_mr)
    
    assert "0 file(s) changed" in changes_summary
    assert files_changed == "_None_"
    assert documentation_updates == "_None_"
    assert breaking_changes == "_None_"


def test_analyze_mr_changes_no_breaking_changes():
    """Test analyze_mr_changes with no breaking changes."""
    mock_mr = Mock()
    mock_mr.changes.return_value = {
        "changes": [
            {
                "new_path": "src/file1.py",
                "old_path": "src/file1.py",
                "diff": "Some normal changes here"
            },
            {
                "new_path": "docs/README.md",
                "old_path": "docs/README.md",
                "diff": "Documentation update"
            },
        ]
    }
    
    changes_summary, files_changed, documentation_updates, breaking_changes = analyze_mr_changes(mock_mr)
    
    assert "2 file(s) changed" in changes_summary
    assert "src/file1.py" in files_changed
    assert "docs/README.md" in documentation_updates
    assert breaking_changes == "_None_"
