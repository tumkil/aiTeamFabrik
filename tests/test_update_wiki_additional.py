"""Additional tests for update_wiki command edge cases and comprehensive coverage."""

import pytest
from unittest.mock import Mock, patch
from factory.commands.update_wiki import analyze_mr_changes, cmd_update_wiki


def test_analyze_mr_changes_empty():
    """Test analyze_mr_changes with empty changes."""
    mock_mr = Mock()
    mock_mr.changes.return_value = {"changes": []}

    changes_summary, files_changed, documentation_updates, breaking_changes = analyze_mr_changes(mock_mr)

    assert "0 file(s) changed" in changes_summary
    assert files_changed == "_None_"
    assert documentation_updates == "_None_"
    assert breaking_changes == "_None_"


def test_analyze_mr_changes_no_diff():
    """Test analyze_mr_changes with changes that have no diff content."""
    mock_mr = Mock()
    mock_mr.changes.return_value = {
        "changes": [
            {
                "new_path": "src/file1.py",
                "old_path": "src/file1.py",
                "diff": ""
            }
        ]
    }

    changes_summary, files_changed, documentation_updates, breaking_changes = analyze_mr_changes(mock_mr)

    assert "1 file(s) changed" in changes_summary
    assert "src/file1.py" in files_changed
    assert documentation_updates == "_None_"
    assert breaking_changes == "_None_"


def test_analyze_mr_changes_multiple_breaking_changes():
    """Test analyze_mr_changes with multiple breaking changes."""
    mock_mr = Mock()
    mock_mr.changes.return_value = {
        "changes": [
            {
                "new_path": "src/api.py",
                "old_path": "src/api.py",
                "diff": "BREAKING CHANGE: API endpoint removed"
            },
            {
                "new_path": "src/database.py",
                "old_path": "src/database.py",
                "diff": "BREAKING: Database schema changed"
            },
            {
                "new_path": "src/config.py",
                "old_path": "src/config.py",
                "diff": "Some normal change"
            }
        ]
    }

    changes_summary, files_changed, documentation_updates, breaking_changes = analyze_mr_changes(mock_mr)

    assert "3 file(s) changed" in changes_summary
    assert "src/api.py" in breaking_changes
    assert "src/database.py" in breaking_changes
    assert "src/config.py" not in breaking_changes


def test_analyze_mr_changes_documentation_files():
    """Test analyze_mr_changes with various documentation file patterns."""
    mock_mr = Mock()
    mock_mr.changes.return_value = {
        "changes": [
            {
                "new_path": "README.md",
                "old_path": "README.md",
                "diff": "Updated README"
            },
            {
                "new_path": "docs/API.md",
                "old_path": "docs/API.md",
                "diff": "API documentation"
            },
            {
                "new_path": "docs/guides/installation.md",
                "old_path": "docs/guides/installation.md",
                "diff": "Installation guide"
            },
            {
                "new_path": "CHANGELOG.md",
                "old_path": "CHANGELOG.md",
                "diff": "Version updates"
            },
            {
                "new_path": "src/code.py",
                "old_path": "src/code.py",
                "diff": "Code change"
            }
        ]
    }

    changes_summary, files_changed, documentation_updates, breaking_changes = analyze_mr_changes(mock_mr)

    # Check that documentation files are identified
    assert "README.md" in documentation_updates
    assert "docs/API.md" in documentation_updates
    assert "docs/guides/installation.md" in documentation_updates
    assert "CHANGELOG.md" in documentation_updates
    assert "src/code.py" not in documentation_updates


def test_cmd_update_wiki_multiple_documentation_updates():
    """Test cmd_update_wiki with multiple documentation updates."""
    with patch("factory.commands.update_wiki.GitLabClient") as mock_gl_client, \
         patch("factory.commands.update_wiki.Console") as mock_console:

        # Setup mock GitLab client
        mock_gl = Mock()
        mock_gl.connect.return_value = (True, "Connected")
        mock_gl.project = Mock()
        mock_gl.project.mergerequests.get.return_value = Mock(
            title="Test MR with docs",
            author={"name": "Test User"},
            source_branch="test-branch",
            target_branch="main",
            state="opened",
            changes=lambda: {
                "changes": [
                    {
                        "new_path": "README.md",
                        "old_path": "README.md",
                        "diff": "Updated README"
                    },
                    {
                        "new_path": "docs/API.md",
                        "old_path": "docs/API.md",
                        "diff": "API documentation update"
                    },
                    {
                        "new_path": "src/file1.py",
                        "old_path": "src/file1.py",
                        "diff": "Some code changes"
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
        content = args[0]["content"]

        # Check that documentation updates are mentioned
        assert "README.md" in content
        assert "docs/API.md" in content
        assert "Documentation Updates" in content


def test_cmd_update_wiki_breaking_changes():
    """Test cmd_update_wiki with breaking changes."""
    with patch("factory.commands.update_wiki.GitLabClient") as mock_gl_client, \
         patch("factory.commands.update_wiki.Console") as mock_console:

        # Setup mock GitLab client
        mock_gl = Mock()
        mock_gl.connect.return_value = (True, "Connected")
        mock_gl.project = Mock()
        mock_gl.project.mergerequests.get.return_value = Mock(
            title="Breaking changes MR",
            author={"name": "Test User"},
            source_branch="test-branch",
            target_branch="main",
            state="opened",
            changes=lambda: {
                "changes": [
                    {
                        "new_path": "src/api.py",
                        "old_path": "src/api.py",
                        "diff": "BREAKING CHANGE: Removed deprecated endpoint"
                    },
                    {
                        "new_path": "src/config.py",
                        "old_path": "src/config.py",
                        "diff": "Some normal change"
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
        content = args[0]["content"]

        # Check that breaking changes are mentioned
        assert "Breaking Changes" in content
        assert "src/api.py" in content
        # The actual diff content is not included in the wiki, just the file path


def test_cmd_update_wiki_closed_mr():
    """Test cmd_update_wiki with a closed merge request."""
    with patch("factory.commands.update_wiki.GitLabClient") as mock_gl_client, \
         patch("factory.commands.update_wiki.Console") as mock_console:

        # Setup mock GitLab client
        mock_gl = Mock()
        mock_gl.connect.return_value = (True, "Connected")
        mock_gl.project = Mock()
        mock_gl.project.mergerequests.get.return_value = Mock(
            title="Closed MR",
            author={"name": "Test User"},
            source_branch="test-branch",
            target_branch="main",
            state="closed",
            changes=lambda: {
                "changes": [
                    {
                        "new_path": "src/file1.py",
                        "old_path": "src/file1.py",
                        "diff": "Some changes"
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

        # Verify wiki creation was called (should still work for closed MRs)
        mock_gl.project.wikis.create.assert_called_once()
        args, _ = mock_gl.project.wikis.create.call_args
        assert "closed" in args[0]["content"].lower()


def test_cmd_update_wiki_connection_failure():
    """Test cmd_update_wiki when GitLab connection fails."""
    with patch("factory.commands.update_wiki.GitLabClient") as mock_gl_client, \
         patch("factory.commands.update_wiki.Console") as mock_console:

        # Setup mock GitLab client to fail connection
        mock_gl = Mock()
        mock_gl.connect.return_value = (False, "Connection failed")
        mock_gl_client.return_value = mock_gl

        # Run the command - should raise an exception due to connection failure
        with pytest.raises(Exception) as exc_info:
            cmd_update_wiki(mr_id=1, config="config/factory.yml")
        assert str(exc_info.value) == "1"


def test_cmd_update_wiki_existing_wiki_with_content():
    """Test updating an existing wiki page that already has content."""
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
                        "diff": "Updated content"
                    }
                ]
            }
        )

        mock_wiki = Mock()
        mock_wiki.slug = "MR-1-Update"
        mock_wiki.content = "Existing wiki content"
        mock_gl.project.wikis.list.return_value = [mock_wiki]
        mock_gl.project.wikis.get.return_value = mock_wiki
        mock_gl._url = "https://gitlab.example.com"
        mock_gl.project_path = "test/project"
        mock_gl_client.return_value = mock_gl

        # Run the command
        cmd_update_wiki(mr_id=1, config="config/factory.yml")

        # Verify wiki update was called
        mock_wiki.save.assert_called_once()
        # The content should be updated (implementation specific)
        assert mock_wiki.content != "Existing wiki content"  # Should be updated


class TestAnalyzeMrChangesObjectType:
    """Tests for analyze_mr_changes handling python-gitlab 3.x+ object returns."""

    def test_changes_with_attrs_object(self):
        """Test analyze_mr_changes with a python-gitlab 3.x+ object that has _attrs."""
        # Simulate python-gitlab 3.x+ object: changes() returns an object with _attrs
        mock_changes_obj = Mock()
        # Remove dict-like 'get' so it doesn't match the dict path
        del mock_changes_obj.get
        mock_changes_obj._attrs = {
            "changes": [
                {
                    "new_path": "src/main.py",
                    "old_path": "src/main.py",
                    "diff": "refactor: update main entry point",
                },
                {
                    "new_path": "docs/guide.md",
                    "old_path": "docs/guide.md",
                    "diff": "Updated installation guide",
                },
                {
                    "new_path": "src/api.py",
                    "old_path": "src/api.py",
                    "diff": "BREAKING CHANGE: removed deprecated endpoint",
                },
            ]
        }

        mock_mr = Mock()
        mock_mr.changes.return_value = mock_changes_obj

        changes_summary, files_changed, documentation_updates, breaking_changes = analyze_mr_changes(mock_mr)

        assert "3 file(s) changed" in changes_summary
        assert "src/main.py" in files_changed
        assert "docs/guide.md" in files_changed
        assert "src/api.py" in files_changed
        assert "docs/guide.md" in documentation_updates
        assert "src/api.py" in breaking_changes

    def test_changes_with_attrs_object_no_changes_key(self):
        """Test analyze_mr_changes with _attrs object that has no 'changes' key."""
        mock_changes_obj = Mock()
        del mock_changes_obj.get
        mock_changes_obj._attrs = {}

        mock_mr = Mock()
        mock_mr.changes.return_value = mock_changes_obj

        changes_summary, files_changed, documentation_updates, breaking_changes = analyze_mr_changes(mock_mr)

        assert "0 file(s) changed" in changes_summary
        assert files_changed == "_None_"
        assert documentation_updates == "_None_"
        assert breaking_changes == "_None_"

    def test_changes_with_attrs_object_empty_changes_list(self):
        """Test analyze_mr_changes with _attrs object that has an empty changes list."""
        mock_changes_obj = Mock()
        del mock_changes_obj.get
        mock_changes_obj._attrs = {"changes": []}

        mock_mr = Mock()
        mock_mr.changes.return_value = mock_changes_obj

        changes_summary, files_changed, documentation_updates, breaking_changes = analyze_mr_changes(mock_mr)

        assert "0 file(s) changed" in changes_summary
        assert files_changed == "_None_"
        assert documentation_updates == "_None_"
        assert breaking_changes == "_None_"

    def test_changes_with_object_fallback_to_dict(self):
        """Test analyze_mr_changes with an object that has neither get nor _attrs but can be cast to dict."""
        # Simulate an object that falls through to the dict() conversion path.
        # dict() uses the iterable protocol when keys() is not present.
        class DictLikeObject:
            """An object that can be converted to dict via dict()."""
            def __init__(self, data):
                self._data = data

            def __iter__(self):
                return iter(self._data.items())

        changes_data = {
            "changes": [
                {
                    "new_path": "src/utils.py",
                    "old_path": "src/utils.py",
                    "diff": "Added helper function",
                }
            ]
        }
        mock_changes_obj = DictLikeObject(changes_data)

        mock_mr = Mock()
        mock_mr.changes.return_value = mock_changes_obj

        changes_summary, files_changed, documentation_updates, breaking_changes = analyze_mr_changes(mock_mr)

        assert "1 file(s) changed" in changes_summary
        assert "src/utils.py" in files_changed

    def test_changes_with_unconvertible_object(self):
        """Test analyze_mr_changes with an object that cannot be converted to dict, triggering fallback to empty dict."""
        # An object with no get, no _attrs, and dict() raises TypeError
        class UnconvertibleObject:
            pass

        mock_changes_obj = UnconvertibleObject()
        # This object has no 'get', no '_attrs', and dict() will fail

        mock_mr = Mock()
        mock_mr.changes.return_value = mock_changes_obj

        changes_summary, files_changed, documentation_updates, breaking_changes = analyze_mr_changes(mock_mr)

        # Falls back to empty dict, so no changes detected
        assert "0 file(s) changed" in changes_summary
        assert files_changed == "_None_"
        assert documentation_updates == "_None_"
        assert breaking_changes == "_None_"

    def test_changes_with_attrs_object_only_markdown_docs(self):
        """Test that only markdown/rst/txt files are detected as documentation updates."""
        mock_changes_obj = Mock()
        del mock_changes_obj.get
        mock_changes_obj._attrs = {
            "changes": [
                {
                    "new_path": "src/app.py",
                    "old_path": "src/app.py",
                    "diff": "Fixed bug",
                },
                {
                    "new_path": "README.md",
                    "old_path": "README.md",
                    "diff": "Updated readme",
                },
                {
                    "new_path": "docs/api.rst",
                    "old_path": "docs/api.rst",
                    "diff": "Updated API docs",
                },
                {
                    "new_path": "NOTES.txt",
                    "old_path": "NOTES.txt",
                    "diff": "Added notes",
                },
            ]
        }

        mock_mr = Mock()
        mock_mr.changes.return_value = mock_changes_obj

        changes_summary, files_changed, documentation_updates, breaking_changes = analyze_mr_changes(mock_mr)

        assert "4 file(s) changed" in changes_summary
        assert "README.md" in documentation_updates
        assert "docs/api.rst" in documentation_updates
        assert "NOTES.txt" in documentation_updates
        assert "src/app.py" not in documentation_updates

    def test_changes_with_attrs_object_breaking_change_case_insensitive(self):
        """Test that breaking changes are detected regardless of case."""
        mock_changes_obj = Mock()
        del mock_changes_obj.get
        mock_changes_obj._attrs = {
            "changes": [
                {
                    "new_path": "src/low.py",
                    "old_path": "src/low.py",
                    "diff": "breaking: removed old api",
                },
                {
                    "new_path": "src/upper.py",
                    "old_path": "src/upper.py",
                    "diff": "BREAKING CHANGE: removed endpoint",
                },
            ]
        }

        mock_mr = Mock()
        mock_mr.changes.return_value = mock_changes_obj

        changes_summary, files_changed, documentation_updates, breaking_changes = analyze_mr_changes(mock_mr)

        assert "src/low.py" in breaking_changes
        assert "src/upper.py" in breaking_changes