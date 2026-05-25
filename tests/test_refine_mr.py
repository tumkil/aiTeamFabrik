"""Unit tests for factory.commands.refine module."""
import pytest
from unittest.mock import MagicMock, patch
from factory.commands.refine import (
    cmd_refine,
    run_mr_refine,
    _get_mr_issue_iid,
    _extract_issue_iids,
    _is_approval_note,
)


# ---------------------------------------------------------------------------
# _is_approval_note
# ---------------------------------------------------------------------------

class TestIsApprovalNote:
    """Tests for the _is_approval_note helper."""

    # --- Positive cases (should return True) ---

    def test_simple_approve(self):
        assert _is_approval_note("APPROVE") is True

    def test_simple_approved(self):
        assert _is_approval_note("APPROVED") is True

    def test_approve_lowercase(self):
        assert _is_approval_note("approve") is True

    def test_approved_mixed_case(self):
        assert _is_approval_note("Approved") is True

    def test_approve_in_sentence(self):
        assert _is_approval_note("I APPROVE this change") is True

    def test_checkmark_approve(self):
        assert _is_approval_note("✅ APPROVE") is True

    def test_verdict_heading_with_checkmark(self):
        assert _is_approval_note("## Verdict\n✅ Looks good") is True

    def test_verdict_heading_approve(self):
        assert _is_approval_note("## Verdict\n✅ APPROVE\nAll good") is True

    # --- Negative cases (should return False) ---

    def test_not_approve(self):
        assert _is_approval_note("I do NOT APPROVE") is False

    def test_not_approved(self):
        assert _is_approval_note("NOT APPROVED") is False

    def test_dont_approve(self):
        assert _is_approval_note("I DON'T APPROVE") is False

    def test_dont_approved(self):
        assert _is_approval_note("DON'T APPROVED") is False

    def test_do_not_approve(self):
        assert _is_approval_note("DO NOT APPROVE") is False

    def test_do_not_approved(self):
        assert _is_approval_note("DO NOT APPROVED") is False

    def test_never_approve(self):
        assert _is_approval_note("NEVER APPROVE") is False

    def test_cannot_approve(self):
        assert _is_approval_note("CANNOT APPROVE") is False

    def test_cant_approve(self):
        assert _is_approval_note("CAN'T APPROVE") is False

    def test_wont_approve(self):
        assert _is_approval_note("WON'T APPROVE") is False

    def test_unable_to_approve(self):
        assert _is_approval_note("UNABLE TO APPROVE") is False

    def test_refuse_to_approve(self):
        assert _is_approval_note("REFUSE TO APPROVE") is False

    def test_never_approved(self):
        assert _is_approval_note("NEVER APPROVED") is False

    def test_cannot_approved(self):
        assert _is_approval_note("CANNOT APPROVED") is False

    def test_cant_approved(self):
        assert _is_approval_note("CAN'T APPROVED") is False

    def test_wont_approved(self):
        assert _is_approval_note("WON'T APPROVED") is False

    def test_unable_to_approved(self):
        assert _is_approval_note("UNABLE TO APPROVED") is False

    def test_dont_approved_no_apostrophe(self):
        assert _is_approval_note("DONT APPROVED") is False

    def test_cant_approve_no_apostrophe(self):
        assert _is_approval_note("CANT APPROVE") is False

    # --- Empty / no relevant keyword ---

    def test_empty_string(self):
        assert _is_approval_note("") is False

    def test_none_like_empty(self):
        assert _is_approval_note("") is False

    def test_unrelated_note(self):
        assert _is_approval_note("Looks fine to me") is False

    def test_request_changes(self):
        assert _is_approval_note("REQUEST CHANGES") is False

    # --- Edge cases ---

    def test_negation_with_positive_in_same_note(self):
        """A note that says both NOT APPROVE and APPROVE should be rejected."""
        assert _is_approval_note("I do NOT APPROVE this, but someone else might APPROVE") is False

    def test_approve_word_boundary(self):
        """'APPROVE' must be a word, not a substring like in 'DISAPPROVE'."""
        # DISAPPROVE contains APPROVE but the \b word boundary should not match
        # inside it. However, \bAPPROVE\b won't match DISAPPROVE because
        # there's no word boundary between DIS and APPROVE… actually there IS
        # a word boundary inside DIS-APPROVE if there's a hyphen. Let's verify
        # the simple case.
        # "DISAPPROVE" should NOT match as an approval.
        assert _is_approval_note("DISAPPROVE") is False


# ---------------------------------------------------------------------------
# Original tests (unchanged)
# ---------------------------------------------------------------------------

def test_extract_issue_iids():
    """Test extracting issue IIDs from text."""
    assert _extract_issue_iids("Fix #1 and #2") == [1, 2]
    assert _extract_issue_iids("Issue #123") == [123]
    assert _extract_issue_iids("No issues here") == []
    assert _extract_issue_iids("Multiple #1, #2, #3") == [1, 2, 3]


def test_get_mr_issue_iid():
    """Test extracting issue IID from MR title and description."""
    mr = MagicMock()
    mr.title = "Fix #456 issue"
    mr.description = "This addresses issue #789"
    assert _get_mr_issue_iid(mr) == 456

    mr.title = "No issue mentioned"
    mr.description = "Just a regular MR"
    assert _get_mr_issue_iid(mr) is None


@patch("factory.commands.refine.GitLabClient")
@patch("factory.commands.refine.CompetenceManager")
@patch("factory.commands.refine.LlmRouter")
@patch("factory.commands.refine.CodeExecutionEngine")
@patch("factory.commands.refine.console")
def test_cmd_refine_basic(mock_console, mock_engine_class, mock_router_class, mock_cm_class, mock_gl_class):
    """Test basic refine command execution."""
    # Setup mocks
    mock_gl = MagicMock()
    mock_gl.connect.return_value = (True, "connected")
    mock_gl._url = "https://gitlab.example.com"
    
    project = MagicMock()
    project.id = 123
    project.http_url_to_repo = "https://gitlab.example.com/test/project.git"
    mock_gl.project = project
    
    mr = MagicMock()
    mr.iid = 1
    mr.title = "Test MR"
    mr.source_branch = "test-branch"
    mr.description = "Test description"
    mr.work_in_progress = False
    mr.web_url = "https://gitlab.example.com/test/project/-/merge_requests/1"
    mr.commits.return_value = []
    mr.notes.list.return_value = []
    mr.changes.return_value = {"changes": []}
    
    project.mergerequests.get.return_value = mr
    mock_gl_class.return_value = mock_gl
    
    # Setup competence manager
    mock_cm = MagicMock()
    developer = MagicMock()
    developer.name = "developer"
    developer.display_name = "Developer"
    developer.model = "claude-sonnet"
    developer.provider = "anthropic"
    developer.execution_mode = "execute"
    mock_cm.get.return_value = developer
    mock_cm.agents = [developer]
    mock_cm_class.return_value = mock_cm
    
    # Setup router
    mock_router = MagicMock()
    mock_router_class.return_value = mock_router
    
    # Setup execution engine
    mock_engine = MagicMock()
    result = MagicMock()
    result.needs_continuation = False
    result.iterations = 1
    result.files_changed = ["file1.py", "file2.py"]
    result.commit_sha = "abc123def456"
    result.response = "Refinement complete"
    mock_engine.run.return_value = result
    mock_engine_class.return_value = mock_engine
    
    # Run command
    cmd_refine(mr=1, config="config/factory.yml", agents_dir="config/agents", provider="", auto_merge=False)
    
    # Assertions
    mock_gl.connect.assert_called_once()
    mock_cm.get.assert_called_with("developer")
    mock_engine_class.assert_called_once()
    mock_engine.run.assert_called_once()


@patch("factory.commands.refine.GitLabClient")
@patch("factory.commands.refine.CompetenceManager")
@patch("factory.commands.refine.LlmRouter")
@patch("factory.commands.refine.console")
def test_cmd_refine_stub_provider(mock_console, mock_router_class, mock_cm_class, mock_gl_class):
    """Test refine command with stub provider."""
    # Setup mocks
    mock_gl = MagicMock()
    mock_gl.connect.return_value = (True, "connected")
    mock_gl._url = "https://gitlab.example.com"
    
    project = MagicMock()
    project.id = 123
    project.http_url_to_repo = "https://gitlab.example.com/test/project.git"
    mock_gl.project = project
    
    mr = MagicMock()
    mr.iid = 1
    mr.title = "Test MR"
    mr.source_branch = "test-branch"
    mr.description = "Test description"
    mr.work_in_progress = False
    mr.web_url = "https://gitlab.example.com/test/project/-/merge_requests/1"
    mr.commits.return_value = []
    mr.notes.list.return_value = []
    mr.changes.return_value = {"changes": []}
    
    project.mergerequests.get.return_value = mr
    mock_gl_class.return_value = mock_gl
    
    # Setup competence manager
    mock_cm = MagicMock()
    developer = MagicMock()
    developer.name = "developer"
    developer.display_name = "Developer"
    developer.model = "claude-sonnet"
    developer.provider = "stub"
    developer.execution_mode = "plan"
    mock_cm.get.return_value = developer
    mock_cm.agents = [developer]
    mock_cm_class.return_value = mock_cm
    
    # Setup router
    mock_router = MagicMock()
    response = MagicMock()
    response.content = "Refinement analysis complete"
    mock_router.complete.return_value = response
    mock_router_class.return_value = mock_router
    
    # Run command
    cmd_refine(mr=1, config="config/factory.yml", agents_dir="config/agents", provider="stub", auto_merge=False)
    
    # Assertions
    mock_router.complete.assert_called_once()


@patch("factory.commands.refine.GitLabClient")
@patch("factory.commands.refine.CompetenceManager")
@patch("factory.commands.refine.LlmRouter")
@patch("factory.commands.refine.CodeExecutionEngine")
@patch("factory.commands.refine.console")
def test_cmd_refine_with_issue(mock_console, mock_engine_class, mock_router_class, mock_cm_class, mock_gl_class):
    """Test refine command with associated issue."""
    # Setup mocks
    mock_gl = MagicMock()
    mock_gl.connect.return_value = (True, "connected")
    mock_gl._url = "https://gitlab.example.com"
    
    project = MagicMock()
    project.id = 123
    project.http_url_to_repo = "https://gitlab.example.com/test/project.git"
    mock_gl.project = project
    
    issue = MagicMock()
    issue.iid = 456
    issue.title = "Test Issue"
    
    mr = MagicMock()
    mr.iid = 1
    mr.title = "Fix #456"
    mr.source_branch = "test-branch"
    mr.description = "Test description"
    mr.work_in_progress = False
    mr.web_url = "https://gitlab.example.com/test/project/-/merge_requests/1"
    mr.commits.return_value = []
    mr.notes.list.return_value = []
    mr.changes.return_value = {"changes": []}
    
    project.mergerequests.get.return_value = mr
    project.issues.get.return_value = issue
    mock_gl_class.return_value = mock_gl
    
    # Setup competence manager
    mock_cm = MagicMock()
    developer = MagicMock()
    developer.name = "developer"
    developer.display_name = "Developer"
    developer.model = "claude-sonnet"
    developer.provider = "anthropic"
    developer.execution_mode = "execute"
    mock_cm.get.return_value = developer
    mock_cm.agents = [developer]
    mock_cm_class.return_value = mock_cm
    
    # Setup router
    mock_router = MagicMock()
    mock_router_class.return_value = mock_router
    
    # Setup execution engine
    mock_engine = MagicMock()
    result = MagicMock()
    result.needs_continuation = False
    result.iterations = 1
    result.files_changed = ["file1.py"]
    result.commit_sha = "abc123def456"
    result.response = "Refinement complete"
    mock_engine.run.return_value = result
    mock_engine_class.return_value = mock_engine
    
    # Run command
    cmd_refine(mr=1, config="config/factory.yml", agents_dir="config/agents", provider="", auto_merge=False)
    
    # Assertions
    project.issues.get.assert_called_with(456)
    mock_engine_class.assert_called_once()


@patch("factory.commands.refine.GitLabClient")
@patch("factory.commands.refine.CompetenceManager")
@patch("factory.commands.refine.LlmRouter")
@patch("factory.commands.refine.CodeExecutionEngine")
@patch("factory.commands.refine.console")
def test_cmd_refine_with_review_notes(mock_console, mock_engine_class, mock_router_class, mock_cm_class, mock_gl_class):
    """Test refine command with review notes."""
    # Setup mocks
    mock_gl = MagicMock()
    mock_gl.connect.return_value = (True, "connected")
    mock_gl._url = "https://gitlab.example.com"
    
    project = MagicMock()
    project.id = 123
    project.http_url_to_repo = "https://gitlab.example.com/test/project.git"
    mock_gl.project = project
    
    mr = MagicMock()
    mr.iid = 1
    mr.title = "Test MR"
    mr.source_branch = "test-branch"
    mr.description = "Test description"
    mr.work_in_progress = False
    mr.web_url = "https://gitlab.example.com/test/project/-/merge_requests/1"
    mr.commits.return_value = []
    
    # Setup review notes
    note1 = MagicMock()
    note1.body = "## Verdict\n⚠️ REQUEST CHANGES\nPlease fix the bug in file1.py"
    note1.author = {"name": "Code Reviewer"}
    note1.created_at = "2023-01-01T00:00:00Z"
    
    note2 = MagicMock()
    note2.body = "Human review: needs improvement"
    note2.author = {"name": "Human Reviewer"}
    note2.created_at = "2023-01-02T00:00:00Z"
    
    mr.notes.list.return_value = [note1, note2]
    mr.changes.return_value = {"changes": []}
    
    project.mergerequests.get.return_value = mr
    mock_gl_class.return_value = mock_gl
    
    # Setup competence manager
    mock_cm = MagicMock()
    developer = MagicMock()
    developer.name = "developer"
    developer.display_name = "Developer"
    developer.model = "claude-sonnet"
    developer.provider = "anthropic"
    developer.execution_mode = "execute"
    mock_cm.get.return_value = developer
    mock_cm.agents = [developer]
    mock_cm_class.return_value = mock_cm
    
    # Setup router
    mock_router = MagicMock()
    mock_router_class.return_value = mock_router
    
    # Setup execution engine
    mock_engine = MagicMock()
    result = MagicMock()
    result.needs_continuation = False
    result.iterations = 1
    result.files_changed = ["file1.py"]
    result.commit_sha = "abc123def456"
    result.response = "Refinement complete"
    mock_engine.run.return_value = result
    mock_engine_class.return_value = mock_engine
    
    # Run command
    cmd_refine(mr=1, config="config/factory.yml", agents_dir="config/agents", provider="", auto_merge=False)
    
    # Assertions
    mock_engine.run.assert_called_once()
    # Check that the engine was called with the correct description containing review feedback
    call_args = mock_engine_class.call_args
    assert "Review Feedback to Address" in call_args[1]["issue_description"]


@patch("factory.commands.refine.GitLabClient")
@patch("factory.commands.refine.CompetenceManager")
@patch("factory.commands.refine.LlmRouter")
@patch("factory.commands.refine.CodeExecutionEngine")
@patch("factory.commands.refine.console")
def test_cmd_refine_continuation(mock_console, mock_engine_class, mock_router_class, mock_cm_class, mock_gl_class):
    """Test refine command when iteration limit is hit."""
    # Setup mocks
    mock_gl = MagicMock()
    mock_gl.connect.return_value = (True, "connected")
    mock_gl._url = "https://gitlab.example.com"
    
    project = MagicMock()
    project.id = 123
    project.http_url_to_repo = "https://gitlab.example.com/test/project.git"
    mock_gl.project = project
    
    mr = MagicMock()
    mr.iid = 1
    mr.title = "Test MR"
    mr.source_branch = "test-branch"
    mr.description = "Test description"
    mr.work_in_progress = False
    mr.web_url = "https://gitlab.example.com/test/project/-/merge_requests/1"
    mr.commits.return_value = []
    mr.notes.list.return_value = []
    mr.changes.return_value = {"changes": []}
    
    project.mergerequests.get.return_value = mr
    mock_gl_class.return_value = mock_gl
    
    # Setup competence manager
    mock_cm = MagicMock()
    developer = MagicMock()
    developer.name = "developer"
    developer.display_name = "Developer"
    developer.model = "claude-sonnet"
    developer.provider = "anthropic"
    developer.execution_mode = "execute"
    mock_cm.get.return_value = developer
    mock_cm.agents = [developer]
    mock_cm_class.return_value = mock_cm
    
    # Setup router
    mock_router = MagicMock()
    mock_router_class.return_value = mock_router
    
    # Setup execution engine with continuation
    mock_engine = MagicMock()
    result = MagicMock()
    result.needs_continuation = True
    result.iterations = 3
    result.files_changed = []
    result.commit_sha = None
    result.response = "Iteration limit hit"
    mock_engine.run.return_value = result
    mock_engine_class.return_value = mock_engine
    
    # Run command
    cmd_refine(mr=1, config="config/factory.yml", agents_dir="config/agents", provider="", auto_merge=False)
    
    # Assertions
    mock_engine.run.assert_called_once()


@patch("factory.commands.refine.GitLabClient")
@patch("factory.commands.refine.CompetenceManager")
@patch("factory.commands.refine.LlmRouter")
@patch("factory.commands.refine.CodeExecutionEngine")
@patch("factory.commands.refine.console")
def test_cmd_refine_auto_merge(mock_console, mock_engine_class, mock_router_class, mock_cm_class, mock_gl_class):
    """Test refine command with auto-merge when all reviews are approved."""
    # Setup mocks
    mock_gl = MagicMock()
    mock_gl.connect.return_value = (True, "connected")
    mock_gl._url = "https://gitlab.example.com"
    
    project = MagicMock()
    project.id = 123
    project.http_url_to_repo = "https://gitlab.example.com/test/project.git"
    mock_gl.project = project
    
    mr = MagicMock()
    mr.iid = 1
    mr.title = "Test MR"
    mr.source_branch = "test-branch"
    mr.description = "Test description"
    mr.work_in_progress = False
    mr.web_url = "https://gitlab.example.com/test/project/-/merge_requests/1"
    mr.commits.return_value = []
    
    # Setup approved review notes
    note1 = MagicMock()
    note1.body = "✅ APPROVE"
    note1.author = {"name": "Code Reviewer"}
    note1.created_at = "2023-01-01T00:00:00Z"
    
    note2 = MagicMock()
    note2.body = "Approved"
    note2.author = {"name": "Human Reviewer"}
    note2.created_at = "2023-01-02T00:00:00Z"
    
    mr.notes.list.return_value = [note1, note2]
    mr.changes.return_value = {"changes": []}
    
    project.mergerequests.get.return_value = mr
    mock_gl_class.return_value = mock_gl
    
    # Setup competence manager
    mock_cm = MagicMock()
    developer = MagicMock()
    developer.name = "developer"
    developer.display_name = "Developer"
    developer.model = "claude-sonnet"
    developer.provider = "anthropic"
    developer.execution_mode = "execute"
    mock_cm.get.return_value = developer
    mock_cm.agents = [developer]
    mock_cm_class.return_value = mock_cm
    
    # Setup router
    mock_router = MagicMock()
    mock_router_class.return_value = mock_router
    
    # Setup execution engine
    mock_engine = MagicMock()
    result = MagicMock()
    result.needs_continuation = False
    result.iterations = 1
    result.files_changed = ["file1.py"]
    result.commit_sha = "abc123def456"
    result.response = "Refinement complete"
    mock_engine.run.return_value = result
    mock_engine_class.return_value = mock_engine
    
    # Run command with auto_merge
    cmd_refine(mr=1, config="config/factory.yml", agents_dir="config/agents", provider="", auto_merge=True)
    
    # Assertions
    mr.merge.assert_called_once_with(remove_source_branch=True)


@patch("factory.commands.refine.GitLabClient")
@patch("factory.commands.refine.CompetenceManager")
@patch("factory.commands.refine.LlmRouter")
@patch("factory.commands.refine.CodeExecutionEngine")
@patch("factory.commands.refine.console")
def test_cmd_refine_auto_merge_not_approve(mock_console, mock_engine_class, mock_router_class, mock_cm_class, mock_gl_class):
    """Test auto-merge does NOT trigger when a note says 'I do NOT APPROVE'."""
    mock_gl = MagicMock()
    mock_gl.connect.return_value = (True, "connected")
    mock_gl._url = "https://gitlab.example.com"

    project = MagicMock()
    project.id = 123
    project.http_url_to_repo = "https://gitlab.example.com/test/project.git"
    mock_gl.project = project

    mr = MagicMock()
    mr.iid = 1
    mr.title = "Test MR"
    mr.source_branch = "test-branch"
    mr.description = "Test description"
    mr.work_in_progress = False
    mr.web_url = "https://gitlab.example.com/test/project/-/merge_requests/1"
    mr.commits.return_value = []

    # One note says "I do NOT APPROVE" — this must block auto-merge
    note1 = MagicMock()
    note1.body = "I do NOT APPROVE this change"
    note1.author = {"name": "Skeptical Reviewer"}
    note1.created_at = "2023-01-01T00:00:00Z"

    mr.notes.list.return_value = [note1]
    mr.changes.return_value = {"changes": []}

    project.mergerequests.get.return_value = mr
    mock_gl_class.return_value = mock_gl

    mock_cm = MagicMock()
    developer = MagicMock()
    developer.name = "developer"
    developer.display_name = "Developer"
    developer.model = "claude-sonnet"
    developer.provider = "anthropic"
    developer.execution_mode = "execute"
    mock_cm.get.return_value = developer
    mock_cm.agents = [developer]
    mock_cm_class.return_value = mock_cm

    mock_router = MagicMock()
    mock_router_class.return_value = mock_router

    mock_engine = MagicMock()
    result = MagicMock()
    result.needs_continuation = False
    result.iterations = 1
    result.files_changed = ["file1.py"]
    result.commit_sha = "abc123def456"
    result.response = "Refinement complete"
    mock_engine.run.return_value = result
    mock_engine_class.return_value = mock_engine

    cmd_refine(mr=1, config="config/factory.yml", agents_dir="config/agents", provider="", auto_merge=True)

    # merge must NOT have been called
    mr.merge.assert_not_called()


@patch("factory.commands.refine.GitLabClient")
@patch("factory.commands.refine.CompetenceManager")
@patch("factory.commands.refine.LlmRouter")
@patch("factory.commands.refine.CodeExecutionEngine")
@patch("factory.commands.refine.console")
def test_cmd_refine_auto_merge_no_approval_notes(mock_console, mock_engine_class, mock_router_class, mock_cm_class, mock_gl_class):
    """Test auto-merge does NOT trigger when there are no approval notes at all."""
    mock_gl = MagicMock()
    mock_gl.connect.return_value = (True, "connected")
    mock_gl._url = "https://gitlab.example.com"

    project = MagicMock()
    project.id = 123
    project.http_url_to_repo = "https://gitlab.example.com/test/project.git"
    mock_gl.project = project

    mr = MagicMock()
    mr.iid = 1
    mr.title = "Test MR"
    mr.source_branch = "test-branch"
    mr.description = "Test description"
    mr.work_in_progress = False
    mr.web_url = "https://gitlab.example.com/test/project/-/merge_requests/1"
    mr.commits.return_value = []

    # A neutral comment that is NOT an approval
    note1 = MagicMock()
    note1.body = "Just a comment, no verdict"
    note1.author = {"name": "Reviewer"}
    note1.created_at = "2023-01-01T00:00:00Z"

    mr.notes.list.return_value = [note1]
    mr.changes.return_value = {"changes": []}

    project.mergerequests.get.return_value = mr
    mock_gl_class.return_value = mock_gl

    mock_cm = MagicMock()
    developer = MagicMock()
    developer.name = "developer"
    developer.display_name = "Developer"
    developer.model = "claude-sonnet"
    developer.provider = "anthropic"
    developer.execution_mode = "execute"
    mock_cm.get.return_value = developer
    mock_cm.agents = [developer]
    mock_cm_class.return_value = mock_cm

    mock_router = MagicMock()
    mock_router_class.return_value = mock_router

    mock_engine = MagicMock()
    result = MagicMock()
    result.needs_continuation = False
    result.iterations = 1
    result.files_changed = ["file1.py"]
    result.commit_sha = "abc123def456"
    result.response = "Refinement complete"
    mock_engine.run.return_value = result
    mock_engine_class.return_value = mock_engine

    cmd_refine(mr=1, config="config/factory.yml", agents_dir="config/agents", provider="", auto_merge=True)

    # merge must NOT have been called
    mr.merge.assert_not_called()


@patch("factory.commands.refine.GitLabClient")
@patch("factory.commands.refine.CompetenceManager")
@patch("factory.commands.refine.LlmRouter")
@patch("factory.commands.refine.CodeExecutionEngine")
@patch("factory.commands.refine.console")
def test_cmd_refine_wip_removal(mock_console, mock_engine_class, mock_router_class, mock_cm_class, mock_gl_class):
    """Test refine command removes WIP status when refinement is complete."""
    # Setup mocks
    mock_gl = MagicMock()
    mock_gl.connect.return_value = (True, "connected")
    mock_gl._url = "https://gitlab.example.com"
    
    project = MagicMock()
    project.id = 123
    project.http_url_to_repo = "https://gitlab.example.com/test/project.git"
    mock_gl.project = project
    
    mr = MagicMock()
    mr.iid = 1
    mr.title = "WIP: Test MR"
    mr.source_branch = "test-branch"
    mr.description = "Test description"
    mr.work_in_progress = True
    mr.web_url = "https://gitlab.example.com/test/project/-/merge_requests/1"
    mr.commits.return_value = []
    mr.notes.list.return_value = []
    mr.changes.return_value = {"changes": []}
    
    project.mergerequests.get.return_value = mr
    mock_gl_class.return_value = mock_gl
    
    # Setup competence manager
    mock_cm = MagicMock()
    developer = MagicMock()
    developer.name = "developer"
    developer.display_name = "Developer"
    developer.model = "claude-sonnet"
    developer.provider = "anthropic"
    developer.execution_mode = "execute"
    mock_cm.get.return_value = developer
    mock_cm.agents = [developer]
    mock_cm_class.return_value = mock_cm
    
    # Setup router
    mock_router = MagicMock()
    mock_router_class.return_value = mock_router
    
    # Setup execution engine
    mock_engine = MagicMock()
    result = MagicMock()
    result.needs_continuation = False
    result.iterations = 1
    result.files_changed = ["file1.py"]
    result.commit_sha = "abc123def456"
    result.response = "Refinement complete"
    mock_engine.run.return_value = result
    mock_engine_class.return_value = mock_engine
    
    # Run command
    cmd_refine(mr=1, config="config/factory.yml", agents_dir="config/agents", provider="", auto_merge=False)
    
    # Assertions
    assert mr.work_in_progress == False


@patch("factory.commands.refine.GitLabClient")
@patch("factory.commands.refine.CompetenceManager")
@patch("factory.commands.refine.LlmRouter")
@patch("factory.commands.refine.CodeExecutionEngine")
def test_run_mr_refine(mock_engine_class, mock_router_class, mock_cm_class, mock_gl_class):
    """Test run_mr_refine function directly."""
    # Setup mocks
    mock_gl = MagicMock()
    mock_gl.connect.return_value = (True, "connected")
    mock_gl._url = "https://gitlab.example.com"
    
    project = MagicMock()
    project.id = 123
    project.http_url_to_repo = "https://gitlab.example.com/test/project.git"
    mock_gl.project = project
    
    mr = MagicMock()
    mr.iid = 1
    mr.title = "Test MR"
    mr.source_branch = "test-branch"
    mr.description = "Test description"
    mr.work_in_progress = False
    mr.commits.return_value = []
    mr.notes.list.return_value = []
    mr.changes.return_value = {"changes": []}
    
    # Setup competence manager
    mock_cm = MagicMock()
    developer = MagicMock()
    developer.name = "developer"
    developer.display_name = "Developer"
    developer.model = "claude-sonnet"
    developer.provider = "anthropic"
    developer.execution_mode = "execute"
    mock_cm.get.return_value = developer
    mock_cm.agents = [developer]
    mock_cm_class.return_value = mock_cm
    
    # Setup router
    mock_router = MagicMock()
    mock_router_class.return_value = mock_router
    
    # Setup execution engine
    mock_engine = MagicMock()
    result = MagicMock()
    result.needs_continuation = False
    result.iterations = 1
    result.files_changed = ["file1.py"]
    result.commit_sha = "abc123def456"
    result.response = "Refinement complete"
    mock_engine.run.return_value = result
    mock_engine_class.return_value = mock_engine
    
    # Run function
    result = run_mr_refine(
        merge_request=mr,
        project=project,
        gl_url="https://gitlab.example.com",
        developer=developer,
        router=mock_router,
        gitlab_token="test-token",
        progress=None,
    )
    
    # Assertions
    assert result.needs_continuation == False
    assert result.iterations == 1
    assert result.files_changed == ["file1.py"]
    assert result.commit_sha == "abc123def456"
    assert result.response == "Refinement complete"