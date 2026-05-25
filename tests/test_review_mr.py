'''Unit tests for factory.commands.review_mr module.'''
import pytest
import typer
from unittest.mock import MagicMock, patch, call
from factory.commands.review_mr import (
    cmd_review_mr,
    _verdict_color,
    _build_comment,
    _merge_verdicts,
    ReviewResult,
    run_mr_review,
)
from factory.core.competence import AgentProfile


def _make_reviewer():
    return AgentProfile(
        name="reviewer",
        display_name="Code Reviewer",
        model="claude-sonnet",
        provider="anthropic",
        system_prompt="",
    )


def _make_meta_reviewer():
    return AgentProfile(
        name="meta_reviewer",
        display_name="Meta Reviewer",
        model="mistral-large",
        provider="mistral",
        system_prompt="",
    )


def _make_gl_mock():
    mock_gl = MagicMock()
    project = MagicMock()
    mr = MagicMock()
    mr.title = "Test MR"
    mr.author = {"name": "Test Author"}
    mr.source_branch = "test-branch"
    mr.target_branch = "main"
    mr.description = "Test description"
    mr.references = {"related_issues": []}
    mr.changes.return_value = {
        "changes": [
            {
                "old_path": "example.py",
                "new_path": "example.py",
                "diff": "@@ -1,3 +1,4 @@\n foo\n+bar\n baz\n",
                "deleted_file": False,
            }
        ]
    }
    project.mergerequests.get.return_value = mr
    mock_gl.connect.return_value = (True, "connected")
    mock_gl.project = project
    mock_gl._url = "https://gitlab.example.com"
    mock_gl.project_path = "test/project"
    return mock_gl, mr


# ---------------------------------------------------------------------------
# ReviewResult dataclass tests
# ---------------------------------------------------------------------------

class TestReviewResult:
    """Tests for the ReviewResult dataclass."""

    def test_review_result_fields(self):
        result = ReviewResult(
            comment_body="review text",
            verdict="green",
            posted=True,
        )
        assert result.comment_body == "review text"
        assert result.verdict == "green"
        assert result.posted is True

    def test_review_result_defaults(self):
        result = ReviewResult(comment_body="review text")
        assert result.verdict == ""
        assert result.posted is False

    def test_review_result_verdict_colors(self):
        for color in ("green", "yellow", "red"):
            result = ReviewResult(comment_body="", verdict=color)
            assert result.verdict == color

    def test_review_result_posted_false(self):
        result = ReviewResult(comment_body="text", verdict="yellow", posted=False)
        assert result.posted is False

    def test_review_result_is_dataclass(self):
        """ReviewResult should be a dataclass with proper attribute access."""
        from dataclasses import is_dataclass
        assert is_dataclass(ReviewResult)

    def test_review_result_equality(self):
        r1 = ReviewResult(comment_body="a", verdict="green", posted=True)
        r2 = ReviewResult(comment_body="a", verdict="green", posted=True)
        assert r1 == r2

    def test_review_result_inequality(self):
        r1 = ReviewResult(comment_body="a", verdict="green", posted=True)
        r2 = ReviewResult(comment_body="b", verdict="green", posted=True)
        assert r1 != r2


# ---------------------------------------------------------------------------
# Verdict color extraction tests
# ---------------------------------------------------------------------------

def test_verdict_color_approve():
    content = "## Verdict\n✅ APPROVE"
    assert _verdict_color(content) == "green"


def test_verdict_color_request_changes():
    content = "## Verdict\n⚠️ REQUEST CHANGES"
    assert _verdict_color(content) == "yellow"


def test_verdict_color_block():
    content = "## Verdict\n🚫 BLOCK"
    assert _verdict_color(content) == "red"


def test_verdict_color_adjusted_verdict():
    """_verdict_color must match ## Adjusted Verdict sections (meta-reviewer output)."""
    content = "## Adjusted Verdict\n⚠️ REQUEST CHANGES"
    assert _verdict_color(content) == "yellow"


def test_verdict_color_no_verdict():
    content = "Some content without a verdict"
    assert _verdict_color(content) is None


def test_verdict_color_ignores_text_in_description():
    content = """## Summary
This could BLOCK deployment if not fixed.
## Verdict
✅ APPROVE"""
    assert _verdict_color(content) == "green"


def test_verdict_color_crlf_line_endings():
    """_verdict_color must parse verdicts in responses with Windows line endings."""
    content = "## Verdict\r\n✅ APPROVE"
    assert _verdict_color(content) == "green"


# ---------------------------------------------------------------------------
# Verdict merging tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("primary,meta,expected", [
    ("red", "green", "yellow"),    # BLOCK→APPROVE capped to REQUEST CHANGES
    ("red", "yellow", "yellow"),   # BLOCK improved by one step
    ("red", "red", "red"),         # BLOCK kept
    ("yellow", "green", "green"),  # REQUEST CHANGES improved by one step
    ("yellow", "yellow", "yellow"),# REQUEST CHANGES kept
    ("green", "yellow", "green"),  # no escalation — primary preserved
    ("green", "red", "green"),     # no escalation — primary preserved
    ("green", "green", "green"),   # APPROVE kept
])
def test_merge_verdicts(primary, meta, expected):
    assert _merge_verdicts(primary, meta) == expected


def test_merge_verdicts_none_meta_keeps_primary():
    """Unparseable meta verdict (None) must not change the primary verdict."""
    assert _merge_verdicts("red", None) == "red"
    assert _merge_verdicts("yellow", None) == "yellow"
    assert _merge_verdicts("green", None) == "green"


def test_merge_verdicts_invalid_primary_raises():
    with pytest.raises(ValueError):
        _merge_verdicts("purple", "green")


def test_merge_verdicts_unparseable_meta_string():
    """Unparseable meta-reviewer verdict string must not change the primary verdict."""
    assert _merge_verdicts("green", "unexpected-verdict-string") == "green"
    assert _merge_verdicts("yellow", "unexpected-verdict-string") == "yellow"
    assert _merge_verdicts("red", "unexpected-verdict-string") == "red"


# ---------------------------------------------------------------------------
# Comment building tests
# ---------------------------------------------------------------------------

def test_build_comment_without_meta():
    primary = MagicMock()
    primary.content = "Primary review content"
    primary.model = "claude-sonnet"
    primary.input_tokens = 100
    primary.output_tokens = 50

    result = _build_comment(primary, None)

    assert "### :mag: Code Review by SoftwareTeamFabrik" in result
    assert "Primary review content" in result
    assert "Model: `claude-sonnet`" in result
    assert "Tokens: 100 in / 50 out" in result
    assert "Meta Review" not in result


def test_build_comment_with_meta():
    primary = MagicMock()
    primary.content = "Primary review content"
    primary.model = "claude-sonnet"
    primary.input_tokens = 100
    primary.output_tokens = 50

    meta = MagicMock()
    meta.content = "Meta review content"
    meta.model = "mistral-large"
    meta.input_tokens = 80
    meta.output_tokens = 40

    result = _build_comment(primary, meta)

    assert "### :mag: Code Review by SoftwareTeamFabrik" in result
    assert "Primary review content" in result
    assert "Model: `claude-sonnet`" in result
    assert "Tokens: 100 in / 50 out" in result
    assert "### :mag::mag: Meta Review (false-positive filter) by SoftwareTeamFabrik" in result
    assert "Meta review content" in result
    assert "Model: `mistral-large`" in result
    assert "Tokens: 80 in / 40 out" in result
    assert "Note: The meta-reviewer can only keep or improve the primary verdict, never escalate it." in result


# ---------------------------------------------------------------------------
# cmd_review_mr CLI tests
# ---------------------------------------------------------------------------

@patch("factory.commands.review_mr.GitLabClient")
@patch("factory.commands.review_mr.CompetenceManager")
@patch("factory.commands.review_mr.LlmRouter")
@patch("factory.commands.review_mr.console")
def test_cmd_review_mr_diff_included_in_prompt(mock_console, mock_router_class, mock_cm_class, mock_gl_class):
    """Diff content from changes() is forwarded to the LLM prompt."""
    mock_gl, _ = _make_gl_mock()
    mock_gl_class.return_value = mock_gl

    mock_cm = MagicMock()
    mock_cm.get.side_effect = lambda name: {"reviewer": _make_reviewer()}.get(name)
    mock_cm_class.return_value = mock_cm

    mock_router = MagicMock()
    response = MagicMock()
    response.content = "## Verdict\n✅ APPROVE"
    response.model = "claude-sonnet"
    response.input_tokens = 100
    response.output_tokens = 50
    mock_router.complete.return_value = response
    mock_router_class.return_value = mock_router

    cmd_review_mr(mr=1, config="config/factory.yml", agents_dir="config/agents", post=False, second_review=False)

    prompt_used = mock_router.complete.call_args[1]["prompt"]
    assert "example.py" in prompt_used
    assert "+bar" in prompt_used


@patch("factory.commands.review_mr.GitLabClient")
@patch("factory.commands.review_mr.CompetenceManager")
@patch("factory.commands.review_mr.LlmRouter")
@patch("factory.commands.review_mr.console")
def test_cmd_review_mr_diff_via_hasattr_path(mock_console, mock_router_class, mock_cm_class, mock_gl_class):
    """Diff content is included in the prompt when changes() returns an object (hasattr path)."""
    mock_gl, mr = _make_gl_mock()
    # Replace the dict response with an object that has a .changes attribute
    changes_obj = MagicMock()
    changes_obj.changes = [
        {
            "old_path": "hasattr_file.py",
            "new_path": "hasattr_file.py",
            "diff": "@@ -1 +1 @@\n-old\n+new\n",
            "deleted_file": False,
        }
    ]
    mr.changes.return_value = changes_obj
    mock_gl_class.return_value = mock_gl

    mock_cm = MagicMock()
    mock_cm.get.side_effect = lambda name: {"reviewer": _make_reviewer()}.get(name)
    mock_cm_class.return_value = mock_cm

    mock_router = MagicMock()
    response = MagicMock()
    response.content = "## Verdict\n✅ APPROVE"
    response.model = "claude-sonnet"
    response.input_tokens = 100
    response.output_tokens = 50
    mock_router.complete.return_value = response
    mock_router_class.return_value = mock_router

    cmd_review_mr(mr=1, config="config/factory.yml", agents_dir="config/agents", post=False, second_review=False)

    prompt_used = mock_router.complete.call_args[1]["prompt"]
    assert "hasattr_file.py" in prompt_used
    assert "+new" in prompt_used


@patch("factory.commands.review_mr.GitLabClient")
@patch("factory.commands.review_mr.CompetenceManager")
@patch("factory.commands.review_mr.LlmRouter")
@patch("factory.commands.review_mr.console")
def test_cmd_review_mr_without_second_review(mock_console, mock_router_class, mock_cm_class, mock_gl_class):
    """Meta review is skipped when second_review=False."""
    mock_gl, _ = _make_gl_mock()
    mock_gl_class.return_value = mock_gl

    mock_cm = MagicMock()
    mock_cm.get.side_effect = lambda name: {"reviewer": _make_reviewer()}.get(name)
    mock_cm_class.return_value = mock_cm

    mock_router = MagicMock()
    response = MagicMock()
    response.content = "## Verdict\n✅ APPROVE"
    response.model = "claude-sonnet"
    response.input_tokens = 100
    response.output_tokens = 50
    mock_router.complete.return_value = response
    mock_router_class.return_value = mock_router

    cmd_review_mr(mr=1, config="config/factory.yml", agents_dir="config/agents", post=False, second_review=False)

    mock_router.complete.assert_called_once()


@patch("factory.commands.review_mr.GitLabClient")
@patch("factory.commands.review_mr.CompetenceManager")
@patch("factory.commands.review_mr.LlmRouter")
@patch("factory.commands.review_mr.console")
def test_cmd_review_mr_with_second_review(mock_console, mock_router_class, mock_cm_class, mock_gl_class):
    """Both reviewer and meta_reviewer are called when second_review=True."""
    mock_gl, _ = _make_gl_mock()
    mock_gl_class.return_value = mock_gl

    agents = {"reviewer": _make_reviewer(), "meta_reviewer": _make_meta_reviewer()}
    mock_cm = MagicMock()
    mock_cm.get.side_effect = agents.get
    mock_cm_class.return_value = mock_cm

    primary_response = MagicMock()
    primary_response.content = "## Verdict\n✅ APPROVE"
    primary_response.model = "claude-sonnet"
    primary_response.input_tokens = 100
    primary_response.output_tokens = 50

    meta_response = MagicMock()
    meta_response.content = "## Adjusted Verdict\n✅ APPROVE"
    meta_response.model = "mistral-large"
    meta_response.input_tokens = 80
    meta_response.output_tokens = 40

    mock_router = MagicMock()
    mock_router.complete.side_effect = [primary_response, meta_response]
    mock_router_class.return_value = mock_router

    cmd_review_mr(mr=1, config="config/factory.yml", agents_dir="config/agents", post=False, second_review=True)

    assert mock_router.complete.call_count == 2
    assert mock_router.complete.call_args_list[0][0][0].name == "reviewer"
    assert mock_router.complete.call_args_list[1][0][0].name == "meta_reviewer"


@patch("factory.commands.review_mr.GitLabClient")
@patch("factory.commands.review_mr.CompetenceManager")
@patch("factory.commands.review_mr.LlmRouter")
@patch("factory.commands.review_mr.console")
def test_cmd_review_mr_missing_meta_reviewer(mock_console, mock_router_class, mock_cm_class, mock_gl_class):
    """Missing meta_reviewer.yml warns and falls back to primary-only (no hard exit)."""
    mock_gl, _ = _make_gl_mock()
    mock_gl_class.return_value = mock_gl

    mock_cm = MagicMock()
    mock_cm.get.side_effect = lambda name: {"reviewer": _make_reviewer()}.get(name)
    mock_cm_class.return_value = mock_cm

    mock_router = MagicMock()
    response = MagicMock()
    response.content = "## Verdict\n✅ APPROVE"
    response.model = "claude-sonnet"
    response.input_tokens = 100
    response.output_tokens = 50
    mock_router.complete.return_value = response
    mock_router_class.return_value = mock_router

    cmd_review_mr(mr=1, config="config/factory.yml", agents_dir="config/agents", post=False, second_review=True)

    mock_console.print.assert_any_call(
        "[yellow]⚠ meta_reviewer agent not found in config/agents/ — "
        "running primary-only review.[/yellow]"
    )
    mock_router.complete.assert_called_once()


@patch("factory.commands.review_mr.GitLabClient")
@patch("factory.commands.review_mr.CompetenceManager")
@patch("factory.commands.review_mr.LlmRouter")
@patch("factory.commands.review_mr.console")
def test_cmd_review_mr_meta_reviewer_failure(mock_console, mock_router_class, mock_cm_class, mock_gl_class):
    """Primary review posts even when the meta-reviewer raises."""
    mock_gl, _ = _make_gl_mock()
    mock_gl_class.return_value = mock_gl

    agents = {"reviewer": _make_reviewer(), "meta_reviewer": _make_meta_reviewer()}
    mock_cm = MagicMock()
    mock_cm.get.side_effect = agents.get
    mock_cm_class.return_value = mock_cm

    primary_response = MagicMock()
    primary_response.content = "## Verdict\n✅ APPROVE"
    primary_response.model = "claude-sonnet"
    primary_response.input_tokens = 100
    primary_response.output_tokens = 50

    mock_router = MagicMock()
    mock_router.complete.side_effect = [primary_response, Exception("Meta reviewer failed")]
    mock_router_class.return_value = mock_router

    cmd_review_mr(mr=1, config="config/factory.yml", agents_dir="config/agents", post=False, second_review=True)

    assert mock_router.complete.call_count == 2
    mock_console.print.assert_any_call("[red]✗ Meta Reviewer failed: Meta reviewer failed[/red]")


@patch("factory.commands.review_mr.GitLabClient")
@patch("factory.commands.review_mr.CompetenceManager")
@patch("factory.commands.review_mr.LlmRouter")
@patch("factory.commands.review_mr.console")
def test_cmd_review_mr_with_post(mock_console, mock_router_class, mock_cm_class, mock_gl_class):
    """notes.create is called with a body containing both review sections when post=True."""
    mock_gl, mr = _make_gl_mock()
    mock_gl_class.return_value = mock_gl

    agents = {"reviewer": _make_reviewer(), "meta_reviewer": _make_meta_reviewer()}
    mock_cm = MagicMock()
    mock_cm.get.side_effect = agents.get
    mock_cm_class.return_value = mock_cm

    primary_response = MagicMock()
    primary_response.content = "Primary review content"
    primary_response.model = "claude-sonnet"
    primary_response.input_tokens = 100
    primary_response.output_tokens = 50

    meta_response = MagicMock()
    meta_response.content = "Meta review content"
    meta_response.model = "mistral-large"
    meta_response.input_tokens = 80
    meta_response.output_tokens = 40

    mock_router = MagicMock()
    mock_router.complete.side_effect = [primary_response, meta_response]
    mock_router_class.return_value = mock_router

    mock_note = MagicMock()
    mock_note.id = 123
    mr.notes.create.return_value = mock_note

    cmd_review_mr(mr=1, config="config/factory.yml", agents_dir="config/agents", post=True, second_review=True)

    mr.notes.create.assert_called_once()
    body = mr.notes.create.call_args[0][0]["body"]
    assert "### :mag: Code Review by SoftwareTeamFabrik" in body
    assert "Primary review content" in body
    assert "### :mag::mag: Meta Review (false-positive filter) by SoftwareTeamFabrik" in body
    assert "Meta review content" in body


@patch("factory.commands.review_mr.GitLabClient")
@patch("factory.commands.review_mr.CompetenceManager")
@patch("factory.commands.review_mr.LlmRouter")
@patch("factory.commands.review_mr.console")
def test_cmd_review_mr_with_post_final_verdict_label(mock_console, mock_router_class, mock_cm_class, mock_gl_class):
    """Final verdict label in the posted comment reflects the merged verdict."""
    mock_gl, mr = _make_gl_mock()
    mock_gl_class.return_value = mock_gl

    agents = {"reviewer": _make_reviewer(), "meta_reviewer": _make_meta_reviewer()}
    mock_cm = MagicMock()
    mock_cm.get.side_effect = agents.get
    mock_cm_class.return_value = mock_cm

    primary_response = MagicMock()
    # Primary says REQUEST CHANGES; meta says APPROVE → expect APPROVE (single-step allowed)
    primary_response.content = "## Verdict\n⚠️ REQUEST CHANGES\nsome findings"
    primary_response.model = "claude-sonnet"
    primary_response.input_tokens = 100
    primary_response.output_tokens = 50

    meta_response = MagicMock()
    meta_response.content = "## Adjusted Verdict\n✅ APPROVE"
    meta_response.model = "mistral-large"
    meta_response.input_tokens = 80
    meta_response.output_tokens = 40

    mock_router = MagicMock()
    mock_router.complete.side_effect = [primary_response, meta_response]
    mock_router_class.return_value = mock_router

    mock_note = MagicMock()
    mock_note.id = 456
    mr.notes.create.return_value = mock_note

    cmd_review_mr(mr=1, config="config/factory.yml", agents_dir="config/agents", post=True, second_review=True)

    body = mr.notes.create.call_args[0][0]["body"]
    assert "**Final verdict (after meta-review): ✅ APPROVE**" in body


# ---------------------------------------------------------------------------
# run_mr_review programmatic API tests
# ---------------------------------------------------------------------------

class TestRunMrReview:
    """Tests for the run_mr_review() programmatic API."""

    def _make_merge_request_mock(self):
        """Create a mock merge request with standard diff data."""
        mr = MagicMock()
        mr.title = "Feature: add login page"
        mr.description = "Implements the login page UI"
        mr.source_branch = "feature-login"
        mr.iid = 42
        mr.references = {"related_issues": []}
        mr.changes.return_value = {
            "changes": [
                {
                    "old_path": "login.py",
                    "new_path": "login.py",
                    "diff": "@@ -1,3 +1,4 @@\n import os\n+import auth\n def login():\n",
                    "deleted_file": False,
                }
            ]
        }
        return mr

    def _make_project_mock(self):
        """Create a mock GitLab project."""
        project = MagicMock()
        return project

    def _make_router_mock(self, content="## Verdict\n✅ APPROVE", model="claude-sonnet"):
        """Create a mock LLM router that returns the given content."""
        router = MagicMock()
        response = MagicMock()
        response.content = content
        response.model = model
        response.input_tokens = 100
        response.output_tokens = 50
        router.complete.return_value = response
        return router

    def test_returns_review_result(self):
        """run_mr_review returns a ReviewResult, not a string."""
        mr = self._make_merge_request_mock()
        project = self._make_project_mock()
        router = self._make_router_mock()
        reviewer = _make_reviewer()

        result = run_mr_review(
            merge_request=mr,
            project=project,
            router=router,
            reviewer=reviewer,
            post=False,
            update_wiki=False,
        )

        assert isinstance(result, ReviewResult)
        assert result.verdict == "green"
        assert result.posted is False

    def test_returns_review_result_with_approve(self):
        """Approve verdict maps to 'green'."""
        mr = self._make_merge_request_mock()
        project = self._make_project_mock()
        router = self._make_router_mock(content="## Verdict\n✅ APPROVE")
        reviewer = _make_reviewer()

        result = run_mr_review(
            merge_request=mr,
            project=project,
            router=router,
            reviewer=reviewer,
            post=False,
            update_wiki=False,
        )

        assert result.verdict == "green"

    def test_returns_review_result_with_request_changes(self):
        """Request changes verdict maps to 'yellow'."""
        mr = self._make_merge_request_mock()
        project = self._make_project_mock()
        router = self._make_router_mock(content="## Verdict\n⚠️ REQUEST CHANGES")
        reviewer = _make_reviewer()

        result = run_mr_review(
            merge_request=mr,
            project=project,
            router=router,
            reviewer=reviewer,
            post=False,
            update_wiki=False,
        )

        assert result.verdict == "yellow"

    def test_returns_review_result_with_block(self):
        """Block verdict maps to 'red'."""
        mr = self._make_merge_request_mock()
        project = self._make_project_mock()
        router = self._make_router_mock(content="## Verdict\n🚫 BLOCK")
        reviewer = _make_reviewer()

        result = run_mr_review(
            merge_request=mr,
            project=project,
            router=router,
            reviewer=reviewer,
            post=False,
            update_wiki=False,
        )

        assert result.verdict == "red"

    def test_comment_body_contains_review(self):
        """The comment body contains the review text."""
        mr = self._make_merge_request_mock()
        project = self._make_project_mock()
        router = self._make_router_mock(content="## Verdict\n✅ APPROVE\n\nCode looks good.")
        reviewer = _make_reviewer()

        result = run_mr_review(
            merge_request=mr,
            project=project,
            router=router,
            reviewer=reviewer,
            post=False,
            update_wiki=False,
        )

        assert "Code looks good." in result.comment_body
        assert "### :mag: Code Review by SoftwareTeamFabrik" in result.comment_body

    def test_posts_comment_when_post_true(self):
        """When post=True, a comment is created on the MR."""
        mr = self._make_merge_request_mock()
        project = self._make_project_mock()
        router = self._make_router_mock()
        reviewer = _make_reviewer()

        result = run_mr_review(
            merge_request=mr,
            project=project,
            router=router,
            reviewer=reviewer,
            post=True,
            update_wiki=False,
        )

        assert result.posted is True
        mr.notes.create.assert_called_once()
        body = mr.notes.create.call_args[0][0]["body"]
        assert "### :mag: Code Review by SoftwareTeamFabrik" in body

    def test_no_comment_when_post_false(self):
        """When post=False, no comment is created on the MR."""
        mr = self._make_merge_request_mock()
        project = self._make_project_mock()
        router = self._make_router_mock()
        reviewer = _make_reviewer()

        result = run_mr_review(
            merge_request=mr,
            project=project,
            router=router,
            reviewer=reviewer,
            post=False,
            update_wiki=False,
        )

        assert result.posted is False
        mr.notes.create.assert_not_called()

    def test_with_meta_reviewer(self):
        """When a meta_reviewer is provided, both reviews are included."""
        mr = self._make_merge_request_mock()
        project = self._make_project_mock()
        reviewer = _make_reviewer()
        meta_reviewer = _make_meta_reviewer()

        primary_response = MagicMock()
        primary_response.content = "## Verdict\n⚠️ REQUEST CHANGES"
        primary_response.model = "claude-sonnet"
        primary_response.input_tokens = 100
        primary_response.output_tokens = 50

        meta_response = MagicMock()
        meta_response.content = "## Adjusted Verdict\n✅ APPROVE"
        meta_response.model = "mistral-large"
        meta_response.input_tokens = 80
        meta_response.output_tokens = 40

        router = MagicMock()
        router.complete.side_effect = [primary_response, meta_response]

        result = run_mr_review(
            merge_request=mr,
            project=project,
            router=router,
            reviewer=reviewer,
            meta_reviewer=meta_reviewer,
            post=False,
            update_wiki=False,
        )

        assert result.verdict == "green"  # meta upgraded from yellow to green
        assert "Meta Review" in result.comment_body
        assert router.complete.call_count == 2

    def test_meta_reviewer_failure_falls_back_to_primary(self):
        """When meta_reviewer raises, the primary review result is still returned."""
        mr = self._make_merge_request_mock()
        project = self._make_project_mock()
        reviewer = _make_reviewer()
        meta_reviewer = _make_meta_reviewer()

        primary_response = MagicMock()
        primary_response.content = "## Verdict\n✅ APPROVE"
        primary_response.model = "claude-sonnet"
        primary_response.input_tokens = 100
        primary_response.output_tokens = 50

        router = MagicMock()
        router.complete.side_effect = [primary_response, Exception("Meta reviewer failed")]

        result = run_mr_review(
            merge_request=mr,
            project=project,
            router=router,
            reviewer=reviewer,
            meta_reviewer=meta_reviewer,
            post=False,
            update_wiki=False,
        )

        assert result.verdict == "green"
        assert "### :mag: Code Review by SoftwareTeamFabrik" in result.comment_body
        # Meta review section should NOT be in the comment when meta fails
        assert "Meta Review" not in result.comment_body

    def test_verdict_label_set_on_post(self):
        """When post=True, set_verdict_label is called with the correct color."""
        mr = self._make_merge_request_mock()
        project = self._make_project_mock()
        router = self._make_router_mock(content="## Verdict\n🚫 BLOCK")
        reviewer = _make_reviewer()

        with patch("factory.commands.review_mr.set_verdict_label") as mock_set_label:
            result = run_mr_review(
                merge_request=mr,
                project=project,
                router=router,
                reviewer=reviewer,
                post=True,
                update_wiki=False,
            )

            mock_set_label.assert_called_once_with(mr, "red")

    @patch("factory.commands.review_mr.cmd_update_wiki")
    def test_wiki_triggered_on_approve(self, mock_wiki):
        """Wiki agent is spawned when review approves and update_wiki=True."""
        mr = self._make_merge_request_mock()
        project = self._make_project_mock()
        router = self._make_router_mock(content="## Verdict\n✅ APPROVE")
        reviewer = _make_reviewer()

        result = run_mr_review(
            merge_request=mr,
            project=project,
            router=router,
            reviewer=reviewer,
            post=False,
            update_wiki=True,
        )

        assert result.verdict == "green"
        mock_wiki.assert_called_once_with(mr_id=42, config="config/factory.yml")

    @patch("factory.commands.review_mr.cmd_update_wiki")
    def test_no_wiki_on_request_changes(self, mock_wiki):
        """Wiki agent is NOT spawned when review requests changes."""
        mr = self._make_merge_request_mock()
        project = self._make_project_mock()
        router = self._make_router_mock(content="## Verdict\n⚠️ REQUEST CHANGES")
        reviewer = _make_reviewer()

        result = run_mr_review(
            merge_request=mr,
            project=project,
            router=router,
            reviewer=reviewer,
            post=False,
            update_wiki=True,
        )

        assert result.verdict == "yellow"
        mock_wiki.assert_not_called()

    @patch("factory.commands.review_mr.cmd_update_wiki")
    def test_no_wiki_on_block(self, mock_wiki):
        """Wiki agent is NOT spawned when review blocks."""
        mr = self._make_merge_request_mock()
        project = self._make_project_mock()
        router = self._make_router_mock(content="## Verdict\n🚫 BLOCK")
        reviewer = _make_reviewer()

        result = run_mr_review(
            merge_request=mr,
            project=project,
            router=router,
            reviewer=reviewer,
            post=False,
            update_wiki=True,
        )

        assert result.verdict == "red"
        mock_wiki.assert_not_called()

    @patch("factory.commands.review_mr.cmd_update_wiki")
    def test_no_wiki_when_update_wiki_false(self, mock_wiki):
        """Wiki agent is NOT spawned when update_wiki=False, even on approve."""
        mr = self._make_merge_request_mock()
        project = self._make_project_mock()
        router = self._make_router_mock(content="## Verdict\n✅ APPROVE")
        reviewer = _make_reviewer()

        result = run_mr_review(
            merge_request=mr,
            project=project,
            router=router,
            reviewer=reviewer,
            post=False,
            update_wiki=False,
        )

        assert result.verdict == "green"
        mock_wiki.assert_not_called()

    @patch("factory.commands.review_mr.cmd_update_wiki")
    def test_wiki_failure_does_not_crash(self, mock_wiki):
        """Wiki generation failure is caught and does not affect the review result."""
        mr = self._make_merge_request_mock()
        project = self._make_project_mock()
        router = self._make_router_mock(content="## Verdict\n✅ APPROVE")
        reviewer = _make_reviewer()
        mock_wiki.side_effect = Exception("Wiki connection failed")

        # Should not raise
        result = run_mr_review(
            merge_request=mr,
            project=project,
            router=router,
            reviewer=reviewer,
            post=False,
            update_wiki=True,
        )

        # Review result is still valid
        assert result.verdict == "green"
        assert isinstance(result, ReviewResult)

    @patch("factory.commands.review_mr.cmd_update_wiki")
    def test_wiki_uses_meta_verdict_when_approved(self, mock_wiki):
        """Wiki is spawned when the merged verdict (after meta-review) is approved."""
        mr = self._make_merge_request_mock()
        project = self._make_project_mock()
        reviewer = _make_reviewer()
        meta_reviewer = _make_meta_reviewer()

        primary_response = MagicMock()
        primary_response.content = "## Verdict\n⚠️ REQUEST CHANGES"
        primary_response.model = "claude-sonnet"
        primary_response.input_tokens = 100
        primary_response.output_tokens = 50

        meta_response = MagicMock()
        meta_response.content = "## Adjusted Verdict\n✅ APPROVE"
        meta_response.model = "mistral-large"
        meta_response.input_tokens = 80
        meta_response.output_tokens = 40

        router = MagicMock()
        router.complete.side_effect = [primary_response, meta_response]

        result = run_mr_review(
            merge_request=mr,
            project=project,
            router=router,
            reviewer=reviewer,
            meta_reviewer=meta_reviewer,
            post=False,
            update_wiki=True,
        )

        # Meta-reviewer upgraded from yellow to green → wiki should be triggered
        assert result.verdict == "green"
        mock_wiki.assert_called_once()

    @patch("factory.commands.review_mr.cmd_update_wiki")
    def test_no_wiki_when_meta_downgrades_to_changes(self, mock_wiki):
        """Wiki is NOT spawned when meta-reviewer can't upgrade past yellow."""
        mr = self._make_merge_request_mock()
        project = self._make_project_mock()
        reviewer = _make_reviewer()
        meta_reviewer = _make_meta_reviewer()

        # Primary says BLOCK, meta says REQUEST CHANGES → merged is yellow
        primary_response = MagicMock()
        primary_response.content = "## Verdict\n🚫 BLOCK"
        primary_response.model = "claude-sonnet"
        primary_response.input_tokens = 100
        primary_response.output_tokens = 50

        meta_response = MagicMock()
        meta_response.content = "## Adjusted Verdict\n⚠️ REQUEST CHANGES"
        meta_response.model = "mistral-large"
        meta_response.input_tokens = 80
        meta_response.output_tokens = 40

        router = MagicMock()
        router.complete.side_effect = [primary_response, meta_response]

        result = run_mr_review(
            merge_request=mr,
            project=project,
            router=router,
            reviewer=reviewer,
            meta_reviewer=meta_reviewer,
            post=False,
            update_wiki=True,
        )

        # Verdict is yellow (REQUEST CHANGES) — wiki should NOT be triggered
        assert result.verdict == "yellow"
        mock_wiki.assert_not_called()

    def test_diff_fetching_handles_hasattr_changes(self):
        """run_mr_review handles MR changes via hasattr path."""
        mr = MagicMock()
        mr.title = "Test MR"
        mr.description = "Test"
        mr.source_branch = "test"
        mr.iid = 42
        mr.references = {"related_issues": []}

        changes_obj = MagicMock()
        changes_obj.changes = [
            {
                "old_path": "file.py",
                "new_path": "file.py",
                "diff": "@@ -1 +1 @@\n-old\n+new\n",
                "deleted_file": False,
            }
        ]
        mr.changes.return_value = changes_obj

        project = MagicMock()
        router = self._make_router_mock()
        reviewer = _make_reviewer()

        result = run_mr_review(
            merge_request=mr,
            project=project,
            router=router,
            reviewer=reviewer,
            post=False,
            update_wiki=False,
        )

        assert isinstance(result, ReviewResult)
        # Verify the diff was included in the prompt
        prompt_used = router.complete.call_args[1]["prompt"]
        assert "file.py" in prompt_used

    def test_empty_changes_still_returns_result(self):
        """run_mr_review returns a valid result even with no diff changes."""
        mr = MagicMock()
        mr.title = "Empty MR"
        mr.description = ""
        mr.source_branch = "test"
        mr.iid = 99
        mr.references = {"related_issues": []}
        mr.changes.return_value = {"changes": []}

        project = MagicMock()
        router = self._make_router_mock()
        reviewer = _make_reviewer()

        result = run_mr_review(
            merge_request=mr,
            project=project,
            router=router,
            reviewer=reviewer,
            post=False,
            update_wiki=False,
        )

        assert isinstance(result, ReviewResult)
        assert result.verdict == "green"  # default fallback for no verdict

    def test_diff_fetch_error_handled(self):
        """run_mr_review handles diff fetch errors gracefully."""
        mr = MagicMock()
        mr.title = "Broken MR"
        mr.description = ""
        mr.source_branch = "test"
        mr.iid = 50
        mr.references = {"related_issues": []}
        mr.changes.side_effect = Exception("GitLab API error")

        project = MagicMock()
        router = self._make_router_mock()
        reviewer = _make_reviewer()

        result = run_mr_review(
            merge_request=mr,
            project=project,
            router=router,
            reviewer=reviewer,
            post=False,
            update_wiki=False,
        )

        assert isinstance(result, ReviewResult)
        # The prompt should contain the error marker
        prompt_used = router.complete.call_args[1]["prompt"]
        assert "[Error fetching diff:" in prompt_used or "(empty diff)" in prompt_used


# ---------------------------------------------------------------------------
# cmd_review_mr wiki integration tests
# ---------------------------------------------------------------------------

class TestCmdReviewMrWiki:
    """Tests for wiki agent integration in cmd_review_mr."""

    @patch("factory.commands.review_mr.cmd_update_wiki")
    @patch("factory.commands.review_mr.GitLabClient")
    @patch("factory.commands.review_mr.CompetenceManager")
    @patch("factory.commands.review_mr.LlmRouter")
    @patch("factory.commands.review_mr.console")
    def test_wiki_triggered_on_approve_cli(self, mock_console, mock_router_class, mock_cm_class, mock_gl_class, mock_wiki):
        """Wiki agent is spawned when review approves via CLI command."""
        mock_gl, _ = _make_gl_mock()
        mock_gl_class.return_value = mock_gl

        mock_cm = MagicMock()
        mock_cm.get.side_effect = lambda name: {"reviewer": _make_reviewer()}.get(name)
        mock_cm_class.return_value = mock_cm

        mock_router = MagicMock()
        response = MagicMock()
        response.content = "## Verdict\n✅ APPROVE"
        response.model = "claude-sonnet"
        response.input_tokens = 100
        response.output_tokens = 50
        mock_router.complete.return_value = response
        mock_router_class.return_value = mock_router

        cmd_review_mr(mr=1, config="config/factory.yml", agents_dir="config/agents", post=False, second_review=False, update_wiki=True)

        mock_wiki.assert_called_once_with(mr_id=1, config="config/factory.yml")

    @patch("factory.commands.review_mr.cmd_update_wiki")
    @patch("factory.commands.review_mr.GitLabClient")
    @patch("factory.commands.review_mr.CompetenceManager")
    @patch("factory.commands.review_mr.LlmRouter")
    @patch("factory.commands.review_mr.console")
    def test_no_wiki_on_request_changes_cli(self, mock_console, mock_router_class, mock_cm_class, mock_gl_class, mock_wiki):
        """Wiki agent is NOT spawned when review requests changes via CLI command."""
        mock_gl, _ = _make_gl_mock()
        mock_gl_class.return_value = mock_gl

        mock_cm = MagicMock()
        mock_cm.get.side_effect = lambda name: {"reviewer": _make_reviewer()}.get(name)
        mock_cm_class.return_value = mock_cm

        mock_router = MagicMock()
        response = MagicMock()
        response.content = "## Verdict\n⚠️ REQUEST CHANGES"
        response.model = "claude-sonnet"
        response.input_tokens = 100
        response.output_tokens = 50
        mock_router.complete.return_value = response
        mock_router_class.return_value = mock_router

        cmd_review_mr(mr=1, config="config/factory.yml", agents_dir="config/agents", post=False, second_review=False, update_wiki=True)

        mock_wiki.assert_not_called()

    @patch("factory.commands.review_mr.cmd_update_wiki")
    @patch("factory.commands.review_mr.GitLabClient")
    @patch("factory.commands.review_mr.CompetenceManager")
    @patch("factory.commands.review_mr.LlmRouter")
    @patch("factory.commands.review_mr.console")
    def test_no_wiki_when_flag_disabled_cli(self, mock_console, mock_router_class, mock_cm_class, mock_gl_class, mock_wiki):
        """Wiki agent is NOT spawned when --no-update-wiki is passed, even on approve."""
        mock_gl, _ = _make_gl_mock()
        mock_gl_class.return_value = mock_gl

        mock_cm = MagicMock()
        mock_cm.get.side_effect = lambda name: {"reviewer": _make_reviewer()}.get(name)
        mock_cm_class.return_value = mock_cm

        mock_router = MagicMock()
        response = MagicMock()
        response.content = "## Verdict\n✅ APPROVE"
        response.model = "claude-sonnet"
        response.input_tokens = 100
        response.output_tokens = 50
        mock_router.complete.return_value = response
        mock_router_class.return_value = mock_router

        cmd_review_mr(mr=1, config="config/factory.yml", agents_dir="config/agents", post=False, second_review=False, update_wiki=False)

        mock_wiki.assert_not_called()

    @patch("factory.commands.review_mr.cmd_update_wiki")
    @patch("factory.commands.review_mr.GitLabClient")
    @patch("factory.commands.review_mr.CompetenceManager")
    @patch("factory.commands.review_mr.LlmRouter")
    @patch("factory.commands.review_mr.console")
    def test_wiki_failure_does_not_crash_cli(self, mock_console, mock_router_class, mock_cm_class, mock_gl_class, mock_wiki):
        """Wiki generation failure is caught and doesn't crash the CLI command."""
        mock_gl, _ = _make_gl_mock()
        mock_gl_class.return_value = mock_gl

        mock_cm = MagicMock()
        mock_cm.get.side_effect = lambda name: {"reviewer": _make_reviewer()}.get(name)
        mock_cm_class.return_value = mock_cm

        mock_router = MagicMock()
        response = MagicMock()
        response.content = "## Verdict\n✅ APPROVE"
        response.model = "claude-sonnet"
        response.input_tokens = 100
        response.output_tokens = 50
        mock_router.complete.return_value = response
        mock_router_class.return_value = mock_router

        mock_wiki.side_effect = Exception("Wiki connection failed")

        # Should not raise
        cmd_review_mr(mr=1, config="config/factory.yml", agents_dir="config/agents", post=False, second_review=False, update_wiki=True)

        # Verify the error was printed
        mock_console.print.assert_any_call("[yellow]⚠ Wiki documentation generation failed: Wiki connection failed[/yellow]")