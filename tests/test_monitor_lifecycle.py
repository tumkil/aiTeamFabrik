"""Integration tests for FactoryMonitor lifecycle decisions.

Builds an in-memory fake GitLab (project, MRs, issues) and drives
`_check_mrs` / `_check_issues` end-to-end with the heavy collaborators
(`_run_review`, `_run_refinement`, `_run_issue_processing`, `_auto_merge`)
patched to recording stubs.

These tests are regression coverage for the lifecycle bugs fixed in the
recent monitor work:
- ``is_draft`` only blocks auto-merge, not review/refinement.
- ``_has_request_changes`` and ``_all_approved`` look at the *latest*
  factory review note, not aggregated history.
- A new issue is routed through ``_run_issue_processing``.
- ``_run_refinement`` failures break the refinement loop instead of
  silently retrying the same failing iteration.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Iterable, Optional

import pytest

from factory.core.monitor import FactoryMonitor, WIP_LIMITS


# ---------------------------------------------------------------------------
# Fake GitLab — minimal subset of the python-gitlab surface monitor uses.
# ---------------------------------------------------------------------------

class FakeNote:
    def __init__(self, body: str, author: str = "factory-bot") -> None:
        self.body = body
        self.author = {"name": author}


class FakeNotesManager:
    def __init__(self) -> None:
        self._notes: list[FakeNote] = []

    def list(self, *, all: bool = True, **_kwargs) -> list[FakeNote]:
        return list(self._notes)

    def create(self, data: dict) -> FakeNote:
        n = FakeNote(body=data.get("body", ""))
        self._notes.append(n)
        return n


class FakeMR:
    def __init__(
        self,
        iid: int,
        title: str = "MR",
        source_branch: str = "feature/x",
        draft: bool = False,
        notes: Optional[Iterable[str]] = None,
        labels: Optional[Iterable[str]] = None,
    ) -> None:
        self.iid = iid
        self.title = title
        self.description = ""
        self.source_branch = source_branch
        self.draft = draft
        self.work_in_progress = draft  # mirror legacy field
        self.state = "opened"
        self.notes = FakeNotesManager()
        if notes:
            for body in notes:
                self.notes._notes.append(FakeNote(body))
        self.labels = list(labels or [])
        self.merged = False
        self.saved = 0

    def merge(self, *, remove_source_branch: bool = False) -> None:
        self.merged = True
        self.state = "merged"

    def save(self) -> None:
        self.saved += 1


class FakeIssue:
    def __init__(
        self,
        iid: int,
        title: str = "Issue",
        description: str = "",
        labels: Optional[list[str]] = None,
    ) -> None:
        self.iid = iid
        self.title = title
        self.description = description
        self.labels = list(labels or [])
        self.notes = FakeNotesManager()
        self.state = "opened"
        self.state_event: Optional[str] = None
        self.saved = 0

    def save(self) -> None:
        self.saved += 1


class FakeMRsManager:
    def __init__(self, mrs: list[FakeMR]) -> None:
        self._mrs = mrs

    def list(self, *, state: str = "opened", all: bool = True, **_kwargs) -> list[FakeMR]:
        return [m for m in self._mrs if m.state == state]

    def get(self, iid: int) -> FakeMR:
        for m in self._mrs:
            if m.iid == iid:
                return m
        raise KeyError(iid)


class FakeIssuesManager:
    def __init__(self, issues: list[FakeIssue]) -> None:
        self._issues = issues

    def list(self, *, state: str = "opened", all: bool = True, **_kwargs) -> list[FakeIssue]:
        # Match the real python-gitlab behaviour: ``state="all"`` returns
        # every issue regardless of open/closed status, while any other
        # value filters by that exact state string.
        if state == "all":
            return list(self._issues)
        return [i for i in self._issues if i.state == state]


class FakeProject:
    def __init__(
        self,
        mrs: Optional[list[FakeMR]] = None,
        issues: Optional[list[FakeIssue]] = None,
    ) -> None:
        self.mergerequests = FakeMRsManager(mrs or [])
        self.issues = FakeIssuesManager(issues or [])
        self.default_branch = "main"
        self.id = 1
        self.http_url_to_repo = "https://example.invalid/repo.git"


# ---------------------------------------------------------------------------
# Monitor builder — bypass __init__ which requires a real GitLab connection.
# ---------------------------------------------------------------------------

_FACTORY_REVIEW_HEADER = "### :mag: Code Review by SoftwareTeamFabrik"


def _build_monitor(project: FakeProject) -> FactoryMonitor:
    """Construct a FactoryMonitor without touching the real GitLab."""
    mon = FactoryMonitor.__new__(FactoryMonitor)
    mon._gl = SimpleNamespace(
        project=project,
        hostname="example.invalid",
        _url="https://example.invalid",
        create_merge_request=lambda **_: None,
    )
    mon._competence = SimpleNamespace(get=lambda _name: SimpleNamespace(
        name=_name, system_prompt="", display_name=_name,
    ))
    mon._scrum = None
    mon._router = None
    mon._token = ""
    mon._budget_manager = SimpleNamespace(can_consume=lambda *_a, **_k: (True, ""))
    mon._poll_interval = 30
    mon._active_issues = {}
    mon._last_processed_mrs = {}
    mon._last_processed_issues = {}
    return mon


# ---------------------------------------------------------------------------
# _check_mrs lifecycle
# ---------------------------------------------------------------------------

class TestCheckMRsLifecycle:
    def test_unreviewed_mr_triggers_review(self, monkeypatch):
        mr = FakeMR(iid=10)
        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        calls: dict[str, list] = {"review": [], "refine": [], "merge": []}
        monkeypatch.setattr(mon, "_run_review", lambda mr, *a, **kw: calls["review"].append(mr.iid))
        monkeypatch.setattr(mon, "_run_refinement", lambda mr, *a, **kw: calls["refine"].append(mr.iid))
        monkeypatch.setattr(mon, "_auto_merge", lambda mr: calls["merge"].append(mr.iid))

        mon._check_mrs(project)

        assert calls["review"] == [10]
        assert calls["refine"] == []
        assert calls["merge"] == []

    def test_request_changes_triggers_refinement_then_rereview(self, monkeypatch):
        mr = FakeMR(iid=11, notes=[
            f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n⚠️ REQUEST CHANGES\n\nFix X.",
        ])
        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        calls = {"review": 0, "refine": 0, "merge": 0}

        def fake_refine(mr, *a, **kw):
            # Simulate the developer pushing a fix; subsequent review will approve.
            mr.notes.create({"body": f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n✅ APPROVE"})
            calls["refine"] += 1

        monkeypatch.setattr(mon, "_run_review", lambda mr, *a, **kw: calls.update(review=calls["review"] + 1))
        monkeypatch.setattr(mon, "_run_refinement", fake_refine)
        monkeypatch.setattr(mon, "_auto_merge", lambda mr: calls.update(merge=calls["merge"] + 1))

        mon._check_mrs(project)

        # One refinement, then a re-review, then the auto-merge.
        assert calls["refine"] == 1
        assert calls["review"] >= 1, "expected a re-review after refinement"
        assert calls["merge"] == 1, "approved-after-refine MR must be auto-merged"

    def test_approved_mr_is_auto_merged(self, monkeypatch):
        mr = FakeMR(iid=12, notes=[
            f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n✅ APPROVE",
        ])
        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        calls = {"review": 0, "refine": 0, "merge": 0}
        monkeypatch.setattr(mon, "_run_review", lambda *a, **kw: calls.update(review=calls["review"] + 1))
        monkeypatch.setattr(mon, "_run_refinement", lambda *a, **kw: calls.update(refine=calls["refine"] + 1))
        monkeypatch.setattr(mon, "_auto_merge", lambda mr: calls.update(merge=calls["merge"] + 1))

        mon._check_mrs(project)

        assert calls["refine"] == 0
        assert calls["merge"] == 1

    def test_latest_approve_supersedes_earlier_request_changes(self, monkeypatch):
        """Regression: a fresh APPROVE after an older REQUEST CHANGES must
        be honoured. Earlier monitor scanned all history and never auto-merged
        such MRs."""
        mr = FakeMR(iid=13, notes=[
            f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n⚠️ REQUEST CHANGES\n\nFix Y.",
            f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n✅ APPROVE",  # newer
        ])
        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        calls = {"refine": 0, "merge": 0}
        monkeypatch.setattr(mon, "_run_review", lambda *a, **kw: None)
        monkeypatch.setattr(mon, "_run_refinement", lambda *a, **kw: calls.update(refine=calls["refine"] + 1))
        monkeypatch.setattr(mon, "_auto_merge", lambda mr: calls.update(merge=calls["merge"] + 1))

        mon._check_mrs(project)

        assert calls["refine"] == 0, "must not refine an MR whose latest verdict is APPROVE"
        assert calls["merge"] == 1, "latest APPROVE must trigger auto-merge"

    def test_draft_mr_is_reviewed_and_refined_but_not_merged(self, monkeypatch):
        """Regression: ``is_draft`` once blocked review and refinement; it
        should only block auto-merge."""
        mr = FakeMR(iid=14, draft=True, notes=[
            f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n✅ APPROVE",
        ])
        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        calls = {"merge": 0}
        monkeypatch.setattr(mon, "_run_review", lambda *a, **kw: None)
        monkeypatch.setattr(mon, "_run_refinement", lambda *a, **kw: None)
        monkeypatch.setattr(mon, "_auto_merge", lambda mr: calls.update(merge=calls["merge"] + 1))

        mon._check_mrs(project)

        assert calls["merge"] == 0, "draft MRs must not be auto-merged"

    def test_draft_mr_with_request_changes_still_refined(self, monkeypatch):
        mr = FakeMR(iid=15, draft=True, notes=[
            f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n🚫 BLOCK\n\nfix it",
        ])
        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        calls = {"refine": 0}
        # Refinement makes the verdict approve so the loop terminates.
        def fake_refine(mr, *a, **kw):
            mr.notes.create({"body": f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n✅ APPROVE"})
            calls["refine"] += 1

        monkeypatch.setattr(mon, "_run_review", lambda *a, **kw: None)
        monkeypatch.setattr(mon, "_run_refinement", fake_refine)
        monkeypatch.setattr(mon, "_auto_merge", lambda *a, **kw: None)

        mon._check_mrs(project)

        assert calls["refine"] == 1

    def test_recently_processed_mr_is_skipped(self, monkeypatch):
        mr = FakeMR(iid=16)
        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)
        mon._last_processed_mrs[16] = datetime.now()

        calls = {"review": 0}
        monkeypatch.setattr(mon, "_run_review", lambda *a, **kw: calls.update(review=calls["review"] + 1))

        mon._check_mrs(project)

        assert calls["review"] == 0, "recently-processed MR must not be re-reviewed"

    def test_mixed_human_blocker_notes_trigger_refinement(self, monkeypatch):
        """Test that human blocker notes trigger refinement when no factory review exists."""
        mr = FakeMR(iid=17, notes=[
            "This looks good overall",
            "REQUEST CHANGES: Please fix the typo in line 42",  # human blocker
            "Thanks for the contribution!",
        ])
        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        calls = {"refine": 0}
        
        def fake_refine(mr, *a, **kw):
            # Simulate the developer pushing a fix
            mr.notes.create({"body": f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n✅ APPROVE"})
            calls["refine"] += 1

        monkeypatch.setattr(mon, "_run_review", lambda *a, **kw: None)
        monkeypatch.setattr(mon, "_run_refinement", fake_refine)
        monkeypatch.setattr(mon, "_auto_merge", lambda *a, **kw: None)

        mon._check_mrs(project)

        assert calls["refine"] == 1, "Human blocker notes must trigger refinement when no factory review exists"


class TestVerdictLabelDrivesLifecycle:
    """Verdict labels (factory-verdict-{approve,changes,block}) are the
    source of truth for review state. Comment-scan remains as a fallback
    for unlabelled MRs and is exercised by the older tests above."""

    def test_label_approve_triggers_auto_merge(self, monkeypatch):
        from factory.core.verdict import VERDICT_APPROVE
        mr = FakeMR(iid=30, labels=[VERDICT_APPROVE], notes=[
            f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n✅ APPROVE",
        ])
        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        calls = {"merge": 0, "refine": 0}
        monkeypatch.setattr(mon, "_run_review", lambda *a, **kw: None)
        monkeypatch.setattr(mon, "_run_refinement",
                            lambda *a, **kw: calls.update(refine=calls["refine"] + 1))
        monkeypatch.setattr(mon, "_auto_merge",
                            lambda mr: calls.update(merge=calls["merge"] + 1))

        mon._check_mrs(project)

        assert calls["merge"] == 1
        assert calls["refine"] == 0

    def test_label_changes_triggers_refinement_even_when_comment_says_approve(self, monkeypatch):
        """Label is authoritative: if the label says CHANGES, refine, even if
        a stale comment scan would have read APPROVE."""
        from factory.core.verdict import VERDICT_APPROVE, VERDICT_CHANGES
        mr = FakeMR(iid=31, labels=[VERDICT_CHANGES], notes=[
            f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n✅ APPROVE",  # contradicts label
        ])
        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        calls = {"refine": 0, "merge": 0}

        def fake_refine(mr, *a, **kw):
            mr.labels = [VERDICT_APPROVE]
            mr.notes.create({"body": f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n✅ APPROVE"})
            calls["refine"] += 1

        monkeypatch.setattr(mon, "_run_review", lambda *a, **kw: None)
        monkeypatch.setattr(mon, "_run_refinement", fake_refine)
        monkeypatch.setattr(mon, "_auto_merge",
                            lambda mr: calls.update(merge=calls["merge"] + 1))

        mon._check_mrs(project)

        assert calls["refine"] == 1, "label CHANGES must drive the refinement loop"
        assert calls["merge"] == 1, "after refinement flips the label, auto-merge must fire"

    def test_label_block_triggers_refinement(self, monkeypatch):
        from factory.core.verdict import VERDICT_APPROVE, VERDICT_BLOCK
        mr = FakeMR(iid=32, labels=[VERDICT_BLOCK])
        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        calls = {"refine": 0}

        def fake_refine(mr, *a, **kw):
            mr.labels = [VERDICT_APPROVE]
            calls["refine"] += 1

        monkeypatch.setattr(mon, "_run_review", lambda *a, **kw: None)
        monkeypatch.setattr(mon, "_run_refinement", fake_refine)
        monkeypatch.setattr(mon, "_auto_merge", lambda *a, **kw: None)

        mon._check_mrs(project)

        assert calls["refine"] == 1


# ---------------------------------------------------------------------------
# _check_issues lifecycle
# ---------------------------------------------------------------------------

class TestCheckIssuesLifecycle:
    def test_new_issue_without_mr_triggers_processing(self, monkeypatch):
        issue = FakeIssue(iid=20, title="Add feature", description="please", labels=["ready"])
        project = FakeProject(issues=[issue])
        mon = _build_monitor(project)

        calls: list[int] = []
        monkeypatch.setattr(
            mon, "_run_issue_processing",
            lambda iss, *a, **kw: (calls.append(iss.iid), None)[1],
        )

        mon._check_issues(project)

        assert calls == [20]
        assert mon._last_processed_issues.get(20) is not None

    def test_issue_with_open_mr_routes_to_mr_lifecycle(self, monkeypatch):
        # Issue 21 has its branch issue-21 already open as MR.
        mr = FakeMR(iid=99, source_branch="issue-21", notes=[
            f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n⚠️ REQUEST CHANGES\n\nfix",
        ])
        issue = FakeIssue(iid=21, labels=["ready"])
        project = FakeProject(mrs=[mr], issues=[issue])
        mon = _build_monitor(project)

        calls = {"refine": 0, "process": 0, "review": 0}
        monkeypatch.setattr(mon, "_run_issue_processing",
                            lambda *a, **kw: calls.update(process=calls["process"] + 1))
        monkeypatch.setattr(mon, "_run_refinement",
                            lambda *a, **kw: calls.update(refine=calls["refine"] + 1))
        monkeypatch.setattr(mon, "_run_review",
                            lambda *a, **kw: calls.update(review=calls["review"] + 1))

        mon._check_issues(project)

        assert calls["process"] == 0, "must not re-process an issue that already has an MR"
        assert calls["refine"] == 1, "REQUEST CHANGES on issue-bound MR must trigger refinement"

    def test_wip_limit_blocks_new_issue_processing(self, monkeypatch):
        issue = FakeIssue(iid=22, labels=["ready"])
        project = FakeProject(issues=[issue])
        mon = _build_monitor(project)
        # Saturate WIP for developer.
        mon._active_issues["developer"] = WIP_LIMITS["developer"]

        calls: list[int] = []
        monkeypatch.setattr(
            mon, "_run_issue_processing",
            lambda iss, *a, **kw: (calls.append(iss.iid), None)[1],
        )

        mon._check_issues(project)

        assert calls == [], "WIP-saturated developer must not pick up new issues"

    def test_issue_without_ready_label_is_skipped(self, monkeypatch):
        issue = FakeIssue(iid=23, title="Add feature", description="please")
        project = FakeProject(issues=[issue])
        mon = _build_monitor(project)

        calls: list[int] = []
        monkeypatch.setattr(
            mon, "_run_issue_processing",
            lambda iss, *a, **kw: (calls.append(iss.iid), None)[1],
        )

        mon._check_issues(project)

        assert calls == [], "Issue without 'ready' label must not be processed"


# ---------------------------------------------------------------------------
# _run_issue_processing — already-implemented auto-close
# ---------------------------------------------------------------------------

class _StubExecutionResult:
    def __init__(self, response: str, commit_sha: str = "", needs_continuation: bool = False) -> None:
        self.response = response
        self.commit_sha = commit_sha
        self.needs_continuation = needs_continuation


class _StubEngine:
    """Drop-in for CodeExecutionEngine; returns a preset result from .run()."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs) -> None:
        type(self).last_kwargs = kwargs

    def run(self, *, router=None):
        return type(self)._result


def _patch_engine(monkeypatch, result: _StubExecutionResult) -> None:
    _StubEngine._result = result
    import factory.core.monitor as monitor_mod
    monkeypatch.setattr(monitor_mod, "CodeExecutionEngine", _StubEngine)


class TestIssueAutoCloseAlreadyImplemented:
    def test_already_implemented_response_closes_issue(self, monkeypatch):
        issue = FakeIssue(iid=40, title="Add foo", description="please add foo", labels=["ready"])
        project = FakeProject(issues=[issue])
        mon = _build_monitor(project)
        _patch_engine(monkeypatch, _StubExecutionResult(
            response="Already implemented: foo() exists in module bar since v1.2",
            commit_sha="",
            needs_continuation=False,
        ))

        developer = SimpleNamespace(name="developer", system_prompt="", display_name="Developer")
        mon._run_issue_processing(issue, developer, architect=None)

        assert issue.state_event == "close", "issue must be closed when already implemented"
        assert "already-implemented" in issue.labels
        assert "processed" in issue.labels
        assert issue.saved == 1

    def test_no_commits_without_already_implemented_does_not_close(self, monkeypatch):
        issue = FakeIssue(iid=41, labels=["ready"])
        project = FakeProject(issues=[issue])
        mon = _build_monitor(project)
        _patch_engine(monkeypatch, _StubExecutionResult(
            response="Looked at the code but didn't change anything.",
            commit_sha="",
            needs_continuation=False,
        ))

        developer = SimpleNamespace(name="developer", system_prompt="", display_name="Developer")
        mon._run_issue_processing(issue, developer, architect=None)

        assert issue.state_event is None, "must not close without 'Already implemented' marker"
        assert "already-implemented" not in issue.labels
        assert "processed" in issue.labels

    def test_commits_present_does_not_close(self, monkeypatch):
        issue = FakeIssue(iid=42, labels=["ready"])
        project = FakeProject(issues=[issue])
        mon = _build_monitor(project)
        _patch_engine(monkeypatch, _StubExecutionResult(
            response="Already implemented: xyz",  # phrase present but commits exist
            commit_sha="abc1234",
            needs_continuation=False,
        ))

        # _create_mr_for_issue would call the real GitLab; stub it to a no-op.
        monkeypatch.setattr(mon, "_create_mr_for_issue", lambda **_kw: None)

        developer = SimpleNamespace(name="developer", system_prompt="", display_name="Developer")
        mon._run_issue_processing(issue, developer, architect=None)

        assert issue.state_event is None, "MR-creating runs must not auto-close"


# ---------------------------------------------------------------------------
# _run_refinement failure path
# ---------------------------------------------------------------------------

class TestRunRefinementFailurePath:
    """Test the failure handling in _run_refinement method."""

    def test_run_refinement_handles_exception_gracefully(self, monkeypatch):
        """Test that when _run_refinement raises, the refinement loop breaks
        and monitoring continues gracefully."""
        mr = FakeMR(iid=50, notes=[
            f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n⚠️ REQUEST CHANGES\n\nFix X.",
        ])
        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        # Track calls to the monkeypatched refinement
        refinement_calls = []

        def failing_refine(mr, *a, **kw):
            refinement_calls.append(mr.iid)
            raise Exception("Simulated refinement failure")

        monkeypatch.setattr(mon, "_run_refinement", failing_refine)
        monkeypatch.setattr(mon, "_run_review", lambda *a, **kw: None)

        # Should not raise, but log the error and break the loop
        mon._check_mrs(project)

        # _run_refinement is called once; the exception breaks the loop,
        # preventing further refinement iterations for this cycle.
        assert refinement_calls == [50]

        # Verify MR is still in opened state (not merged or closed)
        assert mr.state == "opened"

    def test_run_refinement_transient_error_breaks_loop(self, monkeypatch):
        """Test that a transient error from _run_refinement breaks the
        refinement loop for this cycle.

        When _run_refinement is monkeypatched to raise ConnectionError,
        its internal retry logic is bypassed. The exception propagates
        to _check_mrs, which catches it and breaks the loop. The MR
        will be retried on the next poll cycle.
        """
        mr = FakeMR(iid=51, notes=[
            f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n⚠️ REQUEST CHANGES\n\nFix X.",
        ])
        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        # Track refinement attempts
        refinement_attempts = []

        def transient_failing_refine(mr, *a, **kw):
            refinement_attempts.append(len(refinement_attempts))
            raise ConnectionError("Simulated transient network error")

        monkeypatch.setattr(mon, "_run_refinement", transient_failing_refine)
        monkeypatch.setattr(mon, "_run_review", lambda *a, **kw: None)

        mon._check_mrs(project)

        # The exception breaks the refinement loop immediately;
        # only one attempt is made within _check_mrs.
        assert len(refinement_attempts) == 1
        assert refinement_attempts == [0]

    def test_run_refinement_handles_non_retryable_error(self, monkeypatch):
        """Test that a non-retryable error from _run_refinement breaks the
        refinement loop for this cycle."""
        mr = FakeMR(iid=52, notes=[
            f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n⚠️ REQUEST CHANGES\n\nFix X.",
        ])
        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        # Track refinement attempts
        refinement_attempts = []

        def non_retryable_failing_refine(mr, *a, **kw):
            refinement_attempts.append(len(refinement_attempts))
            # Non-transient error (e.g., validation error)
            raise ValueError("Invalid input data")

        monkeypatch.setattr(mon, "_run_refinement", non_retryable_failing_refine)
        monkeypatch.setattr(mon, "_run_review", lambda *a, **kw: None)

        # Should not raise, but handle the error gracefully
        mon._check_mrs(project)

        # The exception breaks the refinement loop; only one attempt.
        assert len(refinement_attempts) == 1
        assert refinement_attempts == [0]

    def test_run_refinement_successful_execution(self, monkeypatch):
        """Test that a successful refinement exits the loop after one iteration
        when the re-review approves the MR."""
        mr = FakeMR(iid=53, notes=[
            f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n⚠️ REQUEST CHANGES\n\nFix X.",
        ])
        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        refinement_calls = []

        def successful_refine(mr, *a, **kw):
            refinement_calls.append(mr.iid)
            # Simulate the developer pushing a fix; the re-review will approve.
            mr.notes.create({"body": f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n✅ APPROVE"})

        monkeypatch.setattr(mon, "_run_refinement", successful_refine)
        monkeypatch.setattr(mon, "_run_review", lambda *a, **kw: None)

        mon._check_mrs(project)

        # Should have called refinement once, then the loop exits because
        # the re-reviewed notes now say APPROVE.
        assert refinement_calls == [53]

    def test_run_refinement_failure_in_check_issues_handled_gracefully(self, monkeypatch):
        """Test that when _run_refinement raises in _check_issues, the
        exception is caught and monitoring continues to the next issue."""
        mr = FakeMR(iid=99, source_branch="issue-60", notes=[
            f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n⚠️ REQUEST CHANGES\n\nfix",
        ])
        issue = FakeIssue(iid=60, labels=["ready"])
        project = FakeProject(mrs=[mr], issues=[issue])
        mon = _build_monitor(project)

        refinement_calls = []

        def failing_refine(mr, *a, **kw):
            refinement_calls.append(mr.iid)
            raise Exception("Simulated refinement failure in check_issues")

        monkeypatch.setattr(mon, "_run_refinement", failing_refine)
        monkeypatch.setattr(mon, "_run_review", lambda *a, **kw: None)

        mon._check_issues(project)

        # Refinement was attempted once before the exception
        assert refinement_calls == [99]
        # Issue remains opened (not closed or merged)
        assert issue.state == "opened"
        # Issue is marked as processed so it won't be retried immediately
        assert mon._last_processed_issues.get(60) is not None