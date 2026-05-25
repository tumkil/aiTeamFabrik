"""Regression tests for BUG-03: poll de-duplication uses .seconds instead of .total_seconds().

`timedelta.seconds` only returns the seconds component within a single day (0-86399).
For intervals spanning more than one day, or when there are negative microseconds
that roll over, `.seconds` gives the wrong result.  `.total_seconds()` returns the
true elapsed seconds including days and fractional seconds.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from factory.core.monitor import FactoryMonitor

POLL_INTERVAL = 30  # default poll interval; matches FactoryMonitor default


# ---------------------------------------------------------------------------
# Helpers — build a monitor with enough stubs for _check_mrs / _check_issues.
# ---------------------------------------------------------------------------

def _build_monitor() -> FactoryMonitor:
    """Construct a FactoryMonitor without touching the real GitLab."""
    mon = FactoryMonitor.__new__(FactoryMonitor)
    mon._gl = SimpleNamespace(
        project=SimpleNamespace(
            default_branch="main",
            id=1,
            http_url_to_repo="https://example.invalid/repo.git",
        ),
        hostname="example.invalid",
        _url="https://example.invalid",
        create_merge_request=lambda **_: None,
    )
    mon._competence = SimpleNamespace(
        get=lambda _name: SimpleNamespace(
            name=_name, system_prompt="", display_name=_name,
        ),
    )
    mon._scrum = None
    mon._router = None
    mon._token = ""
    mon._budget_manager = SimpleNamespace(can_consume=lambda *_a, **_k: (True, ""))
    mon._poll_interval = POLL_INTERVAL
    mon._active_issues = {}
    mon._last_processed_mrs = {}
    mon._last_processed_issues = {}
    return mon


class FakeMR:
    def __init__(self, iid: int) -> None:
        self.iid = iid
        self.notes = SimpleNamespace(list=lambda **_: [])  # type: ignore[assignment]


class FakeIssue:
    def __init__(self, iid: int) -> None:
        self.iid = iid
        self.labels = ["ready"]
        self.title = "title"
        self.description = "desc"


class FakeProject:
    def __init__(self, mrs=None, issues=None) -> None:
        self.mergerequests = SimpleNamespace(
            list=lambda **_: mrs or []
        )
        self.issues = SimpleNamespace(
            list=lambda **_: issues or []
        )


# ---------------------------------------------------------------------------
# Regression: .seconds vs .total_seconds()
# ---------------------------------------------------------------------------

class TestDedupSecondsBug:
    """If the last-processed timestamp is more than a day old, .seconds
    wraps around and may incorrectly report a small elapsed time, causing
    the monitor to skip work items that should be re-evaluated."""

    def test_mr_reprocessed_after_long_interval(self, monkeypatch):
        """A gap of >1 day must not be treated as <POLL_INTERVAL."""
        mon = _build_monitor()
        # Mark processed 2 days ago
        mon._last_processed_mrs[1] = datetime.now() - timedelta(days=2)

        calls: list[int] = []
        monkeypatch.setattr(mon, "_run_review", lambda mr, *a, **kw: calls.append(mr.iid))
        monkeypatch.setattr(mon, "_run_refinement", lambda mr, *a, **kw: None)
        monkeypatch.setattr(mon, "_auto_merge", lambda mr: None)

        project = FakeProject(mrs=[FakeMR(1)])
        mon._check_mrs(project)

        assert calls == [1], (
            "MR processed 2 days ago should be re-evaluated, not skipped"
        )

    def test_mr_skipped_within_poll_interval(self, monkeypatch):
        """If the gap is smaller than POLL_INTERVAL, the MR must still be skipped."""
        mon = _build_monitor()
        mon._last_processed_mrs[1] = datetime.now() - timedelta(seconds=POLL_INTERVAL - 1)

        calls: list[int] = []
        monkeypatch.setattr(mon, "_run_review", lambda mr, *a, **kw: calls.append(mr.iid))
        monkeypatch.setattr(mon, "_run_refinement", lambda mr, *a, **kw: None)
        monkeypatch.setattr(mon, "_auto_merge", lambda mr: None)

        project = FakeProject(mrs=[FakeMR(1)])
        mon._check_mrs(project)

        assert calls == [], (
            "MR processed 1 second ago should still be skipped"
        )

    def test_issue_reprocessed_after_long_interval(self, monkeypatch):
        mon = _build_monitor()
        mon._last_processed_issues[10] = datetime.now() - timedelta(days=2)

        calls: list[int] = []
        monkeypatch.setattr(
            mon, "_run_issue_processing",
            lambda iss, *a, **kw: calls.append(iss.iid),
        )

        project = FakeProject(issues=[FakeIssue(10)])
        mon._check_issues(project)

        assert calls == [10], (
            "Issue processed 2 days ago should be re-evaluated, not skipped"
        )

    def test_issue_skipped_within_poll_interval(self, monkeypatch):
        mon = _build_monitor()
        mon._last_processed_issues[10] = datetime.now() - timedelta(seconds=POLL_INTERVAL - 1)

        calls: list[int] = []
        monkeypatch.setattr(
            mon, "_run_issue_processing",
            lambda iss, *a, **kw: calls.append(iss.iid),
        )

        project = FakeProject(issues=[FakeIssue(10)])
        mon._check_issues(project)

        assert calls == [], (
            "Issue processed 1 second ago should still be skipped"
        )
