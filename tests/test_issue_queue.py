"""Unit tests for IssueQueue and BacklogManager."""

from __future__ import annotations

import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
import yaml

from factory.core.issue_queue import (
    IssueQueue,
    Priority,
    QueueEntry,
    QueueStatus,
    QUEUE_STATUS_LABELS,
    ALL_FACTORY_LABELS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_queue(tmp_path: Path, gl=None) -> IssueQueue:
    """Return a fresh IssueQueue backed by a temp file."""
    return IssueQueue(queue_path=tmp_path / "queue.yml", gl=gl)

def _make_mock_gl(issue_labels: list[str] | None = None):
    """Return a minimal GitLabClient mock.

    ``project.issues.get`` uses a side_effect so the returned mock's ``iid``
    reflects the IID passed in, making assertions on ``issue.iid`` reliable
    across tests that use different IIDs.  The same mock object is returned
    each time so tests can inspect ``issue.labels`` directly.
    """
    issue = MagicMock()
    issue.iid = 1
    issue.labels = list(issue_labels or [])
    issue.notes = MagicMock()
    issue.notes.create.return_value = MagicMock()

    def _get_issue(iid):
        issue.iid = iid  # reflect the requested IID on the shared mock
        return issue

    project = MagicMock()
    project.issues.get.side_effect = _get_issue

    gl = MagicMock()
    gl.project = project
    return gl, issue


# ---------------------------------------------------------------------------
# QueueEntry serialisation
# ---------------------------------------------------------------------------

class TestQueueEntrySerialisation:
    def test_to_dict_roundtrip(self):
        entry = QueueEntry(
            issue_iid=42,
            title="Build widget",
            agent="developer",
            status=QueueStatus.QUEUED,
            priority=Priority.HIGH,
            labels=["feature", "high"],
        )
        d = entry.to_dict()
        restored = QueueEntry.from_dict(d)

        assert restored.issue_iid == 42
        assert restored.title == "Build widget"
        assert restored.agent == "developer"
        assert restored.status == QueueStatus.QUEUED
        assert restored.priority == Priority.HIGH
        assert restored.labels == ["feature", "high"]

    def test_from_dict_with_timestamps(self):
        now = datetime.now(timezone.utc)
        d = {
            "issue_iid": 7,
            "title": "Fix bug",
            "agent": "reviewer",
            "status": "in_progress",
            "priority": 2,
            "labels": ["bug"],
            "enqueued_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "started_at": now.isoformat(),
            "completed_at": None,
            "error": None,
        }
        entry = QueueEntry.from_dict(d)
        assert entry.status == QueueStatus.IN_PROGRESS
        assert entry.started_at is not None

    def test_from_dict_minimal(self):
        """from_dict handles missing optional keys gracefully."""
        entry = QueueEntry.from_dict({"issue_iid": 1, "title": "T", "agent": "qa"})
        assert entry.status == QueueStatus.QUEUED
        assert entry.priority == Priority.UNKNOWN


# ---------------------------------------------------------------------------
# Priority inference
# ---------------------------------------------------------------------------

class TestPriorityInference:
    @pytest.mark.parametrize(
        "labels,expected",
        [
            (["critical", "feature"], Priority.CRITICAL),
            (["blocker"], Priority.CRITICAL),
            (["high", "backend"], Priority.HIGH),
            (["urgent"], Priority.HIGH),
            (["medium"], Priority.MEDIUM),
            (["normal"], Priority.MEDIUM),
            (["low"], Priority.LOW),
            (["nice-to-have"], Priority.LOW),
            (["feature", "backend"], Priority.UNKNOWN),  # no priority label
            ([], Priority.UNKNOWN),
        ],
    )
    def test_infer_priority(self, labels, expected):
        assert IssueQueue._infer_priority(labels) == expected

    def test_highest_priority_wins(self):
        # Both high and low present — should pick high (lower int)
        priority = IssueQueue._infer_priority(["low", "critical", "medium"])
        assert priority == Priority.CRITICAL


# ---------------------------------------------------------------------------
# Enqueue
# ---------------------------------------------------------------------------

class TestEnqueue:
    def test_enqueue_creates_entry(self, tmp_path):
        q = _make_queue(tmp_path)
        entry = q.enqueue(10, "My issue", "developer", ["feature"])

        assert entry.issue_iid == 10
        assert entry.title == "My issue"
        assert entry.agent == "developer"
        assert entry.status == QueueStatus.QUEUED
        assert entry.labels == ["feature"]

    def test_enqueue_infers_priority(self, tmp_path):
        q = _make_queue(tmp_path)
        entry = q.enqueue(11, "Critical fix", "developer", ["bug", "critical"])
        assert entry.priority == Priority.CRITICAL

    def test_enqueue_does_not_reset_in_progress(self, tmp_path):
        q = _make_queue(tmp_path)
        q.enqueue(20, "Running issue", "developer", ["feature"])
        q.dequeue("developer")  # moves to IN_PROGRESS

        # Re-enqueue should be a no-op
        entry = q.enqueue(20, "Running issue", "developer", ["feature"])
        assert entry.status == QueueStatus.IN_PROGRESS

    def test_enqueue_does_not_reset_queued(self, tmp_path):
        q = _make_queue(tmp_path)
        q.enqueue(21, "Issue", "developer", ["feature"])
        entry2 = q.enqueue(21, "Issue", "developer", ["feature", "high"])
        # Still QUEUED, labels NOT updated (idempotent)
        assert entry2.status == QueueStatus.QUEUED

    def test_enqueue_requeues_done_issue(self, tmp_path):
        q = _make_queue(tmp_path)
        q.enqueue(30, "Completed issue", "developer", [])
        q.dequeue("developer")
        q.mark_done(30)

        # Now re-enqueue → should become QUEUED again
        entry = q.enqueue(30, "Completed issue", "developer", ["feature"])
        assert entry.status == QueueStatus.QUEUED

    def test_enqueue_requeues_error_issue(self, tmp_path):
        q = _make_queue(tmp_path)
        q.enqueue(31, "Failed issue", "developer", [])
        q.dequeue("developer")
        q.mark_error(31, "Something went wrong")

        entry = q.enqueue(31, "Failed issue", "developer", [])
        assert entry.status == QueueStatus.QUEUED
        assert entry.error is None

    def test_enqueue_updates_gitlab_labels(self, tmp_path):
        gl, issue = _make_mock_gl(issue_labels=["feature"])
        q = _make_queue(tmp_path, gl=gl)
        q.enqueue(1, "GL issue", "developer", ["feature"])

        # Label should be updated to factory-queued
        assert "factory-queued" in issue.labels
        issue.save.assert_called()

    def test_enqueue_posts_gitlab_comment(self, tmp_path):
        gl, issue = _make_mock_gl()
        q = _make_queue(tmp_path, gl=gl)
        q.enqueue(1, "GL issue", "developer", [])

        issue.notes.create.assert_called_once()
        comment_body = issue.notes.create.call_args[0][0]["body"]
        assert "added to the factory queue" in comment_body.lower() or "queue" in comment_body.lower()

    def test_enqueue_no_gl_does_not_raise(self, tmp_path):
        q = _make_queue(tmp_path, gl=None)
        entry = q.enqueue(99, "No-GL issue", "qa", [])
        assert entry.issue_iid == 99

    def test_requeue_triggers_gitlab_label_and_requeue_comment(self, tmp_path):
        """Re-queuing a terminal issue must update labels and post a re-queue comment."""
        gl, issue = _make_mock_gl()
        q = _make_queue(tmp_path, gl=gl)
        q.enqueue(5, "Issue", "developer", [])
        q.dequeue("developer")
        q.mark_done(5)

        issue.notes.create.reset_mock()
        issue.save.reset_mock()

        q.enqueue(5, "Issue", "developer", [])

        issue.save.assert_called()
        assert "factory-queued" in issue.labels
        comment_body = issue.notes.create.call_args[0][0]["body"]
        assert "re-queue" in comment_body.lower() or "re-queued" in comment_body.lower()


# ---------------------------------------------------------------------------
# Dequeue
# ---------------------------------------------------------------------------

class TestDequeue:
    def test_dequeue_returns_highest_priority(self, tmp_path):
        q = _make_queue(tmp_path)
        q.enqueue(1, "Low priority", "developer", ["low"])
        q.enqueue(2, "High priority", "developer", ["high"])
        q.enqueue(3, "Medium priority", "developer", ["medium"])

        entry = q.dequeue("developer")
        assert entry is not None
        assert entry.issue_iid == 2  # high priority

    def test_dequeue_fifo_within_same_priority(self, tmp_path):
        q = _make_queue(tmp_path)
        q.enqueue(1, "First", "developer", ["medium"])
        q.enqueue(2, "Second", "developer", ["medium"])

        entry = q.dequeue("developer")
        assert entry.issue_iid == 1

    def test_dequeue_marks_in_progress(self, tmp_path):
        q = _make_queue(tmp_path)
        q.enqueue(5, "My issue", "developer", [])
        entry = q.dequeue("developer")

        assert entry.status == QueueStatus.IN_PROGRESS
        assert entry.started_at is not None

    def test_dequeue_returns_none_when_empty(self, tmp_path):
        q = _make_queue(tmp_path)
        assert q.dequeue("developer") is None

    def test_dequeue_filters_by_agent(self, tmp_path):
        q = _make_queue(tmp_path)
        q.enqueue(10, "Dev issue", "developer", [])
        q.enqueue(11, "QA issue", "qa", [])

        entry = q.dequeue("qa")
        assert entry is not None
        assert entry.issue_iid == 11

        # Developer queue should still contain issue 10
        assert q.get(10).status == QueueStatus.QUEUED

    def test_dequeue_updates_gitlab_label(self, tmp_path):
        gl, issue = _make_mock_gl(issue_labels=["factory-queued", "feature"])
        q = _make_queue(tmp_path, gl=gl)
        q.enqueue(1, "GL issue", "developer", ["feature"])

        # Reset call count after enqueue
        issue.save.reset_mock()
        issue.notes.create.reset_mock()

        q.dequeue("developer")

        assert "factory-in-progress" in issue.labels
        assert "factory-queued" not in issue.labels

    def test_dequeue_posts_started_comment(self, tmp_path):
        gl, issue = _make_mock_gl()
        q = _make_queue(tmp_path, gl=gl)
        q.enqueue(1, "GL issue", "developer", [])
        issue.notes.create.reset_mock()

        q.dequeue("developer")

        issue.notes.create.assert_called_once()
        body = issue.notes.create.call_args[0][0]["body"]
        assert "developer" in body.lower() or "processing" in body.lower() or "started" in body.lower()


# ---------------------------------------------------------------------------
# Prioritize
# ---------------------------------------------------------------------------

class TestPrioritize:
    def test_prioritize_returns_queued_only(self, tmp_path):
        q = _make_queue(tmp_path)
        q.enqueue(1, "A", "developer", ["high"])
        q.enqueue(2, "B", "developer", ["low"])
        q.enqueue(3, "C", "developer", [])
        q.dequeue("developer")  # moves issue 1 (highest priority) to in_progress

        prioritized = q.prioritize()
        iids = [e.issue_iid for e in prioritized]
        assert 1 not in iids  # in_progress — not returned
        assert 2 in iids
        assert 3 in iids

    def test_prioritize_orders_highest_first(self, tmp_path):
        q = _make_queue(tmp_path)
        q.enqueue(1, "Low", "developer", ["low"])
        q.enqueue(2, "Critical", "developer", ["critical"])
        q.enqueue(3, "Medium", "developer", ["medium"])

        prioritized = q.prioritize()
        assert prioritized[0].issue_iid == 2  # critical
        assert prioritized[1].issue_iid == 3  # medium
        assert prioritized[2].issue_iid == 1  # low

    def test_prioritize_does_not_mutate_state(self, tmp_path):
        q = _make_queue(tmp_path)
        q.enqueue(1, "Issue", "developer", [])
        q.prioritize()
        assert q.get(1).status == QueueStatus.QUEUED


# ---------------------------------------------------------------------------
# list_queue
# ---------------------------------------------------------------------------

class TestListQueue:
    def test_list_queue_returns_all_statuses(self, tmp_path):
        q = _make_queue(tmp_path)
        q.enqueue(1, "A", "developer", [])
        q.enqueue(2, "B", "developer", [])
        q.dequeue("developer")
        q.mark_done(1)
        q.enqueue(3, "C", "developer", [])
        q.mark_blocked(3, "waiting")

        entries = q.list_queue()
        statuses = {e.status for e in entries}
        assert QueueStatus.DONE in statuses
        assert QueueStatus.QUEUED in statuses
        assert QueueStatus.BLOCKED in statuses

    def test_list_queue_sorted_by_priority(self, tmp_path):
        q = _make_queue(tmp_path)
        q.enqueue(1, "Low", "developer", ["low"])
        q.enqueue(2, "High", "developer", ["high"])
        q.enqueue(3, "Critical", "developer", ["critical"])

        entries = q.list_queue()
        assert entries[0].issue_iid == 3  # critical
        assert entries[1].issue_iid == 2  # high
        assert entries[2].issue_iid == 1  # low

    def test_list_queue_empty_when_no_issues(self, tmp_path):
        q = _make_queue(tmp_path)
        assert q.list_queue() == []


# ---------------------------------------------------------------------------
# mark_done / mark_error / mark_blocked
# ---------------------------------------------------------------------------

class TestStatusTransitions:
    def test_mark_done_sets_status(self, tmp_path):
        q = _make_queue(tmp_path)
        q.enqueue(1, "Issue", "developer", [])
        q.dequeue("developer")
        entry = q.mark_done(1)

        assert entry is not None
        assert entry.status == QueueStatus.DONE
        assert entry.completed_at is not None

    def test_mark_error_sets_status_and_error(self, tmp_path):
        q = _make_queue(tmp_path)
        q.enqueue(1, "Issue", "developer", [])
        q.dequeue("developer")
        entry = q.mark_error(1, "Timeout error")

        assert entry.status == QueueStatus.ERROR
        assert entry.error == "Timeout error"

    def test_mark_blocked_sets_status_and_reason(self, tmp_path):
        q = _make_queue(tmp_path)
        q.enqueue(1, "Issue", "developer", [])
        entry = q.mark_blocked(1, "Out of budget")

        assert entry.status == QueueStatus.BLOCKED
        assert entry.blocked_reason == "Out of budget"
        assert entry.error is None  # error field is for ERROR status only

    def test_mark_done_returns_none_for_unknown_iid(self, tmp_path):
        q = _make_queue(tmp_path)
        assert q.mark_done(9999) is None

    def test_mark_error_returns_none_for_unknown_iid(self, tmp_path):
        q = _make_queue(tmp_path)
        assert q.mark_error(9999, "err") is None

    def test_mark_blocked_returns_none_for_unknown_iid(self, tmp_path):
        q = _make_queue(tmp_path)
        assert q.mark_blocked(9999) is None

    def test_mark_done_applies_gitlab_label(self, tmp_path):
        gl, issue = _make_mock_gl(issue_labels=["factory-in-progress"])
        q = _make_queue(tmp_path, gl=gl)
        q.enqueue(1, "GL issue", "developer", [])
        q.dequeue("developer")
        issue.save.reset_mock()

        q.mark_done(1)

        assert "factory-done" in issue.labels
        assert "factory-in-progress" not in issue.labels

    def test_mark_error_applies_gitlab_label(self, tmp_path):
        gl, issue = _make_mock_gl(issue_labels=["factory-in-progress"])
        q = _make_queue(tmp_path, gl=gl)
        q.enqueue(1, "GL issue", "developer", [])
        q.dequeue("developer")
        issue.save.reset_mock()

        q.mark_error(1, "boom")

        assert "factory-error" in issue.labels

    def test_mark_blocked_applies_gitlab_label(self, tmp_path):
        gl, issue = _make_mock_gl(issue_labels=["factory-queued"])
        q = _make_queue(tmp_path, gl=gl)
        q.enqueue(1, "GL issue", "developer", [])
        issue.save.reset_mock()

        q.mark_blocked(1, "budget exhausted")

        assert "factory-blocked" in issue.labels

    def test_mark_done_posts_comment(self, tmp_path):
        gl, issue = _make_mock_gl()
        q = _make_queue(tmp_path, gl=gl)
        q.enqueue(1, "GL issue", "developer", [])
        q.dequeue("developer")
        issue.notes.create.reset_mock()

        q.mark_done(1)

        issue.notes.create.assert_called_once()
        body = issue.notes.create.call_args[0][0]["body"]
        assert "successfully" in body.lower() or "done" in body.lower() or "processed" in body.lower()

    def test_mark_error_posts_comment(self, tmp_path):
        gl, issue = _make_mock_gl()
        q = _make_queue(tmp_path, gl=gl)
        q.enqueue(1, "GL issue", "developer", [])
        q.dequeue("developer")
        issue.notes.create.reset_mock()

        q.mark_error(1, "Something broke")

        issue.notes.create.assert_called_once()
        body = issue.notes.create.call_args[0][0]["body"]
        assert "error" in body.lower() or "failed" in body.lower()


# ---------------------------------------------------------------------------
# get / remove
# ---------------------------------------------------------------------------

class TestGetAndRemove:
    def test_get_returns_entry(self, tmp_path):
        q = _make_queue(tmp_path)
        q.enqueue(42, "Check me", "developer", [])
        entry = q.get(42)
        assert entry is not None
        assert entry.issue_iid == 42

    def test_get_returns_none_for_unknown(self, tmp_path):
        q = _make_queue(tmp_path)
        assert q.get(999) is None

    def test_remove_deletes_entry(self, tmp_path):
        q = _make_queue(tmp_path)
        q.enqueue(1, "Disposable", "developer", [])
        removed = q.remove(1)
        assert removed is True
        assert q.get(1) is None

    def test_remove_returns_false_for_unknown(self, tmp_path):
        q = _make_queue(tmp_path)
        assert q.remove(888) is False


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_queue_survives_restart(self, tmp_path):
        queue_path = tmp_path / "queue.yml"

        q1 = IssueQueue(queue_path=queue_path)
        q1.enqueue(1, "Persist me", "developer", ["high"])
        q1.enqueue(2, "Also me", "qa", ["low"])

        # Simulate restart by loading a new instance
        q2 = IssueQueue(queue_path=queue_path)
        assert q2.get(1) is not None
        assert q2.get(1).priority == Priority.HIGH
        assert q2.get(2) is not None
        assert q2.get(2).agent == "qa"

    def test_status_persisted(self, tmp_path):
        queue_path = tmp_path / "queue.yml"

        q1 = IssueQueue(queue_path=queue_path)
        q1.enqueue(10, "Status issue", "developer", [])
        q1.dequeue("developer")
        q1.mark_done(10)

        q2 = IssueQueue(queue_path=queue_path)
        assert q2.get(10).status == QueueStatus.DONE

    def test_persisted_file_is_valid_yaml(self, tmp_path):
        queue_path = tmp_path / "queue.yml"

        q = IssueQueue(queue_path=queue_path)
        q.enqueue(1, "Issue", "developer", ["feature"])

        with open(queue_path) as f:
            data = yaml.safe_load(f)

        assert data is not None
        assert "entries" in data
        assert data["version"] == 1
        assert len(data["entries"]) == 1

    def test_loads_empty_on_missing_file(self, tmp_path):
        queue_path = tmp_path / "nonexistent.yml"
        q = IssueQueue(queue_path=queue_path)
        assert q.list_queue() == []

    def test_loads_empty_on_corrupt_file(self, tmp_path):
        queue_path = tmp_path / "queue.yml"
        queue_path.write_text("<<<not valid yaml>>>: [")
        q = IssueQueue(queue_path=queue_path)
        assert q.list_queue() == []


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------
# Note: these tests validate the logical correctness of the locking strategy
# under the Python GIL.  The GIL means true low-level atomicity is not tested
# here; the tests are meaningful for pure-Python code where RLock prevents
# logical races (e.g., check-then-act sequences) regardless of the GIL.
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_enqueue_is_safe(self, tmp_path):
        q = _make_queue(tmp_path)
        errors: list[Exception] = []

        def enqueue_many(start: int) -> None:
            for i in range(start, start + 20):
                try:
                    q.enqueue(i, f"Issue {i}", "developer", [])
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

        threads = [threading.Thread(target=enqueue_many, args=(i * 20,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # 5 threads × 20 issues = 100 distinct issues
        assert len(q.list_queue()) == 100

    def test_concurrent_dequeue_gives_unique_issues(self, tmp_path):
        q = _make_queue(tmp_path)
        for i in range(10):
            q.enqueue(i, f"Issue {i}", "developer", [])

        claimed: list[int] = []
        claimed_lock = threading.Lock()

        def worker() -> None:
            entry = q.dequeue("developer")
            if entry:
                with claimed_lock:
                    claimed.append(entry.issue_iid)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No duplicates
        assert len(claimed) == len(set(claimed))


# ---------------------------------------------------------------------------
# GitLab label cleanup
# ---------------------------------------------------------------------------

class TestGitLabLabelCleanup:
    def test_enqueue_removes_other_factory_labels(self, tmp_path):
        """Enqueuing should strip all other factory labels and add factory-queued."""
        gl, issue = _make_mock_gl(issue_labels=["factory-error", "factory-done", "feature"])
        q = _make_queue(tmp_path, gl=gl)
        q.enqueue(1, "Re-queue", "developer", ["feature"])

        # Only factory-queued should remain from factory labels
        factory_labels_on_issue = ALL_FACTORY_LABELS.intersection(set(issue.labels))
        assert factory_labels_on_issue == {"factory-queued"}

    def test_all_factory_labels_defined(self):
        """Ensure every QueueStatus except BACKLOG maps to a non-empty label."""
        for status in QueueStatus:
            if status == QueueStatus.BACKLOG:
                assert QUEUE_STATUS_LABELS[status] == ""
            else:
                assert QUEUE_STATUS_LABELS[status] != "", f"{status} has no label"


# ---------------------------------------------------------------------------
# Queue status label constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_queue_status_values(self):
        assert QueueStatus.QUEUED.value == "queued"
        assert QueueStatus.IN_PROGRESS.value == "in_progress"
        assert QueueStatus.BLOCKED.value == "blocked"
        assert QueueStatus.DONE.value == "done"
        assert QueueStatus.ERROR.value == "error"

    def test_priority_ordering(self):
        assert Priority.CRITICAL < Priority.HIGH
        assert Priority.HIGH < Priority.MEDIUM
        assert Priority.MEDIUM < Priority.LOW
        assert Priority.LOW < Priority.UNKNOWN

    def test_factory_labels_complete(self):
        expected = {"factory-queued", "factory-in-progress", "factory-blocked", "factory-done", "factory-error"}
        assert ALL_FACTORY_LABELS == expected


# ---------------------------------------------------------------------------
# GitLab API failure handling
# ---------------------------------------------------------------------------

class TestGitLabAPIFailures:
    """Test that queue operations remain functional when GitLab API calls fail."""

    def test_enqueue_succeeds_when_gitlab_raises(self, tmp_path):
        """Enqueue should complete even if GitLab API calls fail."""
        gl, issue = _make_mock_gl()
        # Make GitLab API calls raise an exception
        issue.save.side_effect = Exception("API unavailable")
        issue.notes.create.side_effect = Exception("API unavailable")

        q = _make_queue(tmp_path, gl=gl)
        entry = q.enqueue(1, "Issue", "developer", ["feature"])

        # Queue operation should succeed despite API failure
        assert entry is not None
        assert entry.issue_iid == 1
        assert entry.status == QueueStatus.QUEUED

    def test_dequeue_succeeds_when_gitlab_raises(self, tmp_path):
        """Dequeue should complete even if GitLab API calls fail."""
        gl, issue = _make_mock_gl()
        issue.save.side_effect = Exception("API unavailable")
        issue.notes.create.side_effect = Exception("API unavailable")

        q = _make_queue(tmp_path, gl=gl)
        q.enqueue(1, "Issue", "developer", [])
        entry = q.dequeue("developer")

        # Dequeue should succeed despite API failure
        assert entry is not None
        assert entry.issue_iid == 1
        assert entry.status == QueueStatus.IN_PROGRESS

    def test_mark_done_succeeds_when_gitlab_raises(self, tmp_path):
        """Marking done should complete even if GitLab API calls fail."""
        gl, issue = _make_mock_gl()
        issue.save.side_effect = Exception("API unavailable")
        issue.notes.create.side_effect = Exception("API unavailable")

        q = _make_queue(tmp_path, gl=gl)
        q.enqueue(1, "Issue", "developer", [])
        q.dequeue("developer")
        entry = q.mark_done(1)

        # Mark done should succeed despite API failure
        assert entry is not None
        assert entry.status == QueueStatus.DONE

    def test_mark_error_succeeds_when_gitlab_raises(self, tmp_path):
        """Marking error should complete even if GitLab API calls fail."""
        gl, issue = _make_mock_gl()
        issue.save.side_effect = Exception("API unavailable")
        issue.notes.create.side_effect = Exception("API unavailable")

        q = _make_queue(tmp_path, gl=gl)
        q.enqueue(1, "Issue", "developer", [])
        q.dequeue("developer")
        entry = q.mark_error(1, "Something broke")

        # Mark error should succeed despite API failure
        assert entry is not None
        assert entry.status == QueueStatus.ERROR
        assert entry.error == "Something broke"

    def test_mark_blocked_succeeds_when_gitlab_raises(self, tmp_path):
        """Marking blocked should complete even if GitLab API calls fail."""
        gl, issue = _make_mock_gl()
        issue.save.side_effect = Exception("API unavailable")
        issue.notes.create.side_effect = Exception("API unavailable")

        q = _make_queue(tmp_path, gl=gl)
        q.enqueue(1, "Issue", "developer", [])
        entry = q.mark_blocked(1, "Budget exhausted")

        # Mark blocked should succeed despite API failure
        assert entry is not None
        assert entry.status == QueueStatus.BLOCKED
        assert entry.blocked_reason == "Budget exhausted"

    def test_enqueue_when_issues_get_raises(self, tmp_path):
        """Enqueue should handle failure to fetch issue from GitLab."""
        gl = MagicMock()
        project = MagicMock()
        project.issues.get.side_effect = Exception("Issue not found")
        gl.project = project

        q = _make_queue(tmp_path, gl=gl)
        entry = q.enqueue(1, "Issue", "developer", ["feature"])

        # Should still succeed - queue state is updated
        assert entry is not None
        assert entry.issue_iid == 1
        assert entry.status == QueueStatus.QUEUED

    def test_dequeue_when_issues_get_raises(self, tmp_path):
        """Dequeue should handle failure to fetch issue from GitLab."""
        gl = MagicMock()
        project = MagicMock()
        project.issues.get.side_effect = Exception("Issue not found")
        gl.project = project

        q = _make_queue(tmp_path, gl=gl)
        q.enqueue(1, "Issue", "developer", [])
        entry = q.dequeue("developer")

        # Should still succeed - queue state is updated
        assert entry is not None
        assert entry.issue_iid == 1
        assert entry.status == QueueStatus.IN_PROGRESS

    def test_queue_operations_log_warnings_on_api_failure(self, tmp_path, caplog):
        """API failures should be logged as warnings."""
        gl, issue = _make_mock_gl()
        issue.save.side_effect = Exception("API unavailable")
        issue.notes.create.side_effect = Exception("API unavailable")

        q = _make_queue(tmp_path, gl=gl)
        q.enqueue(1, "Issue", "developer", [])

        # Should have logged a warning about the API failure
        assert any("Failed to update GitLab status" in record.message for record in caplog.records)
        assert any(record.levelname == "WARNING" for record in caplog.records)