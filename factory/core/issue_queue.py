"""Issue Queue and Backlog Manager for SoftwareTeamFabrik.

Tracks issues that need processing, with prioritization logic based on
labels, status, and agent assignment. Supports persistence across restarts
and integrates with GitLab to update labels and post comments.
"""

from __future__ import annotations

import copy
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)

# File lock timeout in seconds
_LOCK_TIMEOUT = 10


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class QueueStatus(str, Enum):
    """Status of an issue in the factory queue."""
    BACKLOG = "backlog"          # Known but not yet enqueued
    QUEUED = "queued"            # In queue, waiting for available agent/budget
    IN_PROGRESS = "in_progress"  # Agent is actively working on it
    BLOCKED = "blocked"          # Waiting for dependency or budget
    DONE = "done"                # Successfully processed
    ERROR = "error"              # Processing failed


class Priority(int, Enum):
    """Numeric priority — lower value = higher urgency."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    UNKNOWN = 5


# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------

# GitLab labels applied for each queue status
QUEUE_STATUS_LABELS: dict[QueueStatus, str] = {
    QueueStatus.BACKLOG: "",               # no factory label for backlog
    QueueStatus.QUEUED: "factory-queued",
    QueueStatus.IN_PROGRESS: "factory-in-progress",
    QueueStatus.BLOCKED: "factory-blocked",
    QueueStatus.DONE: "factory-done",
    QueueStatus.ERROR: "factory-error",
}

# All factory-managed labels (used for cleanup).
# Derived programmatically so a new QUEUE_STATUS_LABELS entry is automatically
# included in the cleanup set without requiring a separate manual update.
# Public (no underscore) so tests can import it directly.
ALL_FACTORY_LABELS: set[str] = {v for v in QUEUE_STATUS_LABELS.values() if v}

# Priority inferred from issue labels
_LABEL_PRIORITY: dict[str, Priority] = {
    "critical": Priority.CRITICAL,
    "blocker": Priority.CRITICAL,
    "high": Priority.HIGH,
    "urgent": Priority.HIGH,
    "medium": Priority.MEDIUM,
    "normal": Priority.MEDIUM,
    "low": Priority.LOW,
    "nice-to-have": Priority.LOW,
}


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime.

    Replaces the deprecated ``datetime.utcnow()`` (removed in Python 3.14)
    with the recommended ``datetime.now(timezone.utc)``.
    """
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QueueEntry:
    """A single entry in the issue queue."""
    issue_iid: int
    title: str
    agent: str                         # agent name (e.g. "developer")
    status: QueueStatus = QueueStatus.QUEUED
    priority: Priority = Priority.UNKNOWN
    labels: list[str] = field(default_factory=list)
    enqueued_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None          # failure message when status=ERROR
    blocked_reason: Optional[str] = None # hold reason when status=BLOCKED

    # -----------------------------------------------------------------------
    # Serialisation helpers
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "issue_iid": self.issue_iid,
            "title": self.title,
            "agent": self.agent,
            "status": self.status.value,
            "priority": self.priority.value,
            "labels": list(self.labels),
            "enqueued_at": self.enqueued_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "blocked_reason": self.blocked_reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QueueEntry":
        def _dt(val: Optional[str]) -> Optional[datetime]:
            if not val:
                return None
            dt = datetime.fromisoformat(val)
            # Ensure the datetime is timezone-aware (older persisted entries may
            # be stored without tzinfo).
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt

        return cls(
            issue_iid=data["issue_iid"],
            title=data.get("title", ""),
            agent=data.get("agent", "developer"),
            status=QueueStatus(data.get("status", QueueStatus.QUEUED.value)),
            priority=Priority(data.get("priority", Priority.UNKNOWN.value)),
            labels=data.get("labels", []),
            enqueued_at=_dt(data.get("enqueued_at")) or _utcnow(),
            updated_at=_dt(data.get("updated_at")) or _utcnow(),
            started_at=_dt(data.get("started_at")),
            completed_at=_dt(data.get("completed_at")),
            error=data.get("error"),
            blocked_reason=data.get("blocked_reason"),
        )


# ---------------------------------------------------------------------------
# IssueQueue
# ---------------------------------------------------------------------------

class IssueQueue:
    """Queue of GitLab issues awaiting processing.

    Responsibilities
    ----------------
    - Enqueue / dequeue issues with priority ordering.
    - Track queue status and persist state to YAML.
    - Update GitLab labels and post comments on status changes (when a
      GitLabClient is provided).

    Thread safety
    -------------
    An in-process ``threading.RLock`` guards all mutations.  A ``FileLock``
    ensures cross-process safety when reading / writing the persistence file.

    Lock ordering
    -------------
    The canonical order is ``_file_lock`` first, then ``_lock``.

    ``_save`` is called *inside* ``_lock`` and then acquires ``_file_lock``,
    which appears to invert the order.  This is safe because ``_load`` is
    called **only from** ``__init__``, before the queue is ever shared across
    threads, so ``_load`` (``_file_lock`` → ``_lock``) and ``_save``
    (``_lock`` → ``_file_lock``) can never run concurrently.  An assertion in
    ``_load`` enforces this invariant; do not call ``_load`` after construction.

    ``filelock.FileLock`` uses an in-process ``threading.Lock`` internally, so
    the deadlock analysis above applies to threading, not just cross-process
    scenarios.

    Parameters
    ----------
    queue_path:
        Path to the YAML file used for persistence.  The directory is created
        automatically if it does not exist.
    gl:
        Optional ``GitLabClient`` instance.  When supplied, status changes
        are reflected on the GitLab issue (labels + comments).
    """

    def __init__(
        self,
        queue_path: Path,
        gl=None,  # Optional[GitLabClient] — avoid circular imports
    ) -> None:
        self._queue_path = Path(queue_path)
        self._gl = gl
        self._lock = threading.RLock()
        self._file_lock = FileLock(str(self._queue_path) + ".lock")

        # In-memory store: issue_iid -> QueueEntry
        self._entries: dict[int, QueueEntry] = {}

        # Guards against _load() being called after construction (see Lock ordering).
        self._initialized = False
        self._load()
        self._initialized = True

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def enqueue(self, issue_iid: int, title: str, agent: str, labels: list[str]) -> QueueEntry:
        """Add an issue to the queue (or update if already present).

        Active issues (QUEUED / IN_PROGRESS) are left untouched so that a
        re-scan does not reset in-flight work.  BLOCKED issues follow the
        same rule — they are a hold state, not terminal, so they will not be
        re-queued automatically; to unblock, first transition the issue to a
        terminal state (DONE / ERROR) or call :meth:`remove`, then re-enqueue.
        Terminal issues (DONE / ERROR) are re-queued.

        Parameters
        ----------
        issue_iid:
            GitLab issue IID.
        title:
            Issue title (for display / persistence).
        agent:
            Name of the agent responsible for the issue.
        labels:
            Current GitLab labels of the issue (used for priority inference).

        Returns
        -------
        QueueEntry
            The (new or existing) entry for the issue.
        """
        with self._lock:
            existing = self._entries.get(issue_iid)
            if existing and existing.status in (
                QueueStatus.QUEUED,
                QueueStatus.IN_PROGRESS,
                QueueStatus.BLOCKED,
            ):
                logger.debug(
                    "Issue #%s already in queue with status %s — skipping",
                    issue_iid,
                    existing.status,
                )
                return copy.deepcopy(existing)

            priority = self._infer_priority(labels)
            is_requeue = existing is not None

            if is_requeue:
                # Re-queue a terminal (DONE / ERROR) entry; refresh metadata in case
                # the issue was renamed or reassigned since the first enqueue.
                existing.title = title
                existing.agent = agent
                existing.status = QueueStatus.QUEUED
                existing.priority = priority
                existing.labels = list(labels)
                existing.updated_at = _utcnow()
                existing.error = None
                existing.blocked_reason = None
                existing.started_at = None
                existing.completed_at = None
                entry = existing
            else:
                entry = QueueEntry(
                    issue_iid=issue_iid,
                    title=title,
                    agent=agent,
                    priority=priority,
                    labels=list(labels),
                    status=QueueStatus.QUEUED,
                )
                self._entries[issue_iid] = entry

            # Snapshot the values we need for GitLab calls *before* releasing
            # the lock so the side-effects are consistent with the state we
            # just wrote, even if another thread transitions the entry
            # concurrently after we release.
            iid_snapshot = entry.issue_iid
            result = copy.deepcopy(entry)

            self._save()

        # GitLab side-effects outside _lock to avoid blocking on network I/O
        # while holding the in-process lock.  The values (iid, target status)
        # were captured inside the lock so they are internally consistent.
        comment = (
            f":recycle: Issue #{iid_snapshot} has been **re-queued** by the factory."
            if is_requeue
            else f":inbox_tray: Issue #{iid_snapshot} has been added to the factory queue."
        )
        self._apply_gitlab_status(iid_snapshot, QueueStatus.QUEUED, comment)

        logger.info("Enqueued issue #%s (agent=%s, priority=%s)", issue_iid, agent, priority.name)
        return result

    def dequeue(self, agent: str) -> Optional[QueueEntry]:
        """Return the highest-priority queued issue for *agent* and mark it IN_PROGRESS.

        Returns ``None`` when there are no eligible issues.

        Parameters
        ----------
        agent:
            Name of the agent requesting work.
        """
        with self._lock:
            candidates = [
                e for e in self._entries.values()
                if e.agent == agent and e.status == QueueStatus.QUEUED
            ]
            if not candidates:
                return None

            # Sort by priority (ascending int = higher urgency), then enqueued_at (FIFO)
            candidates.sort(key=lambda e: (e.priority.value, e.enqueued_at))
            entry = candidates[0]

            entry.status = QueueStatus.IN_PROGRESS
            entry.started_at = _utcnow()
            entry.updated_at = _utcnow()

            # Snapshot iid and copy while the lock is held.
            iid_snapshot = entry.issue_iid
            result = copy.deepcopy(entry)

            self._save()

        self._apply_gitlab_status(
            iid_snapshot,
            QueueStatus.IN_PROGRESS,
            f":rocket: Agent **{agent}** has started processing issue #{iid_snapshot}.",
        )

        logger.info("Dequeued issue #%s for agent %s", iid_snapshot, agent)
        return result

    def prioritize(self) -> list[QueueEntry]:
        """Return all queued entries sorted by priority (highest first) for inspection.

        This does **not** mutate state; call :meth:`dequeue` to actually claim
        an issue.

        Returns
        -------
        list[QueueEntry]
            Issues in QUEUED status, sorted highest → lowest priority.
        """
        with self._lock:
            queued = [e for e in self._entries.values() if e.status == QueueStatus.QUEUED]
            queued.sort(key=lambda e: (e.priority.value, e.enqueued_at))
            return [copy.deepcopy(e) for e in queued]

    def list_queue(self) -> list[QueueEntry]:
        """Return all queue entries (all statuses), sorted by priority then enqueued time.

        Returns
        -------
        list[QueueEntry]
            Snapshot of every known entry.
        """
        with self._lock:
            entries = list(self._entries.values())
            entries.sort(key=lambda e: (e.priority.value, e.enqueued_at))
            return [copy.deepcopy(e) for e in entries]

    def mark_done(self, issue_iid: int) -> Optional[QueueEntry]:
        """Mark an issue as successfully completed.

        Parameters
        ----------
        issue_iid:
            GitLab issue IID.

        Returns
        -------
        QueueEntry or None
            The updated entry, or ``None`` if the IID is not tracked.
        """
        return self._set_terminal_status(issue_iid, QueueStatus.DONE, error=None)

    def mark_error(self, issue_iid: int, error: str) -> Optional[QueueEntry]:
        """Mark an issue as failed.

        Parameters
        ----------
        issue_iid:
            GitLab issue IID.
        error:
            Human-readable error description.

        Returns
        -------
        QueueEntry or None
        """
        return self._set_terminal_status(issue_iid, QueueStatus.ERROR, error=error)

    def mark_blocked(self, issue_iid: int, reason: str = "") -> Optional[QueueEntry]:
        """Mark an issue as blocked (e.g. waiting for budget or dependency).

        Parameters
        ----------
        issue_iid:
            GitLab issue IID.
        reason:
            Human-readable reason for the block.

        Returns
        -------
        QueueEntry or None

        Note
        ----
        There is no dedicated ``unblock()`` method.  To re-queue a blocked
        issue, call :meth:`remove` then :meth:`enqueue`.
        """
        with self._lock:
            entry = self._entries.get(issue_iid)
            if entry is None:
                return None
            # Guard against transitioning a terminal issue to BLOCKED.
            if entry.status in (QueueStatus.DONE, QueueStatus.ERROR):
                logger.warning(
                    "Cannot block terminal issue #%s (status=%s)", issue_iid, entry.status
                )
                return copy.deepcopy(entry)
            entry.status = QueueStatus.BLOCKED
            entry.updated_at = _utcnow()
            entry.blocked_reason = reason or None

            iid_snapshot = entry.issue_iid
            result = copy.deepcopy(entry)

            self._save()

        reason_text = f": {reason}" if reason else ""
        self._apply_gitlab_status(
            iid_snapshot,
            QueueStatus.BLOCKED,
            f":warning: Issue #{iid_snapshot} is **blocked**{reason_text}.",
        )
        return result

    def get(self, issue_iid: int) -> Optional[QueueEntry]:
        """Look up a single entry by issue IID. Returns a copy; do not mutate."""
        with self._lock:
            entry = self._entries.get(issue_iid)
            return copy.deepcopy(entry) if entry is not None else None

    def remove(self, issue_iid: int) -> bool:
        """Remove an entry from the queue entirely.

        Returns ``True`` if the entry existed and was removed.
        """
        with self._lock:
            if issue_iid in self._entries:
                del self._entries[issue_iid]
                self._save()
                return True
        return False

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _set_terminal_status(
        self,
        issue_iid: int,
        status: QueueStatus,
        error: Optional[str],
    ) -> Optional[QueueEntry]:
        with self._lock:
            entry = self._entries.get(issue_iid)
            if entry is None:
                return None
            entry.status = status
            entry.completed_at = _utcnow()
            entry.updated_at = _utcnow()
            entry.error = error

            # Snapshot iid and copy while the lock is held; `status` is a parameter.
            iid_snapshot = entry.issue_iid
            result = copy.deepcopy(entry)

            self._save()

        if status == QueueStatus.DONE:
            comment = f":white_check_mark: Issue #{iid_snapshot} has been **successfully processed** by the factory."
        else:
            err_text = f": {error}" if error else ""
            comment = f":x: Issue #{iid_snapshot} encountered an **error**{err_text}."
        self._apply_gitlab_status(iid_snapshot, status, comment)
        return result

    # -----------------------------------------------------------------------
    # Priority inference
    # -----------------------------------------------------------------------

    @staticmethod
    def _infer_priority(labels: list[str]) -> Priority:
        """Derive a :class:`Priority` from an issue's label list.

        Iterates label-to-priority mapping and returns the highest urgency
        (lowest int value) found.  Falls back to :attr:`Priority.UNKNOWN`.

        Note: labels are matched after ``.lower()``; only exact lowercase
        matches (e.g. ``"critical"``, ``"high"``) are recognised.  Mixed-case
        labels like ``"Critical"`` or ``"High Priority"`` are **not** matched.
        """
        best = Priority.UNKNOWN
        for label in labels:
            label_lower = label.lower()
            if label_lower in _LABEL_PRIORITY:
                candidate = _LABEL_PRIORITY[label_lower]
                if candidate.value < best.value:
                    best = candidate
        return best

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def _load(self) -> None:
        """Load queue state from YAML, initialising an empty store if absent.

        **Must only be called once, from** ``__init__``, before the queue is
        accessible from multiple threads.  Calling it after construction risks
        a ``_file_lock`` / ``_lock`` deadlock (see class Lock ordering docs).
        """
        assert not self._initialized, "_load() must only be called from __init__"
        # Hold _file_lock across both the file read and the dict population so no
        # other process can overwrite the file between the two operations.
        try:
            with self._file_lock.acquire(timeout=_LOCK_TIMEOUT):
                with open(self._queue_path) as f:
                    data = yaml.safe_load(f) or {}
                with self._lock:
                    for raw in data.get("entries", []):
                        try:
                            entry = QueueEntry.from_dict(raw)
                            self._entries[entry.issue_iid] = entry
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Skipping malformed queue entry %r: %s", raw, exc)
        except FileNotFoundError:
            logger.debug("Queue file %s does not exist — starting fresh", self._queue_path)
            return
        except Timeout:
            logger.warning("Could not acquire file lock to load queue; starting fresh")
            return
        except yaml.YAMLError as exc:
            logger.warning("Corrupt queue file, starting fresh: %s", exc)
            return

        logger.info("Loaded %d queue entries from %s", len(self._entries), self._queue_path)

    def _save(self) -> None:
        """Persist queue state to YAML.

        Caller **must** hold ``self._lock``.  Uses an atomic write (temp file
        + ``os.replace``) combined with ``FileLock`` for cross-process safety.
        """
        # _is_owned() is a CPython implementation detail; the assertion is a
        # dev-time guard only — it is skipped on non-CPython runtimes (AttributeError)
        # and when Python is run with -O (asserts are stripped).
        try:
            assert self._lock._is_owned(), "_save() must be called while holding self._lock"  # noqa: SLF001
        except AttributeError:
            pass

        self._queue_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._queue_path.with_suffix(self._queue_path.suffix + ".tmp")

        data = {
            "version": 1,
            "updated_at": _utcnow().isoformat(),
            "entries": [e.to_dict() for e in self._entries.values()],
        }

        try:
            with self._file_lock.acquire(timeout=_LOCK_TIMEOUT):
                try:
                    with open(temp_path, "w") as f:
                        yaml.safe_dump(data, f, sort_keys=False)
                    os.replace(temp_path, self._queue_path)
                except Exception:
                    temp_path.unlink(missing_ok=True)
                    raise
        except Timeout:
            # Clean up any partial write before the lock was acquired.
            temp_path.unlink(missing_ok=True)
            logger.warning("Could not acquire file lock to save queue; skipping persist")

    # -----------------------------------------------------------------------
    # GitLab integration
    # -----------------------------------------------------------------------

    def _apply_gitlab_status(
        self, issue_iid: int, status: QueueStatus, comment: str
    ) -> None:
        """Fetch the GitLab issue once, update its labels, and post a comment.

        Combining label and comment into a single method halves the number of
        ``project.issues.get`` API calls per status transition.

        No-ops silently when no :class:`GitLabClient` is configured, when
        *status* is ``BACKLOG`` (no label to apply), or when the API call fails.
        """
        if self._gl is None:
            return

        new_label = QUEUE_STATUS_LABELS.get(status, "")

        try:
            project = self._gl.project
            issue = project.issues.get(issue_iid)

            # Update labels only when there is a label to apply; BACKLOG has none.
            current = set(issue.labels)
            updated = current - ALL_FACTORY_LABELS
            if new_label:
                updated.add(new_label)
            if updated != current:
                issue.labels = list(updated)
                issue.save()
                logger.debug(
                    "Updated GitLab labels for issue #%s → %s",
                    issue_iid,
                    sorted(updated),
                )

            issue.notes.create({"body": comment})
            logger.debug("Posted comment on issue #%s", issue_iid)
        except Exception as exc:
            logger.warning(
                "Failed to update GitLab status for issue #%s: %s",
                issue_iid,
                exc,
                exc_info=True,
            )


# Public API
__all__ = [
    "IssueQueue",
    "QueueEntry",
    "QueueStatus",
    "Priority",
    "QUEUE_STATUS_LABELS",
    "ALL_FACTORY_LABELS",
]
