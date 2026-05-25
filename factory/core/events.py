"""Event system for SoftwareTeamFabrik."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class SprintEndEvent:
    """Emitted when a sprint end date is detected."""
    sprint_number: int
    sprint_name: str
    end_date: str
    detected_at: datetime = field(default_factory=datetime.now)
    message: str = ""
    
    def __str__(self) -> str:
        return (
            f"SprintEndEvent(sprint={self.sprint_number} "
            f"name={self.sprint_name} end_date={self.end_date})"
        )


@dataclass
class TaskSpawnedEvent:
    """Emitted when an agent task is spawned."""
    run_id: str
    issue_iid: int
    agent_name: str
    spawned_at: datetime = field(default_factory=datetime.now)


@dataclass
class TaskCompletedEvent:
    """Emitted when an agent task completes."""
    run_id: str
    issue_iid: int
    agent_name: str
    success: bool
    completed_at: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


@dataclass
class MergeRequestApprovedEvent:
    """Emitted when a merge request is approved."""
    mr_iid: int
    mr_title: str
    project_id: int
    approved_at: datetime = field(default_factory=datetime.now)
    approver_id: Optional[int] = None
    
    def __str__(self) -> str:
        return (
            f"MergeRequestApprovedEvent(mr_iid={self.mr_iid} "
            f"title={self.mr_title} project_id={self.project_id})"
        )