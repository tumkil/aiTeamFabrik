from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from ruamel.yaml import YAML

from factory.core.events import SprintEndEvent


@dataclass
class SprintState:
    number: int
    name: str
    start_date: date
    duration_days: int
    label_in_progress: str
    label_review: str
    label_done: str
    milestone_prefix: str

    @property
    def end_date(self) -> date:
        return self.start_date + timedelta(days=self.duration_days)

    @property
    def days_remaining(self) -> int:
        return max(0, (self.end_date - date.today()).days)

    @property
    def is_over(self) -> bool:
        return date.today() >= self.end_date

    def to_sprint_end_event(self) -> SprintEndEvent:
        """Create a SprintEndEvent from this sprint state."""
        return SprintEndEvent(
            sprint_number=self.number,
            sprint_name=self.name,
            end_date=self.end_date.isoformat(),
            message=f"Sprint {self.number} ({self.name}) ended on {self.end_date}",
        )


class ScrumEngine:
    def __init__(self, config_path: str = "config/factory.yml") -> None:
        yaml = YAML()
        with open(config_path) as f:
            cfg = yaml.load(f)
        self._config_path = config_path
        sc = cfg.get("sprint", {})
        labels = sc.get("labels", {})
        self._state = SprintState(
            number=sc.get("current", 1),
            name=sc.get("name", "Sprint 1"),
            start_date=date.fromisoformat(sc.get("start_date", str(date.today()))),
            duration_days=sc.get("duration_days", 14),
            label_in_progress=labels.get("in_progress", "In Progress"),
            label_review=labels.get("review", "In Review"),
            label_done=labels.get("done", "Done"),
            milestone_prefix=sc.get("milestone_prefix", "Sprint"),
        )

    @property
    def current(self) -> SprintState:
        return self._state

    def detect_sprint_end(self) -> Optional[SprintEndEvent]:
        """Check if current sprint has ended and return event if so.
        
        Returns:
            SprintEndEvent if sprint has ended, None otherwise.
        """
        if self._state.is_over:
            return self._state.to_sprint_end_event()
        return None

    def velocity(self, closed: int, total: int) -> int:
        """Returns sprint completion percentage."""
        return int((closed / total) * 100) if total else 0

    def advance(self) -> None:
        """Bump sprint number in factory.yml for next sprint planning."""
        yaml = YAML()
        yaml.preserve_quotes = True
        with open(self._config_path) as f:
            cfg = yaml.load(f)

        next_num = self._state.number + 1
        next_start = self._state.end_date
        cfg["sprint"]["current"] = next_num
        cfg["sprint"]["name"] = f"Sprint {next_num}"
        cfg["sprint"]["start_date"] = next_start.isoformat()

        with open(self._config_path, "w") as f:
            yaml.dump(cfg, f)

        self._state = SprintState(
            number=next_num,
            name=f"Sprint {next_num}",
            start_date=next_start,
            duration_days=self._state.duration_days,
            label_in_progress=self._state.label_in_progress,
            label_review=self._state.label_review,
            label_done=self._state.label_done,
            milestone_prefix=self._state.milestone_prefix,
        )
