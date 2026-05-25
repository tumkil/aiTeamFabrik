"""Token-Aware Planner for SoftwareTeamFabrik.

Creates optimal schedules for issue processing based on token cost estimates
and available budgets.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml
from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)

# File lock timeout in seconds
_LOCK_TIMEOUT = 10

# Maximum description multiplier cap to prevent absurd token estimates
# from auto-generated or very long descriptions.
_DESC_MULTIPLIER_CAP = 5.0


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PlanningAlgorithm(str, Enum):
    """Available planning algorithms."""
    GREEDY = "greedy"          # Select highest priority affordable issues
    OPTIMAL = "optimal"        # Optimize for maximum value within budget (greedy by value density)
    FAIR = "fair"              # Distribute work evenly across agents


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IssueEstimate:
    """Token cost estimate for a single issue."""
    issue_iid: int
    title: str
    labels: List[str]
    base_cost: int
    description_multiplier: float
    agent_factor: float
    total_tokens: int
    priority: int
    agent: str = "developer"

    def to_dict(self) -> dict:
        return {
            "issue_iid": self.issue_iid,
            "title": self.title,
            "labels": list(self.labels),
            "base_cost": self.base_cost,
            "description_multiplier": self.description_multiplier,
            "agent_factor": self.agent_factor,
            "total_tokens": self.total_tokens,
            "priority": self.priority,
            "agent": self.agent,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IssueEstimate":
        return cls(
            issue_iid=data["issue_iid"],
            title=data["title"],
            labels=data.get("labels", []),
            base_cost=data["base_cost"],
            description_multiplier=data["description_multiplier"],
            agent_factor=data["agent_factor"],
            total_tokens=data["total_tokens"],
            priority=data["priority"],
            agent=data.get("agent", "developer"),
        )


@dataclass
class PlanEntry:
    """A single entry in the execution plan."""
    issue_iid: int
    title: str
    agent: str
    estimated_tokens: int
    priority: int
    scheduled_at: datetime
    status: str = "pending"
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "issue_iid": self.issue_iid,
            "title": self.title,
            "agent": self.agent,
            "estimated_tokens": self.estimated_tokens,
            "priority": self.priority,
            "scheduled_at": self.scheduled_at.isoformat(),
            "status": self.status,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlanEntry":
        def _dt(val: Optional[str]) -> Optional[datetime]:
            if not val:
                return None
            dt = datetime.fromisoformat(val)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt

        return cls(
            issue_iid=data["issue_iid"],
            title=data["title"],
            agent=data["agent"],
            estimated_tokens=data["estimated_tokens"],
            priority=data["priority"],
            scheduled_at=_dt(data["scheduled_at"]) or datetime.now(timezone.utc),
            status=data.get("status", "pending"),
            completed_at=_dt(data.get("completed_at")),
        )


@dataclass
class SprintPlan:
    """Complete plan for a sprint."""
    sprint_number: str
    algorithm: PlanningAlgorithm
    budget: int
    entries: List[PlanEntry] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "sprint_number": self.sprint_number,
            "algorithm": self.algorithm.value,
            "budget": self.budget,
            "entries": [e.to_dict() for e in self.entries],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SprintPlan":
        def _dt(val: str) -> datetime:
            dt = datetime.fromisoformat(val)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt

        return cls(
            sprint_number=data["sprint_number"],
            algorithm=PlanningAlgorithm(data["algorithm"]),
            budget=data["budget"],
            entries=[PlanEntry.from_dict(e) for e in data.get("entries", [])],
            created_at=_dt(data["created_at"]),
            updated_at=_dt(data["updated_at"]),
        )


# ---------------------------------------------------------------------------
# Token-Aware Planner
# ---------------------------------------------------------------------------

class TokenAwarePlanner:
    """Token-Aware Planner for creating optimal issue processing schedules.

    Responsibilities:
    - Estimate token costs for issues
    - Create execution plans based on available budget
    - Track plan progress and update as issues are processed
    - Support multiple planning algorithms (Greedy, Optimal, Fair)
    """

    # Base token costs by label
    LABEL_COSTS = {
        'feature': 50000,
        'bug': 25000,
        'architecture': 100000,
    }

    # Default base cost for unknown labels
    DEFAULT_BASE_COST = 30000

    # Agent-specific factors
    AGENT_FACTORS = {
        'architect': 1.2,
        'developer': 1.0,
        'reviewer': 0.8,
        'default': 1.0,  # Changed to match developer factor as default
    }

    def __init__(self, plan_path: Path):
        """Initialize the planner.

        Parameters
        ----------
        plan_path:
            Path to the YAML file used for persistence.
        """
        self.plan_path = Path(plan_path)
        self._lock = threading.RLock()
        self._file_lock = FileLock(str(self.plan_path) + ".lock")
        self._plan: Optional[SprintPlan] = None

        # Load existing plan if it exists
        self._load()

    # -----------------------------------------------------------------------
    # Token Estimation
    # -----------------------------------------------------------------------

    def estimate_issue_cost(self, issue_iid: int, title: str, labels: List[str],
                            description: str, agent: str = "developer") -> IssueEstimate:
        """Estimate token cost for a single issue.

        Uses the formula:
        total_tokens = base_cost * description_multiplier * agent_factor * 1.2

        The description multiplier is capped at ``_DESC_MULTIPLIER_CAP`` (5.0)
        to prevent absurdly large estimates for very long descriptions.

        Parameters
        ----------
        issue_iid:
            GitLab issue IID.
        title:
            Issue title.
        labels:
            List of issue labels.
        description:
            Issue description text.
        agent:
            Name of the agent that will process the issue.

        Returns
        -------
        IssueEstimate
            Token cost estimate for the issue.
        """
        # Coerce issue_iid to int to handle potential string inputs
        issue_iid = int(issue_iid)

        # Calculate base cost from labels - only consider known labels
        known_costs = [self.LABEL_COSTS[label.lower()] 
                       for label in labels if label.lower() in self.LABEL_COSTS]
        base = max(known_costs) if known_costs else self.DEFAULT_BASE_COST

        # Description complexity multiplier — capped to avoid runaway estimates
        desc_multiplier = min(1 + (len(description) / 2000), _DESC_MULTIPLIER_CAP)

        # Agent factor
        agent_factor = self.AGENT_FACTORS.get(agent.lower(), self.AGENT_FACTORS["default"])

        # Total tokens with safety margin
        total_tokens = int(base * desc_multiplier * agent_factor * 1.2)

        # Priority inference (lower number = higher priority)
        priority_labels = {'critical': 1, 'blocker': 1, 'high': 2, 'urgent': 2,
                           'medium': 3, 'normal': 3, 'low': 4, 'nice-to-have': 4}
        priority = min([priority_labels.get(label.lower(), 5) for label in labels], default=5)

        return IssueEstimate(
            issue_iid=issue_iid,
            title=title,
            labels=labels,
            base_cost=base,
            description_multiplier=desc_multiplier,
            agent_factor=agent_factor,
            total_tokens=total_tokens,
            priority=priority,
            agent=agent,
        )

    # -----------------------------------------------------------------------
    # Planning Algorithms
    # -----------------------------------------------------------------------

    def create_plan(self, sprint_number: str, budget: int,
                    issues: List[Dict[str, Any]],
                    algorithm: PlanningAlgorithm = PlanningAlgorithm.GREEDY) -> SprintPlan:
        """Create an execution plan for the given sprint.

        Parameters
        ----------
        sprint_number:
            Sprint identifier (e.g., "sprint-3").
        budget:
            Total token budget for the sprint.
        issues:
            List of issues to plan, each as a dict with keys:
            - issue_iid: int
            - title: str
            - labels: List[str]
            - description: str
            - agent: str (optional, default: "developer")
        algorithm:
            Planning algorithm to use.

        Returns
        -------
        SprintPlan
            The generated execution plan.
        """
        with self._lock:
            # Validate budget
            if budget <= 0:
                raise ValueError(f"budget must be positive, got {budget}")

            # Estimate costs for all issues
            estimates = []
            for issue in issues:
                estimate = self.estimate_issue_cost(
                    issue_iid=issue["issue_iid"],
                    title=issue["title"],
                    labels=issue["labels"],
                    description=issue["description"],
                    agent=issue.get("agent", "developer")
                )
                estimates.append(estimate)

            # Apply planning algorithm
            if algorithm == PlanningAlgorithm.GREEDY:
                plan_entries = self._greedy_plan(estimates, budget)
            elif algorithm == PlanningAlgorithm.OPTIMAL:
                plan_entries = self._optimal_plan(estimates, budget)
            elif algorithm == PlanningAlgorithm.FAIR:
                plan_entries = self._fair_plan(estimates, budget)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            # Create and save the plan
            self._plan = SprintPlan(
                sprint_number=sprint_number,
                algorithm=algorithm,
                budget=budget,
                entries=plan_entries,
            )
            self._save()

            return self._plan

    def _greedy_plan(self, estimates: List[IssueEstimate], budget: int) -> List[PlanEntry]:
        """Greedy algorithm: select highest priority affordable issues until budget is exhausted.
        
        Issues are sorted by priority (ascending) and then by token cost (ascending).
        The algorithm iterates through all issues, selecting those that fit within the
        remaining budget. This continues even after encountering issues that exceed the
        budget, as later issues might still be affordable.
        """
        # Sort by priority (ascending), then by token cost (ascending)
        sorted_estimates = sorted(estimates, key=lambda e: (e.priority, e.total_tokens))

        plan_entries = []
        remaining_budget = budget

        for estimate in sorted_estimates:
            if estimate.total_tokens <= remaining_budget:
                entry = PlanEntry(
                    issue_iid=estimate.issue_iid,
                    title=estimate.title,
                    agent=estimate.agent,
                    estimated_tokens=estimate.total_tokens,
                    priority=estimate.priority,
                    scheduled_at=datetime.now(timezone.utc),
                )
                plan_entries.append(entry)
                remaining_budget -= estimate.total_tokens
            else:
                logger.debug(
                    "Issue #%s (%s) exceeds remaining budget (%d tokens needed, %d remaining)",
                    estimate.issue_iid, estimate.title, estimate.total_tokens, remaining_budget,
                )

        return plan_entries

    def _optimal_plan(self, estimates: List[IssueEstimate], budget: int) -> List[PlanEntry]:
        """Optimal algorithm: maximize value (priority-weighted) within budget using knapsack approach.
        
        This is a simplified knapsack approximation using a greedy approach by value density.
        For a true optimal solution, dynamic programming would be needed, but this provides
        a good approximation for most practical cases.
        """
        # Sort by value density: (priority_weight / tokens) descending
        # Higher priority (lower number) = higher weight
        priority_weight = {1: 10, 2: 5, 3: 3, 4: 1, 5: 0.5}

        def value_density(estimate: IssueEstimate) -> float:
            weight = priority_weight.get(estimate.priority, 1)
            return weight / estimate.total_tokens if estimate.total_tokens > 0 else 0

        sorted_estimates = sorted(estimates, key=lambda e: -value_density(e))

        plan_entries = []
        remaining_budget = budget

        for estimate in sorted_estimates:
            if estimate.total_tokens <= remaining_budget:
                entry = PlanEntry(
                    issue_iid=estimate.issue_iid,
                    title=estimate.title,
                    agent=estimate.agent,
                    estimated_tokens=estimate.total_tokens,
                    priority=estimate.priority,
                    scheduled_at=datetime.now(timezone.utc),
                )
                plan_entries.append(entry)
                remaining_budget -= estimate.total_tokens

        return plan_entries

    def _fair_plan(self, estimates: List[IssueEstimate], budget: int) -> List[PlanEntry]:
        """Fair algorithm: distribute work evenly across different issue types.

        Round-robins through label groups.  Each group is pre-sorted by
        ``(priority, total_tokens)`` and traversed with an index pointer so
        that removing a scheduled or unaffordable issue is O(1) instead of
        the O(n) ``list.remove()`` used previously — bringing the overall
        complexity down from O(n²) to O(n log n).

        If a full pass over every group schedules nothing the loop exits to
        avoid spinning forever.
        """
        # Group by label type
        by_label: Dict[str, List[IssueEstimate]] = {}
        for estimate in estimates:
            # Use the first label that matches our cost model
            for label in estimate.labels:
                if label.lower() in self.LABEL_COSTS:
                    key = label.lower()
                    if key not in by_label:
                        by_label[key] = []
                    by_label[key].append(estimate)
                    break
            else:
                if "other" not in by_label:
                    by_label["other"] = []
                by_label["other"].append(estimate)

        # Pre-sort each group by (priority, total_tokens) ascending so that
        # the best candidate is always at the current index position.
        for key in by_label:
            by_label[key].sort(key=lambda e: (e.priority, e.total_tokens))

        # Index pointer for each group — avoids O(n) list.remove() inside
        # the round-robin loop (was O(n²) overall, now O(n log n)).
        group_idx: Dict[str, int] = {key: 0 for key in by_label}
        group_keys = list(by_label.keys())

        plan_entries: List[PlanEntry] = []
        remaining_budget = budget

        while remaining_budget > 0:
            scheduled_this_round = False

            for key in group_keys:
                group = by_label[key]
                idx = group_idx[key]

                # Fast-forward past items that can never be scheduled.
                # Because the remaining budget only decreases, any item
                # whose total_tokens already exceeds the budget will never
                # fit in a future round either.
                while idx < len(group) and group[idx].total_tokens > remaining_budget:
                    idx += 1
                group_idx[key] = idx

                if idx >= len(group):
                    continue

                estimate = group[idx]

                entry = PlanEntry(
                    issue_iid=estimate.issue_iid,
                    title=estimate.title,
                    agent=estimate.agent,
                    estimated_tokens=estimate.total_tokens,
                    priority=estimate.priority,
                    scheduled_at=datetime.now(timezone.utc),
                )
                plan_entries.append(entry)
                remaining_budget -= estimate.total_tokens
                group_idx[key] = idx + 1
                scheduled_this_round = True

                if remaining_budget <= 0:
                    break

            # If we completed a full pass without scheduling anything every
            # remaining issue is too expensive — stop to avoid an infinite loop.
            if not scheduled_this_round:
                break

        return plan_entries

    # -----------------------------------------------------------------------
    # Plan Management
    # -----------------------------------------------------------------------

    def next_issue(self, agent: str) -> Optional[PlanEntry]:
        """Get the next issue to process for the given agent.

        Parameters
        ----------
        agent:
            Name of the agent requesting work.

        Returns
        -------
        PlanEntry or None
            The next pending issue, or None if no issues are available.
        """
        with self._lock:
            if not self._plan:
                return None

            # Find the first pending issue that matches the requesting agent
            for entry in self._plan.entries:
                if entry.status == "pending" and entry.agent == agent:
                    # Mark as in progress and assign to the requesting agent
                    entry.status = "in_progress"
                    entry.agent = agent
                    self._plan.updated_at = datetime.now(timezone.utc)
                    self._save()
                    return entry

            return None

    def mark_completed(self, issue_iid: int) -> bool:
        """Mark an issue as completed.

        Parameters
        ----------
        issue_iid:
            GitLab issue IID.

        Returns
        -------
        bool
            True if the issue was found and marked completed, False otherwise.
            Returns False if the issue is not found, not in "in_progress" status,
            or if no plan exists.
        """
        with self._lock:
            if not self._plan:
                return False

            for entry in self._plan.entries:
                if entry.issue_iid == issue_iid and entry.status == "in_progress":
                    entry.status = "completed"
                    entry.completed_at = datetime.now(timezone.utc)
                    self._plan.updated_at = datetime.now(timezone.utc)
                    self._save()
                    return True

            return False

    def get_plan(self) -> Optional[SprintPlan]:
        """Get the current plan.

        Returns
        -------
        SprintPlan or None
            The current plan, or None if no plan exists.
        """
        with self._lock:
            return self._plan

    def plan_summary(self) -> Dict[str, Any]:
        """Generate a summary of the current plan.

        Returns
        -------
        dict
            Summary information including:
            - sprint_number
            - algorithm
            - budget
            - total_planned_tokens (includes all issues: pending, in_progress, completed)
            - pending_issues
            - in_progress_issues
            - completed_issues
        """
        with self._lock:
            if not self._plan:
                return {
                    "sprint_number": "none",
                    "algorithm": "none",
                    "budget": 0,
                    "total_planned_tokens": 0,
                    "pending_issues": 0,
                    "in_progress_issues": 0,
                    "completed_issues": 0,
                }

            pending = sum(1 for e in self._plan.entries if e.status == "pending")
            in_progress = sum(1 for e in self._plan.entries if e.status == "in_progress")
            completed = sum(1 for e in self._plan.entries if e.status == "completed")
            total_tokens = sum(e.estimated_tokens for e in self._plan.entries)

            return {
                "sprint_number": self._plan.sprint_number,
                "algorithm": self._plan.algorithm.value,
                "budget": self._plan.budget,
                "total_planned_tokens": total_tokens,
                "pending_issues": pending,
                "in_progress_issues": in_progress,
                "completed_issues": completed,
            }

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def _load(self) -> None:
        """Load plan from YAML file.
        
        This method is only safe to call during initialization before the object
        is shared across threads. For thread-safe access, use get_plan().
        """
        try:
            with self._file_lock.acquire(timeout=_LOCK_TIMEOUT):
                if self.plan_path.exists():
                    with open(self.plan_path) as f:
                        data = yaml.safe_load(f)
                    if data:
                        self._plan = SprintPlan.from_dict(data)
                        logger.info("Loaded plan for sprint %s from %s",
                                    self._plan.sprint_number, self.plan_path)
        except FileNotFoundError:
            logger.debug("Plan file %s does not exist — starting fresh", self.plan_path)
        except Timeout:
            logger.warning("Could not acquire file lock to load plan")
        except yaml.YAMLError as exc:
            logger.warning("Corrupt plan file (YAML parse error): %s", exc)
        except (KeyError, TypeError, ValueError) as exc:
            # yaml.safe_load succeeded but the data structure is unexpected
            # (e.g. missing required keys, wrong types, unknown enum value).
            logger.warning(
                "Corrupt plan file %s — could not deserialise plan, starting fresh: %s",
                self.plan_path, exc,
            )

    def _save(self) -> None:
        """Persist plan to YAML file.

        Caller must hold self._lock.
        """
        if not self._plan:
            return

        temp_path = self.plan_path.with_suffix(self.plan_path.suffix + ".tmp")
        try:
            self.plan_path.parent.mkdir(parents=True, exist_ok=True)

            data = self._plan.to_dict()

            with self._file_lock.acquire(timeout=_LOCK_TIMEOUT):
                with open(temp_path, "w") as f:
                    yaml.safe_dump(data, f, sort_keys=False)
                os.replace(temp_path, self.plan_path)

            logger.debug("Saved plan to %s", self.plan_path)
        except Timeout:
            logger.warning("Could not acquire file lock to save plan")
            temp_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.error("Failed to save plan: %s", exc)
            temp_path.unlink(missing_ok=True)


# Public API
__all__ = [
    "TokenAwarePlanner",
    "IssueEstimate",
    "PlanEntry",
    "SprintPlan",
    "PlanningAlgorithm",
    "_DESC_MULTIPLIER_CAP",
]