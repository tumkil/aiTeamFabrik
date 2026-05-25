"""Unit tests for TokenAwarePlanner."""
import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime, timezone
import yaml

from factory.core.planner import (
    TokenAwarePlanner,
    IssueEstimate,
    PlanEntry,
    SprintPlan,
    PlanningAlgorithm,
    _DESC_MULTIPLIER_CAP,
)


def test_issue_estimate():
    """Test issue token estimation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.yml"
        planner = TokenAwarePlanner(plan_path)

        # Test with feature label
        estimate = planner.estimate_issue_cost(
            issue_iid=1,
            title="Test Feature",
            labels=["feature"],
            description="This is a test feature description.",
            agent="developer"
        )

        assert estimate.issue_iid == 1
        assert estimate.title == "Test Feature"
        assert estimate.labels == ["feature"]
        assert estimate.base_cost == 50000  # feature base cost
        assert estimate.description_multiplier > 1
        assert estimate.agent_factor == 1.0  # developer factor
        assert estimate.total_tokens > 0
        assert estimate.priority == 5  # default priority (no priority label)

        # Test with labels that affect priority.
        # "critical" is NOT in LABEL_COSTS, so it should be ignored for base cost.
        # "bug" maps to 25000, so base cost should be 25000.
        assert "critical" not in TokenAwarePlanner.LABEL_COSTS  # confirm assumption
        estimate = planner.estimate_issue_cost(
            issue_iid=2,
            title="Critical Bug",
            labels=["bug", "critical"],
            description="This is a critical bug.",
            agent="architect"
        )
        assert estimate.base_cost == 25000  # bug base cost (known label takes precedence)
        assert estimate.priority == 1  # critical priority label
        assert estimate.agent_factor == 1.2  # architect factor


def test_greedy_planning():
    """Test greedy planning algorithm."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.yml"
        planner = TokenAwarePlanner(plan_path)

        issues = [
            {
                "issue_iid": 1,
                "title": "High Priority Feature",
                "labels": ["feature", "high"],
                "description": "Important feature.",
                "agent": "developer"
            },
            {
                "issue_iid": 2,
                "title": "Low Priority Bug",
                "labels": ["bug", "low"],
                "description": "Minor bug.",
                "agent": "developer"
            },
            {
                "issue_iid": 3,
                "title": "Critical Bug",
                "labels": ["bug", "critical"],
                "description": "Very important.",
                "agent": "developer"
            }
        ]

        plan = planner.create_plan(
            sprint_number="sprint-1",
            budget=500000,
            issues=issues,
            algorithm=PlanningAlgorithm.GREEDY
        )

        assert plan.sprint_number == "sprint-1"
        assert plan.algorithm == PlanningAlgorithm.GREEDY
        assert plan.budget == 500000
        assert len(plan.entries) > 0

        # Check that entries are sorted by priority
        priorities = [entry.priority for entry in plan.entries]
        assert priorities == sorted(priorities)


def test_optimal_planning():
    """Test optimal planning algorithm."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.yml"
        planner = TokenAwarePlanner(plan_path)

        issues = [
            {
                "issue_iid": 1,
                "title": "High Value Feature",
                "labels": ["feature", "high"],
                "description": "Short description.",
                "agent": "developer"
            },
            {
                "issue_iid": 2,
                "title": "Low Value Bug",
                "labels": ["bug", "low"],
                "description": "Very long description. " * 100,  # Long description = higher cost
                "agent": "developer"
            }
        ]

        plan = planner.create_plan(
            sprint_number="sprint-1",
            budget=500000,
            issues=issues,
            algorithm=PlanningAlgorithm.OPTIMAL
        )

        assert plan.algorithm == PlanningAlgorithm.OPTIMAL
        assert len(plan.entries) >= 1


def test_fair_planning():
    """Test fair planning algorithm."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.yml"
        planner = TokenAwarePlanner(plan_path)

        issues = [
            {
                "issue_iid": 1,
                "title": "Feature 1",
                "labels": ["feature"],
                "description": "Feature description.",
                "agent": "developer"
            },
            {
                "issue_iid": 2,
                "title": "Feature 2",
                "labels": ["feature"],
                "description": "Another feature.",
                "agent": "developer"
            },
            {
                "issue_iid": 3,
                "title": "Bug 1",
                "labels": ["bug"],
                "description": "Bug description.",
                "agent": "developer"
            },
            {
                "issue_iid": 4,
                "title": "Architecture Issue",
                "labels": ["architecture"],
                "description": "Architecture work.",
                "agent": "architect"
            }
        ]

        plan = planner.create_plan(
            sprint_number="sprint-1",
            budget=1000000,
            issues=issues,
            algorithm=PlanningAlgorithm.FAIR
        )

        assert plan.algorithm == PlanningAlgorithm.FAIR
        assert len(plan.entries) >= 1


def test_fair_planning_no_infinite_loop():
    """Test that _fair_plan terminates when no remaining issue fits the budget."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.yml"
        planner = TokenAwarePlanner(plan_path)

        # Each architecture issue costs at least 100000 * 1 * 1.2 * 1.2 = 144000 tokens
        # with a 1-char description.  Use a budget that is too small for any of them.
        issues = [
            {
                "issue_iid": i,
                "title": f"Arch Issue {i}",
                "labels": ["architecture"],
                "description": "x",
                "agent": "developer",
            }
            for i in range(1, 4)
        ]

        # Budget smaller than the cheapest possible estimate — loop must not hang
        plan = planner.create_plan(
            sprint_number="sprint-1",
            budget=1000,  # Tiny — nothing can be scheduled
            issues=issues,
            algorithm=PlanningAlgorithm.FAIR,
        )
        assert plan.entries == []


def test_fair_planning_mixed_groups_no_infinite_loop():
    """Test _fair_plan with mixed affordable/unaffordable groups does not spin forever.

    One group (bug) has issues that fit; another group (architecture) has issues
    that never fit.  The architecture issues must be evicted on first evaluation
    so the loop terminates in O(n) rather than O(n²) passes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.yml"
        planner = TokenAwarePlanner(plan_path)

        # Architecture min cost ≈ 144000 — will never fit in 50000 budget.
        # Bug min cost ≈ 30000 — fits once.
        issues = [
            {
                "issue_iid": 1,
                "title": "Cheap Bug",
                "labels": ["bug"],
                "description": "Short.",
                "agent": "developer",
            },
            {
                "issue_iid": 2,
                "title": "Expensive Arch",
                "labels": ["architecture"],
                "description": "x",
                "agent": "architect",
            },
        ]

        plan = planner.create_plan(
            sprint_number="sprint-1",
            budget=50000,
            issues=issues,
            algorithm=PlanningAlgorithm.FAIR,
        )

        # The bug should be scheduled; the architecture issue should not.
        iids = [e.issue_iid for e in plan.entries]
        assert 1 in iids
        assert 2 not in iids
        total = sum(e.estimated_tokens for e in plan.entries)
        assert total <= 50000


def test_fair_planning_multiple_unaffordable_in_group():
    """Test that _fair_plan removes ALL unaffordable issues from a group, not just the cheapest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.yml"
        planner = TokenAwarePlanner(plan_path)

        # Create a group with multiple expensive architecture issues
        # and one group with affordable bugs
        issues = [
            {
                "issue_iid": 1,
                "title": "Cheap Bug 1",
                "labels": ["bug"],
                "description": "Short.",
                "agent": "developer",
            },
            {
                "issue_iid": 2,
                "title": "Cheap Bug 2",
                "labels": ["bug"],
                "description": "Short.",
                "agent": "developer",
            },
            {
                "issue_iid": 3,
                "title": "Expensive Arch 1",
                "labels": ["architecture"],
                "description": "x",
                "agent": "architect",
            },
            {
                "issue_iid": 4,
                "title": "Expensive Arch 2",
                "labels": ["architecture"],
                "description": "x",
                "agent": "architect",
            },
        ]

        plan = planner.create_plan(
            sprint_number="sprint-1",
            budget=50000,  # Can fit 1-2 bugs but no architecture issues
            issues=issues,
            algorithm=PlanningAlgorithm.FAIR,
        )

        # Only bugs should be scheduled, no architecture issues
        iids = [e.issue_iid for e in plan.entries]
        assert 1 in iids or 2 in iids  # At least one bug scheduled
        assert 3 not in iids  # No architecture issues
        assert 4 not in iids
        
        # Verify the loop terminated correctly (no infinite loop)
        assert len(plan.entries) > 0


def test_plan_persistence():
    """Test that plans are persisted and loaded correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.yml"

        # Create first planner and make a plan
        planner1 = TokenAwarePlanner(plan_path)
        issues = [
            {
                "issue_iid": 1,
                "title": "Test Issue",
                "labels": ["feature"],
                "description": "Test description.",
                "agent": "developer"
            }
        ]

        plan1 = planner1.create_plan(
            sprint_number="sprint-1",
            budget=500000,
            issues=issues,
            algorithm=PlanningAlgorithm.GREEDY
        )

        # Create second planner and load the plan
        planner2 = TokenAwarePlanner(plan_path)
        plan2 = planner2.get_plan()

        assert plan2 is not None
        assert plan2.sprint_number == "sprint-1"
        assert plan2.algorithm == PlanningAlgorithm.GREEDY
        assert len(plan2.entries) == 1
        assert plan2.entries[0].issue_iid == 1


def test_corrupt_plan_file_is_handled_gracefully():
    """Test that a corrupt YAML plan file is silently discarded (starts fresh)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.yml"
        # Write a structurally valid YAML that is missing required keys
        plan_path.write_text("sprint_number: sprint-1\n# missing all other required fields\n")

        planner = TokenAwarePlanner(plan_path)
        # Should start fresh without raising
        assert planner.get_plan() is None


def test_next_issue():
    """Test getting the next issue to process."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.yml"
        planner = TokenAwarePlanner(plan_path)

        issues = [
            {
                "issue_iid": 1,
                "title": "First Issue",
                "labels": ["feature"],
                "description": "First.",
                "agent": "developer"
            },
            {
                "issue_iid": 2,
                "title": "Second Issue",
                "labels": ["bug"],
                "description": "Second.",
                "agent": "developer"
            }
        ]

        plan = planner.create_plan(
            sprint_number="sprint-1",
            budget=500000,
            issues=issues,
            algorithm=PlanningAlgorithm.GREEDY
        )

        # Get first issue - should be the one with lower priority (both have priority 5, so order may vary)
        first_issue = planner.next_issue("developer")
        assert first_issue is not None
        assert first_issue.status == "in_progress"

        # Get second issue
        second_issue = planner.next_issue("developer")
        assert second_issue is not None
        assert second_issue.issue_iid != first_issue.issue_iid

        # No more issues
        third_issue = planner.next_issue("developer")
        assert third_issue is None


def test_next_issue_assigns_requesting_agent():
    """Test that next_issue() assigns work to the requesting agent, not 'developer'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.yml"
        planner = TokenAwarePlanner(plan_path)

        issues = [
            {
                "issue_iid": 1,
                "title": "Arch Issue",
                "labels": ["architecture"],
                "description": "Architecture work.",
                "agent": "architect",
            }
        ]

        planner.create_plan(
            sprint_number="sprint-1",
            budget=1_000_000,
            issues=issues,
            algorithm=PlanningAlgorithm.GREEDY,
        )

        # When an architect agent picks up the issue, the entry should reflect that
        entry = planner.next_issue("architect")
        assert entry is not None
        assert entry.agent == "architect"


def test_plan_entry_preserves_agent_from_issue():
    """Test that PlanEntry.agent reflects the issue's agent, not always 'developer'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.yml"
        planner = TokenAwarePlanner(plan_path)

        issues = [
            {
                "issue_iid": 1,
                "title": "Arch Issue",
                "labels": ["architecture"],
                "description": "Short.",
                "agent": "architect",
            },
            {
                "issue_iid": 2,
                "title": "Dev Issue",
                "labels": ["feature"],
                "description": "Short.",
                "agent": "developer",
            },
        ]

        for algo in PlanningAlgorithm:
            plan = planner.create_plan(
                sprint_number="sprint-1",
                budget=2_000_000,
                issues=issues,
                algorithm=algo,
            )

            by_iid = {e.issue_iid: e for e in plan.entries}
            if 1 in by_iid:
                assert by_iid[1].agent == "architect", (
                    f"Algorithm {algo.value}: expected architect for issue 1, "
                    f"got '{by_iid[1].agent}'"
                )
            if 2 in by_iid:
                assert by_iid[2].agent == "developer", (
                    f"Algorithm {algo.value}: expected developer for issue 2, "
                    f"got '{by_iid[2].agent}'"
                )


def test_mark_completed():
    """Test marking issues as completed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.yml"
        planner = TokenAwarePlanner(plan_path)

        issues = [
            {
                "issue_iid": 1,
                "title": "Test Issue",
                "labels": ["feature"],
                "description": "Test.",
                "agent": "developer"
            }
        ]

        plan = planner.create_plan(
            sprint_number="sprint-1",
            budget=500000,
            issues=issues,
            algorithm=PlanningAlgorithm.GREEDY
        )

        # Mark as in progress first
        issue = planner.next_issue("developer")
        assert issue.status == "in_progress"

        # Mark as completed
        result = planner.mark_completed(1)
        assert result is True

        # Check that it's marked completed
        plan = planner.get_plan()
        assert plan.entries[0].status == "completed"
        assert plan.entries[0].completed_at is not None


def test_mark_completed_returns_false_for_pending():
    """Test that mark_completed returns False for issues not in 'in_progress' status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.yml"
        planner = TokenAwarePlanner(plan_path)

        issues = [
            {
                "issue_iid": 1,
                "title": "Test Issue",
                "labels": ["feature"],
                "description": "Test.",
                "agent": "developer"
            }
        ]

        plan = planner.create_plan(
            sprint_number="sprint-1",
            budget=500000,
            issues=issues,
            algorithm=PlanningAlgorithm.GREEDY
        )

        # Try to mark as completed without moving to in_progress first
        result = planner.mark_completed(1)
        assert result is False

        # Issue should still be pending
        plan = planner.get_plan()
        assert plan.entries[0].status == "pending"


def test_plan_summary():
    """Test plan summary generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.yml"
        planner = TokenAwarePlanner(plan_path)

        issues = [
            {
                "issue_iid": 1,
                "title": "Issue 1",
                "labels": ["feature"],
                "description": "First.",
                "agent": "developer"
            },
            {
                "issue_iid": 2,
                "title": "Issue 2",
                "labels": ["bug"],
                "description": "Second.",
                "agent": "developer"
            }
        ]

        plan = planner.create_plan(
            sprint_number="sprint-1",
            budget=500000,
            issues=issues,
            algorithm=PlanningAlgorithm.GREEDY
        )

        # Mark one as in progress
        planner.next_issue("developer")

        summary = planner.plan_summary()

        assert summary["sprint_number"] == "sprint-1"
        assert summary["algorithm"] == "greedy"
        assert summary["budget"] == 500000
        assert summary["total_planned_tokens"] > 0
        assert summary["pending_issues"] == 1
        assert summary["in_progress_issues"] == 1
        assert summary["completed_issues"] == 0


def test_token_estimation_formula():
    """Test that the token estimation formula matches the specification."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.yml"
        planner = TokenAwarePlanner(plan_path)

        # Test the exact formula from the spec
        issue = {
            "issue_iid": 1,
            "title": "Test",
            "labels": ["feature"],
            "description": "A" * 2000,  # Exactly 2000 chars
            "agent": "developer"
        }

        estimate = planner.estimate_issue_cost(
            issue_iid=issue["issue_iid"],
            title=issue["title"],
            labels=issue["labels"],
            description=issue["description"],
            agent=issue["agent"]
        )

        # Formula: base * min(1 + len(desc)/2000, cap) * agent_factor * 1.2
        # base = 50000 (feature)
        # desc_multiplier = min(1 + (2000/2000), 5.0) = min(2.0, 5.0) = 2.0
        # agent_factor = 1.0 (developer)
        # total = 50000 * 2 * 1.0 * 1.2 = 120000

        assert estimate.base_cost == 50000
        assert estimate.description_multiplier == 2.0
        assert estimate.agent_factor == 1.0
        assert estimate.total_tokens == 120000


def test_description_multiplier_cap():
    """Test that the description multiplier is capped to prevent absurd estimates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.yml"
        planner = TokenAwarePlanner(plan_path)

        # 200 000-char description would give multiplier = 1 + 200000/2000 = 101 uncapped
        huge_description = "x" * 200_000

        estimate = planner.estimate_issue_cost(
            issue_iid=1,
            title="Huge Issue",
            labels=["feature"],
            description=huge_description,
            agent="developer",
        )

        assert estimate.description_multiplier == _DESC_MULTIPLIER_CAP
        # total = 50000 * 5.0 * 1.0 * 1.2 = 300000
        assert estimate.total_tokens == int(50000 * _DESC_MULTIPLIER_CAP * 1.0 * 1.2)


def test_budget_constraints():
    """Test that planning respects budget constraints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.yml"
        planner = TokenAwarePlanner(plan_path)

        # Create issues that would exceed a small budget
        issues = [
            {
                "issue_iid": 1,
                "title": "Expensive Feature",
                "labels": ["architecture"],  # High base cost
                "description": "Very long description. " * 500,  # High multiplier
                "agent": "architect"  # High agent factor
            },
            {
                "issue_iid": 2,
                "title": "Cheap Bug",
                "labels": ["bug"],
                "description": "Short.",
                "agent": "developer"
            }
        ]

        # Small budget that can only fit the cheap bug.
        # Architecture cost: 100000 * 5.0 (cap) * 1.2 * 1.2 = 720000 — does not fit.
        # Bug cost: 25000 * ~1.003 * 1.0 * 1.2 ≈ 30090 — fits within 50000.
        plan = planner.create_plan(
            sprint_number="sprint-1",
            budget=50000,  # Should fit bug but not architecture
            issues=issues,
            algorithm=PlanningAlgorithm.GREEDY
        )

        # Should only include issues that fit in budget
        total_tokens = sum(entry.estimated_tokens for entry in plan.entries)
        assert total_tokens <= 50000

        # With greedy algorithm, should include the bug (lower priority than architecture)
        assert any(entry.issue_iid == 2 for entry in plan.entries)


def test_planning_algorithm_enum_values():
    """Test that all expected PlanningAlgorithm values are available."""
    assert PlanningAlgorithm.GREEDY.value == "greedy"
    assert PlanningAlgorithm.OPTIMAL.value == "optimal"
    assert PlanningAlgorithm.FAIR.value == "fair"


def test_issue_estimate_agent_field():
    """Test that IssueEstimate stores the agent field correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.yml"
        planner = TokenAwarePlanner(plan_path)

        estimate = planner.estimate_issue_cost(
            issue_iid=1,
            title="Architect Issue",
            labels=["architecture"],
            description="Short.",
            agent="architect",
        )

        assert estimate.agent == "architect"

        # Round-trip through dict
        restored = IssueEstimate.from_dict(estimate.to_dict())
        assert restored.agent == "architect"


def test_budget_validation():
    """Test that create_plan validates budget parameter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.yml"
        planner = TokenAwarePlanner(plan_path)

        issues = [
            {
                "issue_iid": 1,
                "title": "Test Issue",
                "labels": ["feature"],
                "description": "Test.",
                "agent": "developer"
            }
        ]

        # Test zero budget
        with pytest.raises(ValueError, match="budget must be positive"):
            planner.create_plan(
                sprint_number="sprint-1",
                budget=0,
                issues=issues,
                algorithm=PlanningAlgorithm.GREEDY
            )

        # Test negative budget
        with pytest.raises(ValueError, match="budget must be positive"):
            planner.create_plan(
                sprint_number="sprint-1",
                budget=-1000,
                issues=issues,
                algorithm=PlanningAlgorithm.GREEDY
            )
