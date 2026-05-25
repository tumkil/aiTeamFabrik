import pytest
from factory.core.token_planner import TokenAwarePlanner
from factory.core.issue import Issue


def test_fair_plan_when_every_issue_exceeds_budget():
    """Test that TokenAwarePlanner._fair_plan handles the case where every issue individually exceeds the budget."""
    planner = TokenAwarePlanner()
    
    # Create issues where each issue's token cost exceeds the budget
    issue1 = Issue("Issue 1", "Description 1", 1000)
    issue2 = Issue("Issue 2", "Description 2", 1000)
    issue3 = Issue("Issue 3", "Description 3", 1000)
    
    issues = [issue1, issue2, issue3]
    budget = 500  # Budget is less than each issue's token cost
    
    # Call the _fair_plan method
    result = planner._fair_plan(issues, budget)
    
    # Verify the result
    assert isinstance(result, list)
    assert len(result) == 0  # No issues should be selected as all exceed the budget
