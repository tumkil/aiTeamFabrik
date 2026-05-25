#"""Unit tests for POAgent and related classes."""

import pytest
from datetime import date
from factory.core.po_agent import POAgent, Issue, SprintCapacity


def test_issue_priority_score():
    """Test that Issue calculates priority score correctly."""
    # High business value, low effort
    issue1 = Issue(
        iid=1,
        title="High value feature",
        description="Important feature",
        labels=["priority"],
        weight=2,
        business_value=9
    )
    assert issue1.priority_score == 4.5
    
    # Low business value, high effort
    issue2 = Issue(
        iid=2,
        title="Low value task",
        description="Not important",
        labels=["tech-debt"],
        weight=8,
        business_value=2
    )
    assert issue2.priority_score == 0.25
    
    # No weight or business value
    issue3 = Issue(
        iid=3,
        title="Unestimated task",
        description="No estimates",
        labels=[],
        weight=None,
        business_value=None
    )
    assert issue3.priority_score == 0.0


def test_sprint_capacity_is_unlimited():
    """Test that SprintCapacity correctly identifies unlimited capacity."""
    # Unlimited capacity
    capacity1 = SprintCapacity(total_points=-1, max_issues=-1)
    assert capacity1.is_unlimited is True
    
    # Limited capacity
    capacity2 = SprintCapacity(total_points=20, max_issues=10)
    assert capacity2.is_unlimited is False


def test_po_agent_initialization():
    """Test that POAgent initializes correctly."""
    agent = POAgent(config_path="config/factory.yml")
    
    assert agent is not None
    assert agent.capacity is not None
    assert isinstance(agent.capacity, SprintCapacity)


def test_po_agent_assess_issue():
    """Test that POAgent assesses issues correctly."""
    agent = POAgent(config_path="config/factory.yml")
    
    # Issue with clear requirements
    issue_data1 = {
        "iid": 1,
        "title": "Implement user authentication",
        "description": "As a user, I want to log in so that I can access my account.\n\nAcceptance criteria:\n1. User can enter email and password\n2. System validates credentials\n3. User is redirected to dashboard",
        "labels": ["feature"],
        "weight": 5
    }
    
    issue1 = agent.assess_issue(issue_data1)
    assert issue1.clarification_needed is False
    assert issue1.business_value >= 1
    assert issue1.weight == 5
    
    # Issue needing clarification
    issue_data2 = {
        "iid": 2,
        "title": "TBD: Fix something",
        "description": "This needs to be fixed.",
        "labels": [],
        "weight": None
    }
    
    issue2 = agent.assess_issue(issue_data2)
    assert issue2.clarification_needed is True
    assert issue2.business_value >= 1


def test_po_agent_needs_clarification():
    """Test the clarification detection logic."""
    agent = POAgent(config_path="config/factory.yml")
    
    # Test vague title
    assert agent._needs_clarification("TODO: Implement feature", "") is True
    
    # Test short description
    assert agent._needs_clarification("Feature", "Short") is True
    
    # Test clarification markers
    assert agent._needs_clarification("Feature", "Description [clarification needed]") is True
    
    # Test clear issue
    assert agent._needs_clarification("Implement authentication", 
                                      "Detailed description with acceptance criteria.") is False


def test_po_agent_calculate_business_value():
    """Test the business value calculation."""
    agent = POAgent(config_path="config/factory.yml")
    
    # High value issue
    value1 = agent._calculate_business_value(
        "Customer feature",
        "This will increase revenue",
        ["priority", "customer-request"]
    )
    assert value1 >= 7  # Should be high due to labels and content
    
    # Low value issue
    value2 = agent._calculate_business_value(
        "Refactor code",
        "Clean up technical debt",
        ["tech-debt"]
    )
    assert value2 <= 5  # Should be lower due to tech-debt label


def test_po_agent_prioritize_issues():
    """Test that issues are prioritized correctly."""
    agent = POAgent(config_path="config/factory.yml")
    
    issues = [
        Issue(iid=1, title="High value", description="Important", labels=[], weight=2, business_value=9),
        Issue(iid=2, title="Low value", description="Less important", labels=[], weight=8, business_value=2),
        Issue(iid=3, title="Medium", description="Medium", labels=[], weight=5, business_value=5),
    ]
    
    prioritized = agent.prioritize_issues(issues)
    
    # Should be sorted by priority score (descending)
    assert prioritized[0].iid == 1  # Highest priority score
    assert prioritized[1].iid == 3  # Medium priority score
    assert prioritized[2].iid == 2  # Lowest priority score


def test_po_agent_plan_sprint():
    """Test sprint planning with capacity constraints."""
    agent = POAgent(config_path="config/factory.yml")
    
    # Set a specific capacity for testing
    agent.update_capacity(total_points=10, max_issues=3)
    
    issues = [
        Issue(iid=1, title="Issue 1", description="Desc", labels=[], weight=3, business_value=8, clarification_needed=False),
        Issue(iid=2, title="Issue 2", description="Desc", labels=[], weight=5, business_value=7, clarification_needed=False),
        Issue(iid=3, title="Issue 3", description="Desc", labels=[], weight=4, business_value=6, clarification_needed=False),
        Issue(iid=4, title="Issue 4", description="Desc", labels=[], weight=2, business_value=9, clarification_needed=False),
        Issue(iid=5, title="Needs clarification", description="Desc", labels=[], weight=1, business_value=5, clarification_needed=True),
    ]
    
    selected = agent.plan_sprint(issues)
    
    # Should select issues that fit within capacity (10 points, max 3 issues)
    # Issue 4 (2 points) + Issue 1 (3 points) + Issue 2 (5 points) = 10 points, 3 issues
    assert len(selected) == 3
    assert sum(issue.weight for issue in selected) <= 10
    
    # Issue 5 should not be selected (needs clarification)
    assert 5 not in [issue.iid for issue in selected]


def test_po_agent_get_issues_needing_clarification():
    """Test filtering issues that need clarification."""
    agent = POAgent(config_path="config/factory.yml")
    
    issues = [
        Issue(iid=1, title="Clear issue", description="Clear description", labels=[], clarification_needed=False),
        Issue(iid=2, title="Needs clarification", description="Vague", labels=[], clarification_needed=True),
        Issue(iid=3, title="Another clear", description="Clear", labels=[], clarification_needed=False),
    ]
    
    needing_clarification = agent.get_issues_needing_clarification(issues)
    
    assert len(needing_clarification) == 1
    assert needing_clarification[0].iid == 2


def test_po_agent_update_capacity():
    """Test updating sprint capacity."""
    agent = POAgent(config_path="config/factory.yml")
    
    # Update capacity
    agent.update_capacity(total_points=25, max_issues=15)
    
    # Verify capacity was updated
    assert agent.capacity.total_points == 25
    assert agent.capacity.max_issues == 15


def test_po_agent_unlimited_capacity():
    """Test behavior with unlimited capacity."""
    agent = POAgent(config_path="config/factory.yml")
    
    # Set unlimited capacity
    agent.update_capacity(total_points=-1, max_issues=-1)
    
    issues = [
        Issue(iid=1, title="Issue 1", description="Desc", labels=[], weight=10, business_value=5, clarification_needed=False),
        Issue(iid=2, title="Issue 2", description="Desc", labels=[], weight=20, business_value=8, clarification_needed=False),
        Issue(iid=3, title="Issue 3", description="Desc", labels=[], weight=30, business_value=3, clarification_needed=False),
    ]
    
    # With unlimited capacity, all issues should be selected (except those needing clarification)
    selected = agent.plan_sprint(issues)
    assert len(selected) == 3


def test_po_agent_engage_in_discussion():
    """Test the conversational engagement functionality."""
    agent = POAgent(config_path="config/factory.yml")
    
    # Test basic conversation
    user_input = "What should we prioritize for the next sprint?"
    response = agent.engage_in_discussion(user_input)
    
    # Verify that the response is not empty and is a string
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Test conversation with history
    conversation_history = [
        {"role": "user", "content": "What are our priorities?"},
        {"role": "assistant", "content": "We should focus on high-value features."}
    ]
    response_with_history = agent.engage_in_discussion(user_input, conversation_history)
    
    # Verify that the response with history is also valid
    assert isinstance(response_with_history, str)
    assert len(response_with_history) > 0
