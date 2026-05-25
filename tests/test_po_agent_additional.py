"""Additional unit tests for POAgent edge cases and comprehensive coverage."""

import pytest
from factory.core.po_agent import POAgent, Issue, SprintCapacity


def test_issue_priority_score_edge_cases():
    """Test priority score calculation with edge cases."""
    # Zero weight (should return 0 to avoid division by zero)
    issue1 = Issue(
        iid=1,
        title="Zero weight issue",
        description="Issue with zero weight",
        labels=[],
        weight=0,
        business_value=5
    )
    assert issue1.priority_score == 0.0
    
    # None weight
    issue2 = Issue(
        iid=2,
        title="None weight issue",
        description="Issue with None weight",
        labels=[],
        weight=None,
        business_value=5
    )
    assert issue2.priority_score == 0.0
    
    # Zero business value
    issue3 = Issue(
        iid=3,
        title="Zero business value",
        description="Issue with zero business value",
        labels=[],
        weight=5,
        business_value=0
    )
    assert issue3.priority_score == 0.0


def test_sprint_capacity_edge_cases():
    """Test SprintCapacity with edge cases."""
    # Zero capacity (should be treated as unlimited according to current implementation)
    capacity1 = SprintCapacity(total_points=0, max_issues=0)
    assert capacity1.is_unlimited is True  # Current implementation treats 0 as unlimited
    
    # Mixed zero and negative
    capacity2 = SprintCapacity(total_points=0, max_issues=-1)
    assert capacity2.is_unlimited is True  # Both <= 0
    
    # Large capacity
    capacity3 = SprintCapacity(total_points=1000, max_issues=100)
    assert capacity3.is_unlimited is False


def test_po_agent_assess_issue_edge_cases():
    """Test POAgent assess_issue with edge cases."""
    agent = POAgent(config_path="config/factory.yml")
    
    # Empty issue data
    issue_data1 = {
        "iid": 1,
        "title": "",
        "description": "",
        "labels": []
    }
    issue1 = agent.assess_issue(issue_data1)
    assert issue1.clarification_needed is True  # Empty description should need clarification
    assert issue1.business_value >= 1
    
    # Issue with vague terms in title
    issue_data2 = {
        "iid": 2,
        "title": "TODO: Implement something",
        "description": "This is a detailed description with acceptance criteria.",
        "labels": []
    }
    issue2 = agent.assess_issue(issue_data2)
    assert issue2.clarification_needed is True  # TODO in title
    
    # Issue with clarification marker in description
    issue_data3 = {
        "iid": 3,
        "title": "Implement feature",
        "description": "This needs more details. [clarification needed]",
        "labels": []
    }
    issue3 = agent.assess_issue(issue_data3)
    assert issue3.clarification_needed is True  # Clarification marker


def test_po_agent_business_value_calculation_comprehensive():
    """Test business value calculation with various label combinations."""
    agent = POAgent(config_path="config/factory.yml")
    
    # Multiple high-value labels
    value1 = agent._calculate_business_value(
        "Customer feature",
        "This will increase revenue and conversion",
        ["priority", "critical", "customer-request"]
    )
    assert value1 >= 8  # Should be high due to multiple high-value labels
    
    # Multiple low-value labels
    value2 = agent._calculate_business_value(
        "Refactor code",
        "Clean up technical debt and improve code quality",
        ["tech-debt", "refactor", "chore"]
    )
    assert value2 <= 4  # Should be low due to multiple low-value labels
    
    # Mixed labels
    value3 = agent._calculate_business_value(
        "Security fix",
        "Fix security vulnerability for customer data",
        ["priority", "tech-debt"]
    )
    # Should balance between priority (+2) and tech-debt (-2)
    assert 4 <= value3 <= 7
    
    # Content with multiple value indicators
    value4 = agent._calculate_business_value(
        "Improve user retention and revenue",
        "This feature will increase customer retention and revenue",
        []
    )
    assert value4 >= 6  # Should be boosted by multiple value indicators


def test_po_agent_prioritize_issues_complex():
    """Test prioritization with complex scenarios."""
    agent = POAgent(config_path="config/factory.yml")
    
    issues = [
        # High business value, low weight = high priority
        Issue(iid=1, title="High value low effort", description="Important", 
              labels=[], weight=1, business_value=9, clarification_needed=False),
        
        # Same priority score but higher business value should come first
        Issue(iid=2, title="Same ratio higher value", description="Important", 
              labels=[], weight=2, business_value=9, clarification_needed=False),
        
        # Same priority score but lower weight should come first
        Issue(iid=3, title="Same ratio lower weight", description="Important", 
              labels=[], weight=1, business_value=4.5, clarification_needed=False),
        
        # Needs clarification (should be ranked lower)
        Issue(iid=4, title="Needs clarification", description="Vague", 
              labels=[], weight=2, business_value=8, clarification_needed=True),
        
        # No weight (should be ranked lower)
        Issue(iid=5, title="No weight", description="Issue", 
              labels=[], weight=None, business_value=5, clarification_needed=False),
    ]
    
    prioritized = agent.prioritize_issues(issues)
    
    # Check ordering
    assert prioritized[0].iid == 1  # Highest priority score (9/1 = 9.0)
    assert prioritized[1].iid == 2  # Priority score 4.5, but higher business value than issue 3
    assert prioritized[2].iid == 3  # Priority score 4.5, but lower weight than issue 2
    assert prioritized[3].iid == 4  # Needs clarification
    assert prioritized[4].iid == 5  # No weight


def test_po_agent_plan_sprint_complex():
    """Test sprint planning with complex scenarios."""
    agent = POAgent(config_path="config/factory.yml")
    
    # Set specific capacity
    agent.update_capacity(total_points=15, max_issues=4)
    
    issues = [
        # High priority, fits in capacity
        Issue(iid=1, title="Issue 1", description="Desc", labels=[], 
              weight=5, business_value=8, clarification_needed=False),
        
        # High priority, fits in capacity
        Issue(iid=2, title="Issue 2", description="Desc", labels=[], 
              weight=6, business_value=9, clarification_needed=False),
        
        # High priority but needs clarification
        Issue(iid=3, title="Issue 3", description="Desc", labels=[], 
              weight=3, business_value=7, clarification_needed=True),
        
        # Lower priority but fits
        Issue(iid=4, title="Issue 4", description="Desc", labels=[], 
              weight=4, business_value=5, clarification_needed=False),
        
        # Would exceed capacity if added
        Issue(iid=5, title="Issue 5", description="Desc", labels=[], 
              weight=8, business_value=6, clarification_needed=False),
        
        # No weight
        Issue(iid=6, title="Issue 6", description="Desc", labels=[], 
              weight=None, business_value=4, clarification_needed=False),
    ]
    
    selected = agent.plan_sprint(issues)
    
    # Should select issues 2 (6), 1 (5), and 4 (4) = 15 points, 3 issues
    # Issue 3 is skipped (needs clarification)
    # Issue 5 would exceed capacity (6+5+4+8 = 23 > 15)
    # Issue 6 has no weight so it's included if there's room
    
    assert len(selected) == 4  # Issues 2, 1, 4, and 6 (no weight doesn't count toward total)
    assert sum(issue.weight for issue in selected if issue.weight) == 15
    assert 3 not in [issue.iid for issue in selected]  # Needs clarification
    assert 5 not in [issue.iid for issue in selected]  # Would exceed capacity


def test_po_agent_plan_sprint_no_weight_issues():
    """Test sprint planning with issues that have no weight."""
    agent = POAgent(config_path="config/factory.yml")
    
    # Set capacity
    agent.update_capacity(total_points=10, max_issues=3)
    
    issues = [
        Issue(iid=1, title="Issue 1", description="Desc", labels=[], 
              weight=None, business_value=5, clarification_needed=False),
        Issue(iid=2, title="Issue 2", description="Desc", labels=[], 
              weight=8, business_value=6, clarification_needed=False),
        Issue(iid=3, title="Issue 3", description="Desc", labels=[], 
              weight=None, business_value=4, clarification_needed=False),
        Issue(iid=4, title="Issue 4", description="Desc", labels=[], 
              weight=5, business_value=7, clarification_needed=False),
    ]
    
    selected = agent.plan_sprint(issues)
    
    # Should select: Issue 4 (5 points) and then issues without weight
    assert len(selected) == 3  # Issue 4, 1, and 3 (max_issues limit)
    assert 4 in [issue.iid for issue in selected]
    assert 2 not in [issue.iid for issue in selected]  # Would exceed capacity


def test_po_agent_capacity_update_edge_cases():
    """Test capacity updates with edge cases."""
    agent = POAgent(config_path="config/factory.yml")
    
    # Update to zero capacity (treated as unlimited in current implementation)
    agent.update_capacity(total_points=0, max_issues=0)
    assert agent.capacity.total_points == 0
    assert agent.capacity.max_issues == 0
    assert agent.capacity.is_unlimited is True  # Current implementation
    
    # Update to negative (unlimited)
    agent.update_capacity(total_points=-1, max_issues=-1)
    assert agent.capacity.is_unlimited is True
    
    # Update to very large values
    agent.update_capacity(total_points=10000, max_issues=1000)
    assert agent.capacity.total_points == 10000
    assert agent.capacity.max_issues == 1000


def test_po_agent_unlimited_capacity_with_clarification():
    """Test unlimited capacity behavior - currently includes issues needing clarification."""
    agent = POAgent(config_path="config/factory.yml")
    
    # Set unlimited capacity
    agent.update_capacity(total_points=-1, max_issues=-1)
    
    issues = [
        Issue(iid=1, title="Issue 1", description="Desc", labels=[], 
              weight=10, business_value=5, clarification_needed=False),
        Issue(iid=2, title="Issue 2", description="Desc", labels=[], 
              weight=20, business_value=8, clarification_needed=True),
        Issue(iid=3, title="Issue 3", description="Desc", labels=[], 
              weight=30, business_value=3, clarification_needed=False),
    ]
    
    selected = agent.plan_sprint(issues)
    
    # Current implementation: unlimited capacity includes ALL issues
    # This test documents the current behavior
    assert len(selected) == 3  # All issues are selected with unlimited capacity
    assert 1 in [issue.iid for issue in selected]
    assert 2 in [issue.iid for issue in selected]  # Included even though needs clarification
    assert 3 in [issue.iid for issue in selected]
