"""Integration tests for POAgent with realistic scenarios."""

import pytest
from factory.core.po_agent import POAgent, Issue, SprintCapacity


def test_po_agent_realistic_sprint_planning():
    """Test sprint planning with a realistic backlog."""
    agent = POAgent(config_path="config/factory.yml")
    
    # Set realistic capacity
    agent.update_capacity(total_points=35, max_issues=8)
    
    # Create a realistic backlog
    issues = [
        # High priority customer features
        Issue(iid=1, title="Implement payment gateway", description="Integrate Stripe for customer payments", 
              labels=["priority", "customer-request"], weight=8, business_value=9, clarification_needed=False),
        
        Issue(iid=2, title="Fix checkout bug", description="Users can't complete checkout on mobile devices", 
              labels=["bug", "critical"], weight=5, business_value=8, clarification_needed=False),
        
        # Medium priority features
        Issue(iid=3, title="Add user profile page", description="Allow users to view and edit their profile", 
              labels=["feature"], weight=5, business_value=6, clarification_needed=False),
        
        Issue(iid=4, title="Improve search functionality", description="Enhance search with filters and sorting", 
              labels=["feature", "enhancement"], weight=8, business_value=7, clarification_needed=False),
        
        # Technical debt
        Issue(iid=5, title="Refactor authentication module", description="Clean up legacy auth code", 
              labels=["tech-debt", "refactor"], weight=13, business_value=4, clarification_needed=False),
        
        # Issues needing clarification
        Issue(iid=6, title="TODO: Implement analytics", description="Add tracking for user behavior", 
              labels=[], weight=8, business_value=5, clarification_needed=True),
        
        Issue(iid=7, title="Investigate performance issues", description="System is slow, need to investigate", 
              labels=["performance"], weight=None, business_value=6, clarification_needed=True),
        
        # Low priority
        Issue(iid=8, title="Update documentation", description="Improve API documentation", 
              labels=["documentation", "chore"], weight=3, business_value=3, clarification_needed=False),
        
        Issue(iid=9, title="Fix typo in UI", description="Minor text correction", 
              labels=["bug", "trivial"], weight=1, business_value=2, clarification_needed=False),
        
        # Another high priority that might not fit
        Issue(iid=10, title="Implement email notifications", description="Send transactional emails to users", 
              labels=["priority", "customer-request"], weight=10, business_value=8, clarification_needed=False),
    ]
    
    selected = agent.plan_sprint(issues)
    
    # Should select issues that fit within 35 points and max 8 issues
    # Issues needing clarification (6, 7) should be excluded
    total_points = sum(issue.weight for issue in selected if issue.weight)
    assert total_points <= 35
    assert len(selected) <= 8
    
    # Verify high priority issues are selected
    selected_ids = [issue.iid for issue in selected]
    assert 1 in selected_ids  # High priority payment gateway
    assert 2 in selected_ids  # Critical bug fix
    
    # Verify issues needing clarification are not selected
    assert 6 not in selected_ids
    assert 7 not in selected_ids


def test_po_agent_prioritization_with_mixed_issues():
    """Test prioritization with a mix of issue types and priorities."""
    agent = POAgent(config_path="config/factory.yml")
    
    issues = [
        # Critical production issue
        Issue(iid=1, title="Production outage", description="System is down affecting all users", 
              labels=["bug", "critical", "production"], weight=13, business_value=10, clarification_needed=False),
        
        # High value customer feature
        Issue(iid=2, title="Customer requested feature", description="Important feature for major client", 
              labels=["priority", "customer-request"], weight=8, business_value=9, clarification_needed=False),
        
        # Technical debt with high weight
        Issue(iid=3, title="Refactor legacy code", description="Clean up old codebase", 
              labels=["tech-debt"], weight=21, business_value=3, clarification_needed=False),
        
        # Small bug fix
        Issue(iid=4, title="Minor UI bug", description="Button alignment issue", 
              labels=["bug", "trivial"], weight=2, business_value=4, clarification_needed=False),
        
        # Issue needing clarification
        Issue(iid=5, title="TBD: Future enhancement", description="Something we might do later", 
              labels=[], weight=5, business_value=5, clarification_needed=True),
    ]
    
    prioritized = agent.prioritize_issues(issues)
    
    # Small bug has the best priority score (4/2 = 2.0)
    # Critical production issue has 10/13 = 0.769
    # Customer feature has 9/8 = 1.125
    # Technical debt has 3/21 = 0.142
    # Issue needing clarification has 5/5 = 1.0
    
    # Verify ordering based on priority score
    assert prioritized[0].iid == 4  # Best ratio
    assert prioritized[1].iid == 2  # Second best ratio
    assert prioritized[2].iid == 5  # Third best ratio (but needs clarification)
    assert prioritized[3].iid == 1  # Fourth best ratio
    assert prioritized[4].iid == 3  # Worst ratio


def test_po_agent_capacity_constraints():
    """Test behavior with different capacity constraints."""
    agent = POAgent(config_path="config/factory.yml")
    
    issues = [
        Issue(iid=1, title="Issue 1", description="Desc", labels=[], weight=5, business_value=8, clarification_needed=False),
        Issue(iid=2, title="Issue 2", description="Desc", labels=[], weight=10, business_value=7, clarification_needed=False),
        Issue(iid=3, title="Issue 3", description="Desc", labels=[], weight=8, business_value=6, clarification_needed=False),
        Issue(iid=4, title="Issue 4", description="Desc", labels=[], weight=3, business_value=9, clarification_needed=False),
        Issue(iid=5, title="Issue 5", description="Desc", labels=[], weight=15, business_value=5, clarification_needed=False),
    ]
    
    # Test with tight capacity
    agent.update_capacity(total_points=15, max_issues=3)
    selected = agent.plan_sprint(issues)
    assert len(selected) <= 3  # Should not exceed max_issues
    assert sum(issue.weight for issue in selected) <= 15  # Should not exceed total_points
    
    # Test with loose capacity but tight issue limit
    agent.update_capacity(total_points=100, max_issues=2)
    selected = agent.plan_sprint(issues)
    assert len(selected) == 2
    
    # Test with tight capacity but loose issue limit
    agent.update_capacity(total_points=10, max_issues=10)
    selected = agent.plan_sprint(issues)
    assert sum(issue.weight for issue in selected) <= 10


def test_po_agent_error_handling():
    """Test error handling in POAgent methods."""
    agent = POAgent(config_path="config/factory.yml")
    
    # Test with empty issue list
    empty_issues = []
    prioritized = agent.prioritize_issues(empty_issues)
    assert prioritized == []
    
    selected = agent.plan_sprint(empty_issues)
    assert selected == []
    
    # Test with None values - convert to empty strings
    issue_data = {
        "iid": 1,
        "title": "",
        "description": "",
        "labels": [],
        "weight": None
    }
    issue = agent.assess_issue(issue_data)
    assert issue.clarification_needed is True
    assert issue.business_value >= 1


def test_po_agent_business_value_edge_cases():
    """Test business value calculation with edge cases."""
    agent = POAgent(config_path="config/factory.yml")
    
    # Test with empty strings
    value1 = agent._calculate_business_value("", "", [])
    assert value1 >= 1  # Should have minimum value
    
    # Test with very long description
    long_desc = "A" * 10000
    value2 = agent._calculate_business_value("Test", long_desc, [])
    assert value2 >= 1
    
    # Test with special characters
    value3 = agent._calculate_business_value("Test !@#$%^&*()", "Description with special chars !@#$%^&*()", [])
    assert value3 >= 1


def test_po_agent_clarification_detection_comprehensive():
    """Test clarification detection with various patterns."""
    agent = POAgent(config_path="config/factory.yml")
    
    # Test various clarification patterns
    clarification_cases = [
        ("TODO: Implement", "Description"),
        ("TBD: Feature", "Description"),
        ("Title", "TODO: Description"),
        ("TBD: Description", "Description"),
        ("Title", "Description [clarification needed]"),
        ("Title", "Description [needs clarification]"),
        ("Title", "Description [question]"),
        ("Title", "Description [blocked]"),
        ("Title", "Description [pending input]"),
        ("Title", "Short"),  # Very short description
        ("Title", ""),  # Empty description
        ("", "Description"),  # Empty title
    ]
    
    for title, description in clarification_cases:
        result = agent._needs_clarification(title, description)
        assert result is True, f"Failed for title='{title}', desc='{description}', got {result}"
    
    # Test clear cases
    clear_cases = [
        ("Implement authentication", "Detailed description with acceptance criteria and examples."),
        ("Fix bug in checkout", "Users cannot complete checkout. Error occurs at payment step."),
        ("Add user profile", "As a user, I want to view my profile so I can see my information."),
    ]
    
    for title, description in clear_cases:
        result = agent._needs_clarification(title, description)
        assert result is False, f"Failed for title='{title}', desc='{description}', got {result}"
