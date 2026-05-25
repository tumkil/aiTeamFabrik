"""
Issue module for representing GitLab issues.

This module provides the Issue class which represents
a GitLab issue with title, description, and token cost.
"""


class Issue:
    """Represents a GitLab issue with token cost information."""
    
    def __init__(self, title, description, token_cost):
        """
        Initialize an Issue.
        
        Args:
            title: Issue title
            description: Issue description
            token_cost: Estimated token cost for this issue
        """
        self.title = title
        self.description = description
        self.token_cost = token_cost
    
    def __repr__(self):
        return f"Issue(title='{self.title}', token_cost={self.token_cost})"
    
    def __eq__(self, other):
        if not isinstance(other, Issue):
            return False
        return (self.title == other.title and 
                self.description == other.description and 
                self.token_cost == other.token_cost)
    
    def __hash__(self):
        return hash((self.title, self.description, self.token_cost))
