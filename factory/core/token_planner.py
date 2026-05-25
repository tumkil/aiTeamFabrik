"""
Token-aware planning module for issue prioritization.

This module provides the TokenAwarePlanner class which implements
a fair planning algorithm for selecting issues based on token budgets.
"""


class TokenAwarePlanner:
    """Plans which issues to address based on token budget constraints."""
    
    def _fair_plan(self, issues, budget):
        """
        Select issues to address within a token budget using fair planning.
        
        Args:
            issues: List of Issue objects to consider
            budget: Maximum token budget available
            
        Returns:
            List of selected Issue objects that fit within budget
        
        The algorithm:
        1. Sorts issues by token cost (ascending)
        2. Selects issues until budget would be exceeded
        3. Skips any individual issue that exceeds the remaining budget
        """
        if not issues:
            return []
        
        # Sort issues by token cost (ascending)
        sorted_issues = sorted(issues, key=lambda x: x.token_cost)
        
        selected = []
        remaining_budget = budget
        
        for issue in sorted_issues:
            if issue.token_cost > remaining_budget:
                # This issue alone exceeds remaining budget; skip it
                continue
            
            selected.append(issue)
            remaining_budget -= issue.token_cost
        
        return selected