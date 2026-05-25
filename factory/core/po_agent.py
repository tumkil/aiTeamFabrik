"""Product Owner (PO) Agent for sprint planning and backlog management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import date

import yaml


@dataclass
class Issue:
    """Represents a GitLab issue with relevant fields for sprint planning."""
    iid: int
    title: str
    description: str
    labels: List[str]
    weight: Optional[int] = None  # Story points or effort estimate
    business_value: Optional[int] = None  # Business value score (1-10)
    clarification_needed: bool = False
    
    @property
    def priority_score(self) -> float:
        """Calculate priority score based on business value and effort.
        
        Higher business value and lower effort = higher priority.
        """
        if self.weight is None or self.weight == 0 or self.business_value is None:
            return 0.0
        return float(self.business_value) / float(self.weight)


@dataclass
class SprintCapacity:
    """Represents the capacity constraints for a sprint."""
    total_points: int
    max_issues: int
    
    @property
    def is_unlimited(self) -> bool:
        """Check if capacity is effectively unlimited."""
        return self.total_points <= 0 and self.max_issues <= 0


class POAgent:
    """Product Owner Agent responsible for sprint planning and backlog management."""
    
    def __init__(self, config_path: str = "config/factory.yml") -> None:
        """Initialize the PO Agent with configuration.
        
        Args:
            config_path: Path to the factory configuration file
        """
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        self._config_path = config_path
        self._config = cfg
        
        # Load sprint capacity configuration
        sprint_cfg = cfg.get("sprint", {})
        capacity_cfg = sprint_cfg.get("capacity", {})
        
        self._capacity = SprintCapacity(
            total_points=capacity_cfg.get("total_points", 20),
            max_issues=capacity_cfg.get("max_issues", 10)
        )
        
        # Load prioritization weights
        self._prioritization_weights = sprint_cfg.get("prioritization", {})
        
        # Initialize conversation state
        self._conversation_history = []
        self._current_context = {}
    
    @property
    def capacity(self) -> SprintCapacity:
        """Get the current sprint capacity constraints."""
        return self._capacity
    
    def assess_issue(self, issue_data: Dict) -> Issue:
        """Assess an issue and determine if clarification is needed.
        
        Args:
            issue_data: Dictionary containing issue data from GitLab
            
        Returns:
            Issue object with assessment flags
        """
        title = issue_data.get("title", "")
        description = issue_data.get("description", "")
        labels = issue_data.get("labels", [])
        
        # Check for clarification needs
        clarification_needed = self._needs_clarification(title, description)
        
        # Extract weight (story points) if available
        weight = issue_data.get("weight", None)
        if weight is not None:
            weight = int(weight)
        
        # Determine business value based on labels and content
        business_value = self._calculate_business_value(title, description, labels)
        
        return Issue(
            iid=issue_data["iid"],
            title=title,
            description=description,
            labels=labels,
            weight=weight,
            business_value=business_value,
            clarification_needed=clarification_needed
        )
    
    def _needs_clarification(self, title: str, description: str) -> bool:
        """Determine if an issue needs clarification before development.
        
        Args:
            title: Issue title
            description: Issue description
            
        Returns:
            True if clarification is needed, False otherwise
        """
        # Check for vague language in title
        vague_terms = ["todo", "fixme", "tbd", "???", "help", "question", "unknown"]
        title_lower = title.lower()
        
        if any(term in title_lower for term in vague_terms):
            return True
        
        # Check if description is too short or missing
        if not description or len(description.strip()) < 20:
            return True
        
        # Check for clarification request markers
        clarification_markers = ["[clarification needed]", "[needs clarification]",
                                  "[question]", "[blocked]", "[pending input]"]
        desc_lower = description.lower()
        
        if any(marker in desc_lower for marker in clarification_markers):
            return True
        
        return False
    
    def _calculate_business_value(self, title: str, description: str, labels: List[str]) -> int:
        """Calculate business value score for an issue.
        
        Args:
            title: Issue title
            description: Issue description
            labels: List of issue labels
            
        Returns:
            Business value score (1-10)
        """
        score = 5  # Default medium value
        
        # Adjust based on labels
        high_value_labels = ["priority", "critical", "urgent", "high-value", "customer-request"]
        low_value_labels = ["tech-debt", "refactor", "cleanup", "chore"]
        
        label_lower = [label.lower() for label in labels]
        
        if any(label in label_lower for label in high_value_labels):
            score += 2
        
        if any(label in label_lower for label in low_value_labels):
            score -= 2
        
        # Adjust based on title/description content
        content_lower = (title + " " + description).lower()
        
        value_indicators = ["customer", "user", "revenue", "conversion", "retention",
                           "performance", "security", "compliance"]
        
        if any(indicator in content_lower for indicator in value_indicators):
            score += 1
        
        # Ensure score is within bounds
        return max(1, min(10, score))
    
    def prioritize_issues(self, issues: List[Issue]) -> List[Issue]:
        """Prioritize a list of issues for sprint planning.
        
        Args:
            issues: List of Issue objects to prioritize
            
        Returns:
            List of Issue objects sorted by priority (highest first)
        """
        # Sort by priority score (descending), then by business value (descending)
        return sorted(
            issues,
            key=lambda x: (-x.priority_score, -x.business_value, x.weight or 0)
        )
    
    def plan_sprint(self, candidate_issues: List[Issue]) -> List[Issue]:
        """Plan a sprint by selecting issues that fit within capacity.
        
        Args:
            candidate_issues: List of candidate Issue objects
            
        Returns:
            List of Issue objects selected for the sprint
        """
        # First prioritize all issues
        prioritized = self.prioritize_issues(candidate_issues)
        
        selected = []
        total_points = 0
        
        if self._capacity.is_unlimited:
            # If no capacity limits, select all issues
            return prioritized
        
        # Select issues until capacity is reached
        for issue in prioritized:
            # Skip issues that need clarification
            if issue.clarification_needed:
                continue
            
            # Check if we have room for this issue
            if issue.weight is not None:
                if total_points + issue.weight > self._capacity.total_points:
                    continue
            
            if len(selected) >= self._capacity.max_issues:
                break
            
            selected.append(issue)
            if issue.weight is not None:
                total_points += issue.weight
        
        return selected
    
    def get_issues_needing_clarification(self, issues: List[Issue]) -> List[Issue]:
        """Filter issues that need clarification.
        
        Args:
            issues: List of Issue objects
            
        Returns:
            List of Issue objects that need clarification
        """
        return [issue for issue in issues if issue.clarification_needed]
    
    def update_capacity(self, total_points: int, max_issues: int) -> None:
        """Update the sprint capacity constraints.
        
        Args:
            total_points: Maximum story points for the sprint
            max_issues: Maximum number of issues for the sprint
        """
        self._capacity = SprintCapacity(
            total_points=total_points,
            max_issues=max_issues
        )
        
        # Update configuration file
        with open(self._config_path) as f:
            cfg = yaml.safe_load(f)
        
        if "sprint" not in cfg:
            cfg["sprint"] = {}
        
        cfg["sprint"]["capacity"] = {
            "total_points": total_points,
            "max_issues": max_issues
        }
        
        with open(self._config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    
    def engage_in_discussion(self, user_input: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """Engage in a conversational discussion about sprint planning or backlog management.
        
        Args:
            user_input: The user's input or question
            conversation_history: Optional list of previous messages in the conversation
            
        Returns:
            The agent's response as a string
        """
        # Update conversation history with user input
        if conversation_history is None:
            conversation_history = self._conversation_history
        
        # Add user input to conversation history
        conversation_history.append({"role": "user", "content": user_input})
        
        # Store the updated conversation history
        self._conversation_history = conversation_history
        
        # Analyze user input and provide context-aware response
        response = self._generate_context_aware_response(user_input, conversation_history)
        
        # Add agent's response to conversation history
        conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _generate_context_aware_response(self, user_input: str, conversation_history: List[Dict]) -> str:
        """Generate a context-aware response based on user input and conversation history.
        
        Args:
            user_input: The current user input
            conversation_history: List of previous messages in the conversation
            
        Returns:
            Context-aware response string
        """
        user_input_lower = user_input.lower()
        
        # Analyze conversation context
        context = self._analyze_conversation_context(conversation_history)
        
        # Check for specific intents and provide appropriate responses
        if "prioritize" in user_input_lower or "priority" in user_input_lower:
            if context.get("discussing_specific_issues"):
                return (
                    "I see you're asking about prioritizing specific issues. "
                    "Based on our current backlog, I can analyze these issues and provide "
                    "recommendations. Would you like me to evaluate them based on business "
                    "value and effort estimates?"
                )
            else:
                return (
                    "As a Product Owner, I prioritize issues based on business value and effort. "
                    "High business value with low effort gets the highest priority. "
                    "Would you like me to analyze specific issues or explain the prioritization process further?"
                )
        
        elif "sprint" in user_input_lower and ("plan" in user_input_lower or "capacity" in user_input_lower):
            if context.get("discussing_capacity"):
                return (
                    f"Our current sprint capacity is {self.capacity.total_points} story points "
                    f"and {self.capacity.max_issues} issues. "
                    "Would you like to adjust this capacity or discuss how we can optimize our sprint planning?"
                )
            else:
                return (
                    "For sprint planning, I consider our current capacity and select issues "
                    "that provide the most business value while fitting within our team's capacity. "
                    f"Our current sprint capacity is {self.capacity.total_points} story points "
                    f"and {self.capacity.max_issues} issues. "
                    "Would you like to discuss specific issues for the next sprint?"
                )
        
        elif "clarification" in user_input_lower:
            if context.get("reviewing_issues"):
                return (
                    "I can help identify issues that need clarification before development. "
                    "These typically have vague titles, short descriptions, or explicit clarification markers. "
                    "Would you like me to review the current backlog and flag issues needing clarification?"
                )
            else:
                return (
                    "Issues needing clarification often have vague titles, short descriptions, "
                    "or explicit markers like '[clarification needed]'. "
                    "Would you like me to review specific issues or explain how I identify clarification needs?"
                )
        
        elif "backlog" in user_input_lower:
            if context.get("managing_backlog"):
                return (
                    "I manage the product backlog by continuously assessing and prioritizing issues. "
                    "The backlog should contain well-defined issues with clear acceptance criteria. "
                    "Would you like me to analyze the current backlog or suggest improvements?"
                )
            else:
                return (
                    "The product backlog contains all potential work items for the product. "
                    "I continuously assess and prioritize these items based on business value and effort. "
                    "Would you like to discuss the current backlog or specific backlog management strategies?"
                )
        
        elif "business value" in user_input_lower:
            if context.get("explaining_value"):
                return (
                    "Business value is determined by factors like customer impact, revenue potential, "
                    "and strategic alignment. I calculate it based on issue labels and content analysis, "
                    "with values ranging from 1 (low) to 10 (high). "
                    "Would you like me to explain how specific labels or content affect business value?"
                )
            else:
                return (
                    "Business value represents the importance of an issue to our business goals. "
                    "I calculate it based on factors like customer impact, revenue potential, "
                    "and strategic alignment. Values range from 1 (low) to 10 (high). "
                    "Would you like me to analyze specific issues for their business value?"
                )
        
        elif "capacity" in user_input_lower:
            return (
                f"Our current sprint capacity is set to {self.capacity.total_points} story points "
                f"and a maximum of {self.capacity.max_issues} issues. "
                "This capacity can be adjusted based on team velocity and business needs. "
                "Would you like to discuss changing our sprint capacity?"
            )
        
        elif "issue" in user_input_lower or "feature" in user_input_lower:
            if context.get("discussing_specific_issues"):
                return (
                    "I can help analyze specific issues or features. "
                    "Would you like me to assess their business value, effort, or priority? "
                    "Or perhaps you'd like to discuss how they fit into our current sprint plan?"
                )
            else:
                return (
                    "I can help with issue and feature analysis. "
                    "This includes assessing business value, effort estimates, priority, "
                    "and how they fit into our sprint planning. "
                    "What specific issues or features would you like to discuss?"
                )
        
        elif any(greeting in user_input_lower for greeting in ["hello", "hi", "hey", "greetings"]):
            return (
                "Hello! I'm the Product Owner Agent. I can help with sprint planning, "
                "backlog management, prioritizing issues, and answering questions about our "
                "development process. What would you like to discuss today?"
            )
        
        elif any(thanks in user_input_lower for thanks in ["thank", "thanks", "appreciate"]):
            return (
                "You're welcome! I'm here to help with all your Product Owner needs. "
                "Is there anything else you'd like to discuss regarding sprint planning, "
                "backlog management, or issue prioritization?"
            )
        
        else:
            # Default response for general questions, with context awareness
            if conversation_history and len(conversation_history) > 1:
                return (
                    "I'm here to help with Product Owner responsibilities. "
                    "Based on our conversation, I can assist with sprint planning, "
                    "backlog management, issue prioritization, or answering specific questions. "
                    "What would you like to focus on?"
                )
            else:
                return (
                    "Hello! I'm the Product Owner Agent. I specialize in sprint planning, "
                    "backlog management, and prioritizing issues based on business value. "
                    "How can I assist you today?"
                )
    
    def _analyze_conversation_context(self, conversation_history: List[Dict]) -> Dict:
        """Analyze conversation history to determine context.
        
        Args:
            conversation_history: List of conversation messages
            
        Returns:
            Dictionary containing context flags
        """
        context = {
            "discussing_specific_issues": False,
            "discussing_capacity": False,
            "reviewing_issues": False,
            "managing_backlog": False,
            "explaining_value": False
        }
        
        if not conversation_history:
            return context
        
        # Analyze the last few messages for context
        recent_messages = conversation_history[-min(3, len(conversation_history)):]
        
        for message in recent_messages:
            content_lower = message.get("content", "").lower()
            
            # Check for specific issue discussion
            if any(keyword in content_lower for keyword in ["specific", "particular", "certain", "this issue", "that feature"]):
                context["discussing_specific_issues"] = True
            
            # Check for capacity discussion
            if any(keyword in content_lower for keyword in ["capacity", "story points", "max issues", "team velocity"]):
                context["discussing_capacity"] = True
            
            # Check for issue review
            if any(keyword in content_lower for keyword in ["review", "analyze", "assess", "evaluate", "check"]):
                context["reviewing_issues"] = True
            
            # Check for backlog management
            if any(keyword in content_lower for keyword in ["backlog", "manage", "organize", "groom", "refine"]):
                context["managing_backlog"] = True
            
            # Check for business value explanation
            if any(keyword in content_lower for keyword in ["explain", "how", "calculate", "determine", "business value"]):
                context["explaining_value"] = True
        
        return context
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the current conversation history.
        
        Returns:
            List of conversation messages
        """
        return self._conversation_history
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self._conversation_history = []
        self._current_context = {}
    
    def set_conversation_context(self, context: Dict) -> None:
        """Set specific context for the conversation.
        
        Args:
            context: Dictionary containing context information
        """
        self._current_context = context