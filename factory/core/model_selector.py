"""Dynamic model selection for agent roles based on task requirements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class TaskComplexity(Enum):
    """Enumeration of task complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class TaskType(Enum):
    """Enumeration of task types."""
    FEATURE = "feature"
    BUG = "bug"
    REFACTOR = "refactor"
    DOCUMENTATION = "documentation"
    REVIEW = "review"
    TESTING = "testing"


@dataclass
class ModelCapability:
    """Represents the capabilities of a specific model."""
    name: str
    provider: str
    complexity_support: List[TaskComplexity]
    task_types: List[TaskType]
    cost_per_token: float
    latency: float  # in seconds
    max_tokens: int


class ModelSelector:
    """Selects optimal models for agent roles based on task requirements."""

    def __init__(self) -> None:
        """Initialize the model selector with default model capabilities."""
        self._model_capabilities: Dict[str, ModelCapability] = {}
        self._initialize_default_models()

    def _initialize_default_models(self) -> None:
        """Initialize default model capabilities."""
        # Define capabilities for various models
        self._model_capabilities = {
            "devstral-2:123b-cloud": ModelCapability(
                name="devstral-2:123b-cloud",
                provider="ollama",
                complexity_support=[TaskComplexity.SIMPLE, TaskComplexity.MODERATE],
                task_types=[
                    TaskType.FEATURE,
                    TaskType.BUG,
                    TaskType.REFACTOR,
                    TaskType.TESTING,
                ],
                cost_per_token=0.000002,
                latency=1.5,
                max_tokens=4096,
            ),
            "mistral-large-3:675b-cloud": ModelCapability(
                name="mistral-large-3:675b-cloud",
                provider="ollama",
                complexity_support=[
                    TaskComplexity.SIMPLE,
                    TaskComplexity.MODERATE,
                    TaskComplexity.COMPLEX,
                ],
                task_types=[
                    TaskType.FEATURE,
                    TaskType.BUG,
                    TaskType.REFACTOR,
                    TaskType.DOCUMENTATION,
                    TaskType.REVIEW,
                ],
                cost_per_token=0.000008,
                latency=2.0,
                max_tokens=8192,
            ),
            "llama3.2:1b": ModelCapability(
                name="llama3.2:1b",
                provider="ollama",
                complexity_support=[TaskComplexity.SIMPLE],
                task_types=[TaskType.FEATURE, TaskType.BUG, TaskType.TESTING],
                cost_per_token=0.000001,
                latency=0.8,
                max_tokens=2048,
            ),
            "llama3.2:8b": ModelCapability(
                name="llama3.2:8b",
                provider="ollama",
                complexity_support=[TaskComplexity.SIMPLE, TaskComplexity.MODERATE],
                task_types=[
                    TaskType.FEATURE,
                    TaskType.BUG,
                    TaskType.REFACTOR,
                    TaskType.TESTING,
                ],
                cost_per_token=0.000003,
                latency=1.2,
                max_tokens=4096,
            ),
            "claude-3-sonnet": ModelCapability(
                name="claude-3-sonnet",
                provider="anthropic",
                complexity_support=[
                    TaskComplexity.SIMPLE,
                    TaskComplexity.MODERATE,
                    TaskComplexity.COMPLEX,
                ],
                task_types=[
                    TaskType.FEATURE,
                    TaskType.BUG,
                    TaskType.REFACTOR,
                    TaskType.DOCUMENTATION,
                    TaskType.REVIEW,
                ],
                cost_per_token=0.00001,
                latency=1.8,
                max_tokens=8192,
            ),
        }

    def add_model_capability(self, capability: ModelCapability) -> None:
        """Add or update a model capability.

        Args:
            capability: The ModelCapability to add or update.
        """
        self._model_capabilities[capability.name] = capability

    def get_model_capability(self, model_name: str) -> Optional[ModelCapability]:
        """Get the capability information for a specific model.

        Args:
            model_name: The name of the model.

        Returns:
            The ModelCapability if found, None otherwise.
        """
        return self._model_capabilities.get(model_name)

    def select_model(
        self,
        available_models: List[str],
        task_complexity: TaskComplexity,
        task_type: TaskType,
        provider: str,
        optimize_for: str = "balance",  # balance | cost | performance
    ) -> Optional[str]:
        """Select the optimal model from available options based on task requirements.

        Args:
            available_models: List of model names to choose from.
            task_complexity: The complexity of the task.
            task_type: The type of task.
            provider: The provider to filter by.
            optimize_for: Optimization strategy ('balance', 'cost', or 'performance').

        Returns:
            The name of the selected model, or None if no suitable model found.
        """
        suitable_models = []

        for model_name in available_models:
            capability = self._model_capabilities.get(model_name)
            if not capability:
                continue

            # Filter by provider
            if capability.provider != provider:
                continue

            # Filter by complexity support
            if task_complexity not in capability.complexity_support:
                continue

            # Filter by task type support
            if task_type not in capability.task_types:
                continue

            suitable_models.append(capability)

        if not suitable_models:
            return None

        # Apply optimization strategy
        if optimize_for == "cost":
            return min(suitable_models, key=lambda m: m.cost_per_token).name
        elif optimize_for == "performance":
            # Performance is inversely related to latency, and directly related to max_tokens
            # Higher max_tokens and lower latency are better
            return max(
                suitable_models,
                key=lambda m: (m.max_tokens / m.latency if m.latency > 0 else float('inf')),
            ).name
        else:  # balance
            # Balanced score: (max_tokens / (cost * latency))
            return max(
                suitable_models,
                key=lambda m: (
                    m.max_tokens / (m.cost_per_token * m.latency)
                    if m.cost_per_token > 0 and m.latency > 0
                    else 0
                ),
            ).name

    def get_available_models_for_provider(self, provider: str) -> List[str]:
        """Get all available model names for a specific provider.

        Args:
            provider: The provider name.

        Returns:
            List of model names available for the provider.
        """
        return [
            model_name
            for model_name, capability in self._model_capabilities.items()
            if capability.provider == provider
        ]
