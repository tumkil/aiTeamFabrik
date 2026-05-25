"""Tests for factory.core.model_selector module."""

import pytest

from factory.core.model_selector import (
    ModelSelector,
    TaskComplexity,
    TaskType,
    ModelCapability,
)


class TestModelSelector:
    """Tests for ModelSelector class."""

    def test_initialization(self):
        """Test that ModelSelector initializes with default models."""
        selector = ModelSelector()
        assert len(selector._model_capabilities) > 0

    def test_add_model_capability(self):
        """Test adding a new model capability."""
        selector = ModelSelector()
        new_capability = ModelCapability(
            name="test-model",
            provider="test-provider",
            complexity_support=[TaskComplexity.SIMPLE],
            task_types=[TaskType.FEATURE],
            cost_per_token=0.000001,
            latency=1.0,
            max_tokens=2048,
        )
        selector.add_model_capability(new_capability)
        assert selector.get_model_capability("test-model") is not None

    def test_get_model_capability(self):
        """Test retrieving a model capability."""
        selector = ModelSelector()
        capability = selector.get_model_capability("devstral-2:123b-cloud")
        assert capability is not None
        assert capability.name == "devstral-2:123b-cloud"

    def test_get_nonexistent_model_capability(self):
        """Test retrieving a non-existent model capability."""
        selector = ModelSelector()
        capability = selector.get_model_capability("nonexistent-model")
        assert capability is None

    def test_select_model_simple_feature(self):
        """Test selecting a model for a simple feature task."""
        selector = ModelSelector()
        available_models = ["devstral-2:123b-cloud", "llama3.2:1b"]
        selected = selector.select_model(
            available_models=available_models,
            task_complexity=TaskComplexity.SIMPLE,
            task_type=TaskType.FEATURE,
            provider="ollama",
        )
        assert selected in available_models

    def test_select_model_complex_review(self):
        """Test selecting a model for a complex review task."""
        selector = ModelSelector()
        available_models = ["mistral-large-3:675b-cloud", "claude-3-sonnet"]
        selected = selector.select_model(
            available_models=available_models,
            task_complexity=TaskComplexity.COMPLEX,
            task_type=TaskType.REVIEW,
            provider="ollama",  # mistral-large-3:675b-cloud is from ollama
        )
        assert selected == "mistral-large-3:675b-cloud"

    def test_select_model_optimize_cost(self):
        """Test selecting a model optimized for cost."""
        selector = ModelSelector()
        available_models = ["devstral-2:123b-cloud", "llama3.2:1b"]
        selected = selector.select_model(
            available_models=available_models,
            task_complexity=TaskComplexity.SIMPLE,
            task_type=TaskType.FEATURE,
            provider="ollama",
            optimize_for="cost",
        )
        # llama3.2:1b should be selected as it has lower cost
        assert selected == "llama3.2:1b"

    def test_select_model_optimize_performance(self):
        """Test selecting a model optimized for performance."""
        selector = ModelSelector()
        available_models = ["devstral-2:123b-cloud", "llama3.2:8b"]
        selected = selector.select_model(
            available_models=available_models,
            task_complexity=TaskComplexity.MODERATE,
            task_type=TaskType.FEATURE,
            provider="ollama",
            optimize_for="performance",
        )
        # llama3.2:8b should be selected as it has better performance metrics
        # (lower latency: 1.2 vs 1.5, same max_tokens: 4096)
        assert selected == "llama3.2:8b"

    def test_select_model_no_suitable(self):
        """Test selecting a model when no suitable models are available."""
        selector = ModelSelector()
        available_models = ["llama3.2:1b"]  # Only supports simple tasks
        selected = selector.select_model(
            available_models=available_models,
            task_complexity=TaskComplexity.COMPLEX,
            task_type=TaskType.REVIEW,
            provider="ollama",
        )
        assert selected is None

    def test_select_model_wrong_provider(self):
        """Test selecting a model with wrong provider."""
        selector = ModelSelector()
        available_models = ["claude-3-sonnet"]  # Anthropic provider
        selected = selector.select_model(
            available_models=available_models,
            task_complexity=TaskComplexity.COMPLEX,
            task_type=TaskType.REVIEW,
            provider="ollama",  # Requesting ollama provider
        )
        assert selected is None

    def test_get_available_models_for_provider(self):
        """Test getting available models for a specific provider."""
        selector = ModelSelector()
        ollama_models = selector.get_available_models_for_provider("ollama")
        assert len(ollama_models) > 0
        assert all("ollama" in selector.get_model_capability(m).provider for m in ollama_models)

    def test_select_model_balance_strategy(self):
        """Test the default balance strategy for model selection."""
        selector = ModelSelector()
        available_models = ["devstral-2:123b-cloud", "llama3.2:8b"]
        selected = selector.select_model(
            available_models=available_models,
            task_complexity=TaskComplexity.MODERATE,
            task_type=TaskType.FEATURE,
            provider="ollama",
            optimize_for="balance",
        )
        # Should select the model with best balance of cost, latency, and max_tokens
        assert selected in available_models
