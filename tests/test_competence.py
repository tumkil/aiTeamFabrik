"""Tests for factory.core.competence module."""

import pytest

from factory.core.competence import AgentProfile


class TestAgentProfileResolveModel:
    """Tests for AgentProfile.resolve_model() method."""

    def test_string_model_returns_directly(self):
        """When model is a string, return it directly."""
        agent = AgentProfile(
            name="test",
            display_name="Test",
            model="single-model",
            provider="stub",
        )
        assert agent.resolve_model() == "single-model"

    def test_dict_model_exact_match(self):
        """When model is a dict with exact provider key, return that value."""
        agent = AgentProfile(
            name="test",
            display_name="Test",
            model={"ollama": "llama3.2", "mistral": "devstral-2512"},
            provider="ollama",
        )
        assert agent.resolve_model() == "llama3.2"

    def test_dict_model_case_insensitive_match(self):
        """Provider matching is case-insensitive."""
        # Test lowercase provider with mixed-case dict keys
        agent1 = AgentProfile(
            name="test1",
            display_name="Test",
            model={"Ollama": "llama3.2", "Mistral": "devstral-2512"},
            provider="ollama",
        )
        assert agent1.resolve_model() == "llama3.2"

        # Test uppercase provider
        agent2 = AgentProfile(
            name="test2",
            display_name="Test",
            model={"Ollama": "llama3.2", "Mistral": "devstral-2512"},
            provider="OLLAMA",
        )
        assert agent2.resolve_model() == "llama3.2"

        # Test mixed-case provider
        agent3 = AgentProfile(
            name="test3",
            display_name="Test",
            model={"Ollama": "llama3.2", "Mistral": "devstral-2512"},
            provider="OlLaMa",
        )
        assert agent3.resolve_model() == "llama3.2"

    def test_dict_model_mixed_case_keys(self):
        """Dict keys can be any case, matching is case-insensitive."""
        agent = AgentProfile(
            name="test",
            display_name="Test",
            model={"OLLAMA": "llama3.2", "mistral": "devstral-2512"},
            provider="ollama",
        )
        assert agent.resolve_model() == "llama3.2"

    def test_dict_model_default_fallback(self):
        """When provider not found, use 'default' key."""
        agent = AgentProfile(
            name="test",
            display_name="Test",
            model={"ollama": "llama3.2", "default": "fallback-model"},
            provider="anthropic",
        )
        assert agent.resolve_model() == "fallback-model"

    def test_dict_model_missing_provider_no_default_raises(self):
        """When provider not found and no default, raise ValueError."""
        agent = AgentProfile(
            name="test",
            display_name="Test",
            model={"ollama": "llama3.2"},
            provider="anthropic",
        )
        with pytest.raises(ValueError) as exc_info:
            agent.resolve_model()
        
        assert "anthropic" in str(exc_info.value)
        assert "test" in str(exc_info.value)
        assert "ollama" in str(exc_info.value)  # Available key in error message

    def test_dict_model_empty_string_value_raises(self):
        """Empty string values in model dict raise ValueError at construction time."""
        with pytest.raises(ValueError) as exc_info:
            AgentProfile(
                name="test",
                display_name="Test",
                model={"ollama": ""},
                provider="ollama",
            )
        assert "empty or non-string model value" in str(exc_info.value)

    def test_dict_model_with_default_empty_string_raises(self):
        """Empty string value for 'default' key raises ValueError at construction time."""
        with pytest.raises(ValueError) as exc_info:
            AgentProfile(
                name="test",
                display_name="Test",
                model={"ollama": "llama3.2", "default": ""},
                provider="anthropic",
            )
        assert "empty or non-string model value" in str(exc_info.value)

    def test_string_model_empty_raises(self):
        """Empty string model raises ValueError at construction time."""
        with pytest.raises(ValueError) as exc_info:
            AgentProfile(
                name="test",
                display_name="Test",
                model="",
                provider="stub",
            )
        assert "empty or whitespace-only" in str(exc_info.value)

    def test_dict_model_empty_dict_raises(self):
        """Empty model dict raises ValueError at construction time."""
        with pytest.raises(ValueError) as exc_info:
            AgentProfile(
                name="test",
                display_name="Test",
                model={},
                provider="stub",
            )
        assert "empty model dict" in str(exc_info.value)

    def test_dict_model_case_duplicate_keys_raises(self):
        """Case-equivalent duplicate keys in model dict raise ValueError."""
        # Use explicit dict construction to ensure both keys are present
        # (Python dict literals would deduplicate to a single key)
        model_dict = dict([("Ollama", "llama3.2"), ("ollama", "mistral")])
        with pytest.raises(ValueError) as exc_info:
            AgentProfile(
                name="test",
                display_name="Test",
                model=model_dict,
                provider="ollama",
            )
        assert "duplicate model keys" in str(exc_info.value).lower()
        assert "'ollama' conflicts with existing key 'Ollama'" in str(exc_info.value)

    def test_provider_empty_raises(self):
        """Empty provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            AgentProfile(
                name="test",
                display_name="Test",
                model="llama3.2",
                provider="",
            )
        assert "empty or whitespace-only provider" in str(exc_info.value)

    def test_provider_whitespace_only_raises(self):
        """Whitespace-only provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            AgentProfile(
                name="test",
                display_name="Test",
                model="llama3.2",
                provider="   ",
            )
        assert "empty or whitespace-only provider" in str(exc_info.value)

    def test_model_invalid_type_raises(self):
        """Non-string/non-dict model raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            AgentProfile(
                name="test",
                display_name="Test",
                model=42,
                provider="stub",
            )
        assert "must be a str or dict" in str(exc_info.value)

    def test_dict_model_multiple_providers(self):
        """Test with multiple providers in dict."""
        agent_mistral = AgentProfile(
            name="test_mistral",
            display_name="Test",
            model={
                "ollama": "llama3.2",
                "mistral": "devstral-2512",
                "anthropic": "claude-3-sonnet",
            },
            provider="mistral",
        )
        assert agent_mistral.resolve_model() == "devstral-2512"

        agent_anthropic = AgentProfile(
            name="test_anthropic",
            display_name="Test",
            model={
                "ollama": "llama3.2",
                "mistral": "devstral-2512",
                "anthropic": "claude-3-sonnet",
            },
            provider="anthropic",
        )
        assert agent_anthropic.resolve_model() == "claude-3-sonnet"
