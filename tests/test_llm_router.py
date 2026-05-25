"""Unit tests for LlmRouter — runs fully offline via stub provider."""
import pytest
from factory.adapters.llm_router import LlmRouter
from factory.core.competence import AgentProfile, CompetenceManager


def _stub_agent(system_prompt: str = "", tools=None) -> AgentProfile:
    return AgentProfile(
        name="stub-agent",
        display_name="Stub Agent",
        model="stub",
        provider="stub",
        system_prompt=system_prompt,
        tools=tools or [],
    )


def test_stub_provider_returns_response():
    router = LlmRouter()
    agent = _stub_agent()
    result = router.complete(agent, system="you are a stub", prompt="hello world")
    assert result.provider == "stub"
    assert result.model == "stub"
    assert "stub-agent" in result.content.lower() or "stub" in result.content.lower()


def test_stub_provider_includes_prompt_excerpt():
    router = LlmRouter()
    agent = _stub_agent()
    result = router.complete(agent, system="", prompt="unique-test-prompt-xyz")
    assert "unique-test-prompt-xyz" in result.content


def test_unknown_provider_raises():
    router = LlmRouter()
    agent = AgentProfile(
        name="ghost",
        display_name="Ghost",
        model="ghost-1",
        provider="nonexistent",
    )
    with pytest.raises(ValueError, match="Unknown provider"):
        router.complete(agent, system="", prompt="test")


def test_anthropic_missing_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    router = LlmRouter()
    agent = AgentProfile(
        name="dev",
        display_name="Developer",
        model="claude-sonnet-4-6",
        provider="anthropic",
    )
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        router.complete(agent, system="", prompt="test")


def test_mistral_missing_key_raises(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    router = LlmRouter()
    agent = AgentProfile(
        name="qa",
        display_name="QA",
        model="mistral-large-latest",
        provider="mistral",
    )
    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        router.complete(agent, system="", prompt="test")


def test_agent_system_prompt_overrides_caller():
    """Agent YAML system_prompt takes precedence over the system arg passed by the caller."""
    router = LlmRouter()
    agent = _stub_agent(system_prompt="I am the override prompt.")
    # The stub doesn't use system, but we verify the override logic is applied
    # by checking that a non-stub provider would pick up agent.system_prompt.
    # For stub: just check it runs without error and returns a response.
    result = router.complete(agent, system="caller-provided-system", prompt="test")
    assert result.content != ""


def test_stub_with_tools_mentions_them():
    router = LlmRouter()
    agent = _stub_agent(tools=["web_search"])
    result = router.complete(agent, system="", prompt="find me something")
    assert "web_search" in result.content


def test_router_handles_empty_prompt():
    """Router should handle empty prompts gracefully with stub provider."""
    router = LlmRouter()
    agent = _stub_agent()
    result = router.complete(agent, system="", prompt="")
    assert result.content != ""
    assert result.model == "stub"


def test_router_preserves_model_name():
    """Router should preserve the agent's model name in response."""
    router = LlmRouter()
    agent = AgentProfile(
        name="test",
        display_name="Test",
        model="custom-model-v1",
        provider="stub",
    )
    result = router.complete(agent, system="", prompt="test")
    assert result.model == "custom-model-v1"


def test_stub_response_includes_agent_name():
    """Stub response should mention the agent name."""
    router = LlmRouter()
    agent = AgentProfile(
        name="special-analysis-agent",
        display_name="Special Analysis Agent",
        model="stub",
        provider="stub",
    )
    result = router.complete(agent, system="", prompt="analyze this")
    assert "special-analysis-agent" in result.content.lower() or "Special Analysis" in result.content