# Copyright 2024 SoftwareTeamFabrik contributors
# SPDX-License-Identifier: MIT

"""Integration tests for LlmRouter with TokenBudgetManager."""
import pytest
import tempfile
import unittest.mock
from pathlib import Path
import yaml

from factory.adapters.llm_router import LlmRouter, BudgetExceededError
from factory.core.competence import AgentProfile


@pytest.fixture(autouse=True)
def _disable_ignore_budget_env(monkeypatch):
    """Ensure FACTORY_IGNORE_BUDGET is not set so budget logic is exercised."""
    monkeypatch.delenv("FACTORY_IGNORE_BUDGET", raising=False)


def _create_test_config(config_path: Path, enforcement: str = "strict"):
    """Create a minimal test configuration."""
    config = {
        "token_budget": {
            "enforcement": enforcement,
            "defaults": {
                "daily": 100000,
                "sprint": 1000000
            },
            "agents": {
                "test-agent": {
                    "daily": 1000,
                    "sprint": 10000
                }
            },
            "sprint": {
                "current": "sprint-3",
                "start": "2026-05-26",
                "end": "2026-06-09"
            }
        }
    }
    
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)


def _create_empty_usage_file(usage_path: Path):
    """Create an empty usage file."""
    usage_data = {
        "version": 1,
        "updated_at": "2026-05-26T10:00:00Z",
        "sprints": {
            "sprint-3": {
                "agents": {}
            }
        },
        "days": {
            "2026-05-26": {
                "agents": {}
            }
        },
        "history": []
    }
    
    with open(usage_path, "w") as f:
        yaml.safe_dump(usage_data, f)


def test_llm_router_budget_check():
    """Test that LlmRouter performs budget checks before making API calls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"
        
        _create_test_config(config_path, enforcement="strict")
        _create_empty_usage_file(usage_path)
        
        router = LlmRouter(config_path=config_path, usage_path=usage_path)
        
        agent = AgentProfile(
            name="test-agent",
            display_name="Test Agent",
            model="stub",
            provider="stub",
            system_prompt=""
        )
        
        # This should work - agent has 1000 token daily limit
        # Estimate for "hello world" is about 3 tokens (len("hello world") // 3)
        response = router.complete(agent, system="test", prompt="hello world")
        assert response.provider == "stub"


def test_llm_router_budget_exceeded():
    """Test that LlmRouter raises BudgetExceededError when budget is exceeded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"

        _create_test_config(config_path, enforcement="strict")
        _create_empty_usage_file(usage_path)

        router = LlmRouter(config_path=config_path, usage_path=usage_path)
        
        agent = AgentProfile(
            name="test-agent",
            display_name="Test Agent",
            model="stub",
            provider="stub",
            system_prompt=""
        )
        
        # Consume tokens directly through the router's budget manager to exceed the limit
        # The limit is 1000, so consume 1001 tokens
        router._budget_manager.consume("test-agent", input_tokens=1001, output_tokens=0)
        
        # Now try to make another call - should fail
        with pytest.raises(BudgetExceededError) as exc_info:
            router.complete(agent, system="test", prompt="this should fail")
        
        assert "budget check failed" in str(exc_info.value)


def test_llm_router_token_recording():
    """Test that LlmRouter records token usage after successful calls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"
        
        _create_test_config(config_path, enforcement="warn")
        _create_empty_usage_file(usage_path)
        
        router = LlmRouter(config_path=config_path, usage_path=usage_path)
        
        agent = AgentProfile(
            name="test-agent",
            display_name="Test Agent",
            model="stub",
            provider="stub",
            system_prompt=""
        )
        
        # Make a call
        response = router.complete(agent, system="test", prompt="hello world")
        
        # Verify usage was recorded
        from factory.core.token_budget import TokenBudgetManager
        budget_manager = TokenBudgetManager(config_path, usage_path)
        report = budget_manager.usage_report()
        
        assert "test-agent" in report["agents"]
        # Stub provider doesn't return token counts, so should be 0
        assert report["agents"]["test-agent"]["daily_used"] == 0


def test_llm_router_warn_mode():
    """Test that LlmRouter allows calls in warn mode even when over budget."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"
        
        _create_test_config(config_path, enforcement="warn")
        _create_empty_usage_file(usage_path)
        
        router = LlmRouter(config_path=config_path, usage_path=usage_path)
        
        agent = AgentProfile(
            name="test-agent",
            display_name="Test Agent",
            model="stub",
            provider="stub",
            system_prompt=""
        )
        
        # Consume tokens directly to exceed the limit
        router._budget_manager.consume("test-agent", input_tokens=1001, output_tokens=0)
        
        # In warn mode, this should still work
        response = router.complete(agent, system="test", prompt="this should work")
        assert response.provider == "stub"


def test_token_estimation():
    """Test the token estimation logic."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"
        
        _create_test_config(config_path)
        _create_empty_usage_file(usage_path)
        
        router = LlmRouter(config_path=config_path, usage_path=usage_path)
        
        # Estimation uses len(text) // 3 (tighter than // 4 to reduce budget bypass risk)
        assert router._estimate_tokens("") == 0
        assert router._estimate_tokens("hello") == 1        # 5 // 3 = 1
        assert router._estimate_tokens("hello world") == 3  # 11 // 3 = 3
        assert router._estimate_tokens("a" * 100) == 33     # 100 // 3 = 33


def test_anthropic_complete_budget_exceeded():
    """Test that anthropic_complete raises BudgetExceededError when budget is exceeded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"
        
        _create_test_config(config_path, enforcement="strict")
        _create_empty_usage_file(usage_path)
        
        router = LlmRouter(config_path=config_path, usage_path=usage_path)
        
        agent = AgentProfile(
            name="test-agent",
            display_name="Test Agent",
            model="claude-3-5-sonnet-20240620",
            provider="anthropic",
            system_prompt=""
        )
        
        # Set up API key to avoid missing key error
        import os
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        
        # Mock the anthropic module to avoid import errors
        with unittest.mock.patch.dict('sys.modules', {'anthropic': unittest.mock.MagicMock()}):
            # Consume tokens directly to exceed the limit
            router._budget_manager.consume("test-agent", input_tokens=1001, output_tokens=0)
            
            # Now try to call anthropic_complete - should fail with BudgetExceededError
            with pytest.raises(BudgetExceededError) as exc_info:
                router.anthropic_complete(
                    agent=agent,
                    system="test system",
                    messages=[{"role": "user", "content": "test message"}],
                    tools=None,
                    max_tokens=8192
                )
            
            assert "budget check failed" in str(exc_info.value)
        
        # Clean up
        del os.environ["ANTHROPIC_API_KEY"]