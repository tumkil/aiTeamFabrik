# Copyright 2024 SoftwareTeamFabrik contributors
# SPDX-License-Identifier: MIT

"""Test that _budget_manager.consume is NOT called when BudgetExceededError is raised.

Verifies the guard in LlmRouter.complete() that prevents token accounting
from recording consumption for requests that were rejected by the budget
pre-flight check.
"""

import tempfile
import unittest.mock
from pathlib import Path
from unittest.mock import patch

import pytest
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
                "sprint": 1000000,
            },
            "agents": {
                "test-agent": {
                    "daily": 1000,
                    "sprint": 10000,
                },
            },
            "sprint": {
                "current": "sprint-3",
                "start": "2026-05-26",
                "end": "2026-06-09",
            },
        },
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
                "agents": {},
            },
        },
        "days": {
            "2026-05-26": {
                "agents": {},
            },
        },
        "history": [],
    }

    with open(usage_path, "w") as f:
        yaml.safe_dump(usage_data, f)


class TestBudgetExceededNoConsume:
    """Verify consume() is never invoked when BudgetExceededError is raised."""

    def test_consume_not_called_on_budget_exceeded(self):
        """When BudgetExceededError is raised, consume must NOT be called.

        This guards against a regression where the finally block might
        still record token usage for a request that was rejected outright.
        """
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
                system_prompt="",
            )

            # Exhaust the daily budget (1000 tokens) so the next call is rejected.
            router._budget_manager.consume("test-agent", input_tokens=1001, output_tokens=0)

            # Patch consume so we can assert it is never called.
            with patch.object(
                router._budget_manager, "consume", side_effect=AssertionError("consume must not be called")
            ) as mock_consume:
                with pytest.raises(BudgetExceededError):
                    router.complete(agent, system="test", prompt="this should fail")

                mock_consume.assert_not_called()

    def test_consume_not_called_on_budget_exceeded_anthropic_complete(self):
        """When anthropic_complete rejects due to budget, consume must NOT be called."""
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
                system_prompt="",
            )

            # Exhaust the daily budget so the next call is rejected.
            router._budget_manager.consume("test-agent", input_tokens=1001, output_tokens=0)

            # Mock the anthropic module to avoid import errors
            import os
            os.environ["ANTHROPIC_API_KEY"] = "test-key"

            # Patch consume so we can assert it is never called.
            with patch.object(
                router._budget_manager, "consume", side_effect=AssertionError("consume must not be called")
            ) as mock_consume, \
                 unittest.mock.patch.dict('sys.modules', {'anthropic': unittest.mock.MagicMock()}):
                with pytest.raises(BudgetExceededError):
                    router.anthropic_complete(
                        agent=agent,
                        system="test system",
                        messages=[{"role": "user", "content": "test message"}],
                    )

                mock_consume.assert_not_called()

            # Clean up
            del os.environ["ANTHROPIC_API_KEY"]

    def test_consume_not_called_on_budget_exceeded_ollama_complete(self):
        """When ollama_complete rejects due to budget, consume must NOT be called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "factory.yml"
            usage_path = Path(tmpdir) / "token_usage.yml"

            _create_test_config(config_path, enforcement="strict")
            _create_empty_usage_file(usage_path)

            router = LlmRouter(config_path=config_path, usage_path=usage_path)

            agent = AgentProfile(
                name="test-agent",
                display_name="Test Agent",
                model="llama3",
                provider="ollama",
                system_prompt="",
            )

            # Exhaust the daily budget so the next call is rejected.
            router._budget_manager.consume("test-agent", input_tokens=1001, output_tokens=0)

            # Patch consume so we can assert it is never called.
            with patch.object(
                router._budget_manager, "consume", side_effect=AssertionError("consume must not be called")
            ) as mock_consume:
                with pytest.raises(BudgetExceededError):
                    router.ollama_complete(
                        agent=agent,
                        system="test system",
                        prompt="test prompt",
                    )

                mock_consume.assert_not_called()

    def test_consume_is_called_on_successful_call(self):
        """Positive control: consume IS called when the budget check passes."""
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
                system_prompt="",
            )

            with patch.object(router._budget_manager, "consume", wraps=router._budget_manager.consume) as mock_consume:
                router.complete(agent, system="test", prompt="hello world")

                # consume must have been called exactly once for the successful call
                mock_consume.assert_called_once()

    def test_budget_exceeded_does_not_modify_usage_file(self):
        """Verify the token_usage.yml file is unchanged after a BudgetExceededError."""
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
                system_prompt="",
            )

            # Exhaust the budget
            router._budget_manager.consume("test-agent", input_tokens=1001, output_tokens=0)

            # Snapshot the usage file contents before the rejected call
            before = usage_path.read_text()

            with pytest.raises(BudgetExceededError):
                router.complete(agent, system="test", prompt="this should fail")

            # The usage file must be identical — no writes from consume()
            after = usage_path.read_text()
            assert before == after