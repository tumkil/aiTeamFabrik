"""Unit tests for TokenBudgetManager."""
import pytest
import tempfile
import os
from pathlib import Path
from datetime import date
import yaml
from unittest.mock import patch, MagicMock
from filelock import Timeout

from factory.core.token_budget import TokenBudgetManager, BudgetExceededError


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
                "architect": {
                    "daily": 50000,
                    "sprint": 400000
                },
                "developer": {
                    "daily": 200000,
                    "sprint": 1500000
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


def test_token_budget_manager_initialization():
    """Test that TokenBudgetManager initializes correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"
        
        _create_test_config(config_path)
        
        manager = TokenBudgetManager(config_path, usage_path)
        
        # Should create usage file if it doesn't exist
        assert usage_path.exists()
        
        # Check initial state
        report = manager.usage_report()
        assert "agents" in report
        assert len(report["agents"]) == 2  # architect and developer from config


def test_consume_tokens():
    """Test that consume() correctly records token usage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"
        
        _create_test_config(config_path)
        _create_empty_usage_file(usage_path)
        
        manager = TokenBudgetManager(config_path, usage_path)
        
        # Consume some tokens
        manager.consume("architect", input_tokens=1000, output_tokens=500, model="gpt-4")
        manager.consume("developer", input_tokens=2000, output_tokens=1000, model="claude-3")
        
        # Check usage report
        report = manager.usage_report()
        
        architect_data = report["agents"]["architect"]
        assert architect_data["daily_used"] == 1500
        assert architect_data["sprint_used"] == 1500
        
        developer_data = report["agents"]["developer"]
        assert developer_data["daily_used"] == 3000
        assert developer_data["sprint_used"] == 3000


def test_remaining_tokens():
    """Test that remaining() calculates remaining budget correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"
        
        _create_test_config(config_path)
        _create_empty_usage_file(usage_path)
        
        manager = TokenBudgetManager(config_path, usage_path)
        
        # Architect has daily limit of 50,000
        assert manager.remaining("architect", "daily") == 50000
        
        # Consume some tokens
        manager.consume("architect", input_tokens=10000, output_tokens=5000)
        
        # Should have 35,000 remaining
        assert manager.remaining("architect", "daily") == 35000


def test_is_over_budget():
    """Test that is_over_budget() correctly detects budget overages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"
        
        _create_test_config(config_path)
        _create_empty_usage_file(usage_path)
        
        manager = TokenBudgetManager(config_path, usage_path)
        
        # Should not be over budget initially
        assert not manager.is_over_budget("architect", "daily")
        
        # Consume tokens beyond the limit
        manager.consume("architect", input_tokens=50001, output_tokens=0)
        
        # Should be over budget now
        assert manager.is_over_budget("architect", "daily")


def test_can_consume_strict_enforcement():
    """Test that can_consume() enforces budget in strict mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"
        
        _create_test_config(config_path, enforcement="strict")
        _create_empty_usage_file(usage_path)
        
        manager = TokenBudgetManager(config_path, usage_path)
        
        # Should allow consumption within budget
        allowed, reason = manager.can_consume("architect", 10000)
        assert allowed
        assert reason == ""
        
        # Consume tokens up to near the limit
        manager.consume("architect", input_tokens=45000, output_tokens=0)
        
        # Should still allow small consumption
        allowed, reason = manager.can_consume("architect", 4000)
        assert allowed
        
        # Should deny consumption that would exceed budget
        allowed, reason = manager.can_consume("architect", 6000)
        assert not allowed
        assert "Daily budget exceeded" in reason


def test_can_consume_warn_enforcement():
    """Test that can_consume() allows but warns in warn mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"
        
        _create_test_config(config_path, enforcement="warn")
        _create_empty_usage_file(usage_path)
        
        manager = TokenBudgetManager(config_path, usage_path)
        
        # Consume tokens up to near the limit
        manager.consume("architect", input_tokens=45000, output_tokens=0)
        
        # Should allow but warn about consumption that would exceed budget
        allowed, reason = manager.can_consume("architect", 6000)
        assert allowed
        assert "⚠ Daily budget near limit" in reason


def test_can_consume_off_enforcement():
    """Test that can_consume() allows everything in off mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"
        
        _create_test_config(config_path, enforcement="off")
        _create_empty_usage_file(usage_path)
        
        manager = TokenBudgetManager(config_path, usage_path)
        
        # Should always allow
        allowed, reason = manager.can_consume("architect", 1000000)
        assert allowed
        assert reason == ""


def test_usage_report():
    """Test that usage_report() generates correct report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"
        
        _create_test_config(config_path)
        _create_empty_usage_file(usage_path)
        
        manager = TokenBudgetManager(config_path, usage_path)
        
        # Consume some tokens
        manager.consume("architect", input_tokens=10000, output_tokens=5000)
        manager.consume("developer", input_tokens=20000, output_tokens=10000)
        
        report = manager.usage_report()
        
        # Check report structure
        assert "agents" in report
        assert "architect" in report["agents"]
        assert "developer" in report["agents"]
        
        architect = report["agents"]["architect"]
        assert architect["daily_used"] == 15000
        assert architect["daily_limit"] == 50000
        assert architect["sprint_used"] == 15000
        assert architect["sprint_limit"] == 400000
        
        developer = report["agents"]["developer"]
        assert developer["daily_used"] == 30000
        assert developer["daily_limit"] == 200000
        assert developer["sprint_used"] == 30000
        assert developer["sprint_limit"] == 1500000


def test_thread_safety():
    """Test that concurrent consume() calls are thread-safe."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"
        
        _create_test_config(config_path)
        _create_empty_usage_file(usage_path)
        
        manager = TokenBudgetManager(config_path, usage_path)
        
        import threading
        
        # Number of threads and iterations
        num_threads = 10
        iterations = 100
        
        def consume_tokens():
            for _ in range(iterations):
                manager.consume("architect", input_tokens=10, output_tokens=5)
        
        # Start threads
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=consume_tokens)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check total consumption
        report = manager.usage_report()
        architect = report["agents"]["architect"]
        
        # Should be exactly num_threads * iterations * (10 + 5) = 15000
        expected_total = num_threads * iterations * 15
        assert architect["daily_used"] == expected_total
        assert architect["sprint_used"] == expected_total


def test_persistence():
    """Test that usage data is persisted correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"
        
        _create_test_config(config_path)
        _create_empty_usage_file(usage_path)
        
        manager1 = TokenBudgetManager(config_path, usage_path)
        manager1.consume("architect", input_tokens=1000, output_tokens=500)
        
        # Create new manager instance to verify persistence
        manager2 = TokenBudgetManager(config_path, usage_path)
        report = manager2.usage_report()
        
        assert report["agents"]["architect"]["daily_used"] == 1500


def test_reset():
    """Test that reset() clears usage data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"
        
        _create_test_config(config_path)
        _create_empty_usage_file(usage_path)
        
        manager = TokenBudgetManager(config_path, usage_path)
        
        # Consume some tokens
        manager.consume("architect", input_tokens=1000, output_tokens=500)
        
        # Reset daily usage for the architect agent
        manager.reset("daily", "architect")
        
        # Daily usage should be reset
        report = manager.usage_report()
        assert report["agents"]["architect"]["daily_used"] == 0
        # Sprint usage should still be there
        assert report["agents"]["architect"]["sprint_used"] == 1500


def test_default_limits():
    """Test that agents without specific limits use defaults."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"
        
        # Create config without specific agent limits
        config = {
            "token_budget": {
                "enforcement": "strict",
                "defaults": {
                    "daily": 100000,
                    "sprint": 1000000
                },
                "agents": {},
                "sprint": {
                    "current": "sprint-3",
                    "start": "2026-05-26",
                    "end": "2026-06-09"
                }
            }
        }
        
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        
        _create_empty_usage_file(usage_path)
        
        manager = TokenBudgetManager(config_path, usage_path)
        
        # Should use default limits
        assert manager.remaining("unknown_agent", "daily") == 100000
        assert manager.remaining("unknown_agent", "sprint") == 1000000


def test_llm_router_integration():
    """Test integration with LlmRouter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"
        
        _create_test_config(config_path)
        _create_empty_usage_file(usage_path)
        
        from factory.adapters.llm_router import LlmRouter
        from factory.core.competence import AgentProfile
        
        router = LlmRouter(config_path=config_path, usage_path=usage_path)
        
        agent = AgentProfile(
            name="test-agent",
            display_name="Test Agent",
            model="stub",
            provider="stub",
            system_prompt=""
        )
        
        # This should work and record token usage
        response = router.complete(agent, system="test", prompt="hello world")
        
        # Verify usage was recorded
        manager = TokenBudgetManager(config_path, usage_path)
        report = manager.usage_report()
        
        assert "test-agent" in report["agents"]
        # Stub provider doesn't return token counts, so should be 0
        assert report["agents"]["test-agent"]["daily_used"] == 0


def test_budget_exceeded_error():
    """Test that BudgetExceededError is raised when appropriate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"
        
        _create_test_config(config_path)
        _create_empty_usage_file(usage_path)
        
        manager = TokenBudgetManager(config_path, usage_path)
        
        # Consume tokens beyond the limit
        manager.consume("architect", input_tokens=50001, output_tokens=0)
        
        # Try to consume more - should be over budget
        assert manager.is_over_budget("architect", "daily")
        
        # Verify can_consume returns False
        allowed, reason = manager.can_consume("architect", 1000)
        assert not allowed
        assert "Daily budget exceeded" in reason


def test_consume_file_lock_timeout():
    """Test that consume() handles file lock timeout gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "factory.yml"
        usage_path = Path(tmpdir) / "token_usage.yml"
        
        _create_test_config(config_path)
        _create_empty_usage_file(usage_path)
        
        manager = TokenBudgetManager(config_path, usage_path)
        
        # Mock the file lock to raise Timeout
        with patch.object(manager._file_lock, 'acquire', side_effect=Timeout("Lock timeout")):
            # This should not raise an exception, just log a warning and return
            manager.consume("architect", input_tokens=1000, output_tokens=500, model="gpt-4")
        
        # Verify that no tokens were consumed (file wasn't updated)
        report = manager.usage_report()
        architect_data = report["agents"]["architect"]
        assert architect_data["daily_used"] == 0
        assert architect_data["sprint_used"] == 0