"""Token Budget Manager — central authority for LLM token accounting and enforcement."""

from __future__ import annotations

import copy
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Literal, Optional

import yaml
from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)

# Lock timeout: 10 seconds max wait for budget I/O
_LOCK_TIMEOUT = 10


class BudgetExceededError(Exception):
    """Raised when a token consumption would exceed the configured budget."""
    pass


@dataclass
class TokenUsage:
    """Aggregate token usage for a single agent."""
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


@dataclass
class UsageHistoryEntry:
    """Single entry in the append-only usage ledger."""
    timestamp: str
    agent: str
    sprint: str
    input_tokens: int
    output_tokens: int
    model: Optional[str] = None


class TokenBudgetManager:
    """Central authority for tracking and enforcing token budgets across all agents.
    
    Responsibilities:
    - Pre-flight budget checks (can_consume)
    - Post-flight accounting (consume)
    - Persistence to config/token_usage.yml
    - Thread-safe concurrent updates
    """
    
    def __init__(self, config_path: Path, usage_path: Path):
        self.config_path = Path(config_path)
        self.usage_path = Path(usage_path)
        self._lock = threading.RLock()
        self._file_lock = FileLock(str(self.usage_path) + ".lock")
        
        # Load configuration
        self._config = self._load_config()
        
        # Load or initialize usage data; persist it if newly created.
        # Use double-checked locking so we don't overwrite a file that another
        # process just created between our existence check and our write.
        self._usage_data = self._load_or_init_usage()
        if not self.usage_path.exists():
            with self._file_lock:
                if not self.usage_path.exists():
                    self._save_usage()
    
    # --- Configuration loading ---
    
    def _load_config(self) -> dict:
        """Load token budget configuration from factory.yml."""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        if "token_budget" not in config:
            raise ValueError("factory.yml must contain a 'token_budget' section")

        tb = copy.deepcopy(config["token_budget"])
        # Inject sprint key from top-level config when not present inside token_budget.
        # Real factory.yml has sprint at the top level; test configs may embed it directly.
        if "sprint" not in tb and "sprint" in config:
            tb["sprint"] = {"current": str(config["sprint"].get("current", "1"))}

        return tb
    
    # --- Usage data management ---
    
    def _load_or_init_usage(self) -> dict:
        """Load usage data from file or initialize empty structure."""
        if not self.usage_path.exists():
            return self._init_empty_usage()

        try:
            with open(self.usage_path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logger.warning("Corrupt usage file, resetting: %s", exc)
            return self._init_empty_usage()

        if data is None or "version" not in data:
            return self._init_empty_usage()

        # Guard against corrupted/hand-edited files where history was set to null
        if not isinstance(data.get("history"), list):
            data["history"] = []

        return data
    
    def _init_empty_usage(self) -> dict:
        """Initialize empty usage data structure."""
        current_sprint = self._config["sprint"]["current"]
        today = date.today().isoformat()
        
        return {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "sprints": {
                current_sprint: {
                    "agents": {}
                }
            },
            "days": {
                today: {
                    "agents": {}
                }
            },
            "history": []
        }
    
    def _save_usage(self) -> None:
        """Atomically save usage data to file.
        
        Uses file-level lock (FileLock) for cross-process safety and
        atomic write pattern (temp file + os.replace) for crash safety.
        Caller must hold the thread lock (_lock).
        """
        # Ensure directory exists
        self.usage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first
        temp_path = self.usage_path.with_suffix(self.usage_path.suffix + ".tmp")
        
        with open(temp_path, "w") as f:
            yaml.safe_dump(self._usage_data, f, sort_keys=False)
        
        # Atomic replace
        os.replace(temp_path, self.usage_path)
    
    # --- Core operations ---
    
    def consume(self, agent: str, input_tokens: int, output_tokens: int,
                model: Optional[str] = None) -> None:
        """Record token consumption for an agent.
        
        Updates daily and sprint aggregates, and appends to history ledger.
        Thread-safe (via _lock) and cross-process safe (via FileLock).
        """
        with self._lock:
            # Acquire file lock for cross-process safety
            try:
                self._file_lock.acquire(timeout=_LOCK_TIMEOUT)
            except Timeout:
                logger.warning("Could not acquire file lock for budget update; retry later")
                return
            try:
                # Re-read from disk to pick up any writes from other processes
                # before mutating so we never overwrite concurrent updates.
                self._usage_data = self._load_or_init_usage()

                today = date.today().isoformat()
                current_sprint = self._config["sprint"]["current"]

                # Initialize agent entries if they don't exist
                if today not in self._usage_data["days"]:
                    self._usage_data["days"][today] = {"agents": {}}
                
                if agent not in self._usage_data["days"][today]["agents"]:
                    self._usage_data["days"][today]["agents"][agent] = {
                        "input": 0,
                        "output": 0,
                        "calls": 0
                    }
                
                if current_sprint not in self._usage_data["sprints"]:
                    self._usage_data["sprints"][current_sprint] = {"agents": {}}
                
                if agent not in self._usage_data["sprints"][current_sprint]["agents"]:
                    self._usage_data["sprints"][current_sprint]["agents"][agent] = {
                        "total_input": 0,
                        "total_output": 0
                    }
                
                # Update daily aggregates
                day_data = self._usage_data["days"][today]["agents"][agent]
                day_data["input"] += input_tokens
                day_data["output"] += output_tokens
                day_data["calls"] += 1
                
                # Update sprint aggregates
                sprint_data = self._usage_data["sprints"][current_sprint]["agents"][agent]
                sprint_data["total_input"] += input_tokens
                sprint_data["total_output"] += output_tokens
                
                # Append to history ledger; cap to avoid unbounded file growth.
                self._usage_data["history"].append({
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "agent": agent,
                    "sprint": current_sprint,
                    "input": input_tokens,
                    "output": output_tokens,
                    "model": model
                })
                _MAX_HISTORY = 500
                if len(self._usage_data["history"]) > _MAX_HISTORY:
                    self._usage_data["history"] = self._usage_data["history"][-_MAX_HISTORY:]
                
                # Update last modified timestamp
                self._usage_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
                
                # Persist changes
                self._save_usage()
            finally:
                # Always release file lock
                self._file_lock.release()
    
    def remaining(self, agent: str, scope: Literal["daily", "sprint"] = "daily") -> int:
        """Return remaining token budget for an agent in the given scope."""
        with self._lock:
            if scope == "daily":
                used = self._get_daily_used(agent)
                limit = self._get_agent_limit(agent, "daily")
            else:  # sprint
                used = self._get_sprint_used(agent)
                limit = self._get_agent_limit(agent, "sprint")
            
            return max(0, limit - used)
    
    def is_over_budget(self, agent: str,
                      scope: Literal["daily", "sprint"] = "daily") -> bool:
        """Check if an agent has exceeded its budget in the given scope."""
        with self._lock:
            if scope == "daily":
                used = self._get_daily_used(agent)
                limit = self._get_agent_limit(agent, "daily")
            else:  # sprint
                used = self._get_sprint_used(agent)
                limit = self._get_agent_limit(agent, "sprint")
            
            return used > limit
    
    def can_consume(self, agent: str, estimated_tokens: int) -> tuple[bool, str]:
        """Check if a proposed token consumption is allowed.

        Returns (allowed, reason). If enforcement is 'strict', returns (False, reason)
        when consumption would exceed budget.

        Note: reads from in-memory state last refreshed by consume(). In a
        multi-process setup, a concurrent process may have updated the file
        between this process's last consume() and this call, so the check is
        best-effort. The window is bounded: consume() always reloads from disk
        before writing, so the on-disk aggregate is always correct even if the
        pre-check passes on slightly stale data.
        """
        with self._lock:
            enforcement = self._config.get("enforcement", "strict")
            
            if enforcement == "off":
                return True, ""
            
            # Check daily budget
            daily_used = self._get_daily_used(agent)
            daily_limit = self._get_agent_limit(agent, "daily")
            daily_remaining = daily_limit - daily_used
            
            if daily_remaining < estimated_tokens:
                if enforcement == "strict":
                    return False, f"Daily budget exceeded (used: {daily_used}, limit: {daily_limit})"
                else:  # warn
                    return True, f"⚠ Daily budget near limit (used: {daily_used}, limit: {daily_limit})"
            
            # Check sprint budget
            sprint_used = self._get_sprint_used(agent)
            sprint_limit = self._get_agent_limit(agent, "sprint")
            sprint_remaining = sprint_limit - sprint_used
            
            if sprint_remaining < estimated_tokens:
                if enforcement == "strict":
                    return False, f"Sprint budget exceeded (used: {sprint_used}, limit: {sprint_limit})"
                else:  # warn
                    return True, f"⚠ Sprint budget near limit (used: {sprint_used}, limit: {sprint_limit})"
            
            return True, ""
    
    # --- Helper methods ---
    
    def _get_daily_used(self, agent: str) -> int:
        """Get total tokens used by agent today (input + output)."""
        today = date.today().isoformat()
        if today not in self._usage_data["days"]:
            return 0
        if agent not in self._usage_data["days"][today]["agents"]:
            return 0
        day_data = self._usage_data["days"][today]["agents"][agent]
        return day_data.get("input", 0) + day_data.get("output", 0)
    
    def _get_sprint_used(self, agent: str) -> int:
        """Get total tokens used by agent in current sprint (input + output)."""
        current_sprint = self._config["sprint"]["current"]
        if current_sprint not in self._usage_data["sprints"]:
            return 0
        if agent not in self._usage_data["sprints"][current_sprint]["agents"]:
            return 0
        sprint_data = self._usage_data["sprints"][current_sprint]["agents"][agent]
        return sprint_data.get("total_input", 0) + sprint_data.get("total_output", 0)
    
    def _get_agent_limit(self, agent: str, scope: Literal["daily", "sprint"]) -> int:
        """Get the token limit for an agent in the given scope."""
        # Check agent-specific limits first
        if "agents" in self._config and agent in self._config["agents"]:
            agent_config = self._config["agents"][agent]
            if scope in agent_config:
                return agent_config[scope]
        
        # Fall back to defaults
        if "defaults" in self._config and scope in self._config["defaults"]:
            return self._config["defaults"][scope]
        
        # Final fallback
        return 100000 if scope == "daily" else 1000000
    
    # --- Reporting ---
    
    def usage_report(self) -> dict:
        """Generate a usage report for all agents.
        
        Consumed by `factory status` CLI.
        """
        with self._lock:
            report = {
                "agents": {},
                "updated_at": self._usage_data.get("updated_at", ""),
                "sprint": self._config["sprint"]["current"]
            }
            
            today = date.today().isoformat()
            current_sprint = self._config["sprint"]["current"]
            
            # Get all known agents from config
            agents = set()
            if "agents" in self._config:
                agents.update(self._config["agents"].keys())
            
            # Also include any agents that have usage data
            if today in self._usage_data["days"]:
                agents.update(self._usage_data["days"][today]["agents"].keys())
            if current_sprint in self._usage_data["sprints"]:
                agents.update(self._usage_data["sprints"][current_sprint]["agents"].keys())
            
            for agent in sorted(agents):
                daily_used = self._get_daily_used(agent)
                daily_limit = self._get_agent_limit(agent, "daily")
                sprint_used = self._get_sprint_used(agent)
                sprint_limit = self._get_agent_limit(agent, "sprint")
                
                daily_pct = (daily_used / daily_limit) * 100 if daily_limit > 0 else 0
                sprint_pct = (sprint_used / sprint_limit) * 100 if sprint_limit > 0 else 0
                
                report["agents"][agent] = {
                    "daily_used": daily_used,
                    "daily_limit": daily_limit,
                    "sprint_used": sprint_used,
                    "sprint_limit": sprint_limit,
                    "daily_pct": daily_pct,
                    "sprint_pct": sprint_pct
                }
            
            return report
    
    def reset(self, scope: Literal["daily", "sprint"], key: str) -> None:
        """Reset usage data for a specific agent in the given scope.

        Used for testing or manual adjustments. key is the agent name.
        """
        with self._lock:
            try:
                self._file_lock.acquire(timeout=_LOCK_TIMEOUT)
            except Timeout:
                logger.warning("Could not acquire file lock for reset; skipping")
                return
            try:
                if scope == "daily":
                    today = date.today().isoformat()
                    day_agents = self._usage_data.get("days", {}).get(today, {}).get("agents", {})
                    day_agents.pop(key, None)
                else:  # sprint
                    current_sprint = self._config["sprint"]["current"]
                    sprint_agents = (
                        self._usage_data.get("sprints", {})
                        .get(current_sprint, {})
                        .get("agents", {})
                    )
                    sprint_agents.pop(key, None)
                self._save_usage()
            finally:
                self._file_lock.release()
