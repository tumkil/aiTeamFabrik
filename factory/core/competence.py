from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class AgentProfile:
    name: str
    display_name: str
    # model can be a string (single model for all providers) or a dict mapping
    # provider names to model names. The dict supports a special "default" key
    # for fallback when the current provider is not found in the mapping.
    # Example: model: {mistral: mistral-large, default: mistral-small}
    model: str | dict[str, str]
    provider: str
    capabilities: list[str] = field(default_factory=list)
    task_labels: list[str] = field(default_factory=list)
    max_concurrent_tasks: int = 1
    system_prompt: str = ""
    tools: list[str] = field(default_factory=list)
    execution_mode: str = "plan"   # plan | execute
    temperature: Optional[float] = None  # provider default when None
    # Runtime state — not persisted to YAML
    current_task: Optional[str] = None
    description: str = ""

    def __post_init__(self):
        """Validate agent configuration after initialization."""
        # Validate provider is not empty
        if not self.provider or not self.provider.strip():
            raise ValueError(
                f"Agent '{self.name}' has empty or whitespace-only provider. "
                "Check the agent YAML configuration."
            )

        # Type guard for model
        if not isinstance(self.model, (str, dict)):
            raise ValueError(
                f"Agent '{self.name}' model must be a str or dict, got "
                f"{type(self.model).__name__}. Check the agent YAML configuration."
            )
        
        # Validate string model is not empty
        if isinstance(self.model, str):
            if not self.model.strip():
                raise ValueError(
                    f"Agent '{self.name}' model is empty or whitespace-only."
                )
        
        # Validate dict model
        if isinstance(self.model, dict):
            if not self.model:
                raise ValueError(
                    f"Agent '{self.name}' has empty model dict. "
                    "Provide at least one provider-model mapping."
                )
            # Check for case-equivalent duplicate keys
            # Note: Python dict literals like {"Ollama": "a", "ollama": "b"} deduplicate
            # at parse time (last key wins), but programmatic construction like
            # dict([("Ollama", "a"), ("ollama", "b")]) preserves both keys.
            # This guard catches the programmatic case to avoid ambiguity in
            # case-insensitive matching.
            seen_keys: dict[str, str] = {}
            for key, value in self.model.items():
                # Validate value is a non-empty string
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Agent '{self.name}' has empty or non-string model value for key '{key}'."
                    )
                normalized = key.lower()
                if normalized in seen_keys:
                    raise ValueError(
                        f"Agent '{self.name}' has duplicate model keys (case-insensitive): "
                        f"'{key}' conflicts with existing key '{seen_keys[normalized]}'."
                    )
                seen_keys[normalized] = key

    @property
    def status(self) -> str:
        return "working" if self.current_task else "idle"

    def resolve_model(self) -> str:
        """Resolve the model string based on the current provider.
        
        If model is a dict, return model[provider] (case-insensitive match)
        or model["default"] (also case-insensitive) if provider not found.
        If model is a string, return it directly.
        
        Note: Provider is validated at construction time by __post_init__.
        Construction-time validation also ensures model is non-empty, has valid
        type, and all dict values are non-empty. This method only raises
        ValueError if the provider is not found in the model dict and no
        "default" key exists.
        
        Raises:
            ValueError: If provider not found in model dict and no "default" key exists.
        """
        if isinstance(self.model, dict):
            # Normalize all keys for case-insensitive matching
            normalized_model = {k.lower(): v for k, v in self.model.items()}
            normalized_provider = self.provider.lower()
            
            # Try case-insensitive match first, then case-insensitive "default"
            resolved = normalized_model.get(normalized_provider)
            if resolved is None:
                resolved = normalized_model.get("default")
            
            if resolved is None:
                raise ValueError(
                    f"No model found for provider '{self.provider}' in agent '{self.name}' "
                    f"and no 'default' model specified. Available: {list(self.model.keys())}"
                )
            # Note: Empty/whitespace values are already validated in __post_init__
            return resolved
        
        # String model - already validated in __post_init__
        return self.model


class CompetenceManager:
    def __init__(self, agents_dir: str = "config/agents") -> None:
        self._dir = Path(agents_dir)
        self._profiles: dict[str, AgentProfile] = {}

    def load(self) -> None:
        for yml_file in sorted(self._dir.glob("*.yml")):
            with open(yml_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            profile = AgentProfile(**{k: v for k, v in data.items()})
            self._profiles[profile.name] = profile

    @classmethod
    def from_profiles(cls, profiles: dict[str, AgentProfile], agents_dir: str = "config/agents") -> CompetenceManager:
        """Create a CompetenceManager pre-populated with the given profiles.

        This is the preferred way to construct instances for testing without
        reaching into the internal ``_profiles`` attribute.
        """
        cm = cls(agents_dir=agents_dir)
        cm._profiles = dict(profiles)
        return cm

    @property
    def profiles(self) -> dict[str, AgentProfile]:
        """Read-only view of loaded agent profiles."""
        return dict(self._profiles)

    @property
    def agents(self) -> list[AgentProfile]:
        return list(self._profiles.values())

    def get(self, name: str) -> Optional[AgentProfile]:
        return self._profiles.get(name)

    def for_capability(self, capability: str) -> Optional[AgentProfile]:
        for profile in self._profiles.values():
            if capability in profile.capabilities:
                return profile
        return None