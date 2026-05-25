"""Shared default model mappings for LLM providers."""

# Default models for each provider when --provider override is used
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "mistral": "mistral-large-latest",
    "ollama": "devstral-2:123b-cloud",
}