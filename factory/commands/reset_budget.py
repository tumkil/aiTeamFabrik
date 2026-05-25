"""CLI command to reset token budget usage for an agent."""

from __future__ import annotations

import logging
import typer
from pathlib import Path
from typing import Optional

from factory.core.token_budget import TokenBudgetManager

logger = logging.getLogger(__name__)


def cmd_reset_budget(
    config: str = typer.Option("config/factory.yml", "--config", "-c", help="Path to factory.yml"),
    usage: Optional[str] = typer.Option(None, "--usage", help="Path to token_usage.yml (default: config dir)"),
    agent: str = typer.Argument(..., help="Agent name to reset budget for"),
    scope: str = typer.Option("daily", "--scope", "-s", help="Scope to reset: 'daily' or 'sprint'"),
) -> None:
    """Reset token budget usage for a specific agent.
    
    This command clears the token usage tracking for the specified agent in the
    given scope (daily or sprint). Useful for testing or manual adjustments.
    """
    try:
        usage_path = Path(usage) if usage is not None else Path(config).parent / "token_usage.yml"
        budget_manager = TokenBudgetManager(
            config_path=Path(config),
            usage_path=usage_path
        )
        
        # Validate scope
        if scope not in ("daily", "sprint"):
            typer.echo("Error: scope must be either 'daily' or 'sprint'", err=True)
            raise typer.Exit(1)
        
        # Perform reset
        budget_manager.reset(scope=scope, key=agent)
        
        typer.echo(f"✓ Reset {scope} token budget for agent '{agent}'")
        
    except Exception as exc:
        logger.exception("Reset budget failed: %s", exc)
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
