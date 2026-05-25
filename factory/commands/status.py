from __future__ import annotations

import logging
from typing import Optional
import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text

from factory.adapters.gitlab_client import GitLabClient
from factory.core.competence import CompetenceManager
from factory.core.token_budget import TokenBudgetManager

logger = logging.getLogger(__name__)

console = Console()


def cmd_status(
    config: str = typer.Option("config/factory.yml", "--config", "-c", help="Path to factory.yml"),
    agents_dir: str = typer.Option("config/agents", "--agents", help="Path to agents config dir"),
    usage: Optional[str] = typer.Option(None, "--usage", help="Path to token_usage.yml (default: config dir)"),
) -> None:
    """Show the current state of the factory: GitLab connection, sprint, and agent roster."""

    console.print()
    console.rule("[bold cyan]SoftwareTeamFabrik — Status Report[/bold cyan]")
    console.print()

    # --- GitLab connection ---
    gl = GitLabClient(config_path=config)
    connected, detail = gl.connect()

    status_icon = "[green]✓ connected[/green]" if connected else f"[red]✗ {detail}[/red]"
    console.print(f"  [bold]GitLab[/bold]      {gl.hostname}  {status_icon}")

    if not connected:
        raise typer.Exit(1)

    console.print(f"  [bold]Project[/bold]     {gl.project_path}")

    # --- Sprint info ---
    try:
        sprint = gl.sprint_info()
        days_left = gl.days_remaining(sprint)
        from datetime import date
        if date.today() < sprint.start_date:
            days_until_start = (sprint.start_date - date.today()).days
            remaining_str = f"[yellow]starts in {days_until_start} days[/yellow]"
        else:
            remaining_str = f"[yellow]{days_left} days remaining[/yellow]"
        console.print(
            f"  [bold]Sprint[/bold]      #{sprint.number} — {sprint.name} "
            f"({remaining_str})"
        )
    except Exception as exc:
        console.print(f"  [bold]Sprint[/bold]      [red]Could not load sprint info: {exc}[/red]")
        sprint = None

    console.print()

    # --- Token Budget Usage ---
    try:
        usage_path = Path(usage) if usage is not None else Path(config).parent / "token_usage.yml"
        budget_manager = TokenBudgetManager(
            config_path=Path(config),
            usage_path=usage_path
        )
        report = budget_manager.usage_report()
        
        if report["agents"]:
            budget_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
            budget_table.add_column("Agent", style="cyan", min_width=12)
            budget_table.add_column("Daily Used / Limit", style="dim", min_width=20)
            budget_table.add_column("Sprint Used / Limit", style="dim", min_width=20)
            budget_table.add_column("Status")
            
            for agent_name, agent_data in report["agents"].items():
                daily_str = f"{agent_data['daily_used']:,} / {agent_data['daily_limit']:,}"
                sprint_str = f"{agent_data['sprint_used']:,} / {agent_data['sprint_limit']:,}"
                
                # Determine status — check both daily and sprint scopes
                over_budget = (
                    budget_manager.is_over_budget(agent_name, scope="daily")
                    or budget_manager.is_over_budget(agent_name, scope="sprint")
                )
                if over_budget:
                    status_text = "❌ OVER BUDGET"
                    status_style = "red"
                elif agent_data['daily_pct'] >= 90 or agent_data['sprint_pct'] >= 90:
                    status_text = "⚠ NEAR LIMIT"
                    status_style = "yellow"
                else:
                    status_text = "✓ OK"
                    status_style = "green"
                
                budget_table.add_row(
                    agent_name,
                    daily_str,
                    sprint_str,
                    Text(status_text, style=status_style)
                )
            
            console.print("  [bold]Token Budget Usage[/bold]")
            console.print(budget_table)
            console.print()
    except Exception as exc:
        logger.warning("Budget manager initialization failed: %s", exc, exc_info=True)
        console.print("  [bold]Token Budget[/bold]  [dim]Budget tracking unavailable[/dim]\n")

    # --- Agent roster ---
    cm = CompetenceManager(agents_dir=agents_dir)
    cm.load()

    agent_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    agent_table.add_column("Agent", style="cyan", min_width=12)
    agent_table.add_column("Model", style="dim", min_width=20)
    agent_table.add_column("Status")
    agent_table.add_column("Task")

    for agent in cm.agents:
        if agent.status == "working":
            status_cell = Text("working", style="yellow")
            task_cell = Text(f"→  {agent.current_task}", style="yellow")
        else:
            status_cell = Text("idle", style="green")
            task_cell = Text("—", style="dim")

        agent_table.add_row(agent.display_name, agent.model, status_cell, task_cell)

    console.print("  [bold]Active Agents[/bold]")
    console.print(agent_table)

    # --- Issue summary ---
    if sprint:
        bar_filled = int(sprint.completion_pct / 10)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)
        console.print(
            f"  [bold]Open Issues[/bold]   "
            f"{sprint.open_issues} open / "
            f"{sprint.in_progress_issues} in progress / "
            f"{sprint.closed_issues} closed"
        )
        console.print(
            f"  [bold]Milestone[/bold]     Sprint {sprint.number}  "
            f"[cyan]{bar}[/cyan]  {sprint.completion_pct}%"
        )

    console.print()
