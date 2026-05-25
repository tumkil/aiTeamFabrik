from __future__ import annotations

import os
import subprocess
import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from factory.adapters.gitlab_client import GitLabClient
from factory.adapters.llm_router import LlmRouter
from factory.core.competence import CompetenceManager
from factory.core.orchestrator import Orchestrator
from factory.core.provider_defaults import DEFAULT_MODELS
from factory.core.scrum import ScrumEngine

console = Console()


def cmd_run(
    issue: int = typer.Option(..., "--issue", "-i", help="GitLab issue IID to process"),
    config: str = typer.Option("config/factory.yml", "--config", "-c"),
    agents_dir: str = typer.Option("config/agents", "--agents"),
    provider: str = typer.Option("", "--provider", help="Override provider: anthropic | mistral | ollama | stub"),
) -> None:
    """Spawn an AI agent for a GitLab issue. Execution-mode agents write and commit real code."""

    console.print()
    console.rule(f"[bold cyan]SoftwareTeamFabrik — Running Issue #{issue}[/bold cyan]")
    console.print()

    gl = GitLabClient(config_path=config)
    connected, detail = gl.connect()
    if not connected:
        console.print(f"[red]✗ GitLab connection failed: {detail}[/red]")
        raise typer.Exit(1)

    cm = CompetenceManager(agents_dir=agents_dir)
    cm.load()

    # Provider override: stub disables execution; anthropic/mistral/ollama swap the backend+model
    if provider and provider not in DEFAULT_MODELS and provider != "stub":
        console.print(f"[red]✗ Unknown provider: {provider}. Choose from: {', '.join(DEFAULT_MODELS.keys())} or 'stub'[/red]")
        raise typer.Exit(1)

    if provider == "stub":
        for agent in cm.agents:
            agent.provider = "stub"
            agent.execution_mode = "plan"  # stub can't clone/push
    elif provider in DEFAULT_MODELS:
        for agent in cm.agents:
            agent.provider = provider
            agent.model = DEFAULT_MODELS[provider]

    scrum = ScrumEngine(config_path=config)
    router = LlmRouter()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=router)

    # Peek at which agent will be assigned before running
    project = gl.project
    issue_obj = project.issues.get(issue)
    agent = orch._resolve_agent(issue_obj)

    mode_label = (
        "[yellow]execute[/yellow] — will write & commit code"
        if agent.execution_mode == "execute"
        else "[blue]plan[/blue] — will post analysis"
    )
    console.print(f"  [bold]Agent[/bold]      {agent.display_name} ({agent.model})")
    console.print(f"  [bold]Mode[/bold]       {mode_label}")
    console.print(f"  [bold]Issue[/bold]      #{issue} {issue_obj.title}")
    console.print()

    status_msg = (
        f"[yellow]{agent.display_name} executing — writing code, running tests…[/yellow]"
        if agent.execution_mode == "execute"
        else f"[yellow]{agent.display_name} analysing issue…[/yellow]"
    )

    console.print(status_msg)
    try:
        # Configure Git for the working directory
        _configure_git()
        
        result = orch.run(issue_iid=issue, progress=console.log)
    except Exception as exc:
        console.print(f"[red]✗ Agent failed: {exc}[/red]")
        raise typer.Exit(1)

    # --- Output ---
    if result.mode == "execute":
        # Surface execution errors clearly
        if not result.response and not result.commit_sha and result.iterations == 0:
            console.print(
                "[red]✗ Execution engine failed to start.[/red]\n"
                "  Check that [bold]ANTHROPIC_API_KEY[/bold] is set correctly in your .env file.\n"
                "  The GitLab issue comment contains the full error.\n"
            )
            if result.comment_url:
                console.print(f"  Error log: {result.comment_url}")
            raise typer.Exit(1)

        t = Table(box=box.SIMPLE, show_header=False)
        t.add_column(style="bold", min_width=14)
        t.add_column()
        t.add_row("Iterations", str(result.iterations))
        t.add_row("Files changed", str(len(result.files_changed)))
        if result.files_changed:
            for file in result.files_changed:
                t.add_row("", f"  - {file}")
        
        if result.commit_sha:
            t.add_row("Commit", result.commit_sha[:8])
        if result.mr_url:
            t.add_row("MR", result.mr_url)
        
        console.print(t)
        
        # Check for continuation needed
        if hasattr(result, 'needs_continuation') and result.needs_continuation:
            console.print()
            console.print("[yellow]⚠️  Implementation incomplete[/yellow]")
            console.print("The agent hit the iteration limit before finishing.")
            if result.mr_url:
                mr_iid = result.mr_url.split('/')[-1]
                console.print(f"  factory continue-mr --mr {mr_iid}")
        else:
            console.print()
            console.print("[green]✓ Implementation completed[/green]")
    else:
        # Plan mode output
        console.print()
        console.print(Panel(result.response, title="Analysis", box=box.ROUNDED))


def _git_config_is_set(key: str) -> bool:
    """Check whether a git config key is set at any scope (local, global, system).

    Uses ``git config --get`` which searches all scopes and returns exit code 0
    when the key has a value, 1 when it is unset.
    """
    result = subprocess.run(
        ["git", "config", "--get", key],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _configure_git() -> None:
    """Configure Git in the working directory similar to the monitor command.
    
    This function ensures that Git is properly configured with the necessary
    user name and email settings to perform commits and other Git operations.

    It **only** sets values that are not already configured at any scope
    (local, global, or system), so it will never overwrite a developer's
    existing git identity.  All writes use ``--local`` to guarantee they
    are scoped to the current repository and never leak to global config.
    """
    # Set Git user name and email only if not already configured at any scope
    if not _git_config_is_set("user.name"):
        subprocess.run(
            ["git", "config", "--local", "user.name", "SoftwareTeamFabrik"],
            check=True,
        )
    if not _git_config_is_set("user.email"):
        subprocess.run(
            ["git", "config", "--local", "user.email", "factory@example.com"],
            check=True,
        )