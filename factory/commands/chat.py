from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich import box

from factory.adapters.gitlab_client import GitLabClient
from factory.adapters.llm_router import LlmRouter
from factory.core.competence import CompetenceManager
from factory.core.orchestrator import Orchestrator
from factory.core.scrum import ScrumEngine

console = Console()


def cmd_po(
    issue: int = typer.Option(..., "--issue", "-i", help="GitLab issue IID to discuss"),
    config: str = typer.Option("config/factory.yml", "--config", "-c"),
    agents_dir: str = typer.Option("config/agents", "--agents"),
) -> None:
    """Initiate a Product Owner (PO) agent discussion flow for an open issue."""

    console.print()
    console.rule(f"[bold cyan]SoftwareTeamFabrik — PO Discussion for Issue #{issue}[/bold cyan]")
    console.print()

    gl = GitLabClient(config_path=config)
    connected, detail = gl.connect()
    if not connected:
        console.print(f"[red]✗ GitLab connection failed: {detail}[/red]")
        raise typer.Exit(1)

    cm = CompetenceManager(agents_dir=agents_dir)
    cm.load()

    scrum = ScrumEngine(config_path=config)
    router = LlmRouter()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=router)

    # Get the issue object
    project = gl.project
    issue_obj = project.issues.get(issue)
    
    # Validate that the issue is open
    if issue_obj.state != "opened":
        console.print(f"[red]✗ Issue #{issue} is not open (current state: {issue_obj.state})[/red]")
        console.print("[yellow]Only open issues can be discussed with the PO agent.[/yellow]")
        raise typer.Exit(1)

    console.print(f"  [bold]Issue[/bold]      #{issue} {issue_obj.title}")
    console.print()

    # Initiate PO discussion
    console.print("[yellow]PO agent initiating discussion with stakeholders…[/yellow]")
    
    try:
        result = orch.initiate_po_discussion(issue_iid=issue, progress=console.log)
    except Exception as exc:
        console.print(f"[red]✗ Discussion failed: {exc}[/red]")
        raise typer.Exit(1)

    # --- Output ---
    console.print()
    console.print(Panel(result.response, title="PO Discussion Summary", box=box.ROUNDED))
    
    if result.comment_url:
        console.print(f"[blue]Discussion logged at: {result.comment_url}[/blue]")
