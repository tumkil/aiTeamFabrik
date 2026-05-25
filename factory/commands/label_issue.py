from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich import box

from factory.adapters.gitlab_client import GitLabClient
from factory.core.scrum import ScrumEngine

console = Console()

app = typer.Typer()


@app.command()
def cmd_label_issue(
    issue: int = typer.Option(..., "--issue", "-i", help="GitLab issue IID to label"),
    ready: bool = typer.Option(False, "--ready", "-r", help="Label issue as ready for development"),
    next_sprint: bool = typer.Option(False, "--next-sprint", "-n", help="Label issue for next sprint"),
    blocked: bool = typer.Option(False, "--blocked", "-b", help="Label issue as blocked"),
    config: str = typer.Option("config/factory.yml", "--config", "-c"),
) -> None:
    """Label an issue as ready, blocked, and/or assign it to the next sprint.
    
    This command allows Product Owners to:
    - Mark issues as 'ready' (indicating they're fully specified)
    - Mark issues as 'blocked' (indicating they cannot proceed)
    - Assign issues to the next sprint by applying a 'sprint-next' label
    """

    console.print()
    console.rule(f"[bold cyan]SoftwareTeamFabrik — Label Issue #{issue}[/bold cyan]")
    console.print()

    gl = GitLabClient(config_path=config)
    connected, detail = gl.connect()
    if not connected:
        console.print(f"[red]✗ GitLab connection failed: {detail}[/red]")
        raise typer.Exit(1)

    scrum = ScrumEngine(config_path=config)
    
    # Get the issue object
    project = gl.project
    issue_obj = project.issues.get(issue)
    
    # Validate that the issue exists
    if issue_obj is None:
        console.print(f"[red]✗ Issue #{issue} does not exist[/red]")
        raise typer.Exit(1)
    
    # Validate that the issue is open
    if issue_obj.state != "opened":
        console.print(f"[red]✗ Issue #{issue} is not open (current state: {issue_obj.state})[/red]")
        console.print("[yellow]Only open issues can be labeled.[/yellow]")
        raise typer.Exit(1)

    console.print(f"  [bold]Issue[/bold]      #{issue} {issue_obj.title}")
    console.print()

    # Validate that ready and blocked are not both specified
    if ready and blocked:
        console.print("[red]✗ Cannot specify both --ready and --blocked[/red]")
        console.print("[yellow]An issue cannot be both ready and blocked.[/yellow]")
        raise typer.Exit(1)

    # Apply labels based on flags
    labels_to_add = []
    labels_to_remove = []
    
    if ready:
        labels_to_add.append("ready")
        labels_to_remove.append("blocked")
        console.print("[green]✓ Adding 'ready' label to issue[/green]")
    
    if blocked:
        labels_to_add.append("blocked")
        labels_to_remove.append("ready")
        console.print("[green]✓ Adding 'blocked' label to issue[/green]")
    
    if next_sprint:
        labels_to_add.append("sprint-next")
        console.print("[green]✓ Adding 'sprint-next' label to issue[/green]")
    
    if not labels_to_add:
        console.print("[yellow]No labeling action specified. Use --ready, --blocked, and/or --next-sprint.[/yellow]")
        raise typer.Exit(1)
    
    # Update the issue with new labels
    try:
        current_labels = set(issue_obj.labels)
        new_labels = list(current_labels.union(labels_to_add) - set(labels_to_remove))
        
        issue_obj.labels = new_labels
        issue_obj.save()
        
        console.print()
        console.print("[green]✓ Issue labels updated successfully![/green]")
        console.print(f"  Current labels: {', '.join(new_labels)}")
        
    except Exception as exc:
        console.print(f"[red]✗ Failed to update labels: {exc}[/red]")
        raise typer.Exit(1)



if __name__ == "__main__":
    app()
