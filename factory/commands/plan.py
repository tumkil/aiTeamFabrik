"""CLI command for creating and managing sprint plans."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from factory.adapters.gitlab_client import GitLabClient
from factory.core.planner import TokenAwarePlanner, PlanningAlgorithm, SprintPlan

logger = logging.getLogger(__name__)
console = Console()

# Valid algorithm names for user-facing validation
_VALID_ALGORITHMS = [a.value for a in PlanningAlgorithm]


def cmd_plan(
    sprint: int = typer.Option(..., "--sprint", "-s", help="Sprint number to plan for"),
    config: str = typer.Option("config/factory.yml", "--config", "-c", help="Path to factory.yml"),
    algorithm: str = typer.Option(
        "greedy",
        "--algorithm", "-a",
        help=f"Planning algorithm: {', '.join(_VALID_ALGORITHMS)}",
    ),
    budget: Optional[int] = typer.Option(None, "--budget", "-b",
                                         help="Token budget for the sprint (default: 1,000,000)"),
    show: bool = typer.Option(False, "--show", help="Show existing plan without creating new one"),
) -> None:
    """Create or display a sprint plan with token-aware scheduling."""

    console.print()
    console.rule(f"[bold cyan]SoftwareTeamFabrik — Sprint {sprint} Planning[/bold cyan]")
    console.print()

    # Validate algorithm early so the user gets a clear error message listing
    # valid choices, rather than a cryptic ValueError from the Enum constructor.
    if algorithm not in _VALID_ALGORITHMS:
        valid_str = ", ".join(_VALID_ALGORITHMS)
        raise typer.BadParameter(
            f"Invalid algorithm '{algorithm}'. Valid: {valid_str}",
            param_hint="'--algorithm'",
        )
    
    planning_algorithm = PlanningAlgorithm(algorithm)

    # Validate budget before connecting to GitLab
    if budget is not None and budget <= 0:
        raise typer.BadParameter(
            "Budget must be a positive integer.",
            param_hint="'--budget'",
        )

    # Connect to GitLab
    gl = GitLabClient(config_path=config)
    connected, detail = gl.connect()

    if not connected:
        console.print(f"[red]✗ GitLab connection failed: {detail}[/red]")
        raise typer.Exit(1)

    # Check if project is accessible
    if not gl.project:
        console.print("[red]✗ GitLab project not found or accessible[/red]")
        raise typer.Exit(1)

    console.print(f"  [bold]Project[/bold]     {gl.project_path}")
    console.print()

    # Determine sprint identifier
    sprint_number = f"sprint-{sprint}"
    plan_path = Path(config).parent / f"{sprint_number}-plan.yml"

    # Initialize planner
    planner = TokenAwarePlanner(plan_path=plan_path)

    # Show existing plan if requested
    if show:
        existing_plan = planner.get_plan()
        if existing_plan:
            _display_plan(existing_plan, planner)
        else:
            console.print("[yellow]No existing plan found.[/yellow]")
        return

    # Check if plan already exists
    existing_plan = planner.get_plan()
    if existing_plan:
        console.print(f"[yellow]⚠ Existing plan found for {sprint_number}[/yellow]")
        console.print(f"  Algorithm: {existing_plan.algorithm.value}")
        console.print(f"  Budget: {existing_plan.budget:,} tokens")
        console.print()

        overwrite = typer.confirm("Overwrite existing plan?")
        if not overwrite:
            console.print("[blue]Keeping existing plan.[/blue]")
            _display_plan(existing_plan, planner)
            return

    # Get issues from GitLab
    console.print("[bold]Fetching issues from GitLab...[/bold]")

    try:
        # Fetch open issues for the sprint label using a page-size-limited
        # iterator to avoid loading the entire issue list into memory at once
        # and to respect API rate limits.
        # Note: GitLab's API label filter is case-sensitive. Projects should use
        # lowercase sprint labels (e.g., 'sprint-3') for consistent filtering.
        sprint_label_lower = f"sprint-{sprint}"
        issues = []
        page = 1
        per_page = 100
        while True:
            batch = gl.project.issues.list(
                state='opened',
                labels=[sprint_label_lower],   # Server-side filter
                per_page=per_page,
                page=page,
            )
            if not batch:
                break
            for issue in batch:
                # Determine agent based on labels
                agent = "developer"  # Default agent
                if any(label.lower() == "architecture" for label in issue.labels):
                    agent = "architect"
                elif any(label.lower() == "reviewer" for label in issue.labels):
                    agent = "reviewer"
                
                issues.append({
                    'issue_iid': issue.iid,
                    'title': issue.title,
                    'labels': issue.labels,
                    'description': issue.description or '',
                    'agent': agent
                })
            if len(batch) < per_page:
                # Last page reached
                break
            page += 1

        console.print(f"  Found {len(issues)} issues for sprint {sprint}")
        console.print()

        if not issues:
            console.print(
                f"[yellow]⚠ No issues found for sprint {sprint}.[/yellow]\n"
                f"  Hint: make sure issues have the label [bold]{sprint_label_lower}[/bold] "
                f"(case-insensitive match)."
            )
            return

        # Determine budget — use the value supplied via --budget, or fall back
        # to the hard-coded default of 1,000,000 tokens.
        effective_budget = budget if budget is not None else 1_000_000
        
        if budget is None:
            console.print(
                f"[yellow]⚠ No budget specified — using default: {effective_budget:,} tokens[/yellow]"
            )

        console.print(f"  [bold]Budget[/bold]      {effective_budget:,} tokens")
        console.print(f"  [bold]Algorithm[/bold]   {algorithm}")
        console.print()

        # Create plan
        with console.status("[yellow]Creating optimal plan...[/yellow]"):
            plan = planner.create_plan(
                sprint_number=sprint_number,
                budget=effective_budget,
                issues=issues,
                algorithm=planning_algorithm,
            )

        console.print("[green]✓ Plan created successfully![/green]")
        console.print()

        # Display the plan
        _display_plan(plan, planner)

    except typer.Exit:
        # Re-raise typer.Exit to preserve exit codes
        raise
    except Exception as exc:
        console.print(f"[red]✗ Failed to create plan: {exc}[/red]")
        logger.error("Plan creation failed", exc_info=True)
        raise typer.Exit(1)


def _display_plan(plan: SprintPlan, planner: TokenAwarePlanner) -> None:
    """Display a sprint plan in a user-friendly format."""
    summary = planner.plan_summary()

    # Summary table
    summary_table = Table(box=box.SIMPLE, show_header=False)
    summary_table.add_column(style="bold", min_width=14)
    summary_table.add_column()
    summary_table.add_row("Sprint", plan.sprint_number)
    summary_table.add_row("Algorithm", plan.algorithm.value)
    summary_table.add_row("Budget", f"{plan.budget:,} tokens")
    summary_table.add_row("Planned tokens", f"{summary['total_planned_tokens']:,} tokens")
    summary_table.add_row("Pending", str(summary['pending_issues']))
    summary_table.add_row("In Progress", str(summary['in_progress_issues']))
    summary_table.add_row("Completed", str(summary['completed_issues']))

    console.print("  [bold]Plan Summary[/bold]")
    console.print(summary_table)
    console.print()

    # Plan entries table
    if plan.entries:
        entries_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
        entries_table.add_column("Issue", style="cyan")
        entries_table.add_column("Title", min_width=40)
        entries_table.add_column("Tokens", style="dim")
        entries_table.add_column("Priority", style="dim")
        entries_table.add_column("Agent", style="dim")
        entries_table.add_column("Status", style="dim")

        for entry in plan.entries:
            status_style = (
                "green" if entry.status == "completed"
                else ("yellow" if entry.status == "in_progress" else "dim")
            )

            entries_table.add_row(
                str(entry.issue_iid),
                entry.title[:60] + "..." if len(entry.title) > 60 else entry.title,
                f"{entry.estimated_tokens:,}",
                str(entry.priority),
                entry.agent,
                f"[{status_style}]{entry.status}[/{status_style}]"
            )

        console.print("  [bold]Planned Issues[/bold]")
        console.print(entries_table)
    else:
        console.print("  [bold]No issues planned.[/bold]")

    console.print()
    console.print(f"  [dim]Plan saved to: {planner.plan_path}[/dim]")
    console.print()
