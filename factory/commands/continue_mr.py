"""Command to continue an incomplete implementation on an existing MR branch."""
from __future__ import annotations

import os
import re
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from factory.adapters.gitlab_client import GitLabClient
from factory.adapters.llm_router import LlmRouter
from factory.core.competence import CompetenceManager
from factory.core.execution_engine import CodeExecutionEngine
from factory.core.provider_defaults import DEFAULT_MODELS

console = Console()

_CONTINUATION_SYSTEM = """\
You are the Developer agent of SoftwareTeamFabrik running in continuation mode.
A previous agent run hit the iteration limit before finishing. Your job is to
continue the implementation from where it left off.

Workflow:
1. Read the issue notes and MR description to understand what was already done.
2. Explore the repository to see the current state of the code (it already has partial changes).
3. Identify what is still missing or incomplete.
4. Implement the remaining work using write_file for every file you create or change.
5. Run the test suite: run_command('python3 -m pytest tests/ -v') and fix any failures.
6. When everything is done and tests pass, call task_complete with a full summary.

CRITICAL: You MUST call task_complete to finish.

Rules:
- Do NOT re-implement things that are already done — read the code first.
- Write complete file contents for any file you modify.
- Follow the existing code style exactly.
- If you are blocked, call task_complete describing the blocker.
"""


def _extract_issue_iid(text: str) -> int | None:
    """Extract issue IID from MR description via Closes/Refs #N pattern."""
    match = re.search(r"(?:Closes|closes|Refs|refs)\s+#(\d+)", text or "")
    return int(match.group(1)) if match else None


def cmd_continue_mr(
    mr: int = typer.Option(..., "--mr", "-m", help="MR IID to continue"),
    config: str = typer.Option("config/factory.yml", "--config", "-c"),
    agents_dir: str = typer.Option("config/agents", "--agents"),
    provider: str = typer.Option("", "--provider", help="Override provider: anthropic | mistral | ollama | stub"),
) -> None:
    """Continue an incomplete implementation on an existing MR branch."""

    console.print()
    console.rule(f"[bold cyan]SoftwareTeamFabrik — Continuing MR !{mr}[/bold cyan]")
    console.print()

    gl = GitLabClient(config_path=config)
    connected, detail = gl.connect()
    if not connected:
        console.print(f"[red]✗ GitLab connection failed: {detail}[/red]")
        raise typer.Exit(1)

    cm = CompetenceManager(agents_dir=agents_dir)
    cm.load()

    if provider and provider not in DEFAULT_MODELS and provider != "stub":
        console.print(f"[red]✗ Unknown provider: {provider}. Choose from: {', '.join(DEFAULT_MODELS.keys())} or 'stub'[/red]")
        raise typer.Exit(1)

    if provider == "stub":
        for agent in cm.agents:
            agent.provider = "stub"
            agent.execution_mode = "plan"
    elif provider in DEFAULT_MODELS:
        for agent in cm.agents:
            agent.provider = provider
            agent.model = DEFAULT_MODELS[provider]

    project = gl.project
    merge_request = project.mergerequests.get(mr)

    issue_iid = _extract_issue_iid(merge_request.description)
    if issue_iid is None:
        console.print("[red]✗ Could not extract issue IID from MR description. "
                      "Add a 'Closes #N' or 'Refs #N' line to the MR description.[/red]")
        raise typer.Exit(1)

    try:
        issue = project.issues.get(issue_iid)
    except Exception as exc:
        console.print(f"[red]✗ Could not retrieve issue #{issue_iid}: {exc}[/red]")
        raise typer.Exit(1)

    developer = cm.get("developer")
    if not developer:
        console.print("[red]✗ No 'developer' agent found[/red]")
        raise typer.Exit(1)

    console.print(f"  [bold]MR[/bold]        !{mr} — {merge_request.title}")
    console.print(f"  [bold]Branch[/bold]    {merge_request.source_branch}")
    console.print(f"  [bold]Issue[/bold]     #{issue_iid}")
    console.print(f"  [bold]Agent[/bold]     {developer.display_name} ({developer.model})")
    console.print()

    # Build continuation context from the issue and the last factory comment on the issue
    issue_title = issue.title
    issue_desc = issue.description or ""

    previous_progress = ""
    try:
        notes = issue.notes.list()
        factory_notes = [
            n.body for n in notes
            if n.body and "SoftwareTeamFabrik" in n.body and "Partial Implementation" in n.body
        ]
        if factory_notes:
            previous_progress = factory_notes[-1]
    except Exception:
        pass

    continuation_prefix = (
        "# Continuation Run\n\n"
        "The previous agent run hit the iteration limit. "
        "Partial code is already committed to the branch — read it before writing anything.\n\n"
    )
    if previous_progress:
        continuation_prefix += f"## Previous Progress\n\n{previous_progress}\n\n---\n\n"

    enhanced_desc = continuation_prefix + f"## Original Issue\n\n{issue_desc}"

    token = os.environ.get("GITLAB_TOKEN", "")
    engine = CodeExecutionEngine(
        agent=developer,
        repo_path=Path.cwd(),
        issue_title=issue_title,
        issue_description=enhanced_desc,
        gitlab_token=token,
        gitlab_project_id=str(project.id),
        gitlab_url=gl._url,
        branch_name=merge_request.source_branch,
        mr_iid=mr,
        progress=console.log,
    )
    router = LlmRouter()

    console.print("[yellow]Developer continuing implementation…[/yellow]")
    try:
        result = engine.run(router=router, system=_CONTINUATION_SYSTEM)
    except Exception as exc:
        console.print(f"[red]✗ Continuation failed: {exc}[/red]")
        raise typer.Exit(1)

    console.print()
    if not result.needs_continuation:
        console.print("[green]✓[/green] Implementation complete!")
    else:
        console.print("[yellow]⚠[/yellow] Iteration limit hit again — run continue-mr once more.")

    t = Table(box=box.SIMPLE, show_header=False)
    t.add_column(style="bold", min_width=14)
    t.add_column()
    t.add_row("Iterations", str(result.iterations))
    t.add_row("Files changed", str(len(result.files_changed)))
    if result.commit_sha:
        t.add_row("Commit", result.commit_sha[:8])
    t.add_row("MR", merge_request.web_url)
    console.print(t)

    if result.files_changed:
        console.print("  [bold]Changed files:[/bold]")
        for f in result.files_changed:
            console.print(f"    [cyan]{f}[/cyan]")
        console.print()

    # Post a comment on the issue summarising this continuation run
    files_note = "\n".join(f"- `{f}`" for f in result.files_changed) or "_no files changed_"
    commit_note = f"\nCommit: `{result.commit_sha[:8]}`" if result.commit_sha else ""
    if not result.needs_continuation:
        body = (
            f"### :white_check_mark: {developer.display_name} — Implementation Complete (continuation)\n\n"
            f"**Summary:** {result.response}\n\n"
            f"**Files changed ({len(result.files_changed)}):**\n{files_note}"
            f"{commit_note}\n\n"
            f"---\n*Model: `{developer.model}` · Iterations: {result.iterations}*"
        )
    else:
        body = (
            f"### :hourglass: {developer.display_name} — Partial Implementation (iteration limit reached again)\n\n"
            f"**Progress:** {result.response}\n\n"
            f"**Files changed ({len(result.files_changed)}):**\n{files_note}"
            f"{commit_note}\n\n"
            f"Run `factory continue-mr --mr {mr}` to continue.\n\n"
            f"---\n*Model: `{developer.model}` · Iterations: {result.iterations}*"
        )
    issue.notes.create({"body": body})

    console.print(Panel(result.response or "(no response)", title="Summary", border_style="cyan", expand=False))
    console.print()
