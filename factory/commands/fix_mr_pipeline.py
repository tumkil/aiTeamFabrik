#!/usr/bin/env python3
"""
Command to act on failed pipelines for a merge request.

This command identifies failed pipelines for a given merge request and attempts
to fix them by triggering a retry or other appropriate actions.
"""

import typer
from typing import Optional
from pathlib import Path
from rich.console import Console

from factory.adapters.gitlab_client import GitLabClient
from factory.core.competence import CompetenceManager
from factory.core.execution_engine import CodeExecutionEngine
from factory.core.architect_feedback import create_architect_context
from factory.commands.refine import run_mr_refine
from factory.adapters.llm_router import LlmRouter

console = Console()

app = typer.Typer()


def fix_mr_pipeline(
    mr_iid: int,
    project_id: Optional[int] = None,
    retry: bool = False,
    debug: bool = False,
    config: str = "config/factory.yml",
    agents_dir: str = "config/agents",
    gl_client: Optional[GitLabClient] = None,
) -> bool:
    """
    Act on failed pipelines for a merge request.

    Args:
        mr_iid: The IID of the merge request.
        project_id: The ID of the project. If not provided, the project of the current branch is used.
        retry: If True, retry failed pipelines.
        debug: Enable debug logging.
        config: Path to factory config.
        agents_dir: Path to agents directory.
        gl_client: Optional GitLab client. If not provided, a new one will be created.

    Returns:
        bool: True if the pipeline was successfully fixed, False otherwise.
    """
    typer.echo(f"Acting on failed pipelines for merge request {mr_iid}")
    
    # Initialize GitLab client if not provided
    if gl_client is None:
        gl_client = GitLabClient(config_path=config)
        connected, detail = gl_client.connect()
        if not connected:
            console.print(f"[red]✗[/red] GitLab connection failed: {detail}")
            return False
    
    project = gl_client.project
    if project_id:
        project = gl_client.gitlab.projects.get(project_id)
    
    # Get the merge request
    try:
        mr = project.mergerequests.get(mr_iid)
        console.print(f"[green]✓[/green] Found MR !{mr.iid}: {mr.title}")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to get MR {mr_iid}: {e}")
        return False
    
    # Check pipeline status
    pipelines = mr.pipelines.list()
    if not pipelines:
        console.print(f"[yellow]⚠[/yellow] No pipelines found for MR !{mr.iid}")
        return True
    
    latest_pipeline = pipelines[0]
    status = latest_pipeline.status
    console.print(f"[blue]📊[/blue] Latest pipeline status: {status}")
    
    if status == "failed":
        console.print(f"[red]✗[/red] Pipeline failed for MR !{mr.iid}")
        if retry:
            console.print(f"[yellow]⚠[/yellow] Retrying failed pipeline...")
            try:
                latest_pipeline.retry()
                console.print(f"[green]✓[/green] Pipeline retry triggered for MR !{mr.iid}")
                return True
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to retry pipeline: {e}")
                return False
        else:
            # Trigger repair by running developer refinement
            console.print(f"[yellow]⚠[/yellow] Triggering repair by running developer refinement...")
            try:
                # Load developer agent
                competence = CompetenceManager(agents_dir=agents_dir)
                competence.load()
                developer = competence.get("developer")
                if not developer:
                    console.print(f"[red]✗[/red] Developer agent not found")
                    return False
                
                # Run refinement
                router = LlmRouter()
                gitlab_token = gl_client._token
                
                result = run_mr_refine(
                    merge_request=mr,
                    project=project,
                    gl_url=gl_client._url,
                    developer=developer,
                    router=router,
                    gitlab_token=gitlab_token,
                    progress=console.log,
                )
                
                if result.commit_sha:
                    console.print(f"[green]✓[/green] Repair successful. New commit: {result.commit_sha[:8]}")
                else:
                    console.print(f"[green]✓[/green] Repair successful. No new commits were made.")
                
                return True
            except Exception as e:
                console.print(f"[red]✗[/red] Repair failed: {e}")
                return False
    elif status in ["running", "pending"]:
        console.print(f"[yellow]⚠[/yellow] Pipeline is still {status}, will check again later")
        return True
    elif status == "success":
        console.print(f"[green]✓[/green] Pipeline passed for MR !{mr.iid}")
        return True
    else:
        console.print(f"[blue]ℹ[/blue] Pipeline status: {status}")
        return True


@app.command("fix-mr-pipeline")
def cmd_fix_mr_pipeline(
    mr_iid: int = typer.Option(..., "--mr-iid", "-m", help="Merge Request IID."),
    project_id: Optional[int] = typer.Option(None, "--project-id", "-p", help="Project ID. Defaults to the project of the current branch."),
    retry: bool = typer.Option(False, "--retry", "-r", help="Retry failed pipelines."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
    config: str = typer.Option("config/factory.yml", "--config", "-c", help="Path to factory config."),
    agents_dir: str = typer.Option("config/agents", "--agents-dir", help="Path to agents directory."),
):
    """
    Act on failed pipelines for a merge request.

    Args:
        mr_iid: The IID of the merge request.
        project_id: The ID of the project. If not provided, the project of the current branch is used.
        retry: If True, retry failed pipelines.
        debug: Enable debug logging.
        config: Path to factory config.
        agents_dir: Path to agents directory.
    """
    try:
        success = fix_mr_pipeline(
            mr_iid=mr_iid,
            project_id=project_id,
            retry=retry,
            debug=debug,
            config=config,
            agents_dir=agents_dir,
        )
        if not success:
            raise typer.Exit(1)
    except Exception:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
