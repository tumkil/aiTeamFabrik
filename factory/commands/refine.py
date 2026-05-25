"""Command for agent-driven MR refinement."""
from __future__ import annotations

import os
import re
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from datetime import datetime

from factory.adapters.gitlab_client import GitLabClient
from factory.adapters.llm_router import LlmRouter, LlmResponse
from factory.core.competence import CompetenceManager
from factory.core.execution_engine import CodeExecutionEngine, ExecutionResult
from factory.core.provider_defaults import DEFAULT_MODELS

console = Console()


REFINEMENT_SYSTEM = """\
You are the Developer agent of SoftwareTeamFabrik running in MR refinement mode.
You are a coding model: your strengths are reading code and writing precise edits.
Your job is to address the review feedback on the merge request branch.

You are working inside a git checkout. You have these tools:
  read_file, write_file, list_directory, search_code, run_command, task_complete.

## Inputs
- The MR title and the current diff (this is the code as it stands — it still has
  the issues described below).
- Review feedback from the Code Reviewer. This may be free-form text or may
  contain structured findings. It may also include a meta-reviewer section that
  classifies each point as:
    CONFIRMED        -> a real problem; you must fix it.
    FALSE POSITIVE   -> reviewer was wrong; skip it.
    OUT OF SCOPE     -> not for this MR; skip it.
  If no meta classification is present, treat all feedback as CONFIRMED.

## Required workflow
Follow these steps in order. Do not skip steps.

1. Re-state the work. Before any tool call, list every actionable issue you
   identified from the review feedback as a short numbered plan:
   "I will fix: 1) <problem> in <file>, 2> ..." Skip anything marked
   FALSE POSITIVE or OUT OF SCOPE.

2. For each actionable issue, do exactly this:
     a. read_file on the file the feedback mentions.
     b. Decide the smallest edit that resolves the issue.
     c. write_file with the FULL new contents of that file.
   You MUST call write_file for every issue you claimed to fix. An issue is not
   "fixed" until write_file has succeeded for it. Do not claim a fix you did
   not write.

3. When several issues touch the same file, read it once and write it once
   with all related edits combined. Do not write the same file repeatedly.

4. After all edits, run the test suite exactly once:
     run_command('python -m pytest tests/ --tb=short -q')
   - If it passes: continue to step 5.
   - If it fails: read the failure output, fix the cause, re-run ONCE.
   - If the second run still fails: stop editing and report the failure in
     task_complete. Do not loop further.

5. Call task_complete exactly once with the structured summary defined below.

## Hard rules
- Always read_file before write_file. Never write blind.
- write_file always receives the full new file contents — never a diff or patch.
- Keep edits minimal. Do not reformat unrelated lines, do not rename things,
  do not add features the feedback did not ask for.
- Match the surrounding code's style: imports, naming, type hints, docstring
  shape. Look at the file you are editing for the pattern.
- No TODOs, no placeholders, no "implement later", no commented-out code.
- run_command does NOT support shell pipes (`|`), redirects (`>`, `2>&1`), or
  background operators (`&`). Use the tool's own flags instead
  (`--tb=short`, `-q`, `-k 'pattern'`).
- Do not delete files unless the feedback explicitly requires it.
- Do not retry the same failing tool call more than twice. If it keeps
  failing, report it via task_complete.
- Stay focused on the review feedback. Do not refactor unrelated code.

## Handling tricky feedback
- If the reviewer's suggested fix is wrong but the underlying problem is real,
  implement a correct fix and note in the summary that you deviated.
- If two feedback items conflict, pick the one that fits the existing code better
  and explain the choice in the summary.
- If an issue requires information you cannot get (missing dependency,
  ambiguous spec), mark it BLOCKED in the summary instead of guessing.

## Final report (task_complete)
Pass a single string to task_complete. It MUST contain one block per actionable
issue you addressed (or skipped):

  Issue #N: <FIXED | ALREADY CORRECT | SKIPPED (FALSE POSITIVE) | OUT OF SCOPE | BLOCKED>
    File(s): <paths you wrote, or "none">
    Change: <one or two sentences describing what you actually changed>
    Note:  <only if you deviated from the reviewer's suggestion or it is BLOCKED>

Then end with:

  Tests: <pass | fail — N passed, M failed>
  Notes: <anything the next reviewer should look at; "none" is fine>

Be concise. The next agent reads this summary verbatim to decide whether to merge.
"""


def _extract_issue_iids(text: str) -> list[int]:
    """Extract issue IIDs from text (e.g., #1, #2, etc.)."""
    matches = re.findall(r'(?:^|[\s,])#(\d+)', text)
    return [int(m) for m in matches]


def _get_mr_issue_iid(mr) -> int:
    """Extract the first issue IID from an MR's title and description."""
    title = mr.title or ""
    description = mr.description or ""
    combined = f"{title} {description}"
    iids = _extract_issue_iids(combined)
    return iids[0] if iids else None


def _parse_ts(ts: str) -> datetime | None:
    """Parse a GitLab ISO 8601 timestamp to a timezone-aware datetime.

    Handles both 'Z' (not supported by fromisoformat in Python < 3.11)
    and '+HH:MM' offset forms so commit authored_date and note created_at
    are compared in the same timezone domain.
    """
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Approval-note detection
# ---------------------------------------------------------------------------

# Negation phrases that, when they appear before APPROVE/APPROVED, mean the
# note is *not* an approval.
_NEGATION_PATTERNS = [
    re.compile(r'\bNOT\s+APPROVE\b', re.IGNORECASE),
    re.compile(r'\bNOT\s+APPROVED\b', re.IGNORECASE),
    re.compile(r"\bDON'?T\s+APPROVE\b", re.IGNORECASE),
    re.compile(r"\bDON'?T\s+APPROVED\b", re.IGNORECASE),
    re.compile(r"\bDO\s+NOT\s+APPROVE\b", re.IGNORECASE),
    re.compile(r"\bDO\s+NOT\s+APPROVED\b", re.IGNORECASE),
    re.compile(r'\bNEVER\s+APPROVE\b', re.IGNORECASE),
    re.compile(r'\bNEVER\s+APPROVED\b', re.IGNORECASE),
    re.compile(r'\bCANNOT\s+APPROVE\b', re.IGNORECASE),
    re.compile(r'\bCANNOT\s+APPROVED\b', re.IGNORECASE),
    re.compile(r"\bCAN'?T\s+APPROVE\b", re.IGNORECASE),
    re.compile(r"\bCAN'?T\s+APPROVED\b", re.IGNORECASE),
    re.compile(r'\bREFUSE\s+TO\s+APPROVE\b', re.IGNORECASE),
    re.compile(r"\bWON'?T\s+APPROVE\b", re.IGNORECASE),
    re.compile(r"\bWON'?T\s+APPROVED\b", re.IGNORECASE),
    re.compile(r'\bUNABLE\s+TO\s+APPROVE\b', re.IGNORECASE),
    re.compile(r'\bUNABLE\s+TO\s+APPROVED\b', re.IGNORECASE),
]

# Positive patterns that indicate a genuine approval.
_POSITIVE_PATTERNS = [
    re.compile(r'✅\s*APPROVE', re.IGNORECASE),   # ✅ APPROVE (code-reviewer verdict)
    re.compile(r'##\s*Verdict\s*\n\s*✅', re.IGNORECASE),  # Verdict heading with checkmark
    re.compile(r'\bAPPROVE\b', re.IGNORECASE),      # standalone word APPROVE
    re.compile(r'\bAPPROVED\b', re.IGNORECASE),     # standalone word APPROVED
]


def _is_approval_note(body: str) -> bool:
    """Return True iff *body* is a genuine approval note.

    A note counts as an approval only when it contains a positive approval
    pattern (``APPROVE``, ``APPROVED``, ``✅ APPROVE``, etc.) **and** does not
    contain any negation pattern (``NOT APPROVE``, ``DON'T APPROVE``, etc.).

    This prevents false positives like *"I do NOT APPROVE"* from triggering
    an auto-merge.
    """
    if not body:
        return False

    # If any negation pattern matches, this is explicitly *not* an approval.
    for pat in _NEGATION_PATTERNS:
        if pat.search(body):
            return False

    # Otherwise, look for a positive approval signal.
    for pat in _POSITIVE_PATTERNS:
        if pat.search(body):
            return True

    return False


def _collect_review_notes(merge_request, latest_push_dt: datetime | None = None) -> list[str]:
    """Collect review notes from a merge request, filtering by timestamp if provided.
    
    Args:
        merge_request: The GitLab merge request object
        latest_push_dt: If provided, only include notes created after this timestamp
    
    Returns:
        List of formatted review note strings
    """
    notes = merge_request.notes.list(all=True)
    review_notes = []
    
    for note in notes:
        body = note.body or ""
        note_dt = _parse_ts(getattr(note, "created_at", ""))
        
        # Skip notes created before/at the latest push if timestamp filtering is enabled
        if latest_push_dt and note_dt and note_dt <= latest_push_dt:
            continue
        
        author_name = (note.author or {}).get("name", "")
        is_factory = "SoftwareTeamFabrik" in author_name or "Code Review by SoftwareTeamFabrik" in body
        
        if "Code Review by SoftwareTeamFabrik" in body:
            content = body.split("### :mag:")[1] if "### :mag:" in body else body
            review_notes.append(f"Code Reviewer:\n{content}")
        elif not is_factory and (
            "REQUEST CHANGES" in body or "BLOCK" in body
            or any(w in body.lower() for w in ["fix", "change", "update", "refactor", "bug"])
        ):
            review_notes.append(f"Human Reviewer ({author_name or 'Unknown'}):\n{body}")
    
    # Fallback: if nothing was posted after the last push, take the latest review overall
    if not review_notes:
        for note in reversed(notes):
            body = note.body or ""
            if "Code Review by SoftwareTeamFabrik" in body:
                content = body.split("### :mag:")[1] if "### :mag:" in body else body
                review_notes.append(f"Code Reviewer:\n{content}")
                break
    
    return review_notes


def _get_mr_diff(merge_request) -> str:
    """Extract the diff text from a merge request.
    
    Args:
        merge_request: The GitLab merge request object
    
    Returns:
        Formatted diff text, truncated if too long
    """
    diff_text = ""
    try:
        changes_data = merge_request.changes()
        changes = changes_data.changes if hasattr(changes_data, "changes") else changes_data.get("changes", [])
        for change in changes:
            diff_text += (
                f"\n--- {change.get('old_path', 'file')}"
                f"\n+++ {change.get('new_path', 'file')}"
                f"\n{change.get('diff', '')}"
            )
        if len(diff_text) > 8000:
            diff_text = diff_text[:8000] + "\n\n[diff truncated]"
    except Exception:
        pass
    
    return diff_text


def cmd_refine(
    mr: int = typer.Option(..., "--mr", "-m", help="Merge Request IID to refine"),
    config: str = typer.Option("config/factory.yml", "--config", "-c"),
    agents_dir: str = typer.Option("config/agents", "--agents"),
    provider: str = typer.Option("", "--provider", help="Override provider: anthropic | mistral | ollama | stub"),
    auto_merge: bool = typer.Option(False, "--auto-merge", help="Auto-merge if all reviews are approved"),
) -> None:
    """Run Developer agent to address review feedback on a merge request and push changes to the MR branch."""

    console.print()
    console.rule(f"[bold cyan]SoftwareTeamFabrik — Refining MR !{mr}[/bold cyan]")
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
            agent.execution_mode = "plan"
    elif provider in DEFAULT_MODELS:
        for agent in cm.agents:
            agent.provider = provider
            agent.model = DEFAULT_MODELS[provider]

    project = gl.project
    merge_request = project.mergerequests.get(mr)

    # Get issue IID from MR
    issue_iid = _get_mr_issue_iid(merge_request)
    if issue_iid:
        issue = project.issues.get(issue_iid)
    else:
        issue = None

    developer = cm.get("developer")
    if not developer:
        console.print("[red]✗ No 'developer' agent found[/red]")
        raise typer.Exit(1)

    # Display MR info
    console.print(f"  [bold]MR[/bold]        !{mr} — {merge_request.title}")
    console.print(f"  [bold]Branch[/bold]    {merge_request.source_branch}")
    console.print(f"  [bold]Issue[/bold]     #{issue_iid or 'N/A'}")
    console.print(f"  [bold]Agent[/bold]     {developer.display_name} ({developer.model})")
    console.print()

    # Find when the last push happened
    latest_push_dt: datetime | None = None
    try:
        for commit in merge_request.commits():
            raw = (getattr(commit, "committed_date", "")
                   or getattr(commit, "created_at", ""))
            dt = _parse_ts(raw)
            if dt is not None:
                latest_push_dt = dt
                break
    except Exception:
        pass

    # Collect review notes
    review_notes = _collect_review_notes(merge_request, latest_push_dt)
    
    if not review_notes:
        console.print("[yellow]⚠ No review feedback found. Agent will check if implementation is complete.[/yellow]")
        feedback_str = "No specific feedback. Please verify the implementation meets all acceptance criteria."
    else:
        feedback_str = "\n\n".join(review_notes)
        console.print(f"  [bold]Reviews since last push ({len(review_notes)} item(s)):[/bold]")
        for r in review_notes:
            preview = r[:100] + "..." if len(r) > 100 else r
            console.print(f"    - {preview}")
        console.print()

    # Fetch the MR diff
    diff_text = _get_mr_diff(merge_request)

    issue_title = issue.title if issue else merge_request.title

    # For refinement the original issue spec is noise — the feedback is all that matters.
    enhanced_desc = f"MR: {merge_request.title}\n\n"
    if diff_text:
        enhanced_desc += f"## Current Diff\n```diff\n{diff_text}\n```\n\n"
    enhanced_desc += f"## Review Feedback to Address:\n{feedback_str}"

    router = LlmRouter()
    token = os.environ.get("GITLAB_TOKEN", "")

    if developer.provider == "stub":
        with console.status("[yellow]Analyzing refinement needed…[/yellow]"):
            try:
                llm_result = router.complete(
                    developer,
                    system=REFINEMENT_SYSTEM,
                    prompt=enhanced_desc,
                )
                result = ExecutionResult(
                    response=llm_result.content,
                    iterations=1,
                )
            except Exception as exc:
                console.print(f"[red]✗ Refinement analysis failed: {exc}[/red]")
                raise typer.Exit(1)
    else:
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
            clone_url=getattr(project, "http_url_to_repo", ""),
            progress=console.log,
        )
        console.print("[yellow]Developer addressing feedback…[/yellow]")
        try:
            result = engine.run(router=router, system=REFINEMENT_SYSTEM)
        except Exception as exc:
            console.print(f"[red]✗ Refinement failed: {exc}[/red]")
            raise typer.Exit(1)

    console.print()
    if not result.needs_continuation:
        console.print("[green]✓[/green] Refinement complete!")

        t = Table(box=box.SIMPLE, show_header=False)
        t.add_column(style="bold", min_width=14)
        t.add_column()
        t.add_row("Iterations", str(result.iterations))
        t.add_row("Files changed", str(len(result.files_changed)))
        if result.commit_sha:
            t.add_row("Commit", result.commit_sha[:8])
        console.print(t)

        if result.files_changed:
            console.print("  [bold]Changed files:[/bold]")
            for f in result.files_changed:
                console.print(f"    [cyan]{f}[/cyan]")
            console.print()

        commit_note = f"\n\n*Commit: {result.commit_sha[:8]}*" if result.commit_sha else ""
        new_desc = f"{merge_request.description or ''}\n\n---\n## Refinement by SoftwareTeamFabrik\n\n{result.response}{commit_note}"
        merge_request.description = new_desc
        merge_request.save()

        if merge_request.work_in_progress:
            merge_request.work_in_progress = False
            merge_request.save()
            console.print("  [green]✓[/green] Removed WIP status")

        if auto_merge:
            all_approvals = all(
                _is_approval_note(n.body or "")
                for n in merge_request.notes.list()
            )
            if all_approvals:
                console.print("  [yellow]All reviews approved, merging...[/yellow]")
                merge_request.merge(remove_source_branch=True)
                console.print("  [green]✓[/green] MR merged to main")

        console.print()
        console.print(f"  [bold]MR[/bold]  {merge_request.web_url}")
        console.print()

    else:
        console.print("[yellow]⚠[/yellow] Iteration limit hit — run refine again to continue.")
        console.print()

    console.print(Panel(
        result.response or "(no response)",
        title="Refinement Summary",
        border_style="cyan",
        expand=False,
    ))
    console.print()


def run_mr_refine(
    merge_request,
    project,
    gl_url: str,
    developer,
    router,
    gitlab_token: str,
    progress=None,
) -> "ExecutionResult":
    """Run the Developer agent to address review feedback on *merge_request*.

    Shared implementation used by both the CLI command and the monitor.
    Returns the ``ExecutionResult`` so callers can check ``commit_sha``,
    ``needs_continuation``, and ``files_changed``.
    """
    # Determine timestamp of the latest push.
    latest_push_dt = None
    try:
        for commit in merge_request.commits():
            raw = (getattr(commit, "committed_date", "")
                   or getattr(commit, "created_at", ""))
            dt = _parse_ts(raw)
            if dt is not None:
                latest_push_dt = dt
                break
    except Exception:
        pass

    # Collect review notes
    review_notes = _collect_review_notes(merge_request, latest_push_dt)
    
    feedback_str = (
        "\n\n".join(review_notes)
        if review_notes
        else "No specific feedback. Verify the implementation meets all acceptance criteria."
    )

    # Fetch the MR diff
    diff_text = _get_mr_diff(merge_request)

    issue_title = merge_request.title
    enhanced_desc = f"MR: {merge_request.title}\n\n"
    if diff_text:
        enhanced_desc += f"## Current Diff\n```diff\n{diff_text}\n```\n\n"
    enhanced_desc += f"## Review Feedback to Address:\n{feedback_str}"

    engine = CodeExecutionEngine(
        agent=developer,
        repo_path=Path.cwd(),
        issue_title=issue_title,
        issue_description=enhanced_desc,
        gitlab_token=gitlab_token,
        gitlab_project_id=str(project.id),
        gitlab_url=gl_url,
        branch_name=merge_request.source_branch,
        mr_iid=merge_request.iid,
        clone_url=getattr(project, "http_url_to_repo", ""),
        progress=progress or (lambda msg: None),
    )
    result = engine.run(router=router, system=REFINEMENT_SYSTEM)

    if result.commit_sha:
        commit_note = f"\n\n*Commit: {result.commit_sha[:8]}*"
        new_desc = (
            f"{merge_request.description or ''}\n\n---\n\n"
            f"## Refinement by SoftwareTeamFabrik\n\n{result.response}{commit_note}"
        )
        merge_request.description = new_desc
        merge_request.save()

    return result