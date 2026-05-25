'''Review Merge Request command — run the Code Reviewer agent and optionally spawn wiki agent.'''

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from factory.adapters.gitlab_client import GitLabClient
from factory.adapters.llm_router import LlmRouter
from factory.core.competence import CompetenceManager
from factory.core.verdict import set_verdict_label
from factory.commands.update_wiki import cmd_update_wiki

console = Console()
logger = logging.getLogger(__name__)

_DIFF_LIMIT = 50000
# Meta-reviewer only needs enough diff to validate findings, not a full review.
_META_DIFF_LIMIT = 10000

_REVIEW_PROMPT = """\
Please review the following merge request.

## MR Title
{title}

## Description
{description}

## Diff
```diff
{diff}
```

Linked issues: {issues}
"""

_META_REVIEW_PROMPT = """\
Please validate the following code review for false positives.

## MR Title
{title}

## Primary Review to Validate
{review}

## Diff (truncated for context)
~~~diff
{diff}
~~~
"""

# Severity order: lower index = less severe (toward APPROVE)
_SEVERITY_ORDER = ["green", "yellow", "red"]


@dataclass
class ReviewResult:
    """Structured result of a merge request review.

    Returned by :func:`run_mr_review` instead of a raw string, providing a
    parallel API to :func:`run_mr_refine`'s ``ExecutionResult``.

    Attributes:
        comment_body: The full Markdown review comment body.
        verdict: Final verdict color — ``"green"`` (APPROVE),
            ``"yellow"`` (REQUEST CHANGES), ``"red"`` (BLOCK), or an
            empty string if the verdict could not be determined.
        posted: Whether the review comment was posted to the MR.
    """

    comment_body: str
    verdict: str = ""
    posted: bool = False


def cmd_review_mr(
    mr: int = typer.Option(..., "--mr", "-m", help="GitLab MR IID to review"),
    config: str = typer.Option("config/factory.yml", "--config", "-c"),
    agents_dir: str = typer.Option("config/agents", "--agents"),
    post: bool = typer.Option(True, "--post/--no-post", help="Post review as a GitLab MR comment"),
    second_review: bool = typer.Option(
        True, "--second-review/--no-second-review",
        help="Run a Mistral meta-reviewer to filter false positives from the primary review",
    ),
    update_wiki: bool = typer.Option(
        True, "--update-wiki/--no-update-wiki",
        help="Spawn wiki agent to write documentation when MR is approved",
    ),
) -> None:
    """Run the Code Reviewer agent against a merge request and optionally spawn wiki agent."""

    console.print()
    console.rule(f"[bold cyan]SoftwareTeamFabrik — Reviewing MR !{mr}[/bold cyan]")
    console.print()

    gl = GitLabClient(config_path=config)
    connected, detail = gl.connect()
    if not connected:
        console.print(f"[red]✗ GitLab connection failed: {detail}[/red]")
        raise typer.Exit(1)

    cm = CompetenceManager(agents_dir=agents_dir)
    cm.load()

    reviewer = cm.get("reviewer")
    if not reviewer:
        console.print("[red]✗ No 'reviewer' agent found in config/agents/. "
                      "Make sure reviewer.yml exists.[/red]")
        raise typer.Exit(1)

    meta_reviewer = None
    if second_review:
        meta_reviewer = cm.get("meta_reviewer")
        if not meta_reviewer:
            console.print("[yellow]⚠ meta_reviewer agent not found in config/agents/ — "
                          "running primary-only review.[/yellow]")
            second_review = False

    project = gl.project
    merge_request = project.mergerequests.get(mr)

    console.print(f"  [bold]MR[/bold]       !{mr} — {merge_request.title}")
    console.print(f"  [bold]Author[/bold]   {merge_request.author['name']}")
    console.print(f"  [bold]Branch[/bold]   {merge_request.source_branch} → {merge_request.target_branch}")
    if second_review:
        console.print(f"  [bold]Mode[/bold]     Primary review + Mistral meta-review")
    if update_wiki:
        console.print(f"  [bold]Wiki[/bold]      Documentation will be generated if approved")
    console.print()

    # Fetch diff via changes() API, then fill in any empty-diff files by reading
    # their full content from the source branch.  GitLab truncates diffs for large
    # files (sets diff=""), so we fall back to the repository file API in those cases.
    diff_text = ""
    try:
        changes = merge_request.changes()
        raw_changes = []
        if isinstance(changes, dict) and "changes" in changes:
            raw_changes = changes["changes"]
        elif hasattr(changes, "changes") and changes.changes:
            raw_changes = changes.changes

        for change in raw_changes:
            old_path = change.get("old_path", "file")
            new_path = change.get("new_path", "file")
            file_diff = change.get("diff", "")

            if not file_diff and not change.get("deleted_file"):
                # Large file: diff was suppressed — fetch full content from branch
                try:
                    f = project.files.get(
                        file_path=new_path,
                        ref=merge_request.source_branch,
                    )
                    content = f.decode().decode("utf-8", errors="replace")
                    lines = content.splitlines(keepends=True)
                    file_diff = "".join(f"+{l}" for l in lines)
                    file_diff = f"[full file — diff suppressed by GitLab]\n{file_diff}"
                except Exception:
                    file_diff = "[diff unavailable and file could not be fetched from source branch]"

            diff_text += f"\n--- {old_path}\n+++ {new_path}\n{file_diff}"

        if not diff_text:
            # Fallback to diffs list
            diffs = merge_request.diffs.list()
            for d in diffs:
                if isinstance(d, dict) and "diff" in d:
                    diff_text += f"\n{d['diff']}"

        if len(diff_text) > _DIFF_LIMIT:
            diff_text = diff_text[:_DIFF_LIMIT] + f"\n\n[diff truncated — showing first {_DIFF_LIMIT:,} chars]"
    except Exception as e:
        import traceback
        diff_text = f"[Error fetching diff: {e}\n{traceback.format_exc()}]"

    linked_issues = (
        ", ".join(f"#{r['iid']}" for r in merge_request.references.get("related_issues", []))
        if hasattr(merge_request, "references") else "none listed"
    )

    prompt = _REVIEW_PROMPT.format(
        title=merge_request.title,
        description=merge_request.description or "(no description)",
        diff=diff_text or "(empty diff)",
        issues=linked_issues,
    )

    router = LlmRouter()

    # --- Primary review ---
    with console.status("[yellow]Code Reviewer analysing diff…[/yellow]"):
        try:
            result = router.complete(reviewer, system="", prompt=prompt)
        except Exception as exc:
            console.print(f"[red]✗ Reviewer failed: {exc}[/red]")
            raise typer.Exit(1)

    primary_color = _verdict_color(result.content) or "green"
    console.print(Panel(
        Text(result.content),
        title=f"[{primary_color}]Code Review — !{mr}[/{primary_color}]",
        border_style=primary_color,
        expand=False,
    ))

    # --- Optional meta-review ---
    meta_result = None
    meta_color: str | None = None
    final_color: str | None = None
    if second_review and meta_reviewer is not None:
        console.print()
        # Don't forward a traceback to the external API if diff fetching failed.
        if diff_text.startswith("[Error fetching diff:"):
            meta_diff = "(diff unavailable)"
        else:
            meta_diff = diff_text[:_META_DIFF_LIMIT]
            if len(diff_text) > _META_DIFF_LIMIT:
                meta_diff += f"\n\n[diff truncated — showing first {_META_DIFF_LIMIT:,} chars]"
        # An empty diff_text (no changes returned by GitLab) and a successful
        # but empty-content MR are both represented as "(empty diff)" here.
        meta_prompt = _META_REVIEW_PROMPT.format(
            title=merge_request.title,
            review=result.content,
            diff=meta_diff or "(empty diff)",
        )
        with console.status("[yellow]Meta Reviewer validating findings…[/yellow]"):
            try:
                meta_result = router.complete(meta_reviewer, system="", prompt=meta_prompt)
            except Exception as exc:
                console.print(f"[red]✗ Meta Reviewer failed: {exc}[/red]")
                meta_result = None

        if meta_result and meta_result.content and meta_result.content.strip():
            meta_color = _verdict_color(meta_result.content)
            final_color = _merge_verdicts(primary_color, meta_color)
            console.print(Panel(
                Text(meta_result.content),
                title=f"[{final_color}]Meta Review (Mistral) — !{mr}[/{final_color}]",
                border_style=final_color,
                expand=False,
            ))
        else:
            meta_result = None
            meta_color = None

    if post:
        comment_body = _build_comment(
            result, meta_result,
            primary_color=primary_color, meta_color=meta_color, final_color=final_color,
        )
        note = merge_request.notes.create({"body": comment_body})
        set_verdict_label(merge_request, final_color or primary_color)
        note_url = f"{gl._url}/{gl.project_path}/-/merge_requests/{mr}#note_{note.id}"
        console.print(f"\n  [green]✓[/green] Review posted: {note_url}")

    # --- Wiki documentation generation ---
    if update_wiki and (final_color or primary_color) == "green":
        console.print()
        console.print("[bold blue]MR approved — spawning wiki agent for documentation[/bold blue]")
        try:
            cmd_update_wiki(mr_id=mr, config=config)
            console.print("[green]✓[/green] Wiki documentation generated successfully")
        except Exception as e:
            console.print(f"[yellow]⚠ Wiki documentation generation failed: {e}[/yellow]")
            console.print("[yellow]⚠ MR can still be merged, but documentation should be added manually[/yellow]")

    console.print()


def _verdict_color(content: str) -> str | None:
    """Extract verdict color from a review.

    Returns one of "green", "yellow", "red", or None when no verdict can be
    parsed.  Callers that need a safe display color should use
    `_verdict_color(x) or "green"`.  Callers that merge verdicts should treat
    None as "keep the other side's verdict unchanged".

    First tries to match a structured verdict section header so text in issue
    descriptions (e.g. "this could BLOCK the release") does not misfire.
    Falls back to a raw content scan so a legitimately blocked MR is never
    silently upgraded to APPROVE due to a formatting quirk.
    """
    match = re.search(
        r'##\s+(?:Adjusted\s+)?Verdict\s*\r?\n\s*(✅ APPROVE|⚠️? REQUEST CHANGES|🚫 BLOCK)',
        content,
        re.MULTILINE,
    )
    if match:
        verdict = match.group(1)
        if "REQUEST CHANGES" in verdict:
            return "yellow"
        if "BLOCK" in verdict:
            return "red"
        return "green"
    # Fallback: structured header not found — scan raw content so a BLOCK
    # verdict that deviates from the expected format is never lost.
    # Note: the regex uses ⚠️? (variation selector optional) while the fallback
    # requires the full two-codepoint sequence.  Models that strip the selector
    # are caught by the regex path; models that include it are caught by both.
    if "🚫 BLOCK" in content:
        return "red"
    if "⚠️ REQUEST CHANGES" in content:
        return "yellow"
    return None


def _merge_verdicts(primary_color: str, meta_color: str) -> str:
    """Return the effective final verdict color.

    The meta-reviewer may improve by at most one severity step or keep the
    primary verdict, but never escalate. If the meta-reviewer emits a more
    severe verdict than the primary (e.g. primary=green, meta=red), the
    primary verdict is preserved. A multi-step jump (e.g. BLOCK→APPROVE) is
    capped to a single step (BLOCK→REQUEST CHANGES) and logged.

    Single-step improvements (REQUEST CHANGES → APPROVE) are intentionally
    allowed uncapped — only the two-step jump (BLOCK → APPROVE) is capped.
    """
    if primary_color not in _SEVERITY_ORDER:
        raise ValueError(f"Unknown primary_color: {primary_color!r}")
    primary_level = _SEVERITY_ORDER.index(primary_color)

    # If the meta-reviewer produced an unparseable verdict, keep the primary.
    if meta_color is None or meta_color not in _SEVERITY_ORDER:
        if meta_color is not None:
            logger.warning("Unknown meta_color %r; keeping primary verdict.", meta_color)
        return primary_color

    meta_level = _SEVERITY_ORDER.index(meta_color)
    if meta_level >= primary_level:
        # No improvement or escalation attempt — keep primary.
        return primary_color
    if meta_level < primary_level - 1:
        # Multi-step jump (e.g. BLOCK→APPROVE) — cap to one step.
        logger.warning(
            "Meta-reviewer attempted multi-step verdict improvement (%s → %s); capped at %s.",
            _SEVERITY_ORDER[primary_level],
            _SEVERITY_ORDER[meta_level],
            _SEVERITY_ORDER[primary_level - 1],
        )
        return _SEVERITY_ORDER[primary_level - 1]
    # Single-step improvement — allowed.
    return _SEVERITY_ORDER[meta_level]


def _build_comment(
    primary,
    meta,
    *,
    primary_color: str | None = None,
    meta_color: str | None = None,
    final_color: str | None = None,
) -> str:
    if primary_color is None:
        primary_color = _verdict_color(primary.content)
    # primary_display is only for panel borders; the merge uses the raw value so
    # an unparseable primary review (None) keeps meta_color unchanged rather than
    # being treated as APPROVE in the "Final verdict" label.
    primary_display = primary_color or "green"
    lines = [
        "### :mag: Code Review by SoftwareTeamFabrik",
        "",
        primary.content,
        "",
        "---",
        f"*Model: `{primary.model}` · " +
        f"Tokens: {primary.input_tokens} in / {primary.output_tokens} out*",
    ]

    if meta is not None:
        if final_color is None:
            if meta_color is None:
                meta_color = _verdict_color(meta.content)
            if primary_color is not None:
                final_color = _merge_verdicts(primary_color, meta_color)
            else:
                # Primary verdict unparseable — can't merge; show meta directly.
                final_color = meta_color or "green"
        verdict_emoji = {"green": "✅ APPROVE", "yellow": "⚠️ REQUEST CHANGES", "red": "🚫 BLOCK"}
        # fall back to the raw color name rather than silently claiming APPROVE
        final_label = verdict_emoji.get(final_color, final_color.upper())
        lines += [
            "",
            "---",
            "### :mag::mag: Meta Review (false-positive filter) by SoftwareTeamFabrik",
            "",
            meta.content,
            "",
            "---",
            f"*Model: `{meta.model}` · " +
            f"Tokens: {meta.input_tokens} in / {meta.output_tokens} out*",
            "",
            f"**Final verdict (after meta-review): {final_label}**",
            "*Note: The meta-reviewer can only keep or improve the primary verdict, never escalate it.*",
        ]

    return "\n".join(lines)


def run_mr_review(
    merge_request,
    project,
    router,
    reviewer,
    meta_reviewer=None,
    post: bool = True,
    update_wiki: bool = True,
    config: str = "config/factory.yml",
) -> ReviewResult:
    """Run primary + optional meta review on *merge_request* and optionally post as a note.

    Returns a :class:`ReviewResult` with the comment body, final verdict, and
    posting status — providing a structured, parallel API to
    :func:`factory.commands.refine.run_mr_refine`'s ``ExecutionResult``.
    When ``post=True`` the comment is also posted to the MR.
    """
    diff_text = ""
    try:
        changes = merge_request.changes()
        raw_changes = []
        if isinstance(changes, dict) and "changes" in changes:
            raw_changes = changes["changes"]
        elif hasattr(changes, "changes") and changes.changes:
            raw_changes = changes.changes

        for change in raw_changes:
            old_path = change.get("old_path", "file")
            new_path = change.get("new_path", "file")
            file_diff = change.get("diff", "")
            if not file_diff and not change.get("deleted_file"):
                try:
                    f = project.files.get(
                        file_path=new_path,
                        ref=merge_request.source_branch,
                    )
                    content = f.decode().decode("utf-8", errors="replace")
                    lines = content.splitlines(keepends=True)
                    file_diff = "".join(f"+{l}" for l in lines)
                    file_diff = f"[full file — diff suppressed by GitLab]\n{file_diff}"
                except Exception:
                    file_diff = "[diff unavailable]"
            diff_text += f"\n--- {old_path}\n+++ {new_path}\n{file_diff}"

        if len(diff_text) > _DIFF_LIMIT:
            diff_text = diff_text[:_DIFF_LIMIT] + f"\n\n[diff truncated]"
    except Exception as exc:
        diff_text = f"[Error fetching diff: {exc}]"

    linked_issues = (
        ", ".join(f"#{r['iid']}" for r in merge_request.references.get("related_issues", []))
        if hasattr(merge_request, "references") else "none listed"
    )

    prompt = _REVIEW_PROMPT.format(
        title=merge_request.title,
        description=merge_request.description or "(no description)",
        diff=diff_text or "(empty diff)",
        issues=linked_issues,
    )

    primary = router.complete(reviewer, system="", prompt=prompt)
    primary_color = _verdict_color(primary.content) or "green"

    meta_result = None
    meta_color: str | None = None
    final_color: str | None = None

    if meta_reviewer is not None:
        meta_diff = diff_text[:_META_DIFF_LIMIT]
        if diff_text.startswith("[Error fetching diff:"):
            meta_diff = "(diff unavailable)"
        meta_prompt = _META_REVIEW_PROMPT.format(
            title=merge_request.title,
            review=primary.content,
            diff=meta_diff or "(empty diff)",
        )
        try:
            meta_result = router.complete(meta_reviewer, system="", prompt=meta_prompt)
            if meta_result and meta_result.content and meta_result.content.strip():
                meta_color = _verdict_color(meta_result.content)
                final_color = _merge_verdicts(primary_color, meta_color)
            else:
                meta_result = None
        except Exception as exc:
            logger.warning("Meta Reviewer failed: %s", exc)
            meta_result = None

    comment_body = _build_comment(
        primary, meta_result,
        primary_color=primary_color, meta_color=meta_color, final_color=final_color,
    )

    posted = False
    if post:
        merge_request.notes.create({"body": comment_body})
        set_verdict_label(merge_request, final_color or primary_color)
        posted = True

    # Spawn wiki agent if approved
    if update_wiki and (final_color or primary_color) == "green":
        logger.info("MR approved — triggering wiki documentation generation")
        try:
            cmd_update_wiki(mr_id=merge_request.iid, config=config)
        except Exception as exc:
            logger.warning("Wiki documentation generation failed: %s", exc)

    return ReviewResult(
        comment_body=comment_body,
        verdict=final_color or primary_color,
        posted=posted,
    )