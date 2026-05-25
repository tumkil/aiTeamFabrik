# Copyright 2024 SoftwareTeamFabrik contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import re
import time
import yaml
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from factory.adapters.gitlab_client import GitLabClient
from factory.adapters.llm_router import LlmRouter
from factory.commands.refine import run_mr_refine
from factory.commands.review_mr import run_mr_review
from factory.core.competence import CompetenceManager
from factory.core.execution_engine import CodeExecutionEngine
from factory.core.scrum import ScrumEngine
from factory.core.token_budget import TokenBudgetManager
from factory.core.architect_feedback import (
    ArchitectFeedbackEnforcer,
    create_architect_context,
)
from factory.core.events import MergeRequestApprovedEvent
from factory.core.wiki_manager import WikiManager
from factory.core.constants import ARCH_NOTE_PREFIX
from factory.core.verdict import set_verdict_label

console = Console()

# WIP limits per agent type
WIP_LIMITS = {
    "developer": 2,  # Max concurrent issues for developer
    "reviewer": 3,    # Max concurrent reviews
}

class FactoryMonitor:
    """Monitors GitLab and triggers agent workflows for MRs and issues."""

    def __init__(
        self,
        config_path: str = "config/factory.yml",
        agents_dir: str = "config/agents",
        poll_interval: int = 30,
    ) -> None:
        self._config_path = config_path
        self._agents_dir = agents_dir
        self._poll_interval = poll_interval

        self._gl = GitLabClient(config_path=config_path)
        connected, detail = self._gl.connect()
        if not connected:
            raise RuntimeError(f"GitLab connection failed: {detail}")

        self._competence = CompetenceManager(agents_dir=agents_dir)
        self._competence.load()

        self._scrum = ScrumEngine(config_path=config_path)
        self._router = LlmRouter()
        self._token = os.environ.get("GITLAB_TOKEN", "")

        # Initialize token budget manager
        usage_path = Path(config_path).parent / "token_usage.yml"
        self._budget_manager = TokenBudgetManager(config_path, usage_path)

        # Initialize wiki manager
        self._wiki_manager = WikiManager()

        # Track active work items
        self._active_issues: dict[str, int] = {}  # agent_type -> count

        # Per-instance "last processed" tracking. Previously module-level
        # globals, which broke test isolation and prevented two monitors from
        # safely coexisting in the same process.
        self._last_processed_mrs: dict[int, datetime] = {}
        self._last_processed_issues: dict[int, datetime] = {}

        # Track whether sprint review has been triggered for the current sprint
        self._sprint_review_triggered: bool = False

        # Track the last known sprint number to detect changes
        self._last_known_sprint_number: int = self._scrum.current.number

        console.print(f"[green]✓[/green] Monitor connected to {self._gl.hostname}")

        if self._competence.get("architect") is None:
            console.print("[yellow]⚠[/yellow] No 'architect' agent profile found — Architect step will be skipped for all issues")

    def run(self) -> None:
        """Main monitoring loop."""
        console.print(Panel(
            "[bold]SoftwareTeamFabrik Monitor[/bold]\n"
            "Watching MRs and issues for autonomous processing...",
            border_style="cyan",
        ))
        console.print()

        try:
            while True:
                self._poll()
                time.sleep(self._poll_interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitor stopped by user[/yellow]")

    def _poll(self) -> None:
        """Poll GitLab for MR and issue events.

        The list of open MRs and all issues are each fetched **once** per poll
        cycle and shared with the lifecycle methods to avoid duplicate GitLab
        API calls.
        """
        timestamp = datetime.now()
        sprint_number = self._scrum.current.number if self._scrum and self._scrum.current else "unknown"
        console.print(f"[{timestamp.isoformat()}] Polling for sprint {sprint_number}...", style="dim")

        project = self._gl.project
        # Fetch open MRs once and share with both lifecycle methods.
        open_mrs = project.mergerequests.list(state="opened", all=True)
        # Fetch all issues (open AND closed) once and share with lifecycle
        # methods.  ``state="all"`` is required so that sprint-completion
        # detection can see closed issues; callers that need only open issues
        # filter locally.
        all_issues = project.issues.list(all=True, state="all")

        # Priority: Check MRs first before processing new issues
        self._check_mrs(project, open_mrs=open_mrs)
        self._check_issues(project, open_mrs=open_mrs, all_issues=all_issues)
        self._check_sprint_completion(project, all_issues=all_issues)

        # Check for sprint changes during each polling cycle
        self._check_sprint_change()

    def _check_mrs(self, project, open_mrs=None) -> None:
        """Check all open MRs for auto-review and refinement.
        
        Implements a refinement loop: when changes are requested, the Developer
        refines the code, then a re-review is triggered. This loop continues
        until all reviews are approved or no more changes are requested.

        Args:
            project: GitLab project object.
            open_mrs: Pre-fetched list of open MRs. If *None*, the list is
                fetched from GitLab (backward-compatible default).
        """
        if open_mrs is None:
            open_mrs = project.mergerequests.list(state="opened", all=True)

        developer = self._competence.get("developer")
        reviewer = self._competence.get("reviewer")
        meta_reviewer = self._competence.get("meta_reviewer")

        for mr in open_mrs:
            iid = mr.iid

            # Check if recently processed (skip only within the same poll cycle)
            if iid in self._last_processed_mrs:
                if (datetime.now() - self._last_processed_mrs[iid]).total_seconds() < self._poll_interval:
                    continue

            # GitLab 14+ uses `draft`; older versions used `work_in_progress`
            is_draft = getattr(mr, "draft", False) or getattr(mr, "work_in_progress", False)

            # === Priority 1: Review First (if no factory review exists) ===
            notes = mr.notes.list()
            has_factory_review = any(
                "Code Review by SoftwareTeamFabrik" in (n.body or "")
                for n in notes
            )

            if not has_factory_review:
                console.print(f"  Auto-reviewing MR !{iid}", style="cyan")
                self._run_review(mr, reviewer, meta_reviewer)
                # Refresh notes after review to get the new verdict
                notes = mr.notes.list()

            # === Priority 2: Refinement Loop ===
            # Continue refining until no more changes are requested or MR is approved
            refinement_iterations = 0
            max_refinement_iterations = 3  # Prevent infinite loops

            while refinement_iterations < max_refinement_iterations:
                has_request_changes = self._has_request_changes(notes, mr=mr)

                if has_request_changes:
                    console.print(f"  Refinement iteration {refinement_iterations + 1} for MR !{iid}", style="magenta")
                    try:
                        self._run_refinement(mr, developer)
                    except Exception as exc:
                        console.print(f"  ✗ Refinement error: {exc}", style="red")
                        console.print(f"  ℹ Error handled gracefully, continuing monitoring loop", style="blue")
                        # _run_refinement already handles its own internal
                        # retries; a propagated exception is unrecoverable in
                        # this cycle, so break out of the refinement loop.
                        break
                    refinement_iterations += 1

                    # Re-fetch MR state after refinement (draft status may have changed).
                    # Use a single-MR fetch instead of re-listing all MRs.
                    try:
                        mr = project.mergerequests.get(iid)
                    except Exception:
                        break
                    if getattr(mr, "state", "opened") != "opened":
                        break
                    is_draft = getattr(mr, "draft", False) or getattr(mr, "work_in_progress", False)

                    # After refinement, trigger re-review
                    console.print(f"  Triggering re-review for MR !{iid}...", style="cyan")
                    self._run_review(mr, reviewer, meta_reviewer)

                    # Re-fetch notes to check if review still requests changes
                    notes = mr.notes.list()
                    has_request_changes = self._has_request_changes(notes, mr=mr)

                    if not has_request_changes:
                        console.print(f"  MR !{iid} refinement loop complete", style="green")
                        break
                else:
                    # No changes requested, exit refinement loop
                    break

            # Mark as processed after refinement loop completes
            self._last_processed_mrs[iid] = datetime.now()

            # === Priority 3: All Approved - Auto Merge to Main ===
            notes = mr.notes.list()
            if self._all_approved(mr, notes) and not is_draft:
                console.print(f"  MR !{iid} approved, auto-merging to main...", style="green")
                self._auto_merge(mr)
                self._last_processed_mrs[iid] = datetime.now()

    def _check_issues(self, project, open_mrs=None, all_issues=None) -> None:
        """Check all open issues for autonomous processing.

        For each open issue the lifecycle is:
        1. No MR for the issue yet → run Developer, push, create MR.
        2. MR exists + needs changes → run Developer refinement.
        3. MR exists + not reviewed → run Reviewer.
        4. MR exists + all approved → auto-merge (handled in _check_mrs).
        5. MR exists + already approved → skip (nothing to do).

        Args:
            project: GitLab project object.
            open_mrs: Pre-fetched list of open MRs. If *None*, the list is
                fetched from GitLab (backward-compatible default).  The list
                is re-filtered to exclude any MRs that may have been merged
                during the preceding ``_check_mrs`` pass.
            all_issues: Pre-fetched list of all issues (open AND closed).
                If *None*, only open issues are fetched from GitLab
                (backward-compatible default).
        """
        if all_issues is not None:
            issues = [issue for issue in all_issues if issue.state == "opened"]
        else:
            issues = project.issues.list(state="opened", all=True)

        # Filter issues to only those in the current sprint
        sprint_issues = self._filter_issues_for_current_sprint(issues)

        # Filter issues to only those labeled as "ready"
        ready_issues = self._filter_issues_by_ready_label(sprint_issues)

        # Sort issues by IID in ascending order (10, 11, 12, ...)
        ready_issues = sorted(ready_issues, key=lambda issue: issue.iid)

        developer = self._competence.get("developer")
        architect = self._competence.get("architect")
        reviewer = self._competence.get("reviewer")
        meta_reviewer = self._competence.get("meta_reviewer")

        # Build a quick lookup: source_branch → mr (only opened MRs).
        # Re-filter the pre-fetched list to drop MRs merged during _check_mrs.
        if open_mrs is None:
            open_mrs = project.mergerequests.list(state="opened", all=True)
        else:
            open_mrs = [m for m in open_mrs if getattr(m, "state", "opened") == "opened"]
        mr_by_branch: dict[str, object] = {mr.source_branch: mr for mr in open_mrs}

        for issue in ready_issues:
            iid = issue.iid

            # Skip if recently processed
            if iid in self._last_processed_issues:
                if (datetime.now() - self._last_processed_issues[iid]).total_seconds() < self._poll_interval:
                    continue

            branch_name = f"issue-{iid}"
            mr = mr_by_branch.get(branch_name)

            if mr is None:
                # --- No MR yet: developer implements, push, create MR ---
                if not self._can_start_work("developer"):
                    console.print(
                        f"  WIP limit reached for developer, skipping issue #{iid}", style="yellow"
                    )
                    continue

                token_cost = self._estimate_issue_cost(issue)
                if not self._has_sufficient_budget(token_cost):
                    console.print(
                        f"  Insufficient budget for issue #{iid} (cost: {token_cost} tokens)", style="red"
                    )
                    continue

                console.print(f"  Processing issue #{iid}: {issue.title[:50]}…", style="blue")
                created_mr = self._run_issue_processing(issue, developer, architect)
                if created_mr is not None:
                    mr_by_branch[branch_name] = created_mr
                self._last_processed_issues[iid] = datetime.now()
                continue

            # --- MR exists: apply the same lifecycle as _check_mrs ---
            notes = mr.notes.list()
            is_draft = getattr(mr, "draft", False) or getattr(mr, "work_in_progress", False)

            if self._has_request_changes(notes, mr=mr):
                console.print(
                    f"  Refinement needed for MR !{mr.iid} (issue #{iid})", style="magenta"
                )
                try:
                    self._run_refinement(mr, developer)
                except Exception as exc:
                    console.print(f"  ✗ Refinement error: {exc}", style="red")
                    console.print(f"  ℹ Error handled gracefully, continuing monitoring loop", style="blue")
                self._last_processed_issues[iid] = datetime.now()
                continue

            has_factory_review = any(
                "Code Review by SoftwareTeamFabrik" in (n.body or "") for n in notes
            )
            if not has_factory_review:
                console.print(
                    f"  Auto-reviewing MR !{mr.iid} for issue #{iid}", style="cyan"
                )
                self._run_review(mr, reviewer, meta_reviewer)
                self._last_processed_issues[iid] = datetime.now()
                continue

            if self._all_approved(mr, notes) and not is_draft:
                console.print(
                    f"  MR !{mr.iid} approved, auto-merging (issue #{iid})…", style="green"
                )
                self._auto_merge(mr)
                self._last_processed_issues[iid] = datetime.now()
                continue

            # Approved or waiting — nothing to do this cycle.
            console.print(
                f"  Issue #{iid}: MR !{mr.iid} in progress, no action needed", style="dim"
            )

    def _check_sprint_change(self) -> None:
        """Check if the sprint has changed and reload the scrum engine if needed."""
        if self._scrum is None:
            return
        current_sprint_number = self._scrum.current.number
        if current_sprint_number != self._last_known_sprint_number:
            console.print(f"  Sprint changed from {self._last_known_sprint_number} to {current_sprint_number}, reloading...", style="bold")
            self._scrum = ScrumEngine(config_path=self._config_path)
            self._last_known_sprint_number = current_sprint_number
            # Reset the sprint review triggered flag for the new sprint
            self._sprint_review_triggered = False

    def _filter_issues_for_current_sprint(self, issues):
        """Filter issues to only those in the current sprint."""
        if self._scrum is None or self._scrum.current is None:
            return issues
        sprint_label = f"{self._scrum.current.milestone_prefix} {self._scrum.current.number}"
        sprint_issues = []
        for issue in issues:
            labels = issue.labels or []
            if sprint_label in labels:
                sprint_issues.append(issue)
        return sprint_issues

    def _filter_issues_by_ready_label(self, issues):
        """Filter issues to only those labeled as 'ready'."""
        ready_issues = []
        for issue in issues:
            labels = issue.labels or []
            if "ready" in labels:
                ready_issues.append(issue)
        return ready_issues

    def _check_sprint_completion(self, project, all_issues=None) -> None:
        """Check if all issues in the current sprint are merged and trigger a sprint review.
        
        In monitor mode, this method does NOT automatically close the sprint or advance to the next sprint.
        It logs the completion and waits for manual intervention. This ensures that the sprint review
        and advancement are controlled by the user in monitor mode.

        Args:
            project: GitLab project object.
            all_issues: Pre-fetched list of all issues (open AND closed). If
                *None*, the list is fetched from GitLab (backward-compatible
                default).
        """
        # Use pre-fetched issues if available, otherwise fetch from GitLab.
        # ``state="all"`` is required so that closed sprint issues are visible;
        # without it the sprint would appear incomplete forever.
        if all_issues is None:
            all_issues = project.issues.list(all=True, state="all")

        sprint_issues = self._filter_issues_for_current_sprint(all_issues)

        # Filter to only open issues in the current sprint
        open_sprint_issues = [issue for issue in sprint_issues if issue.state == "opened"]

        # Sprint is complete when there are no open issues in the current sprint
        # AND we haven't already triggered the sprint review for this sprint
        if not open_sprint_issues and sprint_issues and not self._sprint_review_triggered:
            # Set the flag to True immediately to prevent multiple triggers
            self._sprint_review_triggered = True
            console.print("  All sprint issues are merged. Sprint completion detected.", style="green")
            console.print("  In monitor mode: waiting for manual sprint review and advancement.", style="yellow")
            # Do NOT call _trigger_sprint_review() in monitor mode

    def _has_mr_reference(self, issue) -> bool:
        """Check if issue description contains MR reference."""
        description = issue.description or ""
        return bool(re.search(r'!\d+', description))

    def _can_start_work(self, agent_type: str) -> bool:
        """Check if we can start new work based on WIP limits."""
        current = self._active_issues.get(agent_type, 0)
        limit = WIP_LIMITS.get(agent_type, 1)
        return current < limit

    def _estimate_issue_cost(self, issue) -> int:
        """Estimate token cost for processing an issue.
        
        Enhanced estimation that considers:
        - Issue description length
        - Number of labels (complexity indicator)
        - Presence of code blocks or technical details
        - Issue age (older issues may require more context)
        """
        description = issue.description or ""
        
        # Base cost based on issue description length
        word_count = len(description.split())
        base_cost = word_count * 15  # 15 tokens per word for context processing
        
        # Add cost for labels (more labels = more complex)
        label_cost = len(issue.labels or []) * 100
        
        # Add cost for code blocks (technical issues require more tokens)
        code_block_cost = 0
        if "```" in description:
            code_block_cost = 500  # Fixed cost for code blocks
        
        # Add cost for issue age (older issues may need more context)
        created_at = getattr(issue, 'created_at', None)
        age_cost = 0
        if created_at:
            try:
                if isinstance(created_at, str):
                    from datetime import timezone
                    created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    now_dt = datetime.now(tz=timezone.utc)
                else:
                    created_dt = created_at
                    now_dt = datetime.now()
                age_days = (now_dt - created_dt).days
                age_cost = min(age_days * 20, 1000)  # Cap at 1000 tokens
            except (ValueError, TypeError):
                pass
        
        # Add cost for linked issues/MRs (more context to process)
        linked_cost = 0
        if self._has_mr_reference(issue):
            linked_cost = 300
        
        # Minimum cost to ensure we always have a reasonable estimate
        total_cost = max(2000, base_cost + label_cost + code_block_cost + age_cost + linked_cost)
        
        return total_cost

    def _has_sufficient_budget(self, token_cost: int) -> bool:
        """Check if we have sufficient budget for processing."""
        # Check if budget enforcement is enabled
        if os.environ.get("FACTORY_IGNORE_BUDGET", "").lower() in ("1", "true", "yes"):
            return True
        
        # Use budget manager to check if we can consume these tokens
        can_consume, reason = self._budget_manager.can_consume("developer", token_cost)
        if not can_consume:
            console.print(f"  ⚠ Budget check failed: {reason}", style="yellow")
        return can_consume

    def _run_architect(self, issue, architect) -> str:
        """Run the Architect agent on an issue and post its plan as a note.

        Returns the architect plan text (without the note header), or empty string on failure.
        The note is idempotent: if a previous run already posted an Architect Analysis note
        it is reused without calling the LLM again.
        """
        iid = issue.iid
        title = issue.title
        description = issue.description or ""

        # Check if an architect plan already exists in issue notes.
        # Match on the exact prefix so human comments mentioning "Architect Analysis"
        # are not mistaken for a machine-generated plan.
        try:
            notes = issue.notes.list(all=True)
            existing = [
                n.body for n in notes
                if n.body and n.body.startswith(ARCH_NOTE_PREFIX)
            ]
            if existing:
                console.print(f"  Architect plan already exists for issue #{iid}", style="dim")
                # Use the first match — there should only ever be one machine-generated
                # plan note, and oldest-first ordering means existing[0] is canonical.
                # Strip the note header so the caller gets only the raw plan text,
                # avoiding a double-header when it is prepended in _run_issue_processing.
                return existing[0].removeprefix(ARCH_NOTE_PREFIX)
        except Exception as exc:
            # Fall through and run the Architect anyway — a missing notes list is not fatal.
            console.print(f"  ⚠ Could not fetch notes for issue #{iid}: {exc}", style="yellow")

        console.print(f"  Running Architect on issue #{iid}...", style="magenta")
        # Include the full issue context (title and description) for the Architect
        prompt = f"Issue #{iid}: {title}\n\n{description}"
        try:
            # The router resolves agent.system_prompt as first priority; passing it
            # explicitly here makes that intent clear and guards against a None value.
            response = self._router.complete(architect, system=architect.system_prompt or "", prompt=prompt)
            plan = response.content
            if not plan:
                console.print(f"  ⚠ Architect returned empty plan for issue #{iid}, skipping note", style="yellow")
                return ""

            note_body = f"{ARCH_NOTE_PREFIX}{plan}"
            issue.notes.create({"body": note_body})
            console.print(f"  ✓ Architect plan posted for issue #{iid}", style="green")
            return plan
        except Exception as exc:
            console.print(f"  ✗ Architect failed for issue #{iid}: {exc}", style="red")
            return ""

    def _run_issue_processing(self, issue, developer, architect=None):
        """Run Architect then Developer agent to process an issue and create MR.

        Returns the newly-created MR object, or ``None`` on failure.
        """
        iid = issue.iid
        title = issue.title
        description = issue.description or ""

        # Step 1: run Architect to produce and post a plan
        arch_plan = ""
        if architect is not None:
            arch_plan = self._run_architect(issue, architect)

        # Step 2: build the developer's full context with architect guidance
        issue_description = description
        if arch_plan:
            # Use architect feedback enforcer to create proper context
            enforcer = ArchitectFeedbackEnforcer(arch_plan)
            issue_description = create_architect_context(arch_plan, description)
            # Add architect guidance reminder to the context
            issue_description += enforcer.get_adherence_prompt()

        # Mark as active work
        self._active_issues["developer"] = self._active_issues.get("developer", 0) + 1

        created_mr = None
        try:
            console.print(f"  Running Developer on issue #{iid}...", style="yellow")

            engine = CodeExecutionEngine(
                agent=developer,
                repo_path=Path.cwd(),
                issue_title=title,
                issue_description=issue_description,
                gitlab_token=self._token,
                gitlab_project_id=str(self._gl.project.id),
                gitlab_url=self._gl._url,
                branch_name=f"issue-{iid}",
                clone_url=getattr(self._gl.project, "http_url_to_repo", ""),
                progress=console.log,
            )

            result = engine.run(router=self._router)

            if result.commit_sha:
                # Developer pushed code — create an MR for it.
                created_mr = self._create_mr_for_issue(
                    issue=issue,
                    branch_name=f"issue-{iid}",
                    developer_summary=result.response,
                    commit_sha=result.commit_sha,
                )
            elif not result.needs_continuation:
                # No files changed (already implemented?) — label, and close
                # the issue if the developer explicitly reported it was
                # already implemented.
                console.print(f"  ✓ Issue #{iid} complete (no new commits)", style="green")
                response_text = (result.response or "").strip()
                already_implemented = response_text.lower().startswith("already implemented")
                new_labels = set((issue.labels or []) + ["processed"])
                if already_implemented:
                    new_labels.add("already-implemented")
                issue.labels = list(new_labels)
                if already_implemented:
                    issue.state_event = "close"
                try:
                    issue.save()
                except Exception as exc:
                    console.print(
                        f"  ⚠ Could not update issue #{iid}: {exc}", style="yellow"
                    )
                if already_implemented:
                    console.print(
                        f"  ✓ Issue #{iid} auto-closed (already implemented)", style="green"
                    )
            else:
                console.print(f"  ⚠ Issue #{iid} processing hit iteration limit", style="yellow")

        except Exception as exc:
            console.print(f"  ✗ Issue processing error: {exc}", style="red")
        finally:
            self._active_issues["developer"] = max(
                0, self._active_issues.get("developer", 0) - 1
            )

        return created_mr

    def _create_mr_for_issue(self, issue, branch_name: str, developer_summary: str,
                              commit_sha: str) -> object | None:
        """Create a GitLab MR from ``branch_name`` → default branch for ``issue``."""
        iid = issue.iid
        project = self._gl.project
        try:
            default_branch = project.default_branch or "main"
            mr_title = f"feat: resolve issue #{iid} — {issue.title[:60]}"
            mr_description = (
                f"Closes #{iid}\n\n"
                f"## Summary\n\n{developer_summary}\n\n"
                f"---\n*Automatically created by SoftwareTeamFabrik*  \n"
                f"*Commit: `{commit_sha[:8]}`*"
            )
            # Use the retry-enabled create_merge_request method
            mr = self._gl.create_merge_request(
                source_branch=branch_name,
                target_branch=default_branch,
                title=mr_title,
                description=mr_description,
                remove_source_branch=True,
                max_retries=3,
            )
            if mr:
                console.print(
                    f"  ✓ MR !{mr.iid} created for issue #{iid}", style="green"
                )
                return mr
            else:
                console.print(
                    f"  ✗ Failed to create MR for issue #{iid} after retries", style="red"
                )
                return None
        except Exception as exc:
            console.print(f"  ✗ Failed to create MR for issue #{iid}: {exc}", style="red")
            return None

    def _has_request_changes(self, notes, mr=None) -> bool:
        """Return True if the latest review verdict requests changes.

        Verdict labels (set by ``factory.core.verdict.set_verdict_label`` when
        a review is posted) are the source of truth when present. The legacy
        comment-scan path remains as a fallback for MRs reviewed before the
        labelling refactor and for human blocker notes.
        """
        if mr is not None:
            from factory.core.verdict import is_changes_requested
            label_verdict = is_changes_requested(mr)
            if label_verdict is not None:
                return label_verdict

        # Find the latest factory review verdict.
        latest_factory_body = None
        for n in reversed(notes):
            body = n.body or ""
            if "Code Review by SoftwareTeamFabrik" in body:
                latest_factory_body = body
                break

        if latest_factory_body is not None:
            return "REQUEST CHANGES" in latest_factory_body or "BLOCK" in latest_factory_body

        # No factory review yet — check for human blocker notes.
        for n in notes:
            body = n.body or ""
            if "REQUEST CHANGES" in body or "BLOCK" in body:
                return True
        return False

    def _all_approved(self, mr, notes) -> bool:
        """Return True if the latest factory review is an approval.

        Verdict labels are checked first; the comment-scan fallback handles
        MRs reviewed before the labelling refactor.
        """
        from factory.core.verdict import is_approved
        label_verdict = is_approved(mr)
        if label_verdict is not None:
            return label_verdict

        for n in reversed(notes):
            body = n.body or ""
            if "Code Review by SoftwareTeamFabrik" not in body:
                continue
            # This is the latest factory review.
            approved = "APPROVE" in body or "✅" in body
            rejected = "REQUEST CHANGES" in body or "BLOCK" in body or "❌" in body
            # A note can contain both if meta-review downgraded the verdict;
            # prefer rejection in that case.
            return approved and not rejected
        return False  # no factory review at all
    

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        """Return True for network/connection errors that are safe to retry.

        Thin wrapper around :func:`factory.core.resilience.is_transient`.
        """
        from factory.core.resilience import is_transient
        return is_transient(exc)

    def _run_refinement(self, mr, developer) -> None:
        """Run the refine command logic on an MR to address review feedback."""
        iid = mr.iid
        console.print(f"  Running Developer (refine) on MR !{iid}…", style="yellow")
        for attempt in range(3):
            try:
                result = run_mr_refine(
                    merge_request=mr,
                    project=self._gl.project,
                    gl_url=self._gl._url,
                    developer=developer,
                    router=self._router,
                    gitlab_token=self._token,
                    progress=console.log,
                )

                if not result.needs_continuation:
                    console.print(f"  ✓ Refinement successful", style="green")
                    if result.files_changed:
                        console.print(f"    Files changed: {len(result.files_changed)}")

                    commit_note = f"\n\n*Commit: {result.commit_sha[:8]}*" if result.commit_sha else ""
                    new_desc = f"{mr.description or ''}\n\n---\n## Refinement Applied by SoftwareTeamFabrik\n\n{result.response}{commit_note}\n\n*Iterations: {result.iterations}*"
                    mr.description = new_desc
                    mr.save()

                    if mr.work_in_progress:
                        mr.work_in_progress = False
                        mr.save()
                else:
                    # ExecutionResult has no 'error' field; use continuation_context or response.
                    error_msg = result.continuation_context or result.response or "unknown reason"
                    console.print(f"  ✗ Refinement failed: {error_msg}", style="red")
                return  # success or non-retryable outcome

            except Exception as exc:
                if self._is_transient_error(exc) and attempt < 2:
                    delay = (attempt + 1) * 15
                    console.print(
                        f"  ⚠ Refinement connection error (attempt {attempt + 1}/3), "
                        f"retrying in {delay}s: {exc}", style="yellow"
                    )
                    time.sleep(delay)
                else:
                    console.print(f"  ✗ Refinement error: {exc}", style="red")
                    console.print(f"  ℹ Error handled gracefully, continuing monitoring loop", style="blue")
                    return

    def _run_review(self, mr, reviewer, meta_reviewer=None) -> None:
        """Run the review-mr command logic (primary + meta) and post result."""
        iid = mr.iid
        for attempt in range(3):
            try:
                run_mr_review(
                    merge_request=mr,
                    project=self._gl.project,
                    router=self._router,
                    reviewer=reviewer,
                    meta_reviewer=meta_reviewer,
                    post=True,
                )
                console.print(f"  ✓ Review posted for MR !{iid}", style="green")
                return
            except Exception as exc:
                if self._is_transient_error(exc) and attempt < 2:
                    delay = (attempt + 1) * 15
                    console.print(
                        f"  ⚠ Review connection error (attempt {attempt + 1}/3), "
                        f"retrying in {delay}s: {exc}", style="yellow"
                    )
                    time.sleep(delay)
                else:
                    console.print(f"  ✗ Review failed for MR !{iid}: {exc}", style="red")
                    # Add error handling for reviewer agent trigger
                    console.print(f"  ℹ Error handled gracefully, continuing monitoring loop", style="blue")
                    return

    def _auto_merge(self, mr) -> None:
        """Auto-merge MR to main branch.

        If the merge fails, checks the pipeline status. When the pipeline
        has failed, retrieves the logs, labels the MR as needing changes,
        and triggers the Developer agent to repair the failing tests.
        """
        iid = mr.iid
        try:
            mr.merge(remove_source_branch=True)
            console.print(f"  ✓ MR !{iid} merged to main", style="green")

            # Add merge comment
            comment_body = "### :rocket: Auto-merged by SoftwareTeamFabrik\n\nAll reviews approved."
            mr.notes.create({"body": comment_body})

            # Trigger wiki documentation agent
            self._trigger_wiki_documentation(mr)

            # Wait 10 seconds after successful merge before continuing
            console.print(f"  ⏳ Waiting 10 seconds after merge before next poll...", style="dim")
            time.sleep(10)

        except Exception as e:
            error_str = str(e)
            console.print(f"  ✗ Merge failed: {e}", style="red")
            # On any merge failure, check if the pipeline failed and
            # trigger repair if needed.
            console.print(f"  ℹ Checking pipeline status for MR !{iid}...", style="blue")
            self._check_pipeline_status(mr)

    def _check_pipeline_status(self, mr) -> None:
        """Check the pipeline status of a merge request and handle accordingly.

        When a pipeline has failed, this method delegates to
        :meth:`_handle_pipeline_failure` which retrieves the failure logs,
        labels the MR as needing changes, and triggers the Developer agent
        to repair the failing tests.
        """
        iid = mr.iid
        try:
            # Get the pipelines for this merge request
            pipelines = mr.pipelines.list()
            
            if not pipelines:
                console.print(f"  ⚠ No pipelines found for MR !{iid}", style="yellow")
                return
            
            # Get the latest pipeline
            latest_pipeline = pipelines[0]
            
            # Check the pipeline status
            status = latest_pipeline.status
            console.print(f"  📊 Latest pipeline status: {status}", style="blue")
            
            # Handle different pipeline states
            if status == "running":
                console.print(f"  ⏳ Pipeline is still running for MR !{iid}, will retry merge later", style="yellow")
            elif status == "pending":
                console.print(f"  ⏳ Pipeline is pending for MR !{iid}, will retry merge later", style="yellow")
            elif status == "failed":
                console.print(f"  ✗ Pipeline failed for MR !{iid}", style="red")
                self._handle_pipeline_failure(mr)
            elif status == "success":
                console.print(f"  ✓ Pipeline passed for MR !{iid}, ready to merge", style="green")
            elif status == "canceled":
                console.print(f"  ⚠ Pipeline was canceled for MR !{iid}", style="yellow")
            elif status == "skipped":
                console.print(f"  ℹ Pipeline was skipped for MR !{iid}", style="blue")
            else:
                console.print(f"  ℹ Pipeline status unknown: {status}", style="blue")
            
        except Exception as e:
            console.print(f"  ✗ Failed to check pipeline status for MR !{iid}: {e}", style="red")

    def _handle_pipeline_failure(self, mr) -> None:
        """Handle a failed pipeline by retrieving logs, labeling, and triggering repair.

        When a pipeline fails on an MR, this method:

        1. Retrieves the failed pipeline logs via
           :meth:`~factory.adapters.gitlab_client.GitLabClient.get_failed_pipeline_logs`.
        2. Posts a note on the MR with the logs and a ``REQUEST CHANGES``
           marker so the refinement flow picks it up.
        3. Sets the verdict label to ``factory-verdict-changes`` so the
           monitor's refinement loop recognises the MR needs work.
        4. Triggers the Developer agent to repair the failing tests.
        """
        iid = mr.iid
        console.print(f"  🔧 Handling pipeline failure for MR !{iid}...", style="blue")

        # Step 1: Retrieve failed pipeline logs
        logs = ""
        try:
            logs = self._gl.get_failed_pipeline_logs(iid)
            if logs:
                console.print(f"  📋 Retrieved pipeline failure logs for MR !{iid}", style="blue")
            else:
                console.print(f"  ⚠ No pipeline logs available for MR !{iid}", style="yellow")
                logs = "Pipeline failed but no logs could be retrieved."
        except Exception as exc:
            console.print(f"  ✗ Failed to retrieve pipeline logs for MR !{iid}: {exc}", style="red")
            logs = f"Pipeline failed but log retrieval encountered an error: {exc}"

        # Step 2: Post a note on the MR with the pipeline logs
        # Include "REQUEST CHANGES" so the note is picked up by the
        # refinement flow's _collect_review_notes helper.
        note_body = (
            "### :construction: Pipeline Failure — REQUEST CHANGES\n\n"
            "The pipeline for this merge request has failed. "
            "The Developer agent has been triggered to repair the failing tests.\n\n"
            f"{logs}"
        )
        try:
            mr.notes.create({"body": note_body})
            console.print(f"  📝 Posted pipeline failure note on MR !{iid}", style="blue")
        except Exception as exc:
            console.print(f"  ✗ Failed to post pipeline failure note: {exc}", style="red")

        # Step 3: Set the verdict label to factory-verdict-changes
        try:
            label = set_verdict_label(mr, "yellow")
            if label:
                console.print(f"  🏷 Set verdict label to {label} on MR !{iid}", style="blue")
            else:
                console.print(f"  ⚠ Failed to set verdict label on MR !{iid}", style="yellow")
        except Exception as exc:
            console.print(f"  ✗ Failed to set verdict label: {exc}", style="red")

        # Step 4: Trigger the Developer agent to repair the failing tests
        developer = self._competence.get("developer")
        if developer is None:
            console.print(f"  ✗ No developer agent available for pipeline repair", style="red")
            return

        console.print(f"  🔧 Triggering Developer to repair pipeline failures for MR !{iid}...", style="yellow")
        try:
            self._run_refinement(mr, developer)
            console.print(f"  ✓ Pipeline repair refinement triggered for MR !{iid}", style="green")
        except Exception as exc:
            console.print(f"  ✗ Pipeline repair refinement failed for MR !{iid}: {exc}", style="red")

    def _extract_po_comments(self, mr) -> str:
        """Extract PO (Product Owner) comments from MR notes.
        
        Args:
            mr: Merge request object
            
        Returns:
            String containing all PO comments, or empty string if none found
        """
        po_comments = []
        try:
            notes = mr.notes.list(all=True)
            for note in notes:
                body = note.body or ""
                # Look for PO-specific markers or patterns
                if any(marker in body for marker in ["[PO]", "[Product Owner]", "PO Comment:", "Product Owner Feedback:"]):
                    po_comments.append(body)
                # Also include comments from users with PO-like roles
                elif "priority" in body.lower() or "business value" in body.lower() or "stakeholder" in body.lower():
                    po_comments.append(body)
        except Exception as exc:
            console.print(f"  ⚠ Could not extract PO comments: {exc}", style="yellow")
        
        return "\n\n".join(po_comments) if po_comments else "No PO comments found"

    def _extract_architect_plan(self, mr) -> str:
        """Extract architect plan from the linked issue.
        
        Args:
            mr: Merge request object
            
        Returns:
            String containing the architect plan, or empty string if none found
        """
        # Extract issue IID from MR
        issue_iid = self._extract_issue_iid(mr)
        if issue_iid is None:
            return "No linked issue found for architect plan"
        
        try:
            project = self._gl.project
            issue = project.issues.get(issue_iid)
            
            # Look for architect analysis notes
            notes = issue.notes.list(all=True)
            for note in notes:
                body = note.body or ""
                if body.startswith(ARCH_NOTE_PREFIX):
                    # Return the architect plan without the prefix
                    return body.removeprefix(ARCH_NOTE_PREFIX)
            
            return "No architect plan found for linked issue"
        except Exception as exc:
            console.print(f"  ⚠ Could not extract architect plan: {exc}", style="yellow")
            return f"Error extracting architect plan: {exc}"

    def _trigger_wiki_documentation(self, mr) -> None:
        """Trigger the wiki agent to write documentation for the merged MR."""
        iid = mr.iid
        title = mr.title
        project_id = self._gl.project.id
        
        # Create event for the approved MR
        event = MergeRequestApprovedEvent(
            mr_iid=iid,
            mr_title=title,
            project_id=project_id,
        )
        
        console.print(f"  📝 Spawning wiki agent for MR !{iid}...", style="blue")
        
        # Get the wiki agent profile
        wiki_agent = self._competence.get("wiki")
        if wiki_agent is None:
            console.print(f"  ⚠ No 'wiki' agent profile found — skipping documentation", style="yellow")
            return
        
        # Extract PO comments and architect plan for enhanced context
        po_comments = self._extract_po_comments(mr)
        architect_plan = self._extract_architect_plan(mr)
        
        # Build context for the wiki agent with enhanced information
        context = {
            "mr_title": title,
            "mr_iid": iid,
            "project_id": project_id,
            "date": datetime.now().isoformat(),
            "po_comments": po_comments,
            "architect_plan": architect_plan,
        }
        
        # Generate documentation using the wiki agent
        try:
            prompt = self._wiki_manager.format_template("architecture_overview", **context)
            response = self._router.complete(wiki_agent, system=wiki_agent.system_prompt or "", prompt=prompt)
            documentation = response.content
            
            if documentation:
                # Create or update wiki page
                wiki_title = f"MR-{iid}-Documentation"
                wiki_content = f"# Documentation for MR !{iid}\n\n{documentation}"
                
                # Use GitLab client to create/update wiki page
                wiki_page = self._gl.create_or_update_wiki_page(wiki_title, wiki_content)
                if wiki_page:
                    console.print(f"  ✓ Wiki documentation created/updated for MR !{iid}", style="green")
                else:
                    console.print(f"  ✗ Failed to create/update wiki page for MR !{iid}", style="red")
            else:
                console.print(f"  ⚠ Wiki agent returned empty documentation for MR !{iid}", style="yellow")
        except Exception as exc:
            console.print(f"  ✗ Wiki documentation failed for MR !{iid}: {exc}", style="red")

    def _extract_issue_iid(self, mr) -> int | None:
        """Extract issue IID from MR title and description."""
        title = mr.title or ""
        description = mr.description or ""
        combined = f"{title} {description}"
        matches = re.findall(r'#(\d+)', combined)
        return int(matches[0]) if matches else None


def cmd_monitor(
    config: str = "config/factory.yml",
    agents_dir: str = "config/agents",
    interval: int = 30,
) -> None:
    """Start the SoftwareTeamFabrik monitoring service."""
    monitor = FactoryMonitor(config_path=config, agents_dir=agents_dir, poll_interval=interval)
    monitor.run()


if __name__ == "__main__":
    import typer
    app = typer.Typer()
    app.command("monitor")(cmd_monitor)
    app()