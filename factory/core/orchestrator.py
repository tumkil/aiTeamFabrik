"""
Orchestrator Module

This module coordinates the autonomous software development factory.
It manages agents, tasks, and the overall workflow.
"""

from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Any

import logging

from factory.adapters.gitlab_client import GitLabClient, retry_with_backoff
from factory.adapters.llm_router import LlmRouter, LlmResponse
from factory.core.competence import AgentProfile, CompetenceManager
from factory.core.execution_engine import CodeExecutionEngine, ExecutionResult
from factory.core.scrum import ScrumEngine
from factory.core.architect_feedback import (
    ArchitectFeedbackEnforcer,
    extract_architect_plan_from_notes,
    create_architect_context,
)

# Import MCP server classes
from factory.core.developer_mcp_server import DeveloperMCPServer
from factory.core.reviewer_mcp_server import ReviewerMCPServer
from factory.core.architect_mcp_server import ArchitectMCPServer
from factory.core.planner_mcp_server import PlannerMCPServer
from factory.core.po_mcp_server import POMCPServer

logger = logging.getLogger(__name__)

_PLAN_SYSTEM = """\
You are a {role} agent in the SoftwareTeamFabrik autonomous development factory.
Your job is to analyse the given GitLab issue and produce a concrete, actionable response.

Respond with:
1. A brief summary of what the issue requires.
2. A step-by-step implementation plan (numbered list).
3. Any risks or open questions the team should address.\n"""

_DEFAULT_TASK_RETENTION_TTL = timedelta(hours=24)
_DEFAULT_STALE_TASK_TTL = timedelta(hours=2)


class TaskStatus(Enum):
    """Status of an agent task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RunResult:
    issue_iid: int
    agent: str
    model: str
    mode: str           # plan | execute
    response: str
    mr_url: Optional[str]
    comment_url: Optional[str]
    commit_sha: str = ""
    files_changed: list[str] = field(default_factory=list)
    iterations: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    needs_continuation: bool = False
    continuation_context: str = ""


@dataclass
class SpawnResult:
    """Result of spawning an async task."""
    run_id: str
    issue_iid: int
    agent_name: str
    status: str
    message: str


@dataclass
class PollResult:
    """Result of polling a task's status."""
    run_id: str
    status: str
    result: Optional[RunResult] = None
    error: Optional[str] = None
    progress: str = ""


@dataclass
class TaskRecord:
    """Internal record of a running task."""
    run_id: str
    issue_iid: int
    agent: AgentProfile
    status: TaskStatus
    result: Optional[RunResult] = None
    error: Optional[str] = None
    thread: Optional[threading.Thread] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class Orchestrator:
    def __init__(
        self,
        gl: GitLabClient,
        competence: CompetenceManager,
        scrum: ScrumEngine,
        router: Optional[LlmRouter] = None,
    ) -> None:
        self._gl = gl
        self._competence = competence
        self._scrum = scrum
        self._router = router or LlmRouter()

        # Task registry for spawn/poll functionality
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = threading.Lock()

        # MCP server registry
        self._mcp_servers: dict[str, Any] = {}

        # TTLs for task reaping
        self._task_retention_ttl: timedelta = _DEFAULT_TASK_RETENTION_TTL
        self._stale_task_ttl: timedelta = _DEFAULT_STALE_TASK_TTL

        # Reaper thread placeholder (not started automatically)
        self._reaper_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # TASK RETENTION CONFIGURATION
    # ------------------------------------------------------------------

    def set_task_retention_ttl(self, ttl: timedelta) -> None:
        """Set how long completed/failed tasks are kept before reaping."""
        self._task_retention_ttl = ttl

    def set_stale_task_ttl(self, ttl: timedelta) -> None:
        """Set how long a RUNNING task lives before being considered stale."""
        self._stale_task_ttl = ttl

    # ------------------------------------------------------------------
    # TASK REAPING
    # ------------------------------------------------------------------

    def reap_completed_tasks_now(self) -> int:
        """Remove stale tasks from the registry immediately.

        Returns the number of tasks reaped.
        """
        now = datetime.now()
        to_reap: list[str] = []

        with self._lock:
            for run_id, task in self._tasks.items():
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    if task.completed_at is not None:
                        if now - task.completed_at > self._task_retention_ttl:
                            to_reap.append(run_id)
                elif task.status == TaskStatus.RUNNING:
                    if now - task.created_at > self._stale_task_ttl:
                        to_reap.append(run_id)

            for run_id in to_reap:
                del self._tasks[run_id]

        return len(to_reap)

    def stop(self) -> None:
        """Stop the orchestrator and clear all task state."""
        self._reaper_thread = None
        with self._lock:
            self._tasks.clear()

    # ------------------------------------------------------------------
    # MCP SERVER MANAGEMENT
    # ------------------------------------------------------------------

    def _create_mcp_server_for_agent(self, agent: AgentProfile) -> Any:
        """Create an MCP server instance for the given agent."""
        agent_type = agent.name.lower()

        if "developer" in agent_type:
            return DeveloperMCPServer(agent)
        elif "reviewer" in agent_type:
            return ReviewerMCPServer(agent)
        elif "architect" in agent_type:
            return ArchitectMCPServer(agent)
        elif "planner" in agent_type:
            return PlannerMCPServer(agent)
        elif "po" in agent_type or "product" in agent_type:
            return POMCPServer(agent)
        else:
            logger.warning(f"No specific MCP server class for agent type: {agent_type}")
            return None

    def _get_or_create_mcp_server(self, agent: AgentProfile) -> Any:
        """Get or create an MCP server for the given agent."""
        with self._lock:
            if agent.name not in self._mcp_servers:
                server = self._create_mcp_server_for_agent(agent)
                if server:
                    self._mcp_servers[agent.name] = server
                    server.start()
            return self._mcp_servers.get(agent.name)

    def stop_all_mcp_servers(self) -> None:
        """Stop all running MCP servers."""
        with self._lock:
            for server_name, server in self._mcp_servers.items():
                if server:
                    server.stop()
            self._mcp_servers.clear()

    # ------------------------------------------------------------------
    # SPAWN / POLL API
    # ------------------------------------------------------------------

    def spawn(self, issue_iid: int) -> SpawnResult:
        """Spawn an agent task asynchronously. Returns run_id for polling."""
        project = self._gl.project
        issue = project.issues.get(issue_iid)
        agent = self._resolve_agent(issue)
        sprint = self._scrum.current

        run_id = str(uuid.uuid4())

        task = TaskRecord(
            run_id=run_id,
            issue_iid=issue_iid,
            agent=agent,
            status=TaskStatus.PENDING,
        )

        with self._lock:
            self._tasks[run_id] = task

        self._set_label(issue, sprint.label_in_progress)

        self._get_or_create_mcp_server(agent)

        thread = threading.Thread(
            target=self._run_task,
            args=(run_id, issue_iid, agent),
            daemon=True,
        )
        task.thread = thread
        task.status = TaskStatus.RUNNING
        thread.start()

        return SpawnResult(
            run_id=run_id,
            issue_iid=issue_iid,
            agent_name=agent.name,
            status="spawned",
            message=f"Agent {agent.name} spawned for issue #{issue_iid}",
        )

    def poll(self, run_id: str) -> PollResult:
        """Poll the status of a spawned task."""
        with self._lock:
            task = self._tasks.get(run_id)
            if not task:
                return PollResult(
                    run_id=run_id,
                    status="not_found",
                    error="Task not found",
                )

            if task.status == TaskStatus.COMPLETED:
                return PollResult(
                    run_id=run_id,
                    status="completed",
                    result=task.result,
                )
            elif task.status == TaskStatus.FAILED:
                return PollResult(
                    run_id=run_id,
                    status="failed",
                    error=task.error,
                )
            else:
                return PollResult(
                    run_id=run_id,
                    status=task.status.value,
                    progress=f"Agent {task.agent.name} is {task.status.value}",
                )

    def list_tasks(self) -> list[TaskRecord]:
        """List all running tasks."""
        with self._lock:
            return list(self._tasks.values())

    def retire(self, run_id: str) -> bool:
        """Retire a task from the registry."""
        with self._lock:
            if run_id in self._tasks:
                del self._tasks[run_id]
                return True
            return False

    def _run_task(self, run_id: str, issue_iid: int, agent: AgentProfile) -> None:
        """Internal task execution thread."""
        try:
            result = self.run(issue_iid=issue_iid)
            with self._lock:
                if run_id in self._tasks:
                    task = self._tasks[run_id]
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = datetime.now()
                else:
                    logger.warning(f"Task {run_id} was retired before completion")
        except Exception as exc:
            logger.error("Task %s failed: %s", run_id, exc)
            with self._lock:
                if run_id in self._tasks:
                    task = self._tasks[run_id]
                    task.status = TaskStatus.FAILED
                    task.error = str(exc)
                    task.completed_at = datetime.now()
                else:
                    logger.warning(f"Task {run_id} was retired before failure could be recorded: {exc}")

    def run(self, issue_iid: int, progress=None) -> RunResult:
        """Run an agent on a GitLab issue and return the result."""
        project = self._gl.project
        issue = project.issues.get(issue_iid)

        if self._is_openerge_request(issue):
            agent = self._resolve_openerge_agent()
        else:
            agent = self._resolve_agent(issue)

        architect_plan_text = self._check_and_execute_architect_review(issue)

        result = RunResult(
            issue_iid=issue_iid,
            agent=agent.name,
            model=agent.model,
            mode=agent.execution_mode,
            response="",
            mr_url=None,
            comment_url=None,
        )

        if agent.execution_mode == "execute":
            if architect_plan_text is None:
                try:
                    notes = issue.notes.list(all=True)
                    architect_plan_text = extract_architect_plan_from_notes(notes) or ""
                except Exception:
                    pass

            issue_description = issue.description or ""
            if architect_plan_text:
                enforcer = ArchitectFeedbackEnforcer(architect_plan_text)
                issue_description = create_architect_context(
                    architect_plan_text,
                    issue_description
                )
                issue_description += enforcer.get_adherence_prompt()

            engine = CodeExecutionEngine(
                agent=agent,
                repo_path=Path.cwd(),
                issue_title=issue.title,
                issue_description=issue_description,
                gitlab_token=os.environ.get("GITLAB_TOKEN", ""),
                gitlab_project_id=project.id,
                gitlab_url=self._gl._url,
                branch_name=f"issue-{issue_iid}",
                progress=progress,
            )

            engine_result = engine.run(router=self._router)

            result.response = engine_result.response
            result.files_changed = engine_result.files_changed
            result.iterations = engine_result.iterations
            result.commit_sha = engine_result.commit_sha
            result.needs_continuation = engine_result.needs_continuation
            result.continuation_context = engine_result.continuation_context

            if architect_plan_text:
                adherence_context = self._check_architect_adherence(
                    architect_plan_text,
                    result.files_changed,
                )
                if adherence_context:
                    result.response += "\n\n" + adherence_context

            comment_body = self._build_execution_comment(agent, result)
            note = issue.notes.create({"body": comment_body})
            result.comment_url = note.web_url if hasattr(note, 'web_url') else ""

            if result.files_changed:
                mr = self._open_wip_mr(issue, agent, result)
                if mr:
                    result.mr_url = mr.web_url if hasattr(mr, 'web_url') else ""

            if result.needs_continuation:
                self._post_continuation_comment(issue, result)
        else:
            system = _PLAN_SYSTEM.format(role=agent.display_name)
            prompt = f"Issue #{issue_iid}: {issue.title}\n\n{issue.description}"

            llm_response = self._router.complete(
                agent=agent,
                system=system,
                prompt=prompt,
            )

            result.response = llm_response.content

            comment_body = f"**{agent.display_name} ({agent.model}) Analysis**\n\n{result.response}"
            note = issue.notes.create({"body": comment_body})
            result.comment_url = note.web_url if hasattr(note, 'web_url') else ""

        if agent.name == "reviewer":
            security_reviewer = self._competence.get("security_reviewer")
            if security_reviewer:
                self._run_security_review(issue, security_reviewer)

        return result

    def _check_and_execute_architect_review(self, issue) -> Optional[str]:
        try:
            notes = issue.notes.list(all=True)
            architect_plan_text = extract_architect_plan_from_notes(notes)
            if architect_plan_text:
                return architect_plan_text
        except Exception as exc:
            logger.warning(f"Failed to check for existing architect review: {exc}")

        logger.info(f"No architect review found for issue #{issue.iid}, executing one...")

        architect_agent = self._resolve_architect_agent()
        if not architect_agent:
            logger.warning("No architect agent available")
            return None

        try:
            return self._execute_architect_review(issue, architect_agent)
        except Exception as exc:
            logger.error(f"Failed to execute architect review: {exc}")
            return None

    def _execute_architect_review(self, issue, architect_agent: AgentProfile) -> str:
        system = """\
You are an Architect agent in the SoftwareTeamFabrik autonomous development factory.
Your job is to analyse the given GitLab issue and provide architectural guidance.

Respond with:
1. A brief summary of what the issue requires.
2. A detailed implementation plan (numbered list).
3. Key architectural decisions and requirements.
4. Any risks or open questions the team should address.
5. File structure and naming conventions to follow.
6. Class and function signatures that should be implemented.
\n"""
        prompt = f"Issue #{issue.iid}: {issue.title}\n\n{issue.description}"

        llm_response = self._router.complete(
            agent=architect_agent,
            system=system,
            prompt=prompt,
        )

        architect_plan_text = f"## Architect Analysis\n\n{llm_response.content}"
        issue.notes.create({"body": architect_plan_text})
        return architect_plan_text

    def _resolve_architect_agent(self) -> Optional[AgentProfile]:
        for agent in self._competence.agents:
            if "architect" in agent.name.lower():
                return agent
        return None

    def initiate_po_discussion(self, issue_iid: int, progress=None) -> RunResult:
        """Initiate a Product Owner (PO) agent discussion flow for an open issue."""
        project = self._gl.project
        issue = project.issues.get(issue_iid)

        po_agent = self._resolve_po_agent()

        result = RunResult(
            issue_iid=issue_iid,
            agent=po_agent.name,
            model=po_agent.model,
            mode="discuss",
            response="",
            mr_url=None,
            comment_url=None,
        )

        system = """\
You are a Product Owner (PO) agent in the SoftwareTeamFabrik autonomous development factory.
Your job is to facilitate a structured discussion about the given GitLab issue with stakeholders.

Respond with:
1. A brief summary of the issue and its requirements.
2. Key discussion points that need stakeholder input (as bullet points).
3. Any risks or dependencies that should be considered.
4. Recommendations for next steps.
5. Specific questions for stakeholders to answer.
\n"""
        prompt = f"Issue #{issue_iid}: {issue.title}\n\n{issue.description}"

        llm_response = self._router.complete(
            agent=po_agent,
            system=system,
            prompt=prompt,
        )

        discussion_points = llm_response.content

        final_summary = self._conduct_interactive_discussion(
            issue_iid=issue_iid,
            issue_title=issue.title,
            discussion_points=discussion_points,
            po_agent=po_agent,
            progress=progress
        )

        result.response = final_summary

        comment_body = f"**{po_agent.display_name} ({po_agent.model}) Discussion Summary**\n\n{result.response}"
        note = issue.notes.create({"body": comment_body})
        result.comment_url = note.web_url if hasattr(note, 'web_url') else ""

        return result

    def _conduct_interactive_discussion(self, issue_iid: int, issue_title: str,
                                        discussion_points: str, po_agent: AgentProfile,
                                        progress=None) -> str:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()

        console.print()
        console.rule(f"[bold cyan]PO Discussion — Issue #{issue_iid}: {issue_title}[/bold cyan]")
        console.print()

        console.print(Panel(discussion_points,
                            title="[bold]PO Discussion Points[/bold]",
                            border_style="blue"))
        console.print()

        console.print("[yellow]Stakeholders: Please provide your input on the discussion points above.[/yellow]")
        console.print("[yellow]Enter your comments (press Enter twice to finish):[/yellow]")
        console.print()

        stakeholder_input = ""
        while True:
            try:
                line = input("> ")
                if line == "":
                    break
                stakeholder_input += line + "\n"
            except EOFError:
                break

        if stakeholder_input.strip():
            final_summary = "## PO Discussion Summary\n\n"
            final_summary += f"### Discussion Points (from {po_agent.display_name})\n\n"
            final_summary += discussion_points + "\n\n"
            final_summary += "### Stakeholder Input\n\n"
            final_summary += stakeholder_input.strip() + "\n"

            conclusion_prompt = f"""\
Based on the following PO discussion points and stakeholder input,
provide a concise conclusion and next steps:

PO Discussion Points:
{discussion_points}

Stakeholder Input:
{stakeholder_input}

Conclusion (brief summary and next steps):
"""
            conclusion_response = self._router.complete(
                agent=po_agent,
                system="You are a Product Owner agent summarizing a discussion.",
                prompt=conclusion_prompt,
            )

            final_summary += f"### Conclusion\n\n{conclusion_response.content}"
        else:
            final_summary = "## PO Discussion Summary\n\n"
            final_summary += discussion_points

        return final_summary

    def _check_architect_adherence(
        self,
        architect_plan_text: str,
        files_changed: list[str],
    ) -> str:
        enforcer = ArchitectFeedbackEnforcer(architect_plan_text)
        report = enforcer.record_implementation(files=files_changed)
        summary = report.to_summary()
        alert = enforcer.get_deviation_alert()
        if alert:
            summary = alert + "\n\n" + summary
        return summary

    def _is_openerge_request(self, issue) -> bool:
        if any(label.lower() == "openerge" for label in issue.labels):
            return True
        if "openerge" in issue.title.lower():
            return True
        if issue.description and "openerge" in issue.description.lower():
            return True
        return False

    def _resolve_openerge_agent(self) -> AgentProfile:
        for agent in self._competence.agents:
            if any(label.lower() == "openerge" for label in agent.task_labels):
                return agent
        return self._competence.get("developer") or self._competence.agents[0]

    def _resolve_po_agent(self) -> AgentProfile:
        for agent in self._competence.agents:
            if "po" in agent.name.lower() or "product" in agent.name.lower():
                return agent
        return self._competence.agents[0]

    def _run_security_review(self, issue, security_reviewer: AgentProfile) -> None:
        system = _PLAN_SYSTEM.format(role=security_reviewer.display_name)
        prompt = f"Issue #{issue.iid}: {issue.title}\n\n{issue.description}"

        llm_response = self._router.complete(
            agent=security_reviewer,
            system=system,
            prompt=prompt,
        )

        comment_body = f"**{security_reviewer.display_name} ({security_reviewer.model}) Security Review**\n\n{llm_response.content}"
        issue.notes.create({"body": comment_body})

    def _build_execution_comment(self, agent: AgentProfile, result: RunResult) -> str:
        comment = f"**{agent.display_name} ({agent.model}) Execution Results**\n\n"
        comment += f"- Model: `{agent.model}`\n"
        comment += f"- Iterations: {result.iterations}\n"
        comment += f"- Files changed: {len(result.files_changed)}\n"

        if result.files_changed:
            comment += "\n**Changed files:**\n"
            for file in result.files_changed:
                comment += f"- `{file}`\n"

        if result.commit_sha:
            comment += f"\n- Commit: `{result.commit_sha[:8]}`\n"

        comment += f"\n**Response:**\n\n{result.response}"
        return comment

    def _open_wip_mr(self, issue, agent: AgentProfile, result: RunResult) -> Optional[Any]:
        project = self._gl.project
        branch_name = f"issue-{issue.iid}"

        try:
            project.branches.create({
                'branch': branch_name,
                'ref': project.default_branch
            })
        except Exception:
            try:
                project.branches.get(branch_name)
            except Exception:
                return None

        mr_title = f"WIP: {issue.title}"
        mr_description = f"Implementation for issue #{issue.iid}\n\n"
        mr_description += f"Agent: {agent.display_name}\n"
        mr_description += f"Model: {agent.model}\n"
        mr_description += f"Iterations: {result.iterations}\n"

        if result.files_changed:
            mr_description += "\n**Files changed:**\n"
            for file in result.files_changed:
                mr_description += f"- {file}\n"

        try:
            mr = retry_with_backoff(
                lambda: project.mergerequests.create({
                    'source_branch': branch_name,
                    'target_branch': project.default_branch,
                    'title': mr_title,
                    'description': mr_description,
                    'work_in_progress': True,
                }),
                max_retries=3,
            )
            return mr
        except Exception as exc:
            logger.error("Failed to create WIP MR after retries: %s", exc)
            return None

    def _post_continuation_comment(self, issue, result: RunResult) -> Optional[Any]:
        try:
            comment_text = f"""⏳ Iteration limit reached

The agent hit the iteration limit ({result.iterations} iterations) before completing the implementation.

**To continue:**
```bash
factory continue-mr --mr {result.mr_url.split('/')[-1] if result.mr_url else 'MR_IID'}
```

**Continuation context:**
{result.continuation_context}
"""
            note = issue.notes.create({"body": comment_text})
            return note
        except Exception as exc:
            logger.error("Failed to post continuation comment: %s", exc)
            return None

    def _resolve_agent(self, issue) -> AgentProfile:
        for agent in self._competence.agents:
            for label in issue.labels:
                if label in agent.task_labels:
                    return agent
        return self._competence.get("developer") or self._competence.agents[0]

    def _set_label(self, issue, label: str) -> None:
        try:
            current_labels = list(issue.labels) if issue.labels else []
            conflicting_labels = ["In Review", "Done"]
            current_labels = [lbl for lbl in current_labels if lbl not in conflicting_labels]
            if label not in current_labels:
                current_labels.append(label)
            issue.labels = current_labels
            issue.save()
        except Exception as exc:
            logger.warning("Failed to set label %s: %s", label, exc)
