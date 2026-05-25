from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Callable, Any

import gitlab
import yaml


@dataclass
class SprintInfo:
    number: int
    name: str
    start_date: date
    duration_days: int
    milestone_id: Optional[int]
    open_issues: int
    in_progress_issues: int
    closed_issues: int
    completion_pct: int


def retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Any:
    """
    Execute a function with exponential backoff retry logic.
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay between retries (default: 60.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        exceptions: Tuple of exception types to catch and retry
        
    Returns:
        The result of the function call
        
    Raises:
        The last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as exc:
            last_exception = exc
            
            if attempt < max_retries:
                # Calculate delay with exponential backoff
                delay = min(base_delay * (exponential_base ** attempt), max_delay)
                # Add jitter (±10% to prevent thundering herd)
                import random
                jitter = delay * 0.1 * (2 * random.random() - 1)
                actual_delay = max(0, delay + jitter)
                
                time.sleep(actual_delay)
            else:
                # All retries exhausted
                raise
    
    # Should not reach here, but just in case
    if last_exception:
        raise last_exception


class GitLabClient:
    def __init__(self, config_path: str = "config/factory.yml") -> None:
        # Load configuration from file
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Get GitLab configuration from environment variables or config file
        token = os.environ.get("GITLAB_TOKEN", "")
        if not token:
            raise RuntimeError(
                "GITLAB_TOKEN is not set. Add it to your .env file or export it in your shell."
            )

        url = os.environ.get("GITLAB_URL", cfg.get("gitlab", {}).get("url", ""))
        project = os.environ.get("GITLAB_PROJECT", cfg.get("gitlab", {}).get("project", ""))

        if not url:
            raise RuntimeError(
                "GITLAB_URL is not set. Add it to your .env file or export it in your shell."
            )

        if not project:
            raise RuntimeError(
                "GITLAB_PROJECT is not set. Add it to your .env file or export it in your shell."
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gl = gitlab.Gitlab(url, private_token=token)

        self._project_path = project
        self._sprint_cfg = cfg.get("sprint", {})
        self._url = url
        self._project = None

    # ------------------------------------------------------------------
    def connect(self) -> tuple[bool, str]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._gl.auth()
            return True, self._url
        except Exception as exc:
            return False, str(exc)

    @property
    def project(self):
        if self._project is None:
            self._project = self._gl.projects.get(self._project_path)
        return self._project

    # ------------------------------------------------------------------
    def sprint_info(self) -> SprintInfo:
        sprint_cfg = self._sprint_cfg
        number = sprint_cfg.get("current", 1)
        name = sprint_cfg.get("name", f"Sprint {number}")
        duration = sprint_cfg.get("duration_days", 14)
        start = date.fromisoformat(sprint_cfg.get("start_date", str(date.today())))
        milestone_prefix = sprint_cfg.get("milestone_prefix", "Sprint")

        milestone_id = None
        milestones = self.project.milestones.list(search=f"{milestone_prefix} {number}")
        if milestones:
            milestone_id = milestones[0].id

        label_in_progress = sprint_cfg.get("labels", {}).get("in_progress", "In Progress")
        label_done = sprint_cfg.get("labels", {}).get("done", "Done")

        issues = self.project.issues.list(state="opened", all=True)
        open_count = len(issues)
        in_progress = sum(1 for i in issues if label_in_progress in i.labels)
        closed = len(self.project.issues.list(state="closed", all=True))
        total = open_count + closed
        pct = int((closed / total) * 100) if total else 0

        return SprintInfo(
            number=number,
            name=name,
            start_date=start,
            duration_days=duration,
            milestone_id=milestone_id,
            open_issues=open_count,
            in_progress_issues=in_progress,
            closed_issues=closed,
            completion_pct=pct,
        )

    def days_remaining(self, sprint: SprintInfo) -> int:
        end = sprint.start_date + timedelta(days=sprint.duration_days)
        remaining = (end - date.today()).days
        return max(0, remaining)

    @property
    def hostname(self) -> str:
        return self._url.replace("https://", "").replace("http://", "")

    @property
    def project_path(self) -> str:
        return self._project_path

    # ------------------------------------------------------------------
    # Merge Request Operations with Retry
    # ------------------------------------------------------------------
    
    def create_merge_request(
        self,
        source_branch: str,
        target_branch: str,
        title: str,
        description: str = "",
        work_in_progress: bool = False,
        remove_source_branch: bool = False,
        max_retries: int = 3,
    ) -> Any:
        """
        Create a merge request with exponential backoff retry.
        
        Args:
            source_branch: Source branch name
            target_branch: Target branch name
            title: MR title
            description: MR description
            work_in_progress: Whether to create as WIP
            remove_source_branch: Whether to remove source branch on merge
            max_retries: Maximum retry attempts (default: 3)
            
        Returns:
            The created merge request object, or None if all retries fail
        """
        def _create_mr():
            return self.project.mergerequests.create({
                'source_branch': source_branch,
                'target_branch': target_branch,
                'title': title,
                'description': description,
                'work_in_progress': work_in_progress,
                'remove_source_branch': remove_source_branch,
            })
        
        try:
            return retry_with_backoff(
                _create_mr,
                max_retries=max_retries,
                exceptions=(gitlab.GitlabCreateError, gitlab.GitlabConnectionError, Exception),
            )
        except Exception as exc:
            # Log the failure but return None to maintain backward compatibility
            import logging
            logger = logging.getLogger(__name__)
            logger.error("Failed to create MR after %d retries: %s", max_retries, exc)
            return None

    # ------------------------------------------------------------------
    # Pipeline Failure Log Retrieval
    # ------------------------------------------------------------------

    def get_failed_pipeline_logs(self, mr_iid: int) -> str:
        """Retrieve logs from failed pipeline jobs for a merge request.

        Fetches the trace (console output) of every failed job in the most
        recent failed pipeline associated with the MR.  The output is
        formatted as a single Markdown string suitable for posting as a
        GitLab note.

        Args:
            mr_iid: The IID of the merge request.

        Returns:
            A Markdown-formatted string containing the failed job logs,
            or an empty string if no logs could be retrieved.
        """
        import logging
        logger = logging.getLogger(__name__)

        project = self.project
        try:
            mr = project.mergerequests.get(mr_iid)
        except Exception as exc:
            logger.error("Failed to fetch MR !%d for pipeline logs: %s", mr_iid, exc)
            return ""

        # Collect pipelines associated with this MR
        try:
            pipelines = mr.pipelines.list()
        except Exception:
            # Fallback: search by source branch
            try:
                pipelines = project.pipelines.list(
                    ref=mr.source_branch, order_by="id", sort="desc", per_page=5
                )
            except Exception as exc:
                logger.error("Failed to list pipelines for MR !%d: %s", mr_iid, exc)
                return ""

        if not pipelines:
            logger.info("No pipelines found for MR !%d", mr_iid)
            return ""

        # Find the most recent failed pipeline
        failed_pipeline = None
        for p in pipelines:
            status = getattr(p, "status", "")
            if status == "failed":
                failed_pipeline = p
                break

        if failed_pipeline is None:
            logger.info("No failed pipelines found for MR !%d", mr_iid)
            return ""

        pipeline_id = getattr(failed_pipeline, "id", "?")
        pipeline_sha = getattr(failed_pipeline, "sha", "?")[:8] if hasattr(failed_pipeline, "sha") and failed_pipeline.sha else "?"
        pipeline_ref = getattr(failed_pipeline, "ref", "?")

        sections = []
        sections.append(f"### Pipeline {pipeline_id} failed (commit `{pipeline_sha}`, branch `{pipeline_ref}`)\n")

        # Collect failed jobs in this pipeline
        try:
            jobs = failed_pipeline.jobs.list()
        except Exception:
            try:
                jobs = project.jobs.list(pipeline_id=pipeline_id)
            except Exception as exc:
                logger.error("Failed to list jobs for pipeline %s: %s", pipeline_id, exc)
                sections.append("*Could not retrieve job list.*")
                return "\n".join(sections)

        for job in jobs:
            if getattr(job, "status", "") != "failed":
                continue

            job_name = getattr(job, "name", "unknown")
            job_id = getattr(job, "id", "?")
            job_stage = getattr(job, "stage", "unknown")

            try:
                full_job = project.jobs.get(job_id)
                trace = full_job.trace()
                if trace is None:
                    trace = ""
                if isinstance(trace, bytes):
                    trace = trace.decode("utf-8", errors="replace")
            except Exception as exc:
                logger.warning("Could not retrieve trace for job %s (%s): %s", job_name, job_id, exc)
                sections.append(
                    f"#### Job `{job_name}` (stage: `{job_stage}`, id: {job_id}) — failed\n\n"
                    f"*Log retrieval failed: {exc}*\n"
                )
                continue

            # Trim very long traces to the last 150 lines
            lines = trace.splitlines()
            if len(lines) > 150:
                lines = lines[-150:]
                trimmed = True
            else:
                trimmed = False

            header = f"#### Job `{job_name}` (stage: `{job_stage}`, id: {job_id}) — failed\n"
            log_block = "```log\n" + "\n".join(lines) + "\n```"
            if trimmed:
                log_block += "\n*(showing last 150 lines)*\n"

            sections.append(header + log_block)

        if len(sections) == 1:
            # Only the pipeline header — no failed jobs found with logs
            sections.append("*No failed job logs could be retrieved.*")

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Wiki Operations
    # ------------------------------------------------------------------
    
    def create_or_update_wiki_page(self, title: str, content: str) -> Any:
        """
        Create or update a wiki page.
        
        Args:
            title: Wiki page title
            content: Wiki page content
            
        Returns:
            The created/updated wiki page object, or None if operation fails
        """
        def _wiki_operation():
            # Check if wiki page already exists
            try:
                wiki = self.project.wikis.get(title)
                # Update existing page
                wiki.content = content
                wiki.save()
                return wiki
            except gitlab.GitlabGetError:
                # Create new page
                return self.project.wikis.create({
                    'title': title,
                    'content': content,
                })
        
        try:
            return retry_with_backoff(
                _wiki_operation,
                max_retries=3,
                exceptions=(gitlab.GitlabCreateError, gitlab.GitlabConnectionError, Exception),
            )
        except Exception as exc:
            # Log the failure but return None to maintain backward compatibility
            import logging
            logger = logging.getLogger(__name__)
            logger.error("Failed to create/update wiki page after retries: %s", exc)
            return None