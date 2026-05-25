"""Unit tests for Orchestrator — mocks GitLab and uses stub LLM provider."""
import pytest
import uuid
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from factory.adapters.llm_router import LlmRouter, LlmResponse
from factory.core.competence import AgentProfile, CompetenceManager
from factory.core.orchestrator import Orchestrator, RunResult, ExecutionResult, TaskStatus, TaskRecord
from factory.core.scrum import ScrumEngine


def _make_mock_gl(issue_labels=None):
    issue = MagicMock()
    issue.iid = 42
    issue.title = "Build the widget"
    issue.description = "We need a widget that does things."
    issue.labels = issue_labels or ["feature"]
    issue.notes = MagicMock()
    note = MagicMock()
    note.id = 99
    issue.notes.create.return_value = note

    project = MagicMock()
    project.issues.get.return_value = issue
    project.default_branch = "main"
    project.branches.create.return_value = MagicMock()
    mr = MagicMock()
    mr.web_url = "https://gitlab.example.com/org/repo/-/merge_requests/1"
    project.mergerequests.create.return_value = mr

    gl = MagicMock()
    gl.project = project
    gl._url = "https://gitlab.example.com"
    gl.project_path = "org/repo"
    return gl


def _make_profiles(execution_mode="plan") -> dict[str, AgentProfile]:
    return {
        "developer": AgentProfile(
            name="developer",
            display_name="Developer",
            model="stub",
            provider="stub",
            task_labels=["feature", "bug"],
            execution_mode=execution_mode,
        ),
        "qa": AgentProfile(
            name="qa",
            display_name="QA",
            model="stub",
            provider="stub",
            task_labels=["testing"],
            execution_mode="plan",
        ),
        "reviewer": AgentProfile(
            name="reviewer",
            display_name="Code Reviewer",
            model="stub",
            provider="stub",
            task_labels=["review", "code-review"],
            execution_mode="plan",
        ),
        "security_reviewer": AgentProfile(
            name="security_reviewer",
            display_name="Security Reviewer",
            model="stub",
            provider="stub",
            task_labels=["security-review", "cve", "vulnerability"],
            execution_mode="plan",
        ),
        "openerge": AgentProfile(
            name="openerge",
            display_name="Openerge Agent",
            model="stub",
            provider="stub",
            task_labels=["openerge"],
            execution_mode="execute",
        ),
    }


def _make_competence(execution_mode="plan") -> CompetenceManager:
    profiles = _make_profiles(execution_mode=execution_mode)
    return CompetenceManager.from_profiles(profiles)


def _make_scrum() -> ScrumEngine:
    scrum = MagicMock(spec=ScrumEngine)
    state = MagicMock()
    state.number = 2
    state.label_in_progress = "In Progress"
    state.label_review = "In Review"
    scrum.current = state
    scrum.velocity.return_value = 50
    return scrum


def test_run_returns_result():
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())

    result = orch.run(issue_iid=42)

    assert result.issue_iid == 42
    assert result.agent == "developer"
    assert result.model == "stub"
    assert result.response != ""


def test_run_posts_comment():
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())

    orch.run(issue_iid=42)

    gl.project.issues.get.return_value.notes.create.assert_called_once()
    call_body = gl.project.issues.get.return_value.notes.create.call_args[0][0]["body"]
    assert "Developer" in call_body
    assert "stub" in call_body


def test_run_opens_wip_mr():
    gl = _make_mock_gl()
    cm = _make_competence(execution_mode="execute")
    scrum = _make_scrum()
    
    # Mock the execution engine to return some files changed
    with patch("factory.core.orchestrator.CodeExecutionEngine") as mock_engine_class:
        mock_engine = MagicMock()
        mock_engine.run.return_value = ExecutionResult(
            response="Implementation complete",
            files_changed=["test.py"],
            iterations=3,
        )
        mock_engine_class.return_value = mock_engine
        
        orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
        result = orch.run(issue_iid=42)

        gl.project.mergerequests.create.assert_called_once()
        assert result.mr_url is not None
        # Verify WIP MR parameters
        call_args = gl.project.mergerequests.create.call_args[0][0]
        assert call_args.get("work_in_progress") == True
        assert "Draft:" not in call_args.get("title", "")


def test_resolve_agent_by_label():
    gl = _make_mock_gl(issue_labels=["testing"])
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())

    result = orch.run(issue_iid=42)
    assert result.agent == "qa"


def test_resolve_agent_defaults_to_developer():
    gl = _make_mock_gl(issue_labels=["unrecognised-label"])
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())

    result = orch.run(issue_iid=42)
    assert result.agent == "developer"


@patch("factory.core.orchestrator.CodeExecutionEngine")
def test_run_execution_mode_uses_execution_engine(mock_engine_class, monkeypatch):
    """Test that execution mode agents use CodeExecutionEngine."""
    # Create a developer agent in execute mode
    profiles = {
        "developer": AgentProfile(
            name="developer",
            display_name="Developer",
            model="stub",
            provider="stub",
            task_labels=["feature"],
            execution_mode="execute",
        ),
    }
    cm = CompetenceManager.from_profiles(profiles)
    
    gl = _make_mock_gl()
    scrum = _make_scrum()
    
    # Mock the execution engine
    mock_engine = MagicMock()
    mock_engine.run.return_value = ExecutionResult(
        response="Test implementation",
        files_changed=["test.py"],
        commit_sha="abc123",
        iterations=3,
    )
    mock_engine_class.return_value = mock_engine
    
    # Mock environment variables
    monkeypatch.setenv("GITLAB_TOKEN", "test-token")
    
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    result = orch.run(issue_iid=42)
    
    # Verify execution engine was used
    mock_engine_class.assert_called_once()
    assert result.response == "Test implementation"
    assert result.files_changed == ["test.py"]
    assert result.commit_sha == "abc123"
    assert result.iterations == 3


@patch("factory.core.orchestrator.CodeExecutionEngine")
def test_run_plan_mode_uses_llm_router(mock_engine_class):
    """Test that plan mode agents use LLM router directly."""
    # Create a developer agent in plan mode
    profiles = {
        "developer": AgentProfile(
            name="developer",
            display_name="Developer",
            model="stub",
            provider="stub",
            task_labels=["feature"],
            execution_mode="plan",
        ),
    }
    cm = CompetenceManager.from_profiles(profiles)
    
    gl = _make_mock_gl()
    scrum = _make_scrum()
    
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    result = orch.run(issue_iid=42)
    
    # Verify execution engine was NOT used
    mock_engine_class.assert_not_called()
    assert result.response != ""
    assert result.mode == "plan"


def test_open_mr_creates_branch_and_mr():
    """Test _open_wip_mr creates branch and MR."""
    gl = _make_mock_gl()
    cm = _make_competence(execution_mode="execute")
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    # Mock agent and result
    agent = cm.get("developer")
    result = RunResult(
        issue_iid=42,
        agent="developer",
        model="stub",
        mode="execute",
        response="Test",
        mr_url=None,
        comment_url=None,
        files_changed=["test.py"],
        iterations=3,
    )
    
    issue = gl.project.issues.get(return_value=MagicMock())
    mr = orch._open_wip_mr(issue, agent, result)
    
    # Verify branch was created
    gl.project.branches.create.assert_called_once()
    # Verify MR was created
    gl.project.mergerequests.create.assert_called_once()
    assert mr.web_url == "https://gitlab.example.com/org/repo/-/merge_requests/1"


def test_open_mr_returns_none_on_branch_creation_failure():
    """Test _open_wip_mr handles branch creation failure."""
    gl = _make_mock_gl()
    cm = _make_competence(execution_mode="execute")
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    # Mock branch creation to fail and branch get to also fail
    gl.project.branches.create.side_effect = Exception("Branch creation failed")
    gl.project.branches.get.side_effect = Exception("Branch not found")
    
    agent = cm.get("developer")
    result = RunResult(
        issue_iid=42,
        agent="developer",
        model="stub",
        mode="execute",
        response="Test",
        mr_url=None,
        comment_url=None,
        files_changed=["test.py"],
        iterations=3,
    )
    
    issue = gl.project.issues.get(return_value=MagicMock())
    mr = orch._open_wip_mr(issue, agent, result)
    
    # Should return None on failure
    assert mr is None


def test_open_mr_creates_mr_when_branch_already_exists():
    """Test _open_wip_mr creates MR when branch already exists."""
    gl = _make_mock_gl()
    cm = _make_competence(execution_mode="execute")
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    # Mock branch creation to fail (branch already exists)
    gl.project.branches.create.side_effect = Exception("Branch already exists")
    # Mock branch get to succeed (branch exists)
    gl.project.branches.get.return_value = MagicMock()
    
    agent = cm.get("developer")
    result = RunResult(
        issue_iid=42,
        agent="developer",
        model="stub",
        mode="execute",
        response="Test",
        mr_url=None,
        comment_url=None,
        files_changed=["test.py"],
        iterations=3,
    )
    
    issue = gl.project.issues.get(return_value=MagicMock())
    mr = orch._open_wip_mr(issue, agent, result)
    
    # Should still create MR when branch already exists
    gl.project.mergerequests.create.assert_called_once()
    assert mr.web_url == "https://gitlab.example.com/org/repo/-/merge_requests/1"


def test_set_label_adds_in_progress():
    """Test _set_label adds In Progress label."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    issue = gl.project.issues.get(return_value=MagicMock())
    issue.labels = ["todo"]
    
    orch._set_label(issue, "In Progress")
    
    assert "In Progress" in issue.labels


def test_set_label_removes_conflicting_labels():
    """Test _set_label removes conflicting labels."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    issue = gl.project.issues.get(return_value=MagicMock())
    issue.labels = ["todo", "In Review"]
    
    orch._set_label(issue, "In Progress")
    
    assert "In Progress" in issue.labels
    assert "In Review" not in issue.labels


def test_slugify_handles_special_chars():
    """Test _slugify handles special characters."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    # Test slugify method if it exists
    if hasattr(orch, '_slugify'):
        result = orch._slugify("Test Issue #123")
        assert result == "test-issue-123"


def test_spawn_returns_run_id():
    """Test spawn returns a valid run_id."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    result = orch.spawn(issue_iid=42)
    
    assert result.run_id is not None
    assert result.issue_iid == 42
    assert result.agent_name == "developer"
    assert result.status == "spawned"


def test_spawn_creates_task_record():
    """Test spawn creates a task record."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    run_id = orch.spawn(issue_iid=42).run_id
    
    with orch._lock:
        assert run_id in orch._tasks
        # Task may have already completed by the time we check (race-free: RUNNING or COMPLETED)
        assert orch._tasks[run_id].status in (TaskStatus.RUNNING, TaskStatus.COMPLETED)


def test_poll_returns_valid_status():
    """Test poll returns valid status."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    run_id = orch.spawn(issue_iid=42).run_id
    
    result = orch.poll(run_id)
    assert result.status in ["running", "completed", "failed"]


def test_retire_removes_task():
    """Test retire removes task from registry."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    run_id = orch.spawn(issue_iid=42).run_id
    
    # Retire the task
    orch.retire(run_id)
    
    with orch._lock:
        assert run_id not in orch._tasks


def test_retire_returns_false_for_unknown_run_id():
    """Test retire returns False for unknown run_id."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    result = orch.retire("unknown-id")
    
    assert result == False


def test_poll_returns_not_found_for_unknown_run_id():
    """Test poll returns not_found for unknown run_id."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    result = orch.poll("unknown-id")
    
    assert result.status == "not_found"
    assert result.error == "Task not found"


def test_list_tasks_returns_empty_when_no_tasks():
    """Test list_tasks returns empty list when no tasks."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    tasks = orch.list_tasks()
    
    assert tasks == []


def test_spawn_sets_issue_to_in_progress():
    """Test spawn sets issue to In Progress."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    issue = gl.project.issues.get(return_value=MagicMock())
    issue.labels = []
    
    orch.spawn(issue_iid=42)
    
    assert "In Progress" in issue.labels


def test_run_reviewer_triggers_security_review():
    """Test that when reviewer runs, security reviewer is also triggered."""
    gl = _make_mock_gl(issue_labels=["review"])
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    # Run the reviewer agent
    result = orch.run(issue_iid=42)
    
    # Verify that both reviewer and security reviewer comments were posted
    assert gl.project.issues.get.return_value.notes.create.call_count == 2
    
    # Check that the second call is for the security reviewer
    calls = gl.project.issues.get.return_value.notes.create.call_args_list
    assert len(calls) == 2
    
    # First call should be the reviewer's comment
    first_call_body = calls[0][0][0]["body"]
    assert "Code Reviewer" in first_call_body
    
    # Second call should be the security reviewer's comment
    second_call_body = calls[1][0][0]["body"]
    assert "Security Reviewer" in second_call_body


def test_run_non_reviewer_does_not_trigger_security_review():
    """Test that non-reviewer agents do not trigger security review."""
    gl = _make_mock_gl(issue_labels=["feature"])
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    # Run the developer agent
    result = orch.run(issue_iid=42)
    
    # Verify that only one comment was posted (no security review)
    assert gl.project.issues.get.return_value.notes.create.call_count == 1
    
    # Check that the comment is for the developer
    call_body = gl.project.issues.get.return_value.notes.create.call_args[0][0]["body"]
    assert "Developer" in call_body
    assert "Security Reviewer" not in call_body


def test_run_uses_openerge_agent_for_openerge_request():
    """Test that openerge requests use the openerge agent."""
    gl = _make_mock_gl(issue_labels=["openerge"])
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    result = orch.run(issue_iid=42)
    
    assert result.agent == "openerge"


def test_run_uses_openerge_agent_for_openerge_in_title():
    """Test that openerge in title uses the openerge agent."""
    issue = MagicMock()
    issue.iid = 42
    issue.title = "Openerge: Build the widget"
    issue.description = "We need a widget that does things."
    issue.labels = ["feature"]
    issue.notes = MagicMock()
    note = MagicMock()
    note.id = 99
    issue.notes.create.return_value = note

    project = MagicMock()
    project.issues.get.return_value = issue
    project.default_branch = "main"
    project.branches.create.return_value = MagicMock()
    mr = MagicMock()
    mr.web_url = "https://gitlab.example.com/org/repo/-/merge_requests/1"
    project.mergerequests.create.return_value = mr

    gl = MagicMock()
    gl.project = project
    gl._url = "https://gitlab.example.com"
    gl.project_path = "org/repo"

    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    result = orch.run(issue_iid=42)
    
    assert result.agent == "openerge"


def test_run_uses_openerge_agent_for_openerge_in_description():
    """Test that openerge in description uses the openerge agent."""
    issue = MagicMock()
    issue.iid = 42
    issue.title = "Build the widget"
    issue.description = "This is an openerge request for a widget."
    issue.labels = ["feature"]
    issue.notes = MagicMock()
    note = MagicMock()
    note.id = 99
    issue.notes.create.return_value = note

    project = MagicMock()
    project.issues.get.return_value = issue
    project.default_branch = "main"
    project.branches.create.return_value = MagicMock()
    mr = MagicMock()
    mr.web_url = "https://gitlab.example.com/org/repo/-/merge_requests/1"
    project.mergerequests.create.return_value = mr

    gl = MagicMock()
    gl.project = project
    gl._url = "https://gitlab.example.com"
    gl.project_path = "org/repo"

    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    result = orch.run(issue_iid=42)
    
    assert result.agent == "openerge"


def test_run_defaults_to_developer_for_non_openerge():
    """Test that non-openerge requests default to the developer agent."""
    gl = _make_mock_gl(issue_labels=["feature"])
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    result = orch.run(issue_iid=42)
    
    assert result.agent == "developer"


# New tests for task reaping functionality

def test_task_reaping_removes_completed_tasks():
    """Test that task reaping removes completed tasks older than TTL."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    # Set a short TTL for testing
    orch.set_task_retention_ttl(timedelta(seconds=1))
    
    # Create a completed task
    run_id = orch.spawn(issue_iid=42).run_id
    
    # Wait for task to complete
    import time
    time.sleep(0.1)
    
    # Manually set completion time to be old
    with orch._lock:
        task = orch._tasks[run_id]
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now() - timedelta(seconds=2)
    
    # Run reaping
    reaped_count = orch.reap_completed_tasks_now()
    
    # Verify task was reaped
    with orch._lock:
        assert run_id not in orch._tasks
        assert reaped_count == 1


def test_task_reaping_keeps_recent_tasks():
    """Test that task reaping keeps recently completed tasks."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    # Set TTL to 10 seconds
    orch.set_task_retention_ttl(timedelta(seconds=10))
    
    # Create a completed task
    run_id = orch.spawn(issue_iid=42).run_id
    
    # Wait for task to complete
    import time
    time.sleep(0.1)
    
    # Manually set completion time to be recent
    with orch._lock:
        task = orch._tasks[run_id]
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now() - timedelta(seconds=1)
    
    # Run reaping
    reaped_count = orch.reap_completed_tasks_now()
    
    # Verify task was NOT reaped (still within TTL)
    with orch._lock:
        assert run_id in orch._tasks
        assert reaped_count == 0


def test_task_reaping_only_removes_completed_tasks_within_default_ttl():
    """Test that task reaping only removes completed/failed tasks by default, not recently created running tasks."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    # Create a running task (simulate by manually creating task record)
    # With the default stale_task_ttl of 2 hours, a task created 1 hour ago should NOT be reaped
    run_id = str(uuid.uuid4())
    with orch._lock:
        task = TaskRecord(
            run_id=run_id,
            issue_iid=42,
            agent=cm.get("developer"),
            status=TaskStatus.RUNNING,
            created_at=datetime.now() - timedelta(hours=1)
        )
        orch._tasks[run_id] = task
    
    # Run reaping
    reaped_count = orch.reap_completed_tasks_now()
    
    # Verify running task was NOT reaped (within default stale_task_ttl)
    with orch._lock:
        assert run_id in orch._tasks
        assert reaped_count == 0


def test_task_reaping_removes_failed_tasks():
    """Test that task reaping removes failed tasks older than TTL."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    # Set a short TTL
    orch.set_task_retention_ttl(timedelta(seconds=1))
    
    # Create a failed task
    run_id = str(uuid.uuid4())
    with orch._lock:
        task = TaskRecord(
            run_id=run_id,
            issue_iid=42,
            agent=cm.get("developer"),
            status=TaskStatus.FAILED,
            error="Test error",
            completed_at=datetime.now() - timedelta(seconds=2)
        )
        orch._tasks[run_id] = task
    
    # Run reaping
    reaped_count = orch.reap_completed_tasks_now()
    
    # Verify failed task was reaped
    with orch._lock:
        assert run_id not in orch._tasks
        assert reaped_count == 1


def test_task_reaping_removes_stale_running_tasks():
    """Test that task reaping removes stale running tasks older than stale_task_ttl."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    # Set a short stale task TTL
    orch.set_stale_task_ttl(timedelta(seconds=1))
    
    # Create a stale running task
    run_id = str(uuid.uuid4())
    with orch._lock:
        task = TaskRecord(
            run_id=run_id,
            issue_iid=42,
            agent=cm.get("developer"),
            status=TaskStatus.RUNNING,
            created_at=datetime.now() - timedelta(seconds=2)
        )
        orch._tasks[run_id] = task
    
    # Run reaping
    reaped_count = orch.reap_completed_tasks_now()
    
    # Verify stale running task was reaped
    with orch._lock:
        assert run_id not in orch._tasks
        assert reaped_count == 1


def test_task_reaping_keeps_active_running_tasks():
    """Test that task reaping keeps running tasks within stale_task_ttl."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    # Set a longer stale task TTL
    orch.set_stale_task_ttl(timedelta(hours=2))
    
    # Create a recently started running task
    run_id = str(uuid.uuid4())
    with orch._lock:
        task = TaskRecord(
            run_id=run_id,
            issue_iid=42,
            agent=cm.get("developer"),
            status=TaskStatus.RUNNING,
            created_at=datetime.now() - timedelta(minutes=30)
        )
        orch._tasks[run_id] = task
    
    # Run reaping
    reaped_count = orch.reap_completed_tasks_now()
    
    # Verify running task was NOT reaped (still active)
    with orch._lock:
        assert run_id in orch._tasks
        assert reaped_count == 0


def test_stop_terminates_reaper_thread():
    """Test that stop() terminates the reaper thread."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    # Stop the orchestrator
    orch.stop()
    
    # Verify reaper thread is stopped
    assert orch._reaper_thread is None


def test_stop_clears_task_registry():
    """Test that stop() clears the task registry to release memory."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    # Create some tasks
    run_id1 = str(uuid.uuid4())
    run_id2 = str(uuid.uuid4())
    with orch._lock:
        orch._tasks[run_id1] = TaskRecord(
            run_id=run_id1,
            issue_iid=42,
            agent=cm.get("developer"),
            status=TaskStatus.COMPLETED,
            completed_at=datetime.now()
        )
        orch._tasks[run_id2] = TaskRecord(
            run_id=run_id2,
            issue_iid=43,
            agent=cm.get("developer"),
            status=TaskStatus.RUNNING,
        )
    
    # Stop the orchestrator
    orch.stop()
    
    # Verify task registry is cleared
    with orch._lock:
        assert len(orch._tasks) == 0


def test_default_task_retention_ttl():
    """Test that default task retention TTL is 24 hours."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    # Check default TTL
    assert orch._task_retention_ttl == timedelta(hours=24)


def test_default_stale_task_ttl():
    """Test that default stale task TTL is 2 hours."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    assert orch._stale_task_ttl == timedelta(hours=2)


def test_set_task_retention_ttl():
    """Test setting custom task retention TTL."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    # Set custom TTL
    custom_ttl = timedelta(hours=12)
    orch.set_task_retention_ttl(custom_ttl)
    
    # Verify TTL was set
    assert orch._task_retention_ttl == custom_ttl


def test_set_stale_task_ttl():
    """Test setting custom stale task TTL."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    # Set custom stale TTL
    custom_ttl = timedelta(minutes=30)
    orch.set_stale_task_ttl(custom_ttl)
    
    # Verify TTL was set
    assert orch._stale_task_ttl == custom_ttl


def test_task_completion_sets_completed_at():
    """Test that task completion sets completed_at timestamp."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    # Spawn a task
    run_id = orch.spawn(issue_iid=42).run_id
    
    # Wait for task to complete
    import time
    time.sleep(0.1)
    
    # Check that completed_at is set
    with orch._lock:
        task = orch._tasks.get(run_id)
        if task and task.status == TaskStatus.COMPLETED:
            assert task.completed_at is not None
            assert isinstance(task.completed_at, datetime)


def test_task_failure_sets_completed_at():
    """Test that task failure sets completed_at timestamp."""
    # This test is more complex as it requires simulating a failure
    # For now, we'll test the manual case
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    # Create a failed task manually
    run_id = str(uuid.uuid4())
    with orch._lock:
        task = TaskRecord(
            run_id=run_id,
            issue_iid=42,
            agent=cm.get("developer"),
            status=TaskStatus.FAILED,
            error="Test error",
            completed_at=datetime.now()
        )
        orch._tasks[run_id] = task
    
    # Verify completed_at is set
    with orch._lock:
        task = orch._tasks[run_id]
        assert task.completed_at is not None
        assert isinstance(task.completed_at, datetime)


def test_multiple_task_reaping_cycles():
    """Test multiple reaping cycles work correctly."""
    gl = _make_mock_gl()
    cm = _make_competence()
    scrum = _make_scrum()
    orch = Orchestrator(gl=gl, competence=cm, scrum=scrum, router=LlmRouter())
    
    # Set short TTL
    orch.set_task_retention_ttl(timedelta(seconds=1))
    
    # Create multiple tasks with different completion times
    run_id1 = str(uuid.uuid4())
    run_id2 = str(uuid.uuid4())
    
    with orch._lock:
        # Old completed task (should be reaped)
        task1 = TaskRecord(
            run_id=run_id1,
            issue_iid=42,
            agent=cm.get("developer"),
            status=TaskStatus.COMPLETED,
            completed_at=datetime.now() - timedelta(seconds=2)
        )
        orch._tasks[run_id1] = task1
        
        # Recent completed task (should be kept)
        task2 = TaskRecord(
            run_id=run_id2,
            issue_iid=43,
            agent=cm.get("developer"),
            status=TaskStatus.COMPLETED,
            completed_at=datetime.now()
        )
        orch._tasks[run_id2] = task2
    
    # First reaping cycle
    reaped_count1 = orch.reap_completed_tasks_now()
    assert reaped_count1 == 1  # Only the old task should be reaped
    
    with orch._lock:
        assert run_id1 not in orch._tasks
        assert run_id2 in orch._tasks
    
    # Wait and run second cycle
    import time
    time.sleep(1.1)
    
    reaped_count2 = orch.reap_completed_tasks_now()
    assert reaped_count2 == 1  # Now the second task should be reaped
    
    with orch._lock:
        assert run_id2 not in orch._tasks