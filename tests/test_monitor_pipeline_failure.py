"""Tests for the monitor's pipeline-failure handling flow.

When automerge fails, the monitor must:
1. Check the pipeline status
2. If the pipeline failed, retrieve logs via GitLabClient
3. Post a note on the MR with the logs and a REQUEST CHANGES marker
4. Label the MR with factory-verdict-changes
5. Trigger the Developer agent to repair the failing tests
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import pytest

from factory.core.monitor import FactoryMonitor
from factory.core.verdict import VERDICT_CHANGES


# ---------------------------------------------------------------------------
# Fake GitLab objects
# ---------------------------------------------------------------------------

class FakeNote:
    def __init__(self, body: str) -> None:
        self.body = body


class FakeNotesManager:
    def __init__(self, initial_notes: list[str] | None = None) -> None:
        self._notes: list[FakeNote] = []
        if initial_notes:
            for body in initial_notes:
                self._notes.append(FakeNote(body))

    def list(self, *, all: bool = True, **_kwargs) -> list[FakeNote]:
        return list(self._notes)

    def create(self, data: dict) -> FakeNote:
        n = FakeNote(body=data.get("body", ""))
        self._notes.append(n)
        return n


class FakePipeline:
    """Minimal fake pipeline object."""
    def __init__(self, status: str, pipeline_id: int = 1, sha: str = "abc12345", ref: str = "main") -> None:
        self.status = status
        self.id = pipeline_id
        self.sha = sha
        self.ref = ref


class FakePipelinesManager:
    def __init__(self, pipelines: list | None = None) -> None:
        self._pipelines = pipelines or []

    def list(self, **_kwargs) -> list:
        return list(self._pipelines)


class FakeMR:
    def __init__(
        self,
        iid: int = 1,
        title: str = "Test MR",
        source_branch: str = "feature/x",
        draft: bool = False,
        labels: Optional[list[str]] = None,
        initial_notes: Optional[list[str]] = None,
        pipelines: Optional[FakePipelinesManager] = None,
    ) -> None:
        self.iid = iid
        self.title = title
        self.description = f"Closes #{iid}"
        self.source_branch = source_branch
        self.draft = draft
        self.work_in_progress = draft
        self.state = "opened"
        self.notes = FakeNotesManager(initial_notes=initial_notes)
        self.labels = list(labels or [])
        self.merged = False
        self.saved = 0
        self._merge_error: Optional[Exception] = None
        # Public pipelines attribute — accessed as mr.pipelines by monitor code
        self.pipelines: FakePipelinesManager = pipelines or FakePipelinesManager()

    def merge(self, *, remove_source_branch: bool = False) -> None:
        if self._merge_error:
            raise self._merge_error
        self.merged = True
        self.state = "merged"

    def save(self) -> None:
        self.saved += 1


class FakeProject:
    def __init__(self, mrs: list[FakeMR] | None = None) -> None:
        self.mergerequests = SimpleNamespace(
            list=lambda **kw: [m for m in (mrs or []) if m.state == "opened"],
        )
        self.issues = SimpleNamespace(list=lambda **kw: [])
        self.default_branch = "main"
        self.id = 1
        self.http_url_to_repo = "https://example.invalid/repo.git"


# ---------------------------------------------------------------------------
# Monitor builder
# ---------------------------------------------------------------------------

_FACTORY_REVIEW_HEADER = "### :mag: Code Review by SoftwareTeamFabrik"


def _build_monitor(project: FakeProject, gl_extras: dict | None = None) -> FactoryMonitor:
    """Construct a FactoryMonitor without touching the real GitLab."""
    mon = FactoryMonitor.__new__(FactoryMonitor)
    mon._gl = SimpleNamespace(
        project=project,
        hostname="example.invalid",
        _url="https://example.invalid",
        create_merge_request=lambda **_: None,
        **(gl_extras or {}),
    )
    mon._competence = SimpleNamespace(get=lambda name: SimpleNamespace(
        name=name, system_prompt="", display_name=name,
    ))
    mon._scrum = None
    mon._router = None
    mon._token = ""
    mon._budget_manager = SimpleNamespace(can_consume=lambda *_a, **_k: (True, ""))
    mon._poll_interval = 30
    mon._active_issues = {}
    mon._last_processed_mrs = {}
    mon._last_processed_issues = {}
    mon._wiki_manager = SimpleNamespace(format_template=lambda *_a, **_kw: "")
    return mon


# ---------------------------------------------------------------------------
# _check_pipeline_status tests
# ---------------------------------------------------------------------------

class TestCheckPipelineStatus:
    """Test _check_pipeline_status dispatches correctly per pipeline state."""

    def test_failed_pipeline_triggers_handle_pipeline_failure(self, monkeypatch):
        mr = FakeMR(iid=1, pipelines=FakePipelinesManager([FakePipeline(status="failed")]))

        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        handled = []
        monkeypatch.setattr(mon, "_handle_pipeline_failure", lambda m: handled.append(m.iid))

        mon._check_pipeline_status(mr)

        assert handled == [1]

    def test_success_pipeline_does_not_trigger_handler(self, monkeypatch):
        mr = FakeMR(iid=2, pipelines=FakePipelinesManager([FakePipeline(status="success")]))

        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        handled = []
        monkeypatch.setattr(mon, "_handle_pipeline_failure", lambda m: handled.append(m.iid))

        mon._check_pipeline_status(mr)

        assert handled == []

    def test_running_pipeline_does_not_trigger_handler(self, monkeypatch):
        mr = FakeMR(iid=3, pipelines=FakePipelinesManager([FakePipeline(status="running")]))

        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        handled = []
        monkeypatch.setattr(mon, "_handle_pipeline_failure", lambda m: handled.append(m.iid))

        mon._check_pipeline_status(mr)

        assert handled == []

    def test_no_pipelines_does_not_crash(self, monkeypatch):
        mr = FakeMR(iid=4, pipelines=FakePipelinesManager([]))

        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        handled = []
        monkeypatch.setattr(mon, "_handle_pipeline_failure", lambda m: handled.append(m.iid))

        # Should not raise
        mon._check_pipeline_status(mr)
        assert handled == []

    def test_pipeline_list_exception_handled(self, monkeypatch):
        mr = FakeMR(iid=5)
        # Replace pipelines with an object that raises on list()
        mr.pipelines = SimpleNamespace(list=lambda **kw: (_ for _ in ()).throw(Exception("pipeline API error")))

        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        handled = []
        monkeypatch.setattr(mon, "_handle_pipeline_failure", lambda m: handled.append(m.iid))

        # Should not raise
        mon._check_pipeline_status(mr)
        assert handled == []


# ---------------------------------------------------------------------------
# _handle_pipeline_failure tests
# ---------------------------------------------------------------------------

class TestHandlePipelineFailure:
    """Test _handle_pipeline_failure retrieves logs, posts note, labels, and triggers refinement."""

    def test_posts_note_with_logs_and_request_changes(self, monkeypatch):
        mr = FakeMR(iid=10)
        project = FakeProject(mrs=[mr])

        fake_logs = "### Pipeline 42 failed\n\n#### Job `test` — failed\n```log\nAssertionError\n```"
        gl_extras = {"get_failed_pipeline_logs": lambda mr_iid: fake_logs}

        mon = _build_monitor(project, gl_extras=gl_extras)
        monkeypatch.setattr(mon, "_run_refinement", lambda mr, dev: None)

        mon._handle_pipeline_failure(mr)

        notes = mr.notes.list()
        assert len(notes) == 1
        body = notes[0].body
        assert "REQUEST CHANGES" in body
        assert "Pipeline" in body
        assert "AssertionError" in body

    def test_sets_verdict_label_to_changes(self, monkeypatch):
        mr = FakeMR(iid=11)
        project = FakeProject(mrs=[mr])

        gl_extras = {"get_failed_pipeline_logs": lambda mr_iid: "some logs"}
        mon = _build_monitor(project, gl_extras=gl_extras)
        monkeypatch.setattr(mon, "_run_refinement", lambda mr, dev: None)

        mon._handle_pipeline_failure(mr)

        assert VERDICT_CHANGES in mr.labels

    def test_triggers_developer_refinement(self, monkeypatch):
        mr = FakeMR(iid=12)
        project = FakeProject(mrs=[mr])

        gl_extras = {"get_failed_pipeline_logs": lambda mr_iid: "some logs"}
        mon = _build_monitor(project, gl_extras=gl_extras)

        refined = []
        monkeypatch.setattr(mon, "_run_refinement", lambda m, dev: refined.append(m.iid))

        mon._handle_pipeline_failure(mr)

        assert refined == [12]

    def test_empty_logs_still_posts_note(self, monkeypatch):
        mr = FakeMR(iid=13)
        project = FakeProject(mrs=[mr])

        gl_extras = {"get_failed_pipeline_logs": lambda mr_iid: ""}
        mon = _build_monitor(project, gl_extras=gl_extras)
        monkeypatch.setattr(mon, "_run_refinement", lambda mr, dev: None)

        mon._handle_pipeline_failure(mr)

        notes = mr.notes.list()
        assert len(notes) == 1
        assert "REQUEST CHANGES" in notes[0].body

    def test_logs_retrieval_failure_still_posts_note(self, monkeypatch):
        mr = FakeMR(iid=14)
        project = FakeProject(mrs=[mr])

        def bad_logs(mr_iid):
            raise RuntimeError("API down")
        gl_extras = {"get_failed_pipeline_logs": bad_logs}
        mon = _build_monitor(project, gl_extras=gl_extras)
        monkeypatch.setattr(mon, "_run_refinement", lambda mr, dev: None)

        mon._handle_pipeline_failure(mr)

        notes = mr.notes.list()
        assert len(notes) == 1
        assert "REQUEST CHANGES" in notes[0].body

    def test_no_developer_agent_does_not_crash(self, monkeypatch):
        mr = FakeMR(iid=15)
        project = FakeProject(mrs=[mr])

        gl_extras = {"get_failed_pipeline_logs": lambda mr_iid: "some logs"}
        mon = _build_monitor(project, gl_extras=gl_extras)
        # Override competence to return None for developer
        mon._competence = SimpleNamespace(get=lambda name: None)

        # Should not raise
        mon._handle_pipeline_failure(mr)

        # Note and label should still be posted
        notes = mr.notes.list()
        assert len(notes) == 1
        assert VERDICT_CHANGES in mr.labels

    def test_refinement_failure_does_not_crash(self, monkeypatch):
        mr = FakeMR(iid=16)
        project = FakeProject(mrs=[mr])

        gl_extras = {"get_failed_pipeline_logs": lambda mr_iid: "some logs"}
        mon = _build_monitor(project, gl_extras=gl_extras)
        monkeypatch.setattr(mon, "_run_refinement", lambda mr, dev: (_ for _ in ()).throw(RuntimeError("refine failed")))

        # Should not raise
        mon._handle_pipeline_failure(mr)

        # Note and label should still be posted
        notes = mr.notes.list()
        assert len(notes) == 1
        assert "REQUEST CHANGES" in notes[0].body
        assert VERDICT_CHANGES in mr.labels


# ---------------------------------------------------------------------------
# _auto_merge failure path tests
# ---------------------------------------------------------------------------

class TestAutoMergeFailurePath:
    """When automerge fails, _auto_merge delegates to _check_pipeline_status."""

    def test_merge_failure_triggers_pipeline_check(self, monkeypatch):
        mr = FakeMR(iid=20)
        mr._merge_error = Exception("Merge conflict")

        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        checked = []
        monkeypatch.setattr(mon, "_check_pipeline_status", lambda m: checked.append(m.iid))
        monkeypatch.setattr(mon, "_trigger_wiki_documentation", lambda m: None)

        mon._auto_merge(mr)

        assert checked == [20]

    def test_merge_success_does_not_trigger_pipeline_check(self, monkeypatch):
        mr = FakeMR(iid=21)
        # No merge error — merge succeeds

        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        checked = []
        monkeypatch.setattr(mon, "_check_pipeline_status", lambda m: checked.append(m.iid))
        monkeypatch.setattr(mon, "_trigger_wiki_documentation", lambda m: None)

        mon._auto_merge(mr)

        assert checked == []
        assert mr.merged is True

    def test_merge_failure_pipeline_failed_triggers_full_flow(self, monkeypatch):
        """End-to-end: merge fails → pipeline failed → logs retrieved, note posted,
        label set, developer refinement triggered."""
        mr = FakeMR(iid=22)
        mr._merge_error = Exception("Pipeline failed")
        mr.pipelines = FakePipelinesManager([FakePipeline(status="failed")])

        project = FakeProject(mrs=[mr])
        fake_logs = "### Pipeline 99 failed\n```log\nFAIL test_suite\n```"
        gl_extras = {"get_failed_pipeline_logs": lambda mr_iid: fake_logs}
        mon = _build_monitor(project, gl_extras=gl_extras)

        refined = []
        monkeypatch.setattr(mon, "_run_refinement", lambda m, dev: refined.append(m.iid))
        monkeypatch.setattr(mon, "_trigger_wiki_documentation", lambda m: None)

        mon._auto_merge(mr)

        # Pipeline failure handler was invoked
        assert refined == [22]
        # Note was posted
        notes = mr.notes.list()
        assert len(notes) == 1
        assert "REQUEST CHANGES" in notes[0].body
        assert "FAIL test_suite" in notes[0].body
        # Label was set
        assert VERDICT_CHANGES in mr.labels

    def test_merge_failure_pipeline_succeeded_no_handler(self, monkeypatch):
        """Merge fails but pipeline is green — no repair needed, just log."""
        mr = FakeMR(iid=23)
        mr._merge_error = Exception("Merge conflict (not pipeline)")
        mr.pipelines = FakePipelinesManager([FakePipeline(status="success")])

        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        refined = []
        monkeypatch.setattr(mon, "_run_refinement", lambda m, dev: refined.append(m.iid))
        monkeypatch.setattr(mon, "_trigger_wiki_documentation", lambda m: None)

        mon._auto_merge(mr)

        # No refinement triggered since pipeline passed
        assert refined == []
        # No REQUEST CHANGES note posted
        notes = mr.notes.list()
        assert len(notes) == 0


# ---------------------------------------------------------------------------
# Integration: _check_mrs with pipeline failure
# ---------------------------------------------------------------------------

class TestMRsCheckWithPipelineFailure:
    """When an approved MR fails to merge due to pipeline failure, the
    refinement loop should be triggered to repair the failing tests."""

    def test_approved_mr_merge_failure_pipeline_failed_triggers_repair(self, monkeypatch):
        """Simulate the full lifecycle: review approves, automerge fails,
        pipeline failure detected, repair triggered."""
        from factory.core.verdict import VERDICT_APPROVE

        mr = FakeMR(
            iid=30,
            labels=[VERDICT_APPROVE],
            initial_notes=[
                f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n✅ APPROVE",
            ],
        )
        # Make merge fail
        mr._merge_error = Exception("Pipeline failed")
        mr.pipelines = FakePipelinesManager([FakePipeline(status="failed")])

        project = FakeProject(mrs=[mr])
        fake_logs = "### Pipeline 5 failed\n```log\nAssertionError in test_foo\n```"
        gl_extras = {"get_failed_pipeline_logs": lambda mr_iid: fake_logs}
        mon = _build_monitor(project, gl_extras=gl_extras)

        refined = []
        monkeypatch.setattr(mon, "_run_review", lambda *a, **kw: None)
        monkeypatch.setattr(mon, "_run_refinement", lambda m, dev: refined.append(m.iid))
        monkeypatch.setattr(mon, "_trigger_wiki_documentation", lambda m: None)

        mon._check_mrs(project)

        # The approved MR should trigger auto_merge, which fails and checks
        # the pipeline, finds it failed, and triggers repair.
        assert 30 in refined

    def test_approved_mr_merge_success_no_pipeline_check(self, monkeypatch):
        """When merge succeeds, no pipeline check or repair is triggered."""
        from factory.core.verdict import VERDICT_APPROVE

        mr = FakeMR(
            iid=31,
            labels=[VERDICT_APPROVE],
            initial_notes=[
                f"{_FACTORY_REVIEW_HEADER}\n\n## Verdict\n✅ APPROVE",
            ],
        )
        # Merge succeeds (no error)

        project = FakeProject(mrs=[mr])
        mon = _build_monitor(project)

        refined = []
        pipeline_checked = []
        monkeypatch.setattr(mon, "_run_review", lambda *a, **kw: None)
        monkeypatch.setattr(mon, "_run_refinement", lambda m, dev: refined.append(m.iid))
        monkeypatch.setattr(mon, "_check_pipeline_status", lambda m: pipeline_checked.append(m.iid))
        monkeypatch.setattr(mon, "_trigger_wiki_documentation", lambda m: None)

        mon._check_mrs(project)

        # Merge succeeds, no pipeline check needed
        assert pipeline_checked == []
        assert refined == []
        assert mr.merged is True