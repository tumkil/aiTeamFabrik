"""Integration test: verifies GitLab connectivity and config loading."""
import os
import pytest
from factory.adapters.gitlab_client import GitLabClient
from factory.core.competence import CompetenceManager

_SKIP_REASON = "GITLAB_URL and GITLAB_PROJECT must be set to run integration tests"
_needs_gitlab = pytest.mark.skipif(
    not (os.getenv("GITLAB_URL") and os.getenv("GITLAB_PROJECT")),
    reason=_SKIP_REASON,
)


@pytest.mark.integration
@_needs_gitlab
def test_gitlab_connects():
    gl = GitLabClient()
    connected, detail = gl.connect()
    assert connected, f"GitLab connection failed: {detail}"


@pytest.mark.integration
@_needs_gitlab
def test_project_reachable():
    gl = GitLabClient()
    gl.connect()
    project = gl.project
    assert project is not None


def test_agent_profiles_load():
    cm = CompetenceManager()
    cm.load()
    agents = cm.agents
    assert len(agents) == 9
    names = {a.name for a in agents}

    assert names == {"architect", "developer", "meta_reviewer", "po_agent", "qa", "researcher", "reviewer", "security_reviewer", "reviewedcodeimplementation"}



def test_competence_resolves_capability():
    cm = CompetenceManager()
    cm.load()
    agent = cm.for_capability("system_design")
    assert agent is not None
    assert agent.name == "architect"
