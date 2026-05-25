"""Creates Sprint 2 milestone, labels, and backlog issues on GitLab."""
import os
import sys
import warnings
import datetime
warnings.filterwarnings("ignore")

import gitlab

GITLAB_URL = os.getenv("GITLAB_URL", "")
PROJECT_PATH = os.getenv("GITLAB_PROJECT", "")
TOKEN = os.getenv("GITLAB_TOKEN", "")
for _var, _val in [("GITLAB_URL", GITLAB_URL), ("GITLAB_PROJECT", PROJECT_PATH), ("GITLAB_TOKEN", TOKEN)]:
    if not _val:
        print(f"✗ {_var} is not set.")
        sys.exit(1)

SPRINT2_START = datetime.date(2026, 5, 12)
SPRINT2_END   = datetime.date(2026, 5, 25)

LABELS = [
    {"name": "Sprint 1",      "color": "#6699cc", "description": "Sprint 1 — Bootstrap"},
    {"name": "Sprint 2",      "color": "#cc6699", "description": "Sprint 2 — Agent Execution"},
    {"name": "feature",       "color": "#428bca", "description": "New feature"},
    {"name": "architecture",  "color": "#8e44ad", "description": "Architecture / design work"},
    {"name": "testing",       "color": "#27ae60", "description": "Test coverage"},
    {"name": "dx",            "color": "#e67e22", "description": "Developer experience"},
    {"name": "In Progress",   "color": "#f0ad4e", "description": "Currently being worked on"},
    {"name": "In Review",     "color": "#5bc0de", "description": "Awaiting review"},
    {"name": "Done",          "color": "#5cb85c", "description": "Completed"},
]

ISSUES = [
    {
        "title": "Implement `factory run --issue <id>`",
        "description": (
            "## Goal\n"
            "The core Sprint 2 command. Given a GitLab issue ID, the factory spawns the "
            "correct agent (resolved via Competence Manager), passes the issue description "
            "as the task prompt, and opens a Work-in-Progress MR.\n\n"
            "## Acceptance Criteria\n"
            "- [ ] `factory run --issue 5` resolves the agent from issue labels\n"
            "- [ ] Agent receives issue title + description as prompt context\n"
            "- [ ] A WIP MR is created in GitLab and linked to the issue\n"
            "- [ ] Issue label is moved from backlog to `In Progress`\n"
            "- [ ] Command fails gracefully if issue does not exist\n\n"
            "## Agent\n"
            "Developer (claude-sonnet-4-6)"
        ),
        "labels": ["Sprint 2", "feature"],
        "weight": 5,
    },
    {
        "title": "Implement LLM Router (Claude + Mistral)",
        "description": (
            "## Goal\n"
            "The `LlmRouter` in `factory/adapters/llm_router.py` must dispatch task prompts "
            "to the correct provider based on the agent profile's `provider` and `model` fields.\n\n"
            "## Acceptance Criteria\n"
            "- [ ] `anthropic` provider routes via Anthropic SDK with prompt caching headers\n"
            "- [ ] `mistral` provider routes via Mistral API\n"
            "- [ ] A `stub` provider returns a deterministic response (for offline tests)\n"
            "- [ ] Router reads API keys from environment variables\n"
            "- [ ] Unit tests cover all three provider paths\n\n"
            "## Agent\n"
            "Developer (claude-sonnet-4-6)"
        ),
        "labels": ["Sprint 2", "feature"],
        "weight": 5,
    },
    {
        "title": "Implement Agent Orchestrator (spawn · monitor · retire)",
        "description": (
            "## Goal\n"
            "The `Orchestrator` in `factory/core/orchestrator.py` manages the full lifecycle "
            "of an agent run: spawn, poll for completion, update GitLab issue status, retire.\n\n"
            "## Acceptance Criteria\n"
            "- [ ] `orchestrator.spawn(agent, issue)` starts an agent task and returns a run ID\n"
            "- [ ] `orchestrator.poll(run_id)` returns current status (running / done / failed)\n"
            "- [ ] On completion, issue label is updated to `In Review`\n"
            "- [ ] On failure, issue gets a `factory-error` label and a comment with the error\n"
            "- [ ] `factory status` reflects running agents and their current issue\n\n"
            "## Agent\n"
            "Developer (claude-sonnet-4-6)"
        ),
        "labels": ["Sprint 2", "feature"],
        "weight": 8,
    },
    {
        "title": "Implement Scrum Engine (sprint state + velocity)",
        "description": (
            "## Goal\n"
            "The `ScrumEngine` in `factory/core/scrum.py` tracks sprint state, computes "
            "velocity, and manages the transition into a Sprint Review.\n\n"
            "## Acceptance Criteria\n"
            "- [ ] Reads current sprint config from `factory.yml`\n"
            "- [ ] Computes velocity: closed issues / total sprint issues\n"
            "- [ ] Detects sprint end (by date) and emits a `SprintEndEvent`\n"
            "- [ ] `factory status` shows velocity bar (already partially done)\n\n"
            "## Agent\n"
            "Developer (claude-sonnet-4-6)"
        ),
        "labels": ["Sprint 2", "feature"],
        "weight": 3,
    },
    {
        "title": "Implement `factory review --sprint <n>` (Sprint Review Gate)",
        "description": (
            "## Goal\n"
            "The human-in-the-loop checkpoint. When called, this command:\n"
            "1. Generates a Sprint Review summary from closed issues and posts it to the Wiki\n"
            "2. Waits for stakeholder `--approve` flag\n"
            "3. On approval: closes the current milestone, bumps `sprint.current` in "
            "`factory.yml`, and re-prioritises the backlog by milestone assignment\n\n"
            "## Acceptance Criteria\n"
            "- [ ] `factory review --sprint 1` posts `Sprint_Review_1` wiki page\n"
            "- [ ] `factory review --sprint 1 --approve` triggers next sprint setup\n"
            "- [ ] Old milestone is closed in GitLab\n"
            "- [ ] `factory.yml` sprint number is incremented automatically\n\n"
            "## Agent\n"
            "Architect (claude-opus-4-7)"
        ),
        "labels": ["Sprint 2", "feature"],
        "weight": 5,
    },
    {
        "title": "Add .env auto-loading to factory startup",
        "description": (
            "## Goal\n"
            "On startup, the factory should automatically load a `.env` file from the project "
            "root so users don't need to export environment variables manually during local dev.\n\n"
            "## Acceptance Criteria\n"
            "- [ ] `python-dotenv` loads `.env` before any command runs\n"
            "- [ ] `.env` is listed in `.gitignore` (already done)\n"
            "- [ ] `.env.example` is present and documents all required keys (already done)\n"
            "- [ ] Missing required keys produce a clear error message, not a traceback\n\n"
            "## Agent\n"
            "Developer (claude-sonnet-4-6)"
        ),
        "labels": ["Sprint 2", "dx"],
        "weight": 1,
    },
    {
        "title": "Write unit tests for LLM Router and Orchestrator",
        "description": (
            "## Goal\n"
            "Cover the two most critical Sprint 2 components with unit tests using the "
            "`stub` provider so tests run offline without real API keys.\n\n"
            "## Acceptance Criteria\n"
            "- [ ] `test_llm_router.py`: stub path, provider dispatch, missing key error\n"
            "- [ ] `test_orchestrator.py`: spawn → poll → retire lifecycle with mocked GitLab\n"
            "- [ ] All tests pass in CI without `ANTHROPIC_API_KEY` or `MISTRAL_API_KEY`\n\n"
            "## Agent\n"
            "QA (mistral-large-latest)"
        ),
        "labels": ["Sprint 2", "testing"],
        "weight": 3,
    },
]


def ensure_label(project, name, color, description):
    existing = {l.name for l in project.labels.list(all=True)}
    if name not in existing:
        project.labels.create({"name": name, "color": color, "description": description})
        print(f"  + label: {name}")
    else:
        print(f"  · label exists: {name}")


def main():
    gl = gitlab.Gitlab(GITLAB_URL, private_token=TOKEN)
    gl.auth()
    print(f"✓ Connected to {GITLAB_URL}")

    project = gl.projects.get(PROJECT_PATH)
    print(f"✓ Project: {project.name_with_namespace}\n")

    # --- Labels ---
    print("Creating labels...")
    for label in LABELS:
        ensure_label(project, label["name"], label["color"], label["description"])

    # --- Sprint 2 Milestone ---
    print("\nCreating Sprint 2 milestone...")
    existing_milestones = {m.title for m in project.milestones.list(all=True)}
    if "Sprint 2" not in existing_milestones:
        milestone = project.milestones.create({
            "title": "Sprint 2",
            "description": "Sprint 2 — Agent Execution Engine",
            "start_date": SPRINT2_START.isoformat(),
            "due_date": SPRINT2_END.isoformat(),
        })
        print(f"  + milestone: Sprint 2 (#{milestone.id})")
    else:
        milestones = project.milestones.list(search="Sprint 2")
        milestone = milestones[0]
        print(f"  · milestone exists: Sprint 2 (#{milestone.id})")

    # --- Issues ---
    print("\nCreating backlog issues...")
    for issue_data in ISSUES:
        payload = {
            "title": issue_data["title"],
            "description": issue_data["description"],
            "labels": ",".join(issue_data["labels"]),
            "milestone_id": milestone.id,
        }
        if issue_data.get("weight") is not None:
            payload["weight"] = issue_data["weight"]

        issue = project.issues.create(payload)
        print(f"  + #{issue.iid}: {issue.title}")

    print(f"\n✓ Sprint 2 backlog ready")
    print(f"  → {GITLAB_URL}/{PROJECT_PATH}/-/milestones/{milestone.iid}")
    print(f"  → {GITLAB_URL}/{PROJECT_PATH}/-/issues?milestone_title=Sprint+2")


if __name__ == "__main__":
    main()
