"""One-shot script: posts Architecture_v1 to the GitLab Wiki."""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import gitlab

GITLAB_URL = os.getenv("GITLAB_URL", "")
PROJECT_PATH = os.getenv("GITLAB_PROJECT", "")
TOKEN = os.getenv("GITLAB_TOKEN", "")
for _var, _val in [("GITLAB_URL", GITLAB_URL), ("GITLAB_PROJECT", PROJECT_PATH), ("GITLAB_TOKEN", TOKEN)]:
    if not _val:
        print(f"✗ {_var} is not set.")
        sys.exit(1)

WIKI_SLUG = "Architecture_v1"
WIKI_TITLE = "Architecture v1 — SoftwareTeamFabrik"

WIKI_CONTENT = """# Architecture v1 — SoftwareTeamFabrik

*Initial architecture — Discovery Session 2026-04-28*

---

## Decision Log

| Decision | Choice | Rationale |
|---|---|---|
| CLI Runtime | Python 3.11+ / Typer | Rich ecosystem, fast iteration, native python-gitlab SDK |
| Agent Profiles | YAML config files | GitOps-friendly, auditable, no infra to run |
| Human-in-the-Loop | **Scrum Review Gate** | Structured ceremony → reprioritized backlog → new sprint |
| GitLab Target | Self-hosted + GitLab.com | Configurable base URL; no hardcoded endpoints |
| MVP Command | `factory status` | Proves connectivity, sprint awareness, and agent registry |

---

## System Context

```mermaid
graph TD
    PO["Stakeholder / PO<br/>(runs CLI, approves Sprint Reviews)"]
    CLI["SoftwareTeamFabrik CLI<br/>(Python + Typer)"]
    GL["GitLab self-hosted<br/>(Issues · Wiki · Repo · CI/CD)"]
    CLAUDE["Claude API<br/>(Architect & Developer agents)"]
    MISTRAL["Mistral API<br/>(QA & review agents)"]

    PO -- "factory &lt;command&gt;" --> CLI
    CLI -- "REST / python-gitlab" --> GL
    CLI -- "Anthropic SDK" --> CLAUDE
    CLI -- "Mistral API" --> MISTRAL
    GL -- "email / webhook" --> PO
```

---

## Component Architecture

```mermaid
graph TD
    subgraph CLI ["CLI Layer — Typer"]
        cmd_status["factory status (MVP)"]
        cmd_run["factory run (Sprint 2)"]
        cmd_review["factory review (Sprint 2)"]
    end

    subgraph Core
        competence["Competence Manager<br/>reads agents/*.yml"]
        orchestrator["Agent Orchestrator<br/>spawn · monitor · retire"]
        scrum["Scrum Engine<br/>sprint · backlog · review gate"]
    end

    subgraph Adapters
        gl_client["GitLab Client<br/>configurable base URL"]
        llm_router["LLM Router<br/>Claude · Mistral · stub"]
    end

    subgraph Config ["Config — YAML"]
        factory_yml["factory.yml<br/>gitlab_url, project, sprint_config"]
        agent_profiles["agents/<br/>architect.yml · developer.yml · qa.yml"]
    end

    cmd_status --> scrum
    cmd_status --> gl_client
    cmd_status --> competence

    scrum --> gl_client
    orchestrator --> competence
    orchestrator --> llm_router

    competence --> agent_profiles
    gl_client --> factory_yml
```

---

## Scrum Review — Human-in-the-Loop Flow

```mermaid
sequenceDiagram
    participant PO as Stakeholder / PO
    participant CLI as factory CLI
    participant GL as GitLab
    participant Agents as AI Agents

    PO->>CLI: factory run --sprint 1
    CLI->>Agents: Spawn agents per backlog issues
    Agents-->>GL: Commits, MRs, comments

    Note over CLI,GL: Sprint ends (date or velocity threshold)

    CLI->>GL: POST Sprint Review to Wiki
    GL-->>PO: Notification (email / webhook)
    PO->>GL: Review Wiki page, add comments
    PO->>CLI: factory review --approve --sprint 1

    CLI->>GL: Reprioritize backlog (labels + milestone)
    CLI->>CLI: Plan Sprint 2
    CLI->>Agents: Spawn next sprint agents
```

The **Scrum Review** is the only mandatory human gate. Everything inside a sprint runs autonomously.

---

## MVP Scope — `factory status`

The first working command proves three things: GitLab connectivity, config loading, and agent registry resolution.

**Expected output:**

```
SoftwareTeamFabrik — Status Report
────────────────────────────────────────
GitLab        gitlab.raungart.com  ✓ connected
Project       software production/aiteamfactory
Sprint        #1 — Discovery (2 of 14 days remaining)

Active Agents
  architect   claude-opus-4-7      idle
  developer   claude-sonnet-4-6    working  →  #12 Scaffold CLI structure
  qa          mistral-large         idle

Open Issues   8 open / 3 in progress / 2 closed
Milestone     Sprint 1  ██░░░░░░░░  20%
```

---

## Repository Structure

```
aiteamfactory/
├── factory/
│   ├── __init__.py
│   ├── main.py                  # Typer app entry point
│   ├── commands/
│   │   └── status.py            # MVP
│   ├── core/
│   │   ├── competence.py        # Agent profile loader
│   │   ├── orchestrator.py      # Agent lifecycle
│   │   └── scrum.py             # Sprint/backlog logic
│   └── adapters/
│       ├── gitlab_client.py     # python-gitlab wrapper
│       └── llm_router.py        # Model selection + API calls
├── config/
│   ├── factory.yml              # GitLab URL, project path, sprint settings
│   └── agents/
│       ├── architect.yml
│       ├── developer.yml
│       └── qa.yml
├── tests/
├── pyproject.toml
└── .env.example
```

---

## Sprint 1 Backlog

| Issue | Title | Agent | Status |
|---|---|---|---|
| #1 | Post Architecture_v1 to Wiki | Architect | ✓ Done |
| #2 | Scaffold Python project + pyproject.toml | Developer | In Progress |
| #3 | Implement `factory status` command | Developer | Planned |
| #4 | Write integration test for GitLab connectivity | QA | Planned |

---

*Generated by SoftwareTeamFabrik Architect Agent — Sprint 1*
"""


def main():
    gl = gitlab.Gitlab(GITLAB_URL, private_token=TOKEN)
    try:
        gl.auth()
        print(f"✓ Connected to {GITLAB_URL}")
    except Exception as e:
        print(f"✗ Auth failed: {e}")
        sys.exit(1)

    project = gl.projects.get(PROJECT_PATH)
    print(f"✓ Project: {project.name_with_namespace}")

    existing_slugs = [w.slug for w in project.wikis.list()]
    if WIKI_SLUG in existing_slugs:
        wiki = project.wikis.get(WIKI_SLUG)
        wiki.content = WIKI_CONTENT
        wiki.title = WIKI_TITLE
        wiki.save()
        print(f"✓ Wiki page updated: {WIKI_SLUG}")
    else:
        project.wikis.create({"title": WIKI_TITLE, "content": WIKI_CONTENT})
        print(f"✓ Wiki page created: {WIKI_SLUG}")

    print(f"\n  → {GITLAB_URL}/{PROJECT_PATH}/-/wikis/{WIKI_SLUG}")


if __name__ == "__main__":
    main()
