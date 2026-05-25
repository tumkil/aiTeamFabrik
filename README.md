# AITeamFabrik 🏭

> A self-hosted AI scrum team that manages your entire software development process through GitLab.

AITeamFabrik is an autonomous, multi-agent software development factory. Specialized AI agents act as Architect, Product Owner, Developer, Reviewer, QA, and more — collaborating asynchronously through GitLab to plan, implement, review, test, and document software without manual intervention.

**Your code never leaves your infrastructure. Just run `docker compose up`.**

---

## How It Works

Each agent polls GitLab for issues and merge requests matching its role, does its work via an LLM, and writes results back to GitLab — commits, MR comments, wiki updates, pipeline fixes. GitLab is the communication bus and the audit trail.

```
[You] → create issue headline
   ↓
[PO Agent] → fleshes out requirements, refines backlog
   ↓
[Architect] → analyses issue, posts design plan as GitLab note
   ↓
[Developer] → implements code, opens draft MR
   ↓
[Reviewer + Meta Reviewer] → code review, posts verdict
   ↓
[Security Reviewer] → security and CVE review
   ↓
[Developer] → addresses feedback, refines (up to 3 rounds)
   ↓
[QA] → test planning, acceptance criteria
   ↓
[You] → review at sprint boundaries, merge or intervene
```

In monitor mode, this entire lifecycle runs autonomously. Issues labelled `ready` are picked up automatically — no manual triggering required.

---

## Agent Roster

| Agent | Role |
|-------|------|
| `architect` | System design, ADRs, wiki authoring |
| `developer` | Code implementation |
| `reviewer` | Code review on MRs |
| `meta_reviewer` | False-positive filter on primary review |
| `security_reviewer` | Security and CVE review |
| `qa` | Test planning, acceptance criteria |
| `researcher` | Web research and technology evaluation |
| `po_agent` | Product ownership, backlog management |
| `reviewedcodeimplementation` | Post-review code implementation |

Each agent runs its own LLM, configurable per role. Add or swap agents by dropping a `.yml` file in `config/agents/` — no code changes needed.

---

## Smart Model Routing

AITeamFabrik is model-agnostic. Each agent can run a different LLM based on the demands of its role:

```bash
--provider anthropic   # Claude models
--provider mistral     # Mistral / Devstral models
--provider ollama      # Any Ollama-compatible endpoint (local or cloud)
--provider google      # Gemini models
--provider stub        # Offline stub — no API key needed
```

Token spend is tracked per agent per day and per sprint, with configurable enforcement (`strict` / `warn` / `off`). Claude is reserved for tasks that need it — routine work runs on cheaper models.

---

## Requirements

- Docker + Docker Compose
- A GitLab instance (self-hosted recommended)
- API key for at least one LLM provider

---

## Setup

```bash
git clone https://github.com/yourusername/aiteamfabrik
cd aiteamfabrik
cp .env.example .env
# Edit .env — set GITLAB_URL, GITLAB_PROJECT, GITLAB_TOKEN and at least one LLM API key
docker compose up -d
```

Verify connectivity (no API calls):

```bash
factory status
```

Offline smoke test:

```bash
factory run --issue 1 --provider stub
```

---

## Configuration

### `config/factory.yml`

```yaml
gitlab:
  url: https://your-gitlab-instance.com
  project: your-group/your-project

sprint:
  current: 4
  name: Sprint 4
  start_date: '2026-06-09'
  duration_days: 14

token_budget:
  enforcement: strict   # strict | warn | off
  defaults:
    daily: 100000
    sprint: 1000000
```

### `config/agents/*.yml`

Each file defines one agent — model, provider, labels, capabilities, and system prompt:

```yaml
name: developer
model: your-preferred-model
provider: ollama
execution_mode: execute   # plan = analysis only | execute = writes code
task_labels:
  - feature
  - bug
```

### Agent Prompts

System prompts are not included in this repository — they represent operational knowledge built through real usage. Each agent's `.yml` file contains a placeholder `system_prompt` field. Fill it in based on the agent's role and tune based on your own experience.

---

## CLI Reference

### `factory monitor` — autonomous mode

Start the full autonomous polling loop. Issues labelled `ready` are processed end-to-end without manual intervention.

```bash
factory monitor
factory monitor --interval 60   # poll every 60s (default: 30)
```

**Autonomous lifecycle:**
1. Architect analyses issue, posts design plan
2. Developer implements and opens draft MR
3. Reviewer + Meta Reviewer post verdict
4. On `REQUEST CHANGES`: Developer refines, re-review triggered (up to 3×)
5. On `APPROVE`: MR auto-merged, wiki updated
6. Failed pipelines trigger automatic fix attempts

---

### `factory run` — spawn an agent manually

```bash
factory run --issue 42
factory run -i 42 --provider ollama
factory run -i 42 --provider stub    # offline, no API key needed
```

---

### `factory review-mr` — review a merge request

```bash
factory review-mr --mr 12
factory review-mr -m 12 --no-second-review    # skip meta-reviewer
factory review-mr -m 12 --no-post            # review without posting to GitLab
```

---

### `factory refine` — address review feedback

```bash
factory refine --mr 12
factory refine -m 12 --auto-merge    # merge automatically once approved
```

---

### `factory plan` — sprint planning

```bash
factory plan --sprint 4
factory plan -s 4 --algorithm optimal --budget 500000
factory plan -s 4 --show    # display existing plan
```

Algorithms: `greedy` (default) / `optimal` / `fair`

---

### `factory review` — sprint review

```bash
factory review --sprint 3
factory review -s 3 --approve              # close sprint, advance to next
factory review -s 3 --approve --plan-next  # also plan the next sprint
```

---

### Other commands

```bash
factory status           # GitLab connection, agents, token budget usage
factory po --issue 42    # PO agent discussion for an issue
factory fix-mr-pipeline --mr-iid 12   # act on a failed pipeline
factory continue-mr --mr 12           # continue an incomplete implementation
factory update-wiki --mr-id 12        # update wiki from MR diff
```

---

## Token Budget System

Every LLM call goes through a two-phase budget check — pre-flight estimation and post-call recording. Usage is persisted to `config/token_usage.yml` and visible via `factory status`.

```bash
FACTORY_IGNORE_BUDGET=1 factory run -i 42   # bypass enforcement
```

---

## Architecture

```
[Self-hosted GitLab]
  Issues · MRs · Wiki · Pipelines · Labels
        ↑ python-gitlab API ↓
[AITeamFabrik]
  ├── CLI (Typer)
  ├── Orchestrator + ScrumEngine
  ├── CompetenceManager (agent profiles)
  ├── LlmRouter (provider dispatch)
  ├── TokenBudgetManager
  ├── ExecutionEngine (tool-call loop)
  ├── TokenAwarePlanner
  └── WikiManager
        ↓
[LLM Providers]
  Ollama · Anthropic · Mistral · Google
```

Every agent action is a GitLab commit, comment, or wiki edit. Full audit trail. Nothing implicit.

---

## Testing

```bash
# Full suite (integration tests excluded by default)
python3 -m pytest tests/ -q

# Run integration tests (requires GITLAB_URL, GITLAB_PROJECT, GITLAB_TOKEN in .env)
python3 -m pytest tests/ -q -m integration
```

660 tests across 40 test files.

---

## Philosophy

Most AI coding tools are cloud-only, stateless, and opaque. AITeamFabrik is the opposite:

- **Self-hosted** — your code stays on your infrastructure
- **Transparent** — every agent action is a GitLab commit or comment
- **Async** — agents work on their own schedule, no always-on orchestrator needed
- **Human-in-the-loop** — sprint boundaries give you natural review and intervention points
- **Model-agnostic** — bring your own LLMs, swap them per agent, change anytime
- **Token-aware** — built-in budget tracking so costs never surprise you

---

## Roadmap

### v0.1 (current)
- Full scrum agent roster: PO, Architect, Developer, Reviewer, Meta Reviewer, Security Reviewer, QA, Researcher
- GitLab as async communication bus and audit trail
- Autonomous monitor mode with full issue lifecycle
- Per-agent model configuration and token budget system
- Docker Compose setup

### v2 (in development)
- Agents as independent Docker containers with direct API communication
- Faster feedback loops, parallel agent execution
- Automatic model routing based on output validation signals
- Built by v1 — the factory is building its own next version

### v3 (planned)
- Three-tier memory: identity, role-based, and project-wide
- Self-written skills that accumulate across sprints
- People Competence Manager agent for intelligent model routing

---

## Status

Early release. Built in 3 weeks, currently running in production on personal projects with 100+ completed merge requests.

*v2 is being built by v1. Watch the commit history.*

---

## License

License TBD. Currently all rights reserved.