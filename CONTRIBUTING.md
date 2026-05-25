# Contributing to SoftwareTeamFabrik

## Development Setup

```bash
python3 -m venv .venv          # Python ≥ 3.11 required
source .venv/bin/activate
pip install -e ".[dev,llm]"
cp .env.example .env           # fill in GITLAB_TOKEN + at least one LLM key
```

## Project Layout

```
factory/
├── main.py                        # Typer app + command registration
├── adapters/
│   ├── gitlab_client.py           # python-gitlab wrapper
│   └── llm_router.py              # LLM dispatch + two-phase budget enforcement
├── commands/
│   ├── status.py                  # factory status
│   ├── run.py                     # factory run
│   ├── review.py                  # factory review (sprint wiki)
│   ├── review_mr.py               # factory review-mr
│   ├── refine.py                  # factory refine
│   ├── continue_mr.py             # factory continue-mr
│   ├── plan.py                    # factory plan
│   ├── update_wiki.py             # factory update-wiki
│   ├── fix_mr_pipeline.py         # factory fix-mr-pipeline
│   ├── chat.py                    # factory po
│   ├── reset_budget.py            # token budget reset helper (not yet wired to CLI)
│   ├── document.py                # documentation generator (not yet wired to CLI)
│   └── label_issue.py             # issue labeller (not yet wired to CLI)
└── core/
    ├── competence.py              # AgentProfile + CompetenceManager
    ├── token_budget.py            # TokenBudgetManager — budget tracking + enforcement
    ├── execution_engine.py        # Tool-call loop for execute-mode agents
    ├── orchestrator.py            # Async task spawn / poll / retire
    ├── scrum.py                   # Sprint state + advancement
    ├── monitor.py                 # Autonomous polling loop + factory monitor command
    ├── events.py                  # SprintEndEvent, TaskSpawnedEvent, etc.
    ├── planner.py                 # TokenAwarePlanner — sprint scheduling
    ├── wiki_manager.py            # Wiki templates + formatting helpers
    ├── architect_feedback.py      # ArchitectFeedbackEnforcer — context injection
    ├── issue_queue.py             # IssueQueue — prioritised issue selection
    ├── po_agent.py                # PO agent discussion flow
    ├── verdict.py                 # Review verdict parsing
    ├── resilience.py              # Retry / back-off helpers
    ├── model_selector.py          # Model selection helpers
    ├── provider_defaults.py       # Per-provider default settings
    ├── mcp_server.py              # Base MCP server
    ├── architect_mcp_server.py    # Architect MCP server
    ├── developer_mcp_server.py    # Developer MCP server
    ├── reviewer_mcp_server.py     # Reviewer MCP server
    ├── planner_mcp_server.py      # Planner MCP server
    └── po_mcp_server.py           # PO agent MCP server

config/
├── factory.yml              # GitLab, sprint, and token_budget config
├── agents/                  # One YAML file per agent profile
│   ├── architect.yml
│   ├── developer.yml
│   ├── reviewer.yml
│   ├── meta_reviewer.yml
│   ├── security_reviewer.yml
│   ├── qa.yml
│   ├── researcher.yml
│   ├── po_agent.yml
│   └── reviewedcodeimplementation.yml
└── token_usage.yml          # Auto-generated; per-agent usage tracking

tests/                       # pytest — unit + GitLab integration tests
```

## Running Tests

```bash
python3 -m pytest tests/ -q                   # full suite (~3 min, 491 tests)
python3 -m pytest tests/ -q -k "not gitlab"   # skip live GitLab tests (fast)
```

Integration tests (`test_gitlab_connection.py`) require a reachable instance and a valid `GITLAB_TOKEN`. The `test_thread_safety` and related file-lock tests do real I/O and account for most of the runtime.

## Adding a New Command

1. Create `factory/commands/my_command.py` with a `cmd_my_command(...)` function decorated with `typer.Option` parameters.
2. Register it in `factory/main.py`:
   ```python
   from factory.commands.my_command import cmd_my_command
   app.command("my-command", help="...")(cmd_my_command)
   ```
3. Add tests in `tests/test_my_command.py`.

## Adding a New Agent

Create `config/agents/my_agent.yml`:

```yaml
name: my_agent
display_name: My Agent
model: claude-sonnet-4-6
provider: anthropic
execution_mode: plan          # plan = analysis only | execute = writes code
system_prompt: |
  You are...
tools: []
task_labels:
  - my-label
capabilities:
  - my_capability
max_concurrent_tasks: 1
```

No code changes needed — `CompetenceManager` picks up all `*.yml` files in `config/agents/` at startup. The agent count assertion in `tests/test_gitlab_connection.py` must be updated when adding agents.

To give the agent a token budget, add an entry under `token_budget.agents` in `config/factory.yml`. Without one it inherits `token_budget.defaults`.

## LLM Providers

`LlmRouter` dispatches based on `agent.provider`:

| Value | Backend | Environment variable |
|-------|---------|----------------------|
| `anthropic` | `anthropic` SDK | `ANTHROPIC_API_KEY` |
| `mistral` | `mistralai` SDK | `MISTRAL_API_KEY` |
| `ollama` | HTTP API | `OLLAMA_BASE_URL`, optionally `OLLAMA_API_KEY` |
| `google` | `google-generativeai` SDK | `GOOGLE_API_KEY` |
| `stub` | built-in stub | none |

The stub returns `[STUB] Agent '...' would respond to: ...` and records zero tokens. Use it for all offline testing.

### Ollama notes

`OLLAMA_BASE_URL` defaults to `http://localhost:11434`. For remote/authenticated endpoints set `OLLAMA_API_KEY` (sent as `Bearer` token). Adjust the request timeout with `OLLAMA_TIMEOUT` (seconds, default 120, max 600).

## Token Budget System

`factory/core/token_budget.py` — `TokenBudgetManager`:

- `can_consume(agent, estimated_tokens)` — read-only pre-flight check; returns `(allowed, reason)`.
- `consume(agent, input_tokens, output_tokens, model)` — writes actual usage; reloads from disk inside `FileLock` before writing to prevent cross-process overwrites.
- `remaining(agent, scope)` — returns remaining tokens for `"daily"` or `"sprint"`.
- `is_over_budget(agent, scope)` — boolean over-limit check.
- `reset(agent_name, scope)` — removes a single agent's usage entry.
- `usage_report()` — dict consumed by `factory status`.

`LlmRouter.complete()` integrates the budget in two phases — see `factory/adapters/llm_router.py` for details. Set `FACTORY_IGNORE_BUDGET=1` to bypass enforcement entirely (useful in CI).

## Monitor Behaviour

`factory monitor` runs a continuous polling loop. The autonomous issue lifecycle requires the `ready` label:

1. Issue gets the `ready` label — monitor picks it up.
2. Architect analyses the issue and posts a plan as a GitLab note.
3. Developer implements, commits, and creates a draft MR.
4. Code Reviewer + Meta Reviewer post a verdict comment.
5. On `REQUEST CHANGES` or `BLOCK`: Developer refines the MR branch; re-review triggered (up to 3 times).
6. On `APPROVE` without draft status: MR is auto-merged and wiki is updated.
7. Failed pipelines trigger `fix-mr-pipeline` automatically.

The monitor respects WIP limits from `config/factory.yml` (`sprint.capacity.max_issues`). Issues without the `ready` label are skipped.

## Sprint Planning

```bash
factory plan --sprint 4                          # greedy scheduling within default budget
factory plan -s 4 --algorithm optimal --budget 500000
factory plan -s 4 --show                         # display existing plan
```

Plans are saved as `config/sprint-N-plan.yml` and read by the monitor when selecting issues.

## Code Style

- No comments unless the *why* is non-obvious.
- No docstrings unless the function has a non-obvious contract.
- No backwards-compatibility shims — just change the code.
- Error handling only at system boundaries (user input, external APIs).
- Never use shell pipes (`|`), redirects (`2>&1`, `>`), or background operators (`&`) inside `run_command` tool calls in agent system prompts.

## Workflow

1. Create a GitLab issue with relevant labels.
2. Branch off `master` as `factory/issue-N-short-description`.
3. Open a draft MR; add the `In Review` label when ready.
4. Run `factory review-mr --mr N` to get the AI review.
5. Address findings, push, re-review until `APPROVE`.
6. Merge to `master`.