# SoftwareTeamFabrik — Deployment Guide

## Local Development

```bash
python3 -m venv .venv          # Python ≥ 3.11 required
source .venv/bin/activate
pip install -e ".[dev,llm]"

cp .env.example .env
# Set GITLAB_TOKEN + at least one of: ANTHROPIC_API_KEY, MISTRAL_API_KEY, OLLAMA_BASE_URL

factory status                          # verify connection
factory run --issue 1 --provider stub   # offline smoke test
```

---

## Docker

### Build

```bash
docker build -t softwareteamfabrik .
```

### Run interactively

```bash
docker run -it --rm \
  -e GITLAB_TOKEN=$GITLAB_TOKEN \
  -e OLLAMA_BASE_URL=$OLLAMA_BASE_URL \
  -e OLLAMA_API_KEY=$OLLAMA_API_KEY \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  softwareteamfabrik status
```

### Run with Docker Compose

```bash
# Create .env with API keys
cat > .env <<EOF
GITLAB_TOKEN=$GITLAB_TOKEN
OLLAMA_BASE_URL=$OLLAMA_BASE_URL
OLLAMA_API_KEY=$OLLAMA_API_KEY
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
MISTRAL_API_KEY=$MISTRAL_API_KEY
EOF

# Start autonomous monitor
docker-compose run --rm factory monitor

# Detached
docker-compose up -d
docker-compose logs -f
docker-compose down
```

**Container note:** The execution engine clones the GitLab repository into a temporary directory when no `.git` folder exists. This requires git ≥ 2.31 for `GIT_CONFIG_*` env-var auth; older git versions use the oauth2 token-in-URL fallback automatically.

---

## Kubernetes

```bash
kubectl create secret generic factory-secrets \
  --from-literal=GITLAB_TOKEN=$GITLAB_TOKEN \
  --from-literal=OLLAMA_BASE_URL=$OLLAMA_BASE_URL \
  --from-literal=OLLAMA_API_KEY=$OLLAMA_API_KEY \
  --from-literal=ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  --from-literal=MISTRAL_API_KEY=$MISTRAL_API_KEY \
  --from-literal=GOOGLE_API_KEY=$GOOGLE_API_KEY
```

`deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: softwareteamfabrik
spec:
  replicas: 1
  selector:
    matchLabels:
      app: softwareteamfabrik
  template:
    metadata:
      labels:
        app: softwareteamfabrik
    spec:
      containers:
      - name: factory
        image: softwareteamfabrik:latest
        envFrom:
        - secretRef:
            name: factory-secrets
        args: ["monitor", "--interval", "60"]
        resources:
          limits:
            memory: "512Mi"
            cpu: "500m"
```

```bash
kubectl apply -f deployment.yaml
```

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GITLAB_TOKEN` | Yes | Personal access token with `api` scope |
| `ANTHROPIC_API_KEY` | For `anthropic` provider | Claude models |
| `MISTRAL_API_KEY` | For `mistral` provider | Mistral/Devstral models |
| `OLLAMA_BASE_URL` | For `ollama` provider | Ollama endpoint (default: `http://localhost:11434`) |
| `OLLAMA_API_KEY` | For authenticated Ollama | Bearer token for Ollama API |
| `OLLAMA_TIMEOUT` | No | Request timeout in seconds, 1–600 (default: 120) |
| `GOOGLE_API_KEY` | For `google` provider | Gemini models |
| `FACTORY_IGNORE_BUDGET` | No | Set to `1`/`true`/`yes` to bypass token budget enforcement |

### `config/factory.yml`

```yaml
gitlab:
  url: https://gitlab.your-company.com
  project: your-group/your-project

sprint:
  current: 4
  name: Sprint 4
  start_date: '2026-06-09'
  duration_days: 14
  capacity:
    max_issues: -1    # -1 = unlimited
    total_points: -1
  labels:
    in_progress: In Progress
    review: In Review
    done: Done
  milestone_prefix: Sprint

token_budget:
  # strict = block calls over budget | warn = log only | off = disabled
  enforcement: strict
  defaults:
    daily: 100000       # fallback for agents not listed below
    sprint: 1000000
  agents:
    architect:
      daily: 50000
      sprint: 400000
    developer:
      daily: 400000
      sprint: 2000000
    reviewer:
      daily: 800000
      sprint: 4000000
```

### Agent Profiles (`config/agents/*.yml`)

Nine profiles ship out of the box:

| Name | Model | Provider | Mode | Role |
|------|-------|----------|------|------|
| `architect` | `mistral-large-3:675b-cloud` | `ollama` | plan | System design, ADRs, wiki authoring |
| `developer` | `kimi-k2.6:cloud` | `ollama` | execute | Code implementation |
| `reviewer` | `glm-5.1:cloud` | `ollama` | plan | Code review on MRs |
| `meta_reviewer` | `kimi-k2.6:cloud` | `ollama` | plan | False-positive filter on primary review |
| `security_reviewer` | `devstral-2:123b-cloud` | `ollama` | plan | Security and CVE review |
| `qa` | `mistral-large-3:675b-cloud` | `ollama` | plan | Test planning, acceptance criteria |
| `researcher` | `claude-sonnet-4-6` | `anthropic` | plan | Web research and technology evaluation |
| `po_agent` | `deepseek-v4-pro:cloud` | `ollama` | plan | Product ownership, backlog management |
| `reviewedcodeimplementation` | `devstral-2:123b-cloud` | `ollama` | execute | Post-review code implementation |

Edit `config/agents/*.yml` to change models, budgets, or system prompts without touching code.

### Token Usage File (`config/token_usage.yml`)

Auto-generated on first run. Tracks cumulative tokens per agent, per day, per sprint. Do not edit by hand.

---

## Commands

| Command | Key Options | Description |
|---------|-------------|-------------|
| `factory status` | `--config`, `--agents`, `--usage` | Connection, sprint, agents, token budget |
| `factory run` | `--issue N`, `--provider`, `--config`, `--agents` | Spawn agent for issue; opens draft MR |
| `factory review` | `--sprint N`, `--approve`, `--plan-next` | Sprint-review wiki page |
| `factory review-mr` | `--mr N`, `--post/--no-post`, `--second-review/--no-second-review` | Code review on MR |
| `factory refine` | `--mr N`, `--provider`, `--auto-merge` | Fix MR review feedback |
| `factory continue-mr` | `--mr N`, `--provider` | Resume incomplete implementation |
| `factory plan` | `--sprint N`, `--algorithm`, `--budget`, `--show` | Token-aware sprint planning |
| `factory update-wiki` | `--mr-id N` | Update wiki from MR diff |
| `factory fix-mr-pipeline` | `--mr-iid N`, `--retry` | Act on failed pipeline |
| `factory po` | `--issue N` | PO agent discussion for issue |
| `factory monitor` | `--interval`, `--config`, `--agents` | Autonomous polling loop |

### Provider Override

All commands that spawn agents accept `--provider`:

```bash
--provider anthropic   # Claude models  (requires ANTHROPIC_API_KEY)
--provider mistral     # Mistral models (requires MISTRAL_API_KEY)
--provider ollama      # Ollama endpoint (requires OLLAMA_BASE_URL)
--provider google      # Gemini models  (requires GOOGLE_API_KEY)
--provider stub        # Offline stub — no API key needed
```

---

## Token Budget

### Enforcement Modes

| Mode | Behaviour |
|------|-----------|
| `strict` | Blocks LLM calls when the agent is over budget; raises `BudgetExceededError` |
| `warn` | Allows calls over budget but logs a warning |
| `off` | No enforcement; tracking still occurs |

### How It Works

Each call through `LlmRouter.complete()` runs two phases:

1. **Pre-flight check** — `TokenBudgetManager.can_consume(agent, estimated_tokens)` estimates the call size from `len(system + prompt) // 3`. Returns `(False, reason)` in strict mode when over budget.
2. **Post-call recording** — `TokenBudgetManager.consume(agent, input_tokens, output_tokens)` records actual usage from the API response. Always runs (in a `finally` block) even if the provider raises.

Usage is persisted atomically to `config/token_usage.yml` using `os.replace()` + `filelock`.

### Viewing Usage

```bash
factory status                                    # shows token budget table in terminal
factory status --usage /path/to/token_usage.yml   # custom usage file path
```

### Resetting Usage

```bash
# Reset daily usage for the developer agent
python3 -m factory reset-budget developer --scope daily

# Reset sprint usage
python3 -m factory reset-budget reviewer --scope sprint
```

Or bypass enforcement entirely without resetting: `FACTORY_IGNORE_BUDGET=1 factory run --issue N`.

---

## Monitor Service

The monitor service runs a continuous polling loop implementing the full autonomous issue lifecycle:

1. **Issue gate** — only issues labelled `ready` are processed autonomously.
2. **Architect analysis** — Architect analyses the issue and posts a plan as a GitLab note.
3. **Developer implementation** — Developer implements, commits, and creates a draft MR.
4. **Review** — Code Reviewer + Meta Reviewer post a verdict comment on the MR.
5. **Refinement loop** — On `REQUEST CHANGES` or `BLOCK`: Developer refines; re-review triggered (up to 3 times).
6. **Auto-merge** — On `APPROVE` without draft status: MR is auto-merged and wiki is updated.
7. **Pipeline failures** — `fix-mr-pipeline` is called automatically on pipeline failure.

The WIP limit is controlled by `sprint.capacity.max_issues` in `factory.yml` (`-1` = unlimited).

```bash
factory monitor                  # 30-second polling (default)
factory monitor --interval 60    # 60-second polling
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `BudgetExceededError` | Agent over daily/sprint budget | Check `factory status`, increase limit in `factory.yml`, or set `FACTORY_IGNORE_BUDGET=1` |
| `GITLAB_TOKEN not set` | `.env` not loaded | Run `source .env` or check Docker/K8s env vars |
| `403 Forbidden` | Token scope too narrow | Recreate token with `api` scope |
| Empty diff in review | `changes()` API issue | Check `factory.yml` `gitlab.url`; ensure token has MR read access |
| `Budget tracking unavailable` in status | `token_usage.yml` missing or corrupt | Delete the file — it will be recreated on next run |
| Ollama `connection refused` | Ollama not running or wrong URL | Check `OLLAMA_BASE_URL`; run `ollama serve` locally |
| Ollama `401 Unauthorized` | Missing auth token | Set `OLLAMA_API_KEY` |
| Ollama timeout | Large model, slow inference | Increase `OLLAMA_TIMEOUT` (seconds) |
| Git clone fails in container | Missing auth | Set `GITLAB_TOKEN`; git ≥ 2.31 uses `GIT_CONFIG_*`, older git uses oauth2 URL fallback automatically |

### Verbose Logging

```bash
PYTHONPATH=. python3 -c "
import logging; logging.basicConfig(level=logging.DEBUG)
from factory.main import app; app()
" status
```

### Offline Testing

```bash
factory run --issue 1 --provider stub
factory review-mr --mr 1 --provider stub --no-post
```

---

## Security

- Never commit `.env` to version control (it is in `.gitignore`).
- Use GitLab CI/CD variables or Kubernetes secrets for deployment.
- The GitLab token only needs `api` scope — avoid `sudo` or `admin` scope.
- Token usage files contain only aggregate counts, not prompt content.
- `OLLAMA_API_KEY` is sent as a `Bearer` token over HTTPS only — use TLS for remote Ollama endpoints.
