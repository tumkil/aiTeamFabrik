"""
Wiki Manager for SoftwareTeamFabrik.

Manages wiki templates and formatting for consistent documentation.
"""

import datetime


class WikiManager:
    """Manage wiki templates and formatting."""
    
    def __init__(self):
        """Initialize WikiManager with templates."""
        self.templates = {
            "architecture_overview": self._get_architecture_overview_template(),
            "components_core": self._get_components_core_template(),
            "components_api_gateway": self._get_components_api_gateway_template(),
            "components_data_pipeline": self._get_components_data_pipeline_template(),
            "components_monitoring": self._get_components_monitoring_template(),
            "development_process": self._get_development_process_template(),
            "adrs": self._get_adrs_template(),
            "deployment": self._get_deployment_template(),
            "faq": self._get_faq_template(),
        }
    
    def get_template(self, template_name: str) -> str:
        """Get a template by name.
        
        Args:
            template_name: Name of the template to retrieve
            
        Returns:
            Template content as string
            
        Raises:
            KeyError: If template doesn't exist
        """
        return self.templates[template_name]
    
    def format_template(self, template_name: str, **kwargs) -> str:
        """Format a template with variables.
        
        Args:
            template_name: Name of the template to format
            **kwargs: Variables to use in formatting
            
        Returns:
            Formatted template content
        """
        template = self.get_template(template_name)
        
        # Add date if not provided
        if "date" not in kwargs:
            kwargs["date"] = datetime.date.today().isoformat()
        
        # Replace the date placeholder manually to avoid KeyError
        # when kwargs values contain curly braces
        template = template.replace("{date}", kwargs["date"])
        
        # Append remaining kwargs as raw text
        for key, value in kwargs.items():
            if key != "date":
                template += f"\n\n{key}: {value}"
        
        return template
    
    def _get_architecture_overview_template(self) -> str:
        return """# Architecture Overview

*Last updated: {date}*

SoftwareTeamFabrik is an autonomous AI-driven software development factory that
connects to GitLab to manage issues, merge requests, and wiki content.

## High-Level Architecture

```mermaid
flowchart TD
    A[GitLab] -->|API| B[SoftwareTeamFabrik]
    B --> C[factory CLI]
    C --> D[Status]
    C --> E[Run]
    C --> F[Review]
    C --> G[Review-MR]
    C --> H[Refine]
    C --> I[Monitor]
```

## Core Components

- **GitLab Client**: Handles all interactions with GitLab API
- **Scrum Engine**: Manages sprint state and advancement
- **Competence Manager**: Loads and manages agent profiles
- **Token Budget Manager**: Tracks and enforces API token usage
- **LLM Router**: Dispatches requests to appropriate LLM providers
- **Execution Engine**: Manages tool-call loops for agents
- **Orchestrator**: Spawns, polls, and retires agent tasks

## Data Flow

1. User invokes a factory command (e.g., `factory run --issue 42`)
2. Command loads configuration from `config/factory.yml`
3. GitLab Client authenticates and fetches issue details
4. Competence Manager selects the appropriate agent profile
5. Token Budget Manager checks available token budget
6. LLM Router dispatches the request to the configured provider
7. Execution Engine manages the tool-call loop
8. Results are written back to GitLab (MRs, comments, wiki updates)

## Key Files

- `config/factory.yml`: Main configuration file
- `config/agents/*.yml`: Agent profile definitions
- `config/token_usage.yml`: Token consumption tracking (auto-generated)

## Integration Points

- GitLab Issues: Source of work items
- GitLab Merge Requests: Code changes and reviews
- GitLab Wiki: Documentation and sprint reviews
- GitLab Milestones: Sprint tracking
"""
    
    def _get_components_core_template(self) -> str:
        return """# Core Component

*Last updated: {date}*

The Core component contains the main business logic and orchestration services
for SoftwareTeamFabrik.

## Sub-components

### Competence Manager

Manages agent profiles and their capabilities:

- Loads YAML profiles from `config/agents/`
- Matches issues to appropriate agents based on labels and capabilities
- Tracks agent state (active, idle, retired)

### Token Budget Manager

Enforces token usage limits:

- **Modes**: strict (block), warn (log), off (disabled)
- **Scopes**: daily and sprint budgets per agent
- **Tracking**: Persists usage to `config/token_usage.yml`
- **Concurrency**: Uses file locking for safe multi-process access

### LLM Router

Dispatches LLM requests:

- Supports multiple providers (Anthropic, Mistral, Stub)
- Handles provider-specific API formats
- Manages API keys and authentication
- Implements retry logic for transient failures

### Execution Engine

Manages agent execution:

- Tool-call loop implementation
- Context management between calls
- Error handling and recovery
- Timeout management

### Orchestrator

Coordinates agent workflows:

- Spawns new agent tasks
- Polls for completion
- Retires completed tasks
- Manages task queues

### Scrum Engine

Handles sprint management:

- Tracks current sprint state
- Calculates velocity metrics
- Manages milestone transitions
- Updates configuration on sprint advancement
"""
    
    def _get_components_api_gateway_template(self) -> str:
        return """# API Gateway Component

*Last updated: {date}*

The API Gateway handles all external communications for SoftwareTeamFabrik.

## Responsibilities

- GitLab API integration
- LLM provider APIs
- Webhook handling (future)
- Rate limiting
- Authentication management

## GitLab Integration

The GitLab Client (`factory.adapters.gitlab_client`) provides:

- Issue management (list, create, update)
- Merge request operations
- Wiki page management
- Milestone tracking
- Project metadata access

## LLM Provider Integration

Supported providers:

- **Anthropic**: Claude models (Opus, Haiku)
- **Mistral**: Mistral models
- **Stub**: Offline testing provider

Each provider adapter handles:
- API key management
- Request/response formatting
- Error handling
- Rate limiting
"""
    
    def _get_components_data_pipeline_template(self) -> str:
        return """# Data Pipeline Component

*Last updated: {date}*

The Data Pipeline manages information flow through SoftwareTeamFabrik.

## Data Sources

- GitLab API (issues, MRs, wiki, milestones)
- Configuration files (YAML)
- LLM responses (JSON)
- Token usage logs

## Processing Stages

1. **Ingestion**: Fetch data from GitLab
2. **Transformation**: Convert to internal models
3. **Enrichment**: Add derived metrics (velocity, budgets)
4. **Persistence**: Write back to GitLab or config files
5. **Presentation**: Format for CLI output or wiki pages

## Key Data Models

- `Issue`: GitLab issue with metadata
- `MergeRequest`: MR with diff and review state
- `AgentProfile`: Agent capabilities and configuration
- `SprintState`: Current sprint information
- `TokenUsage`: Token consumption tracking
"""
    
    def _get_components_monitoring_template(self) -> str:
        return """# Monitoring Component

*Last updated: {date}*

The Monitoring component provides observability and operational insights.

## Features

### Token Budget Tracking

- Real-time usage monitoring
- Per-agent and global budgets
- Alerts when approaching limits
- Historical usage reports

### Task Monitoring

- Agent task status
- Execution time tracking
- Success/failure rates
- Queue lengths

### Logging

- Structured logs to stdout
- Error logging with stack traces
- Debug mode for detailed output

### Metrics

- Sprint velocity
- Issue throughput
- Review cycle time
- Token efficiency (tokens per issue)

## Integration

- CLI status command
- Wiki reports
- Future: Prometheus/Grafana integration
"""
    
    def _get_development_process_template(self) -> str:
        return """# Development Process

*Last updated: {date}*

This page documents the development workflow using SoftwareTeamFabrik.

## Typical Workflow

### 1. Issue Creation

- Create issues in GitLab with appropriate labels
- Use sprint milestone for planning
- Add detailed description and acceptance criteria

### 2. Agent Assignment

Run the appropriate factory command:

```bash
# For implementation tasks
factory run --issue 42

# For code reviews
factory review-mr --mr 12

# For refinements based on feedback
factory refine --mr 12
```

### 3. Execution

- Agent analyzes the issue
- Generates implementation plan
- Writes code (Developer agent)
- Creates merge request
- Reviews code (Reviewer agent)

### 4. Review and Refinement

- Human review of MR
- Use `factory refine` for automated fixes
- Iterate until approval

### 5. Merge and Close

- Merge approved MRs
- Close completed issues
- Update wiki documentation

## Command Reference

| Command | Purpose | Example |
|---------|---------|---------|
| `status` | Show current state | `factory status` |
| `run` | Implement issue | `factory run --issue 42` |
| `review-mr` | Review MR | `factory review-mr --mr 12` |
| `refine` | Fix review feedback | `factory refine --mr 12` |
| `review` | Sprint review | `factory review --sprint 3` |
| `monitor` | Auto-polling | `factory monitor` |

## Configuration

Edit `config/factory.yml` for:
- GitLab connection details
- Sprint configuration
- Token budget settings

Edit `config/agents/*.yml` for:
- Agent profiles
- Model selection
- System prompts
- Capabilities

## Best Practices

- Start with `--provider stub` for offline testing
- Monitor token usage with `factory status`
- Review generated wiki pages after sprint reviews
- Backup `config/token_usage.yml` periodically
"""
    
    def _get_adrs_template(self) -> str:
        return """# Architecture Decision Records (ADRs)

*Last updated: {date}*

This page tracks significant architecture decisions for SoftwareTeamFabrik.

## ADR Format

Each ADR should include:

1. **Title**: Brief descriptive name
2. **Status**: Proposed, Accepted, Deprecated, Superseded
3. **Context**: The problem being addressed
4. **Decision**: The chosen solution
5. **Consequences**: Implications and trade-offs
6. **Date**: When the decision was made

## Current ADRs

### ADR-001: Multi-Provider LLM Support

**Status**: Accepted
**Date**: 2024-05-01

**Context**: Need to support multiple LLM providers (Anthropic, Mistral) with fallback capability.

**Decision**: Implement a provider-agnostic LLM Router with pluggable adapters for each provider.

**Consequences**:
- Pros: Easy to add new providers, provider-specific logic isolated
- Cons: Slight overhead in request routing

### ADR-002: Token Budget Enforcement

**Status**: Accepted
**Date**: 2024-05-05

**Context**: Need to prevent runaway API costs from LLM usage.

**Decision**: Implement two-phase budget checking (pre-flight estimate + post-call accounting) with configurable enforcement modes.

**Consequences**:
- Pros: Prevents cost overruns, provides visibility into usage
- Cons: Adds complexity to LLM calling code

### ADR-003: File-Based Configuration

**Status**: Accepted
**Date**: 2024-05-10

**Context**: Need flexible configuration without code changes.

**Decision**: Use YAML configuration files for factory settings, agent profiles, and token tracking.

**Consequences**:
- Pros: No code changes for configuration, easy to version control
- Cons: Need to handle file I/O and locking carefully

## Adding New ADRs

1. Create a new section with ADR number (increment from last)
2. Follow the standard format
3. Update this wiki page
4. Reference the ADR in relevant code comments
"""
    
    def _get_deployment_template(self) -> str:
        return """# Deployment

*Last updated: {date}*

This page documents deployment options for SoftwareTeamFabrik.

## Installation

### Prerequisites

- Python 3.9+
- Git
- GitLab account with API access
- LLM provider API keys (Anthropic and/or Mistral)

### Setup

```bash
# Clone repository
git clone https://gitlab.com/your-group/softwareteamfabrik.git
cd softwareteamfabrik

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev,llm]"

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Test connectivity
factory status
```

## Configuration Files

### factory.yml

Main configuration file with:
- GitLab connection details
- Sprint configuration
- Token budget settings

### agents/*.yml

Agent profile definitions with:
- Agent name and display name
- LLM model and provider
- Execution mode (plan/execute)
- System prompt
- Capabilities

## Running Modes

### Interactive

Run individual commands as needed:

```bash
factory run --issue 42
factory review-mr --mr 12
```

### Monitoring

Run the autonomous monitor:

```bash
factory monitor
```

### CI/CD Integration

Example GitLab CI job:

```yaml
stages:
  - review

review_mr:
  stage: review
  script:
    - factory review-mr --mr $CI_MERGE_REQUEST_IID --no-post
  only:
    - merge_requests
```

## Upgrading

```bash
# Pull latest changes
git pull origin main

# Reinstall
pip install -e ".[dev,llm]" --upgrade

# Check for configuration changes
# Compare config/factory.yml.example with your config
```

## Troubleshooting

### Connection Issues

- Verify `GITLAB_TOKEN` in .env
- Check GitLab URL in factory.yml
- Test with `factory status`

### Token Budget Errors

- Check `config/token_usage.yml`
- Adjust budgets in factory.yml
- Use `--provider stub` for testing

### Permission Errors

- Ensure GitLab token has `api` scope
- Check project visibility settings
- Verify user has maintainer/reporter role
"""
    
    def _get_faq_template(self) -> str:
        return """# Frequently Asked Questions

*Last updated: {date}*

## General

### What is SoftwareTeamFabrik?

SoftwareTeamFabrik is an autonomous AI-driven software development factory that connects to GitLab to manage issues, merge requests, and wiki content using LLM-powered agents.

### What does "Fabrik" mean?

"Fabrik" is German for "factory", reflecting the tool's purpose as a software development factory.

## Setup

### How do I install SoftwareTeamFabrik?

See the [Deployment](Deployment) page for detailed installation instructions.

### What API keys do I need?

- GitLab personal access token (required)
- Anthropic API key (for Claude models)
- Mistral API key (for Mistral models)

You can start with just the GitLab token and use the stub provider for testing.

### Can I run it without LLM API keys?

Yes! Use the `--provider stub` flag with any command that requires LLM calls:

```bash
factory run --issue 42 --provider stub
```

## Usage

### How do I assign an issue to an agent?

Use the `factory run` command:

```bash
factory run --issue 42
```

The system will automatically select the appropriate agent based on issue labels and content.

### How do I review a merge request?

Use the `factory review-mr` command:

```bash
factory review-mr --mr 12
```

### How do I handle review feedback?

Use the `factory refine` command to automatically address review comments:

```bash
factory refine --mr 12
```

### How do I monitor token usage?

Use the `factory status` command to see current token consumption and budget status.

## Configuration

### How do I add a new agent?

Create a new YAML file in `config/agents/` with the agent's profile. See existing files for examples.

### How do I change token budgets?

Edit the `token_budget` section in `config/factory.yml` and adjust the daily and sprint limits.

### How do I switch LLM providers?

Either:
1. Edit the agent's YAML file to change the provider, or
2. Use the `--provider` flag with commands:

```bash
factory run --issue 42 --provider mistral
```

## Troubleshooting

### I get "Budget exceeded" errors

- Check your current usage with `factory status`
- Increase budgets in `config/factory.yml`
- Wait for daily budget to reset
- Use `--provider stub` for testing without token consumption

### Commands hang or time out

- Check your internet connection
- Verify API keys are correct
- Try with `--provider stub` to test locally
- Increase timeout settings if needed

### Wiki pages aren't updating

- Verify GitLab token has wiki write permissions
- Check that the project has a wiki enabled
- Run with `--verbose` to see detailed error messages

## Development

### How do I contribute?

See CONTRIBUTING.md in the repository for contribution guidelines.

### How do I run tests?

```bash
python3 -m pytest tests/ -v
```

### How do I add a new feature?

1. Create a GitLab issue describing the feature
2. Create a feature branch
3. Implement the feature
4. Add tests
5. Open a merge request
"""
