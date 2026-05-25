# SoftwareTeamFabrik Docker Image
# Self-contained deployment with all dependencies for running factory agents

# =============================================================================
# Stage 1: Build stage - install all dependencies
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY factory /app/factory
COPY config /app/config

# Create fab user first so we can install to their home directory
RUN useradd -m -u 1000 fab

# Switch to fab user before installing
USER fab

# Install the package with all extras (dev + llm dependencies)
# This includes: typer, rich, python-gitlab, pyyaml, python-dotenv, filelock
# plus dev extras: pytest, pytest-mock
# plus llm extras: anthropic, mistralai, requests
RUN pip install --no-cache-dir --user .[all]

# =============================================================================
# Stage 2: Runtime stage - minimal image with only what's needed
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

# git is required by the execution engine to commit and push code
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create fab user with consistent home directory
RUN useradd -m -u 1000 fab

# Copy installed Python packages from builder to fab's home
COPY --from=builder --chown=fab:fab /home/fab/.local /home/fab/.local

# Copy application code directly from build context
COPY --chown=fab:fab factory /app/factory
COPY --chown=fab:fab config /app/config
COPY --chown=fab:fab pyproject.toml /app/

# Make sure scripts in .local are usable
ENV PATH=/home/fab/.local/bin:$PATH

# Switch to non-root user
USER fab

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check: verify the CLI loads without errors
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD factory --help > /dev/null 2>&1 || exit 1

# Default: run the monitoring daemon.
# Override at runtime for one-off commands:
#   docker run --rm softwareteamfabrik factory run --issue 12
ENTRYPOINT ["factory"]
CMD ["monitor"]
