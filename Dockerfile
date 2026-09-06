# ==============================================================================
# Stage 1: Build Dependencies in Isolated Virtual Environment
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies required for compiling binary wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for clean layer separation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install python package dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt


# ==============================================================================
# Stage 2: Minimal, Hardened Production Runtime Image
# ==============================================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="Enterprise AI Engineering Team" \
      version="0.1.0" \
      description="Hardened Enterprise AI Knowledge & Decision Agent"

# Set optimal Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    APP_ENV="production"

# Install lightweight runtime utilities for container health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged non-root user (UID 10001) and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Create application directories with non-root ownership
RUN mkdir -p /app/data /app/logs && \
    chown -R appuser:appgroup /app

# Copy application source code and configuration
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup pyproject.toml /app/
COPY --chown=appuser:appgroup README.md /app/

# Switch to unprivileged non-root user
USER appuser

# Container-native health check against /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose standard service port
EXPOSE 8000

# Launch production ASGI server
CMD ["uvicorn", "enterprise_agent.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
