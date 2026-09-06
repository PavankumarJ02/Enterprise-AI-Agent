"""Unit tests validating Docker and Docker Compose production infrastructure."""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).parent.parent.parent


def test_dockerfile_structure_and_security() -> None:
    """Verify Dockerfile exists and enforces multi-stage and non-root security standards."""
    dockerfile_path = WORKSPACE_ROOT / "Dockerfile"
    assert dockerfile_path.is_file(), "Dockerfile must exist at repository root."

    content = dockerfile_path.read_text(encoding="utf-8")

    # 1. Verify multi-stage build structure
    assert "FROM python:3.11-slim AS builder" in content, "Must use builder stage."
    assert "FROM python:3.11-slim AS runtime" in content, "Must use runtime stage."
    assert "COPY --from=builder /opt/venv /opt/venv" in content, (
        "Must copy virtualenv from builder."
    )

    # 2. Verify non-root user hardening
    assert "groupadd -g 10001 appgroup" in content, "Must define appgroup."
    assert "useradd -u 10001" in content, "Must define unprivileged appuser."
    assert "USER appuser" in content, "Must switch to non-root appuser before runtime."

    # 3. Verify healthcheck and container execution
    assert "HEALTHCHECK" in content, "Must include container-native HEALTHCHECK."
    assert "curl -f http://localhost:8000/health" in content, (
        "Healthcheck must probe /health endpoint."
    )
    assert "EXPOSE 8000" in content, "Must declare exposed port 8000."
    assert "CMD [" in content, "Must define default startup CMD."
    assert "uvicorn" in content, "Must launch uvicorn server."

    # 4. Verify Python environment optimization
    assert "PYTHONDONTWRITEBYTECODE=1" in content
    assert "PYTHONUNBUFFERED=1" in content


def test_dockerignore_exclusions() -> None:
    """Verify .dockerignore prevents sensitive or unnecessary files from entering context."""
    dockerignore_path = WORKSPACE_ROOT / ".dockerignore"
    assert dockerignore_path.is_file(), ".dockerignore must exist at repository root."

    content = dockerignore_path.read_text(encoding="utf-8")
    lines = [
        line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")
    ]

    # Critical exclusions
    assert ".git" in lines
    assert ".venv/" in lines or ".venv" in lines
    assert "__pycache__/" in lines or "__pycache__" in lines
    assert ".pytest_cache/" in lines or ".pytest_cache" in lines
    assert ".mypy_cache/" in lines or ".mypy_cache" in lines
    assert ".env" in lines


def test_docker_compose_specification() -> None:
    """Verify docker-compose.yml defines multi-container app and qdrant topology."""
    compose_path = WORKSPACE_ROOT / "docker-compose.yml"
    assert compose_path.is_file(), "docker-compose.yml must exist at repository root."

    content = compose_path.read_text(encoding="utf-8")

    # Verify services
    assert "app:" in content, "Must define 'app' service."
    assert "qdrant:" in content, "Must define 'qdrant' service."

    # Verify container names and ports
    assert "enterprise-agent-app" in content
    assert "enterprise-agent-qdrant" in content
    assert '"8000:8000"' in content
    assert '"6333:6333"' in content

    # Verify healthchecks and service dependency
    assert "service_healthy" in content, "App must depend on qdrant being healthy."
    assert "/health" in content, "App must define healthcheck."
    assert "/readyz" in content, "Qdrant must define healthcheck."

    # Verify persistent storage volumes and networks
    assert "qdrant_storage:" in content, "Must declare qdrant_storage volume."
    assert "agent_data:" in content, "Must declare agent_data volume."
    assert "enterprise-agent-net" in content, "Must define isolated custom bridge network."


def test_docker_env_example_template() -> None:
    """Verify .env.docker.example contains necessary production variables."""
    env_example_path = WORKSPACE_ROOT / ".env.docker.example"
    assert env_example_path.is_file(), ".env.docker.example must exist at repository root."

    content = env_example_path.read_text(encoding="utf-8")

    assert "APP_ENV=production" in content
    assert "QDRANT_URL=http://qdrant:6333" in content
    assert "SQLITE_DB_PATH=/app/data/enterprise.db" in content
    assert "EXPERIMENTS_DB_PATH=/app/data/experiments.db" in content
    assert "RATE_LIMIT_ENABLED=true" in content
    assert "OBSERVABILITY_ENABLED=true" in content
