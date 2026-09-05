"""Pytest fixtures and configuration."""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from enterprise_agent.config.settings import Settings
from enterprise_agent.llm.mock_client import MockLLMProvider
from enterprise_agent.main import create_application


@pytest.fixture
def test_settings() -> Settings:
    """Fixture providing isolated test settings."""
    return Settings(
        app_name="Enterprise AI Agent Test",
        app_env="testing",
        app_version="0.1.0-test",
        debug=True,
        log_level="DEBUG",
        llm_provider="mock",
        llm_model="mock-gpt-4o-mini",
        llm_api_key=SecretStr("test-key"),
        llm_base_url="https://api.openai.com/v1",
        llm_temperature=0.0,
        llm_max_tokens=256,
        llm_request_timeout_seconds=5.0,
        llm_max_retries=1,
    )


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Fixture providing deterministic mock LLM provider."""
    return MockLLMProvider(
        default_response="Mock answer for unit testing.",
        model_name="mock-test-model",
    )


@pytest.fixture
def test_client(
    test_settings: Settings, mock_llm: MockLLMProvider
) -> Generator[TestClient, None, None]:
    """Fixture providing FastAPI TestClient with overridden dependencies."""
    app = create_application(settings=test_settings)

    # Dependency overrides
    from enterprise_agent.api.deps import get_app_settings, get_llm

    app.dependency_overrides[get_app_settings] = lambda: test_settings
    app.dependency_overrides[get_llm] = lambda: mock_llm

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
