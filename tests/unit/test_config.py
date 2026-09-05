"""Unit tests for configuration management."""

import pytest
from pydantic import ValidationError

from enterprise_agent.config.settings import Settings


def test_default_settings() -> None:
    """Verify default settings instantiate correctly."""
    settings = Settings()
    assert settings.app_name == "Enterprise AI Knowledge & Decision Agent"
    assert settings.app_env in {"development", "testing"}
    assert settings.llm_provider in {"mock", "openai", "groq", "ollama", "azure"}
    assert settings.llm_temperature >= 0.0
    assert settings.llm_max_tokens > 0


def test_environment_helpers() -> None:
    """Test environment helper boolean properties."""
    prod_settings = Settings(app_env="production")
    assert prod_settings.is_production is True
    assert prod_settings.is_testing is False

    test_settings = Settings(app_env="testing")
    assert test_settings.is_production is False
    assert test_settings.is_testing is True


def test_invalid_temperature_bounds() -> None:
    """Ensure temperature out of [0.0, 2.0] is rejected by Pydantic."""
    with pytest.raises(ValidationError):
        Settings(llm_temperature=2.5)

    with pytest.raises(ValidationError):
        Settings(llm_temperature=-0.1)


def test_invalid_max_tokens() -> None:
    """Ensure max_tokens must be positive."""
    with pytest.raises(ValidationError):
        Settings(llm_max_tokens=0)

    with pytest.raises(ValidationError):
        Settings(llm_max_tokens=-50)
