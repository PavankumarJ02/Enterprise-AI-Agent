"""Application configuration management using Pydantic Settings.

Adheres to 12-Factor App principles by extracting configuration from environment variables.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration object loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -------------------------------------------------------------------------
    # Core Application Configuration
    # -------------------------------------------------------------------------
    app_name: str = Field(
        default="Enterprise AI Knowledge & Decision Agent",
        description="Public display name for the service.",
    )
    app_env: Literal["development", "staging", "production", "testing"] = Field(
        default="development",
        description="Current deployment environment.",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application semantic version.",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode and verbose logs.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Standard log level filter.",
    )
    host: str = Field(
        default="0.0.0.0",
        description="HTTP server bind host.",
    )
    port: int = Field(
        default=8000,
        description="HTTP server bind port.",
    )

    # -------------------------------------------------------------------------
    # LLM Service Configuration
    # -------------------------------------------------------------------------
    llm_provider: Literal["mock", "openai", "groq", "ollama", "azure"] = Field(
        default="mock",
        description="LLM provider implementation to instantiate.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model identifier to invoke.",
    )
    llm_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key for the external LLM provider.",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible base API URL.",
    )
    llm_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM generation.",
    )
    llm_max_tokens: int = Field(
        default=1024,
        gt=0,
        le=32768,
        description="Maximum tokens allowed in generation response.",
    )
    llm_request_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Timeout in seconds for outbound LLM API requests.",
    )
    llm_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts on transient network or rate-limit errors.",
    )

    @property
    def is_production(self) -> bool:
        """Helper to check if running in production."""
        return self.app_env == "production"

    @property
    def is_testing(self) -> bool:
        """Helper to check if running under test suite."""
        return self.app_env == "testing"


@lru_cache
def get_settings() -> Settings:
    """Provide a cached singleton instance of Settings.

    Can be overridden in tests via FastAPI's `app.dependency_overrides`.
    """
    return Settings()
