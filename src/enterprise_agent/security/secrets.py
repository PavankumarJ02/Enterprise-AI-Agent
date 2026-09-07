"""Secrets management abstractions and telemetry credential redaction filters."""

import logging
import os
import re
from typing import Any

from pydantic import SecretStr

from enterprise_agent.security.base import SecretsManager


class EnvSecretsManager(SecretsManager):
    """Resolves enterprise secrets from system environment variables or configured overrides."""

    def __init__(self, overrides: dict[str, str] | None = None) -> None:
        self._overrides = overrides or {}

    def get_secret(self, key: str, default: str | None = None) -> SecretStr | None:
        """Retrieve a secret by name, returning SecretStr wrapping the value."""
        if key in self._overrides:
            override_val = self._overrides[key]
            return SecretStr(override_val) if override_val else None

        env_val: str | None = os.environ.get(key, default)
        return SecretStr(env_val) if env_val is not None else None

    def has_secret(self, key: str) -> bool:
        """Verify whether a secret exists and has a non-empty value."""
        secret = self.get_secret(key)
        return secret is not None and len(secret.get_secret_value().strip()) > 0


class TelemetryRedactor:
    """Regex-based sanitization engine masking credentials across logs, traces, and telemetry."""

    # Comprehensive compiled patterns for sensitive credentials
    PATTERNS: list[tuple[re.Pattern[str], str]] = [
        # OpenAI / Third-party API keys
        (re.compile(r"sk-[a-zA-Z0-9_-]{20,}"), "[REDACTED_API_KEY]"),
        # Enterprise Agent API keys
        (re.compile(r"ea_[a-zA-Z0-9_-]{20,}"), "[REDACTED_AGENT_KEY]"),
        # Bearer tokens in headers
        (re.compile(r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{20,}"), "Bearer [REDACTED_BEARER_TOKEN]"),
        # GitHub Personal Access Tokens
        (re.compile(r"ghp_[a-zA-Z0-9]{36}"), "[REDACTED_GITHUB_TOKEN]"),
        # AWS Access Keys
        (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_KEY]"),
        # Database connection strings with passwords (e.g. postgresql://user:pass@host)
        (
            re.compile(r"((?:postgresql|mysql|sqlite)://[^:]+:)([^@]+)(@)"),
            r"\g<1>[REDACTED_PASSWORD]\g<3>",
        ),
        # Key-value secret assignments in configuration strings or queries
        (
            re.compile(
                r"(?i)(password|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?([^\s'\"]{6,})['\"]?"
            ),
            r"\g<1>=[REDACTED_CREDENTIAL]",
        ),
    ]

    @classmethod
    def redact_text(cls, text: str) -> str:
        """Redact known credential patterns from a plain string."""
        if not text:
            return text

        redacted = text
        for pattern, replacement in cls.PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        return redacted

    @classmethod
    def redact_data(cls, data: Any) -> Any:
        """Recursively redact strings within nested dictionaries, lists, or tuples."""
        if isinstance(data, str):
            return cls.redact_text(data)
        if isinstance(data, dict):
            return {
                k: cls.redact_data(v)
                if not any(
                    s in str(k).lower()
                    for s in ("password", "secret", "api_key", "token", "authorization")
                )
                else "[REDACTED_SENSITIVE_VALUE]"
                for k, v in data.items()
            }
        if isinstance(data, list):
            return [cls.redact_data(item) for item in data]
        if isinstance(data, tuple):
            return tuple(cls.redact_data(item) for item in data)
        return data


class LoggingRedactionFilter(logging.Filter):
    """Python logging filter preventing confidential credentials from entering logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Sanitize record.msg and record.args in-place prior to formatter emission."""
        if isinstance(record.msg, str):
            record.msg = TelemetryRedactor.redact_text(record.msg)

        if record.args:
            if isinstance(record.args, dict):
                record.args = TelemetryRedactor.redact_data(record.args)
            elif isinstance(record.args, tuple):
                record.args = tuple(TelemetryRedactor.redact_data(arg) for arg in record.args)

        return True
