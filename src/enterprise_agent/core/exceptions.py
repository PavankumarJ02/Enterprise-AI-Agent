"""Domain-specific application exceptions."""


class AppException(Exception):
    """Base class for all enterprise application exceptions."""

    def __init__(self, message: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(AppException):
    """Raised when application configuration is missing, invalid, or conflicting."""

    pass


class LLMException(AppException):
    """Base exception for all LLM client and inference failures."""

    pass


class LLMAuthenticationError(LLMException):
    """Raised when LLM provider returns 401/403 credentials error."""

    pass


class LLMRateLimitError(LLMException):
    """Raised when LLM provider rejects request due to rate limits or quota."""

    pass


class LLMTimeoutError(LLMException):
    """Raised when an LLM provider request exceeds configured timeout."""

    pass


class LLMProviderError(LLMException):
    """Raised when an unrecoverable error occurs in upstream provider."""

    pass
