"""Structured logging configuration."""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure standardized logging for root and application loggers with credential redaction."""
    from enterprise_agent.security.secrets import LoggingRedactionFilter

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(LoggingRedactionFilter())

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        handlers=[handler],
        force=True,
    )
    # Suppress overly chatty HTTP client logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get a scoped logger instance."""
    return logging.getLogger(name)
