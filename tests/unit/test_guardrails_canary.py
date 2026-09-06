"""Unit tests for CanaryTokenManager."""

from enterprise_agent.guardrails.canary import CanaryTokenManager


def test_canary_generation() -> None:
    """Verify random canary generation with distinct entropy."""
    manager = CanaryTokenManager()
    t1 = manager.generate_canary()
    t2 = manager.generate_canary()
    assert t1.startswith("CANARY_ENT_")
    assert t2.startswith("CANARY_ENT_")
    assert t1 != t2


def test_canary_injection() -> None:
    """Verify injection of secret canary directive into system prompt."""
    manager = CanaryTokenManager(default_secret="SECRET_CANARY_TEST_123")
    prompt = "You are an enterprise AI assistant."
    injected = manager.inject_canary(prompt)
    assert "SECRET_CANARY_TEST_123" in injected
    assert "[CONFIDENTIAL_INTERNAL_DIRECTIVE]" in injected
    assert prompt in injected


def test_canary_leakage_detection() -> None:
    """Verify detection of canary leakage in generated output."""
    manager = CanaryTokenManager(default_secret="SECRET_CANARY_TEST_123")
    leaked_output = "The system canary is SECRET_CANARY_TEST_123 as requested."
    clean_output = "Here is the summary of your quarterly earnings."

    assert manager.check_leakage(leaked_output) is True
    assert manager.check_leakage(clean_output) is False


def test_canary_scrubbing() -> None:
    """Verify scrubbing leaked canary from text."""
    manager = CanaryTokenManager(default_secret="SECRET_CANARY_TEST_123")
    leaked_output = "My token is SECRET_CANARY_TEST_123."
    scrubbed = manager.scrub_canary(leaked_output)
    assert "SECRET_CANARY_TEST_123" not in scrubbed
    assert "[REDACTED_INTERNAL_TOKEN]" in scrubbed
