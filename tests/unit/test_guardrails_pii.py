"""Unit tests for PII detection and redaction engine."""

from enterprise_agent.guardrails.pii import PIIRedactor, _is_luhn_valid


def test_luhn_validation() -> None:
    """Verify Luhn checksum calculation for credit card numbers."""
    # Standard test Visa
    assert _is_luhn_valid("4532015112830366") is True
    # Corrupted last digit
    assert _is_luhn_valid("4532015112830367") is False
    # Too short
    assert _is_luhn_valid("12345678") is False


def test_pii_redact_ssn() -> None:
    """Verify SSN detection and redaction."""
    redactor = PIIRedactor()
    text = "Employee John Doe has SSN 123-45-6789 on record."
    redacted, matches = redactor.redact(text)
    assert "[REDACTED_SSN]" in redacted
    assert "123-45-6789" not in redacted
    assert len(matches) == 1
    assert matches[0].entity_type == "SSN"


def test_pii_ignore_invalid_ssn() -> None:
    """Verify invalid area/group SSN prefixes are not falsely matched."""
    redactor = PIIRedactor()
    text = "The part number is 000-12-3456 and internal code is 666-45-1234."
    redacted, matches = redactor.redact(text)
    assert redacted == text
    assert len(matches) == 0


def test_pii_redact_email() -> None:
    """Verify email address detection and masking."""
    redactor = PIIRedactor()
    text = "Contact security at sec-alert@enterprise.corp.com or bob@company.io immediately."
    redacted, matches = redactor.redact(text)
    assert "[REDACTED_EMAIL]" in redacted
    assert "sec-alert@enterprise.corp.com" not in redacted
    assert "bob@company.io" not in redacted
    assert len(matches) == 2
    assert all(m.entity_type == "EMAIL" for m in matches)


def test_pii_redact_phone() -> None:
    """Verify US and international phone number masking."""
    redactor = PIIRedactor()
    text = "Call desk at (415) 555-0199 or direct line 212-555-1234."
    redacted, matches = redactor.redact(text)
    assert "[REDACTED_PHONE]" in redacted
    assert "415" not in redacted
    assert "212" not in redacted
    assert len(matches) == 2


def test_pii_redact_api_keys() -> None:
    """Verify OpenAI, GitHub, and AWS API key detection."""
    redactor = PIIRedactor()
    text = (
        "Leaked keys: openai sk-proj12345678901234567890abcdef "
        "and github ghp_123456789012345678901234567890123456 "
        "and aws AKIAIOSFODNN7EXAMPLE."
    )
    redacted, matches = redactor.redact(text)
    assert "[REDACTED_API_KEY]" in redacted
    assert "sk-proj12345678901234567890abcdef" not in redacted
    assert "ghp_123456789012345678901234567890123456" not in redacted
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted
    assert len(matches) == 3


def test_pii_redact_credit_card() -> None:
    """Verify Luhn-valid credit card masking."""
    redactor = PIIRedactor()
    text = "Payment card: 4532 0151 1283 0366 expires next year."
    redacted, matches = redactor.redact(text)
    assert "[REDACTED_CREDIT_CARD]" in redacted
    assert "4532 0151 1283 0366" not in redacted
    assert len(matches) == 1
    assert matches[0].entity_type == "CREDIT_CARD"


def test_pii_clean_text_unchanged() -> None:
    """Verify clean business text is completely unaltered."""
    redactor = PIIRedactor()
    text = "The quarterly financial report was published on September 15 with 150 participants."
    redacted, matches = redactor.redact(text)
    assert redacted == text
    assert len(matches) == 0
