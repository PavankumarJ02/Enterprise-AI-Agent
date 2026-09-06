"""Personally Identifiable Information (PII) detection and redaction engine."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PIIMatch:
    """Represents a single detected PII entity."""

    entity_type: str
    start: int
    end: int
    raw_value: str
    replacement: str


def _is_luhn_valid(card_digits: str) -> bool:
    """Validate numeric string using the Luhn checksum algorithm."""
    digits = [int(c) for c in card_digits if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for idx, digit in enumerate(reverse_digits):
        if idx % 2 == 1:
            doubled = digit * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += digit
    return checksum % 10 == 0


class PIIRedactor:
    """High-accuracy regex and algorithmic PII detector and sanitizer."""

    # US Social Security Number: 123-45-6789 or 123 45 6789 (rejects 000/666 and 00/0000)
    _SSN_PATTERN = re.compile(r"\b(?!000|666)\d{3}[-\s](?!00)\d{2}[-\s](?!0000)\d{4}\b")

    # Email Addresses: standard RFC-compliant subset
    _EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}\b")

    # North American & International Phone Numbers
    _PHONE_PATTERN = re.compile(
        r"(?:\+?1[-.\s]?)?(?:\([2-9]\d{2}\)|[2-9]\d{2})[-.\s][2-9]\d{2}[-.\s]\d{4}\b"
    )

    # API Keys & Secrets: OpenAI, GitHub, AWS
    _API_KEY_PATTERNS = [
        ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
        ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b")),
        ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ]

    # Credit Cards (Visa, MasterCard, Amex, Discover, Diners, JCB)
    _CREDIT_CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

    def detect(self, text: str) -> list[PIIMatch]:
        """Scan text and return all detected PII entities sorted by start index."""
        matches: list[PIIMatch] = []

        # 1. SSN
        for m in self._SSN_PATTERN.finditer(text):
            matches.append(
                PIIMatch(
                    entity_type="SSN",
                    start=m.start(),
                    end=m.end(),
                    raw_value=m.group(),
                    replacement="[REDACTED_SSN]",
                )
            )

        # 2. Email
        for m in self._EMAIL_PATTERN.finditer(text):
            matches.append(
                PIIMatch(
                    entity_type="EMAIL",
                    start=m.start(),
                    end=m.end(),
                    raw_value=m.group(),
                    replacement="[REDACTED_EMAIL]",
                )
            )

        # 3. Phone
        for m in self._PHONE_PATTERN.finditer(text):
            matches.append(
                PIIMatch(
                    entity_type="PHONE",
                    start=m.start(),
                    end=m.end(),
                    raw_value=m.group(),
                    replacement="[REDACTED_PHONE]",
                )
            )

        # 4. API Keys
        for _key_type, pattern in self._API_KEY_PATTERNS:
            for m in pattern.finditer(text):
                matches.append(
                    PIIMatch(
                        entity_type="API_KEY",
                        start=m.start(),
                        end=m.end(),
                        raw_value=m.group(),
                        replacement="[REDACTED_API_KEY]",
                    )
                )

        # 5. Credit Cards with Luhn validation
        for m in self._CREDIT_CARD_CANDIDATE.finditer(text):
            raw = m.group()
            digits = re.sub(r"\D", "", raw)
            if _is_luhn_valid(digits):
                # Verify not overlapping with an already identified SSN or phone
                overlap = any(
                    not (m.end() <= existing.start or m.start() >= existing.end)
                    for existing in matches
                )
                if not overlap:
                    matches.append(
                        PIIMatch(
                            entity_type="CREDIT_CARD",
                            start=m.start(),
                            end=m.end(),
                            raw_value=raw,
                            replacement="[REDACTED_CREDIT_CARD]",
                        )
                    )

        # Sort matches by start position ascending
        matches.sort(key=lambda x: x.start)
        return matches

    def redact(self, text: str) -> tuple[str, list[PIIMatch]]:
        """Redact all PII entities from text and return sanitized string with matches."""
        matches = self.detect(text)
        if not matches:
            return text, []

        # Replace non-overlapping matches from right to left to preserve offsets
        sanitized = list(text)
        # Filter overlapping matches (prefer longer match or earlier start)
        filtered_matches: list[PIIMatch] = []
        last_end = -1
        for match in matches:
            if match.start >= last_end:
                filtered_matches.append(match)
                last_end = match.end

        for match in reversed(filtered_matches):
            sanitized[match.start : match.end] = list(match.replacement)

        return "".join(sanitized), filtered_matches
