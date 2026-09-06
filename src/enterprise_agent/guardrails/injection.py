"""Prompt injection, jailbreak, and system boundary breakout detector."""

import base64
import re
from dataclasses import dataclass

from enterprise_agent.schemas.guardrails import (
    GuardrailAction,
    GuardrailFinding,
    GuardrailViolationType,
)


@dataclass(frozen=True)
class InjectionScanResult:
    """Detailed result of prompt injection scanning."""

    is_injection: bool
    score: float
    findings: list[GuardrailFinding]


class PromptInjectionDetector:
    """Rule-based and heuristic scanner detecting direct and indirect prompt injection attempts."""

    # 1. Directive Overrides (Highest Risk: 0.95)
    _DIRECTIVE_OVERRIDES = [
        re.compile(
            r"\b(?:ignore|disregard|forget|drop)\s+(?:all\s+)?(?:previous|prior|above|system)\s+"
            r"(?:instructions|directives|prompts|rules|commands|context)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:start\s+over|reset\s+(?:your\s+)?system|clear\s+(?:all\s+)?instructions)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:your\s+new\s+(?:task|instructions?|role)\s+is\s+to\s+ignore)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:overwrite|override)\s+(?:the\s+)?(?:system|base)\s+(?:prompt|instructions)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:forget|clear|erase|delete|drop)\s+(?:everything|all)"
            r"(?:\s+(?:you\s+)?(?:were\s+told|before|prior))?\b",
            re.IGNORECASE,
        ),
    ]

    # 2. Jailbreak Personas & Modes (High Risk: 0.90)
    _JAILBREAK_PERSONAS = [
        re.compile(
            r"\b(?:do\s+anything\s+now|dan\s+mode|jailbreak(?:ed)?\s+mode)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:developer\s+mode\s+(?:enabled|activated|unlocked))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:roleplay\s+as\s+(?:an?\s+)?(?:unrestricted|unfiltered|evil|rogue|hacker))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:you\s+have\s+no\s+(?:rules|restrictions|ethics|safety\s+filters))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:bypass\s+(?:all\s+)?(?:safety|content|ethical)\s+filters?)\b",
            re.IGNORECASE,
        ),
    ]

    # 3. System Prompt & Secret Exfiltration (High Risk: 0.85)
    _EXFILTRATION_DIRECTIVES = [
        re.compile(
            r"\b(?:reveal|print|show|repeat|display|output)\s+(?:your\s+)?"
            r"(?:system\s+prompt|initial\s+instructions?|system\s+instructions?|secret\s+prompt)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:output\s+everything\s+(?:above|before\s+this))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:what\s+(?:are|were)\s+your\s+(?:original|system|hidden)\s+instructions\??)\b",
            re.IGNORECASE,
        ),
    ]

    # 4. Delimiter & Boundary Attacks (Medium/High Risk: 0.75)
    _DELIMITER_ATTACKS = [
        re.compile(
            r"<\s*/?\s*(?:system|instructions|context|admin|human|assistant)\s*>",
            re.IGNORECASE,
        ),
        re.compile(
            r"\[(?:ADMIN_OVERRIDE|SYSTEM_DIRECTIVE|INST|SYS)\]",
            re.IGNORECASE,
        ),
        re.compile(
            r"===\s*(?:SYSTEM\s+PROMPT|NEW\s+SESSION|ADMIN\s+MODE)\s*===",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:Human|Assistant|System):\s*[\r\n]+",
            re.IGNORECASE,
        ),
    ]

    def __init__(self, threshold: float = 0.70) -> None:
        self.threshold = threshold

    def _check_base64_payload(self, text: str) -> GuardrailFinding | None:
        """Inspect potential base64 substrings for encoded injection payloads."""
        # Find continuous base64-like sequences with length >= 16
        b64_matches = re.findall(
            r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{16,}={0,2}(?![A-Za-z0-9+/=])",
            text,
        )
        for cand in b64_matches:
            try:
                decoded = base64.b64decode(cand, validate=True).decode("utf-8", errors="ignore")
                # Check decoded payload for injection patterns
                for pat in self._DIRECTIVE_OVERRIDES + self._JAILBREAK_PERSONAS:
                    if pat.search(decoded):
                        return GuardrailFinding(
                            violation_type=GuardrailViolationType.PROMPT_INJECTION,
                            severity="critical",
                            action=GuardrailAction.BLOCK,
                            message="Base64 obfuscated prompt injection payload detected.",
                            matched_text=cand[:30] + "...",
                            confidence=0.98,
                        )
            except Exception:
                continue
        return None

    def scan(self, text: str) -> InjectionScanResult:
        """Scan input text against directive overrides, jailbreaks, and delimiter breakouts."""
        findings: list[GuardrailFinding] = []
        max_score = 0.0

        # 1. Directive Overrides
        for pat in self._DIRECTIVE_OVERRIDES:
            m = pat.search(text)
            if m:
                score = 0.95
                max_score = max(max_score, score)
                findings.append(
                    GuardrailFinding(
                        violation_type=GuardrailViolationType.PROMPT_INJECTION,
                        severity="critical",
                        action=GuardrailAction.BLOCK,
                        message="Direct instruction override attempt detected.",
                        matched_text=m.group(),
                        confidence=score,
                    )
                )
                break

        # 2. Jailbreaks & Personas
        for pat in self._JAILBREAK_PERSONAS:
            m = pat.search(text)
            if m:
                score = 0.90
                max_score = max(max_score, score)
                findings.append(
                    GuardrailFinding(
                        violation_type=GuardrailViolationType.JAILBREAK,
                        severity="high",
                        action=GuardrailAction.BLOCK,
                        message="Jailbreak persona or security filter bypass pattern detected.",
                        matched_text=m.group(),
                        confidence=score,
                    )
                )
                break

        # 3. Exfiltration Directives
        for pat in self._EXFILTRATION_DIRECTIVES:
            m = pat.search(text)
            if m:
                score = 0.85
                max_score = max(max_score, score)
                findings.append(
                    GuardrailFinding(
                        violation_type=GuardrailViolationType.SYSTEM_PROMPT_LEAK,
                        severity="high",
                        action=GuardrailAction.BLOCK,
                        message="System prompt exfiltration attempt detected.",
                        matched_text=m.group(),
                        confidence=score,
                    )
                )
                break

        # 4. Delimiter & Boundary Attacks
        for pat in self._DELIMITER_ATTACKS:
            m = pat.search(text)
            if m:
                score = 0.75
                max_score = max(max_score, score)
                findings.append(
                    GuardrailFinding(
                        violation_type=GuardrailViolationType.DELIMITER_ATTACK,
                        severity="medium",
                        action=(
                            GuardrailAction.BLOCK
                            if score >= self.threshold
                            else GuardrailAction.SANITIZE
                        ),
                        message="System delimiter or conversation boundary injection detected.",
                        matched_text=m.group(),
                        confidence=score,
                    )
                )
                break

        # 5. Base64 payload scan
        b64_finding = self._check_base64_payload(text)
        if b64_finding:
            max_score = max(max_score, b64_finding.confidence)
            findings.append(b64_finding)

        is_injection = max_score >= self.threshold
        return InjectionScanResult(
            is_injection=is_injection,
            score=max_score,
            findings=findings,
        )
