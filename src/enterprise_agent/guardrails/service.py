"""Enterprise Guardrails Service coordinating input sanitization and output protection."""

import time
from typing import Any

from enterprise_agent.config.settings import Settings, get_settings
from enterprise_agent.core.logging import get_logger
from enterprise_agent.guardrails.canary import CanaryTokenManager
from enterprise_agent.guardrails.injection import PromptInjectionDetector
from enterprise_agent.guardrails.pii import PIIRedactor
from enterprise_agent.schemas.guardrails import (
    GuardrailAction,
    GuardrailCheckResult,
    GuardrailFinding,
    GuardrailViolationType,
)

logger = get_logger(__name__)


class GuardrailSecurityError(Exception):
    """Raised when an operation is blocked by enterprise security guardrails."""

    def __init__(self, message: str, findings: list[GuardrailFinding]) -> None:
        super().__init__(message)
        self.findings = findings


class GuardrailsService:
    """Orchestrates multi-layer defense for untrusted inputs and generated outputs."""

    def __init__(
        self,
        settings: Settings | None = None,
        pii_redactor: PIIRedactor | None = None,
        injection_detector: PromptInjectionDetector | None = None,
        canary_manager: CanaryTokenManager | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.pii_redactor = pii_redactor or PIIRedactor()
        self.injection_detector = injection_detector or PromptInjectionDetector(
            threshold=self.settings.guardrails_injection_threshold
        )
        self.canary_manager = canary_manager or CanaryTokenManager(
            default_secret=self.settings.guardrails_canary_secret
        )

    def inspect_input(
        self,
        text: str,
        check_pii: bool = True,
        check_injection: bool = True,
    ) -> GuardrailCheckResult:
        """Inspect and sanitize user query before passing to downstream pipelines."""
        start_time = time.perf_counter()
        findings: list[GuardrailFinding] = []
        sanitized_text = text

        if not self.settings.guardrails_enabled:
            return GuardrailCheckResult(
                passed=True,
                action=GuardrailAction.PASS,
                findings=[],
                sanitized_text=text,
                latency_ms=0.0,
            )

        # 1. Check prompt injection & jailbreaks
        if check_injection:
            scan_res = self.injection_detector.scan(text)
            if scan_res.findings:
                findings.extend(scan_res.findings)

        # 2. Check and redact PII
        if check_pii and self.settings.guardrails_redact_pii:
            redacted_out, pii_matches = self.pii_redactor.redact(sanitized_text)
            sanitized_text = redacted_out
            for match in pii_matches:
                findings.append(
                    GuardrailFinding(
                        violation_type=GuardrailViolationType.PII_LEAK,
                        severity="medium",
                        action=GuardrailAction.SANITIZE,
                        message=f"Detected sensitive {match.entity_type} entity.",
                        matched_text=match.raw_value,
                        confidence=1.0,
                    )
                )

        # Determine ultimate action
        has_block = any(f.action == GuardrailAction.BLOCK for f in findings)
        if has_block and self.settings.guardrails_block_injections:
            action = GuardrailAction.BLOCK
            passed = False
        elif findings:
            action = GuardrailAction.SANITIZE
            passed = True
        else:
            action = GuardrailAction.PASS
            passed = True

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
        return GuardrailCheckResult(
            passed=passed,
            action=action,
            findings=findings,
            sanitized_text=sanitized_text,
            latency_ms=latency_ms,
        )

    def inspect_output(
        self,
        text: str,
        canary: str | None = None,
        check_pii: bool = True,
        check_canary: bool = True,
    ) -> GuardrailCheckResult:
        """Verify model-generated output for prompt leakage or unmasked sensitive data."""
        start_time = time.perf_counter()
        findings: list[GuardrailFinding] = []
        sanitized_text = text

        if not self.settings.guardrails_enabled:
            return GuardrailCheckResult(
                passed=True,
                action=GuardrailAction.PASS,
                findings=[],
                sanitized_text=text,
                latency_ms=0.0,
            )

        # 1. Check canary token leakage
        if check_canary and self.settings.guardrails_canary_detection:
            if self.canary_manager.check_leakage(text, canary):
                findings.append(
                    GuardrailFinding(
                        violation_type=GuardrailViolationType.SYSTEM_PROMPT_LEAK,
                        severity="critical",
                        action=GuardrailAction.BLOCK,
                        message="System prompt canary token leak detected in generated response.",
                        matched_text="[CONFIDENTIAL_INTERNAL_CANARY]",
                        confidence=1.0,
                    )
                )
                sanitized_text = (
                    "I cannot provide this response because it contains internal system metadata."
                )

        # 2. Check and redact PII in output
        if check_pii and self.settings.guardrails_redact_pii:
            redacted_out, pii_matches = self.pii_redactor.redact(sanitized_text)
            sanitized_text = redacted_out
            for match in pii_matches:
                findings.append(
                    GuardrailFinding(
                        violation_type=GuardrailViolationType.PII_LEAK,
                        severity="medium",
                        action=GuardrailAction.SANITIZE,
                        message=f"Redacted sensitive {match.entity_type} from generated output.",
                        matched_text=match.raw_value,
                        confidence=1.0,
                    )
                )

        # Determine ultimate action
        has_block = any(f.action == GuardrailAction.BLOCK for f in findings)
        action = (
            GuardrailAction.BLOCK
            if has_block
            else (GuardrailAction.SANITIZE if findings else GuardrailAction.PASS)
        )
        passed = not has_block

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
        return GuardrailCheckResult(
            passed=passed,
            action=action,
            findings=findings,
            sanitized_text=sanitized_text,
            latency_ms=latency_ms,
        )

    def redact_pii_standalone(self, text: str) -> tuple[str, list[str], int]:
        """Redact PII from text and return sanitized text, unique types, and total count."""
        redacted_text, matches = self.pii_redactor.redact(text)
        unique_types = sorted({m.entity_type for m in matches})
        return redacted_text, unique_types, len(matches)

    def validate_or_raise(self, text: str, **kwargs: Any) -> str:
        """Validate input text and return sanitized version, or raise on violation."""
        result = self.inspect_input(text, **kwargs)
        if not result.passed:
            first_msg = result.findings[0].message if result.findings else "Security violation"
            raise GuardrailSecurityError(
                f"Security guardrail violation: {first_msg}",
                findings=result.findings,
            )
        return result.sanitized_text
