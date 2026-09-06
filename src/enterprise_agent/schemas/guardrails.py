"""Schemas and data models for enterprise guardrails and prompt-injection defense."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class GuardrailViolationType(StrEnum):
    """Categories of security and policy violations."""

    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    PII_LEAK = "pii_leak"
    TOXICITY = "toxicity"
    DELIMITER_ATTACK = "delimiter_attack"


class GuardrailAction(StrEnum):
    """Enforcement decision taken by guardrails."""

    PASS = "pass"
    SANITIZE = "sanitize"
    BLOCK = "block"


class GuardrailFinding(BaseModel):
    """Individual security finding or policy violation."""

    model_config = ConfigDict(frozen=True)

    violation_type: GuardrailViolationType = Field(
        ..., description="Category of detected violation."
    )
    severity: str = Field(
        ..., description="Severity level: 'low', 'medium', 'high', or 'critical'."
    )
    action: GuardrailAction = Field(
        ..., description="Recommended action: 'pass', 'sanitize', or 'block'."
    )
    message: str = Field(..., description="Human-readable explanation of why this was flagged.")
    matched_text: str | None = Field(
        default=None, description="Exact or truncated snippet that triggered the finding."
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score of detection (0.0 to 1.0)."
    )


class GuardrailCheckResult(BaseModel):
    """Aggregate result of guardrail inspection across all safety layers."""

    model_config = ConfigDict(frozen=True)

    passed: bool = Field(
        ..., description="True if input/output is safe to proceed without blocking."
    )
    action: GuardrailAction = Field(
        ..., description="Ultimate action to take: 'pass', 'sanitize', or 'block'."
    )
    findings: list[GuardrailFinding] = Field(
        default_factory=list, description="List of all detected security findings."
    )
    sanitized_text: str = Field(
        ..., description="Text after redaction and defanging has been applied."
    )
    latency_ms: float = Field(..., ge=0.0, description="Inspection latency in milliseconds.")


class GuardrailValidateRequest(BaseModel):
    """Request to validate an arbitrary text payload against guardrails."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=32000, description="Text string to evaluate.")
    check_input: bool = Field(
        default=True, description="True for user input inspection, False for model output."
    )
    check_pii: bool = Field(default=True, description="Whether to check and redact PII.")
    check_injection: bool = Field(
        default=True, description="Whether to scan for prompt injections and jailbreaks."
    )


class GuardrailRedactRequest(BaseModel):
    """Request to redact sensitive PII entities from text."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ..., min_length=1, max_length=32000, description="Text to scan for PII redaction."
    )


class GuardrailRedactResponse(BaseModel):
    """Response containing PII-redacted text and entity counts."""

    model_config = ConfigDict(frozen=True)

    original_text: str = Field(..., description="Original input text.")
    redacted_text: str = Field(..., description="Sanitized text with masked PII placeholders.")
    pii_types_detected: list[str] = Field(
        default_factory=list, description="Unique PII entity types detected."
    )
    count: int = Field(..., ge=0, description="Total number of PII occurrences redacted.")
