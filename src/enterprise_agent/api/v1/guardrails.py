"""REST API endpoints for Enterprise Guardrails and Prompt-Injection Defense."""

from fastapi import APIRouter, Depends, status

from enterprise_agent.api.deps import get_guardrails_service
from enterprise_agent.guardrails.service import GuardrailsService
from enterprise_agent.schemas.guardrails import (
    GuardrailCheckResult,
    GuardrailRedactRequest,
    GuardrailRedactResponse,
    GuardrailValidateRequest,
)

router = APIRouter(prefix="/guardrails", tags=["Guardrails & Security"])


@router.post(
    "/validate",
    response_model=GuardrailCheckResult,
    status_code=status.HTTP_200_OK,
    summary="Validate text against input/output security guardrails",
    description=(
        "Scans input or output text for prompt injections, jailbreak personas, "
        "system boundary breakouts, canary prompt leakage, and PII entities."
    ),
)
async def validate_guardrails(
    request: GuardrailValidateRequest,
    service: GuardrailsService = Depends(get_guardrails_service),
) -> GuardrailCheckResult:
    """Evaluate text through the enterprise guardrails defense layers."""
    if request.check_input:
        return service.inspect_input(
            text=request.text,
            check_pii=request.check_pii,
            check_injection=request.check_injection,
        )
    return service.inspect_output(
        text=request.text,
        check_pii=request.check_pii,
    )


@router.post(
    "/redact",
    response_model=GuardrailRedactResponse,
    status_code=status.HTTP_200_OK,
    summary="Redact sensitive PII entities from text",
    description=(
        "Detects and masks SSNs, credit cards, emails, phone numbers, and API keys "
        "with typed placeholder tokens."
    ),
)
async def redact_pii(
    request: GuardrailRedactRequest,
    service: GuardrailsService = Depends(get_guardrails_service),
) -> GuardrailRedactResponse:
    """Mask sensitive PII entities within the provided text."""
    redacted_text, detected_types, count = service.redact_pii_standalone(request.text)
    return GuardrailRedactResponse(
        original_text=request.text,
        redacted_text=redacted_text,
        pii_types_detected=detected_types,
        count=count,
    )
