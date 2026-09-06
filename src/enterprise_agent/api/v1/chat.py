"""Chat completion API endpoint."""

import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse

from enterprise_agent.api.deps import get_llm
from enterprise_agent.core.logging import get_logger
from enterprise_agent.llm.base import ChatMessage, LLMProvider, MessageRole
from enterprise_agent.schemas.chat import ChatRequest, ChatResponse, TokenUsageResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Direct LLM Generation",
    description=(
        "Submit a conversational query or business question for direct LLM response generation."
    ),
)
async def chat_completion(
    request: ChatRequest,
    llm: LLMProvider = Depends(get_llm),
) -> ChatResponse:
    """Process incoming query and return structured LLM generation with metrics."""
    logger.info("Processing chat completion request (query length=%d chars)", len(request.query))

    messages: list[ChatMessage] = []

    # 1. Add system prompt if specified
    if request.system_prompt:
        messages.append(ChatMessage(role=MessageRole.SYSTEM, content=request.system_prompt))

    # 2. Append conversation history
    for item in request.history:
        messages.append(ChatMessage(role=MessageRole(item.role), content=item.content))

    # 3. Append current user query
    messages.append(ChatMessage(role=MessageRole.USER, content=request.query))

    # 4. Invoke LLM provider with latency tracking
    start_time = time.perf_counter()
    llm_output = await llm.generate(
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    logger.info(
        "Chat completion succeeded [model=%s, tokens=%d, latency=%.2fms]",
        llm_output.model,
        llm_output.usage.total_tokens,
        latency_ms,
    )

    return ChatResponse(
        answer=llm_output.content,
        model=llm_output.model,
        usage=TokenUsageResponse(
            prompt_tokens=llm_output.usage.prompt_tokens,
            completion_tokens=llm_output.usage.completion_tokens,
            total_tokens=llm_output.usage.total_tokens,
        ),
        latency_ms=latency_ms,
        status="success",
    )


@router.post(
    "/stream",
    status_code=status.HTTP_200_OK,
    summary="Streaming LLM Generation",
    description="Stream conversational tokens in real-time using Server-Sent Events (SSE).",
)
async def chat_stream(
    request: ChatRequest,
    llm: LLMProvider = Depends(get_llm),
) -> StreamingResponse:
    """Stream token chunks via Server-Sent Events (SSE)."""
    logger.info("Processing streaming chat request (query length=%d chars)", len(request.query))

    messages: list[ChatMessage] = []
    if request.system_prompt:
        messages.append(ChatMessage(role=MessageRole.SYSTEM, content=request.system_prompt))
    for item in request.history:
        messages.append(ChatMessage(role=MessageRole(item.role), content=item.content))
    messages.append(ChatMessage(role=MessageRole.USER, content=request.query))

    async def sse_event_generator() -> AsyncIterator[str]:
        try:
            async for chunk in llm.stream(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                payload = json.dumps(
                    {"delta": chunk.delta_text, "finish_reason": chunk.finish_reason}
                )
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error("Error during streaming generation: %s", e)
            error_payload = json.dumps({"error": "STREAM_ERROR", "message": str(e)})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
