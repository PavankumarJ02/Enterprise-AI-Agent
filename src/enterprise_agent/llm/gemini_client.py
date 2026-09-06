"""Google Gemini LLM Provider implementation using the official google-genai SDK."""

from collections.abc import AsyncIterator

from google import genai
from google.genai import errors, types
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from enterprise_agent.core.exceptions import (
    LLMAuthenticationError,
    LLMException,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from enterprise_agent.core.logging import get_logger
from enterprise_agent.llm.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    MessageRole,
    TokenUsage,
)

logger = get_logger(__name__)


def _is_transient_error(exc: BaseException) -> bool:
    """Check if exception is transient and eligible for retry."""
    return isinstance(exc, (LLMRateLimitError, LLMTimeoutError))


class GeminiLLMProvider(LLMProvider):
    """Google Gemini provider supporting Gemini 2.5 Flash, 1.5 Flash, and 1.5 Pro."""

    def __init__(
        self,
        api_key: str,
        default_model: str = "gemini-2.5-flash",
        default_temperature: float = 0.2,
        default_max_tokens: int = 1024,
        max_retries: int = 3,
    ) -> None:
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.max_retries = max_retries

        self._client = genai.Client(api_key=api_key)

    def _map_exception(self, err: Exception) -> LLMException:
        """Map google-genai SDK errors to domain exceptions."""
        if isinstance(err, errors.APIError):
            code = getattr(err, "code", None)
            msg = str(err.message or err)
            if code in {401, 403}:
                return LLMAuthenticationError(f"Gemini authentication failed (code {code}): {msg}")
            if code == 429:
                return LLMRateLimitError(f"Gemini rate limit exceeded (code 429): {msg}")
            if code in {408, 504}:
                return LLMTimeoutError(f"Gemini request timed out (code {code}): {msg}")
            if code is not None and code >= 500:
                return LLMProviderError(f"Gemini upstream server error (code {code}): {msg}")
            return LLMException(f"Gemini API error (code {code}): {msg}")

        if isinstance(err, (TimeoutError, errors.ClientError)):
            return LLMTimeoutError(f"Gemini client timeout: {err}")

        return LLMException(f"Unexpected error during Gemini invocation: {err}")

    def _convert_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[str | None, list[types.Content]]:
        """Convert internal ChatMessages to Gemini system instruction and Content turns."""
        system_instruction: str | None = None
        contents: list[types.Content] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_instruction = msg.content
            elif msg.role == MessageRole.USER:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=msg.content)],
                    )
                )
            elif msg.role == MessageRole.ASSISTANT:
                # Gemini models use 'model' role for assistant history
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=msg.content)],
                    )
                )

        # Gemini requires at least one user content item
        if not contents:
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="")],
                )
            )

        return system_instruction, contents

    async def _execute_generate(
        self,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
    ) -> LLMResponse:
        """Execute async generation with retry handling."""
        try:
            raw_response = await self._client.aio.models.generate_content(
                model=self.default_model,
                contents=contents,
                config=config,
            )

            content = raw_response.text or ""
            usage = TokenUsage()
            if raw_response.usage_metadata:
                usage = TokenUsage(
                    prompt_tokens=raw_response.usage_metadata.prompt_token_count or 0,
                    completion_tokens=raw_response.usage_metadata.candidates_token_count or 0,
                    total_tokens=raw_response.usage_metadata.total_token_count or 0,
                )

            finish_reason = "stop"
            if raw_response.candidates and len(raw_response.candidates) > 0:
                raw_finish = raw_response.candidates[0].finish_reason
                if raw_finish is not None:
                    finish_reason = str(raw_finish).lower()

            return LLMResponse(
                content=content,
                model=self.default_model,
                usage=usage,
                finish_reason=finish_reason,
            )
        except Exception as e:
            logger.error("Error generating content via Gemini: %s", e)
            raise self._map_exception(e) from e

    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate response via Gemini using official SDK with retries on transient errors."""
        temp = self.default_temperature if temperature is None else temperature
        tokens = self.default_max_tokens if max_tokens is None else max_tokens

        system_instruction, contents = self._convert_messages(messages)
        config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=tokens,
            system_instruction=system_instruction,
        )

        @retry(
            stop=stop_after_attempt(self.max_retries if self.max_retries > 0 else 1),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
            retry=retry_if_exception(_is_transient_error),
            reraise=True,
        )
        async def _call_with_retry() -> LLMResponse:
            return await self._execute_generate(contents, config)

        return await _call_with_retry()

    async def stream(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream chunks from Gemini in real time."""
        temp = self.default_temperature if temperature is None else temperature
        tokens = self.default_max_tokens if max_tokens is None else max_tokens

        system_instruction, contents = self._convert_messages(messages)
        config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=tokens,
            system_instruction=system_instruction,
        )

        try:
            stream_resp = await self._client.aio.models.generate_content_stream(
                model=self.default_model,
                contents=contents,
                config=config,
            )

            async for chunk in stream_resp:
                delta = chunk.text or ""
                finish: str | None = None
                if chunk.candidates and len(chunk.candidates) > 0:
                    cand_finish = chunk.candidates[0].finish_reason
                    if cand_finish:
                        finish = str(cand_finish).lower()
                if delta or finish:
                    yield LLMStreamChunk(delta_text=delta, finish_reason=finish)
        except Exception as e:
            logger.error("Error streaming content from Gemini: %s", e)
            raise self._map_exception(e) from e

    async def health_check(self) -> bool:
        """Verify Gemini connectivity by pinging the model metadata."""
        try:
            model_info = await self._client.aio.models.get(model=self.default_model)
            return model_info is not None
        except Exception as e:
            logger.warning("Gemini health check failed: %s", e)
            return False
