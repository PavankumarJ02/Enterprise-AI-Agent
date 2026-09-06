"""RAG orchestration service coordinating retrieval, context assembly, and LLM synthesis."""

import json
import time
from collections.abc import AsyncIterator

from enterprise_agent.core.logging import get_logger
from enterprise_agent.llm.base import LLMProvider
from enterprise_agent.observability.tracer import Tracer
from enterprise_agent.rag.citations import GroundingVerifier
from enterprise_agent.rag.context import RAGContextAssembler
from enterprise_agent.rag.prompts import RAGPromptBuilder
from enterprise_agent.schemas.chat import TokenUsageResponse
from enterprise_agent.schemas.observability import SpanKind
from enterprise_agent.schemas.rag import (
    RAGQueryRequest,
    RAGQueryResponse,
)
from enterprise_agent.vectorstore.service import VectorSearchService

logger = get_logger(__name__)

INSUFFICIENT_CONTEXT_MESSAGE = (
    "Based on the provided documentation, I do not have enough information to answer this question."
)


class RAGService:
    """RAG orchestrator unifying vector retrieval, context budgeting, and LLM generation."""

    def __init__(
        self,
        vector_service: VectorSearchService,
        llm: LLMProvider,
        context_assembler: RAGContextAssembler | None = None,
        prompt_builder: RAGPromptBuilder | None = None,
        grounding_verifier: GroundingVerifier | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.vector_service = vector_service
        self.llm = llm
        self.context_assembler = context_assembler or RAGContextAssembler()
        self.prompt_builder = prompt_builder or RAGPromptBuilder()
        self.grounding_verifier = grounding_verifier or GroundingVerifier()
        self.tracer = tracer

    @property
    def model_name(self) -> str:
        """Return model identifier of the underlying LLM provider."""
        return str(getattr(self.llm, "default_model", getattr(self.llm, "model_name", "unknown")))

    async def query(self, request: RAGQueryRequest) -> RAGQueryResponse:
        """Execute synchronous grounded RAG answer synthesis."""
        start_time = time.perf_counter()
        logger.info("Executing RAG query: '%s' (top_k=%d)", request.query, request.top_k)

        if self.tracer:
            async with self.tracer.async_span(
                "rag.query",
                kind=SpanKind.SERVER,
                attributes={"query": request.query, "top_k": request.top_k},
            ) as span:
                res = await self._execute_query(request, start_time)
                span.set_attribute("rag.status", res.status)
                span.set_attribute("rag.sources_count", len(res.sources))
                span.set_genai_metrics(
                    model=res.model,
                    prompt_tokens=res.usage.prompt_tokens,
                    completion_tokens=res.usage.completion_tokens,
                    temperature=request.temperature,
                )
                return res

        return await self._execute_query(request, start_time)

    async def _execute_query(self, request: RAGQueryRequest, start_time: float) -> RAGQueryResponse:
        """Internal execution flow for RAG query."""
        # 1. Semantic Search
        if self.tracer:
            async with self.tracer.async_span("rag.retrieval", kind=SpanKind.RETRIEVER) as r_span:
                search_res = await self.vector_service.semantic_search(
                    query=request.query,
                    top_k=request.top_k,
                    filters=request.filters,
                    min_score=request.min_score,
                )
                r_span.set_attribute("results_count", len(search_res.results))
        else:
            search_res = await self.vector_service.semantic_search(
                query=request.query,
                top_k=request.top_k,
                filters=request.filters,
                min_score=request.min_score,
            )

        # 2. Short-circuit if no relevant documentation found
        if not search_res.results:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info("Zero matching chunks found. Returning insufficient context directly.")
            grounding_eval = (
                self.grounding_verifier.verify(INSUFFICIENT_CONTEXT_MESSAGE, [])
                if request.verify_grounding
                else None
            )
            return RAGQueryResponse(
                query=request.query,
                answer=INSUFFICIENT_CONTEXT_MESSAGE,
                sources=[],
                model=self.model_name,
                usage=TokenUsageResponse(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                latency_ms=latency_ms,
                status="insufficient_context",
                grounding=grounding_eval,
            )

        # 3. Assemble Context under Token Ceiling
        context, sources = self.context_assembler.assemble(
            results=search_res.results,
            max_context_tokens=request.max_context_tokens,
        )

        # 4. Construct Guardrailed Prompts
        messages = self.prompt_builder.build_messages(
            query=request.query,
            context=context,
            system_prompt=request.system_prompt,
        )

        # 5. Generate Grounded Synthesis
        if self.tracer:
            async with self.tracer.async_span("rag.synthesis", kind=SpanKind.LLM) as s_span:
                llm_res = await self.llm.generate(
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
                s_span.set_genai_metrics(
                    model=llm_res.model,
                    prompt_tokens=llm_res.usage.prompt_tokens,
                    completion_tokens=llm_res.usage.completion_tokens,
                    temperature=request.temperature,
                )
        else:
            llm_res = await self.llm.generate(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

        # 6. Verify Citations & Grounding
        grounding_eval = None
        if request.verify_grounding:
            grounding_eval = self.grounding_verifier.verify(llm_res.content, sources)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "RAG generation complete [model=%s, sources=%d, tokens=%d, "
            "latency=%.2fms, grounded=%s]",
            llm_res.model,
            len(sources),
            llm_res.usage.total_tokens,
            latency_ms,
            grounding_eval.grounding_status if grounding_eval else "unverified",
        )

        return RAGQueryResponse(
            query=request.query,
            answer=llm_res.content,
            sources=sources,
            model=llm_res.model,
            usage=TokenUsageResponse(
                prompt_tokens=llm_res.usage.prompt_tokens,
                completion_tokens=llm_res.usage.completion_tokens,
                total_tokens=llm_res.usage.total_tokens,
            ),
            latency_ms=latency_ms,
            status="success",
            grounding=grounding_eval,
        )

    async def query_stream(self, request: RAGQueryRequest) -> AsyncIterator[str]:
        """Stream RAG response tokens via Server-Sent Events (SSE)."""
        logger.info("Executing streaming RAG query: '%s'", request.query)

        # 1. Semantic Search
        search_res = await self.vector_service.semantic_search(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            min_score=request.min_score,
        )

        # 2. Short-circuit if no relevant documentation found
        if not search_res.results:
            yield "event: sources\ndata: []\n\n"
            no_ctx_payload = json.dumps(
                {"delta": INSUFFICIENT_CONTEXT_MESSAGE, "finish_reason": "stop"}
            )
            yield f"data: {no_ctx_payload}\n\n"
            if request.verify_grounding:
                grounding_eval = self.grounding_verifier.verify(INSUFFICIENT_CONTEXT_MESSAGE, [])
                yield f"event: grounding\ndata: {json.dumps(grounding_eval.model_dump())}\n\n"
            yield "data: [DONE]\n\n"
            return

        # 3. Assemble Context
        context, sources = self.context_assembler.assemble(
            results=search_res.results,
            max_context_tokens=request.max_context_tokens,
        )

        # 4. Emit sources upfront in an initial SSE event
        sources_payload = [s.model_dump() for s in sources]
        yield f"event: sources\ndata: {json.dumps(sources_payload)}\n\n"

        # 5. Construct Prompts
        messages = self.prompt_builder.build_messages(
            query=request.query,
            context=context,
            system_prompt=request.system_prompt,
        )

        # 6. Stream Answer Tokens and Accumulate Text
        accumulated_tokens: list[str] = []
        try:
            async for chunk in self.llm.stream(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                if chunk.delta_text:
                    accumulated_tokens.append(chunk.delta_text)
                    payload = json.dumps(
                        {"delta": chunk.delta_text, "finish_reason": chunk.finish_reason}
                    )
                    yield f"data: {payload}\n\n"

            # 7. Emit Grounding Evaluation Event if requested
            if request.verify_grounding:
                full_answer = "".join(accumulated_tokens)
                grounding_eval = self.grounding_verifier.verify(full_answer, sources)
                yield f"event: grounding\ndata: {json.dumps(grounding_eval.model_dump())}\n\n"

        except Exception as e:
            logger.error("Error during streaming RAG generation: %s", e)
            err_payload = json.dumps({"error": "STREAM_ERROR", "message": str(e)})
            yield f"data: {err_payload}\n\n"

        yield "data: [DONE]\n\n"
