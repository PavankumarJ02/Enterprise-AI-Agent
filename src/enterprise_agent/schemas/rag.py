"""Pydantic schemas for Retrieval-Augmented Generation (RAG) query and stream requests/responses."""

from typing import Any

from pydantic import BaseModel, Field

from enterprise_agent.schemas.chat import TokenUsageResponse


class RetrievedSourceChunk(BaseModel):
    """Source citation representing a document chunk used in RAG generation."""

    chunk_id: str = Field(..., description="Unique chunk identifier.")
    document_id: str = Field(..., description="Parent document identifier.")
    source: str = Field(default="", description="Source filename or origin identifier.")
    chunk_index: int = Field(..., description="Position index of the chunk in document.")
    content: str = Field(..., description="Excerpt text content used as context.")
    score: float = Field(..., description="Vector similarity score.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Associated chunk metadata (e.g. page, section header).",
    )


class Citation(BaseModel):
    """Specific document citation tied to a claim or sentence in the generated answer."""

    source_index: int = Field(..., description="1-based index corresponding to [Source N] tag.")
    document_id: str = Field(..., description="Referenced document identifier.")
    chunk_id: str = Field(..., description="Referenced chunk identifier.")
    source: str = Field(default="", description="Source filename or origin.")
    page_numbers: list[int] | None = Field(
        default=None, description="Page number(s) if applicable from chunk metadata."
    )
    quote_snippet: str = Field(
        ..., description="Most relevant excerpt or supporting span from the source chunk."
    )
    claim_text: str = Field(
        ..., description="The claim or sentence in the generated answer making this citation."
    )
    is_grounded: bool = Field(
        ..., description="Whether the claim has strong factual grounding in the chunk."
    )
    grounding_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence metric of factual entailment."
    )


class GroundingEvaluation(BaseModel):
    """Audit summary of grounding verification and faithfulness for the generated answer."""

    faithfulness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio of verified grounded claims to total claims made (0.0 to 1.0).",
    )
    grounding_status: str = Field(
        ...,
        description=(
            "Verification status ('verified', 'partially_grounded', "
            "'unverified', 'insufficient_context')."
        ),
    )
    total_claims: int = Field(..., ge=0, description="Total number of discrete claims analyzed.")
    grounded_claims: int = Field(
        ..., ge=0, description="Number of claims verified against retrieved sources."
    )
    citations: list[Citation] = Field(
        default_factory=list, description="Verified citations per claim."
    )
    unattributed_claims: list[str] = Field(
        default_factory=list,
        description="Claims without valid source attribution or citing non-existent sources.",
    )


class RAGQueryRequest(BaseModel):
    """Payload requesting grounded answer synthesis using enterprise documentation."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural language question to answer using retrieved documentation.",
        examples=["What is the company policy for travel reimbursement?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of relevant document chunks to retrieve.",
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum vector similarity threshold for included chunks.",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata filters to restrict retrieval (e.g. department, source).",
        examples=[{"department": "Finance"}],
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional custom system instructions overriding default guardrails.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="LLM generation temperature (defaults to settings if omitted).",
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        le=32768,
        description="Maximum tokens allowed in synthesized answer.",
    )
    max_context_tokens: int = Field(
        default=4000,
        gt=100,
        le=32000,
        description="Maximum token budget allocated for injected context passages.",
    )
    verify_grounding: bool = Field(
        default=True,
        description=(
            "Whether to perform automated citation extraction and factual grounding verification."
        ),
    )


class RAGQueryResponse(BaseModel):
    """Synthesized response with source citations and execution telemetry."""

    query: str = Field(..., description="Original user question.")
    answer: str = Field(..., description="Synthesized grounded answer.")
    sources: list[RetrievedSourceChunk] = Field(
        default_factory=list,
        description="Document chunks retrieved and utilized to ground the response.",
    )
    model: str = Field(..., description="Identifier of the LLM model that generated the answer.")
    usage: TokenUsageResponse = Field(..., description="Token consumption metrics.")
    latency_ms: float = Field(..., description="End-to-end execution latency in milliseconds.")
    status: str = Field(default="success", description="Status of the RAG operation.")
    grounding: GroundingEvaluation | None = Field(
        default=None,
        description="Automated grounding audit and faithfulness evaluation.",
    )
