"""Pydantic schemas for semantic vector search and indexing requests/responses."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class SemanticSearchRequest(BaseModel):
    """Payload for executing dense semantic similarity retrieval."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural language query string.",
        examples=["What is the company policy on remote work equipment reimbursement?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of top-matching chunks to return.",
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score threshold (0.0 to 1.0).",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional payload filters to restrict search (e.g. document_id, source).",
        examples=[{"source": "employee_handbook.pdf"}],
    )


class SearchResultItem(BaseModel):
    """Single matching chunk retrieved from the vector database."""

    chunk_id: str = Field(..., description="Unique chunk identifier.")
    document_id: str = Field(..., description="ID of the parent document.")
    content: str = Field(..., description="Extracted text chunk content.")
    score: float = Field(..., description="Similarity score (cosine similarity between 0 and 1).")
    chunk_index: int = Field(..., description="Zero-indexed position in parent document.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Associated chunk metadata (e.g. page_number, section_header).",
    )


class SemanticSearchResponse(BaseModel):
    """Response payload containing ranked search results."""

    query: str = Field(..., description="Original search query.")
    total_results: int = Field(..., description="Number of results matching criteria.")
    results: list[SearchResultItem] = Field(
        default_factory=list,
        description="Ranked list of chunk results ordered by relevance.",
    )


class IndexDocumentResponse(BaseModel):
    """Response payload returned when document chunks are vectorized and indexed."""

    document_id: str = Field(..., description="ID of indexed document.")
    chunks_indexed: int = Field(..., description="Number of chunks stored in vector database.")
    status: str = Field(default="success", description="Indexing status.")
    message: str = Field(..., description="Human-readable status summary.")


class SparseSearchRequest(BaseModel):
    """Payload for executing sparse BM25 keyword search."""

    query: str = Field(
        ...,
        min_length=1,
        description="Keyword query string.",
        examples=["CVE-2024-1234"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of keyword-matching chunks to return.",
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum normalized BM25 score threshold.",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata filters.",
    )


class SparseSearchResponse(BaseModel):
    """Response payload for sparse BM25 search."""

    query: str = Field(..., description="Original keyword query.")
    total_results: int = Field(..., description="Count of matching chunks.")
    results: list[SearchResultItem] = Field(
        default_factory=list,
        description="Ranked list of chunk results ordered by BM25 relevance.",
    )


class HybridSearchResultItem(SearchResultItem):
    """Search result enriched with both dense and sparse retrieval telemetry."""

    dense_score: float | None = Field(
        default=None,
        description="Dense vector cosine similarity score if retrieved by dense search.",
    )
    sparse_score: float | None = Field(
        default=None,
        description="Sparse BM25 relevance score if retrieved by keyword search.",
    )
    dense_rank: int | None = Field(
        default=None,
        description="1-based rank position in dense retrieval list.",
    )
    sparse_rank: int | None = Field(
        default=None,
        description="1-based rank position in sparse retrieval list.",
    )
    combined_score: float = Field(
        ...,
        description="Fused score resulting from Reciprocal Rank Fusion (RRF) or linear weighting.",
    )


class HybridSearchRequest(BaseModel):
    """Payload for executing hybrid search combining dense vectors and sparse BM25."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural language or keyword query string.",
        examples=["Section 4.1 travel reimbursement guidelines"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of hybrid-ranked chunks to return.",
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum combined fusion score threshold.",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata filters applied to both retrievers.",
    )
    fusion_method: str = Field(
        default="rrf",
        description="Rank fusion strategy: 'rrf' (Reciprocal Rank Fusion) or 'linear'.",
    )
    dense_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relative weight assigned to dense semantic search.",
    )
    sparse_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relative weight assigned to sparse BM25 search.",
    )
    rrf_k: int = Field(
        default=60,
        gt=0,
        le=500,
        description="Smoothing constant k for Reciprocal Rank Fusion.",
    )


class HybridSearchResponse(BaseModel):
    """Response payload containing hybrid fused search results."""

    query: str = Field(..., description="Original query.")
    total_results: int = Field(..., description="Number of results matching criteria.")
    fusion_method: str = Field(..., description="Fusion method utilized ('rrf' or 'linear').")
    results: list[HybridSearchResultItem] = Field(
        default_factory=list,
        description="Ranked list of chunk results ordered by fused score.",
    )


class RerankResultItem(SearchResultItem):
    """Search result enriched with cross-encoder relevance score and rank transitions."""

    rerank_score: float = Field(..., description="Cross-encoder relevance score.")
    initial_rank: int = Field(
        ...,
        description="Original rank position from Stage 1 retrieval (1-indexed).",
    )
    final_rank: int = Field(
        ...,
        description="Re-ordered rank position after cross-encoder reranking (1-indexed).",
    )
    initial_score: float = Field(
        ...,
        description="Initial similarity or fusion score before reranking.",
    )


class RerankRequest(BaseModel):
    """Payload for executing two-stage retrieval (candidate recall + cross-encoder rerank)."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural language query string.",
        examples=["What are the travel reimbursement daily limits?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Final number of top reranked results to return.",
    )
    candidate_k: int = Field(
        default=25,
        ge=1,
        le=200,
        description="Candidate chunks to retrieve in Stage 1 before cross-encoder reranking.",
    )
    retrieval_strategy: Literal["hybrid", "semantic", "sparse"] = Field(
        default="hybrid",
        description="Stage 1 coarse candidate retrieval strategy.",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata filters applied to Stage 1 retrieval.",
    )
    fusion_method: str = Field(
        default="rrf",
        description="Fusion method for hybrid retrieval ('rrf' or 'linear').",
    )
    dense_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Dense weight for hybrid retrieval.",
    )
    sparse_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Sparse weight for hybrid retrieval.",
    )
    rrf_k: int = Field(
        default=60,
        gt=0,
        le=500,
        description="Smoothing constant k for Reciprocal Rank Fusion.",
    )


class DirectRerankRequest(BaseModel):
    """Payload for directly reranking an explicitly provided list of search items."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural language query string.",
    )
    items: list[SearchResultItem] = Field(
        ...,
        min_length=1,
        description="Candidate items to rescore and reorder with cross-encoder.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of reranked items to return.",
    )


class RerankResponse(BaseModel):
    """Response payload containing cross-encoder reranked results."""

    query: str = Field(..., description="Original query string.")
    total_candidates: int = Field(..., description="Number of candidates evaluated in Stage 1.")
    total_results: int = Field(..., description="Number of results returned after reranking.")
    retrieval_strategy: str = Field(..., description="Stage 1 coarse retrieval strategy utilized.")
    model: str = Field(..., description="Cross-encoder model identifier utilized.")
    results: list[RerankResultItem] = Field(
        default_factory=list,
        description="Re-ordered results sorted by cross-encoder score.",
    )
