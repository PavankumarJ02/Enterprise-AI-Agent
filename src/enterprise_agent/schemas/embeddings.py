"""Request and response schemas for dense embedding operations."""

from pydantic import BaseModel, Field


class EmbedTextsRequest(BaseModel):
    """Payload for batch embedding a list of document texts or chunks."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of text chunks to vectorize.",
        examples=[["First document chunk", "Second document chunk"]],
    )


class EmbedTextsResponse(BaseModel):
    """Response containing generated dense vectors and operational metadata."""

    vectors: list[list[float]] = Field(..., description="Dense float embedding vectors.")
    dimensions: int = Field(..., description="Dimensionality of vectors.")
    count: int = Field(..., description="Number of vectors returned.")
    model: str = Field(..., description="Embedding model used.")
    token_count: int = Field(default=0, description="Tokens consumed.")
    latency_ms: float = Field(..., description="Generation latency in milliseconds.")


class EmbedQueryRequest(BaseModel):
    """Payload for generating an asymmetric search query embedding."""

    query: str = Field(
        ...,
        min_length=1,
        description="Search query to embed.",
        examples=["What is the vacation policy?"],
    )


class EmbedQueryResponse(BaseModel):
    """Response containing single query vector."""

    vector: list[float] = Field(..., description="Dense float query vector.")
    dimensions: int = Field(..., description="Dimensionality of vector.")
    model: str = Field(..., description="Embedding model used.")
    latency_ms: float = Field(..., description="Generation latency in milliseconds.")
