"""Qdrant Vector Store implementation for enterprise semantic retrieval."""

import uuid
import warnings
from typing import Any

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from enterprise_agent.core.exceptions import VectorStoreError
from enterprise_agent.core.logging import get_logger
from enterprise_agent.ingestion.models import DocumentChunk
from enterprise_agent.vectorstore.base import SearchResult, VectorStore

logger = get_logger(__name__)


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Generate a deterministic UUIDv5 string from an arbitrary chunk identifier."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class QdrantVectorStore(VectorStore):
    """Production vector database adapter using Qdrant.

    Supports both in-memory local execution (:memory:) and remote Qdrant clusters.
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str = "enterprise_knowledge",
    ) -> None:
        self.client = client
        self.collection_name = collection_name

    async def create_collection_if_not_exists(self, dimensions: int) -> None:
        """Create the target Qdrant collection with Cosine metric and payload indexes."""
        try:
            exists = await self.client.collection_exists(self.collection_name)
            if not exists:
                logger.info(
                    "Creating Qdrant collection '%s' with %d dimensions (Cosine distance)",
                    self.collection_name,
                    dimensions,
                )
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=dimensions,
                        distance=models.Distance.COSINE,
                    ),
                )
                # Create keyword payload indexes for fast filtering (server-only)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    try:
                        await self.client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name="document_id",
                            field_schema=models.PayloadSchemaType.KEYWORD,
                        )
                        await self.client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name="source",
                            field_schema=models.PayloadSchemaType.KEYWORD,
                        )
                    except Exception as idx_err:
                        logger.debug("Payload index creation note: %s", idx_err)
            else:
                logger.debug("Qdrant collection '%s' already exists", self.collection_name)
        except Exception as e:
            logger.error("Failed to create collection '%s': %s", self.collection_name, e)
            msg = f"Failed to initialize collection {self.collection_name}: {e}"
            raise VectorStoreError(msg) from e

    async def collection_exists(self) -> bool:
        """Check if collection is initialized."""
        try:
            return bool(await self.client.collection_exists(self.collection_name))
        except Exception as e:
            logger.error("Error checking collection existence: %s", e)
            return False

    async def upsert(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> int:
        """Upsert document chunks and their dense embeddings into Qdrant."""
        if not chunks:
            return 0

        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f"Mismatch: received {len(chunks)} chunks but {len(vectors)} vectors"
            )

        points: list[models.PointStruct] = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            point_id = chunk_id_to_uuid(chunk.id)
            payload: dict[str, Any] = {
                **chunk.metadata,
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "token_count": chunk.token_count,
                "chunk_hash": chunk.chunk_hash,
                "source": chunk.metadata.get("source", ""),
                "metadata": chunk.metadata,
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload,
                )
            )

        try:
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.info(
                "Successfully upserted %d vectors into '%s'",
                len(points),
                self.collection_name,
            )
            return len(points)
        except Exception as e:
            logger.error("Failed to upsert points into '%s': %s", self.collection_name, e)
            raise VectorStoreError(f"Vector upsert failed: {e}") from e

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Perform ANN search against the Qdrant collection with optional payload filtering."""
        query_filter: models.Filter | None = None
        if filters:
            conditions: list[models.Condition] = []
            for key, val in filters.items():
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=val),
                    )
                )
            if conditions:
                query_filter = models.Filter(must=conditions)

        try:
            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=min_score if min_score > 0.0 else None,
            )

            results: list[SearchResult] = []
            for point in response.points:
                payload = point.payload or {}
                results.append(
                    SearchResult(
                        chunk_id=str(payload.get("chunk_id", point.id)),
                        document_id=str(payload.get("document_id", "")),
                        content=str(payload.get("content", "")),
                        score=float(point.score),
                        chunk_index=int(payload.get("chunk_index", 0)),
                        metadata=dict(payload.get("metadata", {})),
                    )
                )

            return results
        except UnexpectedResponse as ue:
            logger.error("Qdrant unexpected response during search: %s", ue)
            raise VectorStoreError(f"Qdrant search error: {ue}") from ue
        except Exception as e:
            logger.error("Vector search failed on '%s': %s", self.collection_name, e)
            raise VectorStoreError(f"Vector search failed: {e}") from e

    async def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks belonging to a document ID."""
        try:
            del_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=del_filter,
            )
            logger.info("Deleted chunk points for document_id '%s'", document_id)
            return 1
        except Exception as e:
            logger.error("Failed to delete points for document_id '%s': %s", document_id, e)
            raise VectorStoreError(f"Document chunk deletion failed: {e}") from e

    async def count(self) -> int:
        """Get total vector count in the collection."""
        try:
            cnt = await self.client.count(collection_name=self.collection_name)
            return int(cnt.count)
        except Exception as e:
            logger.error("Failed to count points in '%s': %s", self.collection_name, e)
            return 0

    async def health_check(self) -> bool:
        """Check if Qdrant service is reachable."""
        try:
            await self.client.get_collections()
            return True
        except Exception as e:
            logger.warning("Qdrant health check failed: %s", e)
            return False

    async def close(self) -> None:
        """Close underlying client connection."""
        try:
            await self.client.close()
        except Exception as e:
            logger.debug("Error closing Qdrant client: %s", e)
