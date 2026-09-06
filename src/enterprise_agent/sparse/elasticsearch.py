"""Elasticsearch adapter for enterprise BM25 sparse keyword search."""

from typing import Any

from elasticsearch import AsyncElasticsearch

from enterprise_agent.core.logging import get_logger
from enterprise_agent.ingestion.models import DocumentChunk
from enterprise_agent.schemas.search import SearchResultItem
from enterprise_agent.sparse.base import SparseStore

logger = get_logger(__name__)


class ElasticsearchStore(SparseStore):
    """Elasticsearch-backed BM25 sparse keyword store."""

    def __init__(
        self,
        client: AsyncElasticsearch,
        index_name: str = "enterprise_knowledge",
    ) -> None:
        self.client = client
        self.index_name = index_name
        self._initialized = False

    async def _ensure_index(self) -> None:
        """Create Elasticsearch index with BM25 mapping if it doesn't already exist."""
        if self._initialized:
            return

        exists = await self.client.indices.exists(index=self.index_name)
        if not exists:
            logger.info("Creating Elasticsearch index '%s' with BM25 mapping", self.index_name)
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text", "analyzer": "standard"},
                        "document_id": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "metadata": {"type": "object", "dynamic": True},
                    }
                }
            }
            await self.client.indices.create(index=self.index_name, body=mapping)

        self._initialized = True

    async def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Index chunks into Elasticsearch."""
        if not chunks:
            return 0

        await self._ensure_index()
        count = 0
        for chunk in chunks:
            doc = {
                "content": chunk.content,
                "document_id": chunk.document_id,
                "source": str(chunk.metadata.get("source", "")),
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            }
            await self.client.index(
                index=self.index_name,
                id=chunk.id,
                document=doc,
            )
            count += 1

        await self.client.indices.refresh(index=self.index_name)
        logger.info("Indexed %d chunks into Elasticsearch index '%s'", count, self.index_name)
        return count

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResultItem]:
        """Execute BM25 keyword search against Elasticsearch."""
        await self._ensure_index()

        must_clause: list[dict[str, Any]] = [{"match": {"content": {"query": query}}}]
        filter_clause: list[dict[str, Any]] = []

        if filters:
            for k, v in filters.items():
                if k in ("document_id", "source"):
                    filter_clause.append({"term": {k: v}})
                else:
                    filter_clause.append({"term": {f"metadata.{k}": v}})

        body: dict[str, Any] = {
            "query": {
                "bool": {
                    "must": must_clause,
                    "filter": filter_clause,
                }
            },
            "size": top_k,
        }

        resp = await self.client.search(index=self.index_name, body=body)
        hits = resp["hits"]["hits"]
        max_score = float(resp["hits"]["max_score"] or 1.0)

        results: list[SearchResultItem] = []
        for hit in hits:
            source = hit["_source"]
            raw_score = float(hit["_score"])
            norm_score = round(raw_score / max_score, 4) if max_score > 0.0 else 0.0

            if norm_score < min_score:
                continue

            results.append(
                SearchResultItem(
                    chunk_id=hit["_id"],
                    document_id=source["document_id"],
                    content=source["content"],
                    score=norm_score,
                    chunk_index=source["chunk_index"],
                    metadata=source.get("metadata", {}),
                )
            )

        return results

    async def delete_by_document_id(self, document_id: str) -> int:
        """Delete chunks for a document ID from Elasticsearch."""
        await self._ensure_index()
        resp = await self.client.delete_by_query(
            index=self.index_name,
            query={"term": {"document_id": document_id}},
            refresh=True,
        )
        deleted = int(resp.get("deleted", 0))
        logger.info("Deleted %d chunks for document %s from Elasticsearch", deleted, document_id)
        return deleted

    async def delete_collection(self) -> None:
        """Delete entire index from Elasticsearch."""
        exists = await self.client.indices.exists(index=self.index_name)
        if exists:
            await self.client.indices.delete(index=self.index_name)
            self._initialized = False
            logger.info("Deleted Elasticsearch index '%s'", self.index_name)
