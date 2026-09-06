"""Factory functions for initializing vector database clients and repositories."""

from qdrant_client import AsyncQdrantClient

from enterprise_agent.config.settings import Settings
from enterprise_agent.core.logging import get_logger
from enterprise_agent.vectorstore.base import VectorStore
from enterprise_agent.vectorstore.qdrant import QdrantVectorStore

logger = get_logger(__name__)


def create_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    """Create an AsyncQdrantClient instance based on configuration."""
    url = settings.qdrant_url.strip()
    if not url or url == ":memory:" or settings.is_testing:
        logger.info("Initializing in-memory local Qdrant client")
        return AsyncQdrantClient(location=":memory:")

    api_key = settings.qdrant_api_key.get_secret_value() or None
    logger.info("Connecting to remote Qdrant cluster at %s", url)
    return AsyncQdrantClient(
        url=url,
        api_key=api_key,
        prefer_grpc=settings.qdrant_prefer_grpc,
    )


def create_vector_store(
    settings: Settings,
    client: AsyncQdrantClient | None = None,
) -> VectorStore:
    """Factory to instantiate the configured VectorStore implementation."""
    qdrant_client = client or create_qdrant_client(settings)
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=settings.qdrant_collection_name,
    )
