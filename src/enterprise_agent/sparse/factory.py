"""Factory for creating SparseStore instances based on application settings."""

from elasticsearch import AsyncElasticsearch

from enterprise_agent.config.settings import Settings
from enterprise_agent.core.logging import get_logger
from enterprise_agent.sparse.base import SparseStore
from enterprise_agent.sparse.bm25 import InMemoryBM25Store
from enterprise_agent.sparse.elasticsearch import ElasticsearchStore

logger = get_logger(__name__)


def create_sparse_store(settings: Settings) -> SparseStore:
    """Instantiate a SparseStore provider based on settings.

    Args:
        settings: Application configuration settings.

    Returns:
        Configured SparseStore instance (InMemoryBM25Store or ElasticsearchStore).
    """
    if settings.sparse_search_provider == "elasticsearch":
        logger.info("Initializing Elasticsearch sparse store at %s", settings.elasticsearch_url)
        api_key = settings.elasticsearch_api_key.get_secret_value() or None
        client = AsyncElasticsearch(
            hosts=[settings.elasticsearch_url],
            api_key=api_key,
        )
        return ElasticsearchStore(client=client, index_name=settings.elasticsearch_index_name)

    logger.info("Initializing in-memory BM25 sparse store")
    return InMemoryBM25Store()
