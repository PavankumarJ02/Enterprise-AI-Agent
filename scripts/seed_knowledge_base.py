"""Executable CLI script to seed the Enterprise AI Agent with synthetic corporate knowledge."""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from enterprise_agent.config.settings import Settings
from enterprise_agent.core.logging import get_logger, setup_logging
from enterprise_agent.embeddings.factory import get_embedding_provider
from enterprise_agent.embeddings.service import EmbeddingsService
from enterprise_agent.ingestion.service import IngestionService
from enterprise_agent.sparse.base import SparseStore
from enterprise_agent.sparse.factory import create_sparse_store
from enterprise_agent.vectorstore.factory import create_vector_store
from enterprise_agent.vectorstore.service import VectorSearchService

logger = get_logger(__name__)


def extract_title_from_markdown(content: str, default_title: str) -> str:
    """Extract first top-level H1 header as document title, or fallback to default."""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# ") and len(line) > 2:
            return line[2:].strip()
    return default_title


async def seed_knowledge_base(
    data_dir: Path,
    settings: Settings | None = None,
    dry_run: bool = False,
    include_sparse: bool = True,
    vector_service: VectorSearchService | None = None,
    sparse_store: SparseStore | None = None,
    ingestion_service: IngestionService | None = None,
) -> dict[str, Any]:
    """Recursively scan data directory and ingest all documents into vector and sparse stores.

    Args:
        data_dir: Path to directory containing departmental documents.
        settings: Application settings configuration.
        dry_run: If True, parse and chunk documents without writing to vector stores.
        include_sparse: If True, index document chunks into BM25/Sparse store.
        vector_service: Optional pre-configured VectorSearchService.
        sparse_store: Optional pre-configured SparseStore.
        ingestion_service: Optional pre-configured IngestionService.

    Returns:
        Dictionary summarizing ingestion statistics and execution telemetry.
    """
    start_time = time.perf_counter()
    app_settings = settings or Settings()

    # 1. Initialize services if not supplied
    if not ingestion_service:
        ingestion_service = IngestionService(
            chunk_size=800,
            chunk_overlap=150,
        )

    if not dry_run and not vector_service:
        v_store = create_vector_store(app_settings)
        e_provider = get_embedding_provider(app_settings)
        e_service = EmbeddingsService(provider=e_provider)
        vector_service = VectorSearchService(
            vector_store=v_store,
            embeddings_service=e_service,
        )

    if not dry_run and include_sparse and not sparse_store:
        sparse_store = create_sparse_store(app_settings)

    # 2. Discover documents
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Knowledge base data directory not found: {data_dir}")

    doc_files = sorted(
        [
            f
            for f in data_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in (".md", ".markdown", ".txt")
        ]
    )

    if not doc_files:
        logger.warning("No markdown or text documents found in %s", data_dir)
        return {
            "documents_ingested": 0,
            "chunks_produced": 0,
            "vectors_indexed": 0,
            "departments": [],
            "elapsed_seconds": round(time.perf_counter() - start_time, 3),
        }

    logger.info("Discovered %d documents across knowledge base directory", len(doc_files))

    total_chunks = 0
    total_vectors = 0
    departments_set: set[str] = set()

    # 3. Process and index each document
    for file_path in doc_files:
        relative_path = file_path.relative_to(data_dir)
        # Determine department from parent folder name
        department = relative_path.parts[0] if len(relative_path.parts) > 1 else "general"
        departments_set.add(department)

        content = file_path.read_text(encoding="utf-8")
        clean_fallback = file_path.stem.replace("_", " ").title()
        title = extract_title_from_markdown(content, default_title=clean_fallback)

        doc_meta = {
            "department": department,
            "filename": file_path.name,
            "relative_path": str(relative_path).replace("\\", "/"),
        }

        # Ingest and chunk
        ingest_result = await ingestion_service.ingest_text(
            text=content,
            title=title,
            source=str(relative_path).replace("\\", "/"),
            metadata=doc_meta,
        )

        chunk_count = len(ingest_result.chunks)
        total_chunks += chunk_count
        logger.info(
            "Ingested '%s' [%s] -> %d chunks (ID: %s)",
            title,
            department,
            chunk_count,
            ingest_result.document.id,
        )

        # Index vectors and keywords if not dry run
        if not dry_run and vector_service:
            indexed_count = await vector_service.index_chunks(ingest_result.chunks)
            total_vectors += indexed_count

            if include_sparse and sparse_store:
                await sparse_store.index_chunks(ingest_result.chunks)

    elapsed = round(time.perf_counter() - start_time, 3)
    summary = {
        "documents_ingested": len(doc_files),
        "chunks_produced": total_chunks,
        "vectors_indexed": total_vectors if not dry_run else 0,
        "departments": sorted(departments_set),
        "dry_run": dry_run,
        "elapsed_seconds": elapsed,
    }

    logger.info(
        "Seeding complete: %d docs, %d chunks, %d vectors indexed across %d departments in %.2fs",
        summary["documents_ingested"],
        summary["chunks_produced"],
        summary["vectors_indexed"],
        len(departments_set),
        elapsed,
    )
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the seeder script."""
    parser = argparse.ArgumentParser(
        description="Seed Enterprise AI Agent with synthetic corporate knowledge."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "knowledge_base"),
        help="Path to documents directory (default: data/knowledge_base).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk documents without writing to vector stores.",
    )
    parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Skip indexing into BM25 / Sparse keyword store.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


async def main() -> int:
    """CLI execution entrypoint."""
    args = parse_args()
    setup_logging(args.log_level)

    data_path = Path(args.data_dir).resolve()
    logger.info("Beginning knowledge base seeding from: %s", data_path)

    try:
        summary = await seed_knowledge_base(
            data_dir=data_path,
            dry_run=args.dry_run,
            include_sparse=not args.no_sparse,
        )
        print("\n" + "=" * 60)
        print(" Enterprise Knowledge Base Seeding Summary")
        print("=" * 60)
        print(f" Documents Ingested: {summary['documents_ingested']}")
        print(f" Chunks Produced:    {summary['chunks_produced']}")
        print(f" Vectors Indexed:    {summary['vectors_indexed']}")
        print(f" Departments:        {', '.join(summary['departments'])}")
        print(f" Elapsed Time:       {summary['elapsed_seconds']}s")
        print(f" Dry Run Mode:       {summary['dry_run']}")
        print("=" * 60 + "\n")
        return 0
    except Exception as exc:
        logger.error("Seeding failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
