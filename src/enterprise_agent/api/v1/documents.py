from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from enterprise_agent.api.deps import (
    get_hybrid_search_service,
    get_ingestion_service,
)
from enterprise_agent.core.logging import get_logger
from enterprise_agent.hybrid.service import HybridSearchService
from enterprise_agent.ingestion.service import IngestionService
from enterprise_agent.schemas.documents import (
    DocumentChunkResponse,
    DocumentListResponse,
    DocumentSummaryResponse,
    IngestResponse,
    IngestTextRequest,
)
from enterprise_agent.schemas.search import IndexDocumentResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/ingest/text",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Text Content",
    description=(
        "Ingest raw text or Markdown string, sanitize, chunk, "
        "and optionally index into the vector store."
    ),
)
async def ingest_text(
    request: IngestTextRequest,
    auto_index: bool = Query(
        default=True,
        description="Automatically embed and index chunks into vector database.",
    ),
    service: IngestionService = Depends(get_ingestion_service),
    hybrid_service: HybridSearchService = Depends(get_hybrid_search_service),
) -> IngestResponse:
    """Ingest plain text or markdown directly."""
    logger.info("Ingesting text document '%s' (%d chars)", request.title, len(request.content))
    result = await service.ingest_text(
        text=request.content,
        title=request.title,
        source=request.source,
        metadata=request.metadata,
    )

    if auto_index and result.chunks and result.status == "success":
        try:
            dense_count, sparse_count = await hybrid_service.index_chunks(result.chunks)
            logger.info(
                "Auto-indexed %d dense chunks and %d sparse chunks into search engines",
                dense_count,
                sparse_count,
            )
        except Exception as e:
            logger.warning("Auto-indexing failed (document saved): %s", e)

    return IngestResponse(
        document_id=result.document.id,
        title=result.document.title,
        source=result.document.source,
        num_chunks=len(result.chunks),
        total_characters=len(result.document.content),
        checksum=result.document.checksum,
        status=result.status,
        message=result.message,
    )


@router.post(
    "/ingest/file",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Document File",
    description="Upload a document (.pdf, .md, .txt) to parse, sanitize, chunk, and index.",
)
async def ingest_file(
    file: UploadFile = File(..., description="Document file to upload (.pdf, .md, .txt)"),
    auto_index: bool = Query(
        default=True,
        description="Automatically embed and index chunks into vector database.",
    ),
    service: IngestionService = Depends(get_ingestion_service),
    hybrid_service: HybridSearchService = Depends(get_hybrid_search_service),
) -> IngestResponse:
    """Upload and process an enterprise document."""
    filename = file.filename or "uploaded_document.txt"
    logger.info("Ingesting uploaded file '%s'", filename)

    content_bytes = await file.read()
    if not content_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    result = await service.ingest_file(
        content_bytes=content_bytes,
        filename=filename,
    )

    if auto_index and result.chunks and result.status == "success":
        try:
            dense_count, sparse_count = await hybrid_service.index_chunks(result.chunks)
            logger.info(
                "Auto-indexed %d dense chunks and %d sparse chunks into search engines",
                dense_count,
                sparse_count,
            )
        except Exception as e:
            logger.warning("Auto-indexing failed (document saved): %s", e)

    return IngestResponse(
        document_id=result.document.id,
        title=result.document.title,
        source=result.document.source,
        num_chunks=len(result.chunks),
        total_characters=len(result.document.content),
        checksum=result.document.checksum,
        status=result.status,
        message=result.message,
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List Ingested Documents",
    description="Retrieve list of all ingested documents and their metadata.",
)
async def list_documents(
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentListResponse:
    """List all registered documents."""
    summaries = service.list_documents()
    formatted = [
        DocumentSummaryResponse(
            id=s.id,
            title=s.title,
            source=s.source,
            num_chunks=s.num_chunks,
            total_characters=s.total_characters,
            checksum=s.checksum,
            created_at=s.created_at,
        )
        for s in summaries
    ]
    return DocumentListResponse(
        documents=formatted,
        total_documents=len(formatted),
    )


@router.get(
    "/{document_id}/chunks",
    response_model=list[DocumentChunkResponse],
    summary="Get Document Chunks",
    description="Retrieve all chunk segments generated for a specific document.",
)
async def get_document_chunks(
    document_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> list[DocumentChunkResponse]:
    """Inspect the chunks produced from a given document."""
    doc = service.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    chunks = service.get_document_chunks(document_id)
    return [
        DocumentChunkResponse(
            id=c.id,
            document_id=c.document_id,
            chunk_index=c.chunk_index,
            content=c.content,
            start_char=c.start_char,
            end_char=c.end_char,
            token_count=c.token_count,
            metadata=c.metadata,
        )
        for c in chunks
    ]


@router.post(
    "/{document_id}/index",
    response_model=IndexDocumentResponse,
    summary="Index Document Chunks into Vector & Keyword Stores",
    description="Index document chunks across both dense vector and sparse keyword indices.",
)
async def index_document(
    document_id: str,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    hybrid_service: HybridSearchService = Depends(get_hybrid_search_service),
) -> IndexDocumentResponse:
    """Index an already ingested document into search indices."""
    doc = ingestion_service.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    chunks = ingestion_service.get_document_chunks(document_id)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document '{document_id}' has no chunks to index.",
        )

    dense_count, sparse_count = await hybrid_service.index_chunks(chunks)
    return IndexDocumentResponse(
        document_id=document_id,
        chunks_indexed=dense_count,
        status="success",
        message=f"Indexed {dense_count} dense vectors and {sparse_count} sparse keyword entries.",
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Document",
    description="Delete document, its chunks, and associated search indices.",
)
async def delete_document(
    document_id: str,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    hybrid_service: HybridSearchService = Depends(get_hybrid_search_service),
) -> dict[str, str]:
    """Delete document and purge vectors from search indices."""
    deleted = ingestion_service.delete_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    dense_del, sparse_del = await hybrid_service.delete_by_document_id(document_id)
    return {
        "document_id": document_id,
        "status": "deleted",
        "message": (
            f"Document '{document_id}' and all associated indices "
            f"({dense_del} dense, {sparse_del} sparse) deleted."
        ),
    }
