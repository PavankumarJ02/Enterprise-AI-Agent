"""Context assembly, delimiter formatting, and dynamic token budgeting for RAG."""

import math

from enterprise_agent.core.logging import get_logger
from enterprise_agent.schemas.rag import RetrievedSourceChunk
from enterprise_agent.schemas.search import SearchResultItem

logger = get_logger(__name__)


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator (approx 4 characters per token for English/code)."""
    return max(1, math.ceil(len(text) / 4.0))


class RAGContextAssembler:
    """Formats retrieved document chunks into clean delimited context within a token budget."""

    def __init__(self, chars_per_token: float = 4.0) -> None:
        self.chars_per_token = max(1.0, chars_per_token)

    def assemble(
        self,
        results: list[SearchResultItem],
        max_context_tokens: int = 4000,
    ) -> tuple[str, list[RetrievedSourceChunk]]:
        """Assemble top-ranked chunks into formatted context blocks under token ceiling."""
        if not results:
            return "", []

        context_blocks: list[str] = []
        sources: list[RetrievedSourceChunk] = []
        accumulated_tokens = 0

        for i, item in enumerate(results, start=1):
            source_label = str(item.metadata.get("source") or item.document_id)
            page_info = f" | Page: {item.metadata['page']}" if "page" in item.metadata else ""
            sec_val = item.metadata.get("section")
            section_info = f" | Section: {sec_val}" if sec_val else ""

            header = (
                f"[Source {i} | Document: {item.document_id} | "
                f"File: {source_label}{page_info}{section_info}]"
            )
            block = f"{header}\n{item.content.strip()}\n"

            block_tokens = estimate_tokens(block)

            # Check if adding this chunk exceeds the token budget
            if accumulated_tokens + block_tokens > max_context_tokens:
                # If even the first chunk exceeds budget, truncate it to fit
                if not context_blocks:
                    char_limit = int(max_context_tokens * self.chars_per_token)
                    truncated_content = item.content[:char_limit].rsplit(" ", 1)[0]
                    truncated_block = (
                        f"{header}\n{truncated_content.strip()}... "
                        f"[Truncated to fit context window]\n"
                    )
                    context_blocks.append(truncated_block)
                    sources.append(
                        RetrievedSourceChunk(
                            chunk_id=item.chunk_id,
                            document_id=item.document_id,
                            source=source_label,
                            chunk_index=item.chunk_index,
                            content=truncated_content,
                            score=item.score,
                            metadata=item.metadata,
                        )
                    )
                    logger.warning(
                        "First chunk exceeded token budget (%d tokens); truncated",
                        max_context_tokens,
                    )
                else:
                    logger.debug(
                        "Context budget reached (%d / %d tokens). Omitted remaining %d chunks.",
                        accumulated_tokens,
                        max_context_tokens,
                        len(results) - len(sources),
                    )
                break

            context_blocks.append(block)
            accumulated_tokens += block_tokens
            sources.append(
                RetrievedSourceChunk(
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    source=source_label,
                    chunk_index=item.chunk_index,
                    content=item.content,
                    score=item.score,
                    metadata=item.metadata,
                )
            )

        full_context = "\n".join(context_blocks)
        logger.debug(
            "Assembled context with %d chunks (%d estimated tokens)",
            len(sources),
            accumulated_tokens,
        )
        return full_context, sources
