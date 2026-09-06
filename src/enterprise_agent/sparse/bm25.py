"""In-memory BM25 lexical keyword store using Lucene BM25 formula."""

import math
import re
from collections import Counter
from typing import Any

from enterprise_agent.core.logging import get_logger
from enterprise_agent.ingestion.models import DocumentChunk
from enterprise_agent.schemas.search import SearchResultItem
from enterprise_agent.sparse.base import SparseStore

logger = get_logger(__name__)


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text preserving hyphens, underscores, codes, and currency symbols.

    Supports exact matching for codes like CVE-2024-1234, SKU-900, ISO-27001, $250.
    """
    return [t.lower() for t in re.findall(r"\$?\w+(?:-\w+)*", text) if len(t) > 0]


class LuceneBM25Index:
    """Lucene-compatible BM25 indexer with strictly positive IDF."""

    def __init__(
        self,
        corpus: list[list[str]],
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 1.0
        self.doc_freqs: dict[str, int] = {}
        self.term_freqs: list[dict[str, int]] = []

        for doc in corpus:
            tf = Counter(doc)
            self.term_freqs.append(tf)
            for term in tf:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        """Compute Lucene BM25 scores for query tokens across all indexed documents."""
        scores = [0.0] * self.corpus_size
        if not query_tokens or self.corpus_size == 0:
            return scores

        for term in query_tokens:
            if term not in self.doc_freqs:
                continue

            df = self.doc_freqs[term]
            # Lucene positive IDF formula: ln(1 + (N - n + 0.5) / (n + 0.5))
            idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

            for i, tf_dict in enumerate(self.term_freqs):
                tf = tf_dict.get(term, 0)
                if tf > 0:
                    len_norm = 1.0 - self.b + self.b * (self.doc_lengths[i] / self.avgdl)
                    term_score = idf * ((tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm))
                    scores[i] += term_score

        return scores


class InMemoryBM25Store(SparseStore):
    """In-memory sparse keyword retriever using Lucene BM25 scoring."""

    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunk] = {}
        self._ordered_ids: list[str] = []
        self._bm25: LuceneBM25Index | None = None

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index over all registered chunks."""
        if not self._chunks:
            self._bm25 = None
            self._ordered_ids = []
            return

        self._ordered_ids = list(self._chunks.keys())
        tokenized_corpus = [
            tokenize_for_bm25(self._chunks[cid].content) for cid in self._ordered_ids
        ]
        self._bm25 = LuceneBM25Index(tokenized_corpus)
        logger.debug("Rebuilt in-memory BM25 index with %d chunks", len(self._ordered_ids))

    async def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Add or update document chunks in the in-memory BM25 store."""
        if not chunks:
            return 0

        for chunk in chunks:
            self._chunks[chunk.id] = chunk

        self._rebuild_index()
        logger.info(
            "Indexed %d chunks into InMemoryBM25Store (total=%d)",
            len(chunks),
            len(self._chunks),
        )
        return len(chunks)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResultItem]:
        """Search chunks matching query terms using Lucene BM25."""
        if not self._bm25 or not self._chunks:
            return []

        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []

        raw_scores = self._bm25.get_scores(tokens)
        max_score = float(max(raw_scores)) if len(raw_scores) > 0 else 0.0

        results: list[SearchResultItem] = []
        for i, chunk_id in enumerate(self._ordered_ids):
            score = float(raw_scores[i])
            if score <= 0.0:
                continue

            chunk = self._chunks[chunk_id]

            # Apply metadata filters if provided
            if filters:
                match = True
                for k, v in filters.items():
                    if k == "document_id":
                        if chunk.document_id != v:
                            match = False
                            break
                        continue
                    val = chunk.metadata.get(k)
                    if val != v:
                        match = False
                        break
                if not match:
                    continue

            # Normalize BM25 score to [0.0, 1.0] range
            norm_score = round(score / max_score, 4) if max_score > 0.0 else 0.0
            if norm_score < min_score:
                continue

            results.append(
                SearchResultItem(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    score=norm_score,
                    chunk_index=chunk.chunk_index,
                    metadata=chunk.metadata,
                )
            )

        # Sort descending by normalized score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def delete_by_document_id(self, document_id: str) -> int:
        """Purge all chunks matching the given document ID."""
        to_delete = [cid for cid, chunk in self._chunks.items() if chunk.document_id == document_id]
        for cid in to_delete:
            del self._chunks[cid]

        if to_delete:
            self._rebuild_index()
            logger.info(
                "Deleted %d chunks for document %s from BM25 store",
                len(to_delete),
                document_id,
            )

        return len(to_delete)

    async def delete_collection(self) -> None:
        """Reset the BM25 store."""
        self._chunks.clear()
        self._ordered_ids.clear()
        self._bm25 = None
        logger.info("Cleared InMemoryBM25Store")
