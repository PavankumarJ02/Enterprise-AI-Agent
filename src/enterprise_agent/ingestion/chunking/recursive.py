"""Production-grade Recursive Character Chunker with overlap and metadata propagation."""

from enterprise_agent.ingestion.chunking.base import BaseChunker
from enterprise_agent.ingestion.models import Document, DocumentChunk


class RecursiveCharacterChunker(BaseChunker):
    """Hierarchical text splitter using natural linguistic and layout delimiters."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively break text down by hierarchical separators until fits in chunk_size."""
        final_chunks: list[str] = []

        # Find first matching separator
        separator = separators[-1]
        new_separators: list[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        splits = text.split(separator) if separator else list(text)

        # Merge splits into chunks of up to chunk_size
        good_splits: list[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                # If a piece alone exceeds chunk_size, recurse with finer separators
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if new_separators:
                    sub_chunks = self._split_text(s, new_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(s)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Combine smaller splits respecting chunk_size and chunk_overlap."""
        docs: list[str] = []
        current_doc: list[str] = []
        total = 0

        for s in splits:
            len_s = len(s) + (len(separator) if current_doc else 0)
            if total + len_s > self.chunk_size and current_doc:
                doc_text = separator.join(current_doc)
                if doc_text.strip():
                    docs.append(doc_text.strip())

                # Calculate overlap: retain tail items that fit within chunk_overlap
                overlap_doc: list[str] = []
                overlap_len = 0
                for item in reversed(current_doc):
                    if overlap_len + len(item) <= self.chunk_overlap:
                        overlap_doc.insert(0, item)
                        overlap_len += len(item) + len(separator)
                    else:
                        break

                current_doc = overlap_doc
                total = sum(len(x) + len(separator) for x in current_doc)

            current_doc.append(s)
            total += len_s

        if current_doc:
            doc_text = separator.join(current_doc)
            if doc_text.strip():
                docs.append(doc_text.strip())

        return docs

    def _resolve_page_number(self, start_pos: int, page_ranges: list[tuple[int, int, int]]) -> int:
        """Find which page a character offset belongs to."""
        if not page_ranges:
            return 1
        for page_num, _start_idx, end_idx in page_ranges:
            if start_pos < end_idx:
                return page_num
        return page_ranges[-1][0]

    def chunk(
        self,
        document: Document,
        pages: list[tuple[int, str]] | None = None,
    ) -> list[DocumentChunk]:
        """Chunk document text with overlap, offset calculation, and page mapping."""
        raw_text = document.content
        if not raw_text.strip():
            return []

        # Precompute page ranges (page_num, start_offset, end_offset)
        page_ranges: list[tuple[int, int, int]] = []
        curr_offset = 0
        if pages:
            for page_num, page_content in pages:
                page_len = len(page_content)
                page_ranges.append((page_num, curr_offset, curr_offset + page_len))
                curr_offset += page_len + 2  # account for joined double newlines

        raw_chunks = self._split_text(raw_text, self.separators)
        chunks: list[DocumentChunk] = []
        search_start = 0

        for idx, chunk_str in enumerate(raw_chunks):
            chunk_content = chunk_str.strip()
            if not chunk_content:
                continue

            # Locate substring offset
            start_char = raw_text.find(chunk_content, search_start)
            if start_char == -1:
                start_char = search_start
            end_char = start_char + len(chunk_content)
            search_start = max(search_start, end_char - self.chunk_overlap)

            page_num = self._resolve_page_number(start_char, page_ranges)

            chunk_meta = {
                **document.metadata,
                "title": document.title,
                "source": document.source,
                "page_number": page_num,
                "chunk_size": len(chunk_content),
            }

            # Approximate token count (roughly words * 1.3 or chars // 4)
            estimated_tokens = max(1, len(chunk_content) // 4)

            chunk_id = f"{document.id}:chunk_{idx}"
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    document_id=document.id,
                    content=chunk_content,
                    chunk_index=idx,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=estimated_tokens,
                    metadata=chunk_meta,
                )
            )

        return chunks
