"""Citation extraction, grounding verification, and faithfulness scoring engine for RAG."""

import re

from enterprise_agent.core.logging import get_logger
from enterprise_agent.schemas.rag import (
    Citation,
    GroundingEvaluation,
    RetrievedSourceChunk,
)

logger = get_logger(__name__)

# Standard English stopwords for content token filtering
STOPWORDS: set[str] = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
    # Common conversational fillers
    "according",
    "stated",
    "provides",
    "mentioned",
    "per",
    "based",
}

# Regex to match citation tags like [Source 1], [Source 1, 2], [Source 1, Source 2], [Doc 1]
CITATION_TAG_PATTERN = re.compile(
    r"\[(?:Source|Doc)s?\s*(?::\s*)?([0-9\s,]+(?:and\s*\d+)?)\]",
    re.IGNORECASE,
)

# Common abbreviations to avoid false sentence splits
ABBREVIATIONS: set[str] = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "inc",
    "ltd",
    "corp",
    "dept",
    "sec",
    "no",
    "fig",
    "e.g",
    "i.e",
    "vs",
    "hrs",
    "min",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
    "p.m",
    "a.m",
    "al",
    "etc",
}


class SentenceSplitter:
    """Rule-based sentence boundary segmenter with abbreviation and citation preservation."""

    @classmethod
    def split(cls, text: str) -> list[str]:
        """Split text into sentences while keeping citation tags attached to sentences."""
        if not text or not text.strip():
            return []

        # Normalize linebreaks: paragraphs / bullet points become distinct blocks
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sentences: list[str] = []

        for line in lines:
            # Strip list bullets / numbers
            clean_line = re.sub(r"^[-*•]\s+|\d+\.\s+", "", line).strip()
            if not clean_line:
                continue

            # Split by punctuation followed by space or end of string, avoiding abbreviations
            tokens = re.split(r"([.!?]+(?:\s+|$))", clean_line)
            current_sent = ""

            i = 0
            while i < len(tokens):
                part = tokens[i]
                current_sent += part
                i += 1
                if i < len(tokens) and re.match(r"[.!?]+(?:\s+|$)", tokens[i]):
                    punct = tokens[i]
                    last_word = (
                        current_sent.strip().split()[-1].lower() if current_sent.strip() else ""
                    )
                    clean_word = re.sub(r"[^\w.]", "", last_word)

                    # Check next token's first non-space character
                    next_token = tokens[i + 1] if i + 1 < len(tokens) else ""
                    next_first_char = next_token.strip()[0] if next_token.strip() else ""

                    is_abbr = clean_word in ABBREVIATIONS or clean_word.rstrip(".") in ABBREVIATIONS
                    is_lowercase_continuation = (
                        next_first_char.islower() and not next_token.strip().startswith("[")
                    )

                    if is_abbr or is_lowercase_continuation:
                        current_sent += punct
                    else:
                        current_sent += punct
                        if current_sent.strip():
                            sentences.append(current_sent.strip())
                        current_sent = ""
                    i += 1

            if current_sent.strip():
                sentences.append(current_sent.strip())

        return [s for s in sentences if len(s) > 5]


class CitationExtractor:
    """Extracts inline citation references and maps sentences to referenced sources."""

    @classmethod
    def extract_source_indices(cls, text: str) -> list[int]:
        """Extract unique integer source indices from text, e.g. '[Source 1, 2]' -> [1, 2]."""
        indices: list[int] = []
        matches = CITATION_TAG_PATTERN.findall(text)
        for match in matches:
            numbers = re.findall(r"\d+", match)
            for num_str in numbers:
                try:
                    indices.append(int(num_str))
                except ValueError:
                    continue
        return sorted(set(indices))

    @classmethod
    def strip_citation_tags(cls, text: str) -> str:
        """Remove citation brackets from text, leaving clean claim content."""
        clean = CITATION_TAG_PATTERN.sub("", text)
        clean = re.sub(r"\s+([.,!?;:])", r"\1", clean)
        return re.sub(r"\s+", " ", clean).strip()


class GroundingVerifier:
    """Factual grounding verifier calculating claim-to-chunk entailment and faithfulness."""

    def __init__(
        self,
        grounding_threshold: float = 0.40,
        unverified_threshold: float = 0.40,
        verified_threshold: float = 0.80,
    ) -> None:
        self.grounding_threshold = grounding_threshold
        self.unverified_threshold = unverified_threshold
        self.verified_threshold = verified_threshold

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase alphanumeric words."""
        return re.findall(r"\b[a-zA-Z0-9_\$]+\b", text.lower())

    @classmethod
    def _content_tokens(cls, text: str) -> set[str]:
        """Extract content tokens excluding common stopwords."""
        tokens = cls._tokenize(text)
        return {t for t in tokens if t not in STOPWORDS and len(t) > 1}

    @classmethod
    def _extract_numbers(cls, text: str) -> set[str]:
        """Extract numbers, currency values, and percentages."""
        return set(re.findall(r"\b\d+(?:\.\d+)?\b|\$\d+", text.lower()))

    @classmethod
    def _find_best_quote_snippet(cls, claim: str, chunk_content: str) -> str:
        """Extract the most relevant supporting sentence or excerpt from the chunk."""
        claim_content_words = cls._content_tokens(claim)
        if not claim_content_words:
            return chunk_content[:200].strip()

        # Split chunk into sentences/clauses
        chunk_sentences = re.split(r"[.!?\n]+", chunk_content)
        chunk_sentences = [s.strip() for s in chunk_sentences if len(s.strip()) > 10]

        if not chunk_sentences:
            return chunk_content[:200].strip()

        best_sentence = chunk_sentences[0]
        max_overlap = -1

        for sent in chunk_sentences:
            sent_words = cls._content_tokens(sent)
            overlap = len(claim_content_words.intersection(sent_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = sent

        return best_sentence.strip()

    def compute_grounding_score(self, claim: str, chunk_content: str) -> float:
        """Compute factual grounding metric between a claim and source chunk (0.0 to 1.0).

        Combines:
        1. Content token recall (what fraction of claim keywords appear in chunk).
        2. Content token Jaccard similarity.
        3. Numbers/quantities preservation check.
        4. N-gram phrase overlap.
        """
        claim_words = self._content_tokens(claim)
        chunk_words = self._content_tokens(chunk_content)

        if not claim_words:
            return 1.0

        # 1. Recall of claim keywords in chunk
        shared_words = claim_words.intersection(chunk_words)
        token_recall = len(shared_words) / len(claim_words)

        # 2. Jaccard similarity
        union_words = claim_words.union(chunk_words)
        jaccard = len(shared_words) / len(union_words) if union_words else 0.0

        # 3. Numeric entity check
        claim_numbers = self._extract_numbers(claim)
        chunk_numbers = self._extract_numbers(chunk_content)
        number_multiplier = 1.0
        if claim_numbers:
            found_numbers = claim_numbers.intersection(chunk_numbers)
            number_multiplier = len(found_numbers) / len(claim_numbers)

        # 4. Bigram overlap
        claim_raw_tokens = self._tokenize(claim)
        chunk_raw_tokens = self._tokenize(chunk_content)
        claim_bigrams = {
            f"{claim_raw_tokens[i]}_{claim_raw_tokens[i + 1]}"
            for i in range(len(claim_raw_tokens) - 1)
        }
        chunk_bigrams = {
            f"{chunk_raw_tokens[i]}_{chunk_raw_tokens[i + 1]}"
            for i in range(len(chunk_raw_tokens) - 1)
        }
        bigram_overlap = (
            len(claim_bigrams.intersection(chunk_bigrams)) / len(claim_bigrams)
            if claim_bigrams
            else 0.0
        )

        # Composite score
        base_score = (0.55 * token_recall) + (0.25 * jaccard) + (0.20 * bigram_overlap)
        final_score = base_score * (0.5 + 0.5 * number_multiplier)
        return min(1.0, max(0.0, round(final_score, 4)))

    def verify(
        self,
        answer: str,
        sources: list[RetrievedSourceChunk],
    ) -> GroundingEvaluation:
        """Verify the generated answer against the retrieved source chunks."""
        # Handle insufficient context short-circuit
        if (
            "insufficient information" in answer.lower()
            or "do not have enough information" in answer.lower()
        ):
            return GroundingEvaluation(
                faithfulness_score=1.0,
                grounding_status="insufficient_context",
                total_claims=0,
                grounded_claims=0,
                citations=[],
                unattributed_claims=[],
            )

        # 1. Split answer into distinct sentences/claims
        sentences = SentenceSplitter.split(answer)
        if not sentences:
            return GroundingEvaluation(
                faithfulness_score=1.0,
                grounding_status="verified",
                total_claims=0,
                grounded_claims=0,
                citations=[],
                unattributed_claims=[],
            )

        # Index sources by 1-based source_index
        source_map: dict[int, RetrievedSourceChunk] = {i + 1: s for i, s in enumerate(sources)}

        citations: list[Citation] = []
        unattributed_claims: list[str] = []
        grounded_claims_count = 0
        total_claims_count = len(sentences)

        for sentence in sentences:
            source_indices = CitationExtractor.extract_source_indices(sentence)
            clean_claim = CitationExtractor.strip_citation_tags(sentence)

            if not source_indices:
                unattributed_claims.append(sentence)
                continue

            sentence_is_grounded = False
            for src_idx in source_indices:
                chunk = source_map.get(src_idx)
                if not chunk:
                    unattributed_claims.append(f"{sentence} (Cited non-existent Source {src_idx})")
                    continue

                score = self.compute_grounding_score(clean_claim, chunk.content)
                is_grounded = score >= self.grounding_threshold
                if is_grounded:
                    sentence_is_grounded = True

                quote_snippet = self._find_best_quote_snippet(clean_claim, chunk.content)
                page_numbers: list[int] | None = None
                if "page" in chunk.metadata:
                    page_val = chunk.metadata["page"]
                    if isinstance(page_val, int):
                        page_numbers = [page_val]
                    elif isinstance(page_val, list):
                        page_numbers = [
                            int(p)
                            for p in page_val
                            if isinstance(p, (int, str)) and str(p).isdigit()
                        ]

                citations.append(
                    Citation(
                        source_index=src_idx,
                        document_id=chunk.document_id,
                        chunk_id=chunk.chunk_id,
                        source=chunk.source,
                        page_numbers=page_numbers,
                        quote_snippet=quote_snippet,
                        claim_text=clean_claim,
                        is_grounded=is_grounded,
                        grounding_score=score,
                    )
                )

            if sentence_is_grounded:
                grounded_claims_count += 1

        # Calculate faithfulness score
        faithfulness = (
            round(grounded_claims_count / total_claims_count, 4) if total_claims_count > 0 else 1.0
        )

        # Determine status
        if faithfulness >= self.verified_threshold:
            status = "verified"
        elif faithfulness >= self.unverified_threshold:
            status = "partially_grounded"
        else:
            status = "unverified"

        return GroundingEvaluation(
            faithfulness_score=faithfulness,
            grounding_status=status,
            total_claims=total_claims_count,
            grounded_claims=grounded_claims_count,
            citations=citations,
            unattributed_claims=unattributed_claims,
        )
