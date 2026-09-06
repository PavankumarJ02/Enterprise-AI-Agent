"""RAG evaluation metrics: Faithfulness, Answer Relevance, Context Precision, and Context Recall."""

import math
import re

from enterprise_agent.embeddings.base import EmbeddingProvider
from enterprise_agent.rag.citations import SentenceSplitter

_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "can",
    "will",
    "just",
    "should",
    "now",
}


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return [w.lower() for w in re.findall(r"\b\w+\b", text) if w.strip()]


def _content_words(text: str) -> set[str]:
    """Extract set of non-stopword tokens from text."""
    return {w for w in _tokenize(text) if w not in _STOPWORDS and len(w) > 1}


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = dot / (norm_a * norm_b)
    return max(0.0, min(1.0, sim))


def _stem(word: str) -> str:
    """Simple rule-based suffix stemming for English tokens."""
    for suff in ("ation", "tion", "ing", "ies", "es", "ed", "s"):
        if word.endswith(suff) and len(word) > len(suff) + 2:
            return word[: -len(suff)]
    return word


def calculate_faithfulness(answer: str, retrieved_contexts: list[str]) -> float:
    """Calculate faithfulness score (0.0 to 1.0) of answer claims grounded in context."""
    if not answer.strip():
        return 1.0
    lower_ans = answer.lower()
    if "insufficient information" in lower_ans or "not have enough information" in lower_ans:
        return 1.0
    if not retrieved_contexts:
        return 0.0

    claims = SentenceSplitter.split(answer)
    if not claims:
        return 1.0

    combined_context = " ".join(retrieved_contexts)
    context_tokens = set(_tokenize(combined_context))
    context_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", combined_context))

    grounded_count = 0
    for claim in claims:
        claim_tokens = _content_words(claim)
        if not claim_tokens:
            grounded_count += 1
            continue

        overlap = len(claim_tokens.intersection(context_tokens)) / len(claim_tokens)
        claim_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim))

        # Check numeric alignment
        num_ok = True
        if claim_nums:
            num_ok = bool(claim_nums.issubset(context_numbers))

        if overlap >= 0.40 and num_ok:
            grounded_count += 1

    return round(grounded_count / len(claims), 4)


def _calculate_lexical_relevance(query: str, answer: str) -> float:
    """Calculate calibrated lexical relevance between query and answer."""
    if not query.strip() or not answer.strip():
        return 0.0
    q_words = {_stem(w) for w in _content_words(query)}
    a_words = {_stem(w) for w in _content_words(answer)}
    if not q_words or not a_words:
        return 0.75
    overlap = len(q_words.intersection(a_words))
    if overlap == 0:
        return 0.20
    cov = overlap / len(q_words)
    base_score = 0.60 + 0.35 * cov
    return round(min(1.0, base_score), 4)


async def calculate_answer_relevance(
    query: str,
    answer: str,
    embedding_provider: EmbeddingProvider | None = None,
) -> float:
    """Calculate semantic relevance score (0.0 to 1.0) of answer to the user query."""
    if not query.strip() or not answer.strip():
        return 0.0

    from enterprise_agent.embeddings.mock import MockEmbeddingProvider

    # If non-mock embedding provider is supplied, calculate cosine similarity
    if embedding_provider is not None and not isinstance(embedding_provider, MockEmbeddingProvider):
        q_vec = await embedding_provider.embed_query(query)
        a_vec = await embedding_provider.embed_query(answer)
        cos_sim = _cosine_similarity(q_vec, a_vec)
        return round(cos_sim, 4)

    return _calculate_lexical_relevance(query, answer)


def calculate_context_precision(retrieved_contexts: list[str], ground_truth: str) -> float:
    """Calculate Mean Average Precision (0.0 to 1.0) of retrieved contexts for ground truth."""
    if not retrieved_contexts or not ground_truth.strip():
        return 0.0

    gt_words = _content_words(ground_truth)
    if not gt_words:
        return 1.0

    relevant_count = 0
    precision_sum = 0.0

    for idx, ctx in enumerate(retrieved_contexts, start=1):
        ctx_words = set(_tokenize(ctx))
        overlap = len(gt_words.intersection(ctx_words)) / len(gt_words)
        # Context is relevant if it shares at least 25% of ground-truth content words
        if overlap >= 0.25:
            relevant_count += 1
            precision_sum += relevant_count / idx

    if relevant_count == 0:
        return 0.0

    return round(precision_sum / relevant_count, 4)


def calculate_context_recall(retrieved_contexts: list[str], ground_truth: str) -> float:
    """Calculate context recall (0.0 to 1.0) measuring ground truth coverage in context."""
    if not ground_truth.strip():
        return 1.0
    if not retrieved_contexts:
        return 0.0

    gt_sentences = SentenceSplitter.split(ground_truth)
    if not gt_sentences:
        return 1.0

    combined_context = " ".join(retrieved_contexts)
    context_tokens = set(_tokenize(combined_context))

    covered_count = 0
    for stmt in gt_sentences:
        stmt_words = _content_words(stmt)
        if not stmt_words:
            covered_count += 1
            continue

        overlap = len(stmt_words.intersection(context_tokens)) / len(stmt_words)
        if overlap >= 0.40:
            covered_count += 1

    return round(covered_count / len(gt_sentences), 4)
