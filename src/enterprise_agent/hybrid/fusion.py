"""Rank fusion algorithms for combining dense semantic and sparse lexical search results."""

from enterprise_agent.schemas.search import HybridSearchResultItem, SearchResultItem


def reciprocal_rank_fusion(
    dense_results: list[SearchResultItem],
    sparse_results: list[SearchResultItem],
    k: int = 60,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> list[HybridSearchResultItem]:
    """Combine dense and sparse search rankings using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(d) = (w_dense / (k + rank_dense(d))) + (w_sparse / (k + rank_sparse(d)))

    Args:
        dense_results: Results retrieved from dense vector similarity search.
        sparse_results: Results retrieved from sparse BM25 keyword search.
        k: Smoothing constant parameter (standard default is 60).
        dense_weight: Importance weighting applied to dense rankings.
        sparse_weight: Importance weighting applied to sparse rankings.

    Returns:
        List of HybridSearchResultItem ordered by fused RRF score descending.
    """
    # Track items by chunk_id
    items_by_id: dict[str, SearchResultItem] = {}
    dense_ranks: dict[str, int] = {}
    dense_scores: dict[str, float] = {}
    sparse_ranks: dict[str, int] = {}
    sparse_scores: dict[str, float] = {}

    for rank, item in enumerate(dense_results, start=1):
        items_by_id[item.chunk_id] = item
        dense_ranks[item.chunk_id] = rank
        dense_scores[item.chunk_id] = item.score

    for rank, item in enumerate(sparse_results, start=1):
        if item.chunk_id not in items_by_id:
            items_by_id[item.chunk_id] = item
        sparse_ranks[item.chunk_id] = rank
        sparse_scores[item.chunk_id] = item.score

    fused_results: list[HybridSearchResultItem] = []
    for chunk_id, base_item in items_by_id.items():
        d_rank = dense_ranks.get(chunk_id)
        s_rank = sparse_ranks.get(chunk_id)

        d_contrib = (dense_weight / (k + d_rank)) if d_rank is not None else 0.0
        s_contrib = (sparse_weight / (k + s_rank)) if s_rank is not None else 0.0
        combined_score = round(d_contrib + s_contrib, 6)

        fused_results.append(
            HybridSearchResultItem(
                chunk_id=base_item.chunk_id,
                document_id=base_item.document_id,
                content=base_item.content,
                score=combined_score,
                chunk_index=base_item.chunk_index,
                metadata=base_item.metadata,
                dense_score=dense_scores.get(chunk_id),
                sparse_score=sparse_scores.get(chunk_id),
                dense_rank=d_rank,
                sparse_rank=s_rank,
                combined_score=combined_score,
            )
        )

    fused_results.sort(key=lambda r: r.combined_score, reverse=True)
    return fused_results


def linear_score_fusion(
    dense_results: list[SearchResultItem],
    sparse_results: list[SearchResultItem],
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> list[HybridSearchResultItem]:
    """Combine dense and sparse search rankings using normalized linear score combination.

    Formula:
        Score(d) = (w_dense * norm_dense(d) + w_sparse * norm_sparse(d)) / (w_dense + w_sparse)

    Args:
        dense_results: Results retrieved from dense vector similarity search.
        sparse_results: Results retrieved from sparse BM25 keyword search.
        dense_weight: Importance weighting applied to dense scores.
        sparse_weight: Importance weighting applied to sparse scores.

    Returns:
        List of HybridSearchResultItem ordered by combined linear score descending.
    """
    items_by_id: dict[str, SearchResultItem] = {}
    dense_ranks: dict[str, int] = {}
    dense_scores: dict[str, float] = {}
    sparse_ranks: dict[str, int] = {}
    sparse_scores: dict[str, float] = {}

    max_dense = max((item.score for item in dense_results), default=1.0)
    if max_dense <= 0.0:
        max_dense = 1.0

    max_sparse = max((item.score for item in sparse_results), default=1.0)
    if max_sparse <= 0.0:
        max_sparse = 1.0

    for rank, item in enumerate(dense_results, start=1):
        items_by_id[item.chunk_id] = item
        dense_ranks[item.chunk_id] = rank
        dense_scores[item.chunk_id] = item.score

    for rank, item in enumerate(sparse_results, start=1):
        if item.chunk_id not in items_by_id:
            items_by_id[item.chunk_id] = item
        sparse_ranks[item.chunk_id] = rank
        sparse_scores[item.chunk_id] = item.score

    total_weight = dense_weight + sparse_weight
    if total_weight <= 0:
        total_weight = 1.0

    fused_results: list[HybridSearchResultItem] = []
    for chunk_id, base_item in items_by_id.items():
        d_score = dense_scores.get(chunk_id, 0.0)
        s_score = sparse_scores.get(chunk_id, 0.0)

        norm_d = d_score / max_dense if d_score > 0.0 else 0.0
        norm_s = s_score / max_sparse if s_score > 0.0 else 0.0

        combined = round(((dense_weight * norm_d) + (sparse_weight * norm_s)) / total_weight, 4)

        fused_results.append(
            HybridSearchResultItem(
                chunk_id=base_item.chunk_id,
                document_id=base_item.document_id,
                content=base_item.content,
                score=combined,
                chunk_index=base_item.chunk_index,
                metadata=base_item.metadata,
                dense_score=dense_scores.get(chunk_id),
                sparse_score=sparse_scores.get(chunk_id),
                dense_rank=dense_ranks.get(chunk_id),
                sparse_rank=sparse_ranks.get(chunk_id),
                combined_score=combined,
            )
        )

    fused_results.sort(key=lambda r: r.combined_score, reverse=True)
    return fused_results
