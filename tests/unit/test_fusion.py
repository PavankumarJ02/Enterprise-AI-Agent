"""Unit tests for Reciprocal Rank Fusion (RRF) and linear score fusion."""

from enterprise_agent.hybrid.fusion import linear_score_fusion, reciprocal_rank_fusion
from enterprise_agent.schemas.search import SearchResultItem


def test_reciprocal_rank_fusion_basic() -> None:
    """Verify RRF calculates correct reciprocal ranks and ranks dual-matched items highest."""
    dense_results = [
        SearchResultItem(
            chunk_id="chunk_both",
            document_id="doc1",
            content="Dual matched item",
            score=0.95,
            chunk_index=0,
        ),
        SearchResultItem(
            chunk_id="chunk_dense_only",
            document_id="doc2",
            content="Dense only item",
            score=0.85,
            chunk_index=0,
        ),
    ]

    sparse_results = [
        SearchResultItem(
            chunk_id="chunk_both",
            document_id="doc1",
            content="Dual matched item",
            score=0.90,
            chunk_index=0,
        ),
        SearchResultItem(
            chunk_id="chunk_sparse_only",
            document_id="doc3",
            content="Sparse only item",
            score=0.75,
            chunk_index=0,
        ),
    ]

    fused = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        k=60,
        dense_weight=0.5,
        sparse_weight=0.5,
    )

    assert len(fused) == 3
    # chunk_both was rank 1 in both dense and sparse:
    # 0.5/(60+1) + 0.5/(60+1) = 1.0 / 61 = 0.016393
    assert fused[0].chunk_id == "chunk_both"
    assert fused[0].dense_rank == 1
    assert fused[0].sparse_rank == 1
    assert abs(fused[0].combined_score - (1.0 / 61)) < 1e-4

    # Other items only have single-retriever contribution
    assert fused[1].chunk_id in ("chunk_dense_only", "chunk_sparse_only")
    assert fused[0].combined_score > fused[1].combined_score


def test_reciprocal_rank_fusion_weighting() -> None:
    """Verify weights modulate score contribution between dense and sparse."""
    dense = [
        SearchResultItem(
            chunk_id="c_dense",
            document_id="d1",
            content="Dense winner",
            score=0.90,
            chunk_index=0,
        )
    ]
    sparse = [
        SearchResultItem(
            chunk_id="c_sparse",
            document_id="d2",
            content="Sparse winner",
            score=0.90,
            chunk_index=0,
        )
    ]

    # Give 80% weight to dense
    fused_dense_heavy = reciprocal_rank_fusion(
        dense, sparse, k=60, dense_weight=0.8, sparse_weight=0.2
    )
    assert fused_dense_heavy[0].chunk_id == "c_dense"

    # Give 80% weight to sparse
    fused_sparse_heavy = reciprocal_rank_fusion(
        dense, sparse, k=60, dense_weight=0.2, sparse_weight=0.8
    )
    assert fused_sparse_heavy[0].chunk_id == "c_sparse"


def test_linear_score_fusion() -> None:
    """Verify linear combination normalizes scores and blends correctly."""
    dense = [
        SearchResultItem(
            chunk_id="c1",
            document_id="d1",
            content="Item 1",
            score=0.80,
            chunk_index=0,
        )
    ]
    sparse = [
        SearchResultItem(
            chunk_id="c1",
            document_id="d1",
            content="Item 1",
            score=10.0,
            chunk_index=0,
        )
    ]

    fused = linear_score_fusion(dense, sparse, dense_weight=0.5, sparse_weight=0.5)
    assert len(fused) == 1
    # Both are highest score in their respective lists -> normalized to 1.0 -> combined is 1.0
    assert fused[0].combined_score == 1.0
    assert fused[0].dense_score == 0.80
    assert fused[0].sparse_score == 10.0
