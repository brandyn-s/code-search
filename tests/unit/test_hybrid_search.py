"""Tests for RRF fusion and hybrid search."""
import pytest


def test_rrf_fusion_boosts_documents_in_both_lists():
    """Documents appearing in both vector and BM25 results should rank higher."""
    from search.searcher import reciprocal_rank_fusion

    vector_results = [("doc_a", 0.9), ("doc_b", 0.8), ("doc_c", 0.7)]
    bm25_results = [("doc_b", -1.0), ("doc_d", -2.0), ("doc_a", -3.0)]

    fused = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    fused_ids = [chunk_id for chunk_id, score in fused]

    # doc_a and doc_b appear in both lists, should rank top
    assert fused_ids[0] in ("doc_a", "doc_b")
    assert fused_ids[1] in ("doc_a", "doc_b")
    # doc_c and doc_d appear in only one list, should rank lower
    assert "doc_c" in fused_ids
    assert "doc_d" in fused_ids


def test_rrf_fusion_handles_no_overlap():
    """Fusion with zero overlap should interleave by rank."""
    from search.searcher import reciprocal_rank_fusion

    vector_results = [("doc_a", 0.9), ("doc_b", 0.8)]
    bm25_results = [("doc_c", -1.0), ("doc_d", -2.0)]

    fused = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    assert len(fused) == 4


def test_rrf_fusion_handles_empty_bm25():
    """If BM25 returns nothing, fusion should return vector results only."""
    from search.searcher import reciprocal_rank_fusion

    vector_results = [("doc_a", 0.9), ("doc_b", 0.8)]
    bm25_results = []

    fused = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    assert len(fused) == 2
    assert fused[0][0] == "doc_a"
