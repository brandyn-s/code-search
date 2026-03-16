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


def test_weighted_rrf_bm25_heavy():
    """Higher BM25 weight should boost BM25-only docs over vector-only docs."""
    from search.searcher import reciprocal_rank_fusion

    # doc_a only in vector, doc_b only in bm25, both at rank 0
    vector_results = [("doc_a", 0.9)]
    bm25_results = [("doc_b", -1.0)]

    fused = reciprocal_rank_fusion(
        vector_results,
        bm25_results,
        k=60,
        vector_weight=0.3,
        bm25_weight=0.7,
    )
    fused_ids = [chunk_id for chunk_id, score in fused]

    # With bm25 weight 0.7 vs vector 0.3, doc_b should rank first
    assert fused_ids[0] == "doc_b"
    assert fused_ids[1] == "doc_a"


def test_weighted_rrf_vector_heavy():
    """Higher vector weight should boost vector-only docs over BM25-only docs."""
    from search.searcher import reciprocal_rank_fusion

    vector_results = [("doc_a", 0.9)]
    bm25_results = [("doc_b", -1.0)]

    fused = reciprocal_rank_fusion(
        vector_results,
        bm25_results,
        k=60,
        vector_weight=0.7,
        bm25_weight=0.3,
    )
    fused_ids = [chunk_id for chunk_id, score in fused]

    assert fused_ids[0] == "doc_a"
    assert fused_ids[1] == "doc_b"


def test_chunk_type_boosts_code_mode():
    """In code mode, function chunks should rank above section chunks at same RRF score."""
    from search.searcher import CHUNK_TYPE_BOOSTS

    boosts = CHUNK_TYPE_BOOSTS["code"]
    assert boosts["function"] > boosts["section"]
    assert boosts["method"] > boosts["section"]
    assert boosts["class"] > boosts["section"]
    assert boosts["decorated_definition"] > boosts["section"]


def test_chunk_type_boosts_docs_mode():
    """In docs mode, section chunks should rank above function chunks."""
    from search.searcher import CHUNK_TYPE_BOOSTS

    boosts = CHUNK_TYPE_BOOSTS["docs"]
    assert boosts["section"] > boosts["function"]
    assert boosts["section"] > boosts["method"]


def test_expand_query_adds_synonyms():
    """Query expansion should add known code-domain synonyms."""
    from search.searcher import expand_code_query

    expanded = expand_code_query("authentication logic")
    assert "auth" in expanded.lower()
    assert "oauth" in expanded.lower() or "jwt" in expanded.lower()


def test_expand_query_passthrough_unknown():
    """Queries with no known synonyms should pass through unchanged."""
    from search.searcher import expand_code_query

    result = expand_code_query("foobar baz")
    assert result == "foobar baz"


def test_expand_query_nix_domain():
    """Nix-domain synonyms should expand network/service queries."""
    from search.searcher import expand_code_query

    expanded = expand_code_query("network configuration")
    assert "networking" in expanded.lower()
    assert "firewall" in expanded.lower() or "interface" in expanded.lower()

    expanded2 = expand_code_query("service daemon")
    assert "systemd" in expanded2.lower()


def test_expand_query_corsair_services():
    """Corsair service synonyms should expand domain queries to daemon names."""
    from search.searcher import expand_code_query

    expanded = expand_code_query("sensor navigation GPS")
    assert "internal-svc-62" in expanded.lower() or "internal-svc-51" in expanded.lower()

    expanded2 = expand_code_query("motor propulsion engine")
    assert "internal-svc-12" in expanded2.lower()

    expanded3 = expand_code_query("camera video perception")
    assert "internal-svc-26" in expanded3.lower() or "internal-svc-27" in expanded3.lower()

    expanded4 = expand_code_query("communication radio mesh")
    assert "internal-svc-44" in expanded4.lower() or "internal-svc-41" in expanded4.lower()
