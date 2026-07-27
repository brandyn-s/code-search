"""Unit contracts for search policy modules extracted from ``searcher``."""

from __future__ import annotations


def test_searcher_reexports_extracted_policy_symbols():
    from search.fusion import CHUNK_TYPE_BOOSTS, reciprocal_rank_fusion
    from search.query_expansion import (
        CODE_SYNONYMS,
        _active_synonyms,
        _query_stem,
        expand_code_query,
    )
    from search.result_models import SearchResult
    import search.searcher as legacy

    assert legacy.reciprocal_rank_fusion is reciprocal_rank_fusion
    assert legacy.CHUNK_TYPE_BOOSTS is CHUNK_TYPE_BOOSTS
    assert legacy.CODE_SYNONYMS is CODE_SYNONYMS
    assert legacy._active_synonyms is _active_synonyms
    assert legacy._query_stem is _query_stem
    assert legacy.expand_code_query is expand_code_query
    assert legacy.SearchResult is SearchResult
