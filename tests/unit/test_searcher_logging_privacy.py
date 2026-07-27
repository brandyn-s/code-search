"""Search orchestration logs must not expose query text by default."""

from __future__ import annotations

import logging

from search.searcher import IntelligentSearcher


class _StubEmbedder:
    def embed_query(self, _query: str) -> list[float]:
        return [0.0]


class _StubIndexManager:
    def search(self, *_args, **_kwargs):
        return []


def test_searcher_redacts_cache_key_and_optimized_query_by_default(
    monkeypatch,
    caplog,
):
    monkeypatch.delenv("CODE_SEARCH_LOG_QUERY_TEXT", raising=False)
    searcher = IntelligentSearcher(_StubIndexManager(), _StubEmbedder())
    cache_query = "sentinel private cache query"
    semantic_query = "sentinel private optimized query"

    with caplog.at_level(logging.DEBUG, logger="search.searcher"):
        searcher._get_query_embedding(cache_query)
        searcher._get_query_embedding(cache_query)
        result = searcher._semantic_search(
            semantic_query,
            k=1,
            context_depth=0,
        )

    assert result == []
    assert cache_query not in caplog.text
    assert semantic_query not in caplog.text
    assert "normalized_query redacted" in caplog.text
    assert "optimized_query redacted" in caplog.text


def test_searcher_plaintext_logs_require_exact_operator_opt_in(
    monkeypatch,
    caplog,
):
    monkeypatch.setenv("CODE_SEARCH_LOG_QUERY_TEXT", "on")
    searcher = IntelligentSearcher(_StubIndexManager(), _StubEmbedder())
    cache_query = "sentinel opted in cache query"
    semantic_query = "sentinel opted in optimized query"

    with caplog.at_level(logging.DEBUG, logger="search.searcher"):
        searcher._get_query_embedding(cache_query)
        searcher._get_query_embedding(cache_query)
        searcher._semantic_search(
            semantic_query,
            k=1,
            context_depth=0,
        )

    assert cache_query in caplog.text
    assert semantic_query in caplog.text
