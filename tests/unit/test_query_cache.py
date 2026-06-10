"""Tests for query-level embedding caching."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from search.searcher import IntelligentSearcher


def test_identical_queries_use_cache():
    """Second identical query should not call embedder again."""
    mock_index = MagicMock()
    mock_embedder = MagicMock()
    fake_embedding = np.random.rand(768).astype(np.float32)
    mock_embedder.embed_query.return_value = fake_embedding

    mock_index.search.return_value = []
    mock_index.search_bm25.return_value = []

    searcher = IntelligentSearcher(mock_index, mock_embedder)

    # First call
    searcher.search(query="find authentication handler", k=5)
    first_count = mock_embedder.embed_query.call_count

    # Second identical call
    searcher.search(query="find authentication handler", k=5)
    second_count = mock_embedder.embed_query.call_count

    assert second_count == first_count, (
        f"Embedder called {second_count - first_count} extra times for cached query"
    )


def test_different_queries_bypass_cache():
    """Different queries should each call embedder."""
    mock_index = MagicMock()
    mock_embedder = MagicMock()
    fake_embedding = np.random.rand(768).astype(np.float32)
    mock_embedder.embed_query.return_value = fake_embedding
    mock_index.search.return_value = []
    mock_index.search_bm25.return_value = []

    searcher = IntelligentSearcher(mock_index, mock_embedder)

    searcher.search(query="find auth handler", k=5)
    searcher.search(query="database connection pool", k=5)

    assert mock_embedder.embed_query.call_count == 2


def test_cache_cleared_on_invalidation():
    """Cache should be invalidated when explicitly cleared."""
    mock_index = MagicMock()
    mock_embedder = MagicMock()
    fake_embedding = np.random.rand(768).astype(np.float32)
    mock_embedder.embed_query.return_value = fake_embedding
    mock_index.search.return_value = []
    mock_index.search_bm25.return_value = []

    searcher = IntelligentSearcher(mock_index, mock_embedder)

    searcher.search(query="find auth handler", k=5)
    searcher.clear_cache()
    searcher.search(query="find auth handler", k=5)

    assert mock_embedder.embed_query.call_count == 2


def test_cache_is_case_insensitive():
    """Queries differing only by case should share cache."""
    mock_index = MagicMock()
    mock_embedder = MagicMock()
    fake_embedding = np.random.rand(768).astype(np.float32)
    mock_embedder.embed_query.return_value = fake_embedding
    mock_index.search.return_value = []
    mock_index.search_bm25.return_value = []

    searcher = IntelligentSearcher(mock_index, mock_embedder)

    searcher.search(query="Find Auth Handler", k=5)
    searcher.search(query="find auth handler", k=5)

    assert mock_embedder.embed_query.call_count == 1


def test_cache_is_lru_bounded():
    """The per-searcher cache must evict oldest entries past the cap — it
    was unbounded, a slow leak in a long-lived MCP server process."""
    mock_index = MagicMock()
    mock_embedder = MagicMock()
    fake_embedding = np.random.rand(8).astype(np.float32)
    mock_embedder.embed_query.return_value = fake_embedding

    searcher = IntelligentSearcher(mock_index, mock_embedder)
    cap = searcher._QUERY_CACHE_MAX

    for i in range(cap + 50):
        searcher._get_query_embedding(f"query number {i}")

    assert len(searcher._query_embedding_cache) <= cap

    # Oldest entry evicted; newest retained.
    assert "query number 0" not in searcher._query_embedding_cache
    assert f"query number {cap + 49}" in searcher._query_embedding_cache
