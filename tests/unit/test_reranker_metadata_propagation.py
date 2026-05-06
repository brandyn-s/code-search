"""Tests for IntelligentSearcher.last_reranker_metadata propagation (Plan-2 A1).

Pin the contract: every search() call leaves `last_reranker_metadata` populated
with {applied, reason, latency_ms}. The reason discriminates between:
- Sonnet was invoked successfully (REASON_OK)
- Sonnet was invoked but fell back (api_key_missing, timeout, rate_limit, etc.)
- Sonnet was NOT invoked because of mode (not_invoked_keyword_mode,
    not_invoked_semantic_mode, not_invoked_cross_encoder_mode, disabled_by_env,
    not_invoked_no_candidates, not_invoked_insufficient_candidates)

The MCP layer reads this attribute to surface `_metadata.reranker` in the
search response, so LLM agents can detect silent fallback.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from search.searcher import IntelligentSearcher


@pytest.fixture
def mock_searcher():
    """Build an IntelligentSearcher with empty-result mocks."""
    mock_index = MagicMock()
    mock_embedder = MagicMock()
    fake_embedding = np.random.rand(768).astype(np.float32)
    mock_embedder.embed_query.return_value = fake_embedding
    mock_index.search.return_value = []
    mock_index.search_bm25.return_value = []
    mock_index.get_stats.return_value = {"files_indexed": 0}
    return IntelligentSearcher(mock_index, mock_embedder)


def _assert_metadata_shape(meta):
    """Same contract as test_sonnet_reranker.py — duplicated for module isolation."""
    assert isinstance(meta, dict)
    assert set(meta.keys()) == {"applied", "reason", "latency_ms"}
    assert isinstance(meta["applied"], bool)
    assert isinstance(meta["reason"], str)
    assert isinstance(meta["latency_ms"], int)
    assert meta["latency_ms"] >= 0


def test_initial_metadata_present_before_any_search(mock_searcher):
    """A fresh searcher must have last_reranker_metadata initialized so the
    MCP layer can always read it without an AttributeError."""
    _assert_metadata_shape(mock_searcher.last_reranker_metadata)
    assert mock_searcher.last_reranker_metadata["applied"] is False
    assert mock_searcher.last_reranker_metadata["reason"] == "not_invoked"


def test_keyword_mode_sets_not_invoked_keyword_mode(mock_searcher):
    """search_mode=keyword does not invoke Sonnet."""
    mock_searcher.search(query="find auth", k=5, search_mode="keyword")
    meta = mock_searcher.last_reranker_metadata
    _assert_metadata_shape(meta)
    assert meta["applied"] is False
    assert meta["reason"] == "not_invoked_keyword_mode"


def test_semantic_mode_sets_not_invoked_semantic_mode(mock_searcher):
    """search_mode=semantic does not invoke Sonnet."""
    mock_searcher.search(query="find auth", k=5, search_mode="semantic")
    meta = mock_searcher.last_reranker_metadata
    _assert_metadata_shape(meta)
    assert meta["applied"] is False
    assert meta["reason"] == "not_invoked_semantic_mode"


def test_hybrid_mode_no_candidates_sets_not_invoked_no_candidates(mock_searcher):
    """Hybrid mode with empty index → no candidates → reason=not_invoked_no_candidates."""
    mock_searcher.search(query="find auth", k=5, search_mode="hybrid")
    meta = mock_searcher.last_reranker_metadata
    _assert_metadata_shape(meta)
    assert meta["applied"] is False
    # No candidates path
    assert meta["reason"] == "not_invoked_no_candidates"


def test_hybrid_mode_disabled_by_env(monkeypatch, mock_searcher):
    """RERANKER=off explicitly disables Sonnet."""
    monkeypatch.setenv("RERANKER", "off")
    # Set up some fake candidates so we exercise the "off" branch with
    # candidates present (otherwise we'd hit no_candidates first).
    mock_searcher.index_manager.search.return_value = [
        ("c1", 0.9, {"content_preview": "x", "file_path": "a.py", "relative_path": "a.py"}),
    ]
    mock_searcher.index_manager.search_bm25.return_value = [
        ("c1", -1.0, {"content_preview": "x", "file_path": "a.py", "relative_path": "a.py"}),
    ]
    mock_searcher.search(query="find auth", k=5, search_mode="hybrid")
    meta = mock_searcher.last_reranker_metadata
    _assert_metadata_shape(meta)
    assert meta["applied"] is False
    assert meta["reason"] == "disabled_by_env"


def test_hybrid_mode_sonnet_api_key_missing(monkeypatch, mock_searcher):
    """RERANKER=sonnet (default) with no ANTHROPIC_API_KEY → api_key_missing."""
    monkeypatch.setenv("RERANKER", "sonnet")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Provide enough candidates that len(candidates) > k to enter the rerank
    # branch (otherwise we hit not_invoked_insufficient_candidates).
    mock_searcher.index_manager.search.return_value = [
        (f"c{i}", 0.9 - i * 0.05, {
            "content_preview": f"chunk {i}", "file_path": f"f{i}.py",
            "relative_path": f"f{i}.py", "chunk_type": "function",
        })
        for i in range(20)
    ]
    mock_searcher.index_manager.search_bm25.return_value = [
        (f"c{i}", -float(i), {
            "content_preview": f"chunk {i}", "file_path": f"f{i}.py",
            "relative_path": f"f{i}.py", "chunk_type": "function",
        })
        for i in range(20)
    ]
    mock_searcher.search(query="find auth", k=5, search_mode="hybrid")
    meta = mock_searcher.last_reranker_metadata
    _assert_metadata_shape(meta)
    assert meta["applied"] is False
    assert meta["reason"] == "api_key_missing"


def test_hybrid_mode_cross_encoder(monkeypatch, mock_searcher):
    """RERANKER=cross-encoder explicitly takes the legacy path."""
    monkeypatch.setenv("RERANKER", "cross-encoder")
    # Only test that metadata is set with the right reason. Mocking the
    # cross-encoder is heavy; simulate by patching rerank_results to identity.
    mock_searcher.index_manager.search.return_value = [
        ("c1", 0.9, {"content_preview": "x", "file_path": "a.py", "relative_path": "a.py"}),
    ]
    mock_searcher.index_manager.search_bm25.return_value = [
        ("c1", -1.0, {"content_preview": "x", "file_path": "a.py", "relative_path": "a.py"}),
    ]
    monkeypatch.setattr(
        "search.reranker.rerank_results",
        lambda _q, items, top_k: items[:top_k],
    )
    mock_searcher.search(query="find auth", k=5, search_mode="hybrid")
    meta = mock_searcher.last_reranker_metadata
    _assert_metadata_shape(meta)
    assert meta["applied"] is False
    assert meta["reason"] == "not_invoked_cross_encoder_mode"


def test_metadata_resets_between_searches(mock_searcher):
    """A second search() call must produce fresh metadata, not stale from prior call."""
    mock_searcher.search(query="q1", k=5, search_mode="keyword")
    assert mock_searcher.last_reranker_metadata["reason"] == "not_invoked_keyword_mode"
    mock_searcher.search(query="q2", k=5, search_mode="semantic")
    assert mock_searcher.last_reranker_metadata["reason"] == "not_invoked_semantic_mode"
