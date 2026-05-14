"""Tests for SONNET_RERANKER_SKIP_THRESHOLD opt-in gate (Phase B'''(b), 2026-05-14).

Pins the contract: when SONNET_RERANKER_SKIP_THRESHOLD is set to a positive
float and the top-1 candidate's similarity_score >= threshold, Sonnet is
skipped entirely. Metadata.reason becomes "skipped_high_confidence".

Default (env unset, or =0) preserves pre-Phase-B'''(b) behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from search.searcher import IntelligentSearcher  # noqa: E402


def _searcher_with_n_candidates(n: int, top_score: float):
    """Build a searcher whose hybrid stage returns n candidates with
    candidates[0].similarity_score equal to top_score.

    The remaining candidates have linearly-decreasing similarity_score so
    the order is stable.
    """
    mock_index = MagicMock()
    mock_embedder = MagicMock()
    fake_embedding = np.random.rand(768).astype(np.float32)
    mock_embedder.embed_query.return_value = fake_embedding

    # Drive the RRF math so candidate 0 ends up with the requested score.
    # The hybrid fusion code multiplies BM25 + dense, applies chunk-type
    # boosts, etc. Easiest path: stub out the search methods to return
    # candidates with explicit scores that are then passed through to
    # SearchResult.similarity_score. The hybrid fusion divides by k and
    # combines, so we use large raw values to drive a target final score.
    mock_index.search.return_value = [
        (f"c{i}", max(top_score * 50 - i * 5, 0.001), {
            "content_preview": f"chunk {i}", "file_path": f"f{i}.py",
            "relative_path": f"f{i}.py", "chunk_type": "function",
        })
        for i in range(n)
    ]
    mock_index.search_bm25.return_value = [
        (f"c{i}", -float(i), {
            "content_preview": f"chunk {i}", "file_path": f"f{i}.py",
            "relative_path": f"f{i}.py", "chunk_type": "function",
        })
        for i in range(n)
    ]
    mock_index.get_stats.return_value = {"files_indexed": 0}
    return IntelligentSearcher(mock_index, mock_embedder)


def test_skip_threshold_unset_preserves_baseline_behavior(monkeypatch):
    """When SONNET_RERANKER_SKIP_THRESHOLD is unset, the rerank path runs
    as before. Without ANTHROPIC_API_KEY the rerank falls back to
    api_key_missing — confirming we hit the rerank branch, not the
    new skip branch.
    """
    monkeypatch.delenv("SONNET_RERANKER_SKIP_THRESHOLD", raising=False)
    monkeypatch.setenv("RERANKER", "sonnet")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    s = _searcher_with_n_candidates(n=20, top_score=0.99)
    s.search(query="x", k=5, search_mode="hybrid")
    assert s.last_reranker_metadata["reason"] == "api_key_missing"


def test_skip_threshold_zero_is_inactive(monkeypatch):
    """SONNET_RERANKER_SKIP_THRESHOLD=0 is the documented "off" sentinel.

    Should not skip; the rerank branch runs normally.
    """
    monkeypatch.setenv("SONNET_RERANKER_SKIP_THRESHOLD", "0")
    monkeypatch.setenv("RERANKER", "sonnet")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    s = _searcher_with_n_candidates(n=20, top_score=0.99)
    s.search(query="x", k=5, search_mode="hybrid")
    assert s.last_reranker_metadata["reason"] == "api_key_missing"


def test_skip_threshold_fires_when_top_score_exceeds_threshold(monkeypatch):
    """High-confidence top-1: SONNET_RERANKER_SKIP_THRESHOLD=0.01 fires
    because the test searcher gives candidate 0 a similarity_score around
    0.5+ after RRF fusion + boosts.
    """
    monkeypatch.setenv("SONNET_RERANKER_SKIP_THRESHOLD", "0.01")
    monkeypatch.setenv("RERANKER", "sonnet")
    # API key irrelevant — the skip branch returns before the rerank call.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")

    s = _searcher_with_n_candidates(n=20, top_score=0.99)
    s.search(query="x", k=5, search_mode="hybrid")
    meta = s.last_reranker_metadata
    assert meta["reason"] == "skipped_high_confidence"
    assert meta["applied"] is False
    assert meta["latency_ms"] == 0
    assert "top_1_score" in meta
    assert "skip_threshold" in meta
    assert meta["skip_threshold"] == 0.01


def test_skip_threshold_does_not_fire_when_top_score_below_threshold(
        monkeypatch):
    """When top-1 score < threshold, Sonnet path runs normally."""
    # Set threshold above any score the test searcher can produce.
    monkeypatch.setenv("SONNET_RERANKER_SKIP_THRESHOLD", "999.0")
    monkeypatch.setenv("RERANKER", "sonnet")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    s = _searcher_with_n_candidates(n=20, top_score=0.5)
    s.search(query="x", k=5, search_mode="hybrid")
    # Should NOT skip; rerank branch runs and falls back to api_key_missing
    assert s.last_reranker_metadata["reason"] == "api_key_missing"


def test_skip_threshold_invalid_value_treated_as_off(monkeypatch):
    """SONNET_RERANKER_SKIP_THRESHOLD=not-a-number doesn't crash; treated
    as off (=0)."""
    monkeypatch.setenv("SONNET_RERANKER_SKIP_THRESHOLD", "not-a-number")
    monkeypatch.setenv("RERANKER", "sonnet")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    s = _searcher_with_n_candidates(n=20, top_score=0.99)
    s.search(query="x", k=5, search_mode="hybrid")
    assert s.last_reranker_metadata["reason"] == "api_key_missing"


def test_skip_threshold_returns_hybrid_top_k(monkeypatch):
    """When skip fires, the returned results are the hybrid top-k in
    their existing order (not reranked)."""
    monkeypatch.setenv("SONNET_RERANKER_SKIP_THRESHOLD", "0.001")
    monkeypatch.setenv("RERANKER", "sonnet")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")

    s = _searcher_with_n_candidates(n=20, top_score=0.99)
    results = s.search(query="x", k=5, search_mode="hybrid")
    # Five results in hybrid (top-k) order.
    assert len(results) == 5
    # Top-1 should be the first candidate (chunk_id c0) — order preserved.
    assert results[0].chunk_id == "c0"
