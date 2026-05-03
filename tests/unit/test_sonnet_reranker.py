"""Tests for sonnet_reranker — verifies always-on graceful-fallback contract.

The reranker MUST NEVER raise. On any failure (no API key, timeout, network
error, parse failure), it returns the input candidates unchanged.
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from search.sonnet_reranker import rerank_with_sonnet


@pytest.fixture
def sample_candidates():
    return [
        {"chunk_id": "c1", "file_path": "src/auth.py",
         "full_content": "def authenticate(user): ..."},
        {"chunk_id": "c2", "file_path": "src/login.py",
         "full_content": "def login_handler(req): ..."},
        {"chunk_id": "c3", "file_path": "tests/test_auth.py",
         "full_content": "def test_authenticate(): ..."},
    ]


def test_empty_input_returns_empty(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert rerank_with_sonnet("query", [], top_k=10) == []


def test_no_api_key_returns_input_unchanged(monkeypatch, sample_candidates):
    """When ANTHROPIC_API_KEY is not set, must return candidates[:top_k] unchanged."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = rerank_with_sonnet("auth", sample_candidates, top_k=2)
    assert len(result) == 2
    assert result == sample_candidates[:2]


def test_top_k_truncation(monkeypatch, sample_candidates):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = rerank_with_sonnet("auth", sample_candidates, top_k=1)
    assert len(result) == 1
    assert result[0] == sample_candidates[0]


def test_top_k_larger_than_candidates(monkeypatch, sample_candidates):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = rerank_with_sonnet("auth", sample_candidates, top_k=100)
    assert len(result) == 3
    assert result == sample_candidates


def test_does_not_raise_on_invalid_api_key(monkeypatch, sample_candidates):
    """Even with an invalid API key, never raise; fall back gracefully."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-invalid-test-key-123")
    monkeypatch.setenv("SONNET_RERANKER_TIMEOUT", "1.0")  # short timeout for test
    # Should NOT raise — should silently fall back to input order
    try:
        result = rerank_with_sonnet("query", sample_candidates, top_k=2)
    except Exception as e:
        pytest.fail(f"rerank_with_sonnet raised {type(e).__name__}: {e}")
    assert len(result) == 2


def test_handles_missing_content_field(monkeypatch):
    """Candidates with no content fields should still be handled gracefully."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    candidates = [
        {"chunk_id": "c1", "file_path": "a.py"},  # no content!
        {"chunk_id": "c2"},  # no file_path either!
    ]
    result = rerank_with_sonnet("query", candidates, top_k=10)
    assert len(result) == 2  # all returned, no crash


def test_preserves_extra_keys(monkeypatch, sample_candidates):
    """Extra keys on candidates (like _orig) must be preserved through reranking."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for c in sample_candidates:
        c["_orig"] = f"marker_{c['chunk_id']}"
    result = rerank_with_sonnet("query", sample_candidates, top_k=10)
    for r in result:
        assert "_orig" in r
        assert r["_orig"].startswith("marker_")


@pytest.mark.asyncio
async def test_hybrid_prior_threshold_low_max_uses_input_order(monkeypatch):
    """When max Sonnet score < threshold, return input order (hybrid prior fallback)."""
    from search.sonnet_reranker import _rerank_async
    candidates = [
        {"chunk_id": f"c{i}", "file_path": f"f{i}.py", "full_content": f"chunk {i}"}
        for i in range(5)
    ]
    # Monkey-patch _score_one to return uniform low scores (max=3, below default threshold=7)
    async def fake_score(client, query, file_path, content):
        return 3
    monkeypatch.setattr("search.sonnet_reranker._score_one", fake_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    result = await _rerank_async("q", candidates, top_k=3, timeout=8.0,
                                  hybrid_prior_threshold=7)
    # Hybrid order preserved (chunks 0, 1, 2 in original order)
    assert [c["chunk_id"] for c in result] == ["c0", "c1", "c2"]


@pytest.mark.asyncio
async def test_hybrid_prior_threshold_high_max_uses_rerank(monkeypatch):
    """When max Sonnet score >= threshold, apply rerank ordering."""
    from search.sonnet_reranker import _rerank_async
    candidates = [
        {"chunk_id": f"c{i}", "file_path": f"f{i}.py", "full_content": f"chunk {i}"}
        for i in range(3)
    ]
    # Mock: c2 scores highest (8), c0 mid (5), c1 low (3) — max=8 above threshold=7
    scores_iter = iter([5, 3, 8])
    async def fake_score(client, query, file_path, content):
        return next(scores_iter)
    monkeypatch.setattr("search.sonnet_reranker._score_one", fake_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    result = await _rerank_async("q", candidates, top_k=3, timeout=8.0,
                                  hybrid_prior_threshold=7)
    # Rerank order: c2 (8), c0 (5), c1 (3)
    assert [c["chunk_id"] for c in result] == ["c2", "c0", "c1"]


@pytest.mark.asyncio
async def test_hybrid_prior_threshold_disabled_when_zero(monkeypatch):
    """threshold=0 disables the hybrid-prior fallback (always rerank)."""
    from search.sonnet_reranker import _rerank_async
    candidates = [
        {"chunk_id": f"c{i}", "file_path": f"f{i}.py", "full_content": f"chunk {i}"}
        for i in range(3)
    ]
    # Mock: uniform low scores (max=2)
    scores_iter = iter([1, 2, 1])
    async def fake_score(client, query, file_path, content):
        return next(scores_iter)
    monkeypatch.setattr("search.sonnet_reranker._score_one", fake_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    # threshold=0: max=2 >= 0 (always true), so rerank applied
    result = await _rerank_async("q", candidates, top_k=3, timeout=8.0,
                                  hybrid_prior_threshold=0)
    # Rerank order: c1 (2), c0 (1), c2 (1) — c0 wins tie (lower input index)
    assert [c["chunk_id"] for c in result] == ["c1", "c0", "c2"]
