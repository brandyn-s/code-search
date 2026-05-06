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

from search.sonnet_reranker import (
    rerank_with_sonnet,
    REASON_OK,
    REASON_EMPTY_INPUT,
    REASON_API_KEY_MISSING,
    REASON_TIMEOUT,
    REASON_RATE_LIMIT,
    REASON_TOO_MANY_FAILURES,
    REASON_HYBRID_PRIOR_FALLBACK,
    REASON_UNEXPECTED_ERROR,
    _ERR_RATE_LIMIT,
    _ERR_TIMEOUT,
    _ERR_HTTP,
)


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


# ─── Plan-2 A1 (PR ?): structured metadata tests ───
# These tests pin the shape of the metadata returned via
# `return_metadata=True`. Schema: {applied: bool, reason: str, latency_ms: int}.
# The reason vocabulary is documented in REASON_* constants. Downstream
# consumers (MCP layer, eval harnesses, telemetry pipelines) must be able
# to discriminate between fallback paths.


def _assert_metadata_shape(meta):
    """Every metadata dict must carry exactly these three keys with right types."""
    assert isinstance(meta, dict), f"metadata must be dict, got {type(meta)}"
    assert set(meta.keys()) == {"applied", "reason", "latency_ms"}, \
        f"metadata keys mismatch: {sorted(meta.keys())}"
    assert isinstance(meta["applied"], bool)
    assert isinstance(meta["reason"], str)
    assert isinstance(meta["latency_ms"], int)
    assert meta["latency_ms"] >= 0


def test_metadata_empty_input(monkeypatch):
    """Empty candidates list yields applied=False, reason=empty_input."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out, meta = rerank_with_sonnet("q", [], top_k=10, return_metadata=True)
    _assert_metadata_shape(meta)
    assert out == []
    assert meta["applied"] is False
    assert meta["reason"] == REASON_EMPTY_INPUT


def test_metadata_api_key_missing(monkeypatch, sample_candidates):
    """No ANTHROPIC_API_KEY → applied=False, reason=api_key_missing."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out, meta = rerank_with_sonnet(
        "q", sample_candidates, top_k=2, return_metadata=True
    )
    _assert_metadata_shape(meta)
    assert out == sample_candidates[:2]  # input order preserved
    assert meta["applied"] is False
    assert meta["reason"] == REASON_API_KEY_MISSING


def test_metadata_default_no_metadata_returned(monkeypatch, sample_candidates):
    """Without return_metadata=True, return type stays list (BC contract)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = rerank_with_sonnet("q", sample_candidates, top_k=2)
    # Must be a plain list, NOT a tuple — preserves backward-compat for all
    # existing callers and tests that don't opt into metadata.
    assert isinstance(out, list)
    assert len(out) == 2


@pytest.mark.asyncio
async def test_metadata_ok_path(monkeypatch):
    """Successful rerank: applied=True, reason=ok."""
    from search.sonnet_reranker import _rerank_async
    candidates = [
        {"chunk_id": f"c{i}", "file_path": f"f{i}.py", "full_content": f"chunk {i}"}
        for i in range(3)
    ]
    scores_iter = iter([5, 3, 8])
    async def fake_score(client, query, file_path, content):
        return next(scores_iter)
    monkeypatch.setattr("search.sonnet_reranker._score_one", fake_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    out, meta = await _rerank_async(
        "q", candidates, top_k=3, timeout=8.0, hybrid_prior_threshold=7,
        return_metadata=True,
    )
    _assert_metadata_shape(meta)
    assert meta["applied"] is True
    assert meta["reason"] == REASON_OK
    assert [c["chunk_id"] for c in out] == ["c2", "c0", "c1"]


@pytest.mark.asyncio
async def test_metadata_hybrid_prior_fallback(monkeypatch):
    """All scores below threshold: applied=False, reason=hybrid_prior_fallback."""
    from search.sonnet_reranker import _rerank_async
    candidates = [
        {"chunk_id": f"c{i}", "file_path": f"f{i}.py", "full_content": f"chunk {i}"}
        for i in range(3)
    ]
    async def fake_score(client, query, file_path, content):
        return 3  # uniformly low, max < threshold=7
    monkeypatch.setattr("search.sonnet_reranker._score_one", fake_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    out, meta = await _rerank_async(
        "q", candidates, top_k=3, timeout=8.0, hybrid_prior_threshold=7,
        return_metadata=True,
    )
    _assert_metadata_shape(meta)
    assert meta["applied"] is False
    assert meta["reason"] == REASON_HYBRID_PRIOR_FALLBACK
    # Hybrid order preserved
    assert [c["chunk_id"] for c in out] == ["c0", "c1", "c2"]


@pytest.mark.asyncio
async def test_metadata_too_many_failures(monkeypatch):
    """All-None scores trip the FAILURE_TOLERANCE threshold."""
    from search.sonnet_reranker import _rerank_async
    candidates = [
        {"chunk_id": f"c{i}", "file_path": f"f{i}.py", "full_content": f"chunk {i}"}
        for i in range(5)
    ]
    async def fake_score(client, query, file_path, content):
        return _ERR_HTTP  # structured failure tag
    monkeypatch.setattr("search.sonnet_reranker._score_one", fake_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    out, meta = await _rerank_async(
        "q", candidates, top_k=3, timeout=8.0, hybrid_prior_threshold=7,
        return_metadata=True,
    )
    _assert_metadata_shape(meta)
    assert meta["applied"] is False
    # All failures classified as _ERR_HTTP → too_many_failures (HTTP isn't
    # rate_limit or timeout)
    assert meta["reason"] == REASON_TOO_MANY_FAILURES
    assert [c["chunk_id"] for c in out] == ["c0", "c1", "c2"]


@pytest.mark.asyncio
async def test_metadata_rate_limit_dominant(monkeypatch):
    """Rate-limit failures should surface as reason=rate_limit (most actionable)."""
    from search.sonnet_reranker import _rerank_async
    candidates = [
        {"chunk_id": f"c{i}", "file_path": f"f{i}.py", "full_content": f"chunk {i}"}
        for i in range(5)
    ]
    async def fake_score(client, query, file_path, content):
        return _ERR_RATE_LIMIT
    monkeypatch.setattr("search.sonnet_reranker._score_one", fake_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    out, meta = await _rerank_async(
        "q", candidates, top_k=3, timeout=8.0, hybrid_prior_threshold=7,
        return_metadata=True,
    )
    _assert_metadata_shape(meta)
    assert meta["applied"] is False
    assert meta["reason"] == REASON_RATE_LIMIT


@pytest.mark.asyncio
async def test_metadata_timeout_dominant(monkeypatch):
    """Per-call timeout failures should surface as reason=timeout."""
    from search.sonnet_reranker import _rerank_async
    candidates = [
        {"chunk_id": f"c{i}", "file_path": f"f{i}.py", "full_content": f"chunk {i}"}
        for i in range(5)
    ]
    async def fake_score(client, query, file_path, content):
        return _ERR_TIMEOUT
    monkeypatch.setattr("search.sonnet_reranker._score_one", fake_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    out, meta = await _rerank_async(
        "q", candidates, top_k=3, timeout=8.0, hybrid_prior_threshold=7,
        return_metadata=True,
    )
    _assert_metadata_shape(meta)
    assert meta["applied"] is False
    assert meta["reason"] == REASON_TIMEOUT


@pytest.mark.asyncio
async def test_metadata_overall_timeout(monkeypatch):
    """asyncio.wait_for timeout (entire batch exceeds budget) → reason=timeout."""
    from search.sonnet_reranker import _rerank_async
    candidates = [
        {"chunk_id": f"c{i}", "file_path": f"f{i}.py", "full_content": f"chunk {i}"}
        for i in range(3)
    ]
    import asyncio as _asyncio
    async def slow_score(client, query, file_path, content):
        await _asyncio.sleep(2.0)  # exceeds tiny timeout below
        return 8
    monkeypatch.setattr("search.sonnet_reranker._score_one", slow_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    out, meta = await _rerank_async(
        "q", candidates, top_k=3, timeout=0.1, hybrid_prior_threshold=7,
        return_metadata=True,
    )
    _assert_metadata_shape(meta)
    assert meta["applied"] is False
    assert meta["reason"] == REASON_TIMEOUT
    assert [c["chunk_id"] for c in out] == ["c0", "c1", "c2"]


def test_metadata_invalid_api_key_returns_metadata_not_raise(monkeypatch, sample_candidates):
    """Invalid API key path returns metadata, doesn't raise.

    The exact reason depends on what anthropic raises (auth error → http_error
    classification → reason ∈ {too_many_failures, rate_limit, timeout}).
    Test the contract: never raises, returns metadata with applied=False.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-invalid-test-key-123")
    monkeypatch.setenv("SONNET_RERANKER_TIMEOUT", "2.0")
    try:
        out, meta = rerank_with_sonnet(
            "q", sample_candidates, top_k=2, return_metadata=True,
        )
    except Exception as e:
        pytest.fail(f"rerank_with_sonnet raised {type(e).__name__}: {e}")
    _assert_metadata_shape(meta)
    assert meta["applied"] is False
    # Reason is one of the failure paths; we don't pin which one
    assert meta["reason"] in {
        REASON_TOO_MANY_FAILURES,
        REASON_RATE_LIMIT,
        REASON_TIMEOUT,
        REASON_UNEXPECTED_ERROR,
    }
    assert len(out) == 2


def test_metadata_latency_ms_is_set(monkeypatch, sample_candidates):
    """latency_ms is always a non-negative integer reflecting wall time."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out, meta = rerank_with_sonnet(
        "q", sample_candidates, top_k=2, return_metadata=True,
    )
    _assert_metadata_shape(meta)
    # API-key-missing path is fast — should be near zero ms
    assert meta["latency_ms"] >= 0
    assert meta["latency_ms"] < 1000  # sanity bound


def test_metadata_reason_vocabulary_is_stable():
    """Reason string constants must remain stable across versions.

    Downstream consumers (telemetry pipelines, MCP clients, eval harnesses)
    pattern-match these strings. Changing the string values is a breaking
    change. This test pins the values; future renames must update here.
    """
    assert REASON_OK == "ok"
    assert REASON_EMPTY_INPUT == "empty_input"
    assert REASON_API_KEY_MISSING == "api_key_missing"
    assert REASON_TIMEOUT == "timeout"
    assert REASON_RATE_LIMIT == "rate_limit"
    assert REASON_TOO_MANY_FAILURES == "too_many_failures"
    assert REASON_HYBRID_PRIOR_FALLBACK == "hybrid_prior_fallback"
    assert REASON_UNEXPECTED_ERROR == "unexpected_error"
