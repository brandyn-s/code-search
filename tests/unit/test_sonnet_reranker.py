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
    async def fake_score(client, query, file_path, content, extra_clauses=""):
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
    async def fake_score(client, query, file_path, content, extra_clauses=""):
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
    async def fake_score(client, query, file_path, content, extra_clauses=""):
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
    async def fake_score(client, query, file_path, content, extra_clauses=""):
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
    async def fake_score(client, query, file_path, content, extra_clauses=""):
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
    async def fake_score(client, query, file_path, content, extra_clauses=""):
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
    async def fake_score(client, query, file_path, content, extra_clauses=""):
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
    async def fake_score(client, query, file_path, content, extra_clauses=""):
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
    async def slow_score(client, query, file_path, content, extra_clauses=""):
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


# ─── Plan D1-Pass-2 A.1 (PR ?): latency diagnostic + concurrency cap ───
# These tests pin the concurrency-limit semaphore behavior and the
# [ANTHROPIC_DIAG] log emission. Used to diagnose Anthropic per-call latency
# regressions without requiring API spend.


@pytest.mark.asyncio
async def test_concurrency_limit_caps_in_flight_calls(monkeypatch):
    """ANTHROPIC_CONCURRENCY_LIMIT=2 caps simultaneous _score_one calls to 2."""
    import asyncio as _asyncio
    from search.sonnet_reranker import _rerank_async

    candidates = [
        {"chunk_id": f"c{i}", "file_path": f"f{i}.py", "full_content": f"chunk {i}"}
        for i in range(6)
    ]

    # Track concurrent call count via a shared counter
    concurrent_now = 0
    max_concurrent = 0
    counter_lock = _asyncio.Lock()

    async def slow_score(client, query, file_path, content, extra_clauses=""):
        nonlocal concurrent_now, max_concurrent
        async with counter_lock:
            concurrent_now += 1
            if concurrent_now > max_concurrent:
                max_concurrent = concurrent_now
        await _asyncio.sleep(0.05)
        async with counter_lock:
            concurrent_now -= 1
        return 8

    monkeypatch.setattr("search.sonnet_reranker._score_one", slow_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_CONCURRENCY_LIMIT", "2")
    out = await _rerank_async(
        "q", candidates, top_k=6, timeout=10.0, hybrid_prior_threshold=7,
    )
    assert len(out) == 6
    # max_concurrent must NEVER exceed the cap
    assert max_concurrent <= 2, f"max_concurrent={max_concurrent} exceeded cap=2"


@pytest.mark.asyncio
async def test_concurrency_limit_unset_is_unbounded(monkeypatch):
    """When ANTHROPIC_CONCURRENCY_LIMIT is unset, all calls run concurrently."""
    import asyncio as _asyncio
    from search.sonnet_reranker import _rerank_async

    candidates = [
        {"chunk_id": f"c{i}", "file_path": f"f{i}.py", "full_content": f"chunk {i}"}
        for i in range(6)
    ]

    concurrent_now = 0
    max_concurrent = 0
    counter_lock = _asyncio.Lock()

    async def slow_score(client, query, file_path, content, extra_clauses=""):
        nonlocal concurrent_now, max_concurrent
        async with counter_lock:
            concurrent_now += 1
            if concurrent_now > max_concurrent:
                max_concurrent = concurrent_now
        await _asyncio.sleep(0.05)
        async with counter_lock:
            concurrent_now -= 1
        return 8

    monkeypatch.setattr("search.sonnet_reranker._score_one", slow_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_CONCURRENCY_LIMIT", raising=False)
    out = await _rerank_async(
        "q", candidates, top_k=6, timeout=10.0, hybrid_prior_threshold=7,
    )
    assert len(out) == 6
    # Without a cap, all 6 should run in parallel
    assert max_concurrent == 6, f"max_concurrent={max_concurrent}, expected 6"


@pytest.mark.asyncio
async def test_concurrency_limit_invalid_value_falls_back_to_unbounded(monkeypatch):
    """ANTHROPIC_CONCURRENCY_LIMIT=garbage doesn't break the reranker."""
    import asyncio as _asyncio
    from search.sonnet_reranker import _rerank_async

    candidates = [
        {"chunk_id": f"c{i}", "file_path": f"f{i}.py", "full_content": f"chunk {i}"}
        for i in range(3)
    ]

    async def fast_score(client, query, file_path, content, extra_clauses=""):
        return 8

    monkeypatch.setattr("search.sonnet_reranker._score_one", fast_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_CONCURRENCY_LIMIT", "not-a-number")
    out = await _rerank_async(
        "q", candidates, top_k=3, timeout=5.0, hybrid_prior_threshold=7,
    )
    assert len(out) == 3
    # Reranker still produces a sensible ordering even with bad env value


@pytest.mark.asyncio
async def test_concurrency_limit_zero_is_treated_as_unbounded(monkeypatch):
    """ANTHROPIC_CONCURRENCY_LIMIT=0 is treated as unbounded (semaphore not created)."""
    from search.sonnet_reranker import _rerank_async

    candidates = [
        {"chunk_id": f"c{i}", "file_path": f"f{i}.py", "full_content": f"chunk {i}"}
        for i in range(3)
    ]

    async def fast_score(client, query, file_path, content, extra_clauses=""):
        return 8

    monkeypatch.setattr("search.sonnet_reranker._score_one", fast_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_CONCURRENCY_LIMIT", "0")
    out = await _rerank_async(
        "q", candidates, top_k=3, timeout=5.0, hybrid_prior_threshold=7,
    )
    assert len(out) == 3


def test_diag_log_emits_per_call(monkeypatch, caplog):
    """Each [ANTHROPIC_DIAG] log line records a per-call timing snapshot.

    Uses a real _score_one call against a mock client (the SDK call inside
    _score_one raises, classified as _err_http; we still expect the diag
    line to fire from the finally block).
    """
    import asyncio as _asyncio
    import logging as _logging
    from search import sonnet_reranker as sr

    class FakeClient:
        class messages:
            @staticmethod
            async def create(**kwargs):
                # Simulate a network round-trip + force an error to trigger
                # the failure-path diag emission
                await _asyncio.sleep(0.001)
                raise RuntimeError("simulated SDK error")

    caplog.set_level(_logging.INFO, logger="search.sonnet_reranker")
    result = _asyncio.run(sr._score_one(FakeClient(), "q", "f.py", "content"))
    # On error, _score_one returns one of the _ERR_* tags
    assert isinstance(result, str) and result.startswith("_err_")

    diag_records = [r for r in caplog.records if "[ANTHROPIC_DIAG]" in r.getMessage()]
    assert len(diag_records) == 1, (
        f"expected exactly 1 diag log line, got {len(diag_records)}"
    )
    msg = diag_records[0].getMessage()
    assert "model=" in msg
    assert "total_ms=" in msg
    assert "in_flight=" in msg
    assert "attempt=" in msg
    assert "outcome=" in msg


def test_diag_log_records_outcome_ok_on_success(monkeypatch, caplog):
    """A successful _score_one call records outcome=ok in the diag line."""
    import asyncio as _asyncio
    import logging as _logging
    from search import sonnet_reranker as sr

    class FakeBlock:
        type = "text"
        text = '{"score": 8, "reasoning": "match"}'

    class FakeResponse:
        content = [FakeBlock()]

    class FakeClient:
        class messages:
            @staticmethod
            async def create(**kwargs):
                await _asyncio.sleep(0.001)
                return FakeResponse()

    caplog.set_level(_logging.INFO, logger="search.sonnet_reranker")
    result = _asyncio.run(sr._score_one(FakeClient(), "q", "f.py", "content"))
    assert result == 8

    diag_records = [r for r in caplog.records if "[ANTHROPIC_DIAG]" in r.getMessage()]
    assert len(diag_records) == 1
    assert "outcome=ok" in diag_records[0].getMessage()


# ─── Plan D1-Pass-2 B.1 (PR ?): SDK retry/timeout + classification ───
# Phase B.1 mitigations diagnosed in PR #133:
#   - Lower SDK max_retries from default 2 → 1 to cut retry-exhaustion wall
#     time without disabling transient-error recovery.
#   - Add per-call SDK timeout=12.0 to bound individual call wall.
#   - Walk the exception cause chain in _classify_call_error so wrapped
#     rate-limit errors surface as REASON_RATE_LIMIT instead of generic _ERR_HTTP.


def test_classify_walks_cause_chain_for_rate_limit(monkeypatch):
    """A wrapper exception whose __cause__ is a RateLimitError-like classifies
    as _ERR_RATE_LIMIT (not _ERR_HTTP), even when the wrapper's own name and
    message don't mention 'rate' or 'limit'.
    """
    from search.sonnet_reranker import _classify_call_error, _ERR_RATE_LIMIT

    class FakeRateLimitError(Exception):
        pass
    FakeRateLimitError.__name__ = "RateLimitError"

    class GenericWrapper(Exception):
        pass

    inner = FakeRateLimitError("429 Too Many Requests")
    outer = GenericWrapper("retry exhausted")
    outer.__cause__ = inner

    assert _classify_call_error(outer) == _ERR_RATE_LIMIT


def test_classify_walks_cause_chain_for_timeout():
    """A wrapper exception whose __cause__ is a timeout classifies as _ERR_TIMEOUT."""
    from search.sonnet_reranker import _classify_call_error, _ERR_TIMEOUT

    class FakeTimeout(Exception):
        pass
    FakeTimeout.__name__ = "APITimeoutError"

    class FakeAPIConnectionError(Exception):
        pass

    inner = FakeTimeout("read timeout")
    outer = FakeAPIConnectionError("connection failed")
    outer.__cause__ = inner

    assert _classify_call_error(outer) == _ERR_TIMEOUT


def test_classify_falls_back_to_http_when_no_signal():
    """A pure HTTP error (no rate-limit/timeout signal in the chain) classifies as _ERR_HTTP."""
    from search.sonnet_reranker import _classify_call_error, _ERR_HTTP

    class FakeHTTPError(Exception):
        pass

    e = FakeHTTPError("500 Internal Server Error")
    assert _classify_call_error(e) == _ERR_HTTP


def test_classify_avoids_infinite_loop_on_circular_cause():
    """A circular __cause__ chain doesn't hang; falls through to _ERR_HTTP."""
    from search.sonnet_reranker import _classify_call_error, _ERR_HTTP

    a = Exception("a")
    b = Exception("b")
    a.__cause__ = b
    b.__cause__ = a  # circular!

    assert _classify_call_error(a) == _ERR_HTTP


def test_resolve_per_call_timeout_default(monkeypatch):
    """Default per-call timeout is the documented constant."""
    from search.sonnet_reranker import (
        _resolve_per_call_timeout, DEFAULT_SDK_PER_CALL_TIMEOUT_S,
    )

    monkeypatch.delenv("ANTHROPIC_PER_CALL_TIMEOUT_S", raising=False)
    assert _resolve_per_call_timeout() == DEFAULT_SDK_PER_CALL_TIMEOUT_S


def test_resolve_per_call_timeout_env_override(monkeypatch):
    """Env var ANTHROPIC_PER_CALL_TIMEOUT_S overrides the default."""
    from search.sonnet_reranker import _resolve_per_call_timeout

    monkeypatch.setenv("ANTHROPIC_PER_CALL_TIMEOUT_S", "5.5")
    assert _resolve_per_call_timeout() == 5.5


def test_resolve_per_call_timeout_invalid_falls_back(monkeypatch):
    """Bad env values fall back to the default (no crash)."""
    from search.sonnet_reranker import (
        _resolve_per_call_timeout, DEFAULT_SDK_PER_CALL_TIMEOUT_S,
    )

    monkeypatch.setenv("ANTHROPIC_PER_CALL_TIMEOUT_S", "not-a-number")
    assert _resolve_per_call_timeout() == DEFAULT_SDK_PER_CALL_TIMEOUT_S
    monkeypatch.setenv("ANTHROPIC_PER_CALL_TIMEOUT_S", "-1")
    assert _resolve_per_call_timeout() == DEFAULT_SDK_PER_CALL_TIMEOUT_S


def test_resolve_sdk_max_retries_default(monkeypatch):
    """Default SDK max_retries is the documented constant."""
    from search.sonnet_reranker import _resolve_sdk_max_retries, DEFAULT_SDK_MAX_RETRIES

    monkeypatch.delenv("ANTHROPIC_MAX_RETRIES", raising=False)
    assert _resolve_sdk_max_retries() == DEFAULT_SDK_MAX_RETRIES


def test_resolve_sdk_max_retries_env_override(monkeypatch):
    """Env var ANTHROPIC_MAX_RETRIES overrides the default."""
    from search.sonnet_reranker import _resolve_sdk_max_retries

    monkeypatch.setenv("ANTHROPIC_MAX_RETRIES", "0")
    assert _resolve_sdk_max_retries() == 0
    monkeypatch.setenv("ANTHROPIC_MAX_RETRIES", "3")
    assert _resolve_sdk_max_retries() == 3


def test_resolve_sdk_max_retries_invalid_falls_back(monkeypatch):
    """Bad env values fall back to the default."""
    from search.sonnet_reranker import _resolve_sdk_max_retries, DEFAULT_SDK_MAX_RETRIES

    monkeypatch.setenv("ANTHROPIC_MAX_RETRIES", "not-a-number")
    assert _resolve_sdk_max_retries() == DEFAULT_SDK_MAX_RETRIES
    monkeypatch.setenv("ANTHROPIC_MAX_RETRIES", "-1")
    assert _resolve_sdk_max_retries() == DEFAULT_SDK_MAX_RETRIES


def test_score_one_passes_per_call_timeout_to_sdk(monkeypatch):
    """_score_one must pass the resolved timeout into messages.create.

    Verifies the SDK gets a per-call cap; without this, the SDK falls back
    to its default (no per-call timeout, just the global client timeout).
    """
    import asyncio as _asyncio

    from search import sonnet_reranker as sr

    captured: dict = {}

    class FakeBlock:
        type = "text"
        text = '{"score": 5}'

    class FakeResponse:
        content = [FakeBlock()]

    class FakeClient:
        class messages:
            @staticmethod
            async def create(**kwargs):
                captured.update(kwargs)
                return FakeResponse()

    monkeypatch.setenv("ANTHROPIC_PER_CALL_TIMEOUT_S", "7.5")
    result = _asyncio.run(sr._score_one(FakeClient(), "q", "f.py", "c"))
    assert result == 5
    assert captured.get("timeout") == 7.5


# ─── Per-path hybrid-prior threshold overrides (D3, 2026-05-09) ───
# Tests for SONNET_RERANKER_HYBRID_PRIOR_THRESHOLD_PATH_OVERRIDES env var.
# Bootstrap CI on 2026-05-09 PSM eval data showed sonnet rerank effect splits
# by subproject (billing: -0.0695 hurts, webapp: +0.173 helps; both CIs
# excluded zero). Per-path override lets the threshold tune by domain.


def test_parse_path_overrides_empty():
    from search.sonnet_reranker import _parse_path_overrides

    assert _parse_path_overrides(None) == {}
    assert _parse_path_overrides("") == {}


def test_parse_path_overrides_valid():
    from search.sonnet_reranker import _parse_path_overrides

    result = _parse_path_overrides('{"billing/": 11, "webapp/": 4}')
    assert result == {"billing/": 11, "webapp/": 4}


def test_parse_path_overrides_normalizes_separators():
    from search.sonnet_reranker import _parse_path_overrides

    # Windows-style path prefix gets normalized to forward-slash
    result = _parse_path_overrides('{"billing\\\\src/": 9}')
    assert result == {"billing/src/": 9}


def test_parse_path_overrides_malformed_json_returns_empty():
    from search.sonnet_reranker import _parse_path_overrides

    assert _parse_path_overrides("{not json}") == {}
    assert _parse_path_overrides("[1, 2, 3]") == {}  # not a dict
    assert _parse_path_overrides("null") == {}


def test_parse_path_overrides_skips_invalid_entries():
    from search.sonnet_reranker import _parse_path_overrides

    # str->non-int entries dropped; valid entries kept
    result = _parse_path_overrides('{"good/": 7, "bad/": "not-int", "ok/": 9}')
    assert result == {"good/": 7, "ok/": 9}


def test_effective_threshold_no_overrides_returns_base():
    from search.sonnet_reranker import _effective_threshold

    candidates = [{"file_path": "billing/src/foo.rs"}]
    assert _effective_threshold(candidates, base_threshold=6, path_overrides={}) == 6


def test_effective_threshold_no_match_returns_base():
    from search.sonnet_reranker import _effective_threshold

    candidates = [{"file_path": "nix/modules/foo.nix"}]
    overrides = {"billing/": 11}
    assert _effective_threshold(candidates, base_threshold=6, path_overrides=overrides) == 6


def test_effective_threshold_match_raises_above_base():
    from search.sonnet_reranker import _effective_threshold

    candidates = [{"file_path": "billing/src/foo.rs"}]
    overrides = {"billing/": 11}
    assert _effective_threshold(candidates, base_threshold=6, path_overrides=overrides) == 11


def test_effective_threshold_takes_max_across_matches():
    """Mixed cohort: cohort threshold = MAX (most restrictive) match."""
    from search.sonnet_reranker import _effective_threshold

    candidates = [
        {"file_path": "billing/src/foo.rs"},      # → 11
        {"file_path": "webapp/src/page.tsx"},  # → 4
    ]
    overrides = {"billing/": 11, "webapp/": 4}
    # Conservative: pick the higher (sonnet hurts billing; better to fall back)
    assert _effective_threshold(candidates, base_threshold=6, path_overrides=overrides) == 11


def test_effective_threshold_below_base_does_not_lower():
    """Override BELOW base never lowers the cohort threshold."""
    from search.sonnet_reranker import _effective_threshold

    candidates = [{"file_path": "webapp/src/page.tsx"}]
    overrides = {"webapp/": 4}  # below base
    # Lowering thresholds via overrides isn't yet supported (would need a
    # different design — the current rule uses MAX). Cohort stays at base.
    assert _effective_threshold(candidates, base_threshold=6, path_overrides=overrides) == 6


def test_effective_threshold_falls_back_to_alt_path_keys():
    """_effective_threshold checks file_path → file → relative_path in that order."""
    from search.sonnet_reranker import _effective_threshold

    candidates = [
        {"file": "billing/src/foo.rs"},          # no file_path key
        {"relative_path": "billing/src/bar.rs"},  # no file or file_path key
    ]
    overrides = {"billing/": 11}
    assert _effective_threshold(candidates, base_threshold=6, path_overrides=overrides) == 11


def test_effective_threshold_handles_windows_separators_in_paths():
    """Candidate paths may use Windows backslashes; normalize before matching."""
    from search.sonnet_reranker import _effective_threshold

    candidates = [{"file_path": "billing\\src\\foo.rs"}]
    overrides = {"billing/": 11}
    assert _effective_threshold(candidates, base_threshold=6, path_overrides=overrides) == 11


# ─── Phase A pool size (Plan 8-Phase Arc, 2026-05-09) ───
# Tests for SONNET_RERANKER_POOL_SIZE env var. The pool sizing limits how
# many candidates Sonnet scores; the rest are appended in hybrid order.


def test_resolve_pool_size_default(monkeypatch):
    """Default pool size is 0 (unbounded — score all candidates)."""
    from search.sonnet_reranker import _resolve_pool_size, DEFAULT_RERANK_POOL_SIZE

    monkeypatch.delenv("SONNET_RERANKER_POOL_SIZE", raising=False)
    assert _resolve_pool_size() == DEFAULT_RERANK_POOL_SIZE
    assert DEFAULT_RERANK_POOL_SIZE == 0  # contract: 0 = unbounded


def test_resolve_pool_size_env_override(monkeypatch):
    """Env var SONNET_RERANKER_POOL_SIZE overrides the default."""
    from search.sonnet_reranker import _resolve_pool_size

    monkeypatch.setenv("SONNET_RERANKER_POOL_SIZE", "5")
    assert _resolve_pool_size() == 5
    monkeypatch.setenv("SONNET_RERANKER_POOL_SIZE", "10")
    assert _resolve_pool_size() == 10


def test_resolve_pool_size_invalid_falls_back(monkeypatch):
    """Bad env values fall back to the default (0/unbounded)."""
    from search.sonnet_reranker import _resolve_pool_size

    monkeypatch.setenv("SONNET_RERANKER_POOL_SIZE", "not-a-number")
    assert _resolve_pool_size() == 0
    monkeypatch.setenv("SONNET_RERANKER_POOL_SIZE", "-1")
    assert _resolve_pool_size() == 0
    monkeypatch.setenv("SONNET_RERANKER_POOL_SIZE", "0")
    assert _resolve_pool_size() == 0  # explicit 0 = unbounded


@pytest.mark.asyncio
async def test_pool_size_truncates_scoring_to_pool(monkeypatch):
    """When pool_size=3, only first 3 candidates are scored."""
    from search.sonnet_reranker import _rerank_async

    candidates = [
        {"chunk_id": f"c{i}", "file_path": f"f{i}.py", "full_content": f"chunk {i}"}
        for i in range(8)
    ]

    scored_chunk_ids = []

    async def track_score(client, query, file_path, content, extra_clauses=""):
        # Extract chunk index from file_path "fN.py"
        idx = int(file_path.split("f")[1].split(".")[0])
        scored_chunk_ids.append(f"c{idx}")
        return 8  # uniformly high so rerank applies

    monkeypatch.setattr("search.sonnet_reranker._score_one", track_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    out = await _rerank_async(
        "q", candidates, top_k=8, timeout=10.0, hybrid_prior_threshold=7,
        pool_size=3,
    )
    # Only first 3 candidates were scored
    assert sorted(scored_chunk_ids) == ["c0", "c1", "c2"]
    # Output has 8 items — 3 reranked pool + 5 tail
    assert len(out) == 8
    # Tail (c3..c7) preserved in hybrid order at the END
    assert [c["chunk_id"] for c in out[3:]] == ["c3", "c4", "c5", "c6", "c7"]


@pytest.mark.asyncio
async def test_pool_size_zero_scores_all_candidates(monkeypatch):
    """pool_size=0 (default) scores all candidates (preserves pre-Phase-A behavior)."""
    from search.sonnet_reranker import _rerank_async

    candidates = [
        {"chunk_id": f"c{i}", "file_path": f"f{i}.py", "full_content": f"chunk {i}"}
        for i in range(5)
    ]

    score_calls = 0

    async def count_score(client, query, file_path, content, extra_clauses=""):
        nonlocal score_calls
        score_calls += 1
        return 8

    monkeypatch.setattr("search.sonnet_reranker._score_one", count_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    out = await _rerank_async(
        "q", candidates, top_k=5, timeout=10.0, hybrid_prior_threshold=7,
        pool_size=0,
    )
    assert score_calls == 5  # all candidates scored
    assert len(out) == 5


@pytest.mark.asyncio
async def test_pool_size_larger_than_candidates_scores_all(monkeypatch):
    """pool_size=20 with 5 candidates scores all (no over-allocation)."""
    from search.sonnet_reranker import _rerank_async

    candidates = [
        {"chunk_id": f"c{i}", "file_path": f"f{i}.py", "full_content": f"chunk {i}"}
        for i in range(5)
    ]

    score_calls = 0

    async def count_score(client, query, file_path, content, extra_clauses=""):
        nonlocal score_calls
        score_calls += 1
        return 8

    monkeypatch.setattr("search.sonnet_reranker._score_one", count_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    out = await _rerank_async(
        "q", candidates, top_k=5, timeout=10.0, hybrid_prior_threshold=7,
        pool_size=20,
    )
    assert score_calls == 5  # only 5 candidates exist; no padding
    assert len(out) == 5


@pytest.mark.asyncio
async def test_pool_size_preserves_tail_hybrid_order(monkeypatch):
    """Pool reranking changes pool order; tail retains exact hybrid order."""
    from search.sonnet_reranker import _rerank_async

    candidates = [
        {"chunk_id": f"c{i}", "file_path": f"f{i}.py", "full_content": f"chunk {i}"}
        for i in range(6)
    ]

    # Score pool [c0, c1, c2] inversely (c2=10, c1=5, c0=2) so reranker
    # puts them in c2, c1, c0 order. Tail [c3, c4, c5] stays as-is.
    pool_scores = {"c0": 2, "c1": 5, "c2": 10}

    async def fake_score(client, query, file_path, content, extra_clauses=""):
        idx = int(file_path.split("f")[1].split(".")[0])
        chunk_id = f"c{idx}"
        return pool_scores.get(chunk_id, 0)

    monkeypatch.setattr("search.sonnet_reranker._score_one", fake_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    out = await _rerank_async(
        "q", candidates, top_k=6, timeout=10.0, hybrid_prior_threshold=2,
        pool_size=3,
    )
    # Pool reranked: c2 (10), c1 (5), c0 (2). Tail unchanged: c3, c4, c5.
    assert [c["chunk_id"] for c in out] == ["c2", "c1", "c0", "c3", "c4", "c5"]


@pytest.mark.asyncio
async def test_pool_size_with_top_k_smaller_than_pool(monkeypatch):
    """top_k=3, pool=5, candidates=10 → returns 3 reranked from pool only."""
    from search.sonnet_reranker import _rerank_async

    candidates = [
        {"chunk_id": f"c{i}", "file_path": f"f{i}.py", "full_content": f"chunk {i}"}
        for i in range(10)
    ]

    async def fake_score(client, query, file_path, content, extra_clauses=""):
        return 8  # all uniformly high; preserves input order via stable sort

    monkeypatch.setattr("search.sonnet_reranker._score_one", fake_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    out = await _rerank_async(
        "q", candidates, top_k=3, timeout=10.0, hybrid_prior_threshold=7,
        pool_size=5,
    )
    # top_k=3 from reranked pool (uniform scores, stable sort preserves input order)
    assert [c["chunk_id"] for c in out] == ["c0", "c1", "c2"]


@pytest.mark.asyncio
async def test_pool_size_with_top_k_larger_than_pool(monkeypatch):
    """top_k=8, pool=5, candidates=10 → 5 reranked + 3 tail in hybrid order."""
    from search.sonnet_reranker import _rerank_async

    candidates = [
        {"chunk_id": f"c{i}", "file_path": f"f{i}.py", "full_content": f"chunk {i}"}
        for i in range(10)
    ]

    # Pool [c0..c4] gets scored; rerank inverts their order
    pool_scores = {"c0": 2, "c1": 4, "c2": 6, "c3": 8, "c4": 10}

    async def fake_score(client, query, file_path, content, extra_clauses=""):
        idx = int(file_path.split("f")[1].split(".")[0])
        chunk_id = f"c{idx}"
        return pool_scores.get(chunk_id, 0)

    monkeypatch.setattr("search.sonnet_reranker._score_one", fake_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    out = await _rerank_async(
        "q", candidates, top_k=8, timeout=10.0, hybrid_prior_threshold=2,
        pool_size=5,
    )
    # Reranked pool (descending by score) + tail in hybrid order
    expected = ["c4", "c3", "c2", "c1", "c0", "c5", "c6", "c7"]
    assert [c["chunk_id"] for c in out] == expected


@pytest.mark.asyncio
async def test_pool_size_hybrid_prior_fallback_returns_full_candidates(monkeypatch):
    """When pool scores are all below threshold, fallback returns ALL candidates in hybrid order."""
    from search.sonnet_reranker import _rerank_async, REASON_HYBRID_PRIOR_FALLBACK

    candidates = [
        {"chunk_id": f"c{i}", "file_path": f"f{i}.py", "full_content": f"chunk {i}"}
        for i in range(8)
    ]

    async def low_score(client, query, file_path, content, extra_clauses=""):
        return 3  # uniformly below threshold=7

    monkeypatch.setattr("search.sonnet_reranker._score_one", low_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    out, meta = await _rerank_async(
        "q", candidates, top_k=8, timeout=10.0, hybrid_prior_threshold=7,
        pool_size=3, return_metadata=True,
    )
    assert meta["applied"] is False
    assert meta["reason"] == REASON_HYBRID_PRIOR_FALLBACK
    # Full hybrid order (NOT just the pool)
    assert [c["chunk_id"] for c in out] == ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"]


def test_pool_size_env_var_end_to_end(monkeypatch, sample_candidates):
    """Setting SONNET_RERANKER_POOL_SIZE=2 with no API key still returns input order."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("SONNET_RERANKER_POOL_SIZE", "2")
    # No API key → API_KEY_MISSING path; pool_size doesn't matter
    result = rerank_with_sonnet("auth", sample_candidates, top_k=3)
    assert result == sample_candidates[:3]


# ─── Per-cohort clause-override tests (Phase 2, 2026-05-24) ───

def test_parse_clause_overrides_unset_returns_empty():
    from search.sonnet_reranker import _parse_clause_overrides
    assert _parse_clause_overrides(None) == {}
    assert _parse_clause_overrides("") == {}


def test_parse_clause_overrides_valid_json():
    from search.sonnet_reranker import _parse_clause_overrides
    raw = '{"webapp/": "- TS clause", "billing/": "- Rust clause"}'
    out = _parse_clause_overrides(raw)
    assert out == {"webapp/": "- TS clause", "billing/": "- Rust clause"}


def test_parse_clause_overrides_malformed_json_returns_empty():
    """Malformed JSON logs warning and returns empty dict (never raises)."""
    from search.sonnet_reranker import _parse_clause_overrides
    assert _parse_clause_overrides("{not valid json") == {}
    assert _parse_clause_overrides("[1, 2, 3]") == {}  # array, not object


def test_parse_clause_overrides_non_string_values_skipped():
    """Entries with non-string values are skipped (warn-not-raise)."""
    from search.sonnet_reranker import _parse_clause_overrides
    raw = '{"webapp/": "- TS clause", "bad/": 42}'
    out = _parse_clause_overrides(raw)
    assert out == {"webapp/": "- TS clause"}


def test_parse_clause_overrides_backslash_normalized():
    """Backslash path separators normalized to forward-slash (Windows compat)."""
    from search.sonnet_reranker import _parse_clause_overrides
    raw = r'{"src\\routes\\": "- route clause"}'
    out = _parse_clause_overrides(raw)
    assert "src/routes/" in out
    assert "src\\routes\\" not in out


def test_matching_clauses_no_overrides_returns_empty():
    from search.sonnet_reranker import _matching_clauses
    assert _matching_clauses("webapp/app.tsx", {}) == []
    assert _matching_clauses("", {"webapp/": "- TS"}) == []


def test_matching_clauses_single_prefix_match():
    from search.sonnet_reranker import _matching_clauses
    overrides = {"webapp/": "- TS clause", "billing/": "- Rust clause"}
    # TS candidate gets ONLY the TS clause
    assert _matching_clauses("webapp/app.tsx", overrides) == ["- TS clause"]
    # Rust candidate gets ONLY the Rust clause
    assert _matching_clauses("billing/src/main.rs", overrides) == ["- Rust clause"]
    # Nix candidate gets nothing (cross-cohort isolation)
    assert _matching_clauses("nix/modules/foo.nix", overrides) == []


def test_matching_clauses_deterministic_alphabetical_order():
    """Multiple matching prefixes → clauses ordered alphabetically by prefix."""
    from search.sonnet_reranker import _matching_clauses
    # Two prefixes both match the same path
    overrides = {
        "src/": "- generic src clause",
        "src/auth/": "- specific auth clause",
    }
    out = _matching_clauses("src/auth/login.py", overrides)
    # Alphabetical-by-prefix: "src/" < "src/auth/"
    assert out == ["- generic src clause", "- specific auth clause"]


def test_matching_clauses_windows_path_normalized():
    """Windows-style file_path (backslashes) matches forward-slash prefix."""
    from search.sonnet_reranker import _matching_clauses
    overrides = {"webapp/": "- TS clause"}
    assert _matching_clauses(r"webapp\app.tsx", overrides) == ["- TS clause"]


@pytest.mark.asyncio
async def test_clause_override_cross_cohort_isolation(monkeypatch):
    """LOAD-BEARING: a clause keyed on 'webapp/' must NOT appear in the
    prompt scoring a 'netlib/' candidate. This is the structural guarantee
    that per-cohort dispatch eliminates cross-cohort interference.
    """
    from search.sonnet_reranker import _rerank_async

    # Mixed cohort: 1 TypeScript + 1 Rust candidate.
    candidates = [
        {"chunk_id": "ts1", "file_path": "webapp/app.tsx", "full_content": "TS code"},
        {"chunk_id": "rust1", "file_path": "netlib/src/tailscale.rs", "full_content": "Rust code"},
    ]

    # Capture extra_clauses passed to _score_one per candidate.
    captured = []
    async def capture_score(client, query, file_path, content, extra_clauses=""):
        captured.append((file_path, extra_clauses))
        return 8

    monkeypatch.setattr("search.sonnet_reranker._score_one", capture_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    overrides = {"webapp/": "- TS-only clause; must not leak to Rust scoring"}
    await _rerank_async(
        "q", candidates, top_k=2, timeout=8.0, hybrid_prior_threshold=7,
        clause_overrides=overrides,
    )

    # TS candidate's prompt contains the TS-only clause.
    ts_extra = next(extra for fp, extra in captured if fp == "webapp/app.tsx")
    assert "TS-only clause" in ts_extra

    # Rust candidate's prompt has NO TS-only clause. Cross-cohort isolation.
    rust_extra = next(extra for fp, extra in captured if fp == "netlib/src/tailscale.rs")
    assert "TS-only clause" not in rust_extra
    assert rust_extra == ""  # No overrides match netlib/, so empty extra_clauses


@pytest.mark.asyncio
async def test_clause_override_empty_when_no_overrides(monkeypatch):
    """When clause_overrides is None/empty, extra_clauses is always "".

    This is the A/A invariant: env-unset → prompt byte-identical to baseline.
    """
    from search.sonnet_reranker import _rerank_async

    candidates = [
        {"chunk_id": "c1", "file_path": "webapp/app.tsx", "full_content": "x"},
    ]
    captured = []
    async def capture_score(client, query, file_path, content, extra_clauses=""):
        captured.append(extra_clauses)
        return 8

    monkeypatch.setattr("search.sonnet_reranker._score_one", capture_score)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    await _rerank_async(
        "q", candidates, top_k=1, timeout=8.0, hybrid_prior_threshold=7,
        clause_overrides=None,
    )
    assert captured == [""]


def test_clause_overrides_env_var_unset_no_change(monkeypatch, sample_candidates):
    """A/A end-to-end: env var unset → no extra clauses, behavior preserved.

    Validates the Phase 2 ship gate guarantee: with the env var unset, the
    rerank result must be byte-identical to the pre-Phase-2 behavior.
    """
    monkeypatch.delenv("SONNET_RERANKER_PROMPT_CLAUSE_OVERRIDES", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # No API key → API_KEY_MISSING short-circuit, returns input order.
    # If the env var unset path is broken, the call could raise; this asserts
    # the unset path is wired correctly all the way through rerank_with_sonnet.
    result = rerank_with_sonnet("auth", sample_candidates, top_k=3)
    assert result == sample_candidates[:3]
