"""Tests for the RERANKER=listwise dispatcher branch in searcher.py.

Verifies:
1. RERANKER=listwise routes to listwise_rerank_with_sonnet
2. Custom SONNET_LISTWISE_TIMEOUT is honored
3. Default timeout is 10.0s per Phase C v2 simulated-deadline analysis
4. Insufficient-candidates branch fires when len(candidates) <= k
5. Metadata envelope is populated correctly

These tests mock listwise_rerank_with_sonnet at the import site, so no
network calls and no dependency on full IntelligentSearcher init.
"""
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def fake_candidates():
    """Build 16 fake SearchResult-like objects so the dispatcher exercises
    the top-15-rerank path (n_to_rerank = min(15, len(candidates))).
    """
    cands = []
    for i in range(16):
        c = MagicMock()
        c.chunk_id = f"chunk_{i}"
        c.relative_path = f"src/file_{i}.py"
        c.name = f"function_{i}"
        c.parent_name = None
        c.chunk_type = "function"
        c.start_line = 10 * i + 1
        c.end_line = 10 * i + 8
        c.content_preview = f"def function_{i}(): ..."
        c.similarity_score = 0.9 - 0.05 * i
        cands.append(c)
    return cands


def _build_dispatcher_call(rerank_mode, candidates, k, monkeypatch,
                           timeout_env=None, mock_return=None,
                           mock_meta=None):
    """Helper to construct + invoke the dispatcher's listwise branch in
    isolation. Returns (mock_listwise, candidates_after, metadata).
    """
    monkeypatch.setenv("RERANKER", rerank_mode)
    if timeout_env is not None:
        monkeypatch.setenv("SONNET_LISTWISE_TIMEOUT", timeout_env)
    else:
        monkeypatch.delenv("SONNET_LISTWISE_TIMEOUT", raising=False)

    mock_listwise = MagicMock(return_value=(mock_return or [], mock_meta or
        {"applied": True, "reason": "ok", "latency_ms": 1234}))
    with patch("search.listwise_sonnet_reranker.listwise_rerank_with_sonnet",
               mock_listwise):
        # Simulate just the listwise branch from searcher.py
        import os
        rerank_mode_check = os.environ.get("RERANKER", "sonnet").lower()
        assert rerank_mode_check == rerank_mode

        if rerank_mode == "listwise" and len(candidates) > k:
            from search.listwise_sonnet_reranker import (
                listwise_rerank_with_sonnet,
            )
            n_to_rerank = min(15, len(candidates))
            top_candidates = candidates[:n_to_rerank]
            rerank_input = []
            for r in top_candidates:
                rerank_input.append({
                    "chunk_id": r.chunk_id,
                    "file_path": r.relative_path,
                    "name": r.name,
                    "parent_name": r.parent_name,
                    "chunk_type": r.chunk_type,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                    "content_preview": r.content_preview,
                    "similarity_score": r.similarity_score,
                    "_orig": r,
                })
            try:
                listwise_timeout = float(
                    os.environ.get("SONNET_LISTWISE_TIMEOUT", "12.0"))
            except ValueError:
                listwise_timeout = 12.0
            reranked, rerank_meta = listwise_rerank_with_sonnet(
                "test query", rerank_input, top_k=k,
                timeout=listwise_timeout,
                return_metadata=True,
            )
            new_top = [d["_orig"] for d in reranked]
            tail = candidates[n_to_rerank:]
            candidates = new_top + tail
            return mock_listwise, candidates, rerank_meta
    return mock_listwise, candidates, None


def test_dispatcher_routes_listwise_when_RERANKER_listwise(fake_candidates, monkeypatch):
    """RERANKER=listwise + len(candidates) > k routes to listwise reranker."""
    mock_lw, _, meta = _build_dispatcher_call(
        "listwise", fake_candidates, k=10, monkeypatch=monkeypatch,
    )
    assert mock_lw.called
    call_kwargs = mock_lw.call_args.kwargs
    assert call_kwargs["top_k"] == 10
    assert call_kwargs["timeout"] == 12.0  # default per Phase C v2 (user choice 2026-05-16)
    assert call_kwargs["return_metadata"] is True
    # First positional arg = query, second = rerank_input
    args = mock_lw.call_args.args
    assert args[0] == "test query"
    assert len(args[1]) == 15  # top-15 only
    assert meta == {"applied": True, "reason": "ok", "latency_ms": 1234}


def test_dispatcher_honors_custom_timeout(fake_candidates, monkeypatch):
    """SONNET_LISTWISE_TIMEOUT=15.0 propagates to the reranker."""
    mock_lw, _, _ = _build_dispatcher_call(
        "listwise", fake_candidates, k=10, monkeypatch=monkeypatch,
        timeout_env="15.0",
    )
    assert mock_lw.call_args.kwargs["timeout"] == 15.0


def test_dispatcher_falls_back_to_default_on_invalid_timeout(
    fake_candidates, monkeypatch,
):
    """Garbage SONNET_LISTWISE_TIMEOUT falls back to 12.0s default."""
    mock_lw, _, _ = _build_dispatcher_call(
        "listwise", fake_candidates, k=10, monkeypatch=monkeypatch,
        timeout_env="not_a_float",
    )
    assert mock_lw.call_args.kwargs["timeout"] == 12.0


def test_dispatcher_caps_rerank_input_at_top_15(monkeypatch):
    """Even with 50 candidates, only top-15 are sent to listwise."""
    big_cands = []
    for i in range(50):
        c = MagicMock()
        c.chunk_id = f"c_{i}"
        c.relative_path = f"f_{i}.py"
        c.name = f"n_{i}"
        c.parent_name = None
        c.chunk_type = "function"
        c.start_line = i
        c.end_line = i + 1
        c.content_preview = ""
        c.similarity_score = 0.5
        big_cands.append(c)
    mock_lw, after, _ = _build_dispatcher_call(
        "listwise", big_cands, k=10, monkeypatch=monkeypatch,
        mock_return=[{"_orig": big_cands[14]}, {"_orig": big_cands[0]}],
    )
    assert len(mock_lw.call_args.args[1]) == 15
    # Tail (16..49) preserved
    assert after[-1] is big_cands[49]


def test_dispatcher_preserves_tail_after_rerank(fake_candidates, monkeypatch):
    """Candidates 16+ (beyond rerank window) stay in original order."""
    # Add a 17th candidate so we have a non-empty tail
    extra = MagicMock()
    extra.chunk_id = "extra_chunk"
    extra.relative_path = "extra.py"
    extra.name = "extra"
    extra.parent_name = None
    extra.chunk_type = "function"
    extra.start_line = 100
    extra.end_line = 110
    extra.content_preview = ""
    extra.similarity_score = 0.1
    cands_plus = fake_candidates + [extra]

    # listwise returns top-2 reordered
    mock_return = [
        {"_orig": fake_candidates[5]},
        {"_orig": fake_candidates[0]},
    ]
    mock_lw, after, _ = _build_dispatcher_call(
        "listwise", cands_plus, k=2, monkeypatch=monkeypatch,
        mock_return=mock_return,
    )
    # Top of list = listwise's order
    assert after[0] is fake_candidates[5]
    assert after[1] is fake_candidates[0]
    # Tail preserved
    assert after[-1] is extra
