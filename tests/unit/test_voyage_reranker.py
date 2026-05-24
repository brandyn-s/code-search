"""Smoke tests for the production VoyageReranker contract.

Validates the graceful-fallback contract WITHOUT making real API calls.
A full integration test would need VOYAGE_API_KEY and is deferred to the
A/B eval step (see docs/EVAL_RUNBOOK.md + the 2026-05-24 finding doc
that will be filed alongside the SHIP/REVERT decision).

What this tests:
  - return_metadata=True returns (list, {applied, reason, latency_ms})
  - missing VOYAGE_API_KEY -> applied=False, reason="api_key_missing"
  - empty candidates -> applied=False, reason="empty_input"
  - REASON vocabulary is the same as sonnet_reranker (no per-reranker
    branching needed in consumers)
  - RERANKER_MODES tuple includes "voyage"
"""
from __future__ import annotations

import os

import pytest

from search import config as cfg_mod
from search.voyage_reranker import (
    REASON_API_KEY_MISSING,
    REASON_EMPTY_INPUT,
    REASON_OK,
    REASON_PACKAGE_NOT_INSTALLED,
    REASON_RATE_LIMIT,
    REASON_TIMEOUT,
    REASON_TOO_MANY_FAILURES,
    REASON_UNEXPECTED_ERROR,
    rerank_with_voyage,
)


def test_voyage_in_reranker_modes():
    """Voyage is a first-class reranker mode."""
    assert "voyage" in cfg_mod.RERANKER_MODES


def test_reason_vocabulary_matches_sonnet():
    """REASON constants are the same strings as sonnet_reranker — consumers
    don't need to switch on reranker type to interpret _metadata.reranker.reason."""
    from search import sonnet_reranker as sr
    assert REASON_OK == sr.REASON_OK
    assert REASON_API_KEY_MISSING == sr.REASON_API_KEY_MISSING
    assert REASON_PACKAGE_NOT_INSTALLED == sr.REASON_PACKAGE_NOT_INSTALLED
    assert REASON_TIMEOUT == sr.REASON_TIMEOUT
    assert REASON_RATE_LIMIT == sr.REASON_RATE_LIMIT
    assert REASON_TOO_MANY_FAILURES == sr.REASON_TOO_MANY_FAILURES
    assert REASON_UNEXPECTED_ERROR == sr.REASON_UNEXPECTED_ERROR


def test_empty_input_returns_empty_with_metadata(monkeypatch):
    """Empty candidate list -> ([], {applied=False, reason='empty_input'})."""
    monkeypatch.setenv("VOYAGE_API_KEY", "fake-key")  # so we don't trip the key check first
    out, meta = rerank_with_voyage(
        query="any",
        candidates=[],
        top_k=5,
        return_metadata=True,
    )
    assert out == []
    assert meta["applied"] is False
    assert meta["reason"] == REASON_EMPTY_INPUT
    assert "latency_ms" in meta


def test_missing_api_key_returns_hybrid_order(monkeypatch):
    """No VOYAGE_API_KEY -> hybrid order at top_k + reason=api_key_missing."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    candidates = [
        {"chunk_id": "a", "full_content": "function login() {}"},
        {"chunk_id": "b", "full_content": "function logout() {}"},
        {"chunk_id": "c", "full_content": "function register() {}"},
    ]
    out, meta = rerank_with_voyage(
        query="auth login",
        candidates=candidates,
        top_k=5,
        return_metadata=True,
    )
    assert meta["applied"] is False
    assert meta["reason"] == REASON_API_KEY_MISSING
    # Hybrid order preserved (input order at top_k)
    assert [c["chunk_id"] for c in out] == ["a", "b", "c"]


def test_legacy_return_shape_without_metadata(monkeypatch):
    """return_metadata=False returns just the list (legacy contract)."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    candidates = [
        {"chunk_id": "x", "full_content": "x"},
        {"chunk_id": "y", "full_content": "y"},
    ]
    out = rerank_with_voyage(
        query="q",
        candidates=candidates,
        top_k=1,
        return_metadata=False,
    )
    assert isinstance(out, list)
    # No metadata tuple
    assert len(out) == 1
    assert out[0]["chunk_id"] == "x"


def test_legacy_voyage_rerank_wrapper(monkeypatch):
    """The thin legacy wrapper (returns (index, score) tuples) preserves API."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    from search.voyage_reranker import voyage_rerank

    # Without an API key, fall back to hybrid order — wrapper returns
    # tuples for the first top_k items.
    out = voyage_rerank(
        query="q",
        documents=["alpha", "beta", "gamma"],
        top_k=2,
    )
    assert isinstance(out, list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in out)


def test_extract_text_priority():
    """Content extraction mirrors sonnet's priority order."""
    from search.voyage_reranker import _extract_text

    # full_content wins
    assert _extract_text(
        {"full_content": "FULL", "content": "C", "content_preview": "P"}
    ) == "FULL"
    # content second
    assert _extract_text(
        {"content": "C", "content_preview": "P"}
    ) == "C"
    # content_preview third
    assert _extract_text(
        {"content_preview": "P"}
    ) == "P"
    # empty default
    assert _extract_text({}) == ""
