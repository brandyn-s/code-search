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
