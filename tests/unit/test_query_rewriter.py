"""Tests for BM25 query rewriter (search/query_rewriter.py).

Covers PR #130's fix for the silent-fallback failure mode: PR #124
discovered that BM25_REWRITE=on with the deprecated default model
(`claude-3-haiku-20240307`) returned 404, the rewriter swallowed the
error, and operators saw Δ MRR=0.0000 vs off (silent no-op).

The fix:
  1. Bump default to `claude-haiku-4-5-20251001`
  2. Log first-occurrence on non-200 OR unparseable 200, so the
     deprecation-shaped failure isn't silent
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock


REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from search import query_rewriter
from search.query_rewriter import (
    DEFAULT_HAIKU_MODEL,
    rewrite_query_for_bm25,
)


def _reset_warned_flag():
    """Tests share module state; reset the first-occurrence sentinel."""
    query_rewriter._warned_fallback = False
    query_rewriter._rewrite_cache.clear()


def test_default_model_is_current_haiku():
    """The hardcoded default must point at a current alias, not deprecated.

    Specifically: NOT `claude-3-haiku-20240307` (which returns 404 today).
    Pinning the alias prevents accidental regression to a deprecated default.
    """
    assert DEFAULT_HAIKU_MODEL == "claude-haiku-4-5-20251001"
    # Must not contain old generation markers
    assert "claude-3-haiku" not in DEFAULT_HAIKU_MODEL
    assert "claude-2" not in DEFAULT_HAIKU_MODEL


def test_rewrite_returns_original_when_disabled(monkeypatch):
    """BM25_REWRITE=off (default) returns input unchanged."""
    monkeypatch.delenv("BM25_REWRITE", raising=False)
    out = rewrite_query_for_bm25("dropdown menu component")
    assert out == "dropdown menu component"


def test_rewrite_returns_original_on_404_and_logs_warning(monkeypatch, caplog):
    """A non-200 response (e.g. deprecated model 404) must log a warning,
    not silently swallow. PR #124's failure mode: a 404 silently returned
    the original query, hiding the deprecation entirely.
    """
    _reset_warned_flag()
    monkeypatch.setenv("BM25_REWRITE", "on")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key-for-test")

    fake_resp = MagicMock()
    fake_resp.status_code = 404
    fake_resp.text = '{"error": {"type": "not_found_error"}}'

    with caplog.at_level(logging.WARNING, logger="search.query_rewriter"):
        with patch("search.query_rewriter._ipv4_post", return_value=fake_resp):
            out = rewrite_query_for_bm25("button component UI element")

    # Original returned (graceful fallback contract preserved)
    assert out == "button component UI element"
    # But warning was logged — no longer silent
    assert any("API call failed" in r.message for r in caplog.records)
    assert any("status=404" in r.message for r in caplog.records)


def test_warning_only_logs_once_per_session(monkeypatch, caplog):
    """The warning is logged on FIRST occurrence; subsequent failures stay
    quiet to avoid log spam on a repeat-deprecated-model session."""
    _reset_warned_flag()
    monkeypatch.setenv("BM25_REWRITE", "on")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key-for-test")

    fake_resp = MagicMock()
    fake_resp.status_code = 404
    fake_resp.text = "{}"

    with caplog.at_level(logging.WARNING, logger="search.query_rewriter"):
        with patch("search.query_rewriter._ipv4_post", return_value=fake_resp):
            rewrite_query_for_bm25("query one")
            rewrite_query_for_bm25("query two")
            rewrite_query_for_bm25("query three")

    # Only ONE "API call failed" warning across 3 invocations
    api_failed_warnings = [r for r in caplog.records if "API call failed" in r.message]
    assert len(api_failed_warnings) == 1


def test_empty_response_text_logs_warning(monkeypatch, caplog):
    """A 200 with empty/unparseable body also logs (different code path
    from the 404 case)."""
    _reset_warned_flag()
    monkeypatch.setenv("BM25_REWRITE", "on")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key-for-test")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"content": [{"text": ""}]}

    with caplog.at_level(logging.WARNING, logger="search.query_rewriter"):
        with patch("search.query_rewriter._ipv4_post", return_value=fake_resp):
            out = rewrite_query_for_bm25("query that produces empty rewrite")

    assert out == "query that produces empty rewrite"
    assert any("empty/unparseable" in r.message for r in caplog.records)


def test_excessively_long_response_is_rejected(monkeypatch, caplog):
    """If the model returns > 500 chars (length sanity check), reject and
    fall back. Same warning-once contract."""
    _reset_warned_flag()
    monkeypatch.setenv("BM25_REWRITE", "on")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key-for-test")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"content": [{"text": "x" * 600}]}  # > 500

    with caplog.at_level(logging.WARNING, logger="search.query_rewriter"):
        with patch("search.query_rewriter._ipv4_post", return_value=fake_resp):
            out = rewrite_query_for_bm25("query producing absurdly long rewrite")

    assert out == "query producing absurdly long rewrite"


def test_default_model_used_when_env_unset(monkeypatch):
    """When BM25_REWRITE_MODEL is unset, _call_haiku must pass
    DEFAULT_HAIKU_MODEL to the API. Pins the env-fallback wiring."""
    _reset_warned_flag()
    monkeypatch.setenv("BM25_REWRITE", "on")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key-for-test")
    monkeypatch.delenv("BM25_REWRITE_MODEL", raising=False)

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"content": [{"text": "rewritten"}]}

    with patch("search.query_rewriter._ipv4_post", return_value=fake_resp) as mock_post:
        rewrite_query_for_bm25("test query unique 1")

    # The model field of the JSON body should match DEFAULT_HAIKU_MODEL
    call_args = mock_post.call_args
    assert call_args.kwargs["json"]["model"] == DEFAULT_HAIKU_MODEL


def test_env_override_takes_precedence(monkeypatch):
    """BM25_REWRITE_MODEL env var overrides the default."""
    _reset_warned_flag()
    monkeypatch.setenv("BM25_REWRITE", "on")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key-for-test")
    monkeypatch.setenv("BM25_REWRITE_MODEL", "claude-opus-4-7")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"content": [{"text": "rewritten"}]}

    with patch("search.query_rewriter._ipv4_post", return_value=fake_resp) as mock_post:
        rewrite_query_for_bm25("test query unique 2")

    call_args = mock_post.call_args
    assert call_args.kwargs["json"]["model"] == "claude-opus-4-7"


def test_rewrite_does_not_mutate_global_socket_getaddrinfo(monkeypatch):
    """Regression (2026-05): the rewrite path must NOT install a process-global
    socket.getaddrinfo override.

    The old code monkeypatched socket.getaddrinfo (and urllib3 HAS_IPV6=False)
    to force IPv4 and never restored it, leaking IPv4-only DNS onto every other
    network call in the MCP server (Voyage, the Anthropic reranker SDK). IPv4 is
    now forced per-request via an httpx transport, so the global must be intact.
    """
    import socket

    _reset_warned_flag()
    monkeypatch.setenv("BM25_REWRITE", "on")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key-for-test")

    before = socket.getaddrinfo

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"content": [{"text": "fetch_data loadItems"}]}

    with patch("search.query_rewriter._ipv4_post", return_value=fake_resp):
        out = rewrite_query_for_bm25("where is the data loading logic")

    assert out == "fetch_data loadItems"
    assert socket.getaddrinfo is before, (
        "rewrite path must not replace the global socket.getaddrinfo"
    )


def test_ipv4_post_forces_ipv4_via_transport_no_global_mutation(monkeypatch):
    """_ipv4_post pins IPv4 by binding the client's local socket to the IPv4
    wildcard (httpx HTTPTransport local_address='0.0.0.0') — not by mutating
    any process-global state."""
    import socket
    import httpx

    before = socket.getaddrinfo
    captured: dict = {}

    class _FakeClient:
        def __init__(self, *, transport=None, timeout=None):
            captured["transport"] = transport
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            return "RESP"

    def _fake_transport(*, local_address=None):
        captured["local_address"] = local_address
        return ("fake-transport", local_address)

    monkeypatch.setattr(httpx, "Client", _FakeClient)
    monkeypatch.setattr(httpx, "HTTPTransport", _fake_transport)

    resp = query_rewriter._ipv4_post(
        "https://example.test/x", headers={}, json={}, timeout=3.0
    )

    assert resp == "RESP"
    assert captured["local_address"] == "0.0.0.0"
    assert captured["timeout"] == 3.0
    assert socket.getaddrinfo is before
