"""Startup preflight for silent reranker degradation (PR #229 follow-up).

The graceful per-query fallback masks a permanently-degraded reranker on
clean installs; these tests pin that the server announces the condition
ONCE at startup, and stays quiet when the configuration is honest
(RERANKER=off) or healthy.
"""
import logging

import pytest

from mcp_server.code_search_server import CodeSearchServer


@pytest.fixture()
def isolated_storage(tmp_path, monkeypatch):
    # Keep the query-log DB (created in __init__) out of real storage.
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    from common_utils import get_storage_dir

    get_storage_dir.cache_clear()
    yield
    get_storage_dir.cache_clear()


def test_warns_when_sonnet_configured_but_key_missing(isolated_storage, monkeypatch, caplog):
    monkeypatch.setenv("RERANKER", "sonnet")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING, logger="mcp_server.code_search_server"):
        CodeSearchServer()
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "RERANKER=sonnet" in m and "fall back to hybrid order" in m for m in messages
    ), f"expected degradation warning, got: {messages}"


def test_silent_when_reranker_off(isolated_storage, monkeypatch, caplog):
    monkeypatch.setenv("RERANKER", "off")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING, logger="mcp_server.code_search_server"):
        CodeSearchServer()
    assert not any("fall back to hybrid order" in r.getMessage() for r in caplog.records)


def test_silent_when_healthy(isolated_storage, monkeypatch, caplog):
    # anthropic IS importable in the dev venv; a present key = healthy.
    monkeypatch.setenv("RERANKER", "sonnet")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    with caplog.at_level(logging.WARNING, logger="mcp_server.code_search_server"):
        CodeSearchServer()
    assert not any("fall back to hybrid order" in r.getMessage() for r in caplog.records)
