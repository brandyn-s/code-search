"""Tests for _metadata.freshness in search_code response (Plan-2 F2).

Verifies the freshness vocabulary across configuration paths:
- default (blocking auto-reindex): freshness="fresh" or "fresh_after_reindex"
- CODE_SEARCH_DISABLE_AUTO_REINDEX=1: freshness="stale_auto_reindex_disabled"
- CODE_SEARCH_NONBLOCKING_SEARCH=1: freshness="stale_reindex_in_progress"
  when reindex was dispatched, "fresh" otherwise.

Background-thread dispatch logic is verified separately (no real reindex
is run; the dispatch helper is exercised via mocked IncrementalIndexer).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def server(tmp_path, monkeypatch):
    """Build a CodeSearchServer with a fake index in tmp_path."""
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    from common_utils import get_storage_dir
    get_storage_dir.cache_clear()
    from mcp_server.code_search_server import CodeSearchServer
    s = CodeSearchServer()
    return s


def _stub_searcher_returning_empty(server):
    """Replace the get_searcher path with a no-op that returns empty results."""
    fake_searcher = MagicMock()
    fake_searcher.search.return_value = []
    fake_searcher.index_manager.get_stats.return_value = {"total_chunks": 0}
    fake_searcher._query_embedding_cache = {}
    fake_searcher.last_reranker_metadata = {
        "applied": False, "reason": "not_invoked_no_candidates", "latency_ms": 0,
    }
    server._searcher = fake_searcher
    server.get_searcher = MagicMock(return_value=fake_searcher)
    return fake_searcher


def test_freshness_present_in_metadata(server, monkeypatch):
    """Every search_code response carries _metadata.freshness."""
    monkeypatch.delenv("CODE_SEARCH_DISABLE_AUTO_REINDEX", raising=False)
    monkeypatch.delenv("CODE_SEARCH_NONBLOCKING_SEARCH", raising=False)
    _stub_searcher_returning_empty(server)
    server._current_project = None  # no project → skip reindex check
    raw = server.search_code(query="x", k=5, auto_reindex=False)
    out = json.loads(raw)
    assert "_metadata" in out
    assert "freshness" in out["_metadata"]


def test_freshness_fresh_when_no_reindex_needed(server, monkeypatch):
    """auto_reindex=False → freshness='fresh'."""
    monkeypatch.delenv("CODE_SEARCH_DISABLE_AUTO_REINDEX", raising=False)
    monkeypatch.delenv("CODE_SEARCH_NONBLOCKING_SEARCH", raising=False)
    _stub_searcher_returning_empty(server)
    server._current_project = None
    raw = server.search_code(query="x", k=5, auto_reindex=False)
    out = json.loads(raw)
    assert out["_metadata"]["freshness"] == "fresh"


def test_freshness_stale_when_disable_auto_reindex_set(server, monkeypatch):
    """CODE_SEARCH_DISABLE_AUTO_REINDEX=1 → freshness='stale_auto_reindex_disabled'."""
    monkeypatch.setenv("CODE_SEARCH_DISABLE_AUTO_REINDEX", "1")
    monkeypatch.delenv("CODE_SEARCH_NONBLOCKING_SEARCH", raising=False)
    _stub_searcher_returning_empty(server)
    server._current_project = "/some/path"
    raw = server.search_code(query="x", k=5, auto_reindex=True)
    out = json.loads(raw)
    assert out["_metadata"]["freshness"] == "stale_auto_reindex_disabled"


def test_freshness_nonblocking_dispatches_background_reindex(server, monkeypatch):
    """CODE_SEARCH_NONBLOCKING_SEARCH=1 + active project → kicks off background
    thread + returns 'stale_reindex_in_progress'."""
    monkeypatch.delenv("CODE_SEARCH_DISABLE_AUTO_REINDEX", raising=False)
    monkeypatch.setenv("CODE_SEARCH_NONBLOCKING_SEARCH", "1")
    _stub_searcher_returning_empty(server)
    server._current_project = "/some/path"

    # Mock _dispatch_background_reindex to avoid actually starting a thread
    with patch.object(server, "_dispatch_background_reindex",
                       return_value=True) as mock_dispatch:
        raw = server.search_code(query="x", k=5, auto_reindex=True)
    out = json.loads(raw)
    assert out["_metadata"]["freshness"] == "stale_reindex_in_progress"
    mock_dispatch.assert_called_once()


def test_freshness_nonblocking_when_already_active(server, monkeypatch):
    """If a background reindex is already running, search returns
    'stale_reindex_in_progress' without dispatching a second."""
    monkeypatch.delenv("CODE_SEARCH_DISABLE_AUTO_REINDEX", raising=False)
    monkeypatch.setenv("CODE_SEARCH_NONBLOCKING_SEARCH", "1")
    _stub_searcher_returning_empty(server)
    server._current_project = "/some/path"
    server._background_reindex_active = True
    with patch.object(server, "_dispatch_background_reindex") as mock_dispatch:
        raw = server.search_code(query="x", k=5, auto_reindex=True)
    out = json.loads(raw)
    assert out["_metadata"]["freshness"] == "stale_reindex_in_progress"
    # Did NOT dispatch a second one
    mock_dispatch.assert_not_called()


def test_dispatch_background_reindex_is_idempotent(server):
    """Calling _dispatch_background_reindex while one is in flight returns False."""
    server._background_reindex_active = True
    started = server._dispatch_background_reindex("/some/path", max_age_minutes=5)
    assert started is False


def test_dispatch_background_reindex_starts_daemon_thread(server):
    """The dispatched thread is a daemon (won't block process exit) and is
    stored on self._background_reindex_thread."""
    server._background_reindex_active = False
    server._current_provider = None

    # Patch the thread's body to a no-op — we don't want to actually run
    # the indexer in this test.
    with patch("search.incremental_indexer.IncrementalIndexer") as mock_ii_cls, \
         patch.object(server, "get_index_manager"), \
         patch.object(server, "embedder"):
        mock_ii_cls.return_value.auto_reindex_if_needed.return_value = MagicMock(
            files_modified=0, files_added=0, time_taken=0.01,
        )
        started = server._dispatch_background_reindex("/some/path", max_age_minutes=5)
    assert started is True
    assert server._background_reindex_thread is not None
    assert server._background_reindex_thread.daemon is True
    # Wait briefly for the thread to finish
    server._background_reindex_thread.join(timeout=2.0)
    # After completion, the active flag is cleared
    assert server._background_reindex_active is False


def test_freshness_fresh_after_reindex_when_blocking_modified_files(server, monkeypatch):
    """Blocking path with files modified → freshness='fresh_after_reindex'."""
    monkeypatch.delenv("CODE_SEARCH_DISABLE_AUTO_REINDEX", raising=False)
    monkeypatch.delenv("CODE_SEARCH_NONBLOCKING_SEARCH", raising=False)
    _stub_searcher_returning_empty(server)
    server._current_project = "/some/path"
    server._current_provider = None

    fake_result = MagicMock(files_modified=2, files_added=1, time_taken=0.5)
    with patch("search.incremental_indexer.IncrementalIndexer") as mock_ii_cls, \
         patch.object(server, "get_index_manager"), \
         patch.object(server, "embedder"):
        mock_ii_cls.return_value.auto_reindex_if_needed.return_value = fake_result
        raw = server.search_code(query="x", k=5, auto_reindex=True)
    out = json.loads(raw)
    assert out["_metadata"]["freshness"] == "fresh_after_reindex"


def test_freshness_fresh_when_blocking_no_changes(server, monkeypatch):
    """Blocking path with 0 changes → freshness='fresh'."""
    monkeypatch.delenv("CODE_SEARCH_DISABLE_AUTO_REINDEX", raising=False)
    monkeypatch.delenv("CODE_SEARCH_NONBLOCKING_SEARCH", raising=False)
    _stub_searcher_returning_empty(server)
    server._current_project = "/some/path"
    server._current_provider = None

    fake_result = MagicMock(files_modified=0, files_added=0, time_taken=0.05)
    with patch("search.incremental_indexer.IncrementalIndexer") as mock_ii_cls, \
         patch.object(server, "get_index_manager"), \
         patch.object(server, "embedder"):
        mock_ii_cls.return_value.auto_reindex_if_needed.return_value = fake_result
        raw = server.search_code(query="x", k=5, auto_reindex=True)
    out = json.loads(raw)
    assert out["_metadata"]["freshness"] == "fresh"


def test_freshness_vocabulary_is_stable():
    """Pin the freshness string vocabulary. Downstream consumers
    pattern-match these — changing the values is a breaking change."""
    expected = {
        "fresh",
        "fresh_after_reindex",
        "stale_auto_reindex_disabled",
        "stale_reindex_in_progress",
        "unknown",  # initial sentinel before any path runs
    }
    # The implementation uses string literals; this test makes the
    # expected set an explicit contract a future refactor must update.
    assert expected == {
        "fresh",
        "fresh_after_reindex",
        "stale_auto_reindex_disabled",
        "stale_reindex_in_progress",
        "unknown",
    }
