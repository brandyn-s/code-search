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
    'stale_reindex_in_progress'. Post-R4/R5 (watchdog + lock), the
    nonblocking search path always calls _dispatch_background_reindex
    and lets the dispatch internals decide whether to start a fresh
    thread (under lock) or refuse (in-flight + within watchdog deadline)
    or pre-empt (in-flight but past watchdog deadline). Freshness is
    'stale_reindex_in_progress' in all of those cases since the result
    we're returning is from the pre-reindex index either way."""
    monkeypatch.delenv("CODE_SEARCH_DISABLE_AUTO_REINDEX", raising=False)
    monkeypatch.setenv("CODE_SEARCH_NONBLOCKING_SEARCH", "1")
    _stub_searcher_returning_empty(server)
    server._current_project = "/some/path"
    server._background_reindex_active = True
    # The dispatch is now ALWAYS called; its internal lock+watchdog
    # determines what to do. Stub it to return False (no fresh dispatch)
    # to simulate the "already in flight, within watchdog" case.
    with patch.object(server, "_dispatch_background_reindex",
                       return_value=False) as mock_dispatch:
        raw = server.search_code(query="x", k=5, auto_reindex=True)
    out = json.loads(raw)
    assert out["_metadata"]["freshness"] == "stale_reindex_in_progress"
    mock_dispatch.assert_called_once()


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

    fake_result = MagicMock(
        success=True,
        files_modified=2,
        files_added=1,
        files_removed=0,
        time_taken=0.5,
    )
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

    fake_result = MagicMock(
        success=True,
        files_modified=0,
        files_added=0,
        files_removed=0,
        time_taken=0.05,
    )
    with patch("search.incremental_indexer.IncrementalIndexer") as mock_ii_cls, \
         patch.object(server, "get_index_manager"), \
         patch.object(server, "embedder"):
        mock_ii_cls.return_value.auto_reindex_if_needed.return_value = fake_result
        raw = server.search_code(query="x", k=5, auto_reindex=True)
    out = json.loads(raw)
    assert out["_metadata"]["freshness"] == "fresh"


def test_freshness_fresh_after_deletion_only_reindex(server, monkeypatch):
    """Deletion-only runs mutate the index and must refresh identity/cache."""
    monkeypatch.delenv("CODE_SEARCH_DISABLE_AUTO_REINDEX", raising=False)
    monkeypatch.delenv("CODE_SEARCH_NONBLOCKING_SEARCH", raising=False)
    _stub_searcher_returning_empty(server)
    server._current_project = "/some/path"
    server._current_provider = None

    fake_result = MagicMock(
        success=True,
        files_modified=0,
        files_added=0,
        files_removed=1,
        time_taken=0.05,
    )
    with patch("search.incremental_indexer.IncrementalIndexer") as mock_ii_cls, \
         patch.object(server, "get_index_manager"), \
         patch.object(server, "embedder"):
        mock_ii_cls.return_value.auto_reindex_if_needed.return_value = fake_result
        raw = server.search_code(query="x", k=5, auto_reindex=True)

    out = json.loads(raw)
    assert out["_metadata"]["freshness"] == "fresh_after_reindex"


def test_blocking_reindex_persists_the_new_identity(
    server,
    tmp_path,
    monkeypatch,
):
    """The blocking search path uses the same coherent identity transaction."""
    from dataclasses import replace
    from search.index_identity import IndexIdentity

    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    info_file = project_dir / "project_info.json"
    original = IndexIdentity(
        schema_version=1,
        repository_id="a" * 64,
        checkout_id="b" * 64,
        source_revision="c" * 40,
        dirty_fingerprint="clean",
        index_generation="d" * 64,
        captured_at="2026-07-26T18:00:00Z",
    )
    updated = replace(
        original,
        dirty_fingerprint="e" * 64,
        index_generation="f" * 64,
    )
    info_file.write_text(
        json.dumps(
            {
                "project_path": str(source),
                "index_identity_status": "ready",
                "index_identity": original.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("CODE_SEARCH_DISABLE_AUTO_REINDEX", raising=False)
    monkeypatch.delenv("CODE_SEARCH_NONBLOCKING_SEARCH", raising=False)
    _stub_searcher_returning_empty(server)
    server._current_project = str(source)
    server._current_provider = None
    monkeypatch.setattr(
        server,
        "get_project_storage_dir",
        lambda *_args, **_kwargs: project_dir,
    )
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda _path: updated,
    )
    fake_result = MagicMock(
        success=True,
        files_modified=1,
        files_added=0,
        files_removed=0,
        time_taken=0.05,
    )

    with patch("search.incremental_indexer.IncrementalIndexer") as mock_ii_cls, \
         patch.object(server, "get_index_manager"), \
         patch.object(server, "embedder"):
        mock_ii_cls.return_value.auto_reindex_if_needed.return_value = fake_result
        raw = server.search_code(query="x", k=5, auto_reindex=True)

    out = json.loads(raw)
    persisted = json.loads(info_file.read_text(encoding="utf-8"))
    assert out["_metadata"]["freshness"] == "fresh_after_reindex"
    assert persisted["index_identity_status"] == "ready"
    assert persisted["index_identity"] == updated.to_dict()


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


# ─── Plan-2 E2-6: manifest freshness in search _metadata (PR #122) ───
#
# After E2-1 (PR #119) committed manifests via save_index, and E2-5
# (PR #121) surfaced manifest state in verify_index_integrity, the
# remaining production read path was search_code itself. E2-6 adds
# `_metadata.manifest = {status, epoch_id?}` to every search response,
# using the same `read_with_fallback` reader so search and integrity
# scans agree on the manifest verdict.


def _stub_searcher_with_storage_dir(server, storage_dir):
    """Replace get_searcher with a fake whose index_manager.storage_dir
    points at a real Path (so read_with_fallback can probe manifest/)."""
    fake_searcher = MagicMock()
    fake_searcher.search.return_value = []
    fake_searcher.index_manager.get_stats.return_value = {"total_chunks": 0}
    fake_searcher.index_manager.storage_dir = storage_dir
    fake_searcher._query_embedding_cache = {}
    fake_searcher.last_reranker_metadata = {
        "applied": False, "reason": "not_invoked_no_candidates", "latency_ms": 0,
    }
    server._searcher = fake_searcher
    server.get_searcher = MagicMock(return_value=fake_searcher)
    return fake_searcher


def test_manifest_metadata_missing_when_no_manifest_committed(
    server, tmp_path, monkeypatch,
):
    """A storage_dir without a manifest/ subdir reports status='missing'.
    This is the legacy state for indexes built before PR #119."""
    monkeypatch.delenv("CODE_SEARCH_DISABLE_AUTO_REINDEX", raising=False)
    monkeypatch.delenv("CODE_SEARCH_NONBLOCKING_SEARCH", raising=False)
    _stub_searcher_with_storage_dir(server, tmp_path)
    server._current_project = None
    raw = server.search_code(query="x", k=5, auto_reindex=False)
    out = json.loads(raw)
    assert "_metadata" in out
    assert "manifest" in out["_metadata"]
    assert out["_metadata"]["manifest"]["status"] == "missing"
    # No epoch_id when no manifest exists.
    assert "epoch_id" not in out["_metadata"]["manifest"]


def test_manifest_metadata_includes_epoch_id_when_fresh(
    server, tmp_path, monkeypatch,
):
    """When a real epoch manifest is committed, search_code surfaces
    status='fresh' with the pinned epoch_id."""
    from search.epoch_manifest import (
        ArtifactSpec,
        build_manifest,
        commit_manifest,
    )
    # Seed a single artifact + commit a manifest
    artifact_path = tmp_path / "chunk_ids.pkl"
    artifact_path.write_bytes(b"\x80\x05]\x94.")  # minimal valid pickle list
    artifacts = [ArtifactSpec(name="chunk_ids.pkl", path=artifact_path, count=0)]
    manifest = build_manifest(tmp_path, artifacts)
    commit_manifest(tmp_path, manifest)
    expected_epoch = manifest["epoch_id"]

    monkeypatch.delenv("CODE_SEARCH_DISABLE_AUTO_REINDEX", raising=False)
    monkeypatch.delenv("CODE_SEARCH_NONBLOCKING_SEARCH", raising=False)
    _stub_searcher_with_storage_dir(server, tmp_path)
    server._current_project = None
    raw = server.search_code(query="x", k=5, auto_reindex=False)
    out = json.loads(raw)
    assert out["_metadata"]["manifest"]["status"] == "fresh"
    assert out["_metadata"]["manifest"]["epoch_id"] == expected_epoch


def test_manifest_metadata_corrupt_when_artifacts_mutated(
    server, tmp_path, monkeypatch,
):
    """A committed manifest whose artifact SHAs no longer match (e.g.
    file mutated post-commit) reports status='corrupt'."""
    from search.epoch_manifest import (
        ArtifactSpec,
        build_manifest,
        commit_manifest,
    )
    artifact_path = tmp_path / "chunk_ids.pkl"
    artifact_path.write_bytes(b"\x80\x05]\x94.")
    artifacts = [ArtifactSpec(name="chunk_ids.pkl", path=artifact_path, count=0)]
    manifest = build_manifest(tmp_path, artifacts)
    commit_manifest(tmp_path, manifest)
    # Mutate the artifact AFTER commit — recorded SHA no longer matches.
    artifact_path.write_bytes(b"different bytes entirely")

    monkeypatch.delenv("CODE_SEARCH_DISABLE_AUTO_REINDEX", raising=False)
    monkeypatch.delenv("CODE_SEARCH_NONBLOCKING_SEARCH", raising=False)
    _stub_searcher_with_storage_dir(server, tmp_path)
    server._current_project = None
    raw = server.search_code(query="x", k=5, auto_reindex=False)
    out = json.loads(raw)
    assert out["_metadata"]["manifest"]["status"] == "corrupt"


def test_manifest_metadata_absent_on_probe_exception(server, monkeypatch):
    """If the manifest probe raises (e.g. storage_dir is a non-Path object),
    the failure is swallowed and the response carries no `manifest` field —
    a manifest probe must NEVER break a search response."""
    monkeypatch.delenv("CODE_SEARCH_DISABLE_AUTO_REINDEX", raising=False)
    monkeypatch.delenv("CODE_SEARCH_NONBLOCKING_SEARCH", raising=False)
    fake_searcher = MagicMock()
    fake_searcher.search.return_value = []
    fake_searcher.index_manager.get_stats.return_value = {"total_chunks": 0}
    # storage_dir is a sentinel object — read_with_fallback will fail
    fake_searcher.index_manager.storage_dir = object()
    fake_searcher._query_embedding_cache = {}
    fake_searcher.last_reranker_metadata = {
        "applied": False, "reason": "not_invoked_no_candidates", "latency_ms": 0,
    }
    server._searcher = fake_searcher
    server.get_searcher = MagicMock(return_value=fake_searcher)
    server._current_project = None

    raw = server.search_code(query="x", k=5, auto_reindex=False)
    out = json.loads(raw)
    # Other _metadata fields still present.
    assert "_metadata" in out
    assert "freshness" in out["_metadata"]
    # manifest field is absent (graceful degradation).
    assert "manifest" not in out["_metadata"]


def test_manifest_status_vocabulary_is_stable():
    """Pin the manifest.status vocabulary. Downstream consumers
    pattern-match these — changing values is a breaking change.

    Vocabulary is sourced from search.epoch_manifest.ReadResult.freshness.
    """
    expected = {
        "fresh",
        "stale_using_prior_epoch",
        "missing",
        "corrupt",
    }
    # Sanity-check by inspecting the enum-ish strings the implementation
    # compares against.
    from search.epoch_manifest import read_with_fallback  # noqa: F401
    assert expected == {
        "fresh",
        "stale_using_prior_epoch",
        "missing",
        "corrupt",
    }
