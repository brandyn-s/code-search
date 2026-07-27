"""Identity transactions for search-triggered automatic reindexing."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import mcp_server.code_search_server as code_search_server_module
import pytest

from embeddings.embedder import EffectiveEmbeddingConfig
from mcp_server.code_search_server import CodeSearchServer
from search.index_identity import IndexIdentity
import search.incremental_indexer as incremental_indexer_module


def _identity() -> IndexIdentity:
    return IndexIdentity(
        schema_version=1,
        repository_id="a" * 64,
        checkout_id="b" * 64,
        source_revision="c" * 40,
        dirty_fingerprint="clean",
        index_generation="d" * 64,
        captured_at="2026-07-26T18:00:00Z",
    )


def _ready_info(info_file: Path, source: Path) -> dict[str, object]:
    project_info = {
        "project_path": str(source),
        "index_identity_status": "ready",
        "index_identity": _identity().to_dict(),
    }
    info_file.write_text(json.dumps(project_info), encoding="utf-8")
    return project_info


class _AutoIndexer:
    def __init__(
        self,
        info_file: Path,
        result: SimpleNamespace | BaseException,
    ) -> None:
        self.info_file = info_file
        self.result = result
        self.observed_state: dict[str, object] | None = None

    def auto_reindex_if_needed(self, *_args, **_kwargs):
        self.observed_state = json.loads(
            self.info_file.read_text(encoding="utf-8")
        )
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def _result(
    *,
    added: int = 0,
    modified: int = 0,
    removed: int = 0,
    success: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        success=success,
        files_added=added,
        files_modified=modified,
        files_removed=removed,
        time_taken=0.1,
        error=None if success else "auto reindex failed",
    )


def test_auto_reindex_publishes_pending_then_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    info_file = project_dir / "project_info.json"
    _ready_info(info_file, source)
    server = CodeSearchServer.__new__(CodeSearchServer)
    monkeypatch.setattr(
        server,
        "get_project_storage_dir",
        lambda *_args, **_kwargs: project_dir,
    )
    updated = replace(
        _identity(),
        dirty_fingerprint="e" * 64,
        index_generation="f" * 64,
    )
    captures = iter((updated, updated))
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda _path: next(captures),
    )
    indexer = _AutoIndexer(info_file, _result(modified=1))
    ready_metadata = {
        "embedding_provider": "voyage",
        "embedding_model": "voyage-code-3",
        "embedding_dimension": 1024,
        "content_mode": "code",
        "pipeline_version": "pipeline-v2",
    }

    _, mutated, state = server._auto_reindex_with_identity(
        indexer,
        source,
        max_age_minutes=5,
        publish_pending=True,
        ready_metadata=ready_metadata,
    )

    assert mutated is True
    assert indexer.observed_state["index_identity_status"] == "pending"
    assert "index_identity" not in indexer.observed_state
    assert state["index_identity_status"] == "ready"
    persisted = json.loads(info_file.read_text(encoding="utf-8"))
    assert persisted["index_identity_status"] == "ready"
    assert persisted["index_identity"] == updated.to_dict()
    for key, value in ready_metadata.items():
        assert persisted[key] == value


def test_failed_auto_reindex_file_counts_do_not_report_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    info_file = project_dir / "project_info.json"
    _ready_info(info_file, source)
    server = CodeSearchServer.__new__(CodeSearchServer)
    monkeypatch.setattr(
        server,
        "get_project_storage_dir",
        lambda *_args, **_kwargs: project_dir,
    )
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda _path: _identity(),
    )

    _, mutated, state = server._auto_reindex_with_identity(
        _AutoIndexer(
            info_file,
            _result(added=2, success=False),
        ),
        source,
        max_age_minutes=5,
        publish_pending=True,
    )

    assert mutated is False
    assert state["index_identity_status"] == "error"
    assert "auto_reindex_failed" in state["index_identity_error"]


def test_auto_reindex_noop_restores_previous_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    info_file = project_dir / "project_info.json"
    previous = _ready_info(info_file, source)
    server = CodeSearchServer.__new__(CodeSearchServer)
    monkeypatch.setattr(
        server,
        "get_project_storage_dir",
        lambda *_args, **_kwargs: project_dir,
    )
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda _path: _identity(),
    )
    indexer = _AutoIndexer(info_file, _result())

    _, mutated, state = server._auto_reindex_with_identity(
        indexer,
        source,
        max_age_minutes=5,
        publish_pending=True,
    )

    assert mutated is False
    assert indexer.observed_state["index_identity_status"] == "pending"
    assert state["index_identity_status"] == "ready"
    persisted = json.loads(info_file.read_text(encoding="utf-8"))
    assert persisted == previous


@pytest.mark.parametrize(
    ("reindex_path", "expected_disposition", "publishes_new_identity"),
    [
        ("completed_no_changes", "completed", True),
        ("age_skipped", "skipped_fresh", False),
        ("disabled", "skipped_disabled", False),
        ("refused", "refused", False),
    ],
)
def test_auto_reindex_identity_commit_requires_completed_scan(
    tmp_path: Path,
    monkeypatch,
    reindex_path: str,
    expected_disposition: str,
    publishes_new_identity: bool,
) -> None:
    from unittest.mock import MagicMock

    from search.incremental_indexer import (
        IncrementalIndexer,
        IncrementalIndexResult,
    )

    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    info_file = project_dir / "project_info.json"
    previous = _ready_info(info_file, source)
    updated = replace(
        _identity(),
        source_revision="e" * 40,
        index_generation="f" * 64,
    )

    server = CodeSearchServer.__new__(CodeSearchServer)
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

    indexer = IncrementalIndexer(
        indexer=MagicMock(),
        embedder=MagicMock(),
        chunker=MagicMock(),
        snapshot_manager=MagicMock(),
    )
    indexer.needs_reindex = MagicMock(
        return_value=reindex_path == "completed_no_changes"
    )
    indexer.incremental_index = MagicMock(
        return_value=IncrementalIndexResult(
            files_added=0,
            files_removed=0,
            files_modified=0,
            chunks_added=0,
            chunks_removed=0,
            time_taken=0.1,
            success=True,
        )
    )
    monkeypatch.delenv("CODE_SEARCH_DISABLE_AUTO_REINDEX", raising=False)
    monkeypatch.setattr(
        incremental_indexer_module,
        "refuse_as_project_root_reason",
        lambda _path: (
            "synthetic refusal" if reindex_path == "refused" else None
        ),
    )
    if reindex_path == "disabled":
        monkeypatch.setenv("CODE_SEARCH_DISABLE_AUTO_REINDEX", "1")

    result, mutated, state = server._auto_reindex_with_identity(
        indexer,
        source,
        max_age_minutes=5,
        publish_pending=True,
    )

    assert mutated is False
    assert result.reindex_disposition == expected_disposition
    assert state["index_identity_status"] == "ready"
    persisted = json.loads(info_file.read_text(encoding="utf-8"))
    if publishes_new_identity:
        assert persisted["index_identity"] == updated.to_dict()
    else:
        assert persisted == previous


def test_auto_reindex_counts_deletion_only_as_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    info_file = project_dir / "project_info.json"
    _ready_info(info_file, source)
    server = CodeSearchServer.__new__(CodeSearchServer)
    monkeypatch.setattr(
        server,
        "get_project_storage_dir",
        lambda *_args, **_kwargs: project_dir,
    )
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda _path: _identity(),
    )

    _, mutated, state = server._auto_reindex_with_identity(
        _AutoIndexer(info_file, _result(removed=1)),
        source,
        max_age_minutes=5,
        publish_pending=False,
    )

    assert mutated is True
    assert state["index_identity_status"] == "ready"


def test_auto_reindex_exception_clears_pending_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    info_file = project_dir / "project_info.json"
    _ready_info(info_file, source)
    server = CodeSearchServer.__new__(CodeSearchServer)
    monkeypatch.setattr(
        server,
        "get_project_storage_dir",
        lambda *_args, **_kwargs: project_dir,
    )
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda _path: _identity(),
    )

    with pytest.raises(RuntimeError, match="exploded"):
        server._auto_reindex_with_identity(
            _AutoIndexer(info_file, RuntimeError("exploded")),
            source,
            max_age_minutes=5,
            publish_pending=True,
        )

    persisted = json.loads(info_file.read_text(encoding="utf-8"))
    assert persisted["index_identity_status"] == "error"
    assert "auto_reindex_exception" in persisted["index_identity_error"]
    assert "index_identity" not in persisted


def test_background_dispatch_commits_identity_for_actual_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    info_file = project_dir / "project_info.json"
    _ready_info(info_file, source)
    updated = replace(
        _identity(),
        dirty_fingerprint="e" * 64,
        index_generation="f" * 64,
    )

    class FakeIncrementalIndexer:
        def __init__(self, **_kwargs):
            pass

        def auto_reindex_if_needed(self, *_args, **_kwargs):
            return _result(removed=1)

    server = CodeSearchServer.__new__(CodeSearchServer)
    server._background_reindex_lock = threading.Lock()
    server._background_reindex_active = False
    server._background_reindex_started_at = None
    server._background_reindex_thread = None
    server._current_provider = None
    server._searcher = None
    configuration = EffectiveEmbeddingConfig(
        provider="voyage",
        model_name="voyage-4-large",
        content_mode="code",
        output_dimension=1024,
    )
    bound: dict[str, object] = {}
    index_manager = SimpleNamespace(
        bind_embedding_configuration=lambda effective, pipeline_version: (
            bound.update(
                {
                    "configuration": effective,
                    "pipeline_version": pipeline_version,
                }
            )
        ),
    )
    monkeypatch.setattr(
        server,
        "get_project_storage_dir",
        lambda *_args, **_kwargs: project_dir,
    )
    monkeypatch.setattr(
        server,
        "get_index_manager",
        lambda *_args, **_kwargs: index_manager,
    )
    monkeypatch.setattr(
        server,
        "embedder",
        lambda *_args, **_kwargs: SimpleNamespace(
            configuration=configuration,
        ),
    )
    monkeypatch.setattr(
        code_search_server_module,
        "get_pipeline_version",
        lambda effective: (
            "effective-background-pipeline"
            if effective is configuration
            else "wrong-pipeline"
        ),
    )
    monkeypatch.setattr(
        code_search_server_module,
        "MultiLanguageChunker",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        incremental_indexer_module,
        "IncrementalIndexer",
        FakeIncrementalIndexer,
    )
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda _path: updated,
    )

    assert server._dispatch_background_reindex(
        str(source),
        max_age_minutes=5,
    )
    server._background_reindex_thread.join(timeout=2)

    persisted = json.loads(info_file.read_text(encoding="utf-8"))
    assert persisted["index_identity_status"] == "ready"
    assert persisted["index_identity"] == updated.to_dict()
    assert bound == {
        "configuration": configuration,
        "pipeline_version": "effective-background-pipeline",
    }


def test_blocking_auto_reindex_binds_effective_embedding_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    configuration = EffectiveEmbeddingConfig(
        provider="openai",
        model_name="text-embedding-3-large",
        content_mode="docs",
        output_dimension=3072,
    )
    bound: dict[str, object] = {}
    index_manager = SimpleNamespace(
        bind_embedding_configuration=lambda effective, pipeline_version: (
            bound.update(
                {
                    "configuration": effective,
                    "pipeline_version": pipeline_version,
                }
            )
        ),
    )
    embedder = SimpleNamespace(configuration=configuration)

    class FakeIncrementalIndexer:
        def __init__(self, **_kwargs):
            pass

    searcher = SimpleNamespace(
        index_manager=SimpleNamespace(
            get_stats=lambda: {"total_chunks": 1},
        ),
        search=lambda **_kwargs: [],
    )
    server = CodeSearchServer.__new__(CodeSearchServer)
    server._current_project = str(source)
    server._current_provider = "openai"
    server._searcher = None
    server._indexing_job_snapshot = lambda: None
    server.get_index_manager = lambda *_args, **_kwargs: index_manager
    server.embedder = lambda *_args, **_kwargs: embedder
    server.get_searcher = lambda *_args, **_kwargs: searcher
    server._auto_reindex_with_identity = lambda *_args, **_kwargs: (
        _result(),
        False,
        {},
    )
    monkeypatch.delenv("CODE_SEARCH_DISABLE_AUTO_REINDEX", raising=False)
    monkeypatch.delenv("CODE_SEARCH_NONBLOCKING_SEARCH", raising=False)
    monkeypatch.setattr(
        incremental_indexer_module,
        "IncrementalIndexer",
        FakeIncrementalIndexer,
    )
    monkeypatch.setattr(
        code_search_server_module,
        "MultiLanguageChunker",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        code_search_server_module,
        "get_pipeline_version",
        lambda effective: (
            "effective-blocking-pipeline"
            if effective is configuration
            else "wrong-pipeline"
        ),
    )

    response = json.loads(server.search_code("find authentication"))

    assert response["results"] == []
    assert bound == {
        "configuration": configuration,
        "pipeline_version": "effective-blocking-pipeline",
    }
