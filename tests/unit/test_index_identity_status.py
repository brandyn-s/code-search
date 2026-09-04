"""MCP status contract for persisted cross-engine index identity."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace

import mcp_server.code_search_server as code_search_server_module
from mcp_server.code_search_server import CodeSearchServer
from embeddings.embedder import EffectiveEmbeddingConfig
import pytest
from search.index_identity import IndexIdentity, derive_index_generation
from search.index_identity import IdentityCaptureError
import search.incremental_indexer as incremental_indexer_module
import merkle.snapshot_manager as snapshot_manager_module


def _identity() -> IndexIdentity:
    repository_id = "a" * 64
    source_revision = "c" * 40
    dirty_fingerprint = "clean"
    return IndexIdentity(
        schema_version=1,
        repository_id=repository_id,
        checkout_id="b" * 64,
        source_revision=source_revision,
        dirty_fingerprint=dirty_fingerprint,
        index_generation=derive_index_generation(
            repository_id=repository_id,
            source_revision=source_revision,
            dirty_fingerprint=dirty_fingerprint,
        ),
        captured_at="2026-07-26T18:00:00Z",
    )


def test_get_index_status_emits_persisted_ready_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    (project_dir / "project_info.json").write_text(
        json.dumps(
            {
                "project_path": str(tmp_path / "source"),
                "index_identity_status": "ready",
                "index_identity": _identity().to_dict(),
                "synonym_profile": {
                    "name": "generic",
                    "version": 1,
                    "id": "generic-v1",
                },
            }
        ),
        encoding="utf-8",
    )
    server = CodeSearchServer.__new__(CodeSearchServer)
    server._current_project = str(tmp_path / "source")
    server._current_provider = None
    monkeypatch.setattr(
        server,
        "get_index_manager",
        lambda: SimpleNamespace(get_stats=lambda: {"total_chunks": 7}),
    )
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

    status = json.loads(server.get_index_status())

    assert status["index_identity_status"] == "ready"
    assert status["index_ready"] is True
    assert status["index_identity"] == _identity().to_dict()
    assert status["synonym_profile"] == {
        "name": "generic",
        "version": 1,
        "id": "generic-v1",
    }


def test_get_index_status_reports_live_source_staleness(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    (project_dir / "project_info.json").write_text(
        json.dumps(
            {
                "project_path": str(source),
                "index_identity_status": "ready",
                "index_identity": _identity().to_dict(),
            }
        ),
        encoding="utf-8",
    )
    server = CodeSearchServer.__new__(CodeSearchServer)
    server._current_project = str(source)
    server._current_provider = None
    monkeypatch.setattr(
        server,
        "get_index_manager",
        lambda: SimpleNamespace(get_stats=lambda: {"total_chunks": 7}),
    )
    monkeypatch.setattr(
        server,
        "get_project_storage_dir",
        lambda *_args, **_kwargs: project_dir,
    )
    changed = replace(
        _identity(),
        dirty_fingerprint="e" * 64,
        index_generation="f" * 64,
    )
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda path: changed if path == source else _identity(),
    )

    status = json.loads(server.get_index_status())

    assert status["index_identity_status"] == "stale_source"
    assert status["index_ready"] is False
    assert "source_changed_since_index" in status["index_identity_error"]
    assert status["index_identity"] == _identity().to_dict()


def test_get_index_status_rejects_same_generation_from_another_checkout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    (project_dir / "project_info.json").write_text(
        json.dumps(
            {
                "project_path": str(source),
                "index_identity_status": "ready",
                "index_identity": _identity().to_dict(),
            }
        ),
        encoding="utf-8",
    )
    server = CodeSearchServer.__new__(CodeSearchServer)
    server._current_project = str(source)
    server._current_provider = None
    monkeypatch.setattr(
        server,
        "get_index_manager",
        lambda: SimpleNamespace(get_stats=lambda: {"total_chunks": 7}),
    )
    monkeypatch.setattr(
        server,
        "get_project_storage_dir",
        lambda *_args, **_kwargs: project_dir,
    )
    retargeted = replace(_identity(), checkout_id="e" * 64)
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda _path: retargeted,
    )

    status = json.loads(server.get_index_status())

    assert status["index_identity_status"] == "stale_source"
    assert status["index_ready"] is False
    assert "checkout_id" in status["index_identity_error"]
    assert f"{'b' * 64} -> {'e' * 64}" in status["index_identity_error"]


def test_get_index_status_rejects_corrupt_persisted_envelope(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    corrupt = _identity().to_dict()
    corrupt["index_generation"] = "0" * 64
    (project_dir / "project_info.json").write_text(
        json.dumps(
            {
                "project_path": str(source),
                "index_identity_status": "ready",
                "index_identity": corrupt,
            }
        ),
        encoding="utf-8",
    )
    server = CodeSearchServer.__new__(CodeSearchServer)
    server._current_project = str(source)
    server._current_provider = None
    monkeypatch.setattr(
        server,
        "get_index_manager",
        lambda: SimpleNamespace(get_stats=lambda: {"total_chunks": 7}),
    )
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

    status = json.loads(server.get_index_status())

    assert status["index_identity_status"] == "error"
    assert status["index_ready"] is False
    assert "index_generation" in status["index_identity_error"]


def test_get_index_status_marks_legacy_identity_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    (project_dir / "project_info.json").write_text(
        json.dumps({"project_path": str(source)}),
        encoding="utf-8",
    )
    server = CodeSearchServer.__new__(CodeSearchServer)
    server._current_project = str(source)
    server._current_provider = None
    monkeypatch.setattr(
        server,
        "get_index_manager",
        lambda: SimpleNamespace(get_stats=lambda: {"total_chunks": 7}),
    )
    monkeypatch.setattr(
        server,
        "get_project_storage_dir",
        lambda *_args, **_kwargs: project_dir,
    )

    status = json.loads(server.get_index_status())

    assert status["index_identity_status"] == "legacy_missing"
    assert status["index_ready"] is False
    assert "index_identity" not in status


def test_successful_finalize_persists_the_end_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    info_file = tmp_path / "project_info.json"
    info_file.write_text(
        json.dumps({"project_path": str(tmp_path / "source")}),
        encoding="utf-8",
    )
    server = CodeSearchServer.__new__(CodeSearchServer)
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda _path: _identity(),
    )

    result = server._finalize_index_identity(
        tmp_path / "source",
        info_file,
        _identity(),
    )

    persisted = json.loads(info_file.read_text(encoding="utf-8"))
    assert result == {
        "index_identity_status": "ready",
        "index_identity": _identity().to_dict(),
    }
    assert persisted["index_identity_status"] == "ready"
    assert persisted["index_identity"] == _identity().to_dict()


def test_finalize_publishes_identity_and_provenance_together(
    tmp_path: Path,
    monkeypatch,
) -> None:
    info_file = tmp_path / "project_info.json"
    info_file.write_text(
        json.dumps({"project_path": str(tmp_path / "source")}),
        encoding="utf-8",
    )
    server = CodeSearchServer.__new__(CodeSearchServer)
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda _path: _identity(),
    )
    ready_metadata = {
        "pipeline_version": "pipeline-v2",
        "synonym_profile": {
            "name": "generic",
            "version": 1,
            "id": "generic-v1",
        },
    }

    result = server._finalize_index_identity(
        tmp_path / "source",
        info_file,
        _identity(),
        ready_metadata=ready_metadata,
    )

    persisted = json.loads(info_file.read_text(encoding="utf-8"))
    assert result["pipeline_version"] == "pipeline-v2"
    assert result["synonym_profile"]["id"] == "generic-v1"
    assert persisted["index_identity_status"] == "ready"
    assert persisted["index_identity"] == _identity().to_dict()
    assert persisted["pipeline_version"] == "pipeline-v2"
    assert persisted["synonym_profile"]["id"] == "generic-v1"


def test_failed_project_metadata_replace_preserves_old_valid_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    info_file = tmp_path / "project_info.json"
    original = {
        "project_path": str(tmp_path / "source"),
        "pipeline_version": "pipeline-v1",
        "index_identity_status": "ready",
        "index_identity": _identity().to_dict(),
    }
    info_file.write_text(json.dumps(original), encoding="utf-8")
    server = CodeSearchServer.__new__(CodeSearchServer)

    def fail_replace(_source, _destination):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(
        code_search_server_module.os,
        "replace",
        fail_replace,
    )

    result = server._persist_index_identity_state(
        info_file,
        {
            "index_identity_status": "error",
            "index_identity_error": "new failure",
        },
    )

    assert result["index_identity_status"] == "error"
    assert "Could not persist index identity" in result["index_identity_error"]
    assert json.loads(info_file.read_text(encoding="utf-8")) == original


def test_finalize_rejects_source_changed_during_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    info_file = tmp_path / "project_info.json"
    info_file.write_text(
        json.dumps(
            {
                "project_path": str(tmp_path / "source"),
                "index_identity_status": "ready",
                "index_identity": _identity().to_dict(),
            }
        ),
        encoding="utf-8",
    )
    server = CodeSearchServer.__new__(CodeSearchServer)
    changed = replace(
        _identity(),
        dirty_fingerprint="e" * 64,
        index_generation="f" * 64,
    )
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda _path: changed,
    )

    result = server._finalize_index_identity(
        tmp_path / "source",
        info_file,
        _identity(),
    )

    persisted = json.loads(info_file.read_text(encoding="utf-8"))
    assert result["index_identity_status"] == "source_changed_during_index"
    assert "stable checkout" in result["index_identity_error"]
    assert persisted["index_identity_status"] == "source_changed_during_index"
    assert "index_identity" not in persisted


def test_finalize_rejects_same_generation_from_another_checkout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    info_file = tmp_path / "project_info.json"
    info_file.write_text(
        json.dumps({"project_path": str(tmp_path / "source")}),
        encoding="utf-8",
    )
    server = CodeSearchServer.__new__(CodeSearchServer)
    retargeted = replace(_identity(), checkout_id="e" * 64)
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda _path: retargeted,
    )

    result = server._finalize_index_identity(
        tmp_path / "source",
        info_file,
        _identity(),
    )

    assert result["index_identity_status"] == "source_changed_during_index"
    assert "checkout_id" in result["index_identity_error"]
    assert f"{'b' * 64} -> {'e' * 64}" in result["index_identity_error"]


def test_index_directory_captures_identity_before_worker_start(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    info_file = project_dir / "project_info.json"
    info_file.write_text(
        json.dumps(
            {
                "project_path": str(source),
                "index_identity_status": "ready",
                "index_identity": _identity().to_dict(),
            }
        ),
        encoding="utf-8",
    )
    server = CodeSearchServer.__new__(CodeSearchServer)
    server._indexing_job = None
    capture_calls: list[Path] = []
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda path: capture_calls.append(path) or _identity(),
    )
    monkeypatch.setattr(
        server,
        "get_project_storage_dir",
        lambda *_args, **_kwargs: project_dir,
    )

    class DeferredThread:
        def __init__(self, *, target, daemon):
            self.target = target
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True

    monkeypatch.setattr(threading, "Thread", DeferredThread)

    response = json.loads(server.index_directory(str(source)))

    assert response["status"] == "indexing"
    assert response["index_ready"] is False
    assert capture_calls == [source.resolve()]
    assert server._indexing_job["identity_start"] == _identity()
    assert server._indexing_thread.started is True
    persisted = json.loads(info_file.read_text(encoding="utf-8"))
    assert persisted["index_identity_status"] == "indexing"
    assert "index_identity" not in persisted


def _configure_deferred_successful_index(
    server: CodeSearchServer,
    project_dir: Path,
    monkeypatch,
    *,
    outcome: str = "success",
) -> None:
    class DeferredThread:
        def __init__(self, *, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            return None

    class FakeIncrementalIndexer:
        def __init__(self, **_kwargs):
            pass

        def incremental_index(self, *_args, **_kwargs):
            if outcome == "cancelled":
                raise InterruptedError("cancelled")
            if outcome == "exception":
                raise RuntimeError("index exploded")
            return SimpleNamespace(
                success=outcome == "success",
                files_added=1,
                files_removed=0,
                files_modified=0,
                chunks_added=2,
                chunks_removed=0,
                time_taken=0.1,
                error=None if outcome == "success" else "embedding failed",
            )

        def get_indexing_stats(self, _path):
            return {"total_chunks": 2}

    monkeypatch.setattr(threading, "Thread", DeferredThread)
    monkeypatch.setattr(
        incremental_indexer_module,
        "IncrementalIndexer",
        FakeIncrementalIndexer,
    )
    monkeypatch.setattr(
        snapshot_manager_module,
        "SnapshotManager",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        code_search_server_module,
        "MultiLanguageChunker",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        server,
        "get_project_storage_dir",
        lambda *_args, **_kwargs: project_dir,
    )
    monkeypatch.setattr(
        server,
        "get_index_manager",
        lambda *_args, **_kwargs: SimpleNamespace(
            get_index_size=lambda: 1,
            bind_embedding_configuration=lambda *_args, **_kwargs: None,
        ),
    )
    configuration = EffectiveEmbeddingConfig(
        provider="local",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        content_mode="code",
        output_dimension=384,
    )
    monkeypatch.setattr(
        server,
        "embedder",
        lambda *_args, **_kwargs: SimpleNamespace(
            configuration=configuration,
        ),
    )
    monkeypatch.setattr(server, "_maybe_start_model_preload", lambda: None)


def test_background_worker_hashes_constructed_config_without_env_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    (project_dir / "project_info.json").write_text(
        json.dumps({"project_path": str(source)}),
        encoding="utf-8",
    )
    server = CodeSearchServer.__new__(CodeSearchServer)
    server._indexing_job = None
    server._index_manager = None
    server._searcher = None
    _configure_deferred_successful_index(server, project_dir, monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    configuration = EffectiveEmbeddingConfig(
        provider="voyage",
        model_name="voyage-4-large",
        content_mode="code",
        output_dimension=1024,
        input_type_enabled=True,
    )

    def build_embedder(*_args, **kwargs):
        assert kwargs["provider"] == "voyage"
        assert (
            os.environ["EMBEDDING_PROVIDER"] == "openai"
        ), "per-call provider must not mutate process-global environment"
        return SimpleNamespace(configuration=configuration)

    captured: dict[str, object] = {}

    def pipeline_version(effective):
        captured["configuration"] = effective
        return "effective-pipeline-version"

    index_manager = SimpleNamespace(
        get_index_size=lambda: 1,
        bind_embedding_configuration=lambda effective, pipeline_version: (
            captured.update(
                {
                    "bound_configuration": effective,
                    "bound_pipeline_version": pipeline_version,
                }
            )
        ),
    )
    monkeypatch.setattr(
        server,
        "get_index_manager",
        lambda *_args, **_kwargs: index_manager,
    )
    monkeypatch.setattr(server, "embedder", build_embedder)
    monkeypatch.setattr(
        code_search_server_module,
        "get_pipeline_version",
        pipeline_version,
    )
    captures = iter((_identity(), _identity()))
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda _path: next(captures),
    )

    server.index_directory(str(source), provider="voyage")
    server._indexing_thread.target()

    assert captured["configuration"] == configuration
    assert captured["bound_configuration"] == configuration
    assert (
        captured["bound_pipeline_version"]
        == "effective-pipeline-version"
    )
    assert os.environ["EMBEDDING_PROVIDER"] == "openai"
    persisted = json.loads(
        (project_dir / "project_info.json").read_text(encoding="utf-8")
    )
    assert persisted["embedding_provider"] == configuration.provider
    assert persisted["embedding_model"] == configuration.model_name
    assert (
        persisted["embedding_dimension"]
        == configuration.output_dimension
    )
    assert persisted["content_mode"] == configuration.content_mode
    assert (
        persisted["embedding_input_type_enabled"]
        is configuration.input_type_enabled
    )


def test_background_worker_publishes_ready_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    info_file = project_dir / "project_info.json"
    info_file.write_text(
        json.dumps({"project_path": str(source)}),
        encoding="utf-8",
    )
    server = CodeSearchServer.__new__(CodeSearchServer)
    server._indexing_job = None
    server._index_manager = None
    server._searcher = None
    _configure_deferred_successful_index(server, project_dir, monkeypatch)
    active_profile = {
        "value": {
            "name": "generic",
            "version": 1,
            "id": "generic-v1",
        }
    }
    monkeypatch.setattr(
        code_search_server_module,
        "_active_synonym_profile_metadata",
        lambda: dict(active_profile["value"]),
    )
    captures = iter((_identity(), _identity()))
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda _path: next(captures),
    )

    start = json.loads(server.index_directory(str(source)))
    active_profile["value"] = {
        "name": "generic",
        "version": 1,
        "id": "generic-v1",
    }
    server._indexing_thread.target()
    progress = json.loads(server.get_indexing_progress())

    assert start["index_ready"] is False
    assert progress["status"] == "completed"
    assert progress["index_ready"] is True
    assert progress["result"]["index_ready"] is True
    assert progress["result"]["index_identity_status"] == "ready"
    assert progress["result"]["synonym_profile"] == {
        "name": "generic",
        "version": 1,
        "id": "generic-v1",
    }
    persisted = json.loads(info_file.read_text(encoding="utf-8"))
    assert persisted["index_identity_status"] == "ready"
    assert persisted["index_identity"] == _identity().to_dict()
    assert persisted["synonym_profile"] == {
        "name": "generic",
        "version": 1,
        "id": "generic-v1",
    }


def test_background_worker_rejects_mid_index_source_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    info_file = project_dir / "project_info.json"
    info_file.write_text(
        json.dumps({"project_path": str(source)}),
        encoding="utf-8",
    )
    server = CodeSearchServer.__new__(CodeSearchServer)
    server._indexing_job = None
    server._index_manager = None
    server._searcher = None
    _configure_deferred_successful_index(server, project_dir, monkeypatch)
    changed = replace(
        _identity(),
        dirty_fingerprint="e" * 64,
        index_generation="f" * 64,
    )
    captures = iter((_identity(), changed))
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda _path: next(captures),
    )

    server.index_directory(str(source))
    server._indexing_thread.target()
    progress = json.loads(server.get_indexing_progress())

    assert progress["status"] == "failed"
    assert progress["index_ready"] is False
    assert progress["result"]["index_ready"] is False
    assert (
        progress["result"]["index_identity_status"]
        == "source_changed_during_index"
    )
    assert "source_changed_during_index" in progress["result"]["error"]
    persisted = json.loads(info_file.read_text(encoding="utf-8"))
    assert (
        persisted["index_identity_status"]
        == "source_changed_during_index"
    )
    assert "index_identity" not in persisted


def test_identity_incoherent_run_preserves_previous_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    info_file = project_dir / "project_info.json"
    previous_profile = {
        "name": "generic",
        "version": 1,
        "id": "generic-v1",
    }
    info_file.write_text(
        json.dumps(
            {
                "project_path": str(source),
                "pipeline_version": "pipeline-v1",
                "synonym_profile": previous_profile,
            }
        ),
        encoding="utf-8",
    )
    server = CodeSearchServer.__new__(CodeSearchServer)
    server._indexing_job = None
    server._index_manager = None
    server._searcher = None
    _configure_deferred_successful_index(server, project_dir, monkeypatch)
    monkeypatch.setattr(
        code_search_server_module,
        "get_pipeline_version",
        lambda _configuration=None: "pipeline-v2",
    )
    monkeypatch.setattr(
        code_search_server_module,
        "_active_synonym_profile_metadata",
        lambda: {
            "name": "generic",
            "version": 1,
            "id": "generic-v1",
        },
    )
    changed = replace(
        _identity(),
        dirty_fingerprint="e" * 64,
        index_generation="f" * 64,
    )
    captures = iter((_identity(), changed))
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda _path: next(captures),
    )

    server.index_directory(str(source))
    server._indexing_thread.target()
    progress = json.loads(server.get_indexing_progress())

    assert progress["status"] == "failed"
    persisted = json.loads(info_file.read_text(encoding="utf-8"))
    assert persisted["pipeline_version"] == "pipeline-v1"
    assert persisted["synonym_profile"] == previous_profile
    assert (
        persisted["index_identity_status"]
        == "source_changed_during_index"
    )


def test_completed_metadata_failure_cannot_publish_ready_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    info_file = project_dir / "project_info.json"
    previous_profile = {
        "name": "generic",
        "version": 1,
        "id": "generic-v1",
    }
    info_file.write_text(
        json.dumps(
            {
                "project_path": str(source),
                "pipeline_version": "pipeline-v1",
                "synonym_profile": previous_profile,
            }
        ),
        encoding="utf-8",
    )
    server = CodeSearchServer.__new__(CodeSearchServer)
    server._indexing_job = None
    server._index_manager = None
    server._searcher = None
    _configure_deferred_successful_index(server, project_dir, monkeypatch)
    monkeypatch.setattr(
        code_search_server_module,
        "get_pipeline_version",
        lambda _configuration=None: "pipeline-v2",
    )
    captures = iter((_identity(), _identity()))
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda _path: next(captures),
    )
    real_update = code_search_server_module._update_project_info

    def fail_completed_metadata(
        target: Path,
        updates: dict,
        *,
        remove_fields: tuple[str, ...] = (),
    ):
        if "pipeline_version" in updates:
            raise OSError("synthetic completed metadata failure")
        return real_update(
            target,
            updates,
            remove_fields=remove_fields,
        )

    monkeypatch.setattr(
        code_search_server_module,
        "_update_project_info",
        fail_completed_metadata,
    )

    server.index_directory(str(source))
    server._indexing_thread.target()
    progress = json.loads(server.get_indexing_progress())

    assert progress["status"] == "failed"
    assert progress["index_ready"] is False
    assert progress["result"]["success"] is False
    assert progress["result"]["index_identity_status"] == "error"
    assert "Could not persist index identity" in progress["result"]["error"]
    persisted = json.loads(info_file.read_text(encoding="utf-8"))
    assert persisted["pipeline_version"] == "pipeline-v1"
    assert persisted["synonym_profile"] == previous_profile
    assert persisted["index_identity_status"] != "ready"
    assert "index_identity" not in persisted


def test_background_worker_rejects_success_without_start_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    info_file = project_dir / "project_info.json"
    info_file.write_text(
        json.dumps({"project_path": str(source)}),
        encoding="utf-8",
    )
    index_artifact = project_dir / "code.index"
    index_artifact.write_bytes(b"non-git-index-data")
    server = CodeSearchServer.__new__(CodeSearchServer)
    server._indexing_job = None
    server._index_manager = None
    server._searcher = None
    _configure_deferred_successful_index(server, project_dir, monkeypatch)
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda _path: (_ for _ in ()).throw(
            IdentityCaptureError("source identity unavailable")
        ),
    )

    server.index_directory(str(source))
    server._indexing_thread.target()
    progress = json.loads(server.get_indexing_progress())

    assert progress["status"] == "failed"
    assert progress["phase"] == "error"
    assert progress["index_ready"] is False
    assert progress["result"]["success"] is False
    assert progress["result"]["index_ready"] is False
    assert progress["result"]["index_identity_status"] == "error"
    assert "identity_capture_start_failed" in progress["result"]["error"]
    assert index_artifact.read_bytes() == b"non-git-index-data"
    persisted = json.loads(info_file.read_text(encoding="utf-8"))
    assert persisted["index_identity_status"] == "error"
    assert "index_identity" not in persisted


@pytest.mark.parametrize(
    ("outcome", "job_status", "identity_status"),
    [
        ("failed", "failed", "error"),
        ("cancelled", "cancelled", "cancelled"),
        ("exception", "failed", "error"),
    ],
)
def test_background_worker_persists_nonready_terminal_identity(
    tmp_path: Path,
    monkeypatch,
    outcome: str,
    job_status: str,
    identity_status: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "stored-project"
    project_dir.mkdir()
    info_file = project_dir / "project_info.json"
    info_file.write_text(
        json.dumps({"project_path": str(source)}),
        encoding="utf-8",
    )
    server = CodeSearchServer.__new__(CodeSearchServer)
    server._indexing_job = None
    server._index_manager = None
    server._searcher = None
    _configure_deferred_successful_index(
        server,
        project_dir,
        monkeypatch,
        outcome=outcome,
    )
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda _path: _identity(),
    )

    server.index_directory(str(source))
    server._indexing_thread.target()
    progress = json.loads(server.get_indexing_progress())

    assert progress["status"] == job_status
    assert progress["index_ready"] is False
    assert progress["result"]["index_ready"] is False
    assert progress["result"]["index_identity_status"] == identity_status
    persisted = json.loads(info_file.read_text(encoding="utf-8"))
    assert persisted["index_identity_status"] == identity_status
    assert persisted["index_identity_status"] != "indexing"
    assert "index_identity" not in persisted
