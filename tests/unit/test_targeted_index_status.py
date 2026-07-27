"""Read-only status inspection for a project that is not active yet."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pytest

import mcp_server.code_search_server as server_module
from common_utils import get_storage_dir
from mcp_server.code_search_mcp import CodeSearchMCP
from mcp_server.code_search_server import CodeSearchServer
from search.index_identity import IndexIdentity, derive_index_generation


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


@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch):
    storage = tmp_path / "storage"
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(storage))
    get_storage_dir.cache_clear()
    yield storage
    get_storage_dir.cache_clear()


def _fresh_server() -> tuple[CodeSearchServer, object, object]:
    manager_sentinel = object()
    searcher_sentinel = object()
    server = CodeSearchServer.__new__(CodeSearchServer)
    server._current_project = None
    server._current_provider = None
    server._index_manager = manager_sentinel
    server._searcher = searcher_sentinel
    server._indexing_job = None
    server._indexing_job_lock = threading.RLock()
    return server, manager_sentinel, searcher_sentinel


def _write_ready_project(
    project_dir: Path,
    source: Path,
    *,
    provider: str,
    model: str = "voyage-4-large",
    total_chunks: int = 7,
) -> None:
    index_dir = project_dir / "index"
    index_dir.mkdir(parents=True)
    (index_dir / "code.index").write_bytes(b"existing-index")
    (index_dir / "stats.json").write_text(
        json.dumps({"total_chunks": total_chunks}),
        encoding="utf-8",
    )
    (project_dir / "project_info.json").write_text(
        json.dumps(
            {
                "project_path": str(source.resolve()),
                "embedding_provider": provider,
                "embedding_model": model,
                "index_identity_status": "ready",
                "index_identity": _identity().to_dict(),
            }
        ),
        encoding="utf-8",
    )


def _forbid_index_manager(monkeypatch) -> None:
    class UnexpectedIndexManager:
        def __init__(self, _storage_dir: str):
            raise AssertionError(
                "targeted status must not construct a mutable index manager"
            )

    monkeypatch.setattr(
        server_module,
        "CodeIndexManager",
        UnexpectedIndexManager,
    )


def test_fresh_server_reports_completed_index_before_switch_project(
    isolated_storage: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    server, manager_sentinel, searcher_sentinel = _fresh_server()
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda path: _identity() if path == source.resolve() else None,
    )

    class DeferredThread:
        def __init__(self, *, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self) -> None:
            return None

    monkeypatch.setattr(threading, "Thread", DeferredThread)
    started = json.loads(server.index_directory(str(source), provider="voyage"))
    project_dir = Path(started["storage_target"])
    _write_ready_project(
        project_dir,
        source,
        provider="voyage",
        total_chunks=11,
    )
    server._update_indexing_job(
        started["job_id"],
        status="completed",
        phase="done",
        index_ready=True,
        result={
            "success": True,
            "index_ready": True,
            "index_identity_status": "ready",
            "index_identity": _identity().to_dict(),
        },
    )
    _forbid_index_manager(monkeypatch)

    status = json.loads(server.get_index_status(project_path=str(source)))

    assert status["project_path"] == str(source.resolve())
    assert status["provider"] == "voyage"
    assert status["storage_target"] == str(project_dir.resolve())
    assert status["index_statistics"]["total_chunks"] == 11
    assert status["index_identity_status"] == "ready"
    assert status["index_ready"] is True
    assert status["index_identity"] == _identity().to_dict()
    assert status["source_identity"] == _identity().to_dict()
    assert status["indexing_job"]["job_id"] == started["job_id"]
    assert status["indexing_job"]["status"] == "completed"
    assert server._current_project is None
    assert server._current_provider is None
    assert server._index_manager is manager_sentinel
    assert server._searcher is searcher_sentinel


def test_targeted_status_resolves_provider_aware_sibling_without_switching(
    isolated_storage: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    active_source = tmp_path / "active"
    target_source = tmp_path / "target"
    active_source.mkdir()
    target_source.mkdir()
    server, manager_sentinel, searcher_sentinel = _fresh_server()
    server._current_project = str(active_source)
    server._current_provider = "openai"
    project_dir = server_module._planned_index_storage_target(
        target_source,
        "voyage-context",
    )
    _write_ready_project(
        project_dir,
        target_source,
        provider="voyage-context",
        model="voyage-context-3",
        total_chunks=13,
    )
    legacy_hash = hashlib.md5(str(target_source.resolve()).encode()).hexdigest()[:8]
    legacy_target = (
        isolated_storage / "projects" / f"{target_source.name}_{legacy_hash}"
    )
    _forbid_index_manager(monkeypatch)
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda path: _identity() if path == target_source.resolve() else None,
    )

    status = json.loads(server.get_index_status(project_path=str(target_source)))

    assert status["provider"] == "voyage-context"
    assert status["model_information"]["model_name"] == "voyage-context-3"
    assert status["index_statistics"]["total_chunks"] == 13
    assert status["index_ready"] is True
    assert not legacy_target.exists()
    assert server._current_project == str(active_source)
    assert server._current_provider == "openai"
    assert server._index_manager is manager_sentinel
    assert server._searcher is searcher_sentinel


def test_targeted_status_rejects_ambiguous_provider_indexes(
    isolated_storage: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    server, manager_sentinel, searcher_sentinel = _fresh_server()
    providers = ("voyage", "voyage-context")
    for provider in providers:
        project_dir = server_module._planned_index_storage_target(
            source,
            provider,
        )
        _write_ready_project(
            project_dir,
            source,
            provider=provider,
        )
    _forbid_index_manager(monkeypatch)
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("ambiguous target must fail before identity capture")
        ),
    )

    status = json.loads(server.get_index_status(project_path=str(source)))

    assert status["index_identity_status"] == "ambiguous_index"
    assert status["index_ready"] is False
    assert set(status["available_providers"]) == set(providers)
    assert "multiple populated indexes" in status["error"]
    assert server._current_project is None
    assert server._current_provider is None
    assert server._index_manager is manager_sentinel
    assert server._searcher is searcher_sentinel


@pytest.mark.parametrize(
    "terminal_result",
    [
        None,
        {"success": False, "index_ready": True, "error": None},
        {"success": True, "index_ready": False, "error": None},
        {"success": True, "index_ready": True, "error": "partial failure"},
    ],
)
def test_targeted_status_rejects_incoherent_terminal_job(
    isolated_storage: Path,
    tmp_path: Path,
    monkeypatch,
    terminal_result: dict | None,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    server, manager_sentinel, searcher_sentinel = _fresh_server()
    project_dir = server_module._planned_index_storage_target(
        source,
        "voyage",
    )
    _write_ready_project(
        project_dir,
        source,
        provider="voyage",
    )
    server._indexing_job = {
        "job_id": "terminal-job",
        "status": "completed",
        "phase": "done",
        "current": 7,
        "total": 7,
        "index_ready": True,
        "directory": str(source.resolve()),
        "provider": "voyage",
        "storage_target": str(project_dir.resolve()),
        "result": terminal_result,
    }
    _forbid_index_manager(monkeypatch)
    monkeypatch.setattr(
        server,
        "_capture_index_identity",
        lambda path: _identity() if path == source.resolve() else None,
    )

    status = json.loads(server.get_index_status(project_path=str(source)))

    assert status["index_ready"] is False
    assert status["index_identity_status"] == "error"
    assert "terminal result is not a coherent success" in status["index_identity_error"]
    assert server._current_project is None
    assert server._current_provider is None
    assert server._index_manager is manager_sentinel
    assert server._searcher is searcher_sentinel


@pytest.mark.parametrize("source_exists", [False, True])
def test_targeted_status_does_not_create_storage_for_unknown_project(
    isolated_storage: Path,
    tmp_path: Path,
    monkeypatch,
    source_exists: bool,
) -> None:
    source = tmp_path / "unknown"
    if source_exists:
        source.mkdir()
    server, manager_sentinel, searcher_sentinel = _fresh_server()

    _forbid_index_manager(monkeypatch)

    status = json.loads(server.get_index_status(project_path=str(source)))

    expected_status = "not_indexed" if source_exists else "not_found"
    assert status["index_identity_status"] == expected_status
    assert status["index_ready"] is False
    assert not (isolated_storage / "projects").exists()
    assert server._current_project is None
    assert server._current_provider is None
    assert server._index_manager is manager_sentinel
    assert server._searcher is searcher_sentinel


def test_targeted_status_reports_invalid_project_info_without_switching(
    isolated_storage: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    server, manager_sentinel, searcher_sentinel = _fresh_server()
    project_dir = server_module._planned_index_storage_target(source, None)
    index_dir = project_dir / "index"
    index_dir.mkdir(parents=True)
    (index_dir / "code.index").write_bytes(b"existing-index")
    (project_dir / "project_info.json").write_text(
        "{not valid json",
        encoding="utf-8",
    )

    _forbid_index_manager(monkeypatch)

    status = json.loads(server.get_index_status(project_path=str(source)))

    assert status["index_identity_status"] == "error"
    assert status["index_ready"] is False
    assert "project_info identity could not be read" in status["index_identity_error"]
    assert server._current_project is None
    assert server._current_provider is None
    assert server._index_manager is manager_sentinel
    assert server._searcher is searcher_sentinel


def test_get_index_status_mcp_schema_exposes_optional_project_path() -> None:
    server = CodeSearchServer.__new__(CodeSearchServer)
    mcp = CodeSearchMCP(server)

    tool = mcp._tool_manager._tools["get_index_status"]

    assert "project_path" in tool.parameters["properties"]
    assert "project_path" not in tool.parameters.get("required", [])
    assert "project_path" in tool.description
    assert tool.annotations.readOnlyHint is True
    assert tool.annotations.idempotentHint is True
