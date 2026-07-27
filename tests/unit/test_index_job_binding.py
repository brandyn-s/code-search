"""A process-global index job must identify the project it actually owns."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import mcp_server.code_search_server as server_module
from mcp_server.code_search_server import CodeSearchServer


def _server_with_active_job(source: Path) -> CodeSearchServer:
    server = CodeSearchServer.__new__(CodeSearchServer)
    server._indexing_job_lock = threading.RLock()
    server._indexing_job = {
        "job_id": "active123",
        "status": "indexing",
        "phase": "embedding",
        "current": 4,
        "total": 10,
        "directory": str(source.resolve()),
        "project_name": source.name,
    }
    return server


def test_index_directory_reports_cross_project_job_conflict(
    tmp_path: Path,
) -> None:
    active_source = tmp_path / "repo-a"
    requested_source = tmp_path / "repo-b"
    active_source.mkdir()
    requested_source.mkdir()
    server = _server_with_active_job(active_source)

    response = json.loads(server.index_directory(str(requested_source)))

    assert response["status"] == "indexing"
    assert response["job_id"] == "active123"
    assert response["directory"] == str(active_source.resolve())
    assert response["project_name"] == "repo-a"
    assert response["requested_directory"] == str(requested_source.resolve())
    assert response["indexing_conflict"] is True
    assert response["conflict_reason"] == "different_project_indexing"
    assert "repo-a" in response["message"]
    assert "repo-b" in response["message"]
    assert server._indexing_job["job_id"] == "active123"


def test_index_directory_reuses_same_project_job_without_conflict(
    tmp_path: Path,
) -> None:
    active_source = tmp_path / "repo-a"
    active_source.mkdir()
    server = _server_with_active_job(active_source)

    response = json.loads(server.index_directory(str(active_source)))

    assert response["job_id"] == "active123"
    assert response["directory"] == str(active_source.resolve())
    assert response["project_name"] == "repo-a"
    assert response["requested_directory"] == str(active_source.resolve())
    assert response["indexing_conflict"] is False
    assert "conflict_reason" not in response


def test_index_directory_rejects_same_project_job_for_different_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import threading

    from common_utils import get_storage_dir

    source = tmp_path / "repo-a"
    source.mkdir()
    storage = tmp_path / "storage"
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(storage))
    get_storage_dir.cache_clear()
    monkeypatch.setattr(threading.Thread, "start", lambda _thread: None)
    server = CodeSearchServer.__new__(CodeSearchServer)
    server._indexing_job_lock = threading.RLock()
    server._indexing_job = None
    server._index_manager = None
    server._searcher = None

    started = json.loads(
        server.index_directory(
            str(source),
            provider="voyage",
        )
    )
    active_target = Path(server._indexing_job["identity_info_file"]).parent
    requested_target = server.get_project_storage_dir(
        str(source),
        provider="voyage-context",
    )

    response = json.loads(
        server.index_directory(
            str(source),
            provider="voyage-context",
        )
    )

    assert started["provider"] == "voyage"
    assert started["storage_target"] == str(active_target)
    assert response["indexing_conflict"] is True
    assert response["conflict_reason"] == "different_provider_indexing"
    assert response["provider"] == "voyage"
    assert response["requested_provider"] == "voyage-context"
    assert response["storage_target"] == str(active_target)
    assert response["requested_storage_target"] == str(requested_target)
    assert "reusing" not in response["message"]


def test_concurrent_index_directory_calls_claim_one_foreground_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from common_utils import get_storage_dir

    source = tmp_path / "repo-a"
    source.mkdir()
    storage = tmp_path / "storage"
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(storage))
    get_storage_dir.cache_clear()

    callers_ready = threading.Barrier(2)

    def synchronize_after_validation(_path: str) -> None:
        callers_ready.wait(timeout=5)

    monkeypatch.setattr(
        server_module,
        "_refuse_as_project_root_reason",
        synchronize_after_validation,
    )

    started_workers = []

    class DeferredWorker:
        def __init__(self, *, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self) -> None:
            started_workers.append(self)

    caller_thread_type = threading.Thread
    monkeypatch.setattr(threading, "Thread", DeferredWorker)

    server = CodeSearchServer.__new__(CodeSearchServer)
    server._indexing_job_lock = threading.RLock()
    server._indexing_job = None
    server._index_manager = None
    server._searcher = None

    responses = []

    def call_index_directory() -> None:
        responses.append(json.loads(server.index_directory(str(source))))

    callers = [
        caller_thread_type(target=call_index_directory),
        caller_thread_type(target=call_index_directory),
    ]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=5)

    assert all(not caller.is_alive() for caller in callers)
    assert len(responses) == 2
    assert {response["job_id"] for response in responses} == {
        server._indexing_job["job_id"]
    }
    assert len(started_workers) == 1


def test_stale_worker_update_cannot_mutate_replacement_job() -> None:
    server = CodeSearchServer.__new__(CodeSearchServer)
    server._indexing_job_lock = threading.RLock()
    server._indexing_job = {
        "job_id": "replacement",
        "status": "indexing",
        "phase": "starting",
        "result": None,
    }
    replacement = dict(server._indexing_job)

    updated = server._update_indexing_job(
        "stale-worker",
        status="failed",
        phase="error",
        result={"error": "stale failure"},
    )

    assert updated is False
    assert server._indexing_job == replacement


def test_terminal_publication_serializes_with_replacement_claim() -> None:
    terminal_publication_started = threading.Event()
    allow_terminal_publication = threading.Event()
    replacement_claimed = threading.Event()

    class BlockingJob(dict):
        def update(self, *args, **kwargs) -> None:
            terminal_publication_started.set()
            assert allow_terminal_publication.wait(timeout=5)
            super().update(*args, **kwargs)

    server = CodeSearchServer.__new__(CodeSearchServer)
    server._indexing_job_lock = threading.RLock()
    active_job = BlockingJob(
        {
            "job_id": "old-worker",
            "status": "indexing",
            "phase": "saving",
            "result": None,
        }
    )
    replacement_job = {
        "job_id": "replacement",
        "status": "indexing",
        "phase": "starting",
        "result": None,
    }
    server._indexing_job = active_job
    update_result = {}

    def publish_terminal_state() -> None:
        update_result["updated"] = server._update_indexing_job(
            "old-worker",
            status="completed",
            phase="done",
            result={"success": True},
        )

    def claim_replacement() -> None:
        with server._indexing_job_state_lock():
            server._indexing_job = replacement_job
        replacement_claimed.set()

    publisher = threading.Thread(target=publish_terminal_state)
    claimant = threading.Thread(target=claim_replacement)
    publisher.start()
    assert terminal_publication_started.wait(timeout=5)
    claimant.start()

    assert not replacement_claimed.wait(timeout=0.1)

    allow_terminal_publication.set()
    publisher.join(timeout=5)
    claimant.join(timeout=5)

    assert not publisher.is_alive()
    assert not claimant.is_alive()
    assert update_result == {"updated": True}
    assert active_job == {
        "job_id": "old-worker",
        "status": "completed",
        "phase": "done",
        "result": {"success": True},
    }
    assert server._indexing_job is replacement_job
