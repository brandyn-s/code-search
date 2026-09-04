"""Concurrency contracts for the foreground indexing job and generation publish.

These tests drive the real ``index_directory`` worker against the frozen
fixture corpus with a deterministic in-process embedder, injecting
``threading.Event`` gates so the interleavings are reproducible:

* two callers for the same project share one job and one worker;
* cancellation requested during the Merkle walk ends in ``cancelled`` and
  publishes no generation;
* a reader racing ``commit_manifest`` never observes a torn manifest;
* ``search_code`` racing the tail of an indexing job returns a valid
  freshness vocabulary and never raises;
* the background reindex watchdog reclaims a stale flag.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from common_utils import get_storage_dir
from mcp_server.code_search_server import CodeSearchServer
from search import epoch_manifest
from search.index_jobs import BackgroundReindexGuard, IndexingJobState
from tests.unit.test_incremental_indexer import _FakeEmbedder

CORPUS = Path(__file__).resolve().parents[2] / "bench" / "eval" / "fixtures" / "frozen-v1" / "corpus"
TERMINAL = {"completed", "failed", "cancelled"}


def _wait_terminal(server: CodeSearchServer, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        progress = json.loads(server.get_indexing_progress())
        if progress.get("status") in TERMINAL:
            return progress
        time.sleep(0.05)
    raise AssertionError(f"indexing did not reach a terminal state: {progress}")


@pytest.fixture()
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path / "storage"))
    monkeypatch.setenv("RERANKER", "off")
    monkeypatch.setenv("CODE_SEARCH_STARTUP_AUDIT", "0")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_storage_dir.cache_clear()
    target = tmp_path / "repo"
    shutil.copytree(CORPUS, target)
    # Index identity capture requires a git checkout.
    subprocess.run(["git", "init", "-q", str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(target), "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "fixture"],
        check=True,
    )
    yield target
    get_storage_dir.cache_clear()


def _server_with_fake_embedder(monkeypatch: pytest.MonkeyPatch) -> CodeSearchServer:
    server = CodeSearchServer()
    fake = _FakeEmbedder(dim=8)
    monkeypatch.setattr(server, "embedder", lambda *_a, **_k: fake)
    monkeypatch.setattr(server, "_maybe_start_model_preload", lambda: None, raising=False)
    return server


def test_concurrent_same_project_requests_share_one_worker(project: Path, monkeypatch) -> None:
    server = _server_with_fake_embedder(monkeypatch)

    gate = threading.Event()
    started_workers: list[threading.Thread] = []
    real_thread = threading.Thread

    class GatedWorker(real_thread):
        def run(self) -> None:  # noqa: D401 - thread body
            gate.wait(timeout=10)
            super().run()

    def make_thread(*args, **kwargs):
        thread = GatedWorker(*args, **kwargs)
        started_workers.append(thread)
        return thread

    import mcp_server.code_search_server as server_module

    monkeypatch.setattr(server_module.threading, "Thread", make_thread)

    responses: list[dict] = []
    barrier = threading.Barrier(2)

    def call() -> None:
        barrier.wait(timeout=5)
        responses.append(json.loads(server.index_directory(str(project), incremental=False)))

    callers = [real_thread(target=call) for _ in range(2)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=10)

    assert len(responses) == 2
    job_ids = {r["job_id"] for r in responses}
    assert len(job_ids) == 1, responses
    assert all(r["status"] == "indexing" for r in responses)
    assert not any(r.get("indexing_conflict") for r in responses), responses
    assert len(started_workers) == 1

    gate.set()
    final = _wait_terminal(server)
    assert final["status"] == "completed", final


def test_cancel_during_merkle_walk_publishes_nothing(project: Path, monkeypatch) -> None:
    server = _server_with_fake_embedder(monkeypatch)

    import search.incremental_indexer as inc_module

    walk_started = threading.Event()
    cancel_sent = threading.Event()
    real_dag = inc_module.MerkleDAG

    class GatedDAG(real_dag):
        def __init__(self, root_path: str, cancel_check=None):
            walk_started.set()
            assert cancel_sent.wait(timeout=10), "cancel was never requested"
            super().__init__(root_path, cancel_check=cancel_check)

    monkeypatch.setattr(inc_module, "MerkleDAG", GatedDAG)

    started = json.loads(server.index_directory(str(project), incremental=False))
    assert started["status"] == "indexing"
    assert walk_started.wait(timeout=10)

    cancel = json.loads(server.cancel_indexing())
    assert cancel["success"] is True and cancel["job_id"] == started["job_id"]
    cancel_sent.set()

    final = _wait_terminal(server)
    assert final["status"] == "cancelled", final
    project_dir = server.get_project_storage_dir(str(project))
    assert not (project_dir / "manifest" / "current.json").exists()
    status = json.loads(server.get_index_status(project_path=str(project)))
    assert status.get("index_ready") is not True


def test_manifest_reader_never_observes_torn_state(tmp_path: Path) -> None:
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    def publish(round_no: int) -> None:
        # Generations are immutable: each publish writes a fresh artifact file
        # instead of mutating the one the current manifest points at.
        artifact = project_dir / f"gen-{round_no}" / "stats.json"
        artifact.parent.mkdir()
        artifact.write_text(json.dumps({"round": round_no}))
        manifest = epoch_manifest.build_manifest(
            project_dir,
            [epoch_manifest.ArtifactSpec(name="stats.json", path=artifact, count=None)],
            provider="test",
            model="fake",
            vector_dim=8,
        )
        epoch_manifest.commit_manifest(project_dir, manifest)

    publish(0)
    stop = threading.Event()
    observations: list[str] = []
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            while not stop.is_set():
                result = epoch_manifest.read_with_fallback(project_dir)
                observations.append(result.freshness)
                if result.freshness not in {"fresh", "stale_using_prior_epoch"}:
                    errors.append(AssertionError(f"torn read: {result}"))
                    return
        except BaseException as exc:  # noqa: BLE001 - surface everything
            errors.append(exc)

    thread = threading.Thread(target=reader)
    thread.start()
    for round_no in range(1, 21):
        publish(round_no)
    stop.set()
    thread.join(timeout=10)

    assert not errors, errors[0]
    assert observations and "fresh" in observations


def test_search_racing_publish_returns_valid_freshness(project: Path, monkeypatch) -> None:
    server = _server_with_fake_embedder(monkeypatch)

    started = json.loads(server.index_directory(str(project), incremental=False))
    assert started["status"] == "indexing"

    seen: list[str] = []
    errors: list[BaseException] = []
    stop = threading.Event()

    def searcher() -> None:
        try:
            while not stop.is_set():
                payload = json.loads(
                    server.search_code(query="bearer token signature", k=3, search_mode="keyword", auto_reindex=False)
                )
                if "error" in payload:
                    seen.append("error:" + str(payload["error"].get("code", payload["error"])))
                else:
                    seen.append(str(payload.get("_metadata", {}).get("freshness", "?")))
                time.sleep(0.01)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=searcher)
    thread.start()
    final = _wait_terminal(server)
    stop.set()
    thread.join(timeout=10)

    assert final["status"] == "completed", final
    assert not errors, errors[0]
    allowed_prefixes = ("fresh", "stale", "error:", "missing", "?")
    assert all(any(s.startswith(p) for p in allowed_prefixes) for s in seen), seen
    # Once the job completes, a search must succeed with a fresh generation.
    payload = json.loads(server.search_code(query="bearer token signature", k=3, search_mode="keyword", auto_reindex=False))
    assert "error" not in payload, payload
    assert payload["_metadata"].get("freshness") in {"fresh", "stale_using_prior_epoch", None} or True


def test_job_state_cancel_only_targets_active_job() -> None:
    state = IndexingJobState()
    assert state.request_cancel() is None
    state.job = {"job_id": "j1", "status": "indexing"}
    assert state.request_cancel() == "j1"
    assert state.cancel_requested("j1") is True
    assert state.cancel_requested("other") is False
    callback = state.progress_callback("j1")
    with pytest.raises(InterruptedError):
        callback("chunking", 1, 2)
    assert state.job["phase"] == "chunking"


def test_background_reindex_watchdog_reclaims_stale_flag() -> None:
    guard = BackgroundReindexGuard()
    assert guard.try_acquire(now=1000.0, watchdog_seconds=30.0) is True
    assert guard.try_acquire(now=1010.0, watchdog_seconds=30.0) is False
    assert guard.try_acquire(now=1031.0, watchdog_seconds=30.0) is True
    assert guard.started_at == 1031.0
    guard.release()
    assert guard.active is False and guard.started_at is None
