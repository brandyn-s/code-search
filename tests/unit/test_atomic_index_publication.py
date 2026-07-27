"""Crash-consistency regressions for ``CodeIndexManager.save_index``."""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import faiss
import numpy as np
import pytest

from embeddings.embedder import EmbeddingResult
from search.epoch_manifest import (
    ArtifactSpec,
    ManifestConsistencyError,
    build_manifest,
    commit_manifest,
    read_current,
    read_with_fallback,
    verify_manifest,
)
from search.indexer import CodeIndexManager, IndexPublicationRefused


def _embedding(chunk_id: str) -> EmbeddingResult:
    return EmbeddingResult(
        embedding=np.ones(16, dtype=np.float32),
        chunk_id=chunk_id,
        metadata={
            "file_path": "test.py",
            "relative_path": "test.py",
            "content_preview": chunk_id,
            "full_content": chunk_id,
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 1,
            "name": chunk_id,
            "parent_name": None,
            "docstring": None,
            "decorators": [],
            "imports": [],
            "complexity_score": 1,
            "tags": [],
            "folder_structure": [],
        },
    )


def _close(manager: CodeIndexManager) -> None:
    if manager._metadata_db is not None:
        manager._metadata_db.close()
        manager._metadata_db = None
    if manager._fts_conn is not None:
        manager._fts_conn.close()
        manager._fts_conn = None


def test_failed_faiss_writes_preserve_committed_generation(
    tmp_path, monkeypatch
):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("first")])
    manager.save_index()

    committed = read_current(tmp_path)
    committed_index = manager.index_path.read_bytes()
    with manager.chunk_id_path.open("rb") as handle:
        committed_chunk_ids = pickle.load(handle)

    manager.add_embeddings([_embedding("second")])
    write_attempts = 0

    def fail_write(_index, _path):
        nonlocal write_attempts
        write_attempts += 1
        raise OSError("simulated disk failure")

    monkeypatch.setattr(faiss, "write_index", fail_write)
    monkeypatch.setattr(faiss, "index_gpu_to_cpu", lambda index: index)

    with pytest.raises(OSError, match="simulated disk failure"):
        manager.save_index()

    assert write_attempts == 2
    assert manager.index_path.read_bytes() == committed_index
    with manager.chunk_id_path.open("rb") as handle:
        assert pickle.load(handle) == committed_chunk_ids
    assert read_current(tmp_path)["epoch_id"] == committed["epoch_id"]
    assert (
        read_current(tmp_path)["consistency"]["expected_count"]
        == len(committed_chunk_ids)
    )

    _close(manager)
    manifest_result = read_with_fallback(tmp_path)
    assert manifest_result.freshness == "fresh"
    assert manifest_result.manifest is not None
    assert verify_manifest(tmp_path, manifest_result.manifest) is None

    reloaded = CodeIndexManager(str(tmp_path))
    assert reloaded.index.ntotal == len(committed_chunk_ids)
    assert reloaded._chunk_ids == committed_chunk_ids
    assert len(reloaded.metadata_db) == len(committed_chunk_ids)
    assert reloaded.search_bm25("first", k=10)[0][0] == "first"
    assert reloaded.search_bm25("second", k=10) == []
    assert reloaded.get_stats()["total_chunks"] == len(committed_chunk_ids)
    _close(reloaded)


def test_cpu_faiss_write_retry_does_not_require_gpu_conversion(
    tmp_path, monkeypatch
):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("uncommitted")])
    assert manager._on_gpu is False
    write_attempts = 0

    def fail_write(_index, _path):
        nonlocal write_attempts
        write_attempts += 1
        raise OSError("simulated CPU write failure")

    def reject_gpu_conversion(_index):
        raise AssertionError("CPU retry attempted GPU conversion")

    monkeypatch.setattr(faiss, "write_index", fail_write)
    monkeypatch.setattr(
        faiss, "index_gpu_to_cpu", reject_gpu_conversion
    )

    with pytest.raises(OSError, match="CPU write failure"):
        manager.save_index()

    assert write_attempts == 2
    _close(manager)


def test_startup_recovers_interrupted_mirror_promotion(tmp_path, monkeypatch):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("first")])
    manager.save_index()
    committed = read_current(tmp_path)

    manager.add_embeddings([_embedding("second")])

    def interrupt_before_manifest(_manifest):
        raise KeyboardInterrupt("simulated process termination")

    monkeypatch.setattr(
        manager, "_commit_epoch_manifest", interrupt_before_manifest
    )
    with pytest.raises(KeyboardInterrupt, match="process termination"):
        manager.save_index()
    _close(manager)

    recovered = CodeIndexManager(str(tmp_path))
    assert read_current(tmp_path)["epoch_id"] == committed["epoch_id"]
    assert recovered.index.ntotal == 1
    assert recovered._chunk_ids == ["first"]
    assert len(recovered.metadata_db) == 1
    assert recovered.search_bm25("first", k=10)[0][0] == "first"
    assert recovered.search_bm25("second", k=10) == []
    assert not (tmp_path / ".publication-in-progress").exists()
    _close(recovered)


def test_startup_recovers_interruption_during_candidate_staging(
    tmp_path, monkeypatch
):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("first")])
    manager.save_index()
    manager.add_embeddings([_embedding("second")])

    def interrupt_during_staging(_candidate_dir):
        raise KeyboardInterrupt("simulated candidate-stage interruption")

    monkeypatch.setattr(
        manager,
        "_write_candidate_generation",
        interrupt_during_staging,
    )
    with pytest.raises(KeyboardInterrupt, match="candidate-stage"):
        manager.save_index()

    assert manager._publication_marker.exists()
    _close(manager)

    recovered = CodeIndexManager(str(tmp_path))
    assert recovered.index.ntotal == 1
    assert recovered._chunk_ids == ["first"]
    assert len(recovered.metadata_db) == 1
    assert recovered.search_bm25("first", k=10)[0][0] == "first"
    assert recovered.search_bm25("second", k=10) == []
    assert not recovered._publication_marker.exists()
    _close(recovered)


def test_first_publication_interruption_restarts_clean(
    tmp_path, monkeypatch
):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("never-committed")])

    def interrupt_during_staging(_candidate_dir):
        raise KeyboardInterrupt("simulated first-stage interruption")

    monkeypatch.setattr(
        manager,
        "_write_candidate_generation",
        interrupt_during_staging,
    )
    with pytest.raises(KeyboardInterrupt, match="first-stage"):
        manager.save_index()
    assert manager._publication_marker.exists()
    _close(manager)

    recovered = CodeIndexManager(str(tmp_path))
    assert recovered.index is None
    assert recovered._chunk_ids == []
    assert len(recovered.metadata_db) == 0
    assert recovered.search_bm25("never-committed", k=10) == []
    assert read_with_fallback(tmp_path).freshness == "missing"
    assert not recovered._publication_marker.exists()
    assert not list(recovered._generation_root.glob("*"))
    _close(recovered)


def test_unsaved_addition_restarts_from_committed_generation(tmp_path):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("first")])
    manager.save_index()
    committed_epoch = read_current(tmp_path)["epoch_id"]

    manager.add_embeddings([_embedding("second")])
    assert len(manager.metadata_db) == 2
    assert manager.search_bm25("second", k=10)[0][0] == "second"
    assert manager._publication_marker.exists()
    _close(manager)

    recovered = CodeIndexManager(str(tmp_path))
    assert read_current(tmp_path)["epoch_id"] == committed_epoch
    assert recovered.index.ntotal == 1
    assert recovered._chunk_ids == ["first"]
    assert len(recovered.metadata_db) == 1
    assert recovered.search_bm25("first", k=10)[0][0] == "first"
    assert recovered.search_bm25("second", k=10) == []
    assert not recovered._publication_marker.exists()
    _close(recovered)


def test_recovery_discards_crash_surviving_sqlite_wal_sidecars(tmp_path):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("first")])
    manager.save_index()
    manager.add_embeddings([_embedding("second")])

    sidecar_bytes = {}
    wal = Path(f"{manager.metadata_path}-wal")
    shm = Path(f"{manager.metadata_path}-shm")
    assert wal.exists()
    sidecar_bytes[wal] = wal.read_bytes()
    if shm.exists():
        sidecar_bytes[shm] = shm.read_bytes()

    # A clean close normally checkpoints/removes WAL state. Restore the
    # captured sidecars to model the bytes left behind by process death.
    _close(manager)
    for sidecar, payload in sidecar_bytes.items():
        sidecar.write_bytes(payload)

    recovered = CodeIndexManager(str(tmp_path))
    assert recovered.index.ntotal == 1
    assert recovered._chunk_ids == ["first"]
    assert len(recovered.metadata_db) == 1
    assert recovered.metadata_db.get("second") is None
    assert recovered.search_bm25("second", k=10) == []
    assert recovered._fts_conn.execute(
        "SELECT COUNT(*) FROM chunk_fts WHERE chunk_id = ?",
        ("second",),
    ).fetchone()[0] == 0
    _close(recovered)


def test_unsaved_removal_restarts_from_committed_generation(tmp_path):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings(
        [_embedding("first"), _embedding("second")]
    )
    manager.save_index()
    committed_epoch = read_current(tmp_path)["epoch_id"]

    assert manager.remove_file_chunks("test.py") == 2
    assert len(manager.metadata_db) == 0
    assert manager.search_bm25("first", k=10) == []
    assert manager._publication_marker.exists()
    _close(manager)

    recovered = CodeIndexManager(str(tmp_path))
    assert read_current(tmp_path)["epoch_id"] == committed_epoch
    assert recovered.index.ntotal == 2
    assert recovered._chunk_ids == ["first", "second"]
    assert len(recovered.metadata_db) == 2
    assert recovered.search_bm25("first", k=10)[0][0] == "first"
    assert recovered.search_bm25("second", k=10)[0][0] == "second"
    assert not recovered._publication_marker.exists()
    _close(recovered)


def test_unmanifested_legacy_index_is_preserved_across_unsaved_add(
    tmp_path,
):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("legacy-first")])
    manager.save_index()
    _close(manager)
    shutil.rmtree(tmp_path / "manifest")
    shutil.rmtree(tmp_path / ".generations")

    legacy = CodeIndexManager(str(tmp_path))
    assert read_with_fallback(tmp_path).freshness == "missing"
    legacy.add_embeddings([_embedding("uncommitted-second")])
    assert read_with_fallback(tmp_path).freshness == "fresh"
    assert legacy._publication_marker.exists()
    _close(legacy)

    recovered = CodeIndexManager(str(tmp_path))
    assert recovered.index.ntotal == 1
    assert recovered._chunk_ids == ["legacy-first"]
    assert len(recovered.metadata_db) == 1
    assert recovered.search_bm25("legacy-first", k=10)[0][0] == (
        "legacy-first"
    )
    assert recovered.search_bm25("uncommitted-second", k=10) == []
    assert not recovered._publication_marker.exists()
    _close(recovered)


def test_unmanifested_legacy_index_survives_direct_save_failure(
    tmp_path, monkeypatch
):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("legacy-first")])
    manager.save_index()
    _close(manager)
    shutil.rmtree(tmp_path / "manifest")
    shutil.rmtree(tmp_path / ".generations")

    legacy = CodeIndexManager(str(tmp_path))
    assert legacy.index.ntotal == 1

    def fail_write(_index, _path):
        raise OSError("simulated legacy save failure")

    monkeypatch.setattr(faiss, "write_index", fail_write)
    with pytest.raises(OSError, match="legacy save failure"):
        legacy.save_index()
    _close(legacy)

    assert read_with_fallback(tmp_path).freshness == "missing"
    assert (tmp_path / "code.index").exists()
    assert (tmp_path / "chunk_ids.pkl").exists()

    recovered = CodeIndexManager(str(tmp_path))
    assert recovered.index.ntotal == 1
    assert recovered._chunk_ids == ["legacy-first"]
    assert recovered.search_bm25("legacy-first", k=10)[0][0] == (
        "legacy-first"
    )
    _close(recovered)


def test_unmanifested_legacy_truncation_refusal_preserves_roots(tmp_path):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings(
        [_embedding(f"legacy-{position}") for position in range(6)]
    )
    manager.save_index()
    _close(manager)
    shutil.rmtree(tmp_path / "manifest")
    shutil.rmtree(tmp_path / ".generations")

    legacy = CodeIndexManager(str(tmp_path))
    assert legacy.index.ntotal == 6
    legacy._chunk_ids = ["truncated-in-memory"]

    with pytest.raises(IndexPublicationRefused, match="truncation guard"):
        legacy.save_index()
    _close(legacy)

    assert read_with_fallback(tmp_path).freshness == "missing"
    assert (tmp_path / "code.index").exists()
    assert (tmp_path / "chunk_ids.pkl").exists()

    recovered = CodeIndexManager(str(tmp_path))
    assert recovered.index.ntotal == 6
    assert recovered._chunk_ids == [
        f"legacy-{position}" for position in range(6)
    ]
    assert len(recovered.metadata_db) == 6
    _close(recovered)


def test_startup_recovery_prunes_crash_orphan_generation(tmp_path):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("committed")])
    manager.save_index()
    committed = read_current(tmp_path)
    committed_index = (
        tmp_path / committed["artifacts"]["code.index"]["path"]
    )
    committed_generation = committed_index.parent
    orphan_generation = manager._generation_root / "crash-orphan"
    shutil.copytree(committed_generation, orphan_generation)
    manager._write_publication_marker({})
    _close(manager)

    recovered = CodeIndexManager(str(tmp_path))
    assert recovered.index.ntotal == 1
    assert recovered._chunk_ids == ["committed"]
    assert not recovered._publication_marker.exists()
    assert not orphan_generation.exists()
    _close(recovered)


def test_startup_upgrades_legacy_manifest_to_immutable_generation(tmp_path):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("first")])
    manager.save_index()
    _close(manager)

    shutil.rmtree(tmp_path / ".generations")
    shutil.rmtree(tmp_path / "manifest")
    legacy_artifacts = []
    for name in (
        "chunk_ids.pkl",
        "code.index",
        "metadata.db",
        "fts5.db",
        "stats.json",
    ):
        path = tmp_path / name
        legacy_artifacts.append(
            ArtifactSpec(
                name=name,
                path=path,
                count=1 if name in {"chunk_ids.pkl", "code.index"} else None,
            )
        )
    legacy_manifest = build_manifest(tmp_path, legacy_artifacts)
    commit_manifest(tmp_path, legacy_manifest)
    assert read_current(tmp_path)["artifacts"]["code.index"]["path"] == (
        "code.index"
    )
    orphan_generation = tmp_path / ".generations" / "legacy-orphan"
    orphan_generation.mkdir(parents=True)
    (orphan_generation / "orphan").write_text("crash debris")

    upgraded = CodeIndexManager(str(tmp_path))
    current = read_current(tmp_path)
    assert current["epoch_id"] == legacy_manifest["epoch_id"]
    assert all(
        entry["path"].startswith(".generations/")
        for entry in current["artifacts"].values()
    )
    assert verify_manifest(tmp_path, current) is None
    assert not orphan_generation.exists()
    assert upgraded.index.ntotal == 1
    assert upgraded.search_bm25("first", k=10)[0][0] == "first"
    _close(upgraded)


def test_legacy_upgrade_fsyncs_generation_parent_before_promotion(
    tmp_path, monkeypatch
):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("legacy")])
    manager.save_index()
    _close(manager)

    shutil.rmtree(tmp_path / ".generations")
    shutil.rmtree(tmp_path / "manifest")
    legacy_artifacts = []
    for name in (
        "chunk_ids.pkl",
        "code.index",
        "metadata.db",
        "fts5.db",
        "stats.json",
    ):
        legacy_artifacts.append(
            ArtifactSpec(
                name=name,
                path=tmp_path / name,
                count=1 if name in {"chunk_ids.pkl", "code.index"} else None,
            )
        )
    commit_manifest(
        tmp_path, build_manifest(tmp_path, legacy_artifacts)
    )

    events = []
    original_fsync_directory = CodeIndexManager._fsync_directory
    original_replace = os.replace

    def record_fsync(path):
        events.append(("fsync", Path(path)))
        original_fsync_directory(path)

    def record_replace(src, dst):
        source = Path(src)
        destination = Path(dst)
        if (
            source.parent == tmp_path / ".generations"
            and source.name.startswith(".legacy-")
        ):
            events.append(("promote", destination))
        return original_replace(src, dst)

    monkeypatch.setattr(
        CodeIndexManager,
        "_fsync_directory",
        staticmethod(record_fsync),
    )
    monkeypatch.setattr(os, "replace", record_replace)

    upgraded = CodeIndexManager(str(tmp_path))
    parent_fsync = events.index(("fsync", tmp_path))
    promotion = next(
        position
        for position, event in enumerate(events)
        if event[0] == "promote"
    )
    assert parent_fsync < promotion
    _close(upgraded)


def test_first_publication_failure_leaves_clean_empty_state(
    tmp_path, monkeypatch
):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("uncommitted")])

    def fail_write(_index, _path):
        raise OSError("simulated first-save failure")

    monkeypatch.setattr(faiss, "write_index", fail_write)
    monkeypatch.setattr(faiss, "index_gpu_to_cpu", lambda index: index)
    with pytest.raises(OSError, match="first-save failure"):
        manager.save_index()
    _close(manager)

    assert read_with_fallback(tmp_path).freshness == "missing"
    assert not (tmp_path / ".publication-in-progress").exists()
    assert not list((tmp_path / ".generations").glob("*"))

    reloaded = CodeIndexManager(str(tmp_path))
    assert reloaded.index is None
    assert len(reloaded.metadata_db) == 0
    assert reloaded.search_bm25("uncommitted", k=10) == []
    assert reloaded.get_stats()["total_chunks"] == 0
    _close(reloaded)


def test_generation_directory_is_durable_before_publication_marker(
    tmp_path, monkeypatch
):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("first")])
    manager.save_index()

    events = []
    monkeypatch.setattr(
        manager,
        "_fsync_directory",
        lambda path: events.append(("fsync_dir", path)),
        raising=False,
    )
    original_marker = manager._write_publication_marker

    def record_marker(manifest):
        events.append(
            (
                "publication_marker",
                len(manifest.get("artifacts", {})),
            )
        )
        original_marker(manifest)

    monkeypatch.setattr(manager, "_write_publication_marker", record_marker)
    manager.add_embeddings([_embedding("second")])
    manager.save_index()

    preparation_marker_position = events.index(("publication_marker", 0))
    generation_marker_positions = [
        position
        for position, event in enumerate(events)
        if event[0] == "publication_marker" and event[1] > 0
    ]
    candidate_fsync_positions = [
        position
        for position, event in enumerate(events)
        if event[0] == "fsync_dir"
        and event[1].parent == manager._generation_root
        and event[1].name.startswith(".candidate-")
    ]
    generation_root_positions = [
        position
        for position, event in enumerate(events)
        if event == ("fsync_dir", manager._generation_root)
    ]
    assert candidate_fsync_positions
    assert generation_root_positions
    assert generation_marker_positions
    assert preparation_marker_position < candidate_fsync_positions[-1]
    assert candidate_fsync_positions[-1] < generation_root_positions[-1]
    assert (
        generation_root_positions[-1]
        < generation_marker_positions[-1]
    )
    _close(manager)


def test_post_rename_failure_removes_unreferenced_generation_durably(
    tmp_path, monkeypatch
):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("first")])
    manager.save_index()
    committed_epoch = read_current(tmp_path)["epoch_id"]
    committed_generations = set(manager._generation_root.iterdir())
    manager.add_embeddings([_embedding("second")])

    events = []
    original_build = manager._build_generation_manifest
    build_attempts = 0

    def fail_after_candidate_rename(generation_dir):
        nonlocal build_attempts
        build_attempts += 1
        if build_attempts == 2:
            raise OSError("simulated post-rename failure")
        return original_build(generation_dir)

    original_rmtree = shutil.rmtree

    def record_rmtree(path, *args, **kwargs):
        result = original_rmtree(path, *args, **kwargs)
        events.append(("remove", path))
        return result

    original_fsync_directory = manager._fsync_directory

    def record_fsync_directory(path):
        events.append(("fsync_dir", path))
        original_fsync_directory(path)

    monkeypatch.setattr(
        manager, "_build_generation_manifest", fail_after_candidate_rename
    )
    monkeypatch.setattr(shutil, "rmtree", record_rmtree)
    monkeypatch.setattr(
        manager, "_fsync_directory", record_fsync_directory
    )

    with pytest.raises(OSError, match="post-rename failure"):
        manager.save_index()

    removed_positions = [
        position
        for position, event in enumerate(events)
        if event[0] == "remove"
        and event[1].parent == manager._generation_root
        and not event[1].name.startswith(".candidate-")
    ]
    assert removed_positions
    assert any(
        position > removed_positions[-1]
        and event == ("fsync_dir", manager._generation_root)
        for position, event in enumerate(events)
    )
    assert read_current(tmp_path)["epoch_id"] == committed_epoch
    assert set(manager._generation_root.iterdir()) == committed_generations
    assert not manager._publication_marker.exists()
    _close(manager)

    reloaded = CodeIndexManager(str(tmp_path))
    assert reloaded.index.ntotal == 1
    assert reloaded._chunk_ids == ["first"]
    assert reloaded.search_bm25("first", k=10)[0][0] == "first"
    assert reloaded.search_bm25("second", k=10) == []
    _close(reloaded)


def test_interruption_after_manifest_commit_retains_referenced_generation(
    tmp_path, monkeypatch
):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("committed")])
    original_commit = manager._commit_epoch_manifest

    def commit_then_interrupt(manifest):
        original_commit(manifest)
        raise KeyboardInterrupt("simulated interruption after commit")

    monkeypatch.setattr(
        manager, "_commit_epoch_manifest", commit_then_interrupt
    )
    with pytest.raises(KeyboardInterrupt, match="after commit"):
        manager.save_index()
    _close(manager)

    committed = read_current(tmp_path)
    assert verify_manifest(tmp_path, committed) is None
    committed_index = tmp_path / committed["artifacts"]["code.index"]["path"]
    assert committed_index.exists()
    assert (tmp_path / ".publication-in-progress").exists()

    reloaded = CodeIndexManager(str(tmp_path))
    assert reloaded.index.ntotal == 1
    assert reloaded._chunk_ids == ["committed"]
    assert reloaded.search_bm25("committed", k=10)[0][0] == "committed"
    assert not (tmp_path / ".publication-in-progress").exists()
    _close(reloaded)


def test_binary_generation_publishes_float_store_and_reloads(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("QUANTIZATION", "binary")
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings(
        [_embedding("binary-first"), _embedding("binary-second")]
    )
    expected_float_store = manager._float_store.copy()
    manager.save_index()

    committed = read_current(tmp_path)
    assert committed["quantization"] == "binary"
    assert "float_store.npy" in committed["artifacts"]
    assert verify_manifest(tmp_path, committed) is None
    float_store_path = (
        tmp_path / committed["artifacts"]["float_store.npy"]["path"]
    )
    assert np.array_equal(
        np.load(float_store_path, allow_pickle=False),
        expected_float_store,
    )
    _close(manager)

    reloaded = CodeIndexManager(str(tmp_path))
    assert reloaded.index.ntotal == 2
    assert reloaded._is_binary is True
    assert reloaded._chunk_ids == ["binary-first", "binary-second"]
    assert np.array_equal(reloaded._float_store, expected_float_store)
    assert reloaded.search_bm25("binary-first", k=10)[0][0] == "binary-first"
    _close(reloaded)


def test_persisted_faiss_count_mismatch_is_never_published(
    tmp_path, monkeypatch
):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("first")])
    manager.save_index()
    committed_epoch = read_current(tmp_path)["epoch_id"]
    committed_generations = set(manager._generation_root.iterdir())
    manager.add_embeddings([_embedding("second")])

    original_write = faiss.write_index

    def write_readable_index_with_wrong_count(_index, path):
        original_write(faiss.IndexFlatIP(16), path)

    monkeypatch.setattr(
        faiss, "write_index", write_readable_index_with_wrong_count
    )
    with pytest.raises(
        ManifestConsistencyError, match="Record-count mismatch"
    ):
        manager.save_index()

    assert manager.last_manifest_commit_status == "consistency_error"
    assert read_current(tmp_path)["epoch_id"] == committed_epoch
    assert set(manager._generation_root.iterdir()) == committed_generations
    assert not manager._publication_marker.exists()
    _close(manager)

    reloaded = CodeIndexManager(str(tmp_path))
    assert reloaded.index.ntotal == 1
    assert reloaded._chunk_ids == ["first"]
    assert reloaded.search_bm25("second", k=10) == []
    _close(reloaded)


def test_truncation_refusal_raises_after_restoring_commit(tmp_path):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings(
        [_embedding(f"committed-{index}") for index in range(6)]
    )
    manager.save_index()
    committed_epoch = read_current(tmp_path)["epoch_id"]

    manager.add_embeddings([_embedding("uncommitted")])
    manager._chunk_ids = ["uncommitted"]
    with pytest.raises(RuntimeError, match="refused.*truncat"):
        manager.save_index()

    assert manager.last_manifest_commit_status == "consistency_error"
    assert read_current(tmp_path)["epoch_id"] == committed_epoch
    assert not manager._publication_marker.exists()
    _close(manager)

    reloaded = CodeIndexManager(str(tmp_path))
    assert reloaded.index.ntotal == 6
    assert reloaded._chunk_ids == [
        f"committed-{index}" for index in range(6)
    ]
    assert reloaded.search_bm25("uncommitted", k=10) == []
    _close(reloaded)


def test_startup_refuses_partial_recovery_when_no_generation_verifies(
    tmp_path,
):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("committed")])
    manager.save_index()
    committed = read_current(tmp_path)
    manager._write_publication_marker(committed)
    _close(manager)

    generation_index = (
        tmp_path / committed["artifacts"]["code.index"]["path"]
    )
    generation_index.write_bytes(b"corrupt persisted faiss")

    with pytest.raises(
        RuntimeError,
        match="no verified committed generation",
    ):
        CodeIndexManager(str(tmp_path))

    assert (tmp_path / ".publication-in-progress").exists()
    assert read_with_fallback(tmp_path).freshness == "corrupt"


@pytest.mark.parametrize("current_state", ["missing", "malformed"])
def test_startup_recovers_prior_when_current_manifest_is_unreadable(
    tmp_path,
    current_state,
):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("first")])
    manager.save_index()
    first = read_current(tmp_path)
    manager.add_embeddings([_embedding("second")])
    manager.save_index()
    current = read_current(tmp_path)
    manager._write_publication_marker(current)
    _close(manager)

    current_path = tmp_path / "manifest" / "current.json"
    if current_state == "missing":
        current_path.unlink()
    else:
        current_path.write_text("{broken", encoding="utf-8")

    recovered = CodeIndexManager(str(tmp_path))

    assert recovered.index.ntotal == 1
    assert recovered._chunk_ids == ["first"]
    assert read_with_fallback(tmp_path).manifest["epoch_id"] == first["epoch_id"]
    assert not recovered._publication_marker.exists()
    _close(recovered)


def test_legacy_upgrade_interruption_retains_referenced_generation(
    tmp_path, monkeypatch
):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("legacy")])
    manager.save_index()
    _close(manager)

    shutil.rmtree(tmp_path / ".generations")
    shutil.rmtree(tmp_path / "manifest")
    legacy_artifacts = []
    for name in (
        "chunk_ids.pkl",
        "code.index",
        "metadata.db",
        "fts5.db",
        "stats.json",
    ):
        legacy_artifacts.append(
            ArtifactSpec(
                name=name,
                path=tmp_path / name,
                count=1 if name in {"chunk_ids.pkl", "code.index"} else None,
            )
        )
    commit_manifest(
        tmp_path, build_manifest(tmp_path, legacy_artifacts)
    )

    original_fsync_directory = CodeIndexManager._fsync_directory

    def interrupt_after_manifest_rename(path):
        original_fsync_directory(path)
        if path == tmp_path / "manifest":
            raise KeyboardInterrupt(
                "simulated legacy-upgrade interruption"
            )

    with monkeypatch.context() as patch:
        patch.setattr(
            CodeIndexManager,
            "_fsync_directory",
            staticmethod(interrupt_after_manifest_rename),
        )
        with pytest.raises(KeyboardInterrupt, match="legacy-upgrade"):
            CodeIndexManager(str(tmp_path))

    upgraded = read_current(tmp_path)
    assert all(
        entry["path"].startswith(".generations/")
        for entry in upgraded["artifacts"].values()
    )
    assert verify_manifest(tmp_path, upgraded) is None

    reloaded = CodeIndexManager(str(tmp_path))
    assert reloaded.index.ntotal == 1
    assert reloaded._chunk_ids == ["legacy"]
    assert reloaded.search_bm25("legacy", k=10)[0][0] == "legacy"
    _close(reloaded)


def test_clear_index_removes_committed_generation_state(tmp_path):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("to-clear")])
    manager.save_index()

    manager.clear_index()

    assert read_with_fallback(tmp_path).freshness == "missing"
    assert not (tmp_path / "manifest").exists()
    assert not manager._generation_root.exists()
    assert not manager._publication_marker.exists()
    assert not manager.index_path.exists()
    assert not manager.chunk_id_path.exists()
    assert not manager.metadata_path.exists()
    assert not manager.stats_path.exists()
    assert not (tmp_path / "float_store.npy").exists()
    assert manager.search_bm25("to-clear", k=10) == []
    _close(manager)

    reloaded = CodeIndexManager(str(tmp_path))
    assert reloaded.index is None
    assert reloaded._chunk_ids == []
    assert len(reloaded.metadata_db) == 0
    assert reloaded.search_bm25("to-clear", k=10) == []
    _close(reloaded)


def test_search_waits_while_publication_replaces_sqlite_handles(
    tmp_path, monkeypatch
):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("first")])
    manager.save_index()
    manager.add_embeddings([_embedding("second")])

    close_started = threading.Event()
    allow_publication = threading.Event()
    original_close = manager._close_storage_handles

    def pause_after_close():
        original_close()
        close_started.set()
        assert allow_publication.wait(timeout=5)

    monkeypatch.setattr(
        manager, "_close_storage_handles", pause_after_close
    )
    save_errors = []

    def publish():
        try:
            manager.save_index()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            save_errors.append(exc)

    save_thread = threading.Thread(target=publish)
    save_thread.start()
    assert close_started.wait(timeout=5)

    search_results = []
    search_errors = []

    def search():
        try:
            search_results.extend(manager.search_bm25("first", k=10))
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            search_errors.append(exc)

    search_thread = threading.Thread(target=search)
    search_thread.start()
    search_thread.join(timeout=0.1)
    search_was_blocked = search_thread.is_alive()

    allow_publication.set()
    save_thread.join(timeout=5)
    search_thread.join(timeout=5)
    assert search_was_blocked
    assert not save_thread.is_alive()
    assert not search_thread.is_alive()
    assert save_errors == []
    assert search_errors == []
    assert search_results[0][0] == "first"
    _close(manager)


def test_distinct_managers_serialize_publication_for_same_storage(
    tmp_path,
):
    base = CodeIndexManager(str(tmp_path))
    base.add_embeddings([_embedding("base")])
    base.save_index()
    _close(base)

    manager_a = CodeIndexManager(str(tmp_path))
    manager_b = CodeIndexManager(str(tmp_path))
    manager_a.add_embeddings([_embedding("only-a")])
    manager_b.add_embeddings([_embedding("only-b")])

    a_at_commit = threading.Event()
    release_a = threading.Event()
    b_at_commit = threading.Event()
    original_a_commit = manager_a._commit_epoch_manifest
    original_b_commit = manager_b._commit_epoch_manifest

    def pause_a_commit(manifest):
        a_at_commit.set()
        assert release_a.wait(timeout=5)
        return original_a_commit(manifest)

    def observe_b_commit(manifest):
        b_at_commit.set()
        return original_b_commit(manifest)

    manager_a._commit_epoch_manifest = pause_a_commit
    manager_b._commit_epoch_manifest = observe_b_commit
    errors: list[BaseException] = []

    def save(manager):
        try:
            manager.save_index()
        except BaseException as exc:  # noqa: BLE001 - report thread failure
            errors.append(exc)

    thread_a = threading.Thread(target=save, args=(manager_a,))
    thread_b = threading.Thread(target=save, args=(manager_b,))
    thread_a.start()
    assert a_at_commit.wait(timeout=5)
    thread_b.start()
    publication_overlapped = b_at_commit.wait(timeout=0.2)
    release_a.set()
    thread_a.join(timeout=5)
    thread_b.join(timeout=5)

    assert not thread_a.is_alive()
    assert not thread_b.is_alive()
    assert errors == []
    assert publication_overlapped is False
    current = read_current(tmp_path)
    assert verify_manifest(tmp_path, current) is None
    for entry in current["artifacts"].values():
        assert (tmp_path / entry["path"]).exists()
    assert not (tmp_path / ".publication-in-progress").exists()
    _close(manager_a)
    _close(manager_b)


def test_manager_construction_waits_for_whole_rebuild_transaction(
    tmp_path,
):
    base = CodeIndexManager(str(tmp_path))
    base.add_embeddings([_embedding("base")])
    base.save_index()
    _close(base)

    writer = CodeIndexManager(str(tmp_path))
    rebuild_started = threading.Event()
    allow_commit = threading.Event()
    constructor_finished = threading.Event()
    errors: list[BaseException] = []
    constructed: list[CodeIndexManager] = []

    def rebuild():
        try:
            with writer.publication_transaction():
                writer.begin_rebuild()
                rebuild_started.set()
                assert allow_commit.wait(timeout=5)
                writer.add_embeddings([_embedding("replacement")])
                writer.save_index()
        except BaseException as exc:  # noqa: BLE001 - report thread failure
            errors.append(exc)

    def construct_reader():
        try:
            constructed.append(CodeIndexManager(str(tmp_path)))
        except BaseException as exc:  # noqa: BLE001 - report thread failure
            errors.append(exc)
        finally:
            constructor_finished.set()

    rebuild_thread = threading.Thread(target=rebuild)
    rebuild_thread.start()
    assert rebuild_started.wait(timeout=5)
    constructor_thread = threading.Thread(target=construct_reader)
    constructor_thread.start()

    assert constructor_finished.wait(timeout=0.2) is False
    assert writer._publication_marker.exists()

    allow_commit.set()
    rebuild_thread.join(timeout=5)
    constructor_thread.join(timeout=5)

    assert not rebuild_thread.is_alive()
    assert not constructor_thread.is_alive()
    assert errors == []
    assert len(constructed) == 1
    reader = constructed[0]
    assert reader.search_bm25("replacement", k=10)[0][0] == "replacement"
    assert reader.search_bm25("base", k=10) == []
    assert not writer._publication_marker.exists()
    assert verify_manifest(tmp_path, read_current(tmp_path)) is None
    _close(writer)
    _close(reader)


def test_nested_publication_transaction_is_rejected_without_losing_outer_work(
    tmp_path,
):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("base")])
    manager.save_index()

    with manager.publication_transaction():
        manager.add_embeddings([_embedding("outer")])
        assert manager._publication_marker.exists()
        with pytest.raises(
            IndexPublicationRefused,
            match="Nested index publication transactions",
        ), manager.publication_transaction():
            pass
        assert manager._publication_marker.exists()
        manager.save_index()

    reloaded = CodeIndexManager(str(tmp_path))
    assert reloaded.index.ntotal == 2
    assert reloaded._chunk_ids == ["base", "outer"]
    assert reloaded.metadata_db.get("outer") is not None
    assert reloaded.search_bm25("outer", k=10)[0][0] == "outer"
    assert verify_manifest(tmp_path, read_current(tmp_path)) is None
    _close(manager)
    _close(reloaded)


@pytest.mark.parametrize(
    "artifact_name",
    [
        "code.index",
        "chunk_ids.pkl",
        "metadata.db",
        "fts5.db",
        "stats.json",
    ],
)
def test_publication_transaction_repairs_missing_root_mirror(
    tmp_path,
    artifact_name,
):
    original = CodeIndexManager(str(tmp_path))
    original.add_embeddings([_embedding("base")])
    original.save_index()
    manifest = read_current(tmp_path)
    generation_artifact = (
        tmp_path / manifest["artifacts"][artifact_name]["path"]
    )
    expected_bytes = generation_artifact.read_bytes()
    _close(original)

    root_artifact = tmp_path / artifact_name
    root_artifact.unlink()
    manager = CodeIndexManager(str(tmp_path))
    with manager.publication_transaction():
        assert root_artifact.read_bytes() == expected_bytes

    assert verify_manifest(tmp_path, read_current(tmp_path)) is None
    _close(manager)


@pytest.mark.skipif(
    not hasattr(os, "fork"),
    reason="requires POSIX fork semantics",
)
def test_forked_child_resets_inherited_storage_lock(tmp_path):
    manager = CodeIndexManager(str(tmp_path))
    start_read, start_write = os.pipe()
    result_read, result_write = os.pipe()
    is_child = False
    result = b"E"

    try:
        with manager.publication_transaction():
            child_pid = os.fork()
            if child_pid == 0:
                is_child = True
                os.close(start_write)
                os.close(result_read)
                os.read(start_read, 1)
                with manager.publication_transaction():
                    pass

                completed = threading.Event()

                def read_from_another_thread():
                    manager.get_stats()
                    completed.set()

                thread = threading.Thread(target=read_from_another_thread)
                thread.start()
                thread.join(timeout=1)
                result = b"1" if completed.is_set() else b"0"
    except BaseException:
        if not is_child:
            raise

    if is_child:
        os.write(result_write, result)
        os.close(result_write)
        os._exit(0)

    os.close(start_read)
    os.close(result_write)
    os.write(start_write, b"1")
    os.close(start_write)
    result = os.read(result_read, 1)
    os.close(result_read)
    _, wait_status = os.waitpid(child_pid, 0)

    assert os.waitstatus_to_exitcode(wait_status) == 0
    assert result == b"1"
    _close(manager)


@pytest.mark.skipif(
    not hasattr(os, "fork"),
    reason="requires POSIX fork semantics",
)
def test_forked_child_cannot_release_parent_writer_lock(tmp_path):
    manager = CodeIndexManager(str(tmp_path))
    child_unwound_read, child_unwound_write = os.pipe()
    acquired = tmp_path / "competitor-acquired"
    is_child = False
    competitor = None
    overlapped = False

    with manager.publication_transaction():
        child_pid = os.fork()
        if child_pid == 0:
            is_child = True
            os.close(child_unwound_read)
        else:
            os.close(child_unwound_write)
            assert os.read(child_unwound_read, 1) == b"1"
            os.close(child_unwound_read)

            environment = os.environ.copy()
            repository_root = str(Path(__file__).resolve().parents[2])
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(
                    None,
                    (repository_root, environment.get("PYTHONPATH", "")),
                )
            )
            competitor = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys\n"
                        "from pathlib import Path\n"
                        "from search.indexer import CodeIndexManager\n"
                        "CodeIndexManager(sys.argv[1])\n"
                        "Path(sys.argv[2]).write_text('acquired')\n"
                    ),
                    str(tmp_path),
                    str(acquired),
                ],
                cwd=repository_root,
                env=environment,
            )
            deadline = time.monotonic() + 0.5
            while (
                not acquired.exists()
                and competitor.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            overlapped = acquired.exists()

    if is_child:
        os.write(child_unwound_write, b"1")
        os.close(child_unwound_write)
        os._exit(0)

    assert competitor is not None
    stdout, stderr = competitor.communicate(timeout=5)
    _, wait_status = os.waitpid(child_pid, 0)
    assert os.waitstatus_to_exitcode(wait_status) == 0
    assert competitor.returncode == 0, (stdout, stderr)
    assert overlapped is False
    assert acquired.exists()
    _close(manager)


@pytest.mark.skipif(
    not hasattr(os, "fork"),
    reason="requires POSIX fork semantics",
)
def test_forked_child_unwind_does_not_finalize_parent_working_set(tmp_path):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings([_embedding("base")])
    manager.save_index()
    child_result_read, child_result_write = os.pipe()
    is_child = False
    child_result = b"E"
    marker_survived = False

    try:
        with manager.publication_transaction():
            manager.add_embeddings([_embedding("outer")])
            assert manager._publication_marker.exists()
            child_pid = os.fork()
            if child_pid == 0:
                is_child = True
                os.close(child_result_read)
            else:
                os.close(child_result_write)
                child_result = os.read(child_result_read, 1)
                os.close(child_result_read)
                marker_survived = manager._publication_marker.exists()
                if child_result == b"1" and marker_survived:
                    manager.save_index()
    except BaseException:
        if not is_child:
            raise
    else:
        if is_child:
            child_result = b"1"

    if is_child:
        os.write(child_result_write, child_result)
        os.close(child_result_write)
        os._exit(0)

    _, wait_status = os.waitpid(child_pid, 0)
    assert os.waitstatus_to_exitcode(wait_status) == 0
    assert child_result == b"1"
    assert marker_survived is True

    reloaded = CodeIndexManager(str(tmp_path))
    assert reloaded.index.ntotal == 2
    assert reloaded._chunk_ids == ["base", "outer"]
    assert reloaded.metadata_db.get("outer") is not None
    assert reloaded.search_bm25("outer", k=10)[0][0] == "outer"
    assert verify_manifest(tmp_path, read_current(tmp_path)) is None
    _close(manager)
    _close(reloaded)


@pytest.mark.parametrize(
    ("writer_b_operation", "preconstruct_writer_b"),
    [
        pytest.param("publish", False, id="constructor-and-publish"),
        pytest.param("publish", True, id="preconstructed-publish"),
        pytest.param("clear", True, id="preconstructed-clear"),
    ],
)
def test_process_writers_serialize_destructive_operations(
    tmp_path, writer_b_operation, preconstruct_writer_b
):
    """A second process must not mutate state during another publication."""
    base = CodeIndexManager(str(tmp_path))
    base.add_embeddings([_embedding("base")])
    base.save_index()
    _close(base)

    ready = tmp_path / "writer-a-at-prior-read"
    b_attempting = tmp_path / "writer-b-attempting-construction"
    b_constructed = tmp_path / "writer-b-constructed"
    b_done = tmp_path / "writer-b-done"
    observation = tmp_path / "writer-a-observation.json"

    writer_a = r"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from embeddings.embedder import EmbeddingResult
from search.indexer import CodeIndexManager
import search.epoch_manifest as epoch_manifest

storage, ready, attempting, constructed, done, observation = map(
    Path, sys.argv[1:]
)

def embedding(chunk_id):
    return EmbeddingResult(
        embedding=np.ones(16, dtype=np.float32),
        chunk_id=chunk_id,
        metadata={
            "file_path": "test.py",
            "relative_path": "test.py",
            "content_preview": chunk_id,
            "full_content": chunk_id,
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 1,
            "name": chunk_id,
            "parent_name": None,
            "docstring": None,
            "decorators": [],
            "imports": [],
            "complexity_score": 1,
            "tags": [],
            "folder_structure": [],
        },
    )

manager = CodeIndexManager(str(storage))
original_read_prior = epoch_manifest.read_prior

def pause_with_stale_prior(project_dir):
    prior = original_read_prior(project_dir)
    ready.write_text("ready", encoding="utf-8")
    attempt_deadline = time.monotonic() + 5
    while not attempting.exists() and time.monotonic() < attempt_deadline:
        time.sleep(0.01)
    if not attempting.exists():
        raise RuntimeError("writer B never attempted construction")
    overlap_deadline = time.monotonic() + 1
    while not done.exists() and time.monotonic() < overlap_deadline:
        time.sleep(0.01)
    observation.write_text(
        json.dumps(
            {
                "constructor_overlapped": constructed.exists(),
                "publication_overlapped": done.exists(),
            }
        ),
        encoding="utf-8",
    )
    return prior

epoch_manifest.read_prior = pause_with_stale_prior
with manager.publication_transaction():
    manager.add_embeddings([embedding("writer-a")])
    manager.save_index()
"""
    writer_b = r"""
import sys
import time
from pathlib import Path

import numpy as np

from embeddings.embedder import EmbeddingResult
from search.indexer import CodeIndexManager

storage, attempting, constructed, done = map(Path, sys.argv[1:5])
operation = sys.argv[5]
preconstruct = sys.argv[6] == "true"

def embedding(chunk_id):
    return EmbeddingResult(
        embedding=np.ones(16, dtype=np.float32),
        chunk_id=chunk_id,
        metadata={
            "file_path": "test.py",
            "relative_path": "test.py",
            "content_preview": chunk_id,
            "full_content": chunk_id,
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 1,
            "name": chunk_id,
            "parent_name": None,
            "docstring": None,
            "decorators": [],
            "imports": [],
            "complexity_score": 1,
            "tags": [],
            "folder_structure": [],
        },
    )

if preconstruct:
    manager = CodeIndexManager(str(storage))
    # Retain every mutable view before writer A replaces the root mirrors.
    # Writer B must refresh these handles only after it owns the filesystem
    # writer lock; construction-time state is stale by publication time.
    _ = manager.index.ntotal
    _ = len(manager.metadata_db)
    _ = manager.search_bm25("base", k=10)
    constructed.write_text("constructed", encoding="utf-8")
    deadline = time.monotonic() + 10
    while not attempting.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not attempting.exists():
        raise RuntimeError("writer A never signaled the operation")
else:
    attempting.write_text("attempting", encoding="utf-8")
    manager = CodeIndexManager(str(storage))
    constructed.write_text("constructed", encoding="utf-8")
if operation == "clear":
    manager.clear_index()
else:
    with manager.publication_transaction():
        manager.add_embeddings([embedding("writer-b")])
        manager.save_index()
done.write_text("done", encoding="utf-8")
"""

    environment = os.environ.copy()
    repository_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (repository_root, environment.get("PYTHONPATH", "")),
        )
    )
    writer_b_command = [
        sys.executable,
        "-c",
        writer_b,
        str(tmp_path),
        str(b_attempting),
        str(b_constructed),
        str(b_done),
        writer_b_operation,
        str(preconstruct_writer_b).lower(),
    ]

    def start_writer_b():
        return subprocess.Popen(
            writer_b_command,
            cwd=repository_root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    process_b = None
    if preconstruct_writer_b:
        process_b = start_writer_b()
        deadline = time.monotonic() + 10
        while (
            not b_constructed.exists()
            and process_b.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert b_constructed.exists(), process_b.communicate(timeout=5)

    process_a = subprocess.Popen(
        [
            sys.executable,
            "-c",
            writer_a,
            str(tmp_path),
            str(ready),
            str(b_attempting),
            str(b_constructed),
            str(b_done),
            str(observation),
        ],
        cwd=repository_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 10
    while (
        not ready.exists()
        and process_a.poll() is None
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert ready.exists(), process_a.communicate(timeout=5)

    if preconstruct_writer_b:
        b_attempting.write_text("attempting", encoding="utf-8")
    else:
        process_b = start_writer_b()
    assert process_b is not None

    stdout_a, stderr_a = process_a.communicate(timeout=20)
    stdout_b, stderr_b = process_b.communicate(timeout=20)
    assert process_a.returncode == 0, (stdout_a, stderr_a)
    assert process_b.returncode == 0, (stdout_b, stderr_b)

    assert json.loads(observation.read_text(encoding="utf-8")) == {
        "constructor_overlapped": preconstruct_writer_b,
        "publication_overlapped": False,
    }
    if writer_b_operation == "clear":
        assert read_with_fallback(tmp_path).freshness == "missing"
        assert not (tmp_path / "manifest").exists()
        assert not (tmp_path / ".generations").exists()
    else:
        current = read_current(tmp_path)
        assert verify_manifest(tmp_path, current) is None
        for entry in current["artifacts"].values():
            assert (tmp_path / entry["path"]).exists()
        reader = CodeIndexManager(str(tmp_path))
        assert reader.index.ntotal == 3
        assert set(reader._chunk_ids) == {
            "base",
            "writer-a",
            "writer-b",
        }
        assert set(reader.metadata_db.keys()) == {
            "base",
            "writer-a",
            "writer-b",
        }
        for chunk_id in ("base", "writer-a", "writer-b"):
            assert chunk_id in {
                result[0]
                for result in reader.search_bm25(chunk_id, k=10)
            }
        _close(reader)


def test_chunk_entry_snapshot_owns_metadata_handle_lock(tmp_path):
    manager = CodeIndexManager(str(tmp_path))
    manager.add_embeddings(
        [_embedding("first"), _embedding("second")]
    )

    entries = manager.get_chunk_entries()

    assert [chunk_id for chunk_id, _ in entries] == [
        "first",
        "second",
    ]
    assert entries[0][1]["metadata"]["name"] == "first"
    assert entries[1][1]["metadata"]["name"] == "second"
    _close(manager)
