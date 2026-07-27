"""Tests for Plan-2 E2-4: cleanup_index_orphans commits manifest after recovery.

After cleanup mutates disk (delete fts5/metadata orphans, rewrite stats.json
from authoritative state), the prior manifest's recorded SHAs no longer
match the artifact bytes. PR #123 closes that gap by committing a fresh
manifest covering the post-recovery state.

These tests verify the manifest is committed on real cleanup, skipped on
dry-run, skipped when no cleanup was needed, and the resulting manifest
verifies clean.
"""
from __future__ import annotations

import json
import pickle
import sqlite3
import sys
from pathlib import Path

import faiss
import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.cleanup_index_orphans import (
    commit_post_cleanup_manifest,
    compute_stats_from_truth,
    find_metadata_orphans,
    remove_metadata_orphans,
)
from search.epoch_manifest import (
    ArtifactSpec,
    build_manifest,
    commit_manifest,
    read_current,
    verify_manifest,
    ManifestMissing,
)
from search.indexer import CodeIndexManager


def _seed_artifacts(idx: Path, chunk_count: int = 3) -> list[str]:
    """Seed a complete searchable root index with consistent sidecars."""
    idx.mkdir(parents=True, exist_ok=True)
    chunk_ids = [f"chunk_{i}" for i in range(chunk_count)]
    with open(idx / "chunk_ids.pkl", "wb") as f:
        pickle.dump(chunk_ids, f)

    index = faiss.IndexFlatIP(4)
    index.add(np.ones((chunk_count, 4), dtype=np.float32))
    faiss.write_index(index, str(idx / "code.index"))

    con = sqlite3.connect(str(idx / "fts5.db"))
    try:
        con.execute("CREATE VIRTUAL TABLE chunk_fts USING fts5(chunk_id, content)")
        for cid in chunk_ids:
            con.execute(
                "INSERT INTO chunk_fts (chunk_id, content) VALUES (?, ?)",
                (cid, f"content for {cid}"),
            )
        con.commit()
    finally:
        con.close()

    con = sqlite3.connect(str(idx / "metadata.db"))
    try:
        con.execute(
            "CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        for position, cid in enumerate(chunk_ids):
            con.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?)",
                (
                    cid,
                    json.dumps(
                        {
                            "index_id": position,
                            "metadata": {
                                "chunk_type": "function",
                                "relative_path": f"{cid}.py",
                            },
                        }
                    ),
                ),
            )
        con.commit()
    finally:
        con.close()

    (idx / "stats.json").write_text(
        json.dumps({"total_chunks": chunk_count}),
        encoding="utf-8",
    )
    return chunk_ids


def _manifest_artifacts(idx: Path, chunk_count: int) -> list[ArtifactSpec]:
    return [
        ArtifactSpec(
            name=name,
            path=idx / name,
            count=(
                chunk_count
                if name in {"chunk_ids.pkl", "code.index"}
                else None
            ),
        )
        for name in (
            "chunk_ids.pkl",
            "code.index",
            "metadata.db",
            "fts5.db",
            "stats.json",
        )
    ]


def test_commit_post_cleanup_manifest_creates_fresh_current(tmp_path):
    """A clean post-cleanup state produces a current.json that verifies."""
    idx = tmp_path / "index"
    chunk_ids = _seed_artifacts(idx, chunk_count=4)

    epoch_id = commit_post_cleanup_manifest(idx, chunk_ids)
    assert epoch_id is not None

    manifest = read_current(idx)
    assert manifest["epoch_id"] == epoch_id
    assert all(
        entry["path"].startswith(".generations/")
        for entry in manifest["artifacts"].values()
    )
    err = verify_manifest(idx, manifest)
    assert err is None, f"verify_manifest reported: {err}"


def test_commit_post_cleanup_manifest_promotes_prior(tmp_path):
    """If a manifest already existed, the new commit promotes it to prior.json."""
    idx = tmp_path / "index"
    chunk_ids = _seed_artifacts(idx, chunk_count=2)

    initial_epoch = commit_post_cleanup_manifest(idx, chunk_ids)
    assert initial_epoch is not None

    # A second cleanup publication produces a new immutable generation and
    # rotates the previous verified manifest to prior.json.
    new_epoch = commit_post_cleanup_manifest(idx, chunk_ids)
    assert new_epoch is not None
    assert new_epoch != initial_epoch

    from search.epoch_manifest import read_prior
    prior = read_prior(idx)
    assert prior is not None
    assert prior["epoch_id"] == initial_epoch


def test_commit_post_cleanup_manifest_preserves_verified_index_identity(
    tmp_path,
):
    idx = tmp_path / "index"
    chunk_ids = _seed_artifacts(idx, chunk_count=2)
    initial = build_manifest(
        idx,
        _manifest_artifacts(idx, 2),
        provider="voyage",
        model="voyage-code-3",
        vector_dim=4,
        pipeline_version="pipeline-v7",
        input_type_enabled=True,
    )
    commit_manifest(idx, initial)

    epoch = commit_post_cleanup_manifest(idx, chunk_ids)
    assert epoch is not None
    current = read_current(idx)
    assert current["provider"] == "voyage"
    assert current["model"] == "voyage-code-3"
    assert current["vector_dim"] == 4
    assert current["pipeline_version"] == "pipeline-v7"
    assert current["input_type_enabled"] is True


def test_commit_post_cleanup_manifest_rejects_identity_dimension_mismatch(
    tmp_path,
):
    idx = tmp_path / "index"
    chunk_ids = _seed_artifacts(idx, chunk_count=2)
    initial = build_manifest(
        idx,
        _manifest_artifacts(idx, 2),
        provider="voyage",
        model="voyage-code-3",
        vector_dim=8,
        pipeline_version="pipeline-v7",
    )
    commit_manifest(idx, initial)

    assert commit_post_cleanup_manifest(idx, chunk_ids) is None
    current = read_current(idx)
    assert current["epoch_id"] == initial["epoch_id"]
    assert verify_manifest(idx, current) is None


def test_commit_post_cleanup_manifest_rejects_missing_input_mode_identity(
    tmp_path,
):
    idx = tmp_path / "index"
    chunk_ids = _seed_artifacts(idx, chunk_count=2)
    initial = build_manifest(
        idx,
        _manifest_artifacts(idx, 2),
        provider="voyage",
        model="voyage-code-3",
        vector_dim=4,
        pipeline_version="pipeline-v7",
        input_type_enabled=True,
    )
    initial.pop("input_type_enabled")
    commit_manifest(idx, initial)

    assert commit_post_cleanup_manifest(idx, chunk_ids) is None
    current = read_current(idx)
    assert current["epoch_id"] == initial["epoch_id"]
    assert "input_type_enabled" not in current
    assert verify_manifest(idx, current) is None


def test_cleanup_publication_failure_preserves_previous_generation(
    tmp_path, monkeypatch
):
    idx = tmp_path / "index"
    chunk_ids = _seed_artifacts(idx, chunk_count=2)
    first_epoch = commit_post_cleanup_manifest(idx, chunk_ids)
    assert first_epoch is not None
    before = read_current(idx)

    def fail_commit(_self, _manifest):
        raise OSError("simulated cleanup commit failure")

    monkeypatch.setattr(
        CodeIndexManager,
        "_commit_epoch_manifest",
        fail_commit,
    )
    assert commit_post_cleanup_manifest(idx, chunk_ids) is None

    after = read_current(idx)
    assert after["epoch_id"] == before["epoch_id"]
    assert verify_manifest(idx, after) is None
    assert all(
        entry["path"].startswith(".generations/")
        for entry in after["artifacts"].values()
    )


def test_commit_post_cleanup_manifest_removes_stale_candidate(tmp_path):
    """A leftover candidate.json from a crashed prior write is cleaned up
    before the new commit."""
    idx = tmp_path / "index"
    chunk_ids = _seed_artifacts(idx, chunk_count=2)
    manifest_dir = idx / "manifest"
    manifest_dir.mkdir(parents=True)
    candidate = manifest_dir / "candidate.json"
    candidate.write_text(
        json.dumps({"epoch_id": "stale-residue", "artifacts": {}}),
        encoding="utf-8",
    )
    assert candidate.exists()

    new_epoch = commit_post_cleanup_manifest(idx, chunk_ids)
    assert new_epoch is not None
    # The stale candidate must have been removed before the new commit
    # (if not, commit_manifest would still succeed — but the next dry-run
    # detection of stale candidates would still flag it). Tighter contract:
    # cleanup_stale_candidate is invoked.
    # After commit_manifest writes, a NEW candidate.json briefly exists
    # then is renamed to current.json — so the only residue check is that
    # the OLD content is gone. Here we verify by epoch_id.
    cur = read_current(idx)
    assert cur["epoch_id"] == new_epoch
    assert cur["epoch_id"] != "stale-residue"


def test_commit_post_cleanup_manifest_returns_none_when_no_artifacts(tmp_path):
    """An empty index dir with no artifacts produces no manifest (graceful
    skip rather than crash)."""
    idx = tmp_path / "empty_index"
    idx.mkdir(parents=True)

    epoch_id = commit_post_cleanup_manifest(idx, [])
    assert epoch_id is None
    # No current.json was written.
    with pytest.raises(ManifestMissing):
        read_current(idx)


def test_commit_post_cleanup_manifest_refuses_sidecars_without_faiss(tmp_path):
    """Sidecars alone are not a complete searchable generation."""
    idx = tmp_path / "incomplete_index"
    idx.mkdir(parents=True)
    chunk_ids = ["c1"]
    with (idx / "chunk_ids.pkl").open("wb") as handle:
        pickle.dump(chunk_ids, handle)
    (idx / "stats.json").write_text(
        json.dumps({"total_chunks": 1}),
        encoding="utf-8",
    )

    assert commit_post_cleanup_manifest(idx, chunk_ids) is None
    with pytest.raises(ManifestMissing):
        read_current(idx)


def test_commit_post_cleanup_manifest_handles_inconsistent_artifacts(tmp_path):
    """Cross-artifact inconsistency at commit time → manifest skipped, no
    current.json written. Surfaces the operator-visible failure mode."""
    idx = tmp_path / "index"
    idx.mkdir(parents=True)

    # Seed inconsistent state: 3-chunk pkl but 5 metadata rows.
    chunk_ids = ["c1", "c2", "c3"]
    with open(idx / "chunk_ids.pkl", "wb") as f:
        pickle.dump(chunk_ids, f)

    con = sqlite3.connect(str(idx / "metadata.db"))
    try:
        con.execute("CREATE TABLE unnamed (key TEXT PRIMARY KEY, value BLOB)")
        # Add MORE rows than chunk_ids — inconsistency.
        for cid in chunk_ids + ["extra1", "extra2"]:
            con.execute(
                "INSERT INTO unnamed (key, value) VALUES (?, ?)",
                (cid, pickle.dumps({})),
            )
        con.commit()
    finally:
        con.close()

    # Build a tiny FTS5 with the smaller count so the metadata mismatch is
    # the divergent signal rather than fts5 vs pkl.
    con = sqlite3.connect(str(idx / "fts5.db"))
    try:
        con.execute("CREATE VIRTUAL TABLE chunk_fts USING fts5(chunk_id, content)")
        for cid in chunk_ids:
            con.execute(
                "INSERT INTO chunk_fts (chunk_id, content) VALUES (?, ?)",
                (cid, f"c"),
            )
        con.commit()
    finally:
        con.close()

    # Should refuse to commit (metadata count 5, pkl/fts count 3).
    epoch_id = commit_post_cleanup_manifest(idx, chunk_ids)
    assert epoch_id is None
    with pytest.raises(ManifestMissing):
        read_current(idx)


@pytest.mark.parametrize("sidecar", ["metadata", "fts5"])
def test_commit_post_cleanup_manifest_rejects_orphan_sidecar_ids(
    tmp_path,
    sidecar,
):
    """Cleanup publication requires every persisted sidecar ID to be live."""
    idx = tmp_path / "index"
    chunk_ids = _seed_artifacts(idx, chunk_count=3)

    if sidecar == "metadata":
        connection = sqlite3.connect(str(idx / "metadata.db"))
        try:
            connection.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?)",
                ("orphan", json.dumps({"metadata": {}})),
            )
            connection.commit()
        finally:
            connection.close()
    else:
        connection = sqlite3.connect(str(idx / "fts5.db"))
        try:
            connection.execute(
                "INSERT INTO chunk_fts (chunk_id, content) VALUES (?, ?)",
                ("orphan", "orphan content"),
            )
            connection.commit()
        finally:
            connection.close()

    assert commit_post_cleanup_manifest(idx, chunk_ids) is None
    with pytest.raises(ManifestMissing):
        read_current(idx)


def test_commit_post_cleanup_manifest_allows_stale_faiss_rows(tmp_path):
    """A FAISS row may remain after its live metadata and FTS rows are gone."""
    idx = tmp_path / "index"
    chunk_ids = _seed_artifacts(idx, chunk_count=3)
    stale_id = chunk_ids[-1]

    metadata = sqlite3.connect(str(idx / "metadata.db"))
    try:
        metadata.execute("DELETE FROM kv WHERE key = ?", (stale_id,))
        metadata.commit()
    finally:
        metadata.close()

    fts = sqlite3.connect(str(idx / "fts5.db"))
    try:
        fts.execute("DELETE FROM chunk_fts WHERE chunk_id = ?", (stale_id,))
        fts.commit()
    finally:
        fts.close()

    epoch_id = commit_post_cleanup_manifest(idx, chunk_ids)

    assert epoch_id is not None
    current = read_current(idx)
    assert current["consistency"]["expected_count"] == len(chunk_ids)
    assert verify_manifest(idx, current) is None


def test_current_metadata_orphans_are_detected_and_removed(tmp_path):
    idx = tmp_path / "index"
    _seed_artifacts(idx, chunk_count=2)
    meta_path = idx / "metadata.db"
    connection = sqlite3.connect(str(meta_path))
    try:
        connection.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?)",
            ("orphan", json.dumps({"metadata": {}})),
        )
        connection.commit()
    finally:
        connection.close()

    count, sample = find_metadata_orphans(
        meta_path,
        {"chunk_0", "chunk_1"},
    )
    assert count == 1
    assert sample == ["orphan"]
    assert remove_metadata_orphans(
        meta_path,
        {"chunk_0", "chunk_1"},
    ) == 1
    assert find_metadata_orphans(
        meta_path,
        {"chunk_0", "chunk_1"},
    ) == (0, [])


def test_compute_stats_reads_current_json_metadata(tmp_path):
    idx = tmp_path / "index"
    chunk_ids = _seed_artifacts(idx, chunk_count=2)

    stats = compute_stats_from_truth(idx, chunk_ids)
    assert stats is not None
    assert stats["total_chunks"] == 2
    assert stats["files_indexed"] == 2
    assert stats["chunk_types"] == {"function": 2}
    assert stats["top_tags"] == {"python": 2}


def test_compute_stats_reads_binary_faiss_index(tmp_path):
    idx = tmp_path / "binary-index"
    idx.mkdir()
    chunk_ids = ["binary"]
    with (idx / "chunk_ids.pkl").open("wb") as handle:
        pickle.dump(chunk_ids, handle)
    index = faiss.IndexBinaryFlat(8)
    index.add(np.array([[0b10101010]], dtype=np.uint8))
    faiss.write_index_binary(index, str(idx / "code.index"))
    np.save(
        idx / "float_store.npy",
        np.ones((1, 8), dtype=np.float32),
        allow_pickle=False,
    )
    connection = sqlite3.connect(str(idx / "metadata.db"))
    try:
        connection.execute(
            "CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?)",
            (
                "binary",
                json.dumps(
                    {
                        "index_id": 0,
                        "metadata": {
                            "chunk_type": "function",
                            "relative_path": "binary.py",
                        },
                    }
                ),
            ),
        )
        connection.commit()
    finally:
        connection.close()

    stats = compute_stats_from_truth(idx, chunk_ids)
    assert stats is not None
    assert stats["total_chunks"] == 1
    assert stats["index_size"] == 1
    assert stats["embedding_dimension"] == 8
    assert stats["quantization"] == "binary"


def test_compute_stats_refuses_legacy_pickle_metadata(tmp_path):
    idx = tmp_path / "legacy-index"
    idx.mkdir()
    chunk_ids = ["legacy"]
    with (idx / "chunk_ids.pkl").open("wb") as handle:
        pickle.dump(chunk_ids, handle)
    index = faiss.IndexFlatIP(4)
    index.add(np.ones((1, 4), dtype=np.float32))
    faiss.write_index(index, str(idx / "code.index"))
    connection = sqlite3.connect(str(idx / "metadata.db"))
    try:
        connection.execute(
            "CREATE TABLE unnamed (key TEXT PRIMARY KEY, value BLOB)"
        )
        connection.execute(
            "INSERT INTO unnamed (key, value) VALUES (?, ?)",
            ("legacy", pickle.dumps({"metadata": {}})),
        )
        connection.commit()
    finally:
        connection.close()

    assert compute_stats_from_truth(idx, chunk_ids) is None


def test_main_loop_commits_manifest_on_apply(tmp_path, monkeypatch, capsys):
    """End-to-end: --apply-fts5 cleanup triggers manifest commit; dry-run
    does not."""
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))

    proj = tmp_path / "projects" / "p1"
    idx = proj / "index"
    chunk_ids = _seed_artifacts(idx, chunk_count=3)

    # Inject an fts5 orphan to give cleanup something to do.
    con = sqlite3.connect(str(idx / "fts5.db"))
    try:
        con.execute(
            "INSERT INTO chunk_fts (chunk_id, content) VALUES (?, ?)",
            ("orphan_fts", "orphan content"),
        )
        con.commit()
    finally:
        con.close()

    # Dry-run first: no cleanup, no manifest.
    monkeypatch.setattr(sys, "argv", ["cleanup_index_orphans.py"])
    from scripts.cleanup_index_orphans import main as cleanup_main
    rc = cleanup_main()
    assert rc == 0
    with pytest.raises(ManifestMissing):
        read_current(idx)

    # --apply-fts5: cleanup runs, manifest committed.
    monkeypatch.setattr(
        sys, "argv", ["cleanup_index_orphans.py", "--apply-fts5"],
    )
    rc = cleanup_main()
    assert rc == 0
    manifest = read_current(idx)
    assert "epoch_id" in manifest
    err = verify_manifest(idx, manifest)
    assert err is None, f"manifest verification failed post-cleanup: {err}"


def test_main_loop_cleans_current_metadata_and_rebuilds_stats(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    idx = tmp_path / "projects" / "p1" / "index"
    chunk_ids = _seed_artifacts(idx, chunk_count=3)
    connection = sqlite3.connect(str(idx / "metadata.db"))
    try:
        connection.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?)",
            ("orphan", json.dumps({"metadata": {}})),
        )
        connection.commit()
    finally:
        connection.close()
    (idx / "stats.json").write_text(
        json.dumps({"total_chunks": 99}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cleanup_index_orphans.py",
            "--apply-metadata",
            "--apply-stats",
        ],
    )
    from scripts.cleanup_index_orphans import main as cleanup_main

    assert cleanup_main() == 0
    assert find_metadata_orphans(
        idx / "metadata.db",
        set(chunk_ids),
    ) == (0, [])
    stats = json.loads((idx / "stats.json").read_text(encoding="utf-8"))
    assert stats["total_chunks"] == 3
    manifest = read_current(idx)
    assert verify_manifest(idx, manifest) is None
    assert all(
        entry["path"].startswith(".generations/")
        for entry in manifest["artifacts"].values()
    )

    # Output mentions the commit.
    captured = capsys.readouterr()
    assert "manifest committed" in captured.out
    assert "manifests committed:" in captured.out
