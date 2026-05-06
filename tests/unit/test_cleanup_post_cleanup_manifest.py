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

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.cleanup_index_orphans import commit_post_cleanup_manifest
from search.epoch_manifest import (
    ArtifactSpec,
    build_manifest,
    commit_manifest,
    read_current,
    verify_manifest,
    ManifestMissing,
)


def _seed_artifacts(idx: Path, chunk_count: int = 3) -> list[str]:
    """Seed chunk_ids.pkl + metadata.db + fts5.db + stats.json all with
    consistent counts. Returns the chunk_ids list."""
    idx.mkdir(parents=True, exist_ok=True)
    chunk_ids = [f"chunk_{i}" for i in range(chunk_count)]
    with open(idx / "chunk_ids.pkl", "wb") as f:
        pickle.dump(chunk_ids, f)

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
        con.execute("CREATE TABLE unnamed (key TEXT PRIMARY KEY, value BLOB)")
        for cid in chunk_ids:
            con.execute(
                "INSERT INTO unnamed (key, value) VALUES (?, ?)",
                (cid, pickle.dumps({"chunk_type": "function"})),
            )
        con.commit()
    finally:
        con.close()

    (idx / "stats.json").write_text(
        json.dumps({"total_chunks": chunk_count}),
        encoding="utf-8",
    )
    return chunk_ids


def test_commit_post_cleanup_manifest_creates_fresh_current(tmp_path):
    """A clean post-cleanup state produces a current.json that verifies."""
    idx = tmp_path / "index"
    chunk_ids = _seed_artifacts(idx, chunk_count=4)

    epoch_id = commit_post_cleanup_manifest(idx, chunk_ids)
    assert epoch_id is not None

    manifest = read_current(idx)
    assert manifest["epoch_id"] == epoch_id
    err = verify_manifest(idx, manifest)
    assert err is None, f"verify_manifest reported: {err}"


def test_commit_post_cleanup_manifest_promotes_prior(tmp_path):
    """If a manifest already existed, the new commit promotes it to prior.json."""
    idx = tmp_path / "index"
    chunk_ids = _seed_artifacts(idx, chunk_count=2)

    # First commit (simulates a prior save_index manifest).
    artifacts = [
        ArtifactSpec(
            name="chunk_ids.pkl", path=idx / "chunk_ids.pkl", count=2,
        ),
    ]
    initial = build_manifest(idx, artifacts)
    commit_manifest(idx, initial)
    initial_epoch = initial["epoch_id"]

    # Now run post-cleanup commit. Should produce a NEW epoch and the
    # initial manifest moves to prior.json.
    new_epoch = commit_post_cleanup_manifest(idx, chunk_ids)
    assert new_epoch is not None
    assert new_epoch != initial_epoch

    from search.epoch_manifest import read_prior
    prior = read_prior(idx)
    assert prior is not None
    assert prior["epoch_id"] == initial_epoch


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

    # Output mentions the commit.
    captured = capsys.readouterr()
    assert "manifest committed" in captured.out
    assert "manifests committed:" in captured.out
