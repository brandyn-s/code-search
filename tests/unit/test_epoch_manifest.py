"""Tests for the epoch-manifest primitive (Plan-2 E1).

Covers the crash-recovery scenarios documented in
docs/epoch_manifest_design.md. Each scenario tests that the
on-disk state stays consistent — at no point can a reader
see a partially-committed epoch.
"""
from __future__ import annotations

import pickle
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from search.epoch_manifest import (  # noqa: E402
    ArtifactSpec,
    ManifestConsistencyError,
    ManifestMissing,
    build_manifest,
    cleanup_stale_candidate,
    commit_manifest,
    count_fts5_db,
    count_metadata_db,
    read_current,
    read_prior,
    read_with_fallback,
    verify_manifest,
)


# ─── helpers ───

def _seed_artifacts(idx_dir: Path, chunk_count: int = 5) -> list[ArtifactSpec]:
    """Seed a project's index/ dir with consistent artifacts."""
    idx_dir.mkdir(parents=True, exist_ok=True)
    chunk_ids = [f"chunk_{i}" for i in range(chunk_count)]

    # chunk_ids.pkl
    pkl = idx_dir / "chunk_ids.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(chunk_ids, f)

    # metadata.db with chunk_count rows
    meta = idx_dir / "metadata.db"
    con = sqlite3.connect(str(meta))
    try:
        con.execute("CREATE TABLE unnamed (key TEXT PRIMARY KEY, value BLOB)")
        for cid in chunk_ids:
            con.execute("INSERT INTO unnamed VALUES (?, ?)", (cid, b"v"))
        con.commit()
    finally:
        con.close()

    # fts5.db with chunk_count rows
    fts = idx_dir / "fts5.db"
    con = sqlite3.connect(str(fts))
    try:
        con.execute("CREATE VIRTUAL TABLE chunk_fts USING fts5(chunk_id, content)")
        for cid in chunk_ids:
            con.execute("INSERT INTO chunk_fts VALUES (?, ?)", (cid, "x"))
        con.commit()
    finally:
        con.close()

    return [
        ArtifactSpec("chunk_ids.pkl", pkl, count=chunk_count),
        ArtifactSpec("metadata.db", meta, count=count_metadata_db(meta)),
        ArtifactSpec("fts5.db", fts, count=count_fts5_db(fts)),
    ]


# ─── happy path ───

def test_clean_write_read_roundtrip(tmp_path):
    proj = tmp_path / "proj"
    artifacts = _seed_artifacts(proj / "index", chunk_count=7)

    manifest = build_manifest(
        proj, artifacts,
        provider="voyage", model="voyage-4-large",
        vector_dim=1024, quantization="int8",
        pipeline_version="abc123",
    )
    committed = commit_manifest(proj, manifest)
    assert committed.exists()
    assert committed.name == "current.json"
    # Candidate is gone after commit
    assert not (proj / "manifest" / "candidate.json").exists()
    # No prior on first commit
    assert read_prior(proj) is None

    loaded = read_current(proj)
    assert loaded["epoch_id"] == manifest["epoch_id"]
    assert loaded["provider"] == "voyage"
    assert loaded["consistency"]["all_artifacts_share_count"] is True
    assert loaded["consistency"]["expected_count"] == 7

    # Verification succeeds against the live files
    err = verify_manifest(proj, loaded)
    assert err is None, f"verify_manifest unexpectedly failed: {err}"


def test_second_commit_promotes_prior(tmp_path):
    proj = tmp_path / "proj"
    artifacts = _seed_artifacts(proj / "generation-1", chunk_count=3)
    m1 = build_manifest(proj, artifacts)
    commit_manifest(proj, m1)

    # Immutable generation artifacts keep the current epoch verifiable while
    # the next candidate is prepared.
    artifacts2 = _seed_artifacts(proj / "generation-2", chunk_count=5)
    m2 = build_manifest(proj, artifacts2)
    commit_manifest(proj, m2)

    current = read_current(proj)
    prior = read_prior(proj)
    assert current["epoch_id"] == m2["epoch_id"]
    assert prior is not None
    assert prior["epoch_id"] == m1["epoch_id"]
    assert verify_manifest(proj, prior) is None


# ─── consistency check ───

def test_consistency_error_on_mismatched_counts(tmp_path):
    """Artifact counts disagreeing must abort BEFORE commit."""
    proj = tmp_path / "proj"
    idx = proj / "index"
    artifacts = _seed_artifacts(idx, chunk_count=5)
    # Manually corrupt: drop one fts5 row
    fts = idx / "fts5.db"
    con = sqlite3.connect(str(fts))
    con.execute("DELETE FROM chunk_fts WHERE chunk_id = ?", ("chunk_4",))
    con.commit()
    con.close()
    artifacts[2] = ArtifactSpec("fts5.db", fts, count=count_fts5_db(fts))

    with pytest.raises(ManifestConsistencyError) as exc:
        build_manifest(proj, artifacts)
    assert "Record-count mismatch" in str(exc.value)
    # No manifest dir was created
    assert not (proj / "manifest").exists()


def test_consistency_error_includes_per_artifact_counts(tmp_path):
    """The error message names which artifacts disagree, for debugging."""
    proj = tmp_path / "proj"
    idx = proj / "index"
    artifacts = _seed_artifacts(idx, chunk_count=5)
    fts = idx / "fts5.db"
    con = sqlite3.connect(str(fts))
    con.execute("DELETE FROM chunk_fts WHERE chunk_id = ?", ("chunk_4",))
    con.commit()
    con.close()
    artifacts[2] = ArtifactSpec("fts5.db", fts, count=count_fts5_db(fts))

    with pytest.raises(ManifestConsistencyError) as exc:
        build_manifest(proj, artifacts)
    msg = str(exc.value)
    # Expect both counts surfaced in some form
    assert "5" in msg and "4" in msg


# ─── crash recovery ───

def test_crash_during_candidate_write_no_corruption(tmp_path):
    """A write that crashes mid-step-4 (partial candidate.json) does not
    corrupt the existing committed state. The next clean write succeeds."""
    proj = tmp_path / "proj"
    artifacts = _seed_artifacts(proj / "index", chunk_count=4)
    m1 = build_manifest(proj, artifacts)
    commit_manifest(proj, m1)
    first_epoch = read_current(proj)["epoch_id"]

    # Simulate crash: write a partial candidate.json directly
    (proj / "manifest" / "candidate.json").write_text("{partial", encoding="utf-8")
    assert (proj / "manifest" / "candidate.json").exists()

    # Reader still sees the original committed state (no half-commit)
    assert read_current(proj)["epoch_id"] == first_epoch

    # Cleanup is idempotent and removes the stale candidate
    assert cleanup_stale_candidate(proj) is True
    assert not (proj / "manifest" / "candidate.json").exists()
    # Calling again on clean state: no-op
    assert cleanup_stale_candidate(proj) is False

    # Next clean commit succeeds despite the prior crash
    m2 = build_manifest(proj, artifacts)
    commit_manifest(proj, m2)
    assert read_current(proj)["epoch_id"] == m2["epoch_id"]


def test_crash_after_prior_promote_before_current_rename(tmp_path):
    """Simulate crash AFTER current → prior rename but BEFORE candidate →
    current rename. State: prior.json exists with old epoch, current.json
    is absent, candidate.json is present. Reader sees ManifestMissing;
    E3 will use prior.json instead. Cleanup is to remove the candidate."""
    proj = tmp_path / "proj"
    artifacts = _seed_artifacts(proj / "index", chunk_count=4)
    m1 = build_manifest(proj, artifacts)
    commit_manifest(proj, m1)

    # Simulate crash by patching os.replace to fail on the second call
    # (which is the candidate -> current rename, AFTER prior was promoted)
    real_replace = __import__("os").replace
    call_count = {"n": 0}

    def fail_on_second(src, dst):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise OSError("simulated crash mid-commit")
        return real_replace(src, dst)

    artifacts2 = _seed_artifacts(proj / "index2", chunk_count=4)
    m2 = build_manifest(proj, artifacts2)
    with patch("os.replace", side_effect=fail_on_second):
        with pytest.raises(OSError, match="simulated crash"):
            commit_manifest(proj, m2)

    # After crash: prior.json holds m1, candidate.json has m2 written, current.json missing
    assert (proj / "manifest" / "prior.json").exists()
    assert (proj / "manifest" / "candidate.json").exists()
    assert not (proj / "manifest" / "current.json").exists()

    # Reader sees ManifestMissing (caller in E3 will fall back to prior)
    with pytest.raises(ManifestMissing):
        read_current(proj)

    # Operator can recover via prior + cleanup
    prior = read_prior(proj)
    assert prior is not None
    assert prior["epoch_id"] == m1["epoch_id"]
    cleanup_stale_candidate(proj)
    assert not (proj / "manifest" / "candidate.json").exists()


def test_failed_commit_does_not_replace_verified_prior_with_corrupt_current(
    tmp_path,
):
    """A corrupt current must not be rotated over the only valid fallback."""
    proj = tmp_path / "proj"
    first_artifacts = _seed_artifacts(proj / "generation-1", chunk_count=4)
    first = build_manifest(proj, first_artifacts)
    commit_manifest(proj, first)

    second_artifacts = _seed_artifacts(proj / "generation-2", chunk_count=4)
    second = build_manifest(proj, second_artifacts)
    commit_manifest(proj, second)
    (proj / "generation-2" / "chunk_ids.pkl").write_bytes(b"corrupt")
    fallback = read_with_fallback(proj)
    assert fallback.freshness == "stale_using_prior_epoch"
    assert fallback.manifest is not None
    assert fallback.manifest["epoch_id"] == first["epoch_id"]

    third_artifacts = _seed_artifacts(proj / "generation-3", chunk_count=4)
    third = build_manifest(proj, third_artifacts)
    real_replace = __import__("os").replace

    def fail_candidate_promotion(src, dst):
        if Path(src).name == "candidate.json" and Path(dst).name == "current.json":
            raise OSError("simulated candidate promotion failure")
        return real_replace(src, dst)

    with patch(
        "search.epoch_manifest.os.replace",
        side_effect=fail_candidate_promotion,
    ):
        with pytest.raises(OSError, match="candidate promotion failure"):
            commit_manifest(proj, third)

    recovered = read_with_fallback(proj)
    assert recovered.freshness == "stale_using_prior_epoch"
    assert recovered.manifest is not None
    assert recovered.manifest["epoch_id"] == first["epoch_id"]


# ─── verification ───

def test_verify_detects_artifact_corruption(tmp_path):
    """Modifying an artifact AFTER commit makes verify return a mismatch
    string (caller decides how to react — E3's reader will fall back)."""
    proj = tmp_path / "proj"
    artifacts = _seed_artifacts(proj / "index", chunk_count=3)
    m = build_manifest(proj, artifacts)
    commit_manifest(proj, m)

    # Corrupt the chunk_ids.pkl post-commit
    (proj / "index" / "chunk_ids.pkl").write_bytes(b"\x80corrupted")

    err = verify_manifest(proj, read_current(proj))
    assert err is not None
    assert "sha256 mismatch" in err
    assert "chunk_ids.pkl" in err


def test_verify_rejects_empty_manifest(tmp_path):
    err = verify_manifest(
        tmp_path,
        {"epoch_id": "empty", "artifacts": {}},
    )
    assert err == "manifest has no artifacts"


def test_verify_detects_missing_artifact(tmp_path):
    """If an artifact file is deleted, verify returns a missing-artifact
    string instead of raising."""
    proj = tmp_path / "proj"
    artifacts = _seed_artifacts(proj / "index", chunk_count=3)
    m = build_manifest(proj, artifacts)
    commit_manifest(proj, m)

    (proj / "index" / "fts5.db").unlink()
    err = verify_manifest(proj, read_current(proj))
    assert err is not None
    assert "artifact missing" in err
    assert "fts5.db" in err


def test_verify_passes_on_clean_state(tmp_path):
    proj = tmp_path / "proj"
    artifacts = _seed_artifacts(proj / "index", chunk_count=3)
    m = build_manifest(proj, artifacts)
    commit_manifest(proj, m)
    assert verify_manifest(proj, read_current(proj)) is None


# ─── manifest content shape ───

def test_manifest_contains_all_required_fields(tmp_path):
    proj = tmp_path / "proj"
    artifacts = _seed_artifacts(proj / "index", chunk_count=3)
    m = build_manifest(
        proj, artifacts,
        provider="voyage", model="voyage-4-large",
        vector_dim=1024, quantization="int8",
        pipeline_version="abc123",
    )
    for key in ("version", "epoch_id", "created_at", "provider", "model",
                "vector_dim", "quantization", "pipeline_version", "artifacts",
                "consistency"):
        assert key in m, f"missing key: {key}"
    assert m["version"] == 1
    # epoch_id is sortable timestamp-prefixed
    assert m["epoch_id"][:4].isdigit()


def test_manifest_artifact_paths_are_relative(tmp_path):
    """Manifest stores paths as project-relative POSIX strings, not absolute
    Windows paths. Critical for cross-machine portability and JSON round-trip."""
    proj = tmp_path / "proj"
    artifacts = _seed_artifacts(proj / "index", chunk_count=2)
    m = build_manifest(proj, artifacts)
    for entry in m["artifacts"].values():
        assert not entry["path"].startswith(("/", "C:", "c:")), (
            f"artifact path is absolute: {entry['path']}"
        )
        assert "\\" not in entry["path"], (
            f"artifact path contains backslash: {entry['path']}"
        )


def test_read_current_raises_when_missing(tmp_path):
    proj = tmp_path / "proj"
    (proj / "manifest").mkdir(parents=True)
    with pytest.raises(ManifestMissing):
        read_current(proj)


def test_read_prior_returns_none_when_no_prior(tmp_path):
    proj = tmp_path / "proj"
    artifacts = _seed_artifacts(proj / "index", chunk_count=2)
    m = build_manifest(proj, artifacts)
    commit_manifest(proj, m)
    # First commit: no prior yet
    assert read_prior(proj) is None
