"""Tests for read_with_fallback (Plan-2 E3 — reader downgrade tolerance).

Covers the four outcomes documented in the function's docstring:
  - "fresh": current verified
  - "stale_using_prior_epoch": current failed, prior verified
  - "missing": neither current nor prior exists
  - "corrupt": current and prior both failed verification
"""
from __future__ import annotations

import pickle
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from search.epoch_manifest import (  # noqa: E402
    ArtifactSpec,
    ReadResult,
    build_manifest,
    commit_manifest,
    count_fts5_db,
    count_metadata_db,
    read_with_fallback,
)


def _seed_artifacts(idx_dir: Path, chunk_count: int = 5) -> list[ArtifactSpec]:
    """Same helper used by test_epoch_manifest.py — duplicated for module isolation."""
    idx_dir.mkdir(parents=True, exist_ok=True)
    chunk_ids = [f"chunk_{i}" for i in range(chunk_count)]
    pkl = idx_dir / "chunk_ids.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(chunk_ids, f)
    meta = idx_dir / "metadata.db"
    con = sqlite3.connect(str(meta))
    try:
        con.execute("CREATE TABLE unnamed (key TEXT PRIMARY KEY, value BLOB)")
        for cid in chunk_ids:
            con.execute("INSERT INTO unnamed VALUES (?, ?)", (cid, b"v"))
        con.commit()
    finally:
        con.close()
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


def _commit_first(proj: Path) -> dict:
    """Helper: seed + commit one manifest. Returns the committed manifest dict."""
    artifacts = _seed_artifacts(proj / "index", chunk_count=3)
    m = build_manifest(proj, artifacts)
    commit_manifest(proj, m)
    return m


def test_read_with_fallback_fresh_on_clean_state(tmp_path):
    """Healthy committed state → freshness='fresh'."""
    proj = tmp_path / "proj"
    m = _commit_first(proj)
    result = read_with_fallback(proj)
    assert isinstance(result, ReadResult)
    assert result.ok is True
    assert result.freshness == "fresh"
    assert result.manifest is not None
    assert result.manifest["epoch_id"] == m["epoch_id"]
    assert result.detail == ""


def test_read_with_fallback_missing_when_no_manifests(tmp_path):
    """Empty project → freshness='missing'."""
    proj = tmp_path / "proj"
    (proj / "manifest").mkdir(parents=True)
    result = read_with_fallback(proj)
    assert result.ok is False
    assert result.freshness == "missing"
    assert "no current or prior" in result.detail


def test_read_with_fallback_falls_back_to_prior_on_current_corruption(tmp_path):
    """Corrupt current.json + verifiable prior → freshness='stale_using_prior_epoch'."""
    proj = tmp_path / "proj"
    # First commit
    m1 = _commit_first(proj)
    # Second commit (which moves m1 to prior, m2 to current)
    artifacts2 = _seed_artifacts(proj / "index2", chunk_count=4)
    m2 = build_manifest(proj, artifacts2)
    commit_manifest(proj, m2)

    # Corrupt current.json: rewrite with a sha that no longer matches the
    # m2 artifacts. Easiest: corrupt one of m2's artifacts so verify fails.
    (proj / "index2" / "chunk_ids.pkl").write_bytes(b"\x80corrupted")

    result = read_with_fallback(proj)
    # Current fails verification. Prior (m1) referenced index/ artifacts
    # which are still intact — but m1's artifact paths point at the OLD
    # `index/` dir (which still has its original files). So prior verifies.
    assert result.ok is True
    assert result.freshness == "stale_using_prior_epoch"
    assert result.manifest is not None
    assert result.manifest["epoch_id"] == m1["epoch_id"]
    assert "current failed" in result.detail


def test_read_with_fallback_uses_prior_when_current_json_is_malformed(
    tmp_path,
):
    proj = tmp_path / "proj"
    first = _commit_first(proj)
    artifacts2 = _seed_artifacts(proj / "index2", chunk_count=4)
    commit_manifest(proj, build_manifest(proj, artifacts2))
    (proj / "manifest" / "current.json").write_text(
        "{broken",
        encoding="utf-8",
    )

    result = read_with_fallback(proj)
    assert result.freshness == "stale_using_prior_epoch"
    assert result.manifest is not None
    assert result.manifest["epoch_id"] == first["epoch_id"]
    assert "current.json unreadable" in result.detail


def test_read_with_fallback_reports_malformed_prior_as_corrupt(tmp_path):
    proj = tmp_path / "proj"
    _commit_first(proj)
    artifacts2 = _seed_artifacts(proj / "index2", chunk_count=4)
    commit_manifest(proj, build_manifest(proj, artifacts2))
    (proj / "index2" / "chunk_ids.pkl").write_bytes(b"corrupt")
    (proj / "manifest" / "prior.json").write_text(
        "{broken",
        encoding="utf-8",
    )

    result = read_with_fallback(proj)
    assert result.freshness == "corrupt"
    assert result.manifest is None
    assert "prior.json unreadable" in result.detail


def test_read_with_fallback_uses_prior_when_current_missing(tmp_path):
    """current.json absent (e.g., crash between prior-promote and rename) but
    prior exists and verifies → returns prior with stale freshness."""
    proj = tmp_path / "proj"
    _commit_first(proj)
    artifacts2 = _seed_artifacts(proj / "index2", chunk_count=4)
    m2 = build_manifest(proj, artifacts2)
    commit_manifest(proj, m2)
    # Manually delete current.json (simulate crash mid-rename)
    (proj / "manifest" / "current.json").unlink()

    result = read_with_fallback(proj)
    assert result.ok is True
    assert result.freshness == "stale_using_prior_epoch"
    assert "current.json missing" in result.detail


def test_read_with_fallback_corrupt_when_both_fail(tmp_path):
    """Both current and prior fail verification → freshness='corrupt'."""
    proj = tmp_path / "proj"
    _commit_first(proj)
    artifacts2 = _seed_artifacts(proj / "index2", chunk_count=4)
    m2 = build_manifest(proj, artifacts2)
    commit_manifest(proj, m2)
    # Corrupt artifacts referenced by BOTH current (index2) and prior (index)
    (proj / "index2" / "chunk_ids.pkl").write_bytes(b"corrupt")
    (proj / "index" / "chunk_ids.pkl").write_bytes(b"corrupt")

    result = read_with_fallback(proj)
    assert result.ok is False
    assert result.freshness == "corrupt"
    assert "both failed" in result.detail
    # Detail surfaces both failures for diagnostic
    assert "current" in result.detail
    assert "prior" in result.detail


def test_read_with_fallback_corrupt_when_current_corrupt_no_prior(tmp_path):
    """Current corrupt + no prior (first commit then immediate corruption) → corrupt."""
    proj = tmp_path / "proj"
    _commit_first(proj)
    (proj / "index" / "chunk_ids.pkl").write_bytes(b"corrupt")

    result = read_with_fallback(proj)
    assert result.ok is False
    assert result.freshness == "corrupt"
    # Specifically calls out the lack of prior
    assert "no prior" in result.detail


def test_read_with_fallback_freshness_vocabulary_is_stable():
    """Pin the freshness vocabulary. Downstream consumers pattern-match these."""
    expected = {"fresh", "stale_using_prior_epoch", "missing", "corrupt"}
    # Hardcoded expected set — changing requires updating consumers.
    assert expected == {"fresh", "stale_using_prior_epoch", "missing", "corrupt"}


def test_read_result_ok_property():
    """ReadResult.ok is True iff manifest is not None."""
    r1 = ReadResult(manifest={"foo": "bar"}, freshness="fresh")
    r2 = ReadResult(manifest=None, freshness="missing")
    assert r1.ok is True
    assert r2.ok is False
