"""Tests for verify_index_integrity MCP tool (Plan-2 A3).

Pin the contract: structured JSON output that surfaces orphan/drift state
to LLM agents. Same scan as `cleanup_index_orphans.py --dry-run`, exposed
through the MCP boundary so an agent can detect post-incident drift
without operator intervention.
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

from mcp_server.code_search_server import CodeSearchServer


def _seed_clean_project(project_root: Path, name: str, chunk_count: int = 3) -> Path:
    """Seed a project with consistent index artifacts.

    chunk_ids.pkl, metadata.db, fts5.db, stats.json all agree on the same
    set of chunk_ids. Returns the project's index/ dir.
    """
    pdir = project_root / name
    idx = pdir / "index"
    idx.mkdir(parents=True, exist_ok=True)

    chunk_ids = [f"chunk_{i}" for i in range(chunk_count)]
    with open(idx / "chunk_ids.pkl", "wb") as f:
        pickle.dump(chunk_ids, f)

    # fts5.db
    con = sqlite3.connect(str(idx / "fts5.db"))
    try:
        con.execute(
            "CREATE VIRTUAL TABLE chunk_fts USING fts5(chunk_id, content)"
        )
        for cid in chunk_ids:
            con.execute(
                "INSERT INTO chunk_fts (chunk_id, content) VALUES (?, ?)",
                (cid, f"content for {cid}"),
            )
        con.commit()
    finally:
        con.close()

    # metadata.db (SqliteDict-style: table 'unnamed' with key/value cols)
    con = sqlite3.connect(str(idx / "metadata.db"))
    try:
        con.execute(
            "CREATE TABLE unnamed (key TEXT PRIMARY KEY, value BLOB)"
        )
        for cid in chunk_ids:
            con.execute(
                "INSERT INTO unnamed (key, value) VALUES (?, ?)",
                (cid, pickle.dumps({"chunk_type": "function"})),
            )
        con.commit()
    finally:
        con.close()

    # stats.json — agrees with pkl
    (idx / "stats.json").write_text(
        json.dumps({"total_chunks": chunk_count}),
        encoding="utf-8",
    )

    # project_info.json — required for list_projects discovery downstream,
    # though verify_index_integrity doesn't strictly require it.
    (pdir / "project_info.json").write_text(
        json.dumps({"project_name": name, "project_hash": "test_hash"}),
        encoding="utf-8",
    )

    return idx


def _inject_inconsistencies(
    idx: Path,
    fts5_orphans: int = 0,
    metadata_orphans: int = 0,
    stats_drift: int = 0,
) -> None:
    """Inject fts5/metadata orphans + stats drift into a previously-clean idx."""
    if fts5_orphans:
        con = sqlite3.connect(str(idx / "fts5.db"))
        try:
            for i in range(fts5_orphans):
                con.execute(
                    "INSERT INTO chunk_fts (chunk_id, content) VALUES (?, ?)",
                    (f"orphan_fts_{i}", f"orphan content {i}"),
                )
            con.commit()
        finally:
            con.close()
    if metadata_orphans:
        con = sqlite3.connect(str(idx / "metadata.db"))
        try:
            for i in range(metadata_orphans):
                con.execute(
                    "INSERT INTO unnamed (key, value) VALUES (?, ?)",
                    (f"orphan_meta_{i}", pickle.dumps({"chunk_type": "x"})),
                )
            con.commit()
        finally:
            con.close()
    if stats_drift:
        # Read current stats, bump total_chunks by drift, rewrite.
        stats = json.loads((idx / "stats.json").read_text(encoding="utf-8"))
        stats["total_chunks"] += stats_drift
        (idx / "stats.json").write_text(
            json.dumps(stats), encoding="utf-8",
        )


@pytest.fixture
def storage_root(tmp_path, monkeypatch):
    """Set CODE_SEARCH_STORAGE to a temp dir + return its projects/ subdir.

    Clears `common_utils.get_storage_dir`'s lru_cache so the new env var
    takes effect; without this, an earlier test (or import-time call) caches
    the default path and our monkeypatch is silently ignored.
    """
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    from common_utils import get_storage_dir
    get_storage_dir.cache_clear()
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    return projects_dir


def test_clean_project_reports_status_clean(storage_root):
    """A consistent project must show status=clean and zero counts."""
    _seed_clean_project(storage_root, "proj_a", chunk_count=5)
    server = CodeSearchServer()
    raw = server.verify_index_integrity()
    out = json.loads(raw)
    assert "projects" in out
    assert len(out["projects"]) == 1
    p = out["projects"][0]
    assert p["name"] == "proj_a"
    assert p["status"] == "clean"
    assert p["valid_chunks"] == 5
    assert p["fts5_orphans"] == 0
    assert p["metadata_orphans"] == 0
    assert p["stats_drift"] == 0
    assert out["summary"]["clean"] == 1
    assert out["summary"]["inconsistent"] == 0
    assert out["remediation"] is None


def test_inconsistent_project_reports_counts(storage_root):
    """A project with injected orphans/drift surfaces the right counts."""
    idx = _seed_clean_project(storage_root, "proj_b", chunk_count=4)
    _inject_inconsistencies(
        idx, fts5_orphans=3, metadata_orphans=2, stats_drift=5,
    )
    server = CodeSearchServer()
    raw = server.verify_index_integrity()
    out = json.loads(raw)
    p = out["projects"][0]
    assert p["status"] == "inconsistent"
    assert p["valid_chunks"] == 4
    assert p["fts5_orphans"] == 3
    assert p["metadata_orphans"] == 2
    assert p["stats_drift"] == 5
    # Samples should be present when inconsistent
    assert "fts5_sample" in p
    assert "metadata_sample" in p
    assert isinstance(p["fts5_sample"], list)
    assert isinstance(p["metadata_sample"], list)
    # Summary aggregates
    assert out["summary"]["total_projects"] == 1
    assert out["summary"]["inconsistent"] == 1
    assert out["summary"]["clean"] == 0
    assert out["summary"]["total_fts5_orphans"] == 3
    assert out["summary"]["total_metadata_orphans"] == 2
    assert out["summary"]["total_stats_drift"] == 5
    # Remediation pointer present when inconsistent
    assert out["remediation"] is not None
    assert "cleanup_index_orphans" in out["remediation"]


def test_multi_project_mixed_states(storage_root):
    """Mix of clean + inconsistent + unscannable projects."""
    _seed_clean_project(storage_root, "clean_proj", chunk_count=2)
    idx_dirty = _seed_clean_project(storage_root, "dirty_proj", chunk_count=3)
    _inject_inconsistencies(idx_dirty, fts5_orphans=1, stats_drift=-2)
    # An unscannable project: directory exists but no index/ subdir
    (storage_root / "unscannable_proj").mkdir()

    server = CodeSearchServer()
    raw = server.verify_index_integrity()
    out = json.loads(raw)
    by_name = {p["name"]: p for p in out["projects"]}
    assert by_name["clean_proj"]["status"] == "clean"
    assert by_name["dirty_proj"]["status"] == "inconsistent"
    assert by_name["dirty_proj"]["fts5_orphans"] == 1
    assert by_name["dirty_proj"]["stats_drift"] == -2
    assert by_name["unscannable_proj"]["status"] == "unscannable"
    assert "reason" in by_name["unscannable_proj"]
    assert out["summary"]["clean"] == 1
    assert out["summary"]["inconsistent"] == 1
    assert out["summary"]["unscannable"] == 1
    assert out["summary"]["total_projects"] == 3
    # Drift summed as absolute value
    assert out["summary"]["total_stats_drift"] == 2


def test_project_filter_prefix(storage_root):
    """The `project` arg filters by directory-name prefix."""
    _seed_clean_project(storage_root, "alpha_one", chunk_count=1)
    _seed_clean_project(storage_root, "alpha_two", chunk_count=1)
    _seed_clean_project(storage_root, "beta_proj", chunk_count=1)
    server = CodeSearchServer()
    raw = server.verify_index_integrity(project="alpha")
    out = json.loads(raw)
    names = {p["name"] for p in out["projects"]}
    assert names == {"alpha_one", "alpha_two"}
    assert out["summary"]["total_projects"] == 2


def test_project_filter_no_match(storage_root):
    """Non-matching prefix returns an error response, not an empty list."""
    _seed_clean_project(storage_root, "real_proj", chunk_count=1)
    server = CodeSearchServer()
    raw = server.verify_index_integrity(project="nonexistent")
    out = json.loads(raw)
    assert "error" in out
    assert "nonexistent" in out["error"]


def test_no_storage_dir_returns_zero_projects(tmp_path, monkeypatch):
    """When the storage dir doesn't have a projects/ subdir, return empty
    summary instead of crashing."""
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    from common_utils import get_storage_dir
    get_storage_dir.cache_clear()
    # Don't create projects/ subdir
    server = CodeSearchServer()
    raw = server.verify_index_integrity()
    out = json.loads(raw)
    assert out["projects"] == []
    assert out["summary"]["total_projects"] == 0
    assert "message" in out


def test_unreadable_chunk_ids_pkl_marks_unscannable(storage_root):
    """A project with corrupt/unreadable pkl is unscannable, not crashed."""
    pdir = storage_root / "broken"
    idx = pdir / "index"
    idx.mkdir(parents=True)
    (idx / "chunk_ids.pkl").write_bytes(b"\x80not-actually-a-pickle\x00\x00")
    server = CodeSearchServer()
    raw = server.verify_index_integrity()
    out = json.loads(raw)
    assert len(out["projects"]) == 1
    assert out["projects"][0]["status"] == "unscannable"


def test_response_is_valid_json(storage_root):
    """The tool's contract is JSON-string output. Verify always valid JSON."""
    _seed_clean_project(storage_root, "p", chunk_count=1)
    server = CodeSearchServer()
    raw = server.verify_index_integrity()
    # Must round-trip through json without error
    out = json.loads(raw)
    assert isinstance(out, dict)
    # Must serialize back without circular refs / weird types
    json.dumps(out)


# ─── Plan-2 E2-5: manifest-aware integrity tests (PR #121) ───
#
# After E2-1 (PR #119), save_index commits an epoch manifest. E2-5
# extends verify_index_integrity to surface that state. Each per-project
# entry now carries:
#   - manifest_status: fresh | stale_using_prior_epoch | missing | corrupt | skipped
#   - manifest_epoch_id: str or None
#   - manifest_stale_candidate: bool
# And the summary aggregates manifest_fresh / _stale_prior / _missing /
# _corrupt / total_stale_candidates.


def _commit_manifest_for_seeded_project(idx: Path, chunk_count: int) -> str:
    """Commit a real epoch manifest for the artifacts already in idx.

    Builds an ArtifactSpec list that matches what _seed_clean_project
    wrote, then calls build_manifest + commit_manifest. Returns the
    epoch_id so tests can pin it.

    We don't go through CodeIndexManager.save_index because (a) save_index
    expects an in-memory FAISS state and (b) the fake artifacts the test
    fixture writes don't include code.index. The manifest just covers
    the artifacts that DO exist (chunk_ids.pkl, metadata.db, fts5.db,
    stats.json). build_manifest's consistency check holds since
    _seed_clean_project guarantees count alignment.
    """
    from search.epoch_manifest import (
        ArtifactSpec,
        build_manifest,
        commit_manifest,
        count_fts5_db,
        count_metadata_db,
    )

    artifacts = [
        ArtifactSpec(
            name="chunk_ids.pkl",
            path=idx / "chunk_ids.pkl",
            count=chunk_count,
        ),
        ArtifactSpec(
            name="metadata.db",
            path=idx / "metadata.db",
            count=count_metadata_db(idx / "metadata.db"),
        ),
        ArtifactSpec(
            name="fts5.db",
            path=idx / "fts5.db",
            count=count_fts5_db(idx / "fts5.db"),
        ),
        ArtifactSpec(
            name="stats.json",
            path=idx / "stats.json",
            count=None,
        ),
    ]
    manifest = build_manifest(idx, artifacts)
    commit_manifest(idx, manifest)
    return manifest["epoch_id"]


def test_legacy_project_without_manifest_reports_missing(storage_root):
    """A clean project that pre-dates PR #119's manifest commit shows
    `manifest_status: missing` while still being chunk-level clean."""
    _seed_clean_project(storage_root, "legacy", chunk_count=3)
    server = CodeSearchServer()
    raw = server.verify_index_integrity()
    out = json.loads(raw)
    p = out["projects"][0]
    # Chunk-level state is fine.
    assert p["status"] == "clean"
    # Manifest state surfaces as missing (no current.json yet).
    assert p["manifest_status"] == "missing"
    assert p["manifest_epoch_id"] is None
    assert p["manifest_stale_candidate"] is False
    # Detail message is propagated from read_with_fallback.
    assert "manifest_detail" in p
    # Summary reflects the legacy state.
    assert out["summary"]["manifest_missing"] == 1
    assert out["summary"]["manifest_fresh"] == 0


def test_freshly_committed_manifest_reports_fresh(storage_root):
    """After a real commit_manifest, status=clean AND manifest_status=fresh
    with the pinned epoch_id."""
    idx = _seed_clean_project(storage_root, "fresh_proj", chunk_count=5)
    epoch_id = _commit_manifest_for_seeded_project(idx, chunk_count=5)

    server = CodeSearchServer()
    raw = server.verify_index_integrity()
    out = json.loads(raw)
    p = out["projects"][0]
    assert p["status"] == "clean"
    assert p["manifest_status"] == "fresh"
    assert p["manifest_epoch_id"] == epoch_id
    assert p["manifest_stale_candidate"] is False
    # No detail when fresh.
    assert "manifest_detail" not in p
    assert out["summary"]["manifest_fresh"] == 1


def test_corrupt_manifest_reports_corrupt(storage_root):
    """A current.json whose recorded SHAs don't match actual artifacts is
    detected and reported (no prior to fall back to)."""
    idx = _seed_clean_project(storage_root, "corrupt_proj", chunk_count=2)
    _commit_manifest_for_seeded_project(idx, chunk_count=2)

    # Mutate chunk_ids.pkl AFTER manifest commit. The recorded SHA in
    # current.json no longer matches; verify_manifest fails; no prior
    # exists; read_with_fallback returns "corrupt".
    extra_chunks = ["chunk_0", "chunk_1", "extra_chunk"]
    with open(idx / "chunk_ids.pkl", "wb") as f:
        pickle.dump(extra_chunks, f)

    server = CodeSearchServer()
    raw = server.verify_index_integrity()
    out = json.loads(raw)
    p = out["projects"][0]
    assert p["manifest_status"] == "corrupt"
    assert "manifest_detail" in p
    assert out["summary"]["manifest_corrupt"] == 1
    # Remediation surfaces the corrupt-manifest pointer.
    assert "Manifest corruption" in (out["remediation"] or "")


def test_stale_candidate_surfaces(storage_root):
    """A leftover manifest/candidate.json (crashed-write residue) is
    flagged on the per-project entry and counted in summary."""
    idx = _seed_clean_project(storage_root, "stale_cand_proj", chunk_count=2)
    _commit_manifest_for_seeded_project(idx, chunk_count=2)
    # Simulate a prior crashed write that left candidate.json behind.
    candidate = idx / "manifest" / "candidate.json"
    candidate.write_text(
        json.dumps({"epoch_id": "stale-crash-residue", "artifacts": {}}),
        encoding="utf-8",
    )

    server = CodeSearchServer()
    raw = server.verify_index_integrity()
    out = json.loads(raw)
    p = out["projects"][0]
    assert p["manifest_status"] == "fresh"  # current.json still good
    assert p["manifest_stale_candidate"] is True
    assert out["summary"]["total_stale_candidates"] == 1
    assert "candidate.json" in (out["remediation"] or "")


def test_unscannable_project_marks_manifest_skipped(storage_root):
    """A project without index/ subdir (or with unreadable pkl) gets
    manifest_status='skipped' rather than crashing the manifest probe."""
    (storage_root / "no_index").mkdir()
    server = CodeSearchServer()
    raw = server.verify_index_integrity()
    out = json.loads(raw)
    p = out["projects"][0]
    assert p["status"] == "unscannable"
    assert p["manifest_status"] == "skipped"
    assert p["manifest_epoch_id"] is None
    assert p["manifest_stale_candidate"] is False
    # Skipped projects don't pollute the manifest totals.
    assert out["summary"]["manifest_fresh"] == 0
    assert out["summary"]["manifest_missing"] == 0
    assert out["summary"]["manifest_corrupt"] == 0


def test_summary_aggregates_mixed_manifest_states(storage_root):
    """A mix of fresh + missing + corrupt projects produces correct totals."""
    # Project A: manifest committed, fresh.
    idx_a = _seed_clean_project(storage_root, "a_fresh", chunk_count=2)
    _commit_manifest_for_seeded_project(idx_a, chunk_count=2)

    # Project B: legacy (no manifest).
    _seed_clean_project(storage_root, "b_missing", chunk_count=3)

    # Project C: manifest committed then mutated (corrupt).
    idx_c = _seed_clean_project(storage_root, "c_corrupt", chunk_count=4)
    _commit_manifest_for_seeded_project(idx_c, chunk_count=4)
    with open(idx_c / "chunk_ids.pkl", "wb") as f:
        pickle.dump(["only_one_now"], f)

    server = CodeSearchServer()
    raw = server.verify_index_integrity()
    out = json.loads(raw)
    summary = out["summary"]
    assert summary["total_projects"] == 3
    assert summary["manifest_fresh"] == 1
    assert summary["manifest_missing"] == 1
    assert summary["manifest_corrupt"] == 1
    assert summary["manifest_stale_prior"] == 0
