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
