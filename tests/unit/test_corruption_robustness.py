"""Torn-write corruption contract for the index read path (2026-06-10 fuzz).

Pre-fix, 13 of 24 (artifact x corruption-mode) shapes raised unhandled
exceptions from search-path entry points — including a corrupt fts5.db
making CodeIndexManager UNCONSTRUCTABLE (the constructor runs _init_fts5).

Contract pinned here:
- DERIVED artifacts (fts5.db, code.index, chunk_ids.pkl, stats.json,
  manifest) degrade gracefully: constructor + search + search_bm25 +
  get_stats + stale_ratio never raise. chunk_ids.pkl additionally recovers
  LOSSLESSLY (rebuilt from metadata.db).
- metadata.db (NOT rebuildable) raises an actionable RuntimeError naming
  the remedy (reindex) instead of a raw sqlitedict traceback.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from embeddings.embedder import EmbeddingResult
from search.indexer import CodeIndexManager


def _make_result(i: int) -> EmbeddingResult:
    rng = np.random.RandomState(i)
    cid = f"f{i}.py:1-3:function:fn{i}"
    return EmbeddingResult(
        embedding=rng.randn(16).astype(np.float32), chunk_id=cid,
        metadata={
            "file_path": f"/abs/f{i}.py", "relative_path": f"f{i}.py",
            "content_preview": f"def fn{i}(): pass",
            "full_content": f"def fn{i}(): pass uniqtok{i}",
            "chunk_type": "function", "start_line": 1, "end_line": 3,
            "name": f"fn{i}", "parent_name": None, "docstring": None,
            "decorators": [], "imports": [], "complexity_score": 1,
            "tags": [], "folder_structure": [],
        })


def _close(mgr: CodeIndexManager) -> None:
    for h in (mgr._metadata_db, getattr(mgr, "_fts_conn", None)):
        try:
            if h is not None:
                h.close()
        except Exception:
            pass
    mgr._metadata_db = None
    mgr._fts_conn = None


@pytest.fixture
def healthy_index(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    mgr = CodeIndexManager(str(base))
    mgr.add_embeddings([_make_result(i) for i in range(12)])
    mgr.save_index()
    _close(mgr)
    return base


def _corrupt(path: Path, mode: str) -> None:
    if mode == "delete":
        path.unlink()
        return
    data = path.read_bytes()
    if mode == "truncate_half":
        path.write_bytes(data[: len(data) // 2])
    elif mode == "truncate_zero":
        path.write_bytes(b"")
    elif mode == "garbage":
        path.write_bytes(b"\x9c\xff\x00GARBAGE" * 64)


def _probe_all(storage: Path):
    """Constructor + all read entry points; returns the manager."""
    q = np.random.RandomState(0).randn(16).astype(np.float32)
    mgr = CodeIndexManager(str(storage))
    try:
        mgr.search(q.copy(), k=5)
        mgr.search_bm25("uniqtok1", k=5)
        mgr.get_stats()
        mgr.stale_ratio()
    finally:
        _close(mgr)


DERIVED = [
    ("code.index", m) for m in ("truncate_half", "truncate_zero", "garbage", "delete")
] + [
    ("chunk_ids.pkl", m) for m in ("truncate_half", "truncate_zero", "garbage", "delete")
] + [
    ("fts5.db", m) for m in ("truncate_half", "truncate_zero", "garbage", "delete")
] + [
    ("stats.json", m) for m in ("truncate_half", "truncate_zero", "garbage", "delete")
] + [
    ("manifest/current.json", m) for m in ("truncate_half", "truncate_zero", "garbage", "delete")
] + [
    ("metadata.db", "truncate_zero"),  # zero-byte sqlite file is re-initialized
    ("metadata.db", "delete"),
]


@pytest.mark.parametrize("artifact,mode", DERIVED)
def test_derived_artifact_corruption_never_raises(healthy_index, tmp_path, artifact, mode):
    work = tmp_path / "work"
    shutil.copytree(healthy_index, work)
    target = work / artifact
    if not target.exists():
        pytest.skip(f"{artifact} not present in healthy index")
    _corrupt(target, mode)
    _probe_all(work)  # must not raise


@pytest.mark.parametrize("mode", ["truncate_half", "garbage"])
def test_metadata_corruption_raises_actionable_error(healthy_index, tmp_path, mode):
    work = tmp_path / "work"
    shutil.copytree(healthy_index, work)
    _corrupt(work / "metadata.db", mode)
    mgr = CodeIndexManager(str(work))
    try:
        q = np.random.RandomState(0).randn(16).astype(np.float32)
        with pytest.raises(RuntimeError, match="reindex"):
            mgr.search(q, k=5)
    finally:
        _close(mgr)


def test_corrupt_chunk_ids_recovers_losslessly(healthy_index, tmp_path):
    """The pkl is rebuilt from metadata index_ids — search must still
    return results, not just avoid crashing."""
    work = tmp_path / "work"
    shutil.copytree(healthy_index, work)
    _corrupt(work / "chunk_ids.pkl", "garbage")
    mgr = CodeIndexManager(str(work))
    try:
        results = mgr.search_bm25("uniqtok3", k=5)
        assert any("f3.py" in (m.get("relative_path") or "") for _, _, m in results)
        q = np.random.RandomState(0).randn(16).astype(np.float32)
        assert mgr.search(q, k=5), "vector search empty after pkl rebuild"
    finally:
        _close(mgr)


def test_corrupt_fts_is_quarantined_and_bm25_degrades(healthy_index, tmp_path):
    work = tmp_path / "work"
    shutil.copytree(healthy_index, work)
    _corrupt(work / "fts5.db", "garbage")
    mgr = CodeIndexManager(str(work))  # constructor must survive
    try:
        assert mgr.search_bm25("uniqtok1", k=5) == []  # degraded, not crashed
        quarantined = list(work.glob("fts5.db.corrupt.*"))
        assert quarantined, "corrupt fts5.db was not quarantined"
        # vector leg unaffected
        q = np.random.RandomState(0).randn(16).astype(np.float32)
        assert mgr.search(q, k=5)
    finally:
        _close(mgr)
