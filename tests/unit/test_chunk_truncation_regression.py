"""Tests pinning the chunk-truncation regression observed 2026-05-04/05.

Failure shape (live-reproduced 2026-05-05 on the .claude project):
- A fresh CodeIndexManager (e.g., after switch_project) starts with
  _index=None and _chunk_ids=[].
- The MCP's auto_reindex_if_needed runs incremental_index on a stale
  snapshot.
- _remove_old_chunks iterates the empty _chunk_ids list — no-op.
- _add_new_chunks calls indexer.add_embeddings, which sees _index is None
  and falls through to create_index() — a brand-new empty FAISS — even
  though a healthy on-disk index exists. The new embedding is added to
  the empty FAISS.
- save_index then dumps the truncated in-memory state (1 entry) over
  the healthy on-disk pkl (30+ entries).

Two-layer fix:
1. add_embeddings and remove_file_chunks lazy-load the on-disk index
   before modifying state, so the in-memory _chunk_ids reflects what's
   on disk.
2. save_index defensively refuses to clobber a healthy on-disk pkl
   with a much smaller in-memory list unless force=True is passed.
"""
from __future__ import annotations

import pickle
import tempfile

import numpy as np

from search.indexer import CodeIndexManager
from embeddings.embedder import EmbeddingResult


def _make_result(chunk_id: str, dim: int = 384) -> EmbeddingResult:
    return EmbeddingResult(
        embedding=np.random.randn(dim).astype(np.float32),
        chunk_id=chunk_id,
        metadata={
            "file_path": f"{chunk_id}.py",
            "relative_path": f"{chunk_id}.py",
            "content_preview": f"def {chunk_id}(): pass",
            "full_content": f"def {chunk_id}(): pass",
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 3,
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


def _close(mgr: CodeIndexManager) -> None:
    if mgr._metadata_db is not None:
        mgr._metadata_db.close()
        mgr._metadata_db = None
    if getattr(mgr, "_fts_conn", None) is not None:
        mgr._fts_conn.close()
        mgr._fts_conn = None


def _seed_30(tmpdir: str) -> CodeIndexManager:
    """Create a healthy 30-chunk index in the given storage dir."""
    mgr = CodeIndexManager(tmpdir)
    mgr.create_index(embedding_dimension=384)
    mgr.add_embeddings([_make_result(f"c{i:03d}") for i in range(30)])
    mgr.save_index()
    return mgr


def test_fresh_manager_add_embeddings_loads_existing_index():
    """A fresh CodeIndexManager (post-switch) must load existing state
    before add_embeddings, so the in-memory _chunk_ids reflects on-disk."""
    with tempfile.TemporaryDirectory() as tmp:
        seed = _seed_30(tmp)
        _close(seed)

        # Fresh instance — simulates post-switch_project state.
        mgr2 = CodeIndexManager(tmp)
        assert mgr2._index is None
        assert mgr2._chunk_ids == []

        # add_embeddings should now load the existing index first, so the
        # 1 new entry is APPENDED, not the only entry remaining.
        mgr2.add_embeddings([_make_result("c_new_001")])
        assert len(mgr2._chunk_ids) == 31, (
            f"expected 31 chunks (30 loaded + 1 new), got {len(mgr2._chunk_ids)}"
        )
        assert mgr2._index is not None
        assert mgr2._index.ntotal == 31
        _close(mgr2)


def test_fresh_manager_remove_file_chunks_loads_existing_index():
    """A fresh CodeIndexManager must load before remove_file_chunks so
    iteration over _chunk_ids actually finds the file's chunks."""
    with tempfile.TemporaryDirectory() as tmp:
        seed = _seed_30(tmp)
        _close(seed)

        mgr2 = CodeIndexManager(tmp)
        assert mgr2._chunk_ids == []

        # Remove chunks for file c005.py — should find and remove 1.
        # Without the load-first fix, this returns 0 (silent failure).
        n = mgr2.remove_file_chunks("c005.py")
        assert n == 1, (
            f"expected to remove 1 chunk, got {n}. The load-first fix did "
            f"not fire — fresh _chunk_ids was empty so no chunks matched."
        )
        _close(mgr2)


def test_save_index_refuses_clobber_without_force():
    """save_index refuses when in-memory is dramatically smaller than
    on-disk pkl, unless force=True is passed."""
    with tempfile.TemporaryDirectory() as tmp:
        seed = _seed_30(tmp)
        _close(seed)
        chunk_id_path = (
            CodeIndexManager(tmp).chunk_id_path
        )  # construct only to read the path
        original_pkl_size = chunk_id_path.stat().st_size
        assert original_pkl_size > 200, "test fixture pkl too small"

        # Simulate the regression: fresh manager, force _index to a
        # 1-entry FAISS, _chunk_ids to ['c_lone']. save_index should
        # REFUSE to overwrite the healthy 30-entry pkl.
        mgr2 = CodeIndexManager(tmp)
        mgr2.create_index(embedding_dimension=384)
        mgr2.add_embeddings([_make_result("c_lone")])
        # Note: thanks to the load-first fix, add_embeddings would actually
        # load the existing 30; manually clobber the in-memory state to
        # simulate a future regression that bypasses the load fix.
        mgr2._chunk_ids = ["c_lone"]
        # FAISS state: leave whatever add_embeddings produced (>=1 vector).

        mgr2.save_index()  # default force=False — should refuse

        # On-disk pkl should be UNCHANGED.
        new_pkl_size = chunk_id_path.stat().st_size
        assert new_pkl_size == original_pkl_size, (
            f"guard failed: pkl shrunk from {original_pkl_size} to "
            f"{new_pkl_size} bytes"
        )

        # On-disk pkl content should be the original 30 entries.
        with open(chunk_id_path, "rb") as f:
            on_disk_ids = pickle.load(f)
        assert len(on_disk_ids) == 30
        _close(mgr2)


def test_save_index_with_force_proceeds():
    """save_index(force=True) bypasses the guard for legitimate shrinks
    (e.g., post-clear_index, intentional reset)."""
    with tempfile.TemporaryDirectory() as tmp:
        seed = _seed_30(tmp)
        _close(seed)
        chunk_id_path = CodeIndexManager(tmp).chunk_id_path

        mgr2 = CodeIndexManager(tmp)
        mgr2.create_index(embedding_dimension=384)
        mgr2.add_embeddings([_make_result("c_new")])
        mgr2._chunk_ids = ["c_new"]

        mgr2.save_index(force=True)

        with open(chunk_id_path, "rb") as f:
            on_disk_ids = pickle.load(f)
        assert on_disk_ids == ["c_new"], (
            f"force=True did not proceed; on-disk has {on_disk_ids}"
        )
        _close(mgr2)


def test_save_index_allows_growth():
    """Normal growth (in-memory > on-disk) is not blocked by the guard."""
    with tempfile.TemporaryDirectory() as tmp:
        seed = _seed_30(tmp)
        _close(seed)

        mgr2 = CodeIndexManager(tmp)
        # add_embeddings auto-loads existing 30 + appends 5 = 35 in memory.
        mgr2.add_embeddings([_make_result(f"c_new_{i}") for i in range(5)])
        assert len(mgr2._chunk_ids) == 35

        mgr2.save_index()

        chunk_id_path = mgr2.chunk_id_path
        with open(chunk_id_path, "rb") as f:
            on_disk_ids = pickle.load(f)
        assert len(on_disk_ids) == 35
        _close(mgr2)


def test_save_index_allows_small_legitimate_shrinks():
    """Removing one file from a 30-chunk index should still save (in-memory
    29, on-disk 30 → ratio 29*32=928 vs 966*0.5=483, 928 > 483 → no trip)."""
    with tempfile.TemporaryDirectory() as tmp:
        seed = _seed_30(tmp)
        _close(seed)

        mgr2 = CodeIndexManager(tmp)
        # remove_file_chunks auto-loads, removes 1, leaves 30 in metadata
        # but only updates the metadata_db (not _chunk_ids — note: the
        # implementation removes from metadata only, not from _chunk_ids,
        # because FAISS removal is not done; so _chunk_ids stays at 30).
        # That means save_index sees in-memory=30, on-disk pkl=966 → no trip.
        n_removed = mgr2.remove_file_chunks("c000.py")
        assert n_removed == 1
        # _chunk_ids stays at 30 (FAISS removal is "rebuild on demand")
        assert len(mgr2._chunk_ids) == 30
        mgr2.save_index()  # should proceed normally
        _close(mgr2)
