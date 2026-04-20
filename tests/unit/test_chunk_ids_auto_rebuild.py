"""Tests for auto-rebuild of chunk_ids.pkl on FAISS/list mismatch.

Regression coverage for the corruption mode where chunk_ids.pkl is
truncated to an empty list while FAISS and metadata.db remain intact.
Before this fix, every search raised `list index out of range` because
CodeIndexManager indexed into an empty list using positions from FAISS.
"""
import pickle
import tempfile
from pathlib import Path

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


def _seed(tmpdir: str, chunk_ids: list[str]) -> Path:
    """Build a fresh index with the given chunk IDs and return storage dir."""
    mgr = CodeIndexManager(tmpdir)
    mgr.create_index(embedding_dimension=384)
    mgr.add_embeddings([_make_result(c) for c in chunk_ids])
    mgr.save_index()
    storage = mgr.storage_dir
    _close(mgr)
    return storage


def test_rebuilds_chunk_ids_from_metadata_when_pkl_is_empty():
    """Truncated chunk_ids.pkl (empty list) should be rebuilt on next load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        expected = ["a.py:1-3:func:foo", "b.py:1-3:func:bar", "c.py:1-3:func:baz"]
        storage = _seed(tmpdir, expected)

        # Corrupt chunk_ids.pkl — truncate to an empty pickle list (5 bytes,
        # the exact shape we observed in the wild).
        pkl = storage / "chunk_ids.pkl"
        pkl.write_bytes(pickle.dumps([], protocol=4))
        assert pkl.stat().st_size < 20, "corruption setup failed"

        # Re-open: loader should detect mismatch and rebuild in place.
        mgr2 = CodeIndexManager(str(storage))
        _ = mgr2.index  # triggers _load_index via lazy property
        assert mgr2._chunk_ids == expected, (
            f"expected rebuilt chunk_ids {expected!r}, got {mgr2._chunk_ids!r}"
        )
        assert len(mgr2._chunk_ids) == mgr2._index.ntotal, (
            "rebuilt chunk_ids length must match FAISS ntotal"
        )

        # Rebuilt pkl should also be persisted.
        with open(pkl, "rb") as f:
            persisted = pickle.load(f)
        assert persisted == expected, "rebuild must persist to chunk_ids.pkl"

        # A backup of the corrupted pkl should exist alongside.
        bak_files = list(storage.glob("chunk_ids.pkl.bak.*"))
        assert len(bak_files) == 1, f"expected one backup, got {bak_files}"

        _close(mgr2)


def test_no_rebuild_when_chunk_ids_match():
    """Healthy load path must not touch chunk_ids.pkl or create a backup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        expected = ["x.py:1-3:func:one", "y.py:1-3:func:two"]
        storage = _seed(tmpdir, expected)
        pkl_before = (storage / "chunk_ids.pkl").read_bytes()

        mgr2 = CodeIndexManager(str(storage))
        _ = mgr2.index
        assert mgr2._chunk_ids == expected

        pkl_after = (storage / "chunk_ids.pkl").read_bytes()
        assert pkl_before == pkl_after, "healthy load must not rewrite pkl"
        bak_files = list(storage.glob("chunk_ids.pkl.bak.*"))
        assert bak_files == [], f"no backup should exist on healthy load: {bak_files}"

        _close(mgr2)


def test_rebuilds_when_pkl_is_missing_entirely():
    """Missing chunk_ids.pkl (not just empty) should also trigger rebuild."""
    with tempfile.TemporaryDirectory() as tmpdir:
        expected = ["m.py:1-3:func:m1", "n.py:1-3:func:n1", "o.py:1-3:func:o1"]
        storage = _seed(tmpdir, expected)

        (storage / "chunk_ids.pkl").unlink()

        mgr2 = CodeIndexManager(str(storage))
        _ = mgr2.index
        assert mgr2._chunk_ids == expected
        assert (storage / "chunk_ids.pkl").exists()

        _close(mgr2)


def test_skip_rebuild_when_metadata_cannot_cover_all_slots():
    """Partial metadata should NOT produce a half-rebuilt list.

    If the rebuild can't fill every FAISS position, leave self._chunk_ids
    alone (as the corrupted empty list) rather than ship a `None`-laced
    result that would silently mismatch FAISS positions at query time.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        expected = ["p.py:1-3:func:p1", "q.py:1-3:func:q1", "r.py:1-3:func:r1"]
        storage = _seed(tmpdir, expected)

        # Drop one metadata row so rebuild can't cover every FAISS slot.
        mgr_rm = CodeIndexManager(str(storage))
        _ = mgr_rm.index  # load
        mgr_rm.metadata_db.__delitem__("q.py:1-3:func:q1")
        mgr_rm.metadata_db.commit()
        _close(mgr_rm)

        # Corrupt the pkl and re-open.
        (storage / "chunk_ids.pkl").write_bytes(pickle.dumps([], protocol=4))
        mgr2 = CodeIndexManager(str(storage))
        _ = mgr2.index

        # Expected: rebuild aborted, chunk_ids stays empty (the observed
        # corruption state), caller will see `list index out of range` on
        # search — which is the correct signal to reindex.
        assert mgr2._chunk_ids == [], (
            "partial metadata must NOT produce a half-rebuilt list; "
            "got {mgr2._chunk_ids!r}"
        )

        _close(mgr2)
