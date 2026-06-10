"""Tests for index hygiene under modify/delete churn.

Pins three fixes from the 2026-06-10 semantic-engine review:

1. remove_file_chunks matches paths exactly (segment-boundary), not by
   substring — removing `test.py` must not delete `conftest.py`'s chunks.
2. remove_file_chunks deletes the FTS5 rows, and add_embeddings is
   FTS-idempotent — a modify→re-add cycle must not duplicate a chunk in
   BM25 results.
3. The vector and BM25 search paths dedupe chunk_ids — stale FAISS
   vectors (never removed in place) must not occupy extra result slots or
   get double-counted by RRF.
"""
from __future__ import annotations

import tempfile

import numpy as np

from embeddings.embedder import EmbeddingResult
from search.indexer import CodeIndexManager


def _make_result(chunk_id: str, relative_path: str, dim: int = 32,
                 seed: int | None = None) -> EmbeddingResult:
    rng = np.random.RandomState(
        seed if seed is not None else (hash(chunk_id) & 0xFFFFFFFF)
    )
    return EmbeddingResult(
        embedding=rng.randn(dim).astype(np.float32),
        chunk_id=chunk_id,
        metadata={
            "file_path": f"/abs/project/{relative_path}",
            "relative_path": relative_path,
            "content_preview": f"def body_of_{chunk_id.replace(':', '_').replace('-', '_').replace('.', '_').replace('/', '_')}(): pass",
            "full_content": f"def chunk(): pass  # {chunk_id}",
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 3,
            "name": chunk_id.split(":")[-1],
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


def _fts_rows_for(mgr: CodeIndexManager, chunk_id: str) -> int:
    cur = mgr._fts_conn.execute(
        "SELECT COUNT(*) FROM chunk_fts WHERE chunk_id = ?", (chunk_id,)
    )
    return int(cur.fetchone()[0])


# ---------------------------------------------------------------------------
# 1. Exact path matching
# ---------------------------------------------------------------------------

class TestRemoveFileChunksPathMatching:

    def test_removing_test_py_keeps_conftest_py(self):
        """The substring-match regression shape: "test.py" is a substring of
        "conftest.py", so the old containment check deleted conftest's
        chunks when test.py was modified — conftest then silently vanished
        from the index (it was never re-embedded, since it didn't change).
        """
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            mgr.add_embeddings([
                _make_result("test.py:1-3:function:t1", "test.py"),
                _make_result("conftest.py:1-3:function:c1", "conftest.py"),
            ])

            removed = mgr.remove_file_chunks("test.py")

            assert removed == 1
            assert mgr.get_chunk_by_id("test.py:1-3:function:t1") is None
            assert mgr.get_chunk_by_id("conftest.py:1-3:function:c1") is not None, (
                "conftest.py chunks were deleted by a substring match on test.py"
            )
            _close(mgr)

    def test_removing_short_name_keeps_suffix_collisions(self):
        """`a.py` vs `data.py` — same shape at the directory level."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            mgr.add_embeddings([
                _make_result("a.py:1-3:function:fa", "a.py"),
                _make_result("data.py:1-3:function:fd", "data.py"),
            ])

            removed = mgr.remove_file_chunks("a.py")

            assert removed == 1
            assert mgr.get_chunk_by_id("data.py:1-3:function:fd") is not None
            _close(mgr)

    def test_absolute_target_matches_relative_metadata(self):
        """Callers may pass an absolute path; metadata stores both forms.
        Segment-boundary suffix matching must still connect them."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            mgr.add_embeddings([
                _make_result("src/util.py:1-3:function:u1", "src/util.py"),
            ])

            removed = mgr.remove_file_chunks("/abs/project/src/util.py")

            assert removed == 1
            _close(mgr)

    def test_relative_root_file_does_not_match_nested_same_basename(self):
        """Removing root-level `util.py` must not match `src/util.py` via
        an absolute-path suffix ("/abs/project/src/util.py" ends with
        "/util.py")."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            mgr.add_embeddings([
                _make_result("util.py:1-3:function:root", "util.py"),
                _make_result("src/util.py:1-3:function:nested", "src/util.py"),
            ])

            removed = mgr.remove_file_chunks("util.py")

            assert removed == 1
            assert mgr.get_chunk_by_id("util.py:1-3:function:root") is None
            assert mgr.get_chunk_by_id("src/util.py:1-3:function:nested") is not None
            _close(mgr)

    def test_relative_subpath_does_not_match_other_directories(self):
        """`util.py` at the root must not remove `src/util.py`... unless it
        IS a segment suffix. `b/util.py` vs `src/util.py` must not match."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            mgr.add_embeddings([
                _make_result("src/util.py:1-3:function:u1", "src/util.py"),
            ])

            removed = mgr.remove_file_chunks("b/util.py")

            assert removed == 0
            assert mgr.get_chunk_by_id("src/util.py:1-3:function:u1") is not None
            _close(mgr)


# ---------------------------------------------------------------------------
# 2. FTS5 hygiene across remove / re-add
# ---------------------------------------------------------------------------

class TestFtsChurnHygiene:

    def test_remove_file_chunks_deletes_fts_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            cid = "mod.py:1-3:function:f1"
            mgr.add_embeddings([_make_result(cid, "mod.py")])
            assert _fts_rows_for(mgr, cid) == 1

            mgr.remove_file_chunks("mod.py")

            assert _fts_rows_for(mgr, cid) == 0, (
                "stale FTS5 row survived remove_file_chunks"
            )
            _close(mgr)

    def test_modify_readd_cycle_does_not_duplicate_fts_rows(self):
        """The incremental flow for a modified file is remove→add. With
        unchanged chunk boundaries the chunk_id is identical; the FTS table
        has no uniqueness constraint, so pre-fix the row count grew by one
        per cycle and BM25 returned the chunk multiple times."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            cid = "mod.py:1-3:function:f1"
            for cycle in range(3):
                if cycle > 0:
                    mgr.remove_file_chunks("mod.py")
                mgr.add_embeddings([_make_result(cid, "mod.py", seed=cycle)])

            assert _fts_rows_for(mgr, cid) == 1

            results = mgr.search_bm25("chunk mod", k=10)
            ids = [r[0] for r in results]
            assert ids.count(cid) <= 1, f"duplicate BM25 rows for {cid}: {ids}"
            _close(mgr)

    def test_add_embeddings_is_fts_idempotent_without_remove(self):
        """Belt-and-suspenders: re-adding the same chunk_id directly (no
        remove first) replaces its FTS row instead of duplicating it."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            cid = "mod.py:1-3:function:f1"
            mgr.add_embeddings([_make_result(cid, "mod.py", seed=0)])
            mgr.add_embeddings([_make_result(cid, "mod.py", seed=1)])

            assert _fts_rows_for(mgr, cid) == 1
            _close(mgr)

    def test_duplicate_chunk_ids_in_chunk_ids_list_do_not_break_remove(self):
        """After a modify→re-add cycle _chunk_ids holds the same chunk_id at
        two FAISS positions. remove_file_chunks must dedupe before deleting
        metadata (pre-fix: KeyError on the second occurrence)."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            cid = "mod.py:1-3:function:f1"
            mgr.add_embeddings([_make_result(cid, "mod.py", seed=0)])
            mgr.add_embeddings([_make_result(cid, "mod.py", seed=1)])
            assert mgr._chunk_ids.count(cid) == 2

            removed = mgr.remove_file_chunks("mod.py")  # must not raise
            assert removed == 1
            assert mgr.get_chunk_by_id(cid) is None
            _close(mgr)


# ---------------------------------------------------------------------------
# 3. Search-path dedupe of stale FAISS duplicates
# ---------------------------------------------------------------------------

class TestVectorSearchDedupe:

    def test_duplicate_faiss_positions_yield_one_result(self):
        """Stale vectors are never removed from FAISS, so after a re-add the
        same chunk_id exists at two positions. The vector search must
        return it once (best-scoring occurrence), not twice — duplicates
        both wasted result slots and got double-counted by RRF."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            cid = "mod.py:1-3:function:f1"
            other = "other.py:1-3:function:g1"
            first = _make_result(cid, "mod.py", seed=0)
            mgr.add_embeddings([first, _make_result(other, "other.py", seed=7)])

            # Re-add the same chunk_id with the SAME vector → two FAISS rows
            # that both match a query for it.
            mgr.add_embeddings([
                EmbeddingResult(
                    embedding=first.embedding.copy(),
                    chunk_id=cid,
                    metadata=dict(first.metadata),
                )
            ])
            assert mgr._index.ntotal == 3

            results = mgr.search(first.embedding.copy(), k=3)
            ids = [r[0] for r in results]
            assert ids.count(cid) == 1, f"duplicate vector results: {ids}"
            assert other in ids, (
                "the duplicate consumed the result slot that belonged to "
                f"the other chunk: {ids}"
            )
            _close(mgr)

    def test_stats_report_live_vs_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            cid = "mod.py:1-3:function:f1"
            mgr.add_embeddings([_make_result(cid, "mod.py", seed=0)])
            mgr.add_embeddings([_make_result(cid, "mod.py", seed=1)])
            mgr.save_index()

            stats = mgr.get_stats()
            assert stats["live_chunks"] == 1
            assert stats["index_size"] == 2
            assert stats["stale_vectors"] == 1
            _close(mgr)


class TestBinaryModeSimilarChunks:

    def test_get_similar_chunks_uses_float_store(self, monkeypatch):
        """Binary indexes reconstruct to packed uint8 codes; pre-fix
        get_similar_chunks fed those codes to the float search path
        (normalize_L2 on uint8 → crash/garbage). The float store holds the
        original vectors and must be used instead."""
        monkeypatch.setenv("QUANTIZATION", "binary")
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            mgr.add_embeddings([
                _make_result("a.py:1-3:function:fa", "a.py", seed=1),
                _make_result("b.py:1-3:function:fb", "b.py", seed=2),
                _make_result("c.py:1-3:function:fc", "c.py", seed=3),
            ])
            assert getattr(mgr, "_is_binary", False) is True

            similar = mgr.get_similar_chunks("a.py:1-3:function:fa", k=2)

            ids = [cid for cid, _sim, _meta in similar]
            assert "a.py:1-3:function:fa" not in ids
            assert len(ids) >= 1
            _close(mgr)


class TestCountChunksInFile:

    def test_counts_live_chunks_for_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            mgr.add_embeddings([
                _make_result("f.py:1-3:function:a", "f.py"),
                _make_result("f.py:4-6:function:b", "f.py"),
                _make_result("g.py:1-3:function:c", "g.py"),
            ])
            assert mgr.count_chunks_in_file("f.py") == 2
            assert mgr.count_chunks_in_file("g.py") == 1
            assert mgr.count_chunks_in_file("missing.py") == 0
            assert mgr.count_chunks_in_file("") == 0
            _close(mgr)
