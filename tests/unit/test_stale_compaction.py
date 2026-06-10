"""Tests for stale-vector accounting + auto-compaction (roadmap P5).

FAISS rows are never removed in place, so modify/delete churn accumulates
stale vectors. PR #224 added the live/stale stats; P5 acts on them:
- CodeIndexManager.stale_ratio() — live measurement, None when unknown
- IncrementalIndexer escalates an incremental run to a full reindex when
  the ratio exceeds STALE_COMPACTION_RATIO (0.5) — self-limiting because
  the full reindex resets the ratio to 0
"""
from __future__ import annotations

import tempfile
from typing import List

import numpy as np
import pytest

from chunking.code_chunk import CodeChunk
from chunking.multi_language_chunker import MultiLanguageChunker
from embeddings.embedder import EmbeddingResult
from merkle.snapshot_manager import SnapshotManager
from search.incremental_indexer import IncrementalIndexer
from search.indexer import CodeIndexManager


def _make_result(chunk_id: str, relative_path: str, seed: int = 0,
                 dim: int = 16) -> EmbeddingResult:
    rng = np.random.RandomState((hash(chunk_id) ^ seed) & 0xFFFFFFFF)
    return EmbeddingResult(
        embedding=rng.randn(dim).astype(np.float32),
        chunk_id=chunk_id,
        metadata={
            "file_path": f"/abs/{relative_path}",
            "relative_path": relative_path,
            "content_preview": "def x(): pass",
            "full_content": "def x(): pass",
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 3,
            "name": "x",
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


class TestStaleRatio:

    def test_empty_index_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            assert mgr.stale_ratio() is None
            _close(mgr)

    def test_healthy_index_ratio_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            mgr.add_embeddings([
                _make_result("a.py:1-3:function:x", "a.py"),
                _make_result("b.py:1-3:function:x", "b.py"),
            ])
            assert mgr.stale_ratio() == 0.0
            _close(mgr)

    def test_churn_inflates_ratio(self):
        """Re-adding the same chunk_id leaves a stale FAISS row: ntotal=2,
        live=1 → ratio 1.0."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            cid = "a.py:1-3:function:x"
            mgr.add_embeddings([_make_result(cid, "a.py", seed=0)])
            mgr.add_embeddings([_make_result(cid, "a.py", seed=1)])
            assert mgr.stale_ratio() == pytest.approx(1.0)
            _close(mgr)

    def test_removed_file_counts_as_stale(self):
        """remove_file_chunks deletes metadata but not FAISS rows: those
        rows are stale until compaction."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            mgr.add_embeddings([
                _make_result("a.py:1-3:function:x", "a.py"),
                _make_result("b.py:1-3:function:x", "b.py"),
            ])
            mgr.remove_file_chunks("a.py")
            # ntotal=2, live=1 → ratio 1.0
            assert mgr.stale_ratio() == pytest.approx(1.0)
            _close(mgr)


# ---------------------------------------------------------------------------
# Escalation through IncrementalIndexer
# ---------------------------------------------------------------------------

class _FakeModelHandle:
    def __init__(self, dim: int):
        self.dim = dim
        self._model_name = "mock-embedder"

    def get_embedding_dimension(self):
        return self.dim


class _WorkingEmbedder:
    def __init__(self, dim: int = 16):
        self.dim = dim
        self._model = _FakeModelHandle(dim)

    def embed_chunks_grouped(self, chunks: List[CodeChunk], batch_size: int = 32):
        results = []
        for c in chunks:
            chunk_id = f"{c.relative_path}:{c.start_line}-{c.end_line}:{c.chunk_type}"
            if c.name:
                chunk_id += f":{c.name}"
            results.append(EmbeddingResult(
                embedding=np.random.RandomState(hash(chunk_id) & 0xFFFFFFFF)
                    .randn(self.dim).astype(np.float32),
                chunk_id=chunk_id,
                metadata={
                    "file_path": c.file_path,
                    "relative_path": c.relative_path,
                    "content_preview": c.content[:100],
                    "full_content": c.content,
                    "chunk_type": c.chunk_type,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "name": c.name,
                    "parent_name": c.parent_name,
                    "docstring": c.docstring,
                    "decorators": c.decorators,
                    "imports": c.imports,
                    "complexity_score": c.complexity_score,
                    "tags": c.tags,
                    "folder_structure": c.folder_structure,
                },
            ))
        return results

    def create_embedding_content(self, chunk: CodeChunk) -> str:
        return chunk.content


@pytest.fixture
def parts(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "alpha.py").write_text("def alpha():\n    return 1\n")
    (proj / "beta.py").write_text("def beta():\n    return 2\n")

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()

    indexer = CodeIndexManager(str(index_dir))
    chunker = MultiLanguageChunker(root_path=str(proj))
    snapshot_manager = SnapshotManager(snapshot_dir)
    ii = IncrementalIndexer(
        indexer=indexer, embedder=_WorkingEmbedder(), chunker=chunker,
        snapshot_manager=snapshot_manager,
    )
    return proj, indexer, ii


def _modify_all(proj, marker: int) -> None:
    """Rewrite both files so chunk boundaries (and hence chunk_ids) shift."""
    filler = "    # churn marker\n" * marker
    (proj / "alpha.py").write_text(
        f"def alpha():\n{filler}    return {marker}\n"
    )
    (proj / "beta.py").write_text(
        f"def beta():\n{filler}    return {marker + 100}\n"
    )


def test_below_threshold_stays_incremental(parts):
    proj, indexer, ii = parts
    result = ii.incremental_index(str(proj), project_name="proj")
    assert result.success

    # No changes, healthy index: must take the cheap no-change path.
    ntotal_before = indexer._index.ntotal
    result2 = ii.incremental_index(str(proj), project_name="proj")
    assert result2.success
    assert result2.files_added == 0
    assert result2.chunks_added == 0
    assert indexer._index.ntotal == ntotal_before, (
        "no-change incremental run must not rebuild the index"
    )


def test_churn_past_threshold_escalates_to_full_reindex(parts):
    proj, indexer, ii = parts
    assert ii.incremental_index(str(proj), project_name="proj").success

    # One modify-everything cycle: both files' chunk_ids shift, leaving the
    # original vectors stale. The cycle itself runs incrementally (the ratio
    # is checked BEFORE the run, when the index is still healthy); afterwards
    # ntotal ≈ 2x live → ratio ≈ 1.0 > 0.5.
    _modify_all(proj, 1)
    cycle = ii.incremental_index(str(proj), project_name="proj")
    assert cycle.success
    assert cycle.files_modified > 0, "modify cycle must run incrementally"

    ratio = indexer.stale_ratio()
    assert ratio is not None and ratio > indexer.STALE_COMPACTION_RATIO, (
        f"test setup failed to inflate stale ratio (got {ratio})"
    )

    # Next call with NO changes: pre-P5 this was a no-op; now it must
    # escalate to a full reindex and compact.
    result = ii.incremental_index(str(proj), project_name="proj")
    assert result.success
    assert result.files_added > 0, (
        "expected full-reindex result shape (files_added>0), got the "
        "no-change incremental shape — escalation did not fire"
    )
    post_ratio = indexer.stale_ratio()
    assert post_ratio == 0.0, (
        f"full reindex must reset the stale ratio, got {post_ratio}"
    )


def test_force_full_unaffected_by_ratio(parts):
    proj, indexer, ii = parts
    assert ii.incremental_index(str(proj), project_name="proj").success
    result = ii.incremental_index(
        str(proj), project_name="proj", force_full=True
    )
    assert result.success
    assert result.files_added > 0
    assert indexer.stale_ratio() == 0.0
