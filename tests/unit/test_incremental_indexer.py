"""Unit tests for IncrementalIndexer.

Coverage focus areas identified by the 2026-05-22 deep-assessment:
- Manifest fatal contract: ManifestConsistencyError now propagates instead
  of being silently swallowed. last_manifest_commit_status surfaces the
  outcome of the most recent commit attempt for observability.
- Chunking diagnostics: per-run summary of chunking outcomes so silent
  zero-chunk failures (parse errors, encoding errors) stop being invisible.
- Dispatch correctness: full vs incremental path selection.

The integration test in `tests/integration/test_incremental_indexing.py`
covers end-to-end flow; these tests pin the new contract surfaces.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from chunking.code_chunk import CodeChunk
from chunking.multi_language_chunker import MultiLanguageChunker
from embeddings.embedder import EmbeddingResult
from merkle.snapshot_manager import SnapshotManager
from search.epoch_manifest import ManifestConsistencyError
from search.incremental_indexer import (
    ChunkingDiagnostics,
    IncrementalIndexer,
    IncrementalIndexResult,
)
from search.indexer import CodeIndexManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedding_result(chunk_id: str, dim: int = 8) -> EmbeddingResult:
    """Small mock embedding so tests run fast (8-dim instead of 768)."""
    return EmbeddingResult(
        embedding=np.random.RandomState(hash(chunk_id) & 0xFFFFFFFF)
            .randn(dim).astype(np.float32),
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


def _close_manager(mgr: CodeIndexManager) -> None:
    if mgr._metadata_db is not None:
        mgr._metadata_db.close()
        mgr._metadata_db = None
    if getattr(mgr, "_fts_conn", None) is not None:
        mgr._fts_conn.close()
        mgr._fts_conn = None


class _FakeEmbedder:
    """Stand-in for CodeEmbedder that returns deterministic mock embeddings.

    Skips the network entirely. Used wherever IncrementalIndexer needs to
    embed chunks but the test doesn't care about embedding quality.
    """

    def __init__(self, dim: int = 8):
        self.dim = dim
        self._model = MagicMock()
        self._model.get_embedding_dimension.return_value = dim
        self._model._model_name = "mock-embedder"

    def embed_chunks_grouped(self, chunks: List[CodeChunk], batch_size: int = 32):
        # Return one EmbeddingResult per chunk with chunk_id derived from
        # the chunk's path+line — matches what the real embedder produces.
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
def project_dir(tmp_path):
    """A small project directory with three Python files."""
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "alpha.py").write_text(
        "def alpha():\n"
        "    return 1\n"
    )
    (proj / "beta.py").write_text(
        "def beta_a():\n"
        "    return 2\n"
        "\n"
        "def beta_b():\n"
        "    return 3\n"
    )
    (proj / "gamma.py").write_text(
        "class Gamma:\n"
        "    def method(self):\n"
        "        return 4\n"
    )
    return proj


@pytest.fixture
def indexer_components(tmp_path):
    """Build the moving parts of an IncrementalIndexer with mocked embedder."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()

    indexer = CodeIndexManager(str(index_dir))
    embedder = _FakeEmbedder(dim=8)
    chunker = MultiLanguageChunker(root_path=None)  # set per test
    snapshot_manager = SnapshotManager(snapshot_dir)

    return indexer, embedder, chunker, snapshot_manager


# ---------------------------------------------------------------------------
# Manifest fatal contract (#1 from the assessment)
# ---------------------------------------------------------------------------

class TestManifestFatalContract:
    """Validates that ManifestConsistencyError propagates instead of being
    silently swallowed. The pre-#1 contract logged the error and returned
    normally, leaving callers unable to distinguish a successful save from
    one that left artifacts in an inconsistent state."""

    def test_consistency_error_raises_when_chunk_ids_diverges_from_faiss(self):
        """chunk_ids.pkl row count != FAISS ntotal is the chunk-truncation
        regression signature. Must surface as a propagated exception."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            mgr.create_index(embedding_dimension=8)
            mgr.add_embeddings([_make_embedding_result(f"f{i}") for i in range(5)])
            # Manually clobber chunk_ids so it disagrees with FAISS ntotal.
            mgr._chunk_ids = ["only_one"]

            with pytest.raises(ManifestConsistencyError):
                mgr.save_index(force=True)  # force bypasses the truncation guard

            assert mgr.last_manifest_commit_status == "consistency_error"
            _close_manager(mgr)

    def test_last_manifest_commit_status_starts_none(self):
        """Before any save, status is None — distinguishes 'no commit yet'
        from 'commit attempted'."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            assert mgr.last_manifest_commit_status is None
            _close_manager(mgr)

    def test_last_manifest_commit_status_ok_on_success(self):
        """Happy path: matched counts → status is 'ok'."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            mgr.add_embeddings([_make_embedding_result(f"f{i}") for i in range(3)])
            mgr.save_index()
            assert mgr.last_manifest_commit_status == "ok"
            _close_manager(mgr)

    def test_metadata_db_lag_does_not_trip_consistency(self):
        """remove_file_chunks intentionally leaves chunk_ids + FAISS at
        previous size while updating metadata.db + fts5.db. This legitimate
        divergence must NOT trip the consistency check — sidecars are
        excluded from strict counting by design (see indexer._commit_epoch_manifest)."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            mgr.add_embeddings([
                _make_embedding_result("a", dim=8),
                _make_embedding_result("b", dim=8),
                _make_embedding_result("c", dim=8),
            ])
            mgr.save_index()
            assert mgr.last_manifest_commit_status == "ok"

            # Remove chunks for one file — sidecars (metadata.db, fts5.db)
            # shrink, but chunk_ids + FAISS stay at 3.
            removed = mgr.remove_file_chunks("a.py")
            assert removed == 1, "test setup: remove_file_chunks should find the chunk"

            # save_index must succeed despite metadata.db count diverging.
            mgr.save_index()
            assert mgr.last_manifest_commit_status == "ok"
            _close_manager(mgr)


# ---------------------------------------------------------------------------
# Chunking diagnostics (#7 from the assessment)
# ---------------------------------------------------------------------------

class TestChunkingDiagnosticsDataclass:
    """Direct exercise of the ChunkingDiagnostics dataclass shape + math."""

    def test_zero_chunk_rate_empty(self):
        diag = ChunkingDiagnostics()
        assert diag.zero_chunk_rate == 0.0  # no div-by-zero

    def test_zero_chunk_rate_math(self):
        diag = ChunkingDiagnostics(
            files_attempted=10,
            files_with_chunks=7,
            files_zero_chunks=3,
            chunks_extracted=22,
        )
        assert diag.zero_chunk_rate == 0.3

    def test_to_dict_shape(self):
        diag = ChunkingDiagnostics(
            files_attempted=10,
            files_with_chunks=7,
            files_zero_chunks=3,
            chunks_extracted=22,
        )
        d = diag.to_dict()
        assert d["files_attempted"] == 10
        assert d["files_with_chunks"] == 7
        assert d["files_zero_chunks"] == 3
        assert d["chunks_extracted"] == 22
        assert d["zero_chunk_rate"] == 0.3


class TestChunkingDiagnosticsLive:
    """Diagnostics populated by an actual indexing run."""

    def test_diag_attached_to_full_index_result(
        self, project_dir, indexer_components
    ):
        indexer, embedder, chunker, snapshot_manager = indexer_components
        chunker.root_path = str(project_dir)
        ii = IncrementalIndexer(
            indexer=indexer, embedder=embedder, chunker=chunker,
            snapshot_manager=snapshot_manager,
        )

        result = ii.incremental_index(str(project_dir), project_name="proj")

        assert result.success, f"indexing failed: {result.error}"
        assert result.chunking_diagnostics is not None, (
            "full-index path must attach chunking_diagnostics"
        )
        diag = result.chunking_diagnostics
        # All 3 files produce ≥1 chunk.
        assert diag.files_attempted == 3
        assert diag.files_with_chunks == 3
        assert diag.files_zero_chunks == 0
        assert diag.chunks_extracted >= 3  # at least one chunk per file
        _close_manager(indexer)

    def test_diag_records_zero_chunk_files(
        self, tmp_path, indexer_components
    ):
        """A file whose chunker raises an exception (or returns []) shows
        up as a zero-chunk file in the summary."""
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "good.py").write_text("def good():\n    return 1\n")
        # File with content that produces 0 chunks: a syntactically valid
        # but empty Python file.
        (proj / "empty.py").write_text("")
        # File with content that exercises the tree-sitter path but yields
        # only a module-level chunk — count it as a "with chunks" file.
        (proj / "comments.py").write_text("# just a comment\n")

        indexer, embedder, chunker, snapshot_manager = indexer_components
        chunker.root_path = str(proj)
        ii = IncrementalIndexer(
            indexer=indexer, embedder=embedder, chunker=chunker,
            snapshot_manager=snapshot_manager,
        )

        result = ii.incremental_index(str(proj), project_name="proj")
        assert result.success
        diag = result.chunking_diagnostics
        assert diag is not None
        assert diag.files_attempted == 3
        # empty.py produces 0 chunks (tree-sitter has nothing to extract
        # and source_code.strip() is empty)
        assert diag.files_zero_chunks >= 1, (
            f"expected ≥1 zero-chunk file (empty.py), diag={diag.to_dict()}"
        )
        assert diag.files_with_chunks + diag.files_zero_chunks == 3
        _close_manager(indexer)

    def test_diag_reset_between_runs_no_changes_path(
        self, project_dir, indexer_components
    ):
        """After a successful indexing run, a second run with no file
        changes must NOT carry the previous run's diagnostics."""
        indexer, embedder, chunker, snapshot_manager = indexer_components
        chunker.root_path = str(project_dir)
        ii = IncrementalIndexer(
            indexer=indexer, embedder=embedder, chunker=chunker,
            snapshot_manager=snapshot_manager,
        )

        # First run: produces a non-None diag.
        first = ii.incremental_index(str(project_dir), project_name="proj")
        assert first.chunking_diagnostics is not None

        # Second run with no source changes: hits the no-changes branch,
        # which doesn't go through any chunking loop.
        second = ii.incremental_index(str(project_dir), project_name="proj")
        assert second.success
        assert second.chunks_added == 0
        # Stale diag must be reset to None so the caller can tell that
        # no chunking happened on this run.
        assert second.chunking_diagnostics is None, (
            "no-changes path must not carry over previous diag"
        )
        _close_manager(indexer)

    def test_diag_attached_to_incremental_result_on_modify(
        self, project_dir, indexer_components
    ):
        """A second run with a modified file goes through _add_new_chunks
        and surfaces its own fresh chunking diagnostics."""
        indexer, embedder, chunker, snapshot_manager = indexer_components
        chunker.root_path = str(project_dir)
        ii = IncrementalIndexer(
            indexer=indexer, embedder=embedder, chunker=chunker,
            snapshot_manager=snapshot_manager,
        )

        # First run: snapshots the project.
        ii.incremental_index(str(project_dir), project_name="proj")

        # Modify one file so detect_changes finds work.
        (project_dir / "alpha.py").write_text(
            "def alpha_v2():\n    return 99\n"
        )

        result = ii.incremental_index(str(project_dir), project_name="proj")
        assert result.success
        assert result.chunking_diagnostics is not None
        # Only the modified file is rechunked.
        assert result.chunking_diagnostics.files_attempted == 1
        _close_manager(indexer)


# ---------------------------------------------------------------------------
# Dispatch correctness
# ---------------------------------------------------------------------------

class TestDispatch:
    """Verifies which code path incremental_index takes under each
    precondition (no snapshot → full; force_full → full; snapshot+changes
    → incremental; snapshot+no-changes → fast-return)."""

    def test_no_snapshot_dispatches_to_full(
        self, project_dir, indexer_components, monkeypatch
    ):
        indexer, embedder, chunker, snapshot_manager = indexer_components
        chunker.root_path = str(project_dir)
        ii = IncrementalIndexer(
            indexer=indexer, embedder=embedder, chunker=chunker,
            snapshot_manager=snapshot_manager,
        )
        full_calls = []
        original = ii._full_index

        def spy(*args, **kwargs):
            full_calls.append(1)
            return original(*args, **kwargs)
        monkeypatch.setattr(ii, "_full_index", spy)

        ii.incremental_index(str(project_dir), project_name="proj")
        assert len(full_calls) == 1, "no snapshot → must call _full_index"
        _close_manager(indexer)

    def test_force_full_dispatches_to_full_even_with_snapshot(
        self, project_dir, indexer_components, monkeypatch
    ):
        indexer, embedder, chunker, snapshot_manager = indexer_components
        chunker.root_path = str(project_dir)
        ii = IncrementalIndexer(
            indexer=indexer, embedder=embedder, chunker=chunker,
            snapshot_manager=snapshot_manager,
        )

        # First run: creates a snapshot.
        ii.incremental_index(str(project_dir), project_name="proj")

        full_calls = []
        original = ii._full_index

        def spy(*args, **kwargs):
            full_calls.append(1)
            return original(*args, **kwargs)
        monkeypatch.setattr(ii, "_full_index", spy)

        ii.incremental_index(
            str(project_dir), project_name="proj", force_full=True
        )
        assert len(full_calls) == 1, (
            "force_full=True must call _full_index even when snapshot exists"
        )
        _close_manager(indexer)

    def test_no_changes_returns_zero_chunks_without_chunking(
        self, project_dir, indexer_components, monkeypatch
    ):
        indexer, embedder, chunker, snapshot_manager = indexer_components
        chunker.root_path = str(project_dir)
        ii = IncrementalIndexer(
            indexer=indexer, embedder=embedder, chunker=chunker,
            snapshot_manager=snapshot_manager,
        )

        # First run: index everything.
        first = ii.incremental_index(str(project_dir), project_name="proj")
        assert first.success
        first_chunks = first.chunks_added

        # Second run, no changes: must not re-chunk.
        chunk_calls = []
        original = chunker.chunk_file

        def spy(*args, **kwargs):
            chunk_calls.append(args[0] if args else kwargs.get("file_path"))
            return original(*args, **kwargs)
        monkeypatch.setattr(chunker, "chunk_file", spy)

        second = ii.incremental_index(str(project_dir), project_name="proj")
        assert second.success
        assert second.chunks_added == 0
        assert second.chunks_removed == 0
        assert len(chunk_calls) == 0, (
            "no-changes path must skip the chunking loop entirely"
        )
        _close_manager(indexer)
