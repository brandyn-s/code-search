"""Tests for _full_index's partial-failure contract.

Pins the 2026-06-10 fix: when an embedding batch fails during a full
index, the run must report success=False and must NOT advance the
snapshot. Pre-fix, failures were logged and the snapshot saved anyway —
the missing chunks' files were recorded as indexed and never retried
(the same silent data-loss shape the incremental path's
embed-before-remove ordering was built to prevent).

Also pins the Voyage Batch API metadata contract: results built on that
path must carry `full_content` (they're routed through
_make_embedding_result now). The previous inline metadata block omitted
it, so Batch-API-built indexes fed only the 200-char preview to FTS5/BM25.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pytest

from chunking.code_chunk import CodeChunk
from chunking.multi_language_chunker import MultiLanguageChunker
from embeddings.embedder import EmbeddingResult
from merkle.snapshot_manager import SnapshotManager
from search.incremental_indexer import IncrementalIndexer
from search.indexer import CodeIndexManager


class _FailingEmbedder:
    """Embedder whose embed_chunks_grouped always raises (transient API 5xx)."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self._model = _FakeModelHandle(dim)

    def embed_chunks_grouped(self, chunks: List[CodeChunk], batch_size: int = 32):
        raise RuntimeError("simulated embedding API failure")

    def create_embedding_content(self, chunk: CodeChunk) -> str:
        return chunk.content


class _FakeModelHandle:
    def __init__(self, dim: int, model_name: str = "mock-embedder"):
        self.dim = dim
        self._model_name = model_name

    def get_embedding_dimension(self):
        return self.dim


class _WorkingEmbedder:
    """Minimal deterministic embedder for the happy-path contrast test."""

    def __init__(self, dim: int = 8, model_name: str = "mock-embedder"):
        self.dim = dim
        self._model = _FakeModelHandle(dim, model_name)

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


def _close_manager(mgr: CodeIndexManager) -> None:
    if mgr._metadata_db is not None:
        mgr._metadata_db.close()
        mgr._metadata_db = None
    if getattr(mgr, "_fts_conn", None) is not None:
        mgr._fts_conn.close()
        mgr._fts_conn = None


@pytest.fixture
def project_dir(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "alpha.py").write_text("def alpha():\n    return 1\n")
    (proj / "beta.py").write_text("def beta():\n    return 2\n")
    return proj


@pytest.fixture
def parts(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    indexer = CodeIndexManager(str(index_dir))
    snapshot_manager = SnapshotManager(snapshot_dir)
    return indexer, snapshot_manager


def test_embed_batch_failure_reports_failure_and_skips_snapshot(
    project_dir, parts
):
    indexer, snapshot_manager = parts
    chunker = MultiLanguageChunker(root_path=str(project_dir))
    ii = IncrementalIndexer(
        indexer=indexer, embedder=_FailingEmbedder(), chunker=chunker,
        snapshot_manager=snapshot_manager,
    )

    result = ii.incremental_index(str(project_dir), project_name="proj")

    assert result.success is False, (
        "a full index with failed embedding batches must not report success"
    )
    assert result.error and "batch" in result.error.lower()
    assert not snapshot_manager.has_snapshot(str(project_dir)), (
        "snapshot must NOT advance after a partial-failure full index — "
        "the missing chunks would never be retried"
    )
    _close_manager(indexer)


def test_healthy_full_index_still_saves_snapshot(project_dir, parts):
    indexer, snapshot_manager = parts
    chunker = MultiLanguageChunker(root_path=str(project_dir))
    ii = IncrementalIndexer(
        indexer=indexer, embedder=_WorkingEmbedder(), chunker=chunker,
        snapshot_manager=snapshot_manager,
    )

    result = ii.incremental_index(str(project_dir), project_name="proj")

    assert result.success, f"healthy run failed: {result.error}"
    assert result.chunks_added > 0
    assert snapshot_manager.has_snapshot(str(project_dir))
    _close_manager(indexer)


def test_batch_api_path_metadata_carries_full_content(
    project_dir, parts, monkeypatch
):
    """The Batch API path must produce the same metadata shape as every
    other embed path (single source of truth: _make_embedding_result)."""
    indexer, snapshot_manager = parts
    chunker = MultiLanguageChunker(root_path=str(project_dir))
    embedder = _WorkingEmbedder(model_name="voyage-4-large")

    class _StubBatchEmbedder:
        def __init__(self, model: str):
            self.model = model

        def embed_all(self, contents, input_type="document"):
            return np.random.RandomState(0).randn(len(contents), 8).astype(np.float32)

        def close(self):
            pass

    import embeddings.voyage_batch_embedder as vbe
    monkeypatch.setattr(vbe, "VoyageBatchEmbedder", _StubBatchEmbedder)
    monkeypatch.setenv("VOYAGE_BATCH_API", "on")
    monkeypatch.setenv("VOYAGE_BATCH_THRESHOLD", "1")

    ii = IncrementalIndexer(
        indexer=indexer, embedder=embedder, chunker=chunker,
        snapshot_manager=snapshot_manager,
    )
    result = ii.incremental_index(str(project_dir), project_name="proj")
    assert result.success, f"batch-path run failed: {result.error}"
    assert result.chunks_added > 0

    for chunk_id in indexer._chunk_ids:
        metadata = indexer.get_chunk_by_id(chunk_id)
        assert metadata is not None
        assert metadata.get("full_content"), (
            f"{chunk_id} missing full_content — FTS5 would index only the "
            "200-char preview"
        )
        assert metadata.get("project_name") == "proj"
    _close_manager(indexer)
