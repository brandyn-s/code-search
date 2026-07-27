"""Tests for Plan-2 E2: save_index commits an epoch manifest.

The E1 primitive (PR #111) and E3 reader (PR #114) shipped without
production write-path adoption. This test pins that save_index now
produces a committed manifest covering the just-written artifacts —
the keystone change that makes the primitive load-bearing instead of
dead code.

The cross-artifact consistency check at build_manifest time
structurally prevents the chunk-truncation regression class:
if FAISS ntotal disagrees with len(chunk_ids), commit_manifest fails
loudly and the previous epoch's manifest stays current.
"""

import tempfile
import numpy as np
import pytest

from search.indexer import CodeIndexManager, IndexPublicationRefused
from search.epoch_manifest import read_current, verify_manifest, ManifestMissing
from embeddings.embedder import (
    EffectiveEmbeddingConfig,
    EmbeddingResult,
)


def _make_result(
    chunk_id: str,
    content: str,
    file_path: str = "test.py",
    dimension: int = 384,
) -> EmbeddingResult:
    return EmbeddingResult(
        embedding=np.random.randn(dimension).astype(np.float32),
        chunk_id=chunk_id,
        metadata={
            "file_path": file_path,
            "relative_path": file_path,
            "content_preview": content,
            "full_content": content,
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 10,
            "name": chunk_id.split(":")[-1] if ":" in chunk_id else None,
            "parent_name": None,
            "docstring": None,
            "decorators": [],
            "imports": [],
            "complexity_score": 1,
            "tags": [],
            "folder_structure": [],
        },
    )


def _close_manager(mgr):
    if mgr._metadata_db is not None:
        mgr._metadata_db.close()
        mgr._metadata_db = None
    if hasattr(mgr, "_fts_conn") and mgr._fts_conn is not None:
        mgr._fts_conn.close()
        mgr._fts_conn = None


def test_save_index_commits_manifest():
    """After save_index, current.json exists and verifies clean."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CodeIndexManager(tmpdir)
        mgr.add_embeddings([
            _make_result("a.py:1-10:func:foo", "def foo(): pass"),
            _make_result("b.py:1-10:func:bar", "def bar(): pass"),
        ])
        mgr.save_index()

        # Manifest committed.
        from pathlib import Path
        manifest = read_current(Path(tmpdir))
        assert "epoch_id" in manifest
        assert "artifacts" in manifest
        assert manifest["consistency"].get("all_artifacts_share_count") is True

        # Manifest verifies — actual artifacts match recorded SHAs/counts.
        err = verify_manifest(Path(tmpdir), manifest)
        assert err is None, f"manifest verification failed: {err}"

        _close_manager(mgr)


def test_save_index_records_chunk_ids_count_in_manifest():
    """The committed manifest records the chunk_ids count we just wrote."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CodeIndexManager(tmpdir)
        mgr.add_embeddings([
            _make_result(f"f{i}.py:1-10:func:fn{i}", f"def fn{i}(): pass")
            for i in range(5)
        ])
        mgr.save_index()

        from pathlib import Path
        manifest = read_current(Path(tmpdir))

        # chunk_ids.pkl should be in artifacts with count=5.
        artifacts = manifest["artifacts"]
        assert "chunk_ids.pkl" in artifacts
        assert artifacts["chunk_ids.pkl"]["count"] == 5

        # FAISS index also recorded.
        assert "code.index" in artifacts
        assert artifacts["code.index"]["count"] == 5

        _close_manager(mgr)


def test_save_index_commits_bound_effective_embedding_identity():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CodeIndexManager(tmpdir)
        configuration = EffectiveEmbeddingConfig(
            provider="openai",
            model_name="text-embedding-custom",
            content_mode="code",
            output_dimension=7,
        )
        mgr.bind_embedding_configuration(
            configuration,
            pipeline_version="effective-pipeline-version",
        )
        mgr.add_embeddings(
            [
                _make_result(
                    "a.py:1-10:func:foo",
                    "def foo(): pass",
                    dimension=7,
                )
            ]
        )
        mgr.save_index()

        from pathlib import Path

        manifest = read_current(Path(tmpdir))
        assert manifest["provider"] == "openai"
        assert manifest["model"] == "text-embedding-custom"
        assert manifest["vector_dim"] == 7
        assert (
            manifest["pipeline_version"]
            == "effective-pipeline-version"
        )

        _close_manager(mgr)


def test_save_index_rejects_bound_dimension_that_disagrees_with_faiss():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CodeIndexManager(tmpdir)
        mgr.bind_embedding_configuration(
            EffectiveEmbeddingConfig(
                provider="openai",
                model_name="text-embedding-custom",
                content_mode="code",
                output_dimension=8,
            ),
            pipeline_version="effective-pipeline-version",
        )
        mgr.add_embeddings(
            [
                _make_result(
                    "a.py:1-10:func:foo",
                    "def foo(): pass",
                    dimension=7,
                )
            ]
        )

        with pytest.raises(
            IndexPublicationRefused,
            match="configured embedding dimension 8 does not match",
        ):
            mgr.save_index()

        from pathlib import Path

        assert not (Path(tmpdir) / "current.json").exists()
        _close_manager(mgr)


def test_save_index_promotes_prior_on_second_save():
    """Two consecutive save_index calls populate prior.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CodeIndexManager(tmpdir)
        mgr.add_embeddings([
            _make_result("a.py:1-10:func:foo", "def foo(): pass"),
        ])
        mgr.save_index()

        from pathlib import Path
        first_manifest = read_current(Path(tmpdir))
        first_epoch = first_manifest["epoch_id"]

        # Second save with more data.
        mgr.add_embeddings([
            _make_result("b.py:1-10:func:bar", "def bar(): pass"),
            _make_result("c.py:1-10:func:baz", "def baz(): pass"),
        ])
        mgr.save_index()

        second_manifest = read_current(Path(tmpdir))
        second_epoch = second_manifest["epoch_id"]
        assert second_epoch != first_epoch, "second save should produce new epoch_id"

        # prior.json now holds the first epoch.
        from search.epoch_manifest import read_prior
        prior_manifest = read_prior(Path(tmpdir))
        assert prior_manifest is not None
        assert prior_manifest["epoch_id"] == first_epoch

        _close_manager(mgr)


def test_save_index_with_empty_index_skips_manifest():
    """An empty index (no embeddings) shouldn't crash; no manifest needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CodeIndexManager(tmpdir)
        # No add_embeddings — try save on empty index.
        mgr.save_index()

        # No manifest committed (or graceful skip; either is acceptable).
        from pathlib import Path
        try:
            manifest = read_current(Path(tmpdir))
            # If a manifest was written, it should at least not be inconsistent.
            assert "epoch_id" in manifest
        except ManifestMissing:
            pass  # expected — empty index, nothing to commit

        _close_manager(mgr)
