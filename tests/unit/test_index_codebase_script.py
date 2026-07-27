"""Contracts for the tracked standalone indexing script."""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embeddings.embedder import EffectiveEmbeddingConfig, EmbeddingResult
from scripts import index_codebase
from search.epoch_manifest import read_current, read_with_fallback
from search.indexer import CodeIndexManager, IndexPublicationRefused


class _RecordingManager:
    def __init__(self, storage_dir: Path | None = None) -> None:
        self.events: list[object] = []
        self.storage_dir = storage_dir
        self.index_path = (
            storage_dir / "code.index"
            if storage_dir is not None
            else Path("code.index")
        )

    @contextmanager
    def publication_transaction(self):
        self.events.append("transaction-enter")
        yield
        self.events.append("transaction-exit")

    def bind_embedding_configuration(
        self,
        configuration,
        *,
        pipeline_version,
    ) -> None:
        self.events.append(
            ("bind", configuration, pipeline_version)
        )

    def begin_rebuild(self) -> None:
        self.events.append("begin-rebuild")

    def add_embeddings(self, embedding_results) -> None:
        self.events.append(("add", embedding_results))

    def save_index(self) -> None:
        self.events.append("save")

    def has_persisted_index_state(self) -> bool:
        return self.index_path.exists()


def test_standalone_publisher_binds_identity_and_replaces_atomically(
    monkeypatch,
) -> None:
    configuration = EffectiveEmbeddingConfig(
        provider="voyage",
        model_name="voyage-code-3",
        content_mode="code",
        output_dimension=1024,
    )
    embedder = SimpleNamespace(configuration=configuration)
    manager = _RecordingManager()
    embeddings = [object()]
    monkeypatch.setattr(
        index_codebase,
        "get_pipeline_version",
        lambda effective: (
            "pipeline-v2"
            if effective is configuration
            else "wrong-pipeline"
        ),
    )

    index_codebase._publish_embeddings(
        manager,
        embedder,
        embeddings,
        replace=True,
    )

    assert manager.events == [
        "transaction-enter",
        ("bind", configuration, "pipeline-v2"),
        "begin-rebuild",
        ("add", embeddings),
        "save",
        "transaction-exit",
    ]


def test_standalone_publisher_preserves_append_mode(
    tmp_path,
    monkeypatch,
) -> None:
    configuration = EffectiveEmbeddingConfig(
        provider="local",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        content_mode="code",
        output_dimension=384,
    )
    embedder = SimpleNamespace(configuration=configuration)
    manager = _RecordingManager(tmp_path)
    monkeypatch.setattr(
        index_codebase,
        "get_pipeline_version",
        lambda _effective: "pipeline-local",
    )

    index_codebase._publish_embeddings(
        manager,
        embedder,
        [],
        replace=False,
    )

    assert "begin-rebuild" not in manager.events
    assert manager.events[1] == (
        "bind",
        configuration,
        "pipeline-local",
    )


def _embedding(chunk_id: str, dimension: int = 4) -> EmbeddingResult:
    return EmbeddingResult(
        embedding=np.ones(dimension, dtype=np.float32),
        chunk_id=chunk_id,
        metadata={
            "file_path": "test.py",
            "relative_path": "test.py",
            "content_preview": chunk_id,
            "full_content": chunk_id,
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 1,
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


def test_standalone_append_requires_persisted_effective_identity(
    tmp_path,
    monkeypatch,
) -> None:
    identity_a = EffectiveEmbeddingConfig(
        provider="openai",
        model_name="model-a",
        content_mode="code",
        output_dimension=4,
    )
    identity_b = EffectiveEmbeddingConfig(
        provider="voyage",
        model_name="model-b",
        content_mode="docs",
        output_dimension=4,
    )
    pipeline_versions = {
        identity_a: "pipeline-a",
        identity_b: "pipeline-b",
    }
    monkeypatch.setattr(
        index_codebase,
        "get_pipeline_version",
        pipeline_versions.__getitem__,
    )
    manager = CodeIndexManager(str(tmp_path))

    try:
        # A brand-new empty index has no identity to conflict with.
        index_codebase._publish_embeddings(
            manager,
            SimpleNamespace(configuration=identity_a),
            [],
            replace=False,
        )
        assert read_with_fallback(tmp_path).freshness == "missing"

        # First publication and a compatible append both remain supported.
        index_codebase._publish_embeddings(
            manager,
            SimpleNamespace(configuration=identity_a),
            [_embedding("a:first")],
            replace=False,
        )
        index_codebase._publish_embeddings(
            manager,
            SimpleNamespace(configuration=identity_a),
            [_embedding("a:second")],
            replace=False,
        )
        last_good = read_current(tmp_path)
        assert last_good["consistency"]["expected_count"] == 2

        # Same-dimensional vectors from another effective pipeline must not
        # be mixed into the already-published generation.
        with pytest.raises(
            IndexPublicationRefused,
            match="effective embedding identity",
        ):
            index_codebase._publish_embeddings(
                manager,
                SimpleNamespace(configuration=identity_b),
                [_embedding("b:must-not-publish")],
                replace=False,
            )

        publication = read_with_fallback(tmp_path)
        assert publication.freshness == "fresh"
        assert publication.manifest == last_good
        assert publication.manifest["provider"] == identity_a.provider
        assert publication.manifest["model"] == identity_a.model_name
        assert publication.manifest["vector_dim"] == 4
        assert publication.manifest["pipeline_version"] == "pipeline-a"
        assert publication.manifest["consistency"]["expected_count"] == 2
    finally:
        manager._close_storage_handles()


def test_standalone_append_refuses_stale_prior_identity(
    tmp_path,
    monkeypatch,
) -> None:
    identity_a = EffectiveEmbeddingConfig(
        provider="openai",
        model_name="model-a",
        content_mode="code",
        output_dimension=4,
    )
    identity_b = EffectiveEmbeddingConfig(
        provider="voyage",
        model_name="model-b",
        content_mode="docs",
        output_dimension=4,
    )
    pipeline_versions = {
        identity_a: "pipeline-a",
        identity_b: "pipeline-b",
    }
    monkeypatch.setattr(
        index_codebase,
        "get_pipeline_version",
        pipeline_versions.__getitem__,
    )
    manager = CodeIndexManager(str(tmp_path))

    try:
        index_codebase._publish_embeddings(
            manager,
            SimpleNamespace(configuration=identity_a),
            [_embedding("a-prior")],
            replace=True,
        )
        index_codebase._publish_embeddings(
            manager,
            SimpleNamespace(configuration=identity_b),
            [_embedding("b-current")],
            replace=True,
        )
        (tmp_path / "manifest" / "current.json").write_text(
            "{not-json",
            encoding="utf-8",
        )
        fallback = read_with_fallback(tmp_path)
        assert fallback.freshness == "stale_using_prior_epoch"
        assert fallback.manifest["model"] == "model-a"

        with pytest.raises(
            IndexPublicationRefused,
            match="verified current generation",
        ):
            index_codebase._publish_embeddings(
                manager,
                SimpleNamespace(configuration=identity_a),
                [_embedding("a-must-not-mix")],
                replace=False,
            )

        _ = manager.index
        assert manager._chunk_ids == ["a-prior"]
        assert manager.metadata_db.get("a-must-not-mix") is None
        assert manager.search_bm25("a-must-not-mix", k=10) == []
    finally:
        manager._close_storage_handles()


@pytest.mark.parametrize("surviving_sidecar", ["metadata.db", "fts5.db"])
def test_standalone_first_append_refuses_residual_sidecar_state(
    tmp_path,
    monkeypatch,
    surviving_sidecar,
) -> None:
    identity_a = EffectiveEmbeddingConfig(
        provider="openai",
        model_name="model-a",
        content_mode="code",
        output_dimension=4,
    )
    identity_b = EffectiveEmbeddingConfig(
        provider="voyage",
        model_name="model-b",
        content_mode="docs",
        output_dimension=4,
    )
    monkeypatch.setattr(
        index_codebase,
        "get_pipeline_version",
        lambda configuration: f"pipeline-{configuration.model_name}",
    )
    original = CodeIndexManager(str(tmp_path))
    index_codebase._publish_embeddings(
        original,
        SimpleNamespace(configuration=identity_a),
        [_embedding("stale-a")],
        replace=True,
    )
    original._close_storage_handles()

    for artifact in (
        "code.index",
        "chunk_ids.pkl",
        "metadata.db",
        "fts5.db",
        "stats.json",
        "float_store.npy",
    ):
        if artifact != surviving_sidecar:
            (tmp_path / artifact).unlink(missing_ok=True)
    shutil.rmtree(tmp_path / "manifest")
    shutil.rmtree(tmp_path / ".generations")

    manager = CodeIndexManager(str(tmp_path))
    try:
        assert read_with_fallback(tmp_path).freshness == "missing"
        with pytest.raises(
            IndexPublicationRefused,
            match="residual index state",
        ):
            index_codebase._publish_embeddings(
                manager,
                SimpleNamespace(configuration=identity_b),
                [_embedding("fresh-b")],
                replace=False,
            )

        if surviving_sidecar == "metadata.db":
            assert manager.metadata_db.get("stale-a") is not None
            assert manager.metadata_db.get("fresh-b") is None
        else:
            rows = manager._fts_conn.execute(
                "SELECT chunk_id FROM chunk_fts ORDER BY chunk_id"
            ).fetchall()
            assert rows == [("stale-a",)]
    finally:
        manager._close_storage_handles()
