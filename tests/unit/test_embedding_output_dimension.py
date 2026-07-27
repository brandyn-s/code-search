"""Fail-closed tests for the effective embedding dimension contract."""

from __future__ import annotations

import numpy as np
import pytest

from chunking.code_chunk import CodeChunk
from embeddings import embedder as embedder_module
from embeddings.embedder import (
    CodeEmbedder,
    EffectiveEmbeddingConfig,
    register_provider,
)


class _WrongDimensionModel:
    _model_name = "wrong-dimension-model"

    def encode(self, texts, **_kwargs):
        return np.ones((len(texts), 2), dtype=np.float32)

    def get_embedding_dimension(self) -> int:
        return 2

    def get_model_info(self) -> dict[str, object]:
        return {}

    def cleanup(self) -> None:
        return None


class _WrongGroupedDimensionModel(_WrongDimensionModel):
    def encode_grouped(self, grouped_texts, **_kwargs):
        count = sum(len(group) for group in grouped_texts)
        return np.ones((count, 2), dtype=np.float32)


def _chunk() -> CodeChunk:
    return CodeChunk(
        content="def example():\n    return 1",
        chunk_type="function",
        start_line=1,
        end_line=2,
        file_path="/project/example.py",
        relative_path="example.py",
        folder_structure=[],
        name="example",
    )


def _embedder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    provider: str,
    model,
) -> CodeEmbedder:
    @register_provider(provider)
    def _factory(_model_name: str, _cache_dir: str, _device: str):
        return model

    embedder = CodeEmbedder(
        cache_dir=str(tmp_path),
        configuration=EffectiveEmbeddingConfig(
            provider=provider,
            model_name="dimension-contract-model",
            content_mode="code",
            output_dimension=3,
        ),
    )
    monkeypatch.setattr(embedder, "_get_disk_cache", lambda: None)
    return embedder


def test_query_output_must_match_effective_dimension(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    provider = "__wrong-query-dimension"
    CodeEmbedder._query_cache.clear()
    embedder = _embedder(
        monkeypatch,
        tmp_path,
        provider=provider,
        model=_WrongDimensionModel(),
    )

    try:
        with pytest.raises(
            ValueError,
            match="expected 3 dimensions.*received 2",
        ):
            embedder.embed_query("dimension contract")
    finally:
        CodeEmbedder._query_cache.clear()
        embedder_module._PROVIDER_REGISTRY.pop(provider, None)


def test_document_output_must_match_effective_dimension_before_indexing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    provider = "__wrong-document-dimension"
    embedder = _embedder(
        monkeypatch,
        tmp_path,
        provider=provider,
        model=_WrongDimensionModel(),
    )

    try:
        with pytest.raises(
            ValueError,
            match="expected 3 dimensions.*received 2",
        ):
            embedder.embed_chunks([_chunk()])
    finally:
        embedder_module._PROVIDER_REGISTRY.pop(provider, None)


def test_grouped_output_must_match_effective_dimension_before_indexing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    provider = "__wrong-grouped-dimension"
    embedder = _embedder(
        monkeypatch,
        tmp_path,
        provider=provider,
        model=_WrongGroupedDimensionModel(),
    )

    try:
        with pytest.raises(
            ValueError,
            match="expected 3 dimensions.*received 2",
        ):
            embedder.embed_chunks_grouped([_chunk()])
    finally:
        embedder_module._PROVIDER_REGISTRY.pop(provider, None)
