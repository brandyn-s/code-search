"""Tests for the single resolved embedding configuration contract."""

from __future__ import annotations

import os

import pytest

from embeddings.embedder import (
    CodeEmbedder,
    EffectiveEmbeddingConfig,
    register_provider,
    resolve_embedding_config,
)


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        (
            {},
            EffectiveEmbeddingConfig(
                provider="local",
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                content_mode="code",
                output_dimension=384,
            ),
        ),
        (
            {"VOYAGE_API_KEY": "test-key"},
            EffectiveEmbeddingConfig(
                provider="voyage",
                model_name="voyage-4-large",
                content_mode="code",
                output_dimension=1024,
            ),
        ),
        (
            {"VOYAGE_API_KEY": "test-key", "CONTENT_MODE": "docs"},
            EffectiveEmbeddingConfig(
                provider="voyage-context",
                model_name="voyage-context-3",
                content_mode="docs",
                output_dimension=1024,
            ),
        ),
        (
            {
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "text-embedding-3-large",
                "VOYAGE_API_KEY": "ignored",
            },
            EffectiveEmbeddingConfig(
                provider="openai",
                model_name="text-embedding-3-large",
                content_mode="code",
                output_dimension=3072,
            ),
        ),
        (
            {
                "EMBEDDING_PROVIDER": "local",
                "LOCAL_EMBEDDING_MODEL": "local/custom-model",
            },
            EffectiveEmbeddingConfig(
                provider="local",
                model_name="local/custom-model",
                content_mode="code",
            ),
        ),
    ],
)
def test_resolves_ambient_embedding_configuration(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    expected: EffectiveEmbeddingConfig,
) -> None:
    for name in (
        "CONTENT_MODE",
        "EMBEDDING_DIMENSION",
        "EMBEDDING_MODEL",
        "EMBEDDING_PROVIDER",
        "JINA_TRUNCATE_DIM",
        "LOCAL_EMBEDDING_MODEL",
        "OPENAI_API_KEY",
        "VOYAGE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    assert resolve_embedding_config() == expected


@pytest.mark.parametrize(
    ("stored", "overrides", "expected"),
    [
        (
            {
                "embedding_provider": "voyage-context",
                "embedding_model": "voyage-context-3",
                "content_mode": "docs",
            },
            {},
            EffectiveEmbeddingConfig(
                provider="voyage-context",
                model_name="voyage-context-3",
                content_mode="docs",
                output_dimension=1024,
            ),
        ),
        (
            {
                "embedding_provider": "voyage",
                "embedding_model": "voyage-4-large",
                "content_mode": "docs",
            },
            {"provider": "local"},
            EffectiveEmbeddingConfig(
                provider="local",
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                content_mode="docs",
                output_dimension=384,
            ),
        ),
        (
            {
                "embedding_provider": "voyage-code-3",
                "embedding_model": "voyage-code-3",
                "content_mode": "code",
            },
            {"provider": "voyage-code-3"},
            EffectiveEmbeddingConfig(
                provider="voyage-code-3",
                model_name="voyage-code-3",
                content_mode="code",
                output_dimension=1024,
            ),
        ),
        (
            {
                "embedding_provider": "voyage",
                "embedding_model": "voyage-4-large",
                "content_mode": "code",
            },
            {
                "provider": "openai",
                "model_name": "text-embedding-3-large",
                "content_mode": "docs",
            },
            EffectiveEmbeddingConfig(
                provider="openai",
                model_name="text-embedding-3-large",
                content_mode="docs",
                output_dimension=3072,
            ),
        ),
    ],
)
def test_stored_and_per_call_configuration_precedence(
    monkeypatch: pytest.MonkeyPatch,
    stored: dict[str, str],
    overrides: dict[str, str],
    expected: EffectiveEmbeddingConfig,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "ambient-openai-model")
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "local/ambient-model")
    monkeypatch.setenv("CONTENT_MODE", "code")

    assert resolve_embedding_config(stored=stored, **overrides) == expected


def test_per_call_provider_does_not_inherit_different_ambient_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")

    assert resolve_embedding_config(
        provider="voyage",
    ) == EffectiveEmbeddingConfig(
        provider="voyage",
        model_name="voyage-4-large",
        content_mode="code",
        output_dimension=1024,
    )


def test_embedder_uses_explicit_config_and_records_actual_dimension(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured: dict[str, str] = {}

    class _SevenDimensionalModel:
        _model_name = "actual-seven-dimensional-model"

        def encode(self, texts, **_kwargs):
            raise AssertionError("encoding is not part of configuration")

        def get_embedding_dimension(self) -> int:
            return 7

        def get_model_info(self) -> dict[str, object]:
            return {}

        def cleanup(self) -> None:
            return None

    @register_provider("__effective-config-test")
    def _factory(model_name: str, _cache_dir: str, _device: str):
        captured["model_name"] = model_name
        return _SevenDimensionalModel()

    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_MODEL", "ambient-model-must-not-win")
    requested = EffectiveEmbeddingConfig(
        provider="__effective-config-test",
        model_name="requested-seven-dimensional-model",
        content_mode="docs",
        output_dimension=None,
    )

    try:
        embedder = CodeEmbedder(
            cache_dir=str(tmp_path),
            configuration=requested,
        )
    finally:
        from embeddings import embedder as embedder_module

        embedder_module._PROVIDER_REGISTRY.pop(
            "__effective-config-test",
            None,
        )

    assert captured["model_name"] == "requested-seven-dimensional-model"
    assert embedder.configuration == EffectiveEmbeddingConfig(
        provider="__effective-config-test",
        model_name="actual-seven-dimensional-model",
        content_mode="docs",
        output_dimension=7,
    )
    assert os.environ["EMBEDDING_PROVIDER"] == "local"
    assert os.environ["EMBEDDING_MODEL"] == "ambient-model-must-not-win"


@pytest.mark.parametrize(
    ("provider", "model_name"),
    [
        ("__not-registered", "some-model"),
        ("openai", "voyage-4-large"),
        ("voyage", "text-embedding-3-small"),
        ("voyage-context", "voyage-4-large"),
        ("gemma", "google/not-embedding-gemma"),
    ],
)
def test_rejects_unknown_provider_model_combinations(
    provider: str,
    model_name: str,
) -> None:
    with pytest.raises(ValueError, match="provider/model"):
        resolve_embedding_config(
            provider=provider,
            model_name=model_name,
        )


def test_custom_remote_model_accepts_explicit_dimension_contract() -> None:
    assert resolve_embedding_config(
        provider="openai",
        model_name="text-embedding-custom",
        output_dimension=2048,
    ) == EffectiveEmbeddingConfig(
        provider="openai",
        model_name="text-embedding-custom",
        content_mode="code",
        output_dimension=2048,
    )


def test_ambient_custom_remote_model_uses_dimension_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("EMBEDDING_MODEL", "voyage-code-3-lite")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1536")

    assert resolve_embedding_config() == EffectiveEmbeddingConfig(
        provider="voyage",
        model_name="voyage-code-3-lite",
        content_mode="code",
        output_dimension=1536,
    )


@pytest.mark.parametrize(
    ("ambient_provider", "overrides", "expected"),
    [
        (
            "openai",
            {},
            EffectiveEmbeddingConfig(
                provider="openai",
                model_name="text-embedding-custom",
                content_mode="code",
                output_dimension=2048,
            ),
        ),
        (
            "openai",
            {"provider": "openai"},
            EffectiveEmbeddingConfig(
                provider="openai",
                model_name="text-embedding-custom",
                content_mode="code",
                output_dimension=2048,
            ),
        ),
        (
            None,
            {"provider": "openai"},
            EffectiveEmbeddingConfig(
                provider="openai",
                model_name="text-embedding-custom",
                content_mode="code",
                output_dimension=2048,
            ),
        ),
        (
            "openai",
            {
                "provider": "openai",
                "model_name": "text-embedding-custom",
            },
            EffectiveEmbeddingConfig(
                provider="openai",
                model_name="text-embedding-custom",
                content_mode="code",
                output_dimension=2048,
            ),
        ),
        (
            "openai",
            {
                "provider": "openai",
                "model_name": "explicit-custom-model",
                "output_dimension": 768,
            },
            EffectiveEmbeddingConfig(
                provider="openai",
                model_name="explicit-custom-model",
                content_mode="code",
                output_dimension=768,
            ),
        ),
        (
            "openai",
            {"output_dimension": 768},
            EffectiveEmbeddingConfig(
                provider="openai",
                model_name="text-embedding-custom",
                content_mode="code",
                output_dimension=768,
            ),
        ),
        (
            "openai",
            {"provider": "voyage"},
            EffectiveEmbeddingConfig(
                provider="voyage",
                model_name="voyage-4-large",
                content_mode="code",
                output_dimension=1024,
            ),
        ),
    ],
    ids=(
        "all-ambient",
        "explicit-matching-provider",
        "explicit-provider-without-ambient-provider",
        "explicit-matching-provider-model",
        "all-explicit",
        "explicit-dimension",
        "explicit-mismatched-provider",
    ),
)
def test_provider_model_dimension_precedence_matrix(
    monkeypatch: pytest.MonkeyPatch,
    ambient_provider: str | None,
    overrides: dict[str, object],
    expected: EffectiveEmbeddingConfig,
) -> None:
    if ambient_provider is None:
        monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("EMBEDDING_PROVIDER", ambient_provider)
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-custom")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "2048")

    assert resolve_embedding_config(**overrides) == expected


def test_explicit_model_does_not_inherit_mismatched_ambient_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "ambient-custom-model")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "2048")

    with pytest.raises(ValueError, match="provider/model"):
        resolve_embedding_config(
            provider="openai",
            model_name="explicit-custom-model",
        )


def test_matching_stored_custom_model_reuses_dimension_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")

    assert resolve_embedding_config(
        stored={
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-custom",
            "embedding_dimension": 768,
        },
    ) == EffectiveEmbeddingConfig(
        provider="openai",
        model_name="text-embedding-custom",
        content_mode="code",
        output_dimension=768,
    )


@pytest.mark.parametrize("output_dimension", [True, 0, -1, "not-an-integer"])
def test_rejects_invalid_explicit_dimension_contract(
    output_dimension: object,
) -> None:
    with pytest.raises(ValueError, match="output dimension"):
        resolve_embedding_config(
            provider="openai",
            model_name="text-embedding-custom",
            output_dimension=output_dimension,
        )


def test_rejects_stored_dimension_for_a_different_model() -> None:
    with pytest.raises(ValueError, match="provider/model"):
        resolve_embedding_config(
            provider="openai",
            model_name="another-custom-model",
            stored={
                "embedding_provider": "openai",
                "embedding_model": "text-embedding-custom",
                "embedding_dimension": 768,
            },
        )


def test_rejects_dimension_that_conflicts_with_known_model_contract() -> None:
    with pytest.raises(ValueError, match="known output dimension"):
        resolve_embedding_config(
            provider="openai",
            model_name="text-embedding-3-large",
            output_dimension=2048,
        )


def test_jina_matryoshka_dimension_round_trips_from_stored_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "jina")
    monkeypatch.setenv("JINA_TRUNCATE_DIM", "512")

    first = resolve_embedding_config()

    assert first == EffectiveEmbeddingConfig(
        provider="jina",
        model_name="jinaai/jina-code-embeddings-0.5b",
        content_mode="code",
        output_dimension=512,
    )

    monkeypatch.delenv("EMBEDDING_PROVIDER")
    monkeypatch.delenv("JINA_TRUNCATE_DIM")
    restored = resolve_embedding_config(
        stored={
            "embedding_provider": first.provider,
            "embedding_model": first.model_name,
            "embedding_dimension": first.output_dimension,
            "content_mode": first.content_mode,
        },
    )

    assert restored == first


@pytest.mark.parametrize(
    ("model_name", "dimension"),
    [
        ("jinaai/jina-code-embeddings-0.5b", 65),
        ("jinaai/jina-code-embeddings-1.5b", 896),
    ],
)
def test_rejects_unsupported_jina_matryoshka_dimension(
    model_name: str,
    dimension: int,
) -> None:
    with pytest.raises(ValueError, match="Matryoshka"):
        resolve_embedding_config(
            provider="jina",
            model_name=model_name,
            output_dimension=dimension,
        )


def test_jina_factory_uses_persisted_dimension_without_ambient_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from embeddings import jina_code_embedder as jina_module

    captured: dict[str, object] = {}

    class _ConfiguredJina:
        def __init__(
            self,
            *,
            model_name: str,
            cache_dir: str,
            device: str,
            truncate_dim: int | None,
        ) -> None:
            captured.update(
                {
                    "model_name": model_name,
                    "cache_dir": cache_dir,
                    "device": device,
                    "truncate_dim": truncate_dim,
                }
            )
            self.model_name = model_name

    monkeypatch.delenv("JINA_TRUNCATE_DIM", raising=False)
    monkeypatch.setattr(jina_module, "JinaCodeEmbedder", _ConfiguredJina)
    configuration = EffectiveEmbeddingConfig(
        provider="jina",
        model_name="jinaai/jina-code-embeddings-0.5b",
        content_mode="code",
        output_dimension=512,
    )

    embedder = CodeEmbedder(
        cache_dir=str(tmp_path),
        configuration=configuration,
    )

    assert embedder.configuration == configuration
    assert captured == {
        "model_name": configuration.model_name,
        "cache_dir": str(tmp_path),
        "device": "auto",
        "truncate_dim": 512,
    }


def test_incomplete_legacy_project_config_uses_verified_manifest_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import json

    import numpy as np

    from embeddings.embedder import EmbeddingResult
    from mcp_server import code_search_server as server_module
    from search.indexer import CodeIndexManager

    source = tmp_path / "source"
    source.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "project_info.json").write_text(
        json.dumps(
            {
                "project_path": str(source),
                "embedding_provider": "voyage",
                "embedding_model": "",
            }
        ),
        encoding="utf-8",
    )
    configuration = EffectiveEmbeddingConfig(
        provider="voyage",
        model_name="voyage-code-3",
        content_mode="code",
        output_dimension=1024,
    )
    manager = CodeIndexManager(str(project_dir / "index"))
    manager.bind_embedding_configuration(
        configuration,
        pipeline_version="verified-pipeline",
    )
    manager.add_embeddings(
        [
            EmbeddingResult(
                embedding=np.ones(1024, dtype=np.float32),
                chunk_id="source.py:1-1:function:source",
                metadata={
                    "file_path": str(source / "source.py"),
                    "relative_path": "source.py",
                    "content_preview": "def source(): pass",
                    "full_content": "def source(): pass",
                    "chunk_type": "function",
                    "start_line": 1,
                    "end_line": 1,
                    "name": "source",
                    "parent_name": None,
                    "docstring": None,
                    "decorators": [],
                    "imports": [],
                    "complexity_score": 1,
                    "tags": [],
                    "folder_structure": [],
                },
            )
        ]
    )
    manager.save_index()

    captured: dict[str, EffectiveEmbeddingConfig] = {}

    class _ConfiguredEmbedder:
        def __init__(
            self,
            *,
            cache_dir: str,
            configuration: EffectiveEmbeddingConfig,
        ) -> None:
            del cache_dir
            captured["configuration"] = configuration
            self.configuration = configuration

    server = server_module.CodeSearchServer.__new__(
        server_module.CodeSearchServer
    )
    monkeypatch.setattr(
        server,
        "get_project_storage_dir",
        lambda *_args, **_kwargs: project_dir,
    )
    monkeypatch.setattr(server_module, "get_storage_dir", lambda: tmp_path)
    monkeypatch.setattr(server_module, "CodeEmbedder", _ConfiguredEmbedder)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    restored = server.embedder(str(source))

    assert restored.configuration == configuration
    assert captured["configuration"] == configuration
    if manager._metadata_db is not None:
        manager._metadata_db.close()
    if manager._fts_conn is not None:
        manager._fts_conn.close()


def test_server_builds_embedder_from_stored_config_without_mutating_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import json

    from mcp_server import code_search_server as server_module

    project_dir = tmp_path / "project-index"
    project_dir.mkdir()
    (project_dir / "project_info.json").write_text(
        json.dumps(
            {
                "embedding_provider": "voyage-context",
                "embedding_model": "voyage-context-3",
                "content_mode": "docs",
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class _ConfiguredEmbedder:
        def __init__(
            self,
            *,
            cache_dir: str,
            configuration: EffectiveEmbeddingConfig,
        ) -> None:
            captured["cache_dir"] = cache_dir
            captured["configuration"] = configuration
            self.configuration = configuration

    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv(
        "LOCAL_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    monkeypatch.setenv("CONTENT_MODE", "code")
    monkeypatch.setattr(
        server_module,
        "get_storage_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        server_module,
        "CodeEmbedder",
        _ConfiguredEmbedder,
    )
    server = server_module.CodeSearchServer()
    monkeypatch.setattr(
        server,
        "get_project_storage_dir",
        lambda *_args, **_kwargs: project_dir,
    )

    embedder = server.embedder(
        "/source/project",
        provider="voyage-context",
    )

    assert isinstance(embedder, _ConfiguredEmbedder)
    assert captured["configuration"] == EffectiveEmbeddingConfig(
        provider="voyage-context",
        model_name="voyage-context-3",
        content_mode="docs",
        output_dimension=1024,
    )
    assert os.environ["EMBEDDING_PROVIDER"] == "local"
    assert (
        os.environ["LOCAL_EMBEDDING_MODEL"]
        == "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert os.environ["CONTENT_MODE"] == "code"
