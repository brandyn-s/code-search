"""Tests for provider/model-namespaced query-embedding caches.

Pins the cross-provider cache-poisoning fix: CodeEmbedder's class-level
in-memory LRU and the on-disk SQLite cache are keyed by
(provider, model, normalized query). Pre-fix, the in-memory key was the
bare query text, so the MCP server's per-project embedder reconstruction
(switch_project between projects with different providers) returned the
PREVIOUS model's embedding — a loud dimension mismatch when dims differ,
silently wrong similarities when they match (voyage-4-large and
voyage-code-3 are both 1024-d).
"""
from __future__ import annotations

import numpy as np
import pytest

from embeddings import embedder as embedder_module
from embeddings.embedder import (
    CodeEmbedder,
    EffectiveEmbeddingConfig,
    register_provider,
)


class _FakeModel:
    """Minimal embedding model returning a constant per-instance vector."""

    def __init__(self, model_name: str, vector: np.ndarray):
        self._model_name = model_name
        self._vector = vector.astype(np.float32)
        self.encode_calls = 0
        self.last_encode_kwargs = {}

    def encode(self, texts, **kwargs):
        self.encode_calls += 1
        self.last_encode_kwargs = kwargs
        return np.stack([self._vector] * len(texts))

    def get_embedding_dimension(self):
        return int(self._vector.shape[0])

    def get_model_info(self):
        return {"model_name": self._model_name}

    def cleanup(self):
        pass


@pytest.fixture
def isolated_caches(tmp_path, monkeypatch):
    """Point the disk cache at a tmp dir and empty the shared in-memory LRU.

    get_storage_dir is lru_cached, so the env override only takes effect
    after a cache_clear — and must be cleared again on teardown so the
    tmp path doesn't leak into other tests.
    """
    from common_utils import get_storage_dir
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path / "storage"))
    get_storage_dir.cache_clear()
    CodeEmbedder._query_cache.clear()
    yield
    CodeEmbedder._query_cache.clear()
    get_storage_dir.cache_clear()


@pytest.fixture
def fake_providers(monkeypatch):
    """Register two fake providers with distinct vectors and dimensions."""
    vec_a = np.full(8, 1.0)
    vec_b = np.full(4, 2.0)
    models = {}

    @register_provider("fake-prov-a")
    def _factory_a(model_name, cache_dir, device):
        m = _FakeModel("fake-model-a", vec_a)
        models["a"] = m
        return m

    @register_provider("fake-prov-b")
    def _factory_b(model_name, cache_dir, device):
        m = _FakeModel("fake-model-b", vec_b)
        models["b"] = m
        return m

    yield models
    embedder_module._PROVIDER_REGISTRY.pop("fake-prov-a", None)
    embedder_module._PROVIDER_REGISTRY.pop("fake-prov-b", None)


@pytest.fixture
def fake_voyage_provider():
    models = []
    original_factory = embedder_module._PROVIDER_REGISTRY["voyage"]

    @register_provider("voyage")
    def _factory(model_name, cache_dir, device):
        model = _FakeModel(model_name, np.full(1024, 1.0))
        models.append(model)
        return model

    yield models
    embedder_module._PROVIDER_REGISTRY["voyage"] = original_factory


def _make_embedder(monkeypatch, provider: str, tmp_path) -> CodeEmbedder:
    monkeypatch.setenv("EMBEDDING_PROVIDER", provider)
    return CodeEmbedder(cache_dir=str(tmp_path / "models"))


def test_same_query_different_providers_not_shared(
    isolated_caches, fake_providers, monkeypatch, tmp_path
):
    """The poisoning scenario: same query, two providers, two embedders."""
    emb_a = _make_embedder(monkeypatch, "fake-prov-a", tmp_path)
    out_a = emb_a.embed_query("find auth handler")
    assert out_a.shape == (8,)

    emb_b = _make_embedder(monkeypatch, "fake-prov-b", tmp_path)
    out_b = emb_b.embed_query("find auth handler")

    # Pre-fix this returned the 8-dim provider-a vector from the shared
    # class-level cache. It must be provider-b's own 4-dim vector.
    assert out_b.shape == (4,), (
        f"provider-b got a {out_b.shape} vector — cross-provider cache hit"
    )
    assert np.allclose(out_b, 2.0)
    assert fake_providers["b"].encode_calls == 1


def test_same_provider_instances_share_cache(
    isolated_caches, fake_providers, monkeypatch, tmp_path
):
    """Class-level sharing is intentional WITHIN a (provider, model) pair:
    the server rebuilds embedders on every switch_project and the cache
    should survive that."""
    emb_1 = _make_embedder(monkeypatch, "fake-prov-a", tmp_path)
    emb_1.embed_query("find auth handler")
    first_model = fake_providers["a"]
    assert first_model.encode_calls == 1

    emb_2 = _make_embedder(monkeypatch, "fake-prov-a", tmp_path)
    out = emb_2.embed_query("find auth handler")
    second_model = fake_providers["a"]

    assert np.allclose(out, 1.0)
    # The second instance's model must not have been called — in-memory hit.
    assert second_model.encode_calls == 0


def test_disk_cache_round_trip_is_provider_scoped(
    isolated_caches, fake_providers, monkeypatch, tmp_path
):
    """Disk layer: provider-a's persisted row must not serve provider-b."""
    emb_a = _make_embedder(monkeypatch, "fake-prov-a", tmp_path)
    emb_a.embed_query("database connection pool")

    # Simulate a process restart for the in-memory layer only: the SQLite
    # cache file survives, the LRU does not.
    CodeEmbedder._query_cache.clear()

    emb_b = _make_embedder(monkeypatch, "fake-prov-b", tmp_path)
    out_b = emb_b.embed_query("database connection pool")
    assert out_b.shape == (4,)
    assert np.allclose(out_b, 2.0)

    # And provider-a still gets its own row back from disk.
    CodeEmbedder._query_cache.clear()
    emb_a2 = _make_embedder(monkeypatch, "fake-prov-a", tmp_path)
    out_a = emb_a2.embed_query("database connection pool")
    assert out_a.shape == (8,)
    assert np.allclose(out_a, 1.0)
    # Served from disk, not re-encoded.
    assert fake_providers["a"].encode_calls == 0


def test_cache_namespace_uses_instance_identity_not_env(
    isolated_caches, fake_providers, monkeypatch, tmp_path
):
    """The namespace must come from the resolved instance, not os.environ at
    query time — the server restores its env override before the first
    query runs."""
    emb_a = _make_embedder(monkeypatch, "fake-prov-a", tmp_path)
    # Env now claims provider-b; the instance is still provider-a.
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake-prov-b")
    assert emb_a._cache_namespace().startswith("fake-prov-a::")
    out = emb_a.embed_query("rate limiter")
    assert out.shape == (8,)


def test_input_type_is_frozen_and_namespaces_query_caches(
    isolated_caches,
    fake_voyage_provider,
    monkeypatch,
    tmp_path,
) -> None:
    base = {
        "provider": "voyage",
        "model_name": "voyage-4-large",
        "content_mode": "code",
        "output_dimension": 1024,
    }
    disabled = CodeEmbedder(
        cache_dir=str(tmp_path / "models"),
        configuration=EffectiveEmbeddingConfig(
            **base,
            input_type_enabled=False,
        ),
    )
    disabled.embed_query("same query")
    assert "input_type" not in fake_voyage_provider[-1].last_encode_kwargs

    CodeEmbedder._query_cache.clear()
    monkeypatch.setenv("VOYAGE_INPUT_TYPE", "off")
    enabled = CodeEmbedder(
        cache_dir=str(tmp_path / "models"),
        configuration=EffectiveEmbeddingConfig(
            **base,
            input_type_enabled=True,
        ),
    )
    enabled.embed_query("same query")

    assert fake_voyage_provider[-1].encode_calls == 1
    assert fake_voyage_provider[-1].last_encode_kwargs["input_type"] == "query"
    monkeypatch.setenv("VOYAGE_INPUT_TYPE", "on")
    assert disabled._doc_encode_kwargs().get("input_type") is None
    monkeypatch.setenv("VOYAGE_INPUT_TYPE", "off")
    assert enabled._doc_encode_kwargs()["input_type"] == "document"
