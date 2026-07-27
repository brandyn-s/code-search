"""Tests for the content-hash document-embedding cache (roadmap P4).

Key = (sha256(content), provider, model, input_mode). Mechanism contract:
identical content under the same (provider, model, input_mode) never hits
the model twice; any key component change re-encodes; grouped/
contextualized providers bypass the cache entirely (their vectors are
document-context-dependent).
"""
from __future__ import annotations

import numpy as np
import pytest

from chunking.code_chunk import CodeChunk
from embeddings import embedder as embedder_module
from embeddings.embedder import CodeEmbedder, register_provider


class _CountingModel:
    """Deterministic per-text vectors + a log of every encode call."""

    def __init__(self, model_name: str, dim: int = 8, salt: float = 1.0):
        self._model_name = model_name
        self.dim = dim
        self.salt = salt
        self.encode_calls = 0
        self.encoded_texts: list = []

    def _vec(self, text: str) -> np.ndarray:
        rng = np.random.RandomState((hash(text) ^ hash(self.salt)) & 0xFFFFFFFF)
        return rng.randn(self.dim).astype(np.float32)

    def encode(self, texts, **kwargs):
        self.encode_calls += 1
        self.encoded_texts.extend(texts)
        return np.stack([self._vec(t) for t in texts])

    def get_embedding_dimension(self):
        return self.dim

    def get_model_info(self):
        return {"model_name": self._model_name}

    def cleanup(self):
        pass


class _GroupedCountingModel(_CountingModel):
    """Counting model with encode_grouped (voyage-context shape)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grouped_calls = 0

    def encode_grouped(self, grouped_texts, input_type="document"):
        self.grouped_calls += 1
        flat = [t for group in grouped_texts for t in group]
        return np.stack([self._vec(t) for t in flat])


def _make_chunk(name: str, body: str, path: str = "mod.py") -> CodeChunk:
    return CodeChunk(
        content=body,
        chunk_type="function",
        start_line=1,
        end_line=3,
        file_path=f"/abs/{path}",
        relative_path=path,
        folder_structure=[],
        name=name,
    )


@pytest.fixture
def isolated_caches(tmp_path, monkeypatch):
    from common_utils import get_storage_dir
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path / "storage"))
    monkeypatch.delenv("VOYAGE_INPUT_TYPE", raising=False)
    get_storage_dir.cache_clear()
    CodeEmbedder._query_cache.clear()
    yield
    CodeEmbedder._query_cache.clear()
    get_storage_dir.cache_clear()


@pytest.fixture
def flat_provider(monkeypatch):
    models = []

    @register_provider("fake-doc-flat")
    def _factory(model_name, cache_dir, device):
        m = _CountingModel("fake-doc-model")
        models.append(m)
        return m

    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake-doc-flat")
    yield models
    embedder_module._PROVIDER_REGISTRY.pop("fake-doc-flat", None)


@pytest.fixture
def grouped_provider(monkeypatch):
    models = []

    @register_provider("fake-doc-grouped")
    def _factory(model_name, cache_dir, device):
        m = _GroupedCountingModel("fake-doc-grouped-model")
        models.append(m)
        return m

    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake-doc-grouped")
    yield models
    embedder_module._PROVIDER_REGISTRY.pop("fake-doc-grouped", None)


@pytest.fixture
def voyage_flat_provider(monkeypatch):
    models = []
    original_factory = embedder_module._PROVIDER_REGISTRY["voyage"]

    @register_provider("voyage")
    def _factory(model_name, cache_dir, device):
        m = _CountingModel(model_name, dim=1024)
        models.append(m)
        return m

    monkeypatch.setenv("EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("EMBEDDING_MODEL", "voyage-4-large")
    yield models
    embedder_module._PROVIDER_REGISTRY["voyage"] = original_factory


def test_second_index_run_serves_all_from_cache(isolated_caches, flat_provider, tmp_path):
    chunks = [_make_chunk(f"f{i}", f"def f{i}():\n    return {i}") for i in range(5)]

    emb1 = CodeEmbedder(cache_dir=str(tmp_path / "m"))
    out1 = emb1.embed_chunks(chunks)
    model1 = flat_provider[-1]
    assert model1.encode_calls >= 1
    assert len(model1.encoded_texts) == 5

    # Fresh embedder instance = the "second full reindex" shape.
    emb2 = CodeEmbedder(cache_dir=str(tmp_path / "m"))
    out2 = emb2.embed_chunks(chunks)
    model2 = flat_provider[-1]
    assert model2.encode_calls == 0, (
        "unchanged content re-hit the model on the second run"
    )
    for r1, r2 in zip(out1, out2):
        assert np.allclose(r1.embedding, r2.embedding), (
            "cached vector differs from freshly encoded vector"
        )


def test_partial_change_encodes_only_novel_content(isolated_caches, flat_provider, tmp_path):
    chunks = [_make_chunk(f"f{i}", f"def f{i}():\n    return {i}") for i in range(10)]
    CodeEmbedder(cache_dir=str(tmp_path / "m")).embed_chunks(chunks)

    # Simulate a branch switch touching 2 of 10 chunks.
    chunks[3] = _make_chunk("f3", "def f3():\n    return 'changed'")
    chunks[7] = _make_chunk("f7", "def f7():\n    return 'changed too'")
    emb2 = CodeEmbedder(cache_dir=str(tmp_path / "m"))
    emb2.embed_chunks(chunks)
    model2 = flat_provider[-1]
    assert len(model2.encoded_texts) == 2, (
        f"expected only the 2 changed texts to encode, got "
        f"{len(model2.encoded_texts)}"
    )


def test_provider_and_model_isolation(isolated_caches, flat_provider, monkeypatch, tmp_path):
    other_models = []

    @register_provider("fake-doc-flat-b")
    def _factory_b(model_name, cache_dir, device):
        m = _CountingModel("fake-doc-model-b", salt=2.0)
        other_models.append(m)
        return m

    try:
        chunk = _make_chunk("f", "def f():\n    return 1")
        CodeEmbedder(cache_dir=str(tmp_path / "m")).embed_chunks([chunk])

        monkeypatch.setenv("EMBEDDING_PROVIDER", "fake-doc-flat-b")
        emb_b = CodeEmbedder(cache_dir=str(tmp_path / "m"))
        out_b = emb_b.embed_chunks([chunk])
        assert other_models[-1].encode_calls == 1, (
            "provider B must not be served provider A's cached vector"
        )
        assert np.allclose(out_b[0].embedding, other_models[-1]._vec(
            emb_b.create_embedding_content(chunk)
        ))
    finally:
        embedder_module._PROVIDER_REGISTRY.pop("fake-doc-flat-b", None)


def test_input_type_mode_is_part_of_the_key(
    isolated_caches,
    voyage_flat_provider,
    monkeypatch,
    tmp_path,
):
    chunk = _make_chunk("f", "def f():\n    return 1")
    CodeEmbedder(cache_dir=str(tmp_path / "m")).embed_chunks([chunk])

    monkeypatch.setenv("VOYAGE_INPUT_TYPE", "on")
    emb2 = CodeEmbedder(cache_dir=str(tmp_path / "m"))
    emb2.embed_chunks([chunk])
    assert voyage_flat_provider[-1].encode_calls == 1, (
        "input_type=on must re-encode — same content embeds differently "
        "with input_type set"
    )


def test_intra_batch_duplicate_content_encodes_once(isolated_caches, flat_provider, tmp_path):
    body = "def same():\n    return 0"
    chunks = [_make_chunk("same", body, path="a.py"),
              _make_chunk("same", body, path="a.py")]
    emb = CodeEmbedder(cache_dir=str(tmp_path / "m"))
    out = emb.embed_chunks(chunks)
    model = flat_provider[-1]
    assert len(model.encoded_texts) == 1
    assert len(out) == 2
    assert np.allclose(out[0].embedding, out[1].embedding)


def test_single_chunk_path_shares_the_cache(isolated_caches, flat_provider, tmp_path):
    chunk = _make_chunk("f", "def f():\n    return 1")
    emb = CodeEmbedder(cache_dir=str(tmp_path / "m"))
    emb.embed_chunks([chunk])
    calls_after_batch = flat_provider[-1].encode_calls

    emb.embed_chunk(chunk)  # single-chunk path, same content
    assert flat_provider[-1].encode_calls == calls_after_batch, (
        "embed_chunk re-encoded content already cached by embed_chunks"
    )


def test_grouped_provider_bypasses_cache(isolated_caches, grouped_provider, tmp_path):
    """Contextualized vectors depend on the file group — never cache them."""
    chunks = [_make_chunk(f"f{i}", f"def f{i}():\n    return {i}") for i in range(3)]
    emb1 = CodeEmbedder(cache_dir=str(tmp_path / "m"))
    emb1.embed_chunks_grouped(chunks)
    assert grouped_provider[-1].grouped_calls == 1

    emb2 = CodeEmbedder(cache_dir=str(tmp_path / "m"))
    emb2.embed_chunks_grouped(chunks)
    assert grouped_provider[-1].grouped_calls == 1, (
        "grouped path must re-encode every run (context-dependent vectors)"
    )


def test_clear_document_cache(isolated_caches, flat_provider, tmp_path):
    chunk = _make_chunk("f", "def f():\n    return 1")
    emb = CodeEmbedder(cache_dir=str(tmp_path / "m"))
    emb.embed_chunks([chunk])

    emb.clear_document_cache()
    emb2 = CodeEmbedder(cache_dir=str(tmp_path / "m"))
    emb2.embed_chunks([chunk])
    assert flat_provider[-1].encode_calls == 1, "cache survived clear_document_cache"


def test_cache_unavailable_degrades_to_plain_encode(isolated_caches, flat_provider, tmp_path, monkeypatch):
    chunk = _make_chunk("f", "def f():\n    return 1")
    emb = CodeEmbedder(cache_dir=str(tmp_path / "m"))
    monkeypatch.setattr(emb, "_get_disk_cache", lambda: None)
    out = emb.embed_chunks([chunk])
    assert len(out) == 1
    assert flat_provider[-1].encode_calls == 1
