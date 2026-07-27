"""Tests for R12: embedder provider registry + metadata dedup.

Pre-R12: CodeEmbedder.__init__ was a 6-branch if/elif over
EMBEDDING_PROVIDER. Adding a new provider meant editing __init__ directly.
The 3 metadata-construction blocks (embed_chunk / embed_chunks /
embed_chunks_grouped) were byte-for-byte identical and any schema
change required editing three places.

Post-R12:
- Providers self-register via @register_provider; CodeEmbedder dispatches
  via _PROVIDER_REGISTRY[name](model_name, cache_dir, device).
- _make_embedding_result is the single source for chunk_id + metadata.
"""
from __future__ import annotations

import numpy as np
import pytest

from chunking.code_chunk import CodeChunk
from embeddings.embedder import (
    EffectiveEmbeddingConfig,
    EmbeddingResult,
    _PROVIDER_REGISTRY,
    _make_embedding_result,
    _resolve_provider_name,
    list_providers,
    register_provider,
)


# ---------------------------------------------------------------------------
# Registry contents — all pre-R12 providers must still be registered
# ---------------------------------------------------------------------------

class TestRegistryContents:
    """The set of registered providers must include every name the
    pre-R12 if/elif accepted. Regression-pin."""

    def test_voyage_registered(self):
        assert "voyage" in _PROVIDER_REGISTRY

    def test_voyage_context_registered(self):
        assert "voyage-context" in _PROVIDER_REGISTRY

    def test_openai_registered(self):
        assert "openai" in _PROVIDER_REGISTRY

    def test_jina_and_alias_registered(self):
        assert "jina" in _PROVIDER_REGISTRY
        assert "jina-code" in _PROVIDER_REGISTRY
        # Aliases must point at the same factory.
        assert _PROVIDER_REGISTRY["jina"] is _PROVIDER_REGISTRY["jina-code"]

    def test_local_registered(self):
        assert "local" in _PROVIDER_REGISTRY

    def test_gemma_registered(self):
        assert "gemma" in _PROVIDER_REGISTRY

    def test_list_providers_returns_sorted(self):
        providers = list_providers()
        assert providers == sorted(providers)
        assert "voyage" in providers


# ---------------------------------------------------------------------------
# register_provider decorator behavior
# ---------------------------------------------------------------------------

class TestRegisterProviderDecorator:
    def test_decorator_registers_single_name(self):
        @register_provider("__test_single")
        def factory(model_name, cache_dir, device):
            return ("test_single", model_name, cache_dir, device)

        try:
            assert "__test_single" in _PROVIDER_REGISTRY
            assert _PROVIDER_REGISTRY["__test_single"] is factory
            # Factory is callable with the documented signature.
            result = _PROVIDER_REGISTRY["__test_single"]("m", "c", "cpu")
            assert result == ("test_single", "m", "c", "cpu")
        finally:
            _PROVIDER_REGISTRY.pop("__test_single", None)

    def test_decorator_registers_multiple_aliases(self):
        @register_provider("__test_a", "__test_b", "__test_c")
        def factory(model_name, cache_dir, device):
            return "aliased"

        try:
            for name in ("__test_a", "__test_b", "__test_c"):
                assert name in _PROVIDER_REGISTRY
                assert _PROVIDER_REGISTRY[name] is factory
        finally:
            for name in ("__test_a", "__test_b", "__test_c"):
                _PROVIDER_REGISTRY.pop(name, None)

    def test_decorator_lowercases_names(self):
        @register_provider("__Test_Mixed_Case")
        def factory(model_name, cache_dir, device):
            return "ok"

        try:
            # Stored as lowercase so lookup at dispatch time is case-insensitive.
            assert "__test_mixed_case" in _PROVIDER_REGISTRY
            assert "__Test_Mixed_Case" not in _PROVIDER_REGISTRY
        finally:
            _PROVIDER_REGISTRY.pop("__test_mixed_case", None)


# ---------------------------------------------------------------------------
# _resolve_provider_name — precedence
# ---------------------------------------------------------------------------

class TestResolveProviderName:
    def test_explicit_provider_wins(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("VOYAGE_API_KEY", "anything")
        assert _resolve_provider_name() == "openai"

    def test_provider_lowercased(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "Voyage")
        assert _resolve_provider_name() == "voyage"

    def test_voyage_key_autopicks_voyage(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
        monkeypatch.setenv("VOYAGE_API_KEY", "fake")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert _resolve_provider_name() == "voyage"

    def test_openai_key_does_not_override_local_default(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "fake")
        assert _resolve_provider_name() == "local"

    def test_no_keys_falls_back_to_local(self, monkeypatch):
        for k in ("EMBEDDING_PROVIDER", "VOYAGE_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        assert _resolve_provider_name() == "local"


# ---------------------------------------------------------------------------
# CodeEmbedder dispatch — unknown provider raises with useful message
# ---------------------------------------------------------------------------

class TestUnknownProviderError:
    def test_unknown_provider_raises_with_available_list(self, monkeypatch):
        from embeddings.embedder import CodeEmbedder
        monkeypatch.setenv("EMBEDDING_PROVIDER", "totally_made_up_provider")
        with pytest.raises(ValueError) as exc_info:
            CodeEmbedder()
        msg = str(exc_info.value)
        assert "totally_made_up_provider" in msg
        # Error must list the actual registered providers, not a stale
        # hardcoded string. Pre-R12's error was a fixed list that drifted
        # from reality.
        assert "voyage" in msg
        assert "openai" in msg


# ---------------------------------------------------------------------------
# _make_embedding_result — the deduplicated metadata helper
# ---------------------------------------------------------------------------

def _make_test_chunk(**overrides) -> CodeChunk:
    defaults = dict(
        content="def foo(): pass",
        chunk_type="function",
        start_line=1,
        end_line=2,
        file_path="/proj/src/foo.py",
        relative_path="src/foo.py",
        folder_structure=["src"],
        name="foo",
    )
    defaults.update(overrides)
    return CodeChunk(**defaults)


class TestMakeEmbeddingResult:
    def test_chunk_id_format(self):
        chunk = _make_test_chunk()
        result = _make_embedding_result(
            chunk, embedding=np.zeros(8, dtype=np.float32),
        )
        assert result.chunk_id == "src/foo.py:1-2:function:foo"

    def test_chunk_id_omits_name_when_absent(self):
        chunk = _make_test_chunk(name=None)
        result = _make_embedding_result(
            chunk, embedding=np.zeros(8, dtype=np.float32),
        )
        # No trailing ":name" when name is None.
        assert result.chunk_id == "src/foo.py:1-2:function"

    def test_metadata_schema_fields(self):
        chunk = _make_test_chunk()
        result = _make_embedding_result(
            chunk, embedding=np.zeros(8, dtype=np.float32),
        )
        expected_keys = {
            "file_path", "relative_path", "folder_structure",
            "chunk_type", "start_line", "end_line",
            "name", "parent_name", "docstring", "decorators",
            "imports", "complexity_score", "tags",
            "content_preview", "full_content",
        }
        assert set(result.metadata.keys()) == expected_keys

    def test_content_preview_truncated_when_long(self):
        long_content = "x" * 500
        chunk = _make_test_chunk(content=long_content)
        result = _make_embedding_result(
            chunk, embedding=np.zeros(8, dtype=np.float32),
        )
        assert result.metadata["content_preview"].endswith("...")
        assert len(result.metadata["content_preview"]) <= 203
        # full_content preserves the original (no truncation).
        assert result.metadata["full_content"] == long_content

    def test_content_preview_unchanged_when_short(self):
        chunk = _make_test_chunk(content="def x(): pass")
        result = _make_embedding_result(
            chunk, embedding=np.zeros(8, dtype=np.float32),
        )
        assert result.metadata["content_preview"] == "def x(): pass"
        assert "..." not in result.metadata["content_preview"]

    def test_embedding_passed_through(self):
        chunk = _make_test_chunk()
        emb = np.random.RandomState(42).randn(8).astype(np.float32)
        result = _make_embedding_result(chunk, embedding=emb)
        assert np.array_equal(result.embedding, emb)
        assert isinstance(result, EmbeddingResult)


# ---------------------------------------------------------------------------
# Schema consistency across the 3 embed paths
# ---------------------------------------------------------------------------

class TestSchemaConsistencyAcrossPaths:
    """Pre-R12, the chunk_id + metadata blocks at embed_chunk:276,
    embed_chunks:344, and embed_chunks_grouped:423 were byte-identical.
    All three now route through _make_embedding_result, so a single
    schema-validation test against the helper covers all three."""

    def test_all_paths_use_same_helper(self):
        import inspect
        from embeddings.embedder import CodeEmbedder
        # Verify the helper is referenced in each method's source.
        # This is a lightweight structural check; the real proof is that
        # the byte-identical metadata blocks are gone (greppable).
        embed_chunk_src = inspect.getsource(CodeEmbedder.embed_chunk)
        embed_chunks_src = inspect.getsource(CodeEmbedder.embed_chunks)
        embed_grouped_src = inspect.getsource(
            CodeEmbedder.embed_chunks_grouped,
        )
        for name, src in [
            ("embed_chunk", embed_chunk_src),
            ("embed_chunks", embed_chunks_src),
            ("embed_chunks_grouped", embed_grouped_src),
        ]:
            assert "_make_embedding_result" in src, (
                f"{name} must route through _make_embedding_result to "
                f"keep the chunk_id + metadata schema in sync"
            )


# ---------------------------------------------------------------------------
# embed_chunks_grouped length-mismatch guard
# ---------------------------------------------------------------------------

class TestEmbedChunksGroupedLengthGuard:
    """Regression: 2026-05-26 knowledge-base index lost 5,886 of 7,886 chunks
    via silent zip() truncation. When voyage-context's
    /v1/contextualizedembeddings API rejects individual chunks (oversized,
    malformed, etc.), it returns fewer embeddings than input chunks. The
    bare zip() at the result-construction site would silently drop the
    surviving chunks at the tail; the outer incremental_indexer's
    batch-failure handler would never see the mismatch and the indexer
    would report success with a partial index.

    Fix: raise ValueError on length mismatch so the indexer surfaces the
    drop as an error instead of producing a silent partial index.
    """

    def _build_embedder_with_stub_model(self, embeddings_to_return):
        """Construct a real CodeEmbedder with the inner _model replaced by a
        stub whose encode_grouped returns the caller-supplied array.

        Uses __new__ to bypass CodeEmbedder.__init__'s provider-registry
        lookup, then sets the minimal attribute surface that
        embed_chunks_grouped touches.
        """
        from embeddings.embedder import CodeEmbedder
        import logging

        class _StubInnerModel:
            def __init__(self, returns):
                self._returns = returns

            def encode_grouped(self, grouped_texts, input_type="document"):
                return self._returns

        embedder = CodeEmbedder.__new__(CodeEmbedder)
        embedder._model = _StubInnerModel(embeddings_to_return)
        embedder._logger = logging.getLogger("test_embedder")
        embedder._provider = "voyage-context"
        embedder._configuration = EffectiveEmbeddingConfig(
            provider="voyage-context",
            model_name="voyage-context-3",
            content_mode="code",
            output_dimension=8,
        )
        embedder._sibling_context = {}
        return embedder

    def test_raises_when_fewer_embeddings_than_chunks(self):
        """Mock voyage-context returning 2 embeddings for 4 input chunks
        must raise ValueError (not silently drop the trailing 2 chunks)."""
        embedder = self._build_embedder_with_stub_model(
            embeddings_to_return=np.zeros((2, 8), dtype=np.float32),
        )
        chunks = [
            _make_test_chunk(
                start_line=i, end_line=i + 1,
                relative_path=f"src/f{i}.py", file_path=f"/proj/src/f{i}.py",
            )
            for i in range(4)
        ]

        with pytest.raises(ValueError, match="voyage-context returned 2"):
            embedder.embed_chunks_grouped(chunks, batch_size=32)

    def test_raises_when_more_embeddings_than_chunks(self):
        """Symmetric guard: more embeddings than chunks is also a bug (would
        leak embeddings into the next batch's space). Must raise."""
        embedder = self._build_embedder_with_stub_model(
            embeddings_to_return=np.zeros((5, 8), dtype=np.float32),
        )
        chunks = [
            _make_test_chunk(
                start_line=i, end_line=i + 1,
                relative_path=f"src/f{i}.py", file_path=f"/proj/src/f{i}.py",
            )
            for i in range(3)
        ]

        with pytest.raises(ValueError, match="voyage-context returned 5"):
            embedder.embed_chunks_grouped(chunks, batch_size=32)

    def test_passes_when_lengths_match(self):
        """Sanity: lengths matching is the normal case — must produce one
        EmbeddingResult per chunk, no error."""
        embedder = self._build_embedder_with_stub_model(
            embeddings_to_return=np.zeros((3, 8), dtype=np.float32),
        )
        chunks = [
            _make_test_chunk(
                start_line=i, end_line=i + 1,
                relative_path=f"src/f{i}.py", file_path=f"/proj/src/f{i}.py",
            )
            for i in range(3)
        ]

        results = embedder.embed_chunks_grouped(chunks, batch_size=32)
        assert len(results) == 3
        # All results are EmbeddingResult instances routed through the helper.
        for r in results:
            assert isinstance(r, EmbeddingResult)
