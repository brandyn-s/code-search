"""Tests for the voyage-code-3 non-default embedding provider.

Pins three contracts from the R12 registry pattern:
1. The provider is registered and discoverable via list_providers().
2. The factory is callable and returns an OpenAIEmbeddingModel instance.
3. The unknown-provider error message lists voyage-code-3 in the available set.

No real API calls are made — the factory import is monkeypatched.
Per docs/findings/2026-05-15-voyage-code-3-ab-finding.md, voyage-4-large
remains the production default; this provider is available for TypeScript-heavy
corpora (mithrandir +0.119 MRR CI excludes zero, enabled via
EMBEDDING_PROVIDER=voyage-code-3).
"""
from __future__ import annotations

import pytest

from embeddings.embedder import (
    _PROVIDER_REGISTRY,
    list_providers,
)


class TestVoyageCode3Registration:
    """voyage-code-3 must appear in the registry and provider list."""

    def test_registered_in_registry(self):
        assert "voyage-code-3" in _PROVIDER_REGISTRY

    def test_appears_in_list_providers(self):
        assert "voyage-code-3" in list_providers()

    def test_list_providers_still_sorted(self):
        providers = list_providers()
        assert providers == sorted(providers)

    def test_factory_is_callable(self):
        factory = _PROVIDER_REGISTRY["voyage-code-3"]
        assert callable(factory)


class TestVoyageCode3Factory:
    """Factory must build an OpenAIEmbeddingModel pointing at the Voyage endpoint."""

    def test_factory_returns_openai_embedding_model(self, monkeypatch):
        """Factory instantiates OpenAIEmbeddingModel — no real API call."""
        captured = {}

        class _FakeModel:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            "embeddings.openai_embedder.OpenAIEmbeddingModel",
            _FakeModel,
        )

        factory = _PROVIDER_REGISTRY["voyage-code-3"]
        result = factory("", "/tmp/cache", "cpu")

        assert isinstance(result, _FakeModel)
        assert captured.get("base_url") == "https://api.voyageai.com/v1"

    def test_factory_uses_voyage_code3_model_name_by_default(self, monkeypatch):
        """When EMBEDDING_MODEL is unset the factory defaults to voyage-code-3."""
        captured = {}

        class _FakeModel:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            "embeddings.openai_embedder.OpenAIEmbeddingModel",
            _FakeModel,
        )
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

        factory = _PROVIDER_REGISTRY["voyage-code-3"]
        factory("", "/tmp/cache", "cpu")

        assert captured.get("model_name") == "voyage-code-3"

    def test_factory_respects_embedding_model_env(self, monkeypatch):
        """EMBEDDING_MODEL override is honoured (same contract as the voyage factory)."""
        captured = {}

        class _FakeModel:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            "embeddings.openai_embedder.OpenAIEmbeddingModel",
            _FakeModel,
        )
        monkeypatch.setenv("EMBEDDING_MODEL", "voyage-code-3-lite")

        factory = _PROVIDER_REGISTRY["voyage-code-3"]
        factory("", "/tmp/cache", "cpu")

        assert captured.get("model_name") == "voyage-code-3-lite"

    def test_factory_reads_voyage_api_key(self, monkeypatch):
        captured = {}

        class _FakeModel:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            "embeddings.openai_embedder.OpenAIEmbeddingModel",
            _FakeModel,
        )
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key-abc")

        factory = _PROVIDER_REGISTRY["voyage-code-3"]
        factory("", "/tmp/cache", "cpu")

        assert captured.get("api_key") == "test-key-abc"


class TestVoyageCode3InErrorMessage:
    """Unknown-provider ValueError must list voyage-code-3 among available providers.

    Pre-R12, the error message was a hardcoded string that drifted from the
    registry. The R12 error calls list_providers() dynamically — this test
    pins that voyage-code-3 shows up, preventing silent omission if the
    provider name is typo'd or unregistered.
    """

    def test_error_lists_voyage_code3(self, monkeypatch):
        from embeddings.embedder import CodeEmbedder

        monkeypatch.setenv("EMBEDDING_PROVIDER", "__no_such_provider__")
        with pytest.raises(ValueError) as exc_info:
            CodeEmbedder()

        assert "voyage-code-3" in str(exc_info.value)
