"""Tests for embedding provider selection via env vars."""

import os
import pytest
from unittest.mock import patch


def test_openai_provider_selected_when_env_set():
    """EMBEDDING_PROVIDER=openai should create OpenAI model."""
    with patch.dict(
        os.environ,
        {
            "EMBEDDING_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
        },
    ):
        from embeddings.embedder import CodeEmbedder

        embedder = CodeEmbedder()
        info = embedder.get_model_info()
        assert info["provider"] == "openai"


def test_local_provider_selected_when_env_set():
    """EMBEDDING_PROVIDER=local should create SentenceTransformerModel."""
    with patch.dict(
        os.environ,
        {
            "EMBEDDING_PROVIDER": "local",
        },
        clear=False,
    ):
        from embeddings.embedder import CodeEmbedder
        from embeddings.sentence_transformer import SentenceTransformerModel

        embedder = CodeEmbedder()
        assert isinstance(embedder._model, SentenceTransformerModel)
        assert "MiniLM" in embedder._model.model_name


def test_default_provider_is_voyage_context_when_voyage_key_present():
    """When no EMBEDDING_PROVIDER set but VOYAGE_API_KEY exists, default to voyage-context."""
    env = os.environ.copy()
    env.pop("EMBEDDING_PROVIDER", None)
    env["VOYAGE_API_KEY"] = "test-key"
    env.pop("OPENAI_API_KEY", None)
    with patch.dict(os.environ, env, clear=True):
        from embeddings.embedder import CodeEmbedder

        embedder = CodeEmbedder()
        info = embedder.get_model_info()
        assert info["provider"] == "voyage-context"


def test_default_provider_is_openai_when_only_openai_key_present():
    """When no EMBEDDING_PROVIDER set and only OPENAI_API_KEY exists, default to openai."""
    env = os.environ.copy()
    env.pop("EMBEDDING_PROVIDER", None)
    env.pop("VOYAGE_API_KEY", None)
    env["OPENAI_API_KEY"] = "test-key"
    with patch.dict(os.environ, env, clear=True):
        from embeddings.embedder import CodeEmbedder

        embedder = CodeEmbedder()
        info = embedder.get_model_info()
        assert info["provider"] == "openai"


def test_voyage_provider_selected_when_env_set():
    """EMBEDDING_PROVIDER=voyage should create Voyage-configured model."""
    with patch.dict(
        os.environ,
        {
            "EMBEDDING_PROVIDER": "voyage",
            "VOYAGE_API_KEY": "test-key",
        },
    ):
        from embeddings.embedder import CodeEmbedder

        embedder = CodeEmbedder()
        assert embedder._model._base_url == "https://api.voyageai.com/v1"
