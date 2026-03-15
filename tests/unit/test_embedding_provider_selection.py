"""Tests for embedding provider selection via env vars."""
import os
import pytest
from unittest.mock import patch


def test_openai_provider_selected_when_env_set():
    """EMBEDDING_PROVIDER=openai should create OpenAI model."""
    with patch.dict(os.environ, {
        "EMBEDDING_PROVIDER": "openai",
        "OPENAI_API_KEY": "test-key",
    }):
        from embeddings.embedder import CodeEmbedder
        embedder = CodeEmbedder()
        info = embedder.get_model_info()
        assert info["provider"] == "openai"


def test_local_provider_selected_when_env_set():
    """EMBEDDING_PROVIDER=local should create SentenceTransformerModel."""
    with patch.dict(os.environ, {
        "EMBEDDING_PROVIDER": "local",
    }, clear=False):
        from embeddings.embedder import CodeEmbedder
        from embeddings.sentence_transformer import SentenceTransformerModel
        embedder = CodeEmbedder()
        assert isinstance(embedder._model, SentenceTransformerModel)
        assert "MiniLM" in embedder._model.model_name


def test_default_provider_is_openai_when_key_present():
    """When no EMBEDDING_PROVIDER set but OPENAI_API_KEY exists, default to openai."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
    }, clear=False):
        env = os.environ.copy()
        env.pop("EMBEDDING_PROVIDER", None)
        with patch.dict(os.environ, env, clear=True):
            from embeddings.embedder import CodeEmbedder
            embedder = CodeEmbedder()
            info = embedder.get_model_info()
            assert info["provider"] == "openai"
