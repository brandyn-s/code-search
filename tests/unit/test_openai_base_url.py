"""OPENAI_BASE_URL: point the ``openai`` provider at any OpenAI-compatible server."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embeddings.openai_embedder import (
    DEFAULT_OPENAI_BASE_URL,
    OpenAIEmbeddingModel,
    build_auth_headers,
    requires_api_key,
    resolve_auth_header_style,
    resolve_openai_base_url,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_AUTH_HEADER",
                 "EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_DIMENSION",
                 "VOYAGE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _ok_response(dim: int, count: int) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": [{"embedding": [0.1] * dim, "index": i} for i in range(count)],
    }
    return response


def test_default_base_url_and_trailing_slash_stripping(monkeypatch):
    assert resolve_openai_base_url() == DEFAULT_OPENAI_BASE_URL
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1///")
    assert resolve_openai_base_url() == "http://localhost:11434/v1"
    # An explicit argument (the Voyage factories) wins over the environment.
    assert resolve_openai_base_url("https://api.voyageai.com/v1/") == "https://api.voyageai.com/v1"


def test_key_required_only_for_openai_host():
    assert requires_api_key(DEFAULT_OPENAI_BASE_URL) is True
    assert requires_api_key("https://api.voyageai.com/v1") is True
    assert requires_api_key("http://localhost:11434/v1") is False
    assert requires_api_key("https://example.openai.azure.com/openai/v1") is False


def test_missing_key_message_names_base_url_alternative():
    with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
        OpenAIEmbeddingModel(api_key="")


def test_self_hosted_server_needs_no_key_and_sends_no_authorization(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1/")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "768")
    captured: dict = {}

    def fake_post(self, url, headers=None, json=None, **kwargs):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json
        return _ok_response(768, len(json["input"]))

    with patch("httpx.Client.post", new=fake_post):
        model = OpenAIEmbeddingModel(model_name="nomic-embed-text")
        vectors = model.encode(["def a(): pass", "def b(): pass"])

    assert captured["url"] == "http://localhost:11434/v1/embeddings"
    assert "Authorization" not in captured["headers"]
    assert "api-key" not in captured["headers"]
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["body"] == {"input": ["def a(): pass", "def b(): pass"], "model": "nomic-embed-text"}
    assert isinstance(vectors, np.ndarray) and vectors.shape == (2, 768)
    assert model.get_embedding_dimension() == 768
    assert model.get_model_info()["base_url"] == "http://localhost:11434/v1"


def test_bearer_header_when_key_present(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: dict = {}

    def fake_post(self, url, headers=None, json=None, **kwargs):
        captured["headers"] = headers
        return _ok_response(1536, 1)

    with patch("httpx.Client.post", new=fake_post):
        OpenAIEmbeddingModel(model_name="text-embedding-3-small").encode(["x"])
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


def test_api_key_header_style_for_azure(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://myres.openai.azure.com/openai/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("OPENAI_AUTH_HEADER", "api-key")
    captured: dict = {}

    def fake_post(self, url, headers=None, json=None, **kwargs):
        captured["headers"] = headers
        return _ok_response(1536, 1)

    with patch("httpx.Client.post", new=fake_post):
        OpenAIEmbeddingModel(model_name="text-embedding-3-small").encode(["x"])
    assert captured["headers"]["api-key"] == "azure-key"
    assert "Authorization" not in captured["headers"]


def test_invalid_auth_header_style_is_rejected(monkeypatch):
    monkeypatch.setenv("OPENAI_AUTH_HEADER", "basic")
    with pytest.raises(ValueError, match="OPENAI_AUTH_HEADER"):
        resolve_auth_header_style()
    assert build_auth_headers("", "bearer") == {}


def test_provider_resolution_with_base_url_and_no_key(monkeypatch):
    """EMBEDDING_PROVIDER=openai against a self-hosted server constructs without a key."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "768")

    from embeddings.embedder import CodeEmbedder, resolve_embedding_config

    config = resolve_embedding_config()
    assert config.provider == "openai"
    assert config.model_name == "nomic-embed-text"
    assert config.output_dimension == 768

    embedder = CodeEmbedder()
    assert embedder._model._base_url == "http://localhost:8000/v1"
    assert embedder._model._api_key == ""
    assert embedder.configuration.output_dimension == 768


def test_custom_model_without_dimension_contract_fails_clearly(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text")

    from embeddings.embedder import resolve_embedding_config

    with pytest.raises(ValueError, match="EMBEDDING_DIMENSION"):
        resolve_embedding_config()


def test_openai_host_without_key_still_fails(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")

    from embeddings.embedder import CodeEmbedder

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        CodeEmbedder()


def test_voyage_factories_keep_their_endpoint(monkeypatch):
    """OPENAI_BASE_URL must not redirect the Voyage providers."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("VOYAGE_API_KEY", "v-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "voyage")

    from embeddings.embedder import CodeEmbedder

    embedder = CodeEmbedder()
    assert embedder._model._base_url == "https://api.voyageai.com/v1"
    assert json.dumps(embedder._model.get_model_info())  # serialisable
