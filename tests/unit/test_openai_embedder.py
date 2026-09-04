"""Tests for OpenAI embedding provider."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def test_openai_embedder_encode_returns_correct_shape():
    """OpenAI embedder should return numpy array with correct dimensions."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1] * 1536, "index": 0},
            {"embedding": [0.2] * 1536, "index": 1},
        ],
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }

    with patch("httpx.Client.post", return_value=mock_response):
        from embeddings.openai_embedder import OpenAIEmbeddingModel

        model = OpenAIEmbeddingModel(api_key="test-key")
        result = model.encode(["hello world", "test query"])

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 1536)


def test_openai_embedder_get_dimension():
    """OpenAI embedder should report correct dimension for text-embedding-3-small."""
    from embeddings.openai_embedder import OpenAIEmbeddingModel

    model = OpenAIEmbeddingModel(
        api_key="test-key", model_name="text-embedding-3-small"
    )
    assert model.get_embedding_dimension() == 1536


def test_openai_embedder_missing_api_key():
    """OpenAI embedder should raise if no API key provided."""
    import os

    old_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        from embeddings.openai_embedder import OpenAIEmbeddingModel

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIEmbeddingModel(api_key="")
    finally:
        if old_key is not None:
            os.environ["OPENAI_API_KEY"] = old_key


def test_openai_embedder_retries_on_500():
    """OpenAI embedder should retry once on server errors."""
    import httpx

    call_count = 0

    def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            resp = MagicMock()
            resp.status_code = 500
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500 Internal Server Error", request=MagicMock(), response=resp
            )
            return resp
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "data": [{"embedding": [0.1] * 1536, "index": 0}],
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        resp.raise_for_status.return_value = None
        return resp

    with patch("httpx.Client.post", side_effect=mock_post):
        from embeddings.openai_embedder import OpenAIEmbeddingModel

        model = OpenAIEmbeddingModel(api_key="test-key")
        # Monkey-patch to skip the actual 5s sleep in tests

        original_encode = model.encode
        import time

        _real_sleep = time.sleep
        time.sleep = lambda x: None
        try:
            result = model.encode(["test"])
        finally:
            time.sleep = _real_sleep

    assert call_count == 2
    assert result.shape == (1, 1536)


def test_voyage_model_dimensions():
    """Voyage AI models should have correct known dimensions."""
    from embeddings.openai_embedder import MODEL_DIMENSIONS

    assert "voyage-code-3" in MODEL_DIMENSIONS
    assert MODEL_DIMENSIONS["voyage-code-3"] == 1024


def test_voyage_provider_creates_embedder():
    """EMBEDDING_PROVIDER=voyage should create OpenAI embedder with Voyage base URL."""
    import os

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
        assert embedder._model._model_name == "voyage-4-large"


def test_voyage_context_provider_creates_embedder():
    """EMBEDDING_PROVIDER=voyage-context should create VoyageContextEmbedder."""
    import os

    with patch.dict(
        os.environ,
        {
            "EMBEDDING_PROVIDER": "voyage-context",
            "VOYAGE_API_KEY": "test-key",
        },
    ):
        from embeddings.embedder import CodeEmbedder

        embedder = CodeEmbedder()
        assert embedder._model._model_name == "voyage-context-3"
        assert hasattr(embedder._model, "encode_grouped")


def test_voyage_embedder_missing_api_key():
    """Voyage embedder should raise if VOYAGE_API_KEY not provided."""
    import os

    old_key = os.environ.pop("VOYAGE_API_KEY", None)
    old_oai = os.environ.pop("OPENAI_API_KEY", None)
    try:
        from embeddings.openai_embedder import OpenAIEmbeddingModel

        with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
            OpenAIEmbeddingModel(api_key="", base_url="https://api.voyageai.com/v1")
    finally:
        if old_key is not None:
            os.environ["VOYAGE_API_KEY"] = old_key
        if old_oai is not None:
            os.environ["OPENAI_API_KEY"] = old_oai


def test_voyage_model_has_smaller_batch_size():
    """Voyage models should default to smaller batch size for rate limit compliance."""
    from embeddings.openai_embedder import OpenAIEmbeddingModel

    model = OpenAIEmbeddingModel(
        api_key="test-key",
        model_name="voyage-code-3",
        base_url="https://api.voyageai.com/v1",
    )
    assert model._batch_size <= 128
