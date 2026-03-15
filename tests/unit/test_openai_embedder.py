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
    model = OpenAIEmbeddingModel(api_key="test-key", model_name="text-embedding-3-small")
    assert model.get_embedding_dimension() == 1536


def test_openai_embedder_missing_api_key():
    """OpenAI embedder should raise if no API key provided."""
    from embeddings.openai_embedder import OpenAIEmbeddingModel
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingModel(api_key="")
