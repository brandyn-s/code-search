"""Tests for Voyage contextualized chunk embedder."""

import numpy as np
from unittest.mock import patch, MagicMock


def test_voyage_context_encode_flat():
    """Standard encode() should work like OpenAI embedder for queries."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "object": "list",
        "data": [
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": [0.1] * 1024, "index": 0}
                ],
                "index": 0,
            }
        ],
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }

    with patch("httpx.Client.post", return_value=mock_response):
        from embeddings.voyage_context_embedder import VoyageContextEmbedder

        model = VoyageContextEmbedder(api_key="test-key")
        result = model.encode(["test query"])

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1024)


def test_voyage_context_encode_grouped():
    """encode_grouped() should send chunks grouped by document."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "object": "list",
        "data": [
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": [0.1] * 1024, "index": 0},
                    {"object": "embedding", "embedding": [0.2] * 1024, "index": 1},
                ],
                "index": 0,
            },
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": [0.3] * 1024, "index": 0},
                ],
                "index": 1,
            },
        ],
        "usage": {"prompt_tokens": 20, "total_tokens": 20},
    }

    with patch("httpx.Client.post", return_value=mock_response):
        from embeddings.voyage_context_embedder import VoyageContextEmbedder

        model = VoyageContextEmbedder(api_key="test-key")
        grouped = [
            ["chunk1_of_file1", "chunk2_of_file1"],
            ["chunk1_of_file2"],
        ]
        result = model.encode_grouped(grouped)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 1024)  # 3 total chunks flattened


def test_voyage_context_dimension():
    """Should report 1024 dimensions for voyage-context-3."""
    from embeddings.voyage_context_embedder import VoyageContextEmbedder

    model = VoyageContextEmbedder(api_key="test-key")
    assert model.get_embedding_dimension() == 1024
