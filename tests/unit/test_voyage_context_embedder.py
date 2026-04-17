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


def test_voyage_context_batches_use_full_api_capacity():
    """Batches should pack many small groups, not cap at 4.

    Regression for the 2026-04-17 incident where _prepare_voyage_batches
    capped at 4 groups per request — 250x below the Voyage contextualized
    API limit of 1000 inputs/request. Small-file repos took hours instead
    of minutes because each request only sent 4 files.
    """
    from embeddings.voyage_context_embedder import _prepare_voyage_batches

    # 100 small groups, each well under token cap — should pack efficiently.
    small_groups = [["short chunk"] for _ in range(100)]
    batches = _prepare_voyage_batches(small_groups)

    # Before fix: 100 / 4 = 25 batches. After fix: 1 batch (all fit under
    # both token cap and input cap).
    assert len(batches) == 1, (
        f"Expected 1 batch for 100 small groups, got {len(batches)}. "
        "The group-count cap is throttling voyage-context throughput."
    )
    assert len(batches[0]) == 100


def test_voyage_context_respects_input_cap():
    """When inputs exceed the API's 1000-input limit, split into batches."""
    from embeddings.voyage_context_embedder import (
        _prepare_voyage_batches,
        _VOYAGE_MAX_INPUTS_PER_BATCH,
    )

    # Build N small groups where N > _VOYAGE_MAX_INPUTS_PER_BATCH
    n = _VOYAGE_MAX_INPUTS_PER_BATCH + 50
    small_groups = [["short"] for _ in range(n)]
    batches = _prepare_voyage_batches(small_groups)

    assert len(batches) >= 2, "Must split when exceeding input cap"
    # First batch should be at the cap
    assert len(batches[0]) == _VOYAGE_MAX_INPUTS_PER_BATCH


def test_voyage_context_respects_token_cap():
    """When tokens exceed per-batch cap, split even if input count is small."""
    from embeddings.voyage_context_embedder import _prepare_voyage_batches

    # 3 groups each ~50K tokens — should split (total 150K > 100K cap)
    # 50K tokens ≈ 200K chars
    large_group = ["x" * 200_000]
    batches = _prepare_voyage_batches([large_group, large_group, large_group])

    # 3 groups × 50K = 150K tokens, exceeds 100K cap → at least 2 batches
    assert len(batches) >= 2
