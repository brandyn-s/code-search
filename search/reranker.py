"""Cross-encoder reranker for search result refinement."""

import logging
import os
from typing import List, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_cross_encoder():
    """Lazy-load the cross-encoder model. Cached after first call."""
    model_name = os.environ.get(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    logger.info(f"Loading cross-encoder reranker: {model_name}")
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    logger.info("Cross-encoder loaded")
    return model


def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Rerank results using a cross-encoder model.

    Args:
        query: The search query.
        results: List of dicts with at least 'chunk_id', 'content', 'score'.
        top_k: Number of results to return.

    Returns:
        Reranked list of result dicts with updated 'score' field.
    """
    if not results:
        return []

    model = _load_cross_encoder()

    # Build query-document pairs
    pairs = [(query, r["content"]) for r in results]

    # Score with cross-encoder
    scores = model.predict(pairs)

    # Attach scores and sort
    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)

    reranked = sorted(results, key=lambda r: r["rerank_score"], reverse=True)
    return reranked[:top_k]
