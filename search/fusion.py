"""Rank-fusion policies used by hybrid search."""

from __future__ import annotations

from typing import Dict, List, Tuple


# Chunk type boost multipliers per content mode.
CHUNK_TYPE_BOOSTS = {
    "code": {
        "function": 1.3,
        "method": 1.3,
        "class": 1.3,
        "decorated_definition": 1.3,
        "let": 1.3,
        "binding": 1.3,
        "option": 1.3,
        "service_config": 1.3,
        "imports": 1.1,
        "section": 0.7,
        "document": 0.7,
        "module": 0.9,
    },
    "docs": {
        "function": 0.8,
        "method": 0.8,
        "class": 0.8,
        "decorated_definition": 0.8,
        "section": 1.3,
        "document": 1.3,
        "module": 0.9,
    },
    "all": {},
}


def reciprocal_rank_fusion(
    vector_results: List[Tuple[str, float]],
    bm25_results: List[Tuple[str, float]],
    k: int = 60,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
) -> List[Tuple[str, float]]:
    """Fuse two ranked lists using Weighted Reciprocal Rank Fusion.

    Args:
        vector_results: List of (chunk_id, score) from vector search, ordered by relevance.
        bm25_results: List of (chunk_id, score) from BM25 search, ordered by relevance.
        k: Smoothing parameter (default 60, industry standard).
        vector_weight: Weight for vector search contributions (default 0.5).
        bm25_weight: Weight for BM25 search contributions (default 0.5).

    Returns:
        List of (chunk_id, rrf_score) sorted by fused relevance.
    """
    scores: Dict[str, float] = {}
    for rank, (chunk_id, _score) in enumerate(vector_results):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + vector_weight * (
            1.0 / (k + rank + 1)
        )
    for rank, (chunk_id, _score) in enumerate(bm25_results):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + bm25_weight * (
            1.0 / (k + rank + 1)
        )
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
