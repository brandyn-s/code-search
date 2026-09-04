"""Reranker registry: one place where `RERANKER=<mode>` maps to an implementation.

Mirrors the embedding-provider registry in ``embeddings.embedder``. A reranker
is a function with the signature::

    def rerank(searcher, query, *, k, config, candidates, metadata_lookup) -> list[SearchResult]

It must set ``searcher.last_reranker_metadata`` (a dict with at least
``applied``, ``reason`` and ``latency_ms``) and return the ranked list
truncated to ``k``. Failures inside an engine must preserve the hybrid order
rather than raise; that is the contract callers rely on.

Register a new mode with::

    @register_reranker("my-mode")
    def rerank_my_mode(searcher, query, *, k, config, candidates, metadata_lookup):
        ...

``search.config.RERANKER_MODES`` is derived from this registry plus ``auto``,
so a registered name is immediately a valid ``RERANKER`` value.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from search.result_models import SearchResult

RerankerFn = Callable[..., List[SearchResult]]

_RERANKER_REGISTRY: Dict[str, RerankerFn] = {}

# Number of top hybrid candidates handed to an LLM reranker.
LLM_RERANK_POOL = 15


def register_reranker(*names: str) -> Callable[[RerankerFn], RerankerFn]:
    """Decorator that registers ``fn`` under one or more mode names."""

    def decorator(fn: RerankerFn) -> RerankerFn:
        for name in names:
            _RERANKER_REGISTRY[name.lower()] = fn
        return fn

    return decorator


def registered_rerankers() -> Tuple[str, ...]:
    """Mode names in registration order (excludes the ``auto`` alias)."""
    return tuple(_RERANKER_REGISTRY)


def get_reranker(mode: str) -> RerankerFn:
    """Return the implementation for ``mode``; ``off`` when unknown.

    ``search.config`` validates ``RERANKER`` before it reaches this point, so
    the fallback only matters for direct callers.
    """
    return _RERANKER_REGISTRY.get(mode.lower(), _RERANKER_REGISTRY["off"])


def _not_invoked(searcher: Any, reason: str) -> None:
    searcher.last_reranker_metadata = {
        "applied": False,
        "reason": reason,
        "latency_ms": 0,
    }


def _full_content(metadata_lookup: dict, result: SearchResult) -> str:
    metadata = metadata_lookup.get(result.chunk_id, {}) or {}
    return (
        metadata.get("full_content")
        or metadata.get("content")
        or result.content_preview
        or ""
    )


@register_reranker("sonnet")
def rerank_sonnet(
    searcher: Any,
    query: str,
    *,
    k: int,
    config: Any,
    candidates: List[SearchResult],
    metadata_lookup: dict,
) -> List[SearchResult]:
    """Pointwise Sonnet rerank of the top hybrid candidates."""
    if len(candidates) <= k:
        _not_invoked(searcher, "not_invoked_insufficient_candidates")
        return candidates[:k]

    skip_threshold = config.sonnet_skip_threshold
    if skip_threshold is not None:
        top_1_score = candidates[0].similarity_score
        if top_1_score >= skip_threshold:
            searcher._logger.info(
                "[RERANK_REASON] skipped_high_confidence "
                "top_1_score=%.4f threshold=%.4f "
                "n_candidates=%d; preserved hybrid order",
                top_1_score,
                skip_threshold,
                len(candidates),
            )
            searcher.last_reranker_metadata = {
                "applied": False,
                "reason": "skipped_high_confidence",
                "latency_ms": 0,
                "top_1_score": top_1_score,
                "skip_threshold": skip_threshold,
            }
            return candidates[:k]

    from search.sonnet_reranker import rerank_with_sonnet

    n_to_rerank = min(LLM_RERANK_POOL, len(candidates))
    rerank_input = [
        {
            "chunk_id": result.chunk_id,
            "file_path": result.relative_path,
            "full_content": _full_content(metadata_lookup, result),
            "_orig": result,
        }
        for result in candidates[:n_to_rerank]
    ]
    reranked, rerank_meta = rerank_with_sonnet(
        query,
        rerank_input,
        top_k=k,
        return_metadata=True,
    )
    searcher.last_reranker_metadata = rerank_meta
    new_top = [item["_orig"] for item in reranked]
    return (new_top + candidates[n_to_rerank:])[:k]


@register_reranker("listwise")
def rerank_listwise(
    searcher: Any,
    query: str,
    *,
    k: int,
    config: Any,
    candidates: List[SearchResult],
    metadata_lookup: dict,
) -> List[SearchResult]:
    """Listwise Sonnet rerank with a hard deadline."""
    if len(candidates) <= k:
        _not_invoked(searcher, "not_invoked_insufficient_candidates")
        return candidates[:k]

    from search.listwise_sonnet_reranker import listwise_rerank_with_sonnet

    n_to_rerank = min(LLM_RERANK_POOL, len(candidates))
    rerank_input = [
        {
            "chunk_id": result.chunk_id,
            "file_path": result.relative_path,
            "name": result.name,
            "parent_name": result.parent_name,
            "chunk_type": result.chunk_type,
            "start_line": result.start_line,
            "end_line": result.end_line,
            "content_preview": _full_content(metadata_lookup, result),
            "similarity_score": result.similarity_score,
            "_orig": result,
        }
        for result in candidates[:n_to_rerank]
    ]
    reranked, rerank_meta = listwise_rerank_with_sonnet(
        query,
        rerank_input,
        top_k=k,
        timeout=config.listwise_timeout_s,
        return_metadata=True,
    )
    searcher.last_reranker_metadata = rerank_meta
    new_top = [item["_orig"] for item in reranked]
    return (new_top + candidates[n_to_rerank:])[:k]


@register_reranker("cross-encoder")
def rerank_cross_encoder(
    searcher: Any,
    query: str,
    *,
    k: int,
    config: Any,
    candidates: List[SearchResult],
    metadata_lookup: dict,
) -> List[SearchResult]:
    """Legacy local cross-encoder rerank over every candidate."""
    from search.reranker import rerank_results

    rerank_input = [
        {
            "chunk_id": result.chunk_id,
            "content": result.content_preview,
            "score": result.similarity_score,
            "result": result,
        }
        for result in candidates
    ]
    reranked = rerank_results(query, rerank_input, top_k=k)
    ranked = [item["result"] for item in reranked]
    for item, candidate in zip(reranked, ranked, strict=False):
        candidate.similarity_score = item.get(
            "rerank_score",
            candidate.similarity_score,
        )
    # Historical behaviour: cross-encoder mode reports itself as not applied.
    _not_invoked(searcher, "not_invoked_cross_encoder_mode")
    return ranked[:k]


@register_reranker("off")
def rerank_off(
    searcher: Any,
    query: str,
    *,
    k: int,
    config: Any,
    candidates: List[SearchResult],
    metadata_lookup: dict,
) -> List[SearchResult]:
    """No reranking: hybrid order by fused score."""
    _not_invoked(searcher, "disabled_by_env")
    candidates.sort(key=lambda result: result.similarity_score, reverse=True)
    return candidates[:k]
