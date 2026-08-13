"""Candidate retrieval, rank fusion, and deterministic hybrid boosts."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from search.query_signals import (
    build_lexical_query,
    calculate_artifact_role_boost,
    calculate_signal_boost,
    extract_query_signals,
)
from search.result_models import SearchResult


@dataclass
class HybridRetrieval:
    """Post-retrieval state consumed by the optional ranking pipeline."""

    candidates: list[SearchResult]
    metadata_lookup: dict[str, dict[str, Any]]


def dedupe_candidates_by_file(candidates: list[SearchResult]) -> list[SearchResult]:
    """Keep the highest-ranked chunk for each file without reordering files."""

    seen: set[str] = set()
    diversified: list[SearchResult] = []
    for candidate in candidates:
        path = candidate.relative_path or candidate.file_path
        if path in seen:
            continue
        seen.add(path)
        diversified.append(candidate)
    return diversified


def rerank_raw_with_query_signals(
    raw_results: list[tuple[str, float, dict[str, Any]]],
    signals: Any,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Promote exact code signals before rank-only fusion discards magnitude."""

    if not signals.explicit:
        return raw_results

    def adjusted(item: tuple[str, float, dict[str, Any]]) -> tuple[float, str]:
        chunk_id, score, metadata = item
        boost = calculate_signal_boost(
            signals,
            relative_path=str(metadata.get("relative_path") or ""),
            name=metadata.get("name"),
            parent_name=metadata.get("parent_name"),
            full_content=str(
                metadata.get("full_content")
                or metadata.get("content")
                or metadata.get("content_preview")
                or ""
            ),
        )
        return (-abs(float(score)) * boost, chunk_id)

    return sorted(raw_results, key=adjusted)


def retrieve_hybrid_candidates(
    searcher: Any,
    query: str,
    *,
    k: int,
    context_depth: int,
    filters: dict[str, Any] | None,
    config: Any,
    chunk_type_boosts: Mapping[str, Mapping[str, float]],
    expand_query: Callable[[str], str],
    fuse_results: Callable[..., list[tuple[str, float]]],
) -> HybridRetrieval:
    """Retrieve, fuse, materialize, boost, and sort hybrid candidates."""
    from search.config import resolve_hybrid_weights

    signals = extract_query_signals(query)
    candidate_k = min(200, max(50, k * 4 if signals.explicit else 50))
    vector_weight, bm25_weight = resolve_hybrid_weights(config)
    if (
        signals.explicit
        and config.vector_weight == 0
        and config.bm25_weight == 0
        and config.content_mode == "code"
    ):
        vector_weight, bm25_weight = (0.35, 0.65)

    optimized_query = searcher._optimize_query(query)
    query_embedding = searcher._get_query_embedding(optimized_query)
    vector_raw = searcher.index_manager.search(
        query_embedding,
        candidate_k,
        filters,
    )
    vector_raw = rerank_raw_with_query_signals(vector_raw, signals)
    vector_pairs = [
        (chunk_id, similarity)
        for chunk_id, similarity, _metadata in vector_raw
    ]

    # Preserve the established order: optional LLM rewrite first, static
    # expansion second, then BM25 retrieval.
    bm25_query = build_lexical_query(query)
    if config.bm25_rewrite:
        from search.query_rewriter import rewrite_query_for_bm25

        bm25_query = rewrite_query_for_bm25(query)
    if config.query_expansion:
        bm25_query = expand_query(bm25_query)
    bm25_raw = searcher.index_manager.search_bm25(
        bm25_query,
        k=candidate_k,
        filters=filters,
    )
    bm25_raw = rerank_raw_with_query_signals(bm25_raw, signals)
    bm25_pairs = [
        (chunk_id, rank)
        for chunk_id, rank, _metadata in bm25_raw
    ]

    fused = fuse_results(
        vector_pairs,
        bm25_pairs,
        k=config.fusion_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )

    metadata_lookup: dict[str, dict[str, Any]] = {}
    for chunk_id, _similarity, metadata in vector_raw:
        metadata_lookup[chunk_id] = metadata
    for chunk_id, _rank, metadata in bm25_raw:
        if chunk_id not in metadata_lookup:
            metadata_lookup[chunk_id] = metadata

    over_fetch = min(k * 3, len(fused))
    candidates: list[SearchResult] = []
    for chunk_id, rrf_score in fused[:over_fetch]:
        metadata = metadata_lookup.get(chunk_id)
        if metadata:
            candidates.append(
                searcher._create_search_result(
                    chunk_id,
                    rrf_score,
                    metadata,
                    context_depth,
                )
            )

    # Deployment overrides layer on the current module-level policy passed by
    # searcher.py. Passing the mapping preserves the legacy
    # search.searcher.CHUNK_TYPE_BOOSTS monkey-patch seam.
    boosts = dict(chunk_type_boosts.get(config.content_mode, {}))
    override_raw = os.environ.get("CHUNK_TYPE_BOOST_OVERRIDE")
    if override_raw:
        try:
            override = json.loads(override_raw)
            if isinstance(override, dict):
                for chunk_type_key, boost_value in override.items():
                    boosts[chunk_type_key] = float(boost_value)
        except (ValueError, TypeError):
            pass

    query_tokens = searcher._normalize_to_tokens(query.lower())
    for result in candidates:
        if boosts:
            result.similarity_score *= boosts.get(result.chunk_type, 1.0)

        name_boost = searcher._calculate_name_boost(
            result.name,
            query,
            query_tokens,
        )
        if name_boost > 1.0:
            name_boost = 1.0 + (name_boost - 1.0) * 2.0
        result.similarity_score *= name_boost

        path_boost = searcher._calculate_path_boost(
            result.relative_path,
            query_tokens,
        )
        if path_boost > 1.0:
            path_boost = 1.0 + (path_boost - 1.0) * 3.0
        result.similarity_score *= path_boost

        metadata = metadata_lookup.get(result.chunk_id, {}) or {}
        result.similarity_score *= calculate_signal_boost(
            signals,
            relative_path=result.relative_path,
            name=result.name,
            parent_name=result.parent_name,
            full_content=str(
                metadata.get("full_content")
                or metadata.get("content")
                or result.content_preview
                or ""
            ),
        )
        result.similarity_score *= calculate_artifact_role_boost(
            query,
            relative_path=result.relative_path,
            content_mode=config.content_mode,
        )

    candidates.sort(key=lambda result: result.similarity_score, reverse=True)
    candidates = dedupe_candidates_by_file(candidates)
    return HybridRetrieval(
        candidates=candidates,
        metadata_lookup=metadata_lookup,
    )
