"""Optional post-retrieval ranking stages for hybrid search."""

from __future__ import annotations

from typing import Any

from search.result_models import SearchResult


def run_hybrid_pipeline(
    searcher: Any,
    query: str,
    *,
    k: int,
    config: Any,
    candidates: list[SearchResult],
    metadata_lookup: dict[str, dict[str, Any]],
) -> list[SearchResult]:
    """Apply PPR and the configured reranker while surfacing metadata."""
    import time as _time

    from search.ppr_scorer import (
        PPRScorer,
        blend_ppr_into_candidates,
        get_env_config,
    )

    ppr_enabled, ppr_alpha = get_env_config()
    ppr_start = _time.monotonic()
    if not ppr_enabled:
        searcher.last_ppr_metadata = {
            "applied": False,
            "reason": "disabled_by_env",
            "latency_ms": 0,
        }
    elif ppr_alpha == 0.0:
        searcher.last_ppr_metadata = {
            "applied": False,
            "reason": "alpha_zero",
            "latency_ms": 0,
        }
    elif not candidates:
        searcher.last_ppr_metadata = {
            "applied": False,
            "reason": "no_candidates",
            "latency_ms": 0,
        }
    else:
        try:
            hint = None
            for candidate in candidates:
                abs_path = (
                    getattr(candidate, "file_path", None)
                    or getattr(candidate, "absolute_path", None)
                )
                if abs_path:
                    hint = str(abs_path)
                    break
            with PPRScorer() as ppr:
                candidate_path_scores = [
                    (candidate.relative_path, candidate.similarity_score)
                    for candidate in candidates
                ]
                ppr_scores = ppr.score(
                    candidate_path_scores,
                    hint_abs_path=hint,
                )
            latency_ms = int((_time.monotonic() - ppr_start) * 1000)
            if ppr_scores:
                blend_ppr_into_candidates(candidates, ppr_alpha, ppr_scores)
                candidates.sort(
                    key=lambda result: result.similarity_score,
                    reverse=True,
                )
                searcher.last_ppr_metadata = {
                    "applied": True,
                    "reason": "ok",
                    "latency_ms": latency_ms,
                    "scored_candidates": len(ppr_scores),
                    "alpha": ppr_alpha,
                }
            else:
                searcher.last_ppr_metadata = {
                    "applied": False,
                    "reason": "no_graph_db",
                    "latency_ms": latency_ms,
                }
        except Exception as ppr_err:  # noqa: BLE001 - graceful PPR fallback
            searcher._logger.warning(
                "[PPR_DIAG] ppr_blend_failed err=%s",
                ppr_err,
            )
            searcher.last_ppr_metadata = {
                "applied": False,
                "reason": "error",
                "latency_ms": int(
                    (_time.monotonic() - ppr_start) * 1000
                ),
                "error_class": type(ppr_err).__name__,
            }

    rerank_mode = config.reranker_mode
    if not candidates:
        searcher.last_reranker_metadata = {
            "applied": False,
            "reason": "not_invoked_no_candidates",
            "latency_ms": 0,
        }
        return []

    if rerank_mode == "sonnet" and len(candidates) > k:
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

        n_to_rerank = min(15, len(candidates))
        top_candidates = candidates[:n_to_rerank]
        rerank_input = []
        for result in top_candidates:
            metadata = metadata_lookup.get(result.chunk_id, {}) or {}
            full_content = (
                metadata.get("full_content")
                or metadata.get("content")
                or result.content_preview
                or ""
            )
            rerank_input.append(
                {
                    "chunk_id": result.chunk_id,
                    "file_path": result.relative_path,
                    "full_content": full_content,
                    "_orig": result,
                }
            )
        reranked, rerank_meta = rerank_with_sonnet(
            query,
            rerank_input,
            top_k=k,
            return_metadata=True,
        )
        searcher.last_reranker_metadata = rerank_meta
        new_top = [item["_orig"] for item in reranked]
        tail = candidates[n_to_rerank:]
        candidates = new_top + tail
    elif rerank_mode == "sonnet" and len(candidates) <= k:
        searcher.last_reranker_metadata = {
            "applied": False,
            "reason": "not_invoked_insufficient_candidates",
            "latency_ms": 0,
        }
    elif rerank_mode == "listwise" and len(candidates) > k:
        from search.listwise_sonnet_reranker import (
            listwise_rerank_with_sonnet,
        )

        n_to_rerank = min(15, len(candidates))
        top_candidates = candidates[:n_to_rerank]
        rerank_input = []
        for result in top_candidates:
            metadata = metadata_lookup.get(result.chunk_id, {}) or {}
            full_content = (
                metadata.get("full_content")
                or metadata.get("content")
                or result.content_preview
                or ""
            )
            rerank_input.append(
                {
                    "chunk_id": result.chunk_id,
                    "file_path": result.relative_path,
                    "name": result.name,
                    "parent_name": result.parent_name,
                    "chunk_type": result.chunk_type,
                    "start_line": result.start_line,
                    "end_line": result.end_line,
                    "content_preview": full_content,
                    "similarity_score": result.similarity_score,
                    "_orig": result,
                }
            )
        reranked, rerank_meta = listwise_rerank_with_sonnet(
            query,
            rerank_input,
            top_k=k,
            timeout=config.listwise_timeout_s,
            return_metadata=True,
        )
        searcher.last_reranker_metadata = rerank_meta
        new_top = [item["_orig"] for item in reranked]
        tail = candidates[n_to_rerank:]
        candidates = new_top + tail
    elif rerank_mode == "listwise" and len(candidates) <= k:
        searcher.last_reranker_metadata = {
            "applied": False,
            "reason": "not_invoked_insufficient_candidates",
            "latency_ms": 0,
        }
    elif rerank_mode == "cross-encoder" and candidates:
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
        candidates = [item["result"] for item in reranked]
        for item, candidate in zip(reranked, candidates, strict=False):
            candidate.similarity_score = item.get(
                "rerank_score",
                candidate.similarity_score,
            )
        searcher.last_reranker_metadata = {
            "applied": False,
            "reason": "not_invoked_cross_encoder_mode",
            "latency_ms": 0,
        }
    else:
        searcher.last_reranker_metadata = {
            "applied": False,
            "reason": "disabled_by_env",
            "latency_ms": 0,
        }
        candidates.sort(
            key=lambda result: result.similarity_score,
            reverse=True,
        )
    return candidates[:k]
