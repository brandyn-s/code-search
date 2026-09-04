"""Optional post-retrieval ranking stages for hybrid search.

PPR blending lives here; reranker modes are looked up in
``search.reranker_registry`` so adding a mode never touches this file.
"""

from __future__ import annotations

from typing import Any

from search.reranker_registry import get_reranker
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

    if not candidates:
        searcher.last_reranker_metadata = {
            "applied": False,
            "reason": "not_invoked_no_candidates",
            "latency_ms": 0,
        }
        return []

    rerank = get_reranker(config.reranker_mode)
    return rerank(
        searcher,
        query,
        k=k,
        config=config,
        candidates=candidates,
        metadata_lookup=metadata_lookup,
    )
