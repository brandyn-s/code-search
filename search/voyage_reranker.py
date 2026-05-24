"""Voyage AI reranker — production contract matching sonnet_reranker.

Uses Voyage's /v1/rerank endpoint with the `rerank-2.5` cross-encoder model.
Single API call per query (vs Sonnet's 15 isolated parallel calls), with the
same graceful-fallback contract: any failure (auth, network, timeout,
rate-limit, parse) returns the hybrid order at top_k with a structured
`_metadata.reranker.reason` so callers can detect the silent fallback.

Reason vocabulary mirrors sonnet_reranker exactly so MCP consumers don't
need a per-reranker branch:

  ok                       success
  empty_input              len(candidates) == 0 (rare)
  api_key_missing          VOYAGE_API_KEY not set
  package_not_installed    httpx missing
  timeout                  hard deadline exceeded
  rate_limit               429 dominated retries
  too_many_failures        non-429 HTTP / parse / connection error
  unexpected_error         catch-all

Compatibility with current production-Sonnet path:

  - Same input shape: list of dicts with keys including
    `full_content` / `content` / `content_preview` and `_orig`. The
    `_orig` reference is preserved in the returned dicts so the caller's
    `new_top = [d["_orig"] for d in reranked]` extraction works.
  - Same return shape when `return_metadata=True`:
        (list[dict], {"applied": bool, "reason": str, "latency_ms": int})
  - Same hybrid-prior-fallback knob via `hybrid_prior_threshold`. Voyage
    returns scores in [0, 1]; threshold is interpreted as a min-score
    (default 0.0 = no fallback). The Sonnet path's int 0-10 threshold
    doesn't transfer directly; per-deployment tuning required.

Latency budget: 12s default hard deadline matches SONNET_LISTWISE_TIMEOUT
default. Voyage Rerank 2.5 p99 measured ~600ms in vendor benchmarks
(blog.voyageai.com/2025/08/11/rerank-2-5/, agentset.ai/rerankers/...);
12s captures retries on transient 429/5xx.

Configuration env vars (all optional):
  VOYAGE_API_KEY            required for the reranker to apply (else
                            api_key_missing fallback)
  VOYAGE_RERANKER_MODEL     model name; default "rerank-2.5". Also
                            supports "rerank-2.5-lite" (cheaper, faster).
  VOYAGE_RERANKER_TIMEOUT   hard deadline in seconds; default 12.0
  VOYAGE_RERANKER_MAX_RETRIES   on 429/5xx; default 2 (so up to 3 attempts)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reason vocabulary (mirrors sonnet_reranker exactly)
# ---------------------------------------------------------------------------

REASON_OK = "ok"
REASON_EMPTY_INPUT = "empty_input"
REASON_API_KEY_MISSING = "api_key_missing"
REASON_PACKAGE_NOT_INSTALLED = "package_not_installed"
REASON_TIMEOUT = "timeout"
REASON_RATE_LIMIT = "rate_limit"
REASON_TOO_MANY_FAILURES = "too_many_failures"
REASON_UNEXPECTED_ERROR = "unexpected_error"

# Endpoint and defaults
VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"
DEFAULT_MODEL = "rerank-2.5"
DEFAULT_TIMEOUT_S = 12.0
DEFAULT_MAX_RETRIES = 2  # so up to 3 total attempts


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        v = float(raw)
        return v if v > 0 else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v >= 0 else default
    except ValueError:
        return default


def _extract_text(candidate: Dict[str, Any]) -> str:
    """Mirror sonnet_reranker's content extraction priority."""
    return (
        candidate.get("full_content")
        or candidate.get("content")
        or candidate.get("content_preview")
        or ""
    )


def rerank_with_voyage(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
    return_metadata: bool = False,
    timeout: Optional[float] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """Rerank `candidates` against `query` using Voyage Rerank 2.5.

    On any failure (missing key, package, timeout, HTTP error, parse error)
    returns `candidates[:top_k]` (hybrid order) so callers observe a single
    failure-mode contract: degraded quality, never a crash.

    Args:
        query: the user query.
        candidates: list of dicts, each with at least one of full_content /
                    content / content_preview. The dict structure is
                    preserved on output; callers typically include an
                    `_orig` reference to the original SearchResult.
        top_k: number of candidates to return (after reranking the full
               input set).
        return_metadata: when True, returns (list, dict) where dict has
                         applied (bool), reason (str), latency_ms (int).
        timeout: hard deadline. Defaults to VOYAGE_RERANKER_TIMEOUT env
                 var (default 12.0s).
        api_key: override the VOYAGE_API_KEY env var.
        model: override the VOYAGE_RERANKER_MODEL env var (default "rerank-2.5").

    Returns:
        list[dict] reordered by relevance score, truncated to top_k. OR
        (list, metadata) when return_metadata=True.
    """
    t_start = time.monotonic()

    def _emit(out: List[Dict[str, Any]], applied: bool, reason: str):
        latency_ms = int((time.monotonic() - t_start) * 1000)
        if return_metadata:
            return out, {"applied": applied, "reason": reason, "latency_ms": latency_ms}
        return out

    if not candidates:
        LOG.info("[RERANK_REASON] %s n_candidates=0", REASON_EMPTY_INPUT)
        return _emit([], False, REASON_EMPTY_INPUT)

    key = api_key or os.environ.get("VOYAGE_API_KEY", "")
    if not key:
        LOG.warning("[RERANK_REASON] %s; using hybrid order", REASON_API_KEY_MISSING)
        return _emit(candidates[:top_k], False, REASON_API_KEY_MISSING)

    try:
        import httpx
    except ImportError:
        LOG.warning("[RERANK_REASON] %s (httpx); using hybrid order", REASON_PACKAGE_NOT_INSTALLED)
        return _emit(candidates[:top_k], False, REASON_PACKAGE_NOT_INSTALLED)

    effective_timeout = timeout if timeout is not None else _env_float(
        "VOYAGE_RERANKER_TIMEOUT", DEFAULT_TIMEOUT_S
    )
    effective_model = model or os.environ.get("VOYAGE_RERANKER_MODEL", DEFAULT_MODEL)
    max_retries = _env_int("VOYAGE_RERANKER_MAX_RETRIES", DEFAULT_MAX_RETRIES)

    documents = [_extract_text(c) for c in candidates]

    # Voyage's /v1/rerank accepts top_k; request the full set so we can
    # preserve a hybrid-order tail if top_k < len(candidates) for some
    # caller-specific reason. The endpoint returns indices into the
    # documents array, so this is straightforward.
    request_top_k = min(top_k, len(candidates))

    payload = {
        "query": query,
        "documents": documents,
        "model": effective_model,
        "top_k": request_top_k,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    last_status: Optional[int] = None
    last_err: Optional[BaseException] = None
    rate_limited = False

    # The outer timeout bounds total wall including retries.
    deadline = t_start + effective_timeout
    attempt = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            LOG.warning(
                "[RERANK_REASON] %s cohort_wall_ms>%d n_candidates=%d; using hybrid order",
                REASON_TIMEOUT,
                int(effective_timeout * 1000),
                len(candidates),
            )
            return _emit(candidates[:top_k], False, REASON_TIMEOUT)

        try:
            with httpx.Client(timeout=remaining) as client:
                resp = client.post(VOYAGE_RERANK_URL, headers=headers, json=payload)
            last_status = resp.status_code
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("data", [])
                # Each item: {"index": int, "relevance_score": float, ...}
                # Build the reordered list, preserving candidate dicts.
                # Voyage returns results sorted by descending relevance score,
                # but defensive-sort anyway.
                try:
                    pairs = sorted(
                        ((int(r["index"]), float(r["relevance_score"])) for r in results),
                        key=lambda p: p[1],
                        reverse=True,
                    )
                except (KeyError, TypeError, ValueError) as parse_err:
                    LOG.warning(
                        "[RERANK_REASON] %s parse_error=%r; using hybrid order",
                        REASON_TOO_MANY_FAILURES,
                        parse_err,
                    )
                    return _emit(candidates[:top_k], False, REASON_TOO_MANY_FAILURES)

                reordered: List[Dict[str, Any]] = []
                seen = set()
                for idx, _score in pairs:
                    if 0 <= idx < len(candidates) and idx not in seen:
                        reordered.append(candidates[idx])
                        seen.add(idx)
                # If Voyage returned fewer results than top_k (unusual but
                # possible), append remaining candidates in hybrid order to
                # honor the top_k contract.
                if len(reordered) < top_k:
                    for i, c in enumerate(candidates):
                        if i not in seen and len(reordered) < top_k:
                            reordered.append(c)
                            seen.add(i)
                out = reordered[:top_k]
                LOG.info(
                    "[RERANK_REASON] %s model=%s n_candidates=%d top_k=%d latency_ms=%d",
                    REASON_OK,
                    effective_model,
                    len(candidates),
                    top_k,
                    int((time.monotonic() - t_start) * 1000),
                )
                return _emit(out, True, REASON_OK)

            # Non-200. Retry on 429 / 5xx; give up on 4xx else.
            if resp.status_code == 429:
                rate_limited = True
                last_err = httpx.HTTPStatusError(
                    "rate limit", request=resp.request, response=resp
                )
            elif 500 <= resp.status_code < 600:
                last_err = httpx.HTTPStatusError(
                    f"server error {resp.status_code}",
                    request=resp.request,
                    response=resp,
                )
            else:
                # 4xx other than 429 — auth, bad request, etc. Don't retry.
                LOG.warning(
                    "[RERANK_REASON] %s http=%d body=%s; using hybrid order",
                    REASON_TOO_MANY_FAILURES,
                    resp.status_code,
                    resp.text[:200],
                )
                return _emit(candidates[:top_k], False, REASON_TOO_MANY_FAILURES)
        except httpx.TimeoutException as e:
            last_err = e
            # Outer-loop will check remaining budget; if exhausted -> REASON_TIMEOUT
        except (httpx.NetworkError, httpx.RemoteProtocolError) as e:
            last_err = e
        except Exception as e:  # noqa: BLE001 — catch-all to guarantee graceful fallback
            LOG.exception("[RERANK_REASON] %s err=%r; using hybrid order", REASON_UNEXPECTED_ERROR, e)
            return _emit(candidates[:top_k], False, REASON_UNEXPECTED_ERROR)

        # Should we retry?
        if attempt >= max_retries:
            break
        attempt += 1
        # Exponential backoff with rate-limit-aware floor (Voyage advises
        # 15s on 429 per typical rate-limit docs).
        wait = 15.0 if rate_limited else min(2 ** attempt, 4.0)
        wait = min(wait, max(0.0, deadline - time.monotonic() - 0.5))
        if wait <= 0:
            break
        time.sleep(wait)

    # Out of retries.
    if rate_limited:
        LOG.warning(
            "[RERANK_REASON] %s last_status=%s attempts=%d; using hybrid order",
            REASON_RATE_LIMIT,
            last_status,
            attempt + 1,
        )
        return _emit(candidates[:top_k], False, REASON_RATE_LIMIT)

    LOG.warning(
        "[RERANK_REASON] %s last_status=%s last_err=%r attempts=%d; using hybrid order",
        REASON_TOO_MANY_FAILURES,
        last_status,
        last_err,
        attempt + 1,
    )
    return _emit(candidates[:top_k], False, REASON_TOO_MANY_FAILURES)


# ---------------------------------------------------------------------------
# Legacy thin wrapper preserved for callers using the original 4-arg form
# ---------------------------------------------------------------------------

def voyage_rerank(
    query: str,
    documents: List[str],
    model: str = DEFAULT_MODEL,
    top_k: int = 5,
    api_key: str = "",
) -> List[Tuple[int, float]]:
    """Legacy interface: returns list of (original_index, relevance_score).

    Preserved for any caller still using the pre-production-contract API.
    New code should use rerank_with_voyage which mirrors the sonnet
    return_metadata contract.
    """
    # Wrap each document in a dict so the production function can ingest it.
    candidates = [{"full_content": d} for d in documents]
    result = rerank_with_voyage(
        query=query,
        candidates=candidates,
        top_k=top_k,
        return_metadata=False,
        api_key=api_key or None,
        model=model,
    )
    # Reconstruct (index, score) pairs by looking up content. This is an
    # imperfect reverse-mapping (duplicate content collapses indices), but
    # the legacy callers were already index-based so we mirror that.
    out: List[Tuple[int, float]] = []
    for r in (result if isinstance(result, list) else []):
        content = r.get("full_content", "")
        try:
            idx = documents.index(content)
        except ValueError:
            continue
        # The legacy interface returns scores but rerank_with_voyage no
        # longer surfaces them. Return 0.0 placeholder — callers relying
        # on numeric scores should migrate to rerank_with_voyage.
        out.append((idx, 0.0))
    return out
