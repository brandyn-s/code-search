"""Sonnet 4.6 reranker — query-time relevance scoring via Anthropic API.

Validated 2026-05-03 via D4b (PR #93+): on n=183 multi-target real_session,
hybrid baseline MRR=0.763 → Sonnet rerank MRR=0.850 (+0.087, +0.137 HR@1).
Reranks top-15 hybrid candidates by Sonnet relevance score, returns top-k.

ALWAYS-ON CONTRACT: this module never raises. Any failure (missing API key,
timeout, HTTP error, parse error) returns the input candidates unchanged
in their existing order. Code-search must continue to function when
Anthropic API is unavailable.

Disable explicitly with `RERANKER=off`. Switch to legacy cross-encoder with
`RERANKER=cross-encoder`. Default mode is `sonnet` (new 2026-05-03).

Latency budget: 8s total (configurable via SONNET_RERANKER_TIMEOUT). Top-15
parallel calls typically complete in 1.5-2.5s on Anthropic real-time API.

Cost: ~$0.005/query at 15 candidates × ~5K tokens × $1.50/M input (real-time
Sonnet 4.6 pricing).

Observability (PR Plan-2 A1, 2026-05-05): callers can opt into structured
metadata via `rerank_with_sonnet(..., return_metadata=True)`. Returns
`(reranked_list, {"applied": bool, "reason": str, "latency_ms": int})`.
Reason values are documented in `REASON_*` constants below. The MCP search
response surfaces this in `_metadata.reranker` so LLM agents can detect
silent fallback (e.g., rotated API key, sustained rate-limit, sustained
hybrid-prior fallback indicating prompt-coverage issues).

Latency diagnostics (Plan D1-Pass-2 A.1, 2026-05-06): each Anthropic call
emits a `[ANTHROPIC_DIAG]` log line with `total_ms`, `in_flight`,
`attempt_seq`, and `outcome`. Optional concurrency cap via
`ANTHROPIC_CONCURRENCY_LIMIT` env var (unbounded when unset, preserving
existing behavior). Together these expose whether degraded latency is
constant-rate (server-side), tail-heavy (per-call slow), concurrency-
related, or rate-limit-driven — see
`bench/research/anthropic_latency_diagnosis.md`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

LOG = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_CONTENT_CHARS = 4000
DEFAULT_TIMEOUT = 8.0
DEFAULT_RERANK_K = 15  # rerank top-15, return top-k of those (D4b validated)
FAILURE_TOLERANCE = 0.3  # if >30% of calls fail, abort and use input order

# ─── Phase B.1 SDK retry/timeout knobs (Plan D1-Pass-2, 2026-05-06) ───
# Default SDK max_retries=2 means 3 total attempts per call. When the API
# returns 429/5xx, the SDK retries with backoff before raising; on retry
# exhaustion we observe ~7.5s of wall time per failure (3 attempts × ~2.5s)
# even though the cohort already has its own FAILURE_TOLERANCE=0.3 fallback.
# Lowering to max_retries=1 cuts retry overhead in half without disabling
# transient-error recovery.
DEFAULT_SDK_MAX_RETRIES = 1
# Default per-call SDK timeout=NOT_GIVEN (no per-call cap). Phase A.2
# diagnostic showed p95 successful=5.9s, p99 successful=8.6s; setting a
# per-call cap above p99 successful (12s here) catches genuinely-stuck calls
# without truncating healthy slow ones. The cohort-level
# SONNET_RERANKER_TIMEOUT (default 8s) is the OUTER bound on the entire
# `asyncio.gather` — it serves as a backstop, not a per-call control.
DEFAULT_SDK_PER_CALL_TIMEOUT_S = 12.0
# If max Sonnet score across the candidate pool is below this threshold,
# Sonnet has not identified anything confidently relevant. When the judge
# is uncertain, Sonnet's score-tie-breaking pushes canonical files down
# via chunk-keyword density. Falling back to hybrid order in those cases
# recovers MRR/HR. Threshold=6 ("Partially relevant or above") empirically
# beats 5 and 7 on n=183 multi-target real_session — see eval_v4/run_a0c_*.
# Default=7 (PR #95) was too aggressive; tuned to 6 in PR #96.
DEFAULT_HYBRID_PRIOR_THRESHOLD = 6

# ─── Structured-metadata reason vocabulary ───
# Stable strings consumed by callers (MCP layer, eval harnesses). Add new
# entries here rather than ad-hoc reason strings; downstream consumers will
# treat unknown strings as "unexpected_error" by default.
REASON_OK = "ok"
REASON_EMPTY_INPUT = "empty_input"
REASON_API_KEY_MISSING = "api_key_missing"
REASON_PACKAGE_NOT_INSTALLED = "package_not_installed"
REASON_TIMEOUT = "timeout"
REASON_RATE_LIMIT = "rate_limit"
REASON_TOO_MANY_FAILURES = "too_many_failures"
REASON_HYBRID_PRIOR_FALLBACK = "hybrid_prior_fallback"
REASON_ASYNC_CONTEXT = "async_context"
REASON_UNEXPECTED_ERROR = "unexpected_error"

# Per-call error tags. _score_one returns one of these strings on failure.
_ERR_RATE_LIMIT = "_err_rate_limit"
_ERR_TIMEOUT = "_err_timeout"
_ERR_HTTP = "_err_http"
_ERR_UNPARSEABLE = "_err_unparseable"
_ERR_EMPTY = "_err_empty"

JUDGE_PROMPT = """You are evaluating whether a code chunk is relevant to a developer search query.

Query: {query}

Code chunk (file: {file_path}):
```
{content}
```

Rate the relevance on a scale of 0-10:
- 10 = This chunk IS exactly what the user is searching for
- 7-9 = Highly relevant; clearly matches the user's intent
- 4-6 = Partially relevant; related but not the primary target
- 1-3 = Tangentially related
- 0 = Not relevant at all

Respond with ONLY valid JSON:
{{"score": <int 0-10>, "reasoning": "<one sentence>"}}"""


def _classify_call_error(exc: BaseException) -> str:
    """Classify a per-call exception into one of the _ERR_* tags.

    Walks the exception's `__cause__` chain so retry-exhausted errors that
    arrive as wrappers still surface their underlying cause. Phase B.1 fix:
    before this change, the SDK's retry-exhaustion produced a wrapper whose
    name/message didn't contain "rate"/"limit"/"429", causing genuine rate
    limits to be misclassified as generic _ERR_HTTP and losing diagnostic
    signal in `_metadata.reranker.reason`.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        cls_name = type(cur).__name__
        msg = str(cur).lower()
        # Anthropic SDK exception class names (RateLimitError, APITimeoutError,
        # etc.) checked alongside string-based detection so we catch both
        # explicitly-typed exceptions and stringified wrappers.
        if (
            "ratelimit" in cls_name.lower()
            or "rate_limit" in cls_name.lower()
            or ("rate" in msg and "limit" in msg)
            or "429" in msg
        ):
            return _ERR_RATE_LIMIT
        if "timeout" in cls_name.lower() or "timeout" in msg:
            return _ERR_TIMEOUT
        # Walk the cause chain (one common SDK pattern: APIConnectionError
        # wraps an httpx ReadTimeout)
        cur = cur.__cause__ or cur.__context__
    return _ERR_HTTP


# ─── Latency diagnostic state (Plan D1-Pass-2 A.1) ───
# Module-level counter for in-flight Sonnet calls. Updated under
# _IN_FLIGHT_LOCK so the diag log line reflects a consistent snapshot.
_IN_FLIGHT_COUNT = 0
_IN_FLIGHT_LOCK: asyncio.Lock | None = None
_ATTEMPT_SEQ = 0


def _get_in_flight_lock() -> asyncio.Lock:
    """Lazy-create the lock so we attach it to the running event loop."""
    global _IN_FLIGHT_LOCK
    if _IN_FLIGHT_LOCK is None:
        _IN_FLIGHT_LOCK = asyncio.Lock()
    return _IN_FLIGHT_LOCK


def _resolve_per_call_timeout() -> float:
    """Resolve per-call SDK timeout from env, falling back to the default.

    `messages.create(timeout=...)` accepts a float seconds value; we read the
    env var here so tests can override per-test without touching defaults.
    """
    raw = os.environ.get("ANTHROPIC_PER_CALL_TIMEOUT_S")
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return DEFAULT_SDK_PER_CALL_TIMEOUT_S


def _resolve_sdk_max_retries() -> int:
    """Resolve SDK max_retries from env, falling back to the default."""
    raw = os.environ.get("ANTHROPIC_MAX_RETRIES")
    if raw:
        try:
            v = int(raw)
            if v >= 0:
                return v
        except ValueError:
            pass
    return DEFAULT_SDK_MAX_RETRIES


async def _score_one(client: Any, query: str, file_path: str, content: str):
    """Score one (query, chunk) pair.

    Returns:
        int 0-10 on success.
        None on legacy parse/empty paths (preserved for backward-compat with
            existing test monkey-patches that return int|None).
        str (one of _ERR_* tags) when a structured failure can be classified.

    Emits one `[ANTHROPIC_DIAG]` log line per call documenting wall-time and
    concurrency. Used to diagnose latency regressions — see PR Plan D1-Pass-2.
    """
    global _IN_FLIGHT_COUNT, _ATTEMPT_SEQ
    truncated = content[:MAX_CONTENT_CHARS] if len(content) > MAX_CONTENT_CHARS else content
    prompt = JUDGE_PROMPT.format(
        query=query,
        file_path=file_path or "(unknown)",
        content=truncated or "(empty content)",
    )

    # Snapshot in-flight + attempt-seq under lock so the diag log line is
    # consistent. Increment counter, capture seq, then make the SDK call.
    lock = _get_in_flight_lock()
    async with lock:
        _ATTEMPT_SEQ += 1
        attempt_seq = _ATTEMPT_SEQ
        _IN_FLIGHT_COUNT += 1
        in_flight = _IN_FLIGHT_COUNT

    t_start = time.monotonic()
    outcome = "ok"
    resp = None
    exc: BaseException | None = None
    per_call_timeout = _resolve_per_call_timeout()
    try:
        try:
            resp = await client.messages.create(
                model=MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
                timeout=per_call_timeout,
            )
        except Exception as e:
            exc = e
            outcome = _classify_call_error(e).removeprefix("_err_")
    finally:
        t_total_ms = int((time.monotonic() - t_start) * 1000)
        async with lock:
            _IN_FLIGHT_COUNT -= 1
        LOG.info(
            f"[ANTHROPIC_DIAG] model={MODEL} total_ms={t_total_ms} "
            f"in_flight={in_flight} attempt={attempt_seq} outcome={outcome}"
        )

    if exc is not None:
        LOG.debug(f"Sonnet score call failed: {exc}")
        return _classify_call_error(exc)

    try:
        text = ""
        for block in resp.content:  # type: ignore[union-attr]
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", "").strip()
                break
        if not text:
            return _ERR_EMPTY
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(line for line in lines if not line.startswith("```"))
        obj = json.loads(text)
        score = int(obj.get("score", 0))
        return max(0, min(10, score))  # clamp to [0, 10]
    except Exception as e:
        LOG.debug(f"Sonnet score parse failed: {e}")
        return _ERR_UNPARSEABLE


async def _bounded_score_one(sem: asyncio.Semaphore | None, *args):
    """Wrap _score_one with optional semaphore-bounded concurrency.

    When sem is None, behaves identically to _score_one (unbounded).
    When sem is set, acquires before the call so at most `sem._value`
    concurrent calls are in flight. Indirection lives outside _score_one
    so existing tests that monkey-patch _score_one with a 4-arg fake keep
    working.
    """
    if sem is None:
        return await _score_one(*args)
    async with sem:
        return await _score_one(*args)


def _aggregate_failure_reason(failures: list[str]) -> str:
    """Pick the most-frequent _ERR_* tag and map to a public REASON_*.

    Tie-breaking: rate_limit > timeout > everything else (rate_limit is the
    most actionable signal for an operator).
    """
    if not failures:
        return REASON_TOO_MANY_FAILURES
    counts: dict[str, int] = {}
    for f in failures:
        counts[f] = counts.get(f, 0) + 1
    if counts.get(_ERR_RATE_LIMIT, 0) > 0 and counts[_ERR_RATE_LIMIT] >= max(counts.values()) - 1:
        return REASON_RATE_LIMIT
    if counts.get(_ERR_TIMEOUT, 0) > 0 and counts[_ERR_TIMEOUT] >= max(counts.values()) - 1:
        return REASON_TIMEOUT
    return REASON_TOO_MANY_FAILURES


async def _rerank_async(
    query: str,
    candidates: list[dict],
    top_k: int,
    timeout: float,
    hybrid_prior_threshold: int,
    return_metadata: bool = False,
):
    """Score candidates in parallel, sort by score, return top-k.

    Falls back to input[:top_k] on timeout, too-many-failures, or any exception.

    When return_metadata=True returns (list, {applied, reason, latency_ms}).
    Otherwise returns list (preserves existing API for tests).
    """
    t_start = time.monotonic()

    def _emit(out_list: list[dict], applied: bool, reason: str):
        latency_ms = int((time.monotonic() - t_start) * 1000)
        if return_metadata:
            return out_list, {"applied": applied, "reason": reason, "latency_ms": latency_ms}
        return out_list

    try:
        import anthropic
    except ImportError:
        LOG.warning("anthropic package not installed; reranker disabled")
        return _emit(candidates[:top_k], False, REASON_PACKAGE_NOT_INSTALLED)

    # Plan-2 (2026-05-06 roundtable rec #1): wrap AsyncAnthropic in `async with`
    # so its underlying httpx AsyncClient is fully closed before this coroutine
    # returns. Without the explicit close, asyncio.run() tears down the event
    # loop while the client's connection-pool cleanup tasks are still scheduled,
    # producing "RuntimeError: Event loop is closed" warnings on every call.
    # This is cosmetic in long-lived MCP servers (the loop never closes) but
    # accumulates catastrophically in eval drivers that call asyncio.run per
    # query (~150+ warnings stalled D1 Pass 2 on 2026-05-06). The async-with
    # block guarantees client.aclose() completes inside this loop's lifetime.
    # Optional concurrency cap (Plan D1-Pass-2 A.1). When unset, behaves
    # exactly as before (unbounded asyncio.gather over all candidates). When
    # ANTHROPIC_CONCURRENCY_LIMIT is set to a positive integer, _score_one
    # acquires from a semaphore before each SDK call, capping in-flight
    # requests. Useful for debugging concurrency-related latency regressions.
    sem: asyncio.Semaphore | None = None
    try:
        limit_str = os.environ.get("ANTHROPIC_CONCURRENCY_LIMIT")
        if limit_str:
            limit = int(limit_str)
            if limit > 0:
                sem = asyncio.Semaphore(limit)
    except ValueError:
        sem = None

    sdk_max_retries = _resolve_sdk_max_retries()
    async with anthropic.AsyncAnthropic(max_retries=sdk_max_retries) as client:
        tasks = []
        for c in candidates:
            full = c.get("full_content") or c.get("content") or c.get("content_preview") or ""
            file_path = c.get("file_path") or c.get("file") or c.get("relative_path") or ""
            tasks.append(_bounded_score_one(sem, client, query, file_path, full))

        try:
            scores = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=False),
                                              timeout=timeout)
        except asyncio.TimeoutError:
            LOG.warning(f"Sonnet reranker timeout >{timeout}s; using hybrid order")
            return _emit(candidates[:top_k], False, REASON_TIMEOUT)

        # _score_one returns: int (success), None (legacy parse/empty paths), or str (_ERR_*).
        failures = [s for s in scores if isinstance(s, str)]
        n_failed = sum(1 for s in scores if not isinstance(s, int))
        if len(scores) > 0 and n_failed > len(scores) * FAILURE_TOLERANCE:
            reason = _aggregate_failure_reason(failures) if failures else REASON_TOO_MANY_FAILURES
            LOG.warning(f"Sonnet reranker {n_failed}/{len(scores)} failed ({reason}); using hybrid order")
            return _emit(candidates[:top_k], False, reason)

        # Hybrid-prior fallback: if the max Sonnet score across the candidate pool
        # is below the threshold (default 7 = "Highly relevant" boundary in
        # JUDGE_PROMPT), Sonnet hasn't identified anything as confidently relevant.
        # Tie-breaking on uniformly-low scores favors keyword-dense chunks over
        # canonical implementations. Preserve hybrid order in that case.
        valid_scores = [s for s in scores if isinstance(s, int)]
        if valid_scores and max(valid_scores) < hybrid_prior_threshold:
            LOG.debug(f"Sonnet max score {max(valid_scores)} < {hybrid_prior_threshold}; "
                      f"using hybrid order")
            return _emit(candidates[:top_k], False, REASON_HYBRID_PRIOR_FALLBACK)

        # Sort: higher score wins; non-int scores (None, _ERR_*) sink to bottom;
        # preserve original order on ties (stable sort).
        def _sort_key(item):
            idx, pair = item
            score = pair[0]
            is_failure = not isinstance(score, int)
            return (is_failure, -(score if isinstance(score, int) else -1), idx)

        indexed = list(enumerate(zip(scores, candidates)))
        indexed.sort(key=_sort_key)
        out = [c for _, (_, c) in indexed[:top_k]]
        return _emit(out, True, REASON_OK)


def rerank_with_sonnet(
    query: str,
    candidates: list[dict],
    top_k: int = 10,
    timeout: float | None = None,
    return_metadata: bool = False,
):
    """Rerank candidates by Sonnet 4.6 relevance score. Returns top-k.

    Args:
        query: original search query string
        candidates: list of dicts, each with at least one of {full_content,
            content, content_preview} and one of {file_path, file, relative_path}.
            Extra keys preserved in output.
        top_k: number of results to return after reranking
        timeout: total budget in seconds (default DEFAULT_TIMEOUT=8.0,
            override via SONNET_RERANKER_TIMEOUT env)
        return_metadata: when True, returns (list, dict) where dict has
            {applied: bool, reason: str (REASON_* constant), latency_ms: int}.
            When False (default), returns list only — preserves existing API.

    Returns:
        Reranked candidates[:top_k]. On any failure (no API key, timeout,
        HTTP error, parse failure, >30% call failures), returns
        candidates[:top_k] unchanged in input order.

    Never raises. Always-on contract.
    """
    t_start = time.monotonic()

    def _emit(out_list: list[dict], applied: bool, reason: str):
        latency_ms = int((time.monotonic() - t_start) * 1000)
        if return_metadata:
            return out_list, {"applied": applied, "reason": reason, "latency_ms": latency_ms}
        return out_list

    if not candidates:
        return _emit([], False, REASON_EMPTY_INPUT)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _emit(candidates[:top_k], False, REASON_API_KEY_MISSING)
    if timeout is None:
        try:
            timeout = float(os.environ.get("SONNET_RERANKER_TIMEOUT", DEFAULT_TIMEOUT))
        except ValueError:
            timeout = DEFAULT_TIMEOUT

    try:
        threshold = int(os.environ.get(
            "SONNET_RERANKER_HYBRID_PRIOR_THRESHOLD",
            DEFAULT_HYBRID_PRIOR_THRESHOLD,
        ))
    except ValueError:
        threshold = DEFAULT_HYBRID_PRIOR_THRESHOLD

    try:
        result = asyncio.run(
            _rerank_async(query, candidates, top_k, timeout, threshold,
                          return_metadata=return_metadata)
        )
        return result
    except RuntimeError as e:
        # asyncio.run fails if already in an event loop; in that case
        # caller is async — we don't have a sync fallback here, so just
        # return input order
        msg = str(e).lower()
        if "already running" in msg or "running event loop" in msg:
            LOG.warning("Sonnet reranker called from async context; not yet supported, "
                        "using hybrid order")
            return _emit(candidates[:top_k], False, REASON_ASYNC_CONTEXT)
        LOG.warning(f"Sonnet reranker runtime error: {e}; using hybrid order")
        return _emit(candidates[:top_k], False, REASON_UNEXPECTED_ERROR)
    except Exception as e:
        LOG.warning(f"Sonnet reranker unexpected error: {e}; using hybrid order")
        return _emit(candidates[:top_k], False, REASON_UNEXPECTED_ERROR)
