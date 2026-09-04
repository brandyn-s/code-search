"""OpenAI-compatible pointwise LLM reranker (``RERANKER=openai``).

Bring your own model: any server that implements ``POST /chat/completions``
(OpenAI, Azure OpenAI, OpenRouter, Ollama, vLLM, LM Studio, LiteLLM and other
gateways in front of Gemini or Bedrock). The prompt and score parsing are the
shared judge in ``search.llm_judge``, so scores are comparable with the
Anthropic engine.

Configuration (all read through ``search.env``):

- ``RERANKER_LLM_MODEL`` (required): model name the server exposes.
- ``RERANKER_LLM_BASE_URL``: endpoint root including the version path;
  defaults to ``OPENAI_BASE_URL``, then ``https://api.openai.com/v1``.
- ``RERANKER_LLM_API_KEY``: falls back to ``OPENAI_API_KEY``; required only
  for api.openai.com. ``OPENAI_AUTH_HEADER`` selects ``bearer`` or ``api-key``.
- ``RERANKER_LLM_TIMEOUT_S``: per-request timeout (default 12 s).
- ``SONNET_RERANKER_TIMEOUT``, ``SONNET_RERANKER_POOL_SIZE``,
  ``SONNET_RERANKER_HYBRID_PRIOR_THRESHOLD`` (and its path overrides) apply
  unchanged; they are engine-agnostic despite the historical prefix.

Failure contract: identical to the Anthropic engine. Any error preserves the
hybrid order and records ``_metadata.reranker.reason``. Never raises.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, wait

import httpx

from embeddings.openai_embedder import (
    build_auth_headers,
    requires_api_key,
    resolve_auth_header_style,
    resolve_openai_base_url,
)
from search.env import env_get
from search.llm_judge import build_judge_prompt, parse_score
from search.logging_privacy import format_query_exception_for_log
from search.sonnet_reranker import (
    DEFAULT_HYBRID_PRIOR_THRESHOLD,
    DEFAULT_TIMEOUT,
    FAILURE_TOLERANCE,
    REASON_API_KEY_MISSING,
    REASON_EMPTY_INPUT,
    REASON_HYBRID_PRIOR_FALLBACK,
    REASON_OK,
    REASON_RATE_LIMIT,
    REASON_TIMEOUT,
    REASON_TOO_MANY_FAILURES,
    REASON_UNEXPECTED_ERROR,
    _effective_threshold,
    _matching_clauses,
    _parse_clause_overrides,
    _parse_path_overrides,
    _resolve_pool_size,
)

LOG = logging.getLogger(__name__)

REASON_MODEL_NOT_CONFIGURED = "model_not_configured"
DEFAULT_PER_CALL_TIMEOUT_S = 12.0
MAX_TOKENS = 200

_ERR_TIMEOUT = "timeout"
_ERR_RATE_LIMIT = "rate_limit"
_ERR_HTTP = "http"
_ERR_UNPARSEABLE = "unparseable"


def resolve_model() -> str:
    return (env_get("RERANKER_LLM_MODEL", "") or "").strip()


def resolve_base_url() -> str:
    explicit = (env_get("RERANKER_LLM_BASE_URL", "") or "").strip()
    return resolve_openai_base_url(explicit)


def resolve_api_key() -> str:
    return (env_get("RERANKER_LLM_API_KEY", "") or env_get("OPENAI_API_KEY", "") or "").strip()


def resolve_per_call_timeout() -> float:
    raw = env_get("RERANKER_LLM_TIMEOUT_S")
    if raw is None or not raw.strip():
        return DEFAULT_PER_CALL_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_PER_CALL_TIMEOUT_S
    return value if value > 0 else DEFAULT_PER_CALL_TIMEOUT_S


def preflight_problems() -> list[str]:
    """Configuration problems that would make every rerank fall back.

    Used by the server's startup preflight so a misconfigured engine is
    announced once instead of failing silently per query.
    """
    problems: list[str] = []
    if not resolve_model():
        problems.append("RERANKER_LLM_MODEL is not set")
    base_url = resolve_base_url()
    if requires_api_key(base_url) and not resolve_api_key():
        problems.append(
            "RERANKER_LLM_API_KEY (or OPENAI_API_KEY) is not set for "
            f"{base_url}"
        )
    try:
        resolve_auth_header_style()
    except ValueError as exc:
        problems.append(str(exc))
    return problems


def _classify(exc: BaseException) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return _ERR_TIMEOUT
    if isinstance(exc, httpx.HTTPStatusError):
        return _ERR_RATE_LIMIT if exc.response.status_code == 429 else _ERR_HTTP
    return _ERR_HTTP


def _score_one(
    client: httpx.Client,
    *,
    base_url: str,
    model: str,
    headers: dict[str, str],
    timeout: float,
    prompt: str,
) -> int | str:
    """Return an int score or an error tag; never raises."""
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response = client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        text = body["choices"][0]["message"]["content"] or ""
    except Exception as exc:  # noqa: BLE001 - classified below, never raised
        LOG.debug("openai reranker call failed: %s", format_query_exception_for_log(exc))
        return _classify(exc)
    score = parse_score(text)
    return _ERR_UNPARSEABLE if score is None else score


def _aggregate_reason(failures: list[str]) -> str:
    if not failures:
        return REASON_TOO_MANY_FAILURES
    counts: dict[str, int] = {}
    for tag in failures:
        counts[tag] = counts.get(tag, 0) + 1
    top = max(counts.values())
    if counts.get(_ERR_RATE_LIMIT, 0) >= top - 1 and counts.get(_ERR_RATE_LIMIT, 0) > 0:
        return REASON_RATE_LIMIT
    if counts.get(_ERR_TIMEOUT, 0) >= top - 1 and counts.get(_ERR_TIMEOUT, 0) > 0:
        return REASON_TIMEOUT
    return REASON_TOO_MANY_FAILURES


def rerank_with_openai(
    query: str,
    candidates: list[dict],
    top_k: int = 10,
    timeout: float | None = None,
    return_metadata: bool = False,
):
    """Rerank ``candidates`` with an OpenAI-compatible chat model. Never raises."""
    t_start = time.monotonic()

    def _emit(out_list: list[dict], applied: bool, reason: str):
        latency_ms = int((time.monotonic() - t_start) * 1000)
        if return_metadata:
            return out_list, {"applied": applied, "reason": reason, "latency_ms": latency_ms}
        return out_list

    if not candidates:
        return _emit([], False, REASON_EMPTY_INPUT)

    model = resolve_model()
    if not model:
        LOG.warning("[RERANK_REASON] %s RERANKER_LLM_MODEL is not set; using hybrid order",
                    REASON_MODEL_NOT_CONFIGURED)
        return _emit(candidates[:top_k], False, REASON_MODEL_NOT_CONFIGURED)

    base_url = resolve_base_url()
    api_key = resolve_api_key()
    if requires_api_key(base_url) and not api_key:
        return _emit(candidates[:top_k], False, REASON_API_KEY_MISSING)

    try:
        headers = {"Content-Type": "application/json"}
        headers.update(build_auth_headers(api_key, resolve_auth_header_style()))
    except ValueError as exc:
        LOG.warning("openai reranker misconfigured: %s; using hybrid order", exc)
        return _emit(candidates[:top_k], False, REASON_UNEXPECTED_ERROR)

    if timeout is None:
        try:
            timeout = float(env_get("SONNET_RERANKER_TIMEOUT", DEFAULT_TIMEOUT))
        except ValueError:
            timeout = DEFAULT_TIMEOUT
    try:
        threshold = int(env_get("SONNET_RERANKER_HYBRID_PRIOR_THRESHOLD", DEFAULT_HYBRID_PRIOR_THRESHOLD))
    except ValueError:
        threshold = DEFAULT_HYBRID_PRIOR_THRESHOLD
    path_overrides = _parse_path_overrides(env_get("SONNET_RERANKER_HYBRID_PRIOR_THRESHOLD_PATH_OVERRIDES"))
    clause_overrides = _parse_clause_overrides(env_get("SONNET_RERANKER_PROMPT_CLAUSE_OVERRIDES"))
    pool_size = _resolve_pool_size()
    per_call_timeout = min(resolve_per_call_timeout(), timeout)

    if 0 < pool_size < len(candidates):
        pool, tail = candidates[:pool_size], candidates[pool_size:]
    else:
        pool, tail = candidates, []

    try:
        prompts: list[str] = []
        for item in pool:
            content = item.get("full_content") or item.get("content") or item.get("content_preview") or ""
            file_path = item.get("file_path") or item.get("file") or item.get("relative_path") or ""
            matched = _matching_clauses(file_path, clause_overrides or {})
            extra = ("\n" + "\n".join(matched)) if matched else ""
            prompts.append(build_judge_prompt(query, file_path, content, extra))

        scores: list[int | str] = [_ERR_TIMEOUT] * len(pool)
        with httpx.Client() as client, ThreadPoolExecutor(max_workers=max(1, len(pool))) as pool_exec:
            futures = {
                pool_exec.submit(
                    _score_one, client, base_url=base_url, model=model,
                    headers=headers, timeout=per_call_timeout, prompt=prompt,
                ): idx
                for idx, prompt in enumerate(prompts)
            }
            done, not_done = wait(futures, timeout=timeout)
            for future in done:
                scores[futures[future]] = future.result()
            for future in not_done:
                future.cancel()
            if not_done:
                LOG.warning(
                    "[RERANK_REASON] %s cohort_wall_ms>%d n_candidates=%d; using hybrid order",
                    REASON_TIMEOUT, int(timeout * 1000), len(candidates),
                )
                return _emit(candidates[:top_k], False, REASON_TIMEOUT)

        failures = [s for s in scores if isinstance(s, str)]
        if scores and len(failures) > len(scores) * FAILURE_TOLERANCE:
            reason = _aggregate_reason(failures)
            LOG.warning(
                "[RERANK_REASON] %s n_failed=%d n_total=%d model=%s; using hybrid order",
                reason, len(failures), len(scores), model,
            )
            return _emit(candidates[:top_k], False, reason)

        effective_threshold = _effective_threshold(candidates, threshold, path_overrides or {})
        valid = [s for s in scores if isinstance(s, int)]
        if valid and max(valid) < effective_threshold:
            LOG.info(
                "[RERANK_REASON] %s max_score=%d effective_threshold=%d model=%s",
                REASON_HYBRID_PRIOR_FALLBACK, max(valid), effective_threshold, model,
            )
            return _emit(candidates[:top_k], False, REASON_HYBRID_PRIOR_FALLBACK)

        def _sort_key(entry):
            idx, (score, _item) = entry
            is_failure = not isinstance(score, int)
            return (is_failure, -(score if isinstance(score, int) else -1), idx)

        indexed = sorted(enumerate(zip(scores, pool, strict=False)), key=_sort_key)
        reranked = [item for _, (_, item) in indexed]
        return _emit((reranked + tail)[:top_k], True, REASON_OK)
    except Exception as exc:  # noqa: BLE001 - always-on contract
        LOG.warning(
            "openai reranker unexpected error: %s; using hybrid order",
            format_query_exception_for_log(exc),
        )
        return _emit(candidates[:top_k], False, REASON_UNEXPECTED_ERROR)
