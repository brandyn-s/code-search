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
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

LOG = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_CONTENT_CHARS = 4000
DEFAULT_TIMEOUT = 8.0
DEFAULT_RERANK_K = 15  # rerank top-15, return top-k of those (D4b validated)
FAILURE_TOLERANCE = 0.3  # if >30% of calls fail, abort and use input order

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


async def _score_one(client: Any, query: str, file_path: str, content: str) -> int | None:
    """Score one (query, chunk) pair. Returns int 0-10 on success, None on any failure."""
    truncated = content[:MAX_CONTENT_CHARS] if len(content) > MAX_CONTENT_CHARS else content
    prompt = JUDGE_PROMPT.format(
        query=query,
        file_path=file_path or "(unknown)",
        content=truncated or "(empty content)",
    )
    try:
        resp = await client.messages.create(
            model=MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        LOG.debug(f"Sonnet score call failed: {e}")
        return None
    try:
        text = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", "").strip()
                break
        if not text:
            return None
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(line for line in lines if not line.startswith("```"))
        obj = json.loads(text)
        score = int(obj.get("score", 0))
        return max(0, min(10, score))  # clamp to [0, 10]
    except Exception as e:
        LOG.debug(f"Sonnet score parse failed: {e}")
        return None


async def _rerank_async(
    query: str,
    candidates: list[dict],
    top_k: int,
    timeout: float,
) -> list[dict]:
    """Score candidates in parallel, sort by score, return top-k.
    Falls back to input[:top_k] on timeout, too-many-failures, or any exception.
    """
    try:
        import anthropic
    except ImportError:
        LOG.warning("anthropic package not installed; reranker disabled")
        return candidates[:top_k]

    client = anthropic.AsyncAnthropic()
    tasks = []
    for c in candidates:
        full = c.get("full_content") or c.get("content") or c.get("content_preview") or ""
        file_path = c.get("file_path") or c.get("file") or c.get("relative_path") or ""
        tasks.append(_score_one(client, query, file_path, full))

    try:
        scores = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=False),
                                          timeout=timeout)
    except asyncio.TimeoutError:
        LOG.warning(f"Sonnet reranker timeout >{timeout}s; using hybrid order")
        return candidates[:top_k]

    n_failed = sum(1 for s in scores if s is None)
    if len(scores) > 0 and n_failed > len(scores) * FAILURE_TOLERANCE:
        LOG.warning(f"Sonnet reranker {n_failed}/{len(scores)} failed; using hybrid order")
        return candidates[:top_k]

    # Sort: higher score wins; None scores sink to bottom; preserve original
    # order on ties (stable sort)
    indexed = list(enumerate(zip(scores, candidates)))
    indexed.sort(key=lambda x: (x[1][0] is None, -(x[1][0] or -1), x[0]))
    return [c for _, (_s, c) in indexed[:top_k]]


def rerank_with_sonnet(
    query: str,
    candidates: list[dict],
    top_k: int = 10,
    timeout: float | None = None,
) -> list[dict]:
    """Rerank candidates by Sonnet 4.6 relevance score. Returns top-k.

    Args:
        query: original search query string
        candidates: list of dicts, each with at least one of {full_content,
            content, content_preview} and one of {file_path, file, relative_path}.
            Extra keys preserved in output.
        top_k: number of results to return after reranking
        timeout: total budget in seconds (default DEFAULT_TIMEOUT=8.0,
            override via SONNET_RERANKER_TIMEOUT env)

    Returns:
        Reranked candidates[:top_k]. On any failure (no API key, timeout,
        HTTP error, parse failure, >30% call failures), returns
        candidates[:top_k] unchanged in input order.

    Never raises. Always-on contract.
    """
    if not candidates:
        return []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return candidates[:top_k]
    if timeout is None:
        try:
            timeout = float(os.environ.get("SONNET_RERANKER_TIMEOUT", DEFAULT_TIMEOUT))
        except ValueError:
            timeout = DEFAULT_TIMEOUT

    try:
        return asyncio.run(_rerank_async(query, candidates, top_k, timeout))
    except RuntimeError as e:
        # asyncio.run fails if already in an event loop; in that case
        # caller is async — we don't have a sync fallback here, so just
        # return input order
        if "already running" in str(e).lower() or "running event loop" in str(e).lower():
            LOG.warning("Sonnet reranker called from async context; not yet supported, "
                        "using hybrid order")
        else:
            LOG.warning(f"Sonnet reranker runtime error: {e}; using hybrid order")
        return candidates[:top_k]
    except Exception as e:
        LOG.warning(f"Sonnet reranker unexpected error: {e}; using hybrid order")
        return candidates[:top_k]
