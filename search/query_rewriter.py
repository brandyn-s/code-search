"""LLM-based query rewriting for BM25 search.

Rewrites natural language queries into code-specific keywords to improve
BM25 lexical matching. Uses Haiku for fast, cheap rewrites (~500ms, $0.001/query).

Only applied to the BM25 side of hybrid search — vector search keeps the
original NL query (voyage-context-3 handles NL-to-code matching natively).

Evidence: +0.090 MRR on TypeScript, +0.019 avg across 4 languages (A/B eval
2026-04-07, 102 queries). The NL-to-code semantic gap is largest for UI
component queries where BM25 can't match "dropdown menu" to "DropdownMenu".

Design decisions:
- Haiku not local rules: static synonym expansion can't generate identifiers
  it hasn't seen. Haiku generates codebase-appropriate terms (CamelCase,
  snake_case) that BM25 needs for lexical matching.
- BM25 only, not vector: voyage-context-3 already bridges NL-to-code.
  Rewriting the vector query HURTS (-0.067 MRR avg) because it loses the
  semantic intent that the embedding model needs.
- LRU cache: identical queries return cached rewrites (no API call).
- Graceful fallback: any error returns the original query unchanged.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Cache rewrites to avoid redundant API calls
_rewrite_cache: dict = {}
_MAX_CACHE = 500


def rewrite_query_for_bm25(query: str) -> str:
    """Rewrite an NL query into code-specific BM25 search terms.

    Returns the rewritten query, or the original query on any error.
    Controlled by BM25_REWRITE env var (default: off).

    Args:
        query: Natural language search query

    Returns:
        Code-oriented query for BM25 matching
    """
    if os.environ.get("BM25_REWRITE", "off") != "on":
        return query

    # Check cache
    if query in _rewrite_cache:
        return _rewrite_cache[query]

    try:
        rewritten = _call_haiku(query)
        if rewritten and rewritten != query:
            # Cache the result
            if len(_rewrite_cache) >= _MAX_CACHE:
                # Evict oldest entry
                _rewrite_cache.pop(next(iter(_rewrite_cache)))
            _rewrite_cache[query] = rewritten
            logger.debug(f"BM25 rewrite: '{query}' -> '{rewritten}'")
            return rewritten
    except Exception as e:
        logger.debug(f"BM25 rewrite failed, using original: {e}")

    return query


def _get_api_key() -> Optional[str]:
    """Get Anthropic API key from environment or config.

    Checks multiple env vars because ANTHROPIC_API_KEY is scrubbed
    from Claude Code subprocesses (CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1).
    """
    # Direct key (if set explicitly for the MCP server)
    for env_var in ["ANTHROPIC_API_KEY", "AUTO_LEARN_API_KEY"]:
        key = os.environ.get(env_var, "")
        if key and key.startswith("sk-ant-"):
            return key

    # Fallback: read from a config file
    key_file = os.path.expanduser("~/.anthropic/api_key")
    if os.path.exists(key_file):
        try:
            with open(key_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass

    return None


# Default Haiku alias used when BM25_REWRITE_MODEL env var is unset.
# Bumped 2026-05-06 from `claude-3-haiku-20240307` (deprecated, returns 404)
# after PR #124 found the silent-fallback failure mode: operators set
# BM25_REWRITE=on without overriding the model env, the API returned 404,
# the graceful-fallback path swallowed the error, and the rewriter
# silently returned the original query. PR #124's empirical test
# confirmed BM25_REWRITE=off-vs-on produced Δ MRR=0.0000 (no-op) until
# the model was overridden via BM25_REWRITE_MODEL.
DEFAULT_HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Sentinel — log a warning the FIRST time the rewriter falls back to
# original query in a session, so operators see deprecation-shaped
# silent-no-ops instead of having them swallowed entirely. Re-set on
# module reload (test isolation).
_warned_fallback = False


def _call_haiku(query: str) -> Optional[str]:
    """Call Haiku to rewrite query. Returns None on failure."""
    api_key = _get_api_key()
    if not api_key:
        return None

    import httpx

    # Force IPv4 (Tailscale split DNS causes IPv6 hangs)
    import socket
    _orig = socket.getaddrinfo
    socket.getaddrinfo = lambda host, port, family=0, type=0, proto=0, flags=0: _orig(
        host, port, socket.AF_INET, type, proto, flags
    )
    try:
        import urllib3.util.connection
        urllib3.util.connection.HAS_IPV6 = False
    except ImportError:
        pass  # urllib3 not installed, socket patch is sufficient

    model = os.environ.get("BM25_REWRITE_MODEL", DEFAULT_HAIKU_MODEL)
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": (
                    "Rewrite this search query into code-specific keywords for "
                    "searching a codebase. Include function names, module names, "
                    "variable names, and technical terms that would appear in "
                    "source code. Return ONLY the rewritten query on one line, "
                    "no explanation.\n\n"
                    f"Query: {query}"
                ),
            }],
        },
        timeout=5.0,
    )

    global _warned_fallback
    if resp.status_code == 200:
        data = resp.json()
        text = data.get("content", [{}])[0].get("text", "").strip()
        # Sanity check: rewrite shouldn't be empty or excessively long
        if text and len(text) < 500:
            return text
        # 200 but unparseable. Log the first occurrence so it isn't silent.
        if not _warned_fallback:
            _warned_fallback = True
            logger.warning(
                "BM25 rewriter received 200 but empty/unparseable response; "
                "falling back to original query. model=%s", model,
            )
        return None

    # Non-200: surface the deprecation-shaped failure mode that caused
    # PR #124's silent no-op. Log the FIRST occurrence per session so
    # operators see "model deprecated" rather than debug a silent gap.
    if not _warned_fallback:
        _warned_fallback = True
        logger.warning(
            "BM25 rewriter API call failed (status=%d); falling back to "
            "original query. model=%s. Override via BM25_REWRITE_MODEL env. "
            "Body: %s",
            resp.status_code, model, resp.text[:200] if resp.text else "(empty)",
        )
    return None
