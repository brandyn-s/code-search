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

CS-3 (2026-05-06) — short-natural-query detection + multi-alternative
rewriting. The 2026-05-03 PSM eval showed real-session MRR 0.353 vs
golden 0.828 (1.4× gap dominated by query-shape mismatch). Real users
send short, natural-language queries (median 4 words) that BM25 can't
lexically match. The existing single-rewrite path helps but produces
ONE alternative; some queries need MULTIPLE phrasings to surface the
right files. CS-3 adds:
  - is_short_natural_query(query) detector
  - rewrite_short_natural_query(query) — returns up to 3 alternative
    phrasings using a tighter prompt focused on code naming conventions
  - SHORT_QUERY_REWRITE env var (default off) — opt-in
"""

import logging
import os
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# Cache rewrites to avoid redundant API calls
_rewrite_cache: dict = {}
_MAX_CACHE = 500

# CS-3 cache for multi-alternative short-query rewrites. Separate from
# _rewrite_cache because the value type differs (list vs str).
_short_query_cache: dict = {}
_MAX_SHORT_CACHE = 500


# CS-3: code-token signals that mark a query as ALREADY code-shaped.
# Queries containing CamelCase identifiers, snake_case, dotted paths,
# parens, or backticks aren't "short natural-language" — the user has
# already provided code-anchored search terms. Skip the short-query
# rewriter for these to avoid wasting API budget on queries the
# existing BM25 path already handles well.
_CODE_TOKEN_PATTERNS = [
    re.compile(r"[A-Z][a-z]+[A-Z]"),  # CamelCase like AlertToast
    re.compile(r"[a-z]+_[a-z]+"),     # snake_case like fetch_data
    re.compile(r"[a-zA-Z]\.[a-zA-Z]"),  # dotted like obj.method
    re.compile(r"[()`/]"),            # parens, backticks, paths
]
_SHORT_QUERY_MAX_TOKENS = 5


def is_short_natural_query(query: str) -> bool:
    """CS-3: detect short, natural-language queries that the multi-
    alternative rewriter targets.

    A query is "short natural" when:
      - it splits into FEWER than _SHORT_QUERY_MAX_TOKENS whitespace
        tokens (median real-session length is 4 words), AND
      - it contains NO code-token signals (CamelCase, snake_case,
        dotted identifiers, parens, etc.) — those queries are already
        code-shaped and don't need rewriting

    The empty/whitespace-only case returns False (no rewrite to do).
    """
    if not query or not query.strip():
        return False
    tokens = query.strip().split()
    if len(tokens) >= _SHORT_QUERY_MAX_TOKENS:
        return False
    for pat in _CODE_TOKEN_PATTERNS:
        if pat.search(query):
            return False
    return True


def rewrite_short_natural_query(query: str, n_alternatives: int = 3) -> List[str]:
    """CS-3: produce up to N alternative phrasings for a short natural-
    language query, mirroring code naming conventions.

    Returns a list of alternative queries (excluding the original) on
    success, or an empty list on any failure. Caller is responsible for
    running the alternatives through the search pipeline (typically by
    fanning out hybrid search and merging via existing RRF fusion).

    Controlled by `SHORT_QUERY_REWRITE` env var (default off) — opt-in
    because it adds 1-3 LLM calls on the search hot path and Anthropic
    has shown 28-30% retry-exhausted failure rates (see
    `bench/research/anthropic_latency_diagnosis.md`). Graceful fallback
    on any error returns [].
    """
    if os.environ.get("SHORT_QUERY_REWRITE", "off") != "on":
        return []
    if not is_short_natural_query(query):
        return []
    if n_alternatives <= 0:
        return []

    # Cache hit — return the cached alternatives without re-calling the API.
    cache_key = f"{query}\x00{n_alternatives}"
    if cache_key in _short_query_cache:
        return list(_short_query_cache[cache_key])

    try:
        alternatives = _call_haiku_short_query(query, n_alternatives)
        if alternatives:
            if len(_short_query_cache) >= _MAX_SHORT_CACHE:
                _short_query_cache.pop(next(iter(_short_query_cache)))
            _short_query_cache[cache_key] = list(alternatives)
            logger.debug(
                "short-query rewrite: '%s' -> %r", query, alternatives,
            )
            return alternatives
    except Exception as e:
        logger.debug("short-query rewrite failed, returning []: %s", e)

    return []


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


def _call_haiku_short_query(query: str, n_alternatives: int) -> List[str]:
    """CS-3: call Haiku to produce N alternative phrasings of a short
    natural-language query, mirroring code naming conventions.

    Returns a list of alternative queries on success, or [] on failure.
    Tighter prompt than the BM25 single-rewrite path: explicitly asks
    for code-naming-convention variants (function-name shape,
    snake_case, identifier tokens, file-path hint).
    """
    api_key = _get_api_key()
    if not api_key:
        return []

    import httpx

    # Reuse the IPv4 patch from _call_haiku above. Module-level state.
    import socket
    _orig = socket.getaddrinfo
    socket.getaddrinfo = lambda host, port, family=0, type=0, proto=0, flags=0: _orig(
        host, port, socket.AF_INET, type, proto, flags
    )
    try:
        import urllib3.util.connection
        urllib3.util.connection.HAS_IPV6 = False
    except ImportError:
        pass

    model = os.environ.get("BM25_REWRITE_MODEL", DEFAULT_HAIKU_MODEL)
    prompt = (
        f"Given a {len(query.split())}-word natural-language query "
        "targeting a code repository, produce exactly "
        f"{n_alternatives} alternative phrasings that mirror common "
        "code-naming conventions. Each alternative should use a "
        "different style: function-name shape (CamelCase identifier "
        "tokens), snake_case identifier tokens, or a file-path hint "
        "(directory tokens + extension). Return the alternatives as "
        "a numbered list, ONE alternative per line, no explanation.\n\n"
        f"Query: {query}"
    )

    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=5.0,
    )

    if resp.status_code != 200:
        return []

    data = resp.json()
    text = data.get("content", [{}])[0].get("text", "").strip()
    if not text:
        return []

    # Parse numbered list. Tolerant of `1.`, `1)`, `-`, `*` markers and
    # trailing whitespace. Accept lines that look like substantive
    # alternatives (≥2 chars, ≤200 chars).
    out: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Strip leading numbering / bullet markers
        line = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line)
        if 2 <= len(line) <= 200 and line.lower() != query.lower():
            out.append(line)
        if len(out) >= n_alternatives:
            break
    return out
