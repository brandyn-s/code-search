"""Typed configuration for search-time behavior (R11 phase 1).

Replaces scattered ``os.environ.get(...)`` calls across ``search/searcher.py``
with a single dataclass that parses, validates, and surfaces config at one
place. Built on top of the ``_parse_env_int`` / ``_parse_env_float``
helpers introduced in R3 (PR #192) so all parsing has consistent failure
behavior: bad values log a warning and fall back to defaults rather than
crashing the search call.

Scope (phase 1):
    - Search hot-path knobs read inside ``_hybrid_search``: FUSION_K,
      VECTOR_WEIGHT, BM25_WEIGHT, CONTENT_MODE, plus the booleans
      QUERY_EXPANSION and BM25_REWRITE.
    - Reranker knobs read inside ``_hybrid_search``: RERANKER (mode),
      SONNET_LISTWISE_TIMEOUT, SONNET_RERANKER_SKIP_THRESHOLD.

Out of scope (deliberate; future phases):
    - Secrets (ANTHROPIC_API_KEY, VOYAGE_API_KEY) — read at use site
      because they should never live in a long-lived config object.
    - Diagnostic-only flags (SONNET_RERANKER_LOG_PER_CANDIDATE_SCORE) —
      read at use site for parity with operator-driven debugging.
    - Provider knobs (EMBEDDING_PROVIDER, EMBEDDING_MODEL, JINA_TRUNCATE_DIM)
      — owned by the embedder module and will move in R12's registry work.
    - Per-call SDK knobs (ANTHROPIC_PER_CALL_TIMEOUT_S,
      ANTHROPIC_CONCURRENCY_LIMIT, ANTHROPIC_MAX_RETRIES) — owned by the
      reranker module's internal client; surface separately.
    - JSON-shaped knobs (CHUNK_TYPE_BOOST_OVERRIDE,
      SONNET_RERANKER_HYBRID_PRIOR_THRESHOLD_PATH_OVERRIDES) — need a
      richer parser; treat as a follow-up.

Usage:
    from search.config import get_search_config
    cfg = get_search_config()
    if cfg.query_expansion:
        ...

Test pattern:
    monkeypatch.setenv("FUSION_K", "30")
    from search.config import get_search_config
    get_search_config.cache_clear()  # invalidate memoized config
    cfg = get_search_config()
    assert cfg.fusion_k == 30
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Tuple

_module_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parsing primitives — extracted from searcher.py (R3, PR #192) so config
# loading reuses the same graceful-fallback contract.
# ---------------------------------------------------------------------------


def parse_env_int(
    name: str,
    default: int,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Parse an int from an env var with graceful fallback on bad input."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except (TypeError, ValueError):
        (logger or _module_logger).warning(
            "[CONFIG] env var %s=%r is not a valid int; using default %s",
            name, raw, default,
        )
        return default
    if min_value is not None and val < min_value:
        (logger or _module_logger).warning(
            "[CONFIG] env var %s=%s below min_value=%s; using default %s",
            name, val, min_value, default,
        )
        return default
    if max_value is not None and val > max_value:
        (logger or _module_logger).warning(
            "[CONFIG] env var %s=%s above max_value=%s; using default %s",
            name, val, max_value, default,
        )
        return default
    return val


def parse_env_float(
    name: str,
    default: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> float:
    """Parse a float from an env var with graceful fallback on bad input."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        (logger or _module_logger).warning(
            "[CONFIG] env var %s=%r is not a valid float; using default %s",
            name, raw, default,
        )
        return default
    if min_value is not None and val < min_value:
        (logger or _module_logger).warning(
            "[CONFIG] env var %s=%s below min_value=%s; using default %s",
            name, val, min_value, default,
        )
        return default
    if max_value is not None and val > max_value:
        (logger or _module_logger).warning(
            "[CONFIG] env var %s=%s above max_value=%s; using default %s",
            name, val, max_value, default,
        )
        return default
    return val


def parse_env_bool(name: str, default: bool = False) -> bool:
    """Parse a bool from an env var. Truthy: 1/true/yes/on (case-insensitive)."""
    raw = os.environ.get(name, "")
    return raw.strip().lower() in ("1", "true", "yes", "on") if raw else default


def parse_env_enum(
    name: str,
    default: str,
    allowed: Tuple[str, ...],
    logger: Optional[logging.Logger] = None,
) -> str:
    """Parse a string from an env var, restricted to an allowed set.

    Unknown values log a warning and fall back to the default — replaces
    the silent ``.get(mode, fallback)`` pattern that previously hid typos
    (e.g., CONTENT_MODE=cod silently became balanced weights).
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val not in allowed:
        (logger or _module_logger).warning(
            "[CONFIG] env var %s=%r not in allowed=%s; using default %r",
            name, raw, allowed, default,
        )
        return default
    return val


# ---------------------------------------------------------------------------
# SearchConfig — the typed dataclass for search-time behavior
# ---------------------------------------------------------------------------


# Content-mode allowlist. Pre-R11 the searcher accepted any string and
# silently mapped unknowns to the "code" defaults — typos like
# CONTENT_MODE=cod were undetectable.
CONTENT_MODES: Tuple[str, ...] = ("code", "docs", "all")

# Reranker mode allowlist. The dispatcher in searcher.py routes by this string.
RERANKER_MODES: Tuple[str, ...] = (
    "sonnet", "listwise", "voyage", "cross-encoder", "off",
)

# Search mode allowlist (search_code's search_mode arg).
SEARCH_MODES: Tuple[str, ...] = ("auto", "hybrid", "keyword", "semantic")


@dataclass(frozen=True)
class SearchConfig:
    """Validated search-time configuration.

    Frozen because consumers cache this and pass it across helpers; an
    accidental mutation in one caller would propagate. To override in
    tests, monkey-patch env vars and clear the ``get_search_config`` cache
    rather than mutating an instance.
    """

    # RRF + hybrid weights
    fusion_k: int
    """Smoothing parameter for RRF. k=20 wins over k=60 (sharper rank fusion).
    Lower bound 1 — k=0 would div-by-zero in RRF."""

    vector_weight: float
    """Override for vector arm RRF weight. 0.0 means 'use content-mode default'."""

    bm25_weight: float
    """Override for BM25 arm RRF weight. 0.0 means 'use content-mode default'."""

    # Content mode controls both weights and chunk-type boosts.
    content_mode: str

    # Behavioral toggles
    query_expansion: bool
    """Domain synonym expansion on BM25 query (default on)."""

    bm25_rewrite: bool
    """LLM-based BM25 query rewrite (default off, opt-in)."""

    short_query_rewrite: bool
    """Generate alt phrasings for short queries (default off, opt-in)."""

    agentic_search: bool
    """Blended agentic validation step after formatted_results (default off)."""

    # Reranker selection + tuning
    reranker_mode: str
    """One of RERANKER_MODES. Default 'sonnet' (pointwise rubric, with R9
    Nix-aware clause). The 2026-05-23 listwise default-flip (PR #199) was
    REVERTED 2026-05-23 after the rule-9 re-eval on current main: listwise
    harvested MRR delta −0.0456 CI [−0.0891, −0.0024], real_session_v1
    delta −0.0622 CI [−0.108, −0.017] — both CIs exclude zero unfavorable.
    See docs/findings/2026-05-23-listwise-default-eval-finding.md. Listwise
    stays selectable via RERANKER=listwise for callers who want the
    single-call latency profile and accept the harvested MRR cost."""

    listwise_timeout_s: float
    """Hard deadline for listwise reranker (RERANKER=listwise only).
    Default 12s per Phase C v2 simulated-deadline analysis. Below 8s
    degrades quality below hybrid baseline."""

    sonnet_skip_threshold: Optional[float]
    """When set, skip Sonnet rerank if top-1 hybrid score >= threshold.
    None disables the gate (Phase B'''(b) opt-in)."""

    # PPR (Personalized PageRank over code-graph) — consolidated R11 phase 2.
    ppr_enabled: bool
    """Whether the PPR re-ranking signal blends into post-boost scores.
    Default False (opt-in)."""

    ppr_alpha: float
    """Blend strength; ``score *= (1 + alpha * ppr_norm)``. alpha=0.0 is a
    bit-exact correctness gate documented in ``search.ppr_scorer``."""

    # Dispatch-level fallback (R11 phase 2)
    default_search_mode: str
    """Used by ``search()`` when the caller doesn't pass ``search_mode``.
    Pre-R11 this read ``SEARCH_MODE`` env directly with no allowlist."""
    sonnet_skip_threshold: Optional[float]
    """When set, skip Sonnet rerank if top-1 hybrid score >= threshold.
    None disables the gate (Phase B'''(b) opt-in).

    NOTE on PPR knobs: ``CODE_SEARCH_PPR_ENABLED`` and
    ``CODE_SEARCH_PPR_ALPHA`` are intentionally NOT carried by SearchConfig
    in phase 1 — ``search.ppr_scorer.get_env_config()`` is their existing
    operational source and is consumed directly by the PPR block in
    ``_hybrid_search``. A phase 2 consolidation can fold PPR into
    SearchConfig once we settle whether the PPR scorer should depend on
    the search-level config or stay self-contained."""


@lru_cache(maxsize=1)
def get_search_config() -> SearchConfig:
    """Return the validated SearchConfig, memoized across calls.

    Tests that override env vars must call ``get_search_config.cache_clear()``
    after the override to force a re-read. Production callers get a single
    parsed-and-validated instance per process.
    """
    return SearchConfig(
        # RRF / weights
        fusion_k=parse_env_int("FUSION_K", default=20, min_value=1),
        vector_weight=parse_env_float("VECTOR_WEIGHT", default=0.0, min_value=0.0),
        bm25_weight=parse_env_float("BM25_WEIGHT", default=0.0, min_value=0.0),

        # Modes
        content_mode=parse_env_enum("CONTENT_MODE", default="code", allowed=CONTENT_MODES),
        reranker_mode=parse_env_enum("RERANKER", default="sonnet", allowed=RERANKER_MODES),

        # Toggles (default-on for query_expansion is documented; rest default-off)
        query_expansion=parse_env_bool("QUERY_EXPANSION", default=True),
        bm25_rewrite=parse_env_bool("BM25_REWRITE", default=False),
        short_query_rewrite=parse_env_bool("SHORT_QUERY_REWRITE", default=False),
        agentic_search=parse_env_bool("AGENTIC_SEARCH", default=False),

        # Reranker tuning
        listwise_timeout_s=parse_env_float(
            "SONNET_LISTWISE_TIMEOUT", default=12.0, min_value=0.1,
        ),
        sonnet_skip_threshold=_parse_optional_float("SONNET_RERANKER_SKIP_THRESHOLD"),

        # R11 phase 2: PPR consolidation. Previously read by
        # ppr_scorer.get_env_config() directly; that helper now delegates
        # here so both modules share one source of truth.
        ppr_enabled=parse_env_bool("CODE_SEARCH_PPR_ENABLED", default=False),
        ppr_alpha=parse_env_float(
            "CODE_SEARCH_PPR_ALPHA", default=0.5, min_value=0.0,
        ),

        # R11 phase 2: SEARCH_MODE fallback. Pre-fix this was read inline
        # in search() with no allowlist — typos silently became "hybrid"-ish.
        default_search_mode=parse_env_enum(
            "SEARCH_MODE", default="hybrid", allowed=SEARCH_MODES,
        ),
    )


def _parse_optional_float(name: str) -> Optional[float]:
    """Optional float — None if unset OR if value is malformed/non-positive.

    Returns None instead of a default when the env var is unset, so the
    caller can distinguish 'gate not configured' from 'gate at threshold=X'.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        _module_logger.warning(
            "[CONFIG] env var %s=%r is not a valid float; treating as unset",
            name, raw,
        )
        return None
    if val <= 0:
        return None
    return val


# ---------------------------------------------------------------------------
# Resolved hybrid weights — derived helper, not a separate config knob
# ---------------------------------------------------------------------------


# Hybrid weights per content mode. Tuned per PR #90 A/B eval (n=99 queries):
# code mode = vector-heavy; docs mode = even more vector-heavy.
CONTENT_MODE_WEIGHTS: dict[str, Tuple[float, float]] = {
    "code": (0.65, 0.35),  # vector_weight, bm25_weight
    "docs": (0.7, 0.3),
    "all": (0.5, 0.5),
}


def resolve_hybrid_weights(cfg: SearchConfig) -> Tuple[float, float]:
    """Resolve (vector_weight, bm25_weight) honoring env overrides + mode default.

    If either VECTOR_WEIGHT or BM25_WEIGHT is positive, use them (filling
    the unset one with 0.5). Otherwise look up the content-mode default.
    Same semantics as the pre-R11 inline logic in searcher.py:410-417,
    just centralized.
    """
    if cfg.vector_weight > 0 or cfg.bm25_weight > 0:
        return (cfg.vector_weight or 0.5, cfg.bm25_weight or 0.5)
    return CONTENT_MODE_WEIGHTS.get(cfg.content_mode, (0.5, 0.5))
