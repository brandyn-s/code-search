"""Intelligent search functionality with query optimization."""

import json
import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from search.indexer import CodeIndexManager
from embeddings.embedder import CodeEmbedder


# R11 (PR forthcoming): the env-var parsing primitives moved to
# search.config so all callers go through the same validation contract.
# Re-exported here for backwards compatibility with tests/external code that
# imported the R3 helpers directly from this module.
from search.config import (
    parse_env_int as _parse_env_int,
    parse_env_float as _parse_env_float,
)


def reciprocal_rank_fusion(
    vector_results: List[Tuple[str, float]],
    bm25_results: List[Tuple[str, float]],
    k: int = 60,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
) -> List[Tuple[str, float]]:
    """Fuse two ranked lists using Weighted Reciprocal Rank Fusion.

    Args:
        vector_results: List of (chunk_id, score) from vector search, ordered by relevance.
        bm25_results: List of (chunk_id, score) from BM25 search, ordered by relevance.
        k: Smoothing parameter (default 60, industry standard).
        vector_weight: Weight for vector search contributions (default 0.5).
        bm25_weight: Weight for BM25 search contributions (default 0.5).

    Returns:
        List of (chunk_id, rrf_score) sorted by fused relevance.
    """
    scores: Dict[str, float] = {}
    for rank, (chunk_id, _score) in enumerate(vector_results):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + vector_weight * (
            1.0 / (k + rank + 1)
        )
    for rank, (chunk_id, _score) in enumerate(bm25_results):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + bm25_weight * (
            1.0 / (k + rank + 1)
        )
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# Content mode configurations re-exported from search.config for callers
# (CONTENT_MODE_WEIGHTS values: code=(0.65, 0.35) tuned 2026-05-03,
# vw=0.65 wins over vw=0.5 by MRR +0.016 on n=99 multi-target gold,
# B2 per-arm sweep PR #90). Source-of-truth lives in search.config now.
from search.config import CONTENT_MODE_WEIGHTS  # noqa: E402

# Chunk type boost multipliers per content mode
CHUNK_TYPE_BOOSTS = {
    "code": {
        "function": 1.3,
        "method": 1.3,
        "class": 1.3,
        "decorated_definition": 1.3,
        "let": 1.3,
        "binding": 1.3,
        "option": 1.3,          # NixOS mkOption declarations
        "service_config": 1.3,  # NixOS service configurations
        "imports": 1.1,         # NixOS imports lists
        "section": 0.7,
        "document": 0.7,
        "module": 0.9,
    },
    "docs": {
        "function": 0.8,
        "method": 0.8,
        "class": 0.8,
        "decorated_definition": 0.8,
        "section": 1.3,
        "document": 1.3,
        "module": 0.9,
    },
    "all": {},
}

# Code-domain synonym map for BM25 query expansion
CODE_SYNONYMS = {
    "auth": ["authentication", "oauth", "jwt", "token", "credential", "login", "entra"],
    "authentication": ["auth", "oauth", "jwt", "token", "credential", "login", "entra"],
    "error": ["exception", "raise", "ToolError", "HTTPException", "error_handling"],
    "retry": ["backoff", "retryable", "retry_delay", "429", "529"],
    "rate": ["rate_limit", "throttle", "RPM", "TPM"],
    "middleware": ["ASGI", "middleware", "intercept"],
    "route": ["Route", "endpoint", "path", "handler", "Starlette"],
    "network": ["networking", "internal-svc-19", "interface", "vlan", "firewall", "nftables", "allowedTCPPorts"],
    "service": ["systemd", "daemon", "enable", "wantedBy", "serviceConfig", "systemd.services", "mkEnableOption"],
    "package": ["pkgs", "nix", "derivation", "buildInputs", "nativeBuildInputs", "stdenv", "mkDerivation", "fetchurl"],
    "option": ["mkOption", "mkEnableOption", "types", "default", "description"],
    # NixOS-specific expansions
    "nix": ["nixos", "nixpkgs", "derivation", "flake", "overlay"],
    "module": ["nixos-module", "imports", "options", "mkIf"],
    "derivation": ["stdenv", "mkDerivation", "buildInputs", "nativeBuildInputs", "fetchurl"],
    "flake": ["flake.nix", "inputs", "outputs", "nixpkgs"],
    "enable": ["mkEnableOption", "mkIf", "cfg.enable"],
    "firewall": ["nftables", "allowedTCPPorts", "allowedUDPPorts", "networking.firewall"],
    "systemd": ["systemd.services", "serviceConfig", "wantedBy", "ExecStart"],
    "boot": ["bootloader", "grub", "systemd-boot", "initrd", "kernelModules"],
    "nixos": ["nix", "nixpkgs", "nix-module", "mkOption"],
    "environment": ["systemPackages", "environment.systemPackages"],
    # Corsair service domains
    "sensor": [
        "internal-svc-62",
        "internal-svc-28",
        "internal-svc-51",
        "internal-svc-25",
        "internal-svc-20",
        "internal-svc-23",
        "internal-svc-15",
        "internal-svc-40",
    ],
    "navigation": [
        "internal-svc-62",
        "internal-svc-51",
        "internal-svc-28",
        "internal-svc-20",
        "internal-svc-23",
        "internal-svc-60",
        "internal-svc-14",
        "internal-svc-57",
    ],
    "gps": ["internal-svc-62", "internal-svc-51", "internal-svc-25", "internal-svc-10", "internal-svc-36"],
    "imu": ["internal-svc-28", "internal-svc-20", "internal-svc-23"],
    "perception": ["internal-svc-26", "internal-svc-27", "internal-svc-56", "internal-svc-13", "internal-svc-6"],
    "camera": ["internal-svc-26", "internal-svc-56", "internal-svc-37", "internal-svc-50", "internal-svc-3"],
    "video": ["internal-svc-56", "internal-svc-8", "internal-svc-26", "internal-svc-3", "internal-svc-58"],
    "tracking": ["internal-svc-27", "internal-svc-26", "internal-svc-5", "internal-svc-63"],
    "motor": ["internal-svc-12", "internal-svc-34", "throttled", "internal-svc-31", "internal-svc-30"],
    "propulsion": ["internal-svc-12", "internal-svc-34", "throttled", "internal-svc-30", "internal-svc-43"],
    "engine": ["internal-svc-30", "internal-svc-12", "internal-svc-34"],
    "steering": ["internal-svc-12", "internal-svc-34", "rudder"],
    "communication": ["internal-svc-41", "internal-svc-44", "internal-svc-47", "internal-svc-64", "internal-svc-45", "internal-svc-49"],
    "radio": ["internal-svc-44", "internal-svc-41", "internal-svc-24"],
    "satellite": ["internal-svc-47", "internal-svc-32", "internal-svc-48"],
    "safety": ["internal-svc-55", "internal-svc-31", "internal-svc-39", "internal-svc-9", "internal-svc-16"],
    "emergency": ["internal-svc-39", "internal-svc-31", "internal-svc-55", "internal-svc-16"],
    "power": ["internal-svc-43", "internal-svc-18", "charged", "internal-svc-54", "internal-svc-53"],
    "battery": ["internal-svc-18", "charged", "internal-svc-43"],
    "radar": ["internal-svc-35", "internal-svc-61", "internal-svc-63"],
    "autonomy": ["internal-svc-52", "internal-svc-17", "internal-svc-57", "internal-svc-42", "internal-svc-46"],
    "planning": ["internal-svc-57", "internal-svc-52", "internal-svc-17", "internal-svc-42"],
    "fleet": ["internal-svc-46", "internal-svc-65", "internal-svc-24", "internal-svc-21"],
    "logging": ["internal-svc-45", "internal-svc-33", "internal-svc-49", "internal-svc-4"],
    "monitoring": ["internal-svc-7", "internal-svc-2", "internal-svc-59", "metrics"],
    "configuration": ["internal-svc-29", "internal-svc-11", "internal-svc-19", "internal-svc-38"],
    "calibration": ["internal-svc-38", "internal-svc-1", "internal-svc-22"],
}


# Order matters: longest suffix first so "navigations" -> strip "s" (not "tion").
# Each token has AT MOST one suffix stripped — we only need stem-equivalence to a
# CODE_SYNONYMS key, not full English morphology. `str.rstrip(<chars>)` is a
# character-class strip and was the prior implementation's bug
# (e.g. "navigation".rstrip("ing") removes trailing n, producing "navigatio";
# subsequent rstrip("tion") strips o/i/t producing "naviga" — never matches "navigation").
_QUERY_STEM_SUFFIXES = ("tion", "ing", "ed", "s")


def _query_stem(token: str) -> str:
    for suffix in _QUERY_STEM_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def expand_code_query(query: str) -> str:
    """Expand a query with code-domain synonyms for better BM25 recall."""
    tokens = query.lower().split()
    expanded_tokens = list(tokens)

    for token in tokens:
        stem = _query_stem(token)
        for key, synonyms in CODE_SYNONYMS.items():
            if token == key or stem == key or token in synonyms:
                for syn in synonyms:
                    if syn.lower() not in [t.lower() for t in expanded_tokens]:
                        expanded_tokens.append(syn)
                break

    if expanded_tokens == tokens:
        return query  # No expansion happened, return original

    return " ".join(expanded_tokens)


@dataclass
class SearchResult:
    """Enhanced search result with rich metadata."""

    chunk_id: str
    similarity_score: float
    content_preview: str
    file_path: str
    relative_path: str
    folder_structure: List[str]
    chunk_type: str
    name: Optional[str]
    parent_name: Optional[str]
    start_line: int
    end_line: int
    docstring: Optional[str]
    tags: List[str]
    context_info: Dict[str, Any]


class IntelligentSearcher:
    """Intelligent code search with query optimization and context awareness."""

    def __init__(self, index_manager: CodeIndexManager, embedder: CodeEmbedder):
        self.index_manager = index_manager
        self.embedder = embedder
        self._logger = logging.getLogger(__name__)
        self._query_embedding_cache: Dict[str, Any] = {}  # normalized_query -> embedding
        # PR Plan-2 A1 (2026-05-05): structured reranker metadata from the
        # most recent search() call. MCP layer reads this and emits
        # `_metadata.reranker = {applied, reason, latency_ms}` so LLM agents
        # can detect silent fallback (rotated API key, sustained rate-limit,
        # prolonged hybrid-prior fallback). Reset at the top of every
        # search() call. See docstring of rerank_with_sonnet for reason vocab.
        self.last_reranker_metadata: Dict[str, Any] = {
            "applied": False,
            "reason": "not_invoked",
            "latency_ms": 0,
        }
        # R8 (2026-05-23): structured PPR metadata, mirroring the
        # reranker envelope. PPR is an opt-in feature (CODE_SEARCH_PPR_ENABLED)
        # whose enable/disable, missing-graph-db, and empty-subgraph paths
        # were invisible to consumers before this — only sidecar [PPR_DIAG]
        # log lines signaled anything. The MCP layer emits this as
        # `_metadata.ppr = {applied, reason, latency_ms}`. Reason vocab:
        #   ok                  PPR applied; scores blended into candidates
        #   disabled_by_env     CODE_SEARCH_PPR_ENABLED is off (default)
        #   alpha_zero          CODE_SEARCH_PPR_ALPHA=0.0 (correctness gate)
        #   no_candidates       upstream produced empty candidate list
        #   no_graph_db         graph DB missing (PPRScorer returned {})
        #   error               exception caught; hybrid order preserved
        self.last_ppr_metadata: Dict[str, Any] = {
            "applied": False,
            "reason": "not_invoked",
            "latency_ms": 0,
        }

        # Query patterns for intent detection
        self.query_patterns = {
            "function_search": [
                r"\bfunction\b",
                r"\bdef\b",
                r"\bmethod\b",
                r"\bclass\b",
                r"how.*work",
                r"implement.*",
                r"algorithm.*",
            ],
            "error_handling": [
                r"\berror\b",
                r"\bexception\b",
                r"\btry\b",
                r"\bcatch\b",
                r"handle.*error",
                r"exception.*handling",
            ],
            "database": [
                r"\bdatabase\b",
                r"\bdb\b",
                r"\bquery\b",
                r"\bsql\b",
                r"\bmodel\b",
                r"\btable\b",
                r"connection",
            ],
            "api": [
                r"\bapi\b",
                r"\bendpoint\b",
                r"\broute\b",
                r"\brequest\b",
                r"\bresponse\b",
                r"\bhttp\b",
                r"rest.*api",
            ],
            "authentication": [
                r"\bauth\b",
                r"\blogin\b",
                r"\btoken\b",
                r"\bpassword\b",
                r"\bsession\b",
                r"authenticate",
                r"permission",
            ],
            "testing": [
                r"\btest\b",
                r"\bmock\b",
                r"\bassert\b",
                r"\bfixture\b",
                r"unit.*test",
                r"integration.*test",
            ],
        }

    def _get_query_embedding(self, query: str):
        """Get embedding for a query, using cache for repeated queries."""
        cache_key = query.strip().lower()
        if cache_key in self._query_embedding_cache:
            self._logger.debug(f"Query embedding cache hit for: '{cache_key}'")
            return self._query_embedding_cache[cache_key]
        embedding = self.embedder.embed_query(query)
        self._query_embedding_cache[cache_key] = embedding
        return embedding

    def clear_cache(self):
        """Clear query embedding cache. Call after reindex."""
        self._query_embedding_cache.clear()

    def search(
        self,
        query: str,
        k: int = 5,
        search_mode: str = "",
        context_depth: int = 1,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for code using semantic, keyword, or hybrid mode.

        Args:
            query: Natural language query
            k: Number of results
            search_mode: "hybrid", "semantic", or "keyword" (default from SEARCH_MODE env)
            context_depth: Include related chunks
            filters: Optional filters
        """

        # R11 phase 2: dispatch-mode fallback via validated SearchConfig.
        from search.config import get_search_config
        mode = search_mode or get_search_config().default_search_mode

        # Reset reranker metadata for this call. _hybrid_search overwrites with
        # the actual reason+latency from rerank_with_sonnet. Other modes leave
        # the "not_invoked_<mode>" sentinel so the MCP layer can distinguish
        # "Sonnet was skipped because mode=keyword" from "Sonnet failed for X".
        if mode == "keyword":
            self.last_reranker_metadata = {
                "applied": False, "reason": "not_invoked_keyword_mode", "latency_ms": 0,
            }
            self.last_ppr_metadata = {
                "applied": False, "reason": "not_invoked_keyword_mode", "latency_ms": 0,
            }
            return self._keyword_search(query, k, filters)
        elif mode == "semantic":
            self.last_reranker_metadata = {
                "applied": False, "reason": "not_invoked_semantic_mode", "latency_ms": 0,
            }
            self.last_ppr_metadata = {
                "applied": False, "reason": "not_invoked_semantic_mode", "latency_ms": 0,
            }
            return self._semantic_search(query, k, context_depth, filters)
        else:  # hybrid — _hybrid_search will populate both metadata fields
            self.last_reranker_metadata = {
                "applied": False, "reason": "not_invoked", "latency_ms": 0,
            }
            self.last_ppr_metadata = {
                "applied": False, "reason": "not_invoked", "latency_ms": 0,
            }
            return self._hybrid_search(query, k, context_depth, filters)

    def _semantic_search(
        self,
        query: str,
        k: int = 5,
        context_depth: int = 1,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Pure semantic search implementation."""

        # Detect query intent and optimize
        optimized_query = self._optimize_query(query)
        intent_tags = self._detect_query_intent(query)

        self._logger.info(
            f"Searching for: '{optimized_query}' with intent: {intent_tags}"
        )

        # Generate query embedding (cached)
        query_embedding = self._get_query_embedding(optimized_query)

        # Search with expanded result set for better filtering and recall
        search_k = min(k * 10, 200)  # Increased from k*3 to k*10 for better recall
        self._logger.info(
            f"Query embedding shape: {query_embedding.shape if hasattr(query_embedding, 'shape') else 'unknown'}"
        )
        self._logger.info(f"Using original filters: {filters}")
        self._logger.info(f"Calling index_manager.search with k={search_k}")

        raw_results = self.index_manager.search(query_embedding, search_k, filters)
        self._logger.info(f"Index manager returned {len(raw_results)} raw results")

        # Convert to rich search results
        search_results = []
        for chunk_id, similarity, metadata in raw_results:
            result = self._create_search_result(
                chunk_id, similarity, metadata, context_depth
            )
            search_results.append(result)

        # Post-process and rank results
        ranked_results = self._rank_results(search_results, query, intent_tags)

        return ranked_results[:k]

    def _keyword_search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Pure BM25 keyword search."""
        raw_results = self.index_manager.search_bm25(query, k=k, filters=filters)
        return [
            self._create_search_result(chunk_id, abs(rank), metadata, 0)
            for chunk_id, rank, metadata in raw_results
        ][:k]

    def _hybrid_search(
        self,
        query: str,
        k: int = 5,
        context_depth: int = 1,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Hybrid BM25 + vector search with weighted RRF fusion and content mode boosting."""

        # R11: read all search-time knobs through the validated SearchConfig.
        # Pre-R11 these were scattered `os.environ.get(...)` calls with
        # inconsistent failure modes (some crashed on malformed input,
        # some silently mapped to defaults). SearchConfig parses + validates
        # once, logs warnings for bad values, and provides typed access.
        from search.config import get_search_config, resolve_hybrid_weights
        cfg = get_search_config()

        fusion_k = cfg.fusion_k
        candidate_k = 50  # Retrieve 50 from each source

        # Determine content mode and weights
        content_mode = cfg.content_mode
        vector_weight, bm25_weight = resolve_hybrid_weights(cfg)

        # Vector search
        optimized_query = self._optimize_query(query)
        query_embedding = self._get_query_embedding(optimized_query)
        vector_raw = self.index_manager.search(query_embedding, candidate_k, filters)
        vector_pairs = [(chunk_id, sim) for chunk_id, sim, _meta in vector_raw]

        # BM25 search: LLM rewrite (if enabled) then static expansion
        bm25_query = query
        if cfg.bm25_rewrite:
            from search.query_rewriter import rewrite_query_for_bm25
            bm25_query = rewrite_query_for_bm25(query)
        if cfg.query_expansion:
            bm25_query = expand_code_query(bm25_query)
        bm25_raw = self.index_manager.search_bm25(bm25_query, k=candidate_k, filters=filters)
        bm25_pairs = [(chunk_id, rank) for chunk_id, rank, _meta in bm25_raw]

        # Weighted RRF fusion
        fused = reciprocal_rank_fusion(
            vector_pairs,
            bm25_pairs,
            k=fusion_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )

        # Build SearchResult objects for top-k fused results
        metadata_lookup = {}
        for chunk_id, _sim, metadata in vector_raw:
            metadata_lookup[chunk_id] = metadata
        for chunk_id, _rank, metadata in bm25_raw:
            if chunk_id not in metadata_lookup:
                metadata_lookup[chunk_id] = metadata

        # Collect more candidates than k so chunk type boosting can re-order
        over_fetch = min(k * 3, len(fused))
        candidates = []
        for chunk_id, rrf_score in fused[:over_fetch]:
            metadata = metadata_lookup.get(chunk_id)
            if metadata:
                result = self._create_search_result(
                    chunk_id, rrf_score, metadata, context_depth
                )
                candidates.append(result)

        # Apply chunk type boosts and name-match boost based on content mode.
        # `CHUNK_TYPE_BOOST_OVERRIDE` env var (Phase B3, 2026-05-08) layers a
        # JSON dict on top of the static defaults at search-time, enabling
        # the sweep harness to test alternative boost values without
        # restarting the server. Keys not in the override fall through to the
        # static dict; malformed JSON is silently ignored.
        boosts = dict(CHUNK_TYPE_BOOSTS.get(content_mode, {}))
        override_raw = os.environ.get("CHUNK_TYPE_BOOST_OVERRIDE")
        if override_raw:
            try:
                override = json.loads(override_raw)
                if isinstance(override, dict):
                    # NOTE: avoid using `k` as a loop variable here — `k` is the
                    # search top-k argument at function scope and shadowing it
                    # silently breaks `candidates[:k]` slicing later.
                    for chunk_type_key, boost_value in override.items():
                        boosts[chunk_type_key] = float(boost_value)
            except (ValueError, TypeError):
                pass
        query_tokens = self._normalize_to_tokens(query.lower())

        for result in candidates:
            # Chunk type boost
            if boosts:
                result.similarity_score *= boosts.get(result.chunk_type, 1.0)

            # Name-match boost (amplified 2x — A/B eval: +0.041 avg MRR)
            name_boost = self._calculate_name_boost(result.name, query, query_tokens)
            if name_boost > 1.0:
                name_boost = 1.0 + (name_boost - 1.0) * 2.0
            result.similarity_score *= name_boost

            # Path relevance boost (amplified 3x — fixes Nix +0.036 MRR
            # where correct files are retrieved but ranked too low)
            path_boost = self._calculate_path_boost(result.relative_path, query_tokens)
            if path_boost > 1.0:
                path_boost = 1.0 + (path_boost - 1.0) * 3.0
            result.similarity_score *= path_boost

        # Phase H fix (2026-05-10): sort candidates by post-boost
        # similarity_score BEFORE the rerank branch. This ensures all paths
        # (sonnet success, sonnet override-fallback, RERANKER=off) start
        # from the same boost-sorted order. Previously, only the
        # RERANKER=off path applied this sort (line ~600); sonnet's
        # hybrid_prior_fallback returned `candidates[:top_k]` in
        # RRF-fused-rank order, which differs from the boost-sorted order
        # on subprojects whose chunk-type / name / path boosts re-order
        # the top-15 (libnet: 13/18 queries had different top-1 between
        # the two paths; see bench/research/2026-05-10-assetman-override-refresh.md).
        candidates.sort(key=lambda r: r.similarity_score, reverse=True)

        # Arc A (2026-05-11): Personalized PageRank over code-graph as
        # post-boost-sort, pre-rerank re-ranking signal. Gated by env var
        # `CODE_SEARCH_PPR_ENABLED=1` (default off). Blends
        # `similarity_score *= (1 + alpha * ppr_score)` where alpha is read
        # from `CODE_SEARCH_PPR_ALPHA` (default 0.5). Mechanism-correctness
        # gate (Plan A2.4 falsifier): with PPR disabled OR alpha=0.0, this
        # block is a no-op and candidates pass through unchanged.
        #
        # R8 (2026-05-23): write `self.last_ppr_metadata` on every path so
        # the MCP `_metadata.ppr` envelope can surface enable/disable/
        # missing-DB to consumers. Previously this was visible only via
        # sidecar log lines.
        import time as _time
        from search.ppr_scorer import (
            PPRScorer, blend_ppr_into_candidates, get_env_config,
        )
        ppr_enabled, ppr_alpha = get_env_config()
        ppr_start = _time.monotonic()
        if not ppr_enabled:
            self.last_ppr_metadata = {
                "applied": False,
                "reason": "disabled_by_env",
                "latency_ms": 0,
            }
        elif ppr_alpha == 0.0:
            self.last_ppr_metadata = {
                "applied": False,
                "reason": "alpha_zero",
                "latency_ms": 0,
            }
        elif not candidates:
            self.last_ppr_metadata = {
                "applied": False,
                "reason": "no_candidates",
                "latency_ms": 0,
            }
        else:
            try:
                hint = None
                for c in candidates:
                    abs_path = getattr(c, "file_path", None) or getattr(c, "absolute_path", None)
                    if abs_path:
                        hint = str(abs_path)
                        break
                with PPRScorer() as ppr:
                    cps = [(c.relative_path, c.similarity_score) for c in candidates]
                    ppr_scores = ppr.score(cps, hint_abs_path=hint)
                latency_ms = int((_time.monotonic() - ppr_start) * 1000)
                if ppr_scores:
                    blend_ppr_into_candidates(candidates, ppr_alpha, ppr_scores)
                    candidates.sort(key=lambda r: r.similarity_score, reverse=True)
                    self.last_ppr_metadata = {
                        "applied": True,
                        "reason": "ok",
                        "latency_ms": latency_ms,
                        "scored_candidates": len(ppr_scores),
                        "alpha": ppr_alpha,
                    }
                else:
                    # Empty dict from PPRScorer.score() means either the
                    # graph DB is missing or the subgraph was too small.
                    # The scorer logs which via [PPR_DIAG]; both surface
                    # here as no_graph_db (the dominant cause in practice).
                    self.last_ppr_metadata = {
                        "applied": False,
                        "reason": "no_graph_db",
                        "latency_ms": latency_ms,
                    }
            except Exception as ppr_err:
                self._logger.warning("[PPR_DIAG] ppr_blend_failed err=%s", ppr_err)
                self.last_ppr_metadata = {
                    "applied": False,
                    "reason": "error",
                    "latency_ms": int((_time.monotonic() - ppr_start) * 1000),
                    "error_class": type(ppr_err).__name__,
                }

        # Reranking. Default mode is "sonnet" (validated 2026-05-03 PR #93+:
        # +0.087 MRR, +0.137 HR@1 on n=183 multi-target real_session). The
        # Sonnet reranker is graceful: on missing ANTHROPIC_API_KEY, timeout,
        # or any error, it silently returns input candidates unchanged.
        # Disable explicitly with RERANKER=off. Legacy cross-encoder via
        # RERANKER=cross-encoder (off-by-default since A/B showed quality
        # regression).
        rerank_mode = cfg.reranker_mode
        # Surface "no candidates" as the most specific signal, regardless of
        # mode. This catches empty-index searches; downstream consumers don't
        # need to disambiguate "no candidates because mode=off" vs "no
        # candidates because index empty" — the latter is the real signal.
        if not candidates:
            self.last_reranker_metadata = {
                "applied": False,
                "reason": "not_invoked_no_candidates",
                "latency_ms": 0,
            }
            return []
        if rerank_mode == "sonnet" and len(candidates) > k:
            # Phase B'''(b) skip-threshold gate (opt-in, default off):
            # SONNET_RERANKER_SKIP_THRESHOLD allows operators to skip Sonnet
            # entirely when the hybrid top-1 score already exceeds a
            # confidence floor. Motivation: the 2026-05-14 Phase B''
            # labeling analysis identified ~7% of harvested queries where
            # Sonnet at pool=5 CORRUPTS already-perfect hybrid rank-1
            # results. Skipping Sonnet on high-confidence queries preserves
            # the rank-1 + saves ~4-5s latency. Threshold is corpus-
            # specific — set per-deployment based on local similarity_score
            # distribution. See CLAUDE.md SONNET_RERANKER_SKIP_THRESHOLD
            # for tuning guidance.
            # R11: skip threshold is None when unset / non-positive (handled
            # by SearchConfig._parse_optional_float).
            skip_threshold = cfg.sonnet_skip_threshold
            if skip_threshold is not None:
                top_1_score = candidates[0].similarity_score
                if top_1_score >= skip_threshold:
                    self._logger.info(
                        "[RERANK_REASON] skipped_high_confidence "
                        "top_1_score=%.4f threshold=%.4f "
                        "n_candidates=%d; preserved hybrid order",
                        top_1_score, skip_threshold, len(candidates),
                    )
                    self.last_reranker_metadata = {
                        "applied": False,
                        "reason": "skipped_high_confidence",
                        "latency_ms": 0,
                        "top_1_score": top_1_score,
                        "skip_threshold": skip_threshold,
                    }
                    return candidates[:k]

            from search.sonnet_reranker import rerank_with_sonnet

            # Rerank only the top-15 candidates (D4b validated: top-30 is
            # equivalent to top-15 with 2x cost). Build dicts with
            # full_content so the LLM scores against actual code, not
            # 200-char snippets.
            n_to_rerank = min(15, len(candidates))
            top_candidates = candidates[:n_to_rerank]
            rerank_input = []
            for r in top_candidates:
                meta = metadata_lookup.get(r.chunk_id, {}) or {}
                full = (meta.get("full_content")
                        or meta.get("content")
                        or r.content_preview
                        or "")
                rerank_input.append({
                    "chunk_id": r.chunk_id,
                    "file_path": r.relative_path,
                    "full_content": full,
                    "_orig": r,
                })
            # PR Plan-2 A1: opt into structured metadata so the MCP layer
            # can surface reranker outcome to LLM agents.
            reranked, rerank_meta = rerank_with_sonnet(
                query, rerank_input, top_k=k, return_metadata=True,
            )
            self.last_reranker_metadata = rerank_meta
            # Extract original SearchResult objects in new order; tail any
            # candidates beyond top-15 in their existing order.
            new_top = [d["_orig"] for d in reranked]
            tail = candidates[n_to_rerank:]
            candidates = new_top + tail
        elif rerank_mode == "sonnet" and len(candidates) <= k:
            # Sonnet path entered but no reranking needed (candidate pool
            # is already <= k). Surface as a non-error reason.
            self.last_reranker_metadata = {
                "applied": False,
                "reason": "not_invoked_insufficient_candidates",
                "latency_ms": 0,
            }
        elif rerank_mode == "listwise" and len(candidates) > k:
            # Listwise reranker (opt-in canary, validated 2026-05-16 Phase C
            # v2 with bootstrap CI gates ALL PASS):
            #   - PSM golden: +0.047 nDCG@10 CI [+0.004, +0.095] favorable
            #   - PSM harvested: +0.044 MRR CI [+0.003, +0.084] favorable
            #   - PSM nix subset: +0.008 MRR (parity confirmed, regression
            #     from v1 reversed by nix-aware rubric clause)
            #   - flask/requests adversarial: +0.13 to +0.22 MRR favorable
            # Replaces pointwise (15 isolated Sonnet calls) with ONE
            # comparative call. Architecturally cleaner: removes the
            # slowest-of-15 latency pattern + arbitrary-tie behavior +
            # per-domain pointwise inconsistency. Retires the
            # SONNET_RERANKER_HYBRID_PRIOR_THRESHOLD_PATH_OVERRIDES hack.
            #
            # Hard deadline default 12s per Phase C v2 simulated-deadline
            # analysis. 10s is the smallest deadline where all 4 fixtures
            # stay favorable; 12s captures more of the listwise lift
            # (harvested applied 93.4% vs 76.5% at 10s; worst Δ nDCG@10
            # +0.010 vs +0.004) at the cost of 2s more p99. User picked
            # 12s 2026-05-16 for the higher applied rate.
            # On deadline/error/parse-failure, listwise returns baseline
            # order — graceful fallback per the always-on contract.
            #
            # Override deadline via SONNET_LISTWISE_TIMEOUT env var.
            from search.listwise_sonnet_reranker import (
                listwise_rerank_with_sonnet,
            )

            n_to_rerank = min(15, len(candidates))
            top_candidates = candidates[:n_to_rerank]
            rerank_input = []
            for r in top_candidates:
                meta = metadata_lookup.get(r.chunk_id, {}) or {}
                full = (meta.get("full_content")
                        or meta.get("content")
                        or r.content_preview
                        or "")
                rerank_input.append({
                    "chunk_id": r.chunk_id,
                    "file_path": r.relative_path,
                    "name": r.name,
                    "parent_name": r.parent_name,
                    "chunk_type": r.chunk_type,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                    "content_preview": full,
                    "similarity_score": r.similarity_score,
                    "_orig": r,
                })
            # R11: timeout validated + parsed by SearchConfig.
            reranked, rerank_meta = listwise_rerank_with_sonnet(
                query, rerank_input, top_k=k,
                timeout=cfg.listwise_timeout_s,
                return_metadata=True,
            )
            self.last_reranker_metadata = rerank_meta
            new_top = [d["_orig"] for d in reranked]
            tail = candidates[n_to_rerank:]
            candidates = new_top + tail
        elif rerank_mode == "listwise" and len(candidates) <= k:
            self.last_reranker_metadata = {
                "applied": False,
                "reason": "not_invoked_insufficient_candidates",
                "latency_ms": 0,
            }
        elif rerank_mode == "cross-encoder" and candidates:
            # Legacy cross-encoder path (off by default; degrades quality
            # per 2026-03-22 A/B eval but kept for fallback/comparison).
            from search.reranker import rerank_results

            rerank_input = [
                {
                    "chunk_id": r.chunk_id,
                    "content": r.content_preview,
                    "score": r.similarity_score,
                    "result": r,
                }
                for r in candidates
            ]
            reranked = rerank_results(query, rerank_input, top_k=k)
            candidates = [item["result"] for item in reranked]
            for item, candidate in zip(reranked, candidates):
                candidate.similarity_score = item.get(
                    "rerank_score", candidate.similarity_score
                )
            # PR Plan-2 A1: cross-encoder path doesn't invoke Sonnet — surface
            # explicit reason so MCP consumers don't misinterpret a default
            # "not_invoked".
            self.last_reranker_metadata = {
                "applied": False,
                "reason": "not_invoked_cross_encoder_mode",
                "latency_ms": 0,
            }
        else:
            # rerank_mode == "off" — explicit disable. The empty-candidates
            # path is handled earlier and returns immediately, so candidates
            # is non-empty here.
            self.last_reranker_metadata = {
                "applied": False,
                "reason": "disabled_by_env",
                "latency_ms": 0,
            }
            candidates.sort(key=lambda r: r.similarity_score, reverse=True)
        return candidates[:k]

    def _optimize_query(self, query: str) -> str:
        """Optimize query for better embedding generation."""
        # Basic query cleaning only - avoid expanding technical terms
        # that might distort code-specific queries
        return query.strip()

    def _detect_query_intent(self, query: str) -> List[str]:
        """Detect the intent/domain of the search query."""
        query_lower = query.lower()
        detected_intents = []

        for intent, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    detected_intents.append(intent)
                    break

        return detected_intents

    def _create_search_result(
        self,
        chunk_id: str,
        similarity: float,
        metadata: Dict[str, Any],
        context_depth: int,
    ) -> SearchResult:
        """Create a rich search result with context information."""

        # Basic metadata extraction
        content_preview = metadata.get("content_preview", "")
        file_path = metadata.get("file_path", "")
        relative_path = metadata.get("relative_path", "")
        folder_structure = metadata.get("folder_structure", [])

        # Context information
        context_info = {}

        if context_depth > 0:
            # Add related chunks context
            similar_chunks = self.index_manager.get_similar_chunks(chunk_id, k=3)
            context_info["similar_chunks"] = [
                {
                    "chunk_id": cid,
                    "similarity": sim,
                    "name": meta.get("name"),
                    "chunk_type": meta.get("chunk_type"),
                }
                for cid, sim, meta in similar_chunks[:2]  # Top 2 similar
            ]

            # Add file context
            context_info["file_context"] = {
                "total_chunks_in_file": self._count_chunks_in_file(relative_path),
                "folder_path": "/".join(folder_structure) if folder_structure else None,
            }

        return SearchResult(
            chunk_id=chunk_id,
            similarity_score=similarity,
            content_preview=content_preview,
            file_path=file_path,
            relative_path=relative_path,
            folder_structure=folder_structure,
            chunk_type=metadata.get("chunk_type", "unknown"),
            name=metadata.get("name"),
            parent_name=metadata.get("parent_name"),
            start_line=metadata.get("start_line", 0),
            end_line=metadata.get("end_line", 0),
            docstring=metadata.get("docstring"),
            tags=metadata.get("tags", []),
            context_info=context_info,
        )

    def _count_chunks_in_file(self, relative_path: str) -> int:
        """Count total chunks in a specific file."""
        count = 0
        stats = self.index_manager.get_stats()

        # This is a simplified implementation
        # In a real scenario, you might want to maintain this as a separate index
        return stats.get("files_indexed", 0)

    def _rank_results(
        self, results: List[SearchResult], original_query: str, intent_tags: List[str]
    ) -> List[SearchResult]:
        """Advanced ranking based on multiple factors."""

        def calculate_rank_score(result: SearchResult) -> float:
            score = result.similarity_score

            # Detect if query looks like an entity/class name
            query_tokens = self._normalize_to_tokens(original_query.lower())
            is_entity_query = self._is_entity_like_query(original_query, query_tokens)
            has_class_keyword = "class" in original_query.lower()

            # Dynamic chunk type boosts based on query type
            if has_class_keyword:
                # Strong preference for classes when "class" is mentioned
                type_boosts = {
                    "class": 1.3,
                    "function": 1.05,
                    "method": 1.05,
                    "module": 0.9,
                }
            elif is_entity_query:
                # Moderate preference for classes on entity-like queries
                type_boosts = {
                    "class": 1.15,
                    "function": 1.1,
                    "method": 1.1,
                    "module": 0.92,
                }
            else:
                # Default boosts for general queries
                type_boosts = {
                    "function": 1.1,
                    "method": 1.1,
                    "class": 1.05,
                    "module": 0.95,
                }

            score *= type_boosts.get(result.chunk_type, 1.0)

            # Enhanced name matching with token-based comparison
            name_boost = self._calculate_name_boost(
                result.name, original_query, query_tokens
            )
            score *= name_boost

            # Path/filename relevance boost
            path_boost = self._calculate_path_boost(result.relative_path, query_tokens)
            score *= path_boost

            # Boost based on tag matches
            if intent_tags and result.tags:
                tag_overlap = len(set(intent_tags) & set(result.tags))
                score *= 1.0 + tag_overlap * 0.1

            # Boost based on docstring presence (but less for module chunks on entity queries)
            if result.docstring:
                if is_entity_query and result.chunk_type == "module":
                    score *= (
                        1.02  # Smaller boost for module docstrings on entity queries
                    )
                else:
                    score *= 1.05

            # Slight penalty for very complex chunks (might be too specific)
            if len(result.content_preview) > 1000:
                score *= 0.98

            return score

        # Sort by calculated rank score
        ranked_results = sorted(results, key=calculate_rank_score, reverse=True)
        return ranked_results

    def _normalize_to_tokens(self, text: str) -> List[str]:
        """Convert text to normalized tokens, handling CamelCase."""
        import re

        # Split CamelCase and snake_case
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = text.replace("_", " ").replace("-", " ")

        # Extract alphanumeric tokens
        tokens = re.findall(r"\w+", text.lower())
        return tokens

    def _is_entity_like_query(self, query: str, query_tokens: List[str]) -> bool:
        """Detect if query looks like an entity/type name."""
        # Short queries with 1-3 tokens that don't contain action words
        if len(query_tokens) > 3:
            return False

        action_words = {
            "find",
            "search",
            "get",
            "show",
            "list",
            "how",
            "what",
            "where",
            "when",
            "create",
            "build",
            "make",
            "handle",
            "process",
            "manage",
            "implement",
        }

        # If any token is an action word, it's not an entity query
        if any(token in action_words for token in query_tokens):
            return False

        # If original query has CamelCase or looks like a class name, it's entity-like
        import re

        if re.search(r"[A-Z][a-z]+[A-Z]", query):  # CamelCase pattern
            return True

        return len(query_tokens) <= 2  # Short noun phrases

    def _calculate_name_boost(
        self, name: Optional[str], original_query: str, query_tokens: List[str]
    ) -> float:
        """Calculate boost based on name matching with robust token comparison."""
        if not name:
            return 1.0

        name_tokens = self._normalize_to_tokens(name)

        # Exact match (case insensitive)
        if original_query.lower() == name.lower():
            return 1.4

        # Token overlap calculation
        query_set = set(query_tokens)
        name_set = set(name_tokens)

        if not query_set or not name_set:
            return 1.0

        overlap = len(query_set & name_set)
        total_query_tokens = len(query_set)

        if overlap == 0:
            return 1.0

        # Strong boost for high overlap
        overlap_ratio = overlap / total_query_tokens
        if overlap_ratio >= 0.8:  # 80%+ of query tokens match
            return 1.3
        elif overlap_ratio >= 0.5:  # 50%+ match
            return 1.2
        elif overlap_ratio >= 0.3:  # 30%+ match
            return 1.1
        else:
            return 1.05

    def _calculate_path_boost(
        self, relative_path: str, query_tokens: List[str]
    ) -> float:
        """Calculate boost based on path/filename relevance."""
        if not relative_path or not query_tokens:
            return 1.0

        # Extract path components and filename
        path_parts = relative_path.lower().replace("/", " ").replace("\\", " ")
        path_tokens = self._normalize_to_tokens(path_parts)

        # Check for token overlap with path
        query_set = set(query_tokens)
        path_set = set(path_tokens)

        overlap = len(query_set & path_set)
        if overlap > 0:
            # Modest boost for path relevance
            return 1.0 + (overlap * 0.05)  # 5% boost per matching token

        return 1.0

    def search_by_file_pattern(
        self, query: str, file_patterns: List[str], k: int = 5
    ) -> List[SearchResult]:
        """Search within specific file patterns."""
        filters = {"file_pattern": file_patterns}
        return self.search(query, k=k, filters=filters)

    def search_by_chunk_type(
        self, query: str, chunk_type: str, k: int = 5
    ) -> List[SearchResult]:
        """Search for specific types of code chunks."""
        filters = {"chunk_type": chunk_type}
        return self.search(query, k=k, filters=filters)

    def find_similar_to_chunk(self, chunk_id: str, k: int = 5) -> List[SearchResult]:
        """Find chunks similar to a given chunk."""
        similar_chunks = self.index_manager.get_similar_chunks(chunk_id, k)

        results = []
        for chunk_id, similarity, metadata in similar_chunks:
            result = self._create_search_result(
                chunk_id, similarity, metadata, context_depth=1
            )
            results.append(result)

        return results

    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """Generate search suggestions based on indexed content."""
        # This is a simplified implementation
        # In a full system, you might maintain a separate suggestions index

        suggestions = []
        stats = self.index_manager.get_stats()

        # Suggest based on top tags
        top_tags = stats.get("top_tags", {})
        for tag in top_tags:
            if partial_query.lower() in tag.lower():
                suggestions.append(f"Find {tag} related code")

        # Suggest based on chunk types
        chunk_types = stats.get("chunk_types", {})
        for chunk_type in chunk_types:
            if partial_query.lower() in chunk_type.lower():
                suggestions.append(f"Show all {chunk_type}s")

        return suggestions[:5]
