"""Intelligent search functionality with query optimization."""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from search.indexer import CodeIndexManager
from embeddings.embedder import CodeEmbedder


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


# Content mode configurations: (vector_weight, bm25_weight)
CONTENT_MODE_WEIGHTS = {
    "code": (0.65, 0.35),  # Tuned 2026-05-03: vw=0.65 wins over vw=0.5 by MRR +0.016 on n=99 multi-target gold (B2 per-arm sweep, PR #90)
    "docs": (0.7, 0.3),
    "all": (0.5, 0.5),
}

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


def expand_code_query(query: str) -> str:
    """Expand a query with code-domain synonyms for better BM25 recall."""
    tokens = query.lower().split()
    expanded_tokens = list(tokens)

    for token in tokens:
        stem = token.rstrip("s").rstrip("ing").rstrip("tion").rstrip("ed")
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

        mode = search_mode or os.environ.get("SEARCH_MODE", "hybrid")

        if mode == "keyword":
            return self._keyword_search(query, k)
        elif mode == "semantic":
            return self._semantic_search(query, k, context_depth, filters)
        else:  # hybrid
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

    def _keyword_search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Pure BM25 keyword search."""
        raw_results = self.index_manager.search_bm25(query, k=k)
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

        fusion_k = int(os.environ.get("FUSION_K", "20"))  # k=20 wins over k=60 (sharper rank fusion)
        candidate_k = 50  # Retrieve 50 from each source

        # Determine content mode and weights
        content_mode = os.environ.get("CONTENT_MODE", "code").lower()
        vw = float(os.environ.get("VECTOR_WEIGHT", "0"))
        bw = float(os.environ.get("BM25_WEIGHT", "0"))
        if vw > 0 or bw > 0:
            vector_weight, bm25_weight = vw or 0.5, bw or 0.5
        else:
            vector_weight, bm25_weight = CONTENT_MODE_WEIGHTS.get(
                content_mode, (0.5, 0.5)
            )

        # Vector search
        optimized_query = self._optimize_query(query)
        query_embedding = self._get_query_embedding(optimized_query)
        vector_raw = self.index_manager.search(query_embedding, candidate_k, filters)
        vector_pairs = [(chunk_id, sim) for chunk_id, sim, _meta in vector_raw]

        # BM25 search: LLM rewrite (if enabled) then static expansion
        bm25_query = query
        if os.environ.get("BM25_REWRITE", "off") == "on":
            from search.query_rewriter import rewrite_query_for_bm25
            bm25_query = rewrite_query_for_bm25(query)
        if os.environ.get("QUERY_EXPANSION", "on") == "on":
            bm25_query = expand_code_query(bm25_query)
        bm25_raw = self.index_manager.search_bm25(bm25_query, k=candidate_k)
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

        # Apply chunk type boosts and name-match boost based on content mode
        boosts = CHUNK_TYPE_BOOSTS.get(content_mode, {})
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

        # Reranking. Default mode is "sonnet" (validated 2026-05-03 PR #93+:
        # +0.087 MRR, +0.137 HR@1 on n=183 multi-target real_session). The
        # Sonnet reranker is graceful: on missing ANTHROPIC_API_KEY, timeout,
        # or any error, it silently returns input candidates unchanged.
        # Disable explicitly with RERANKER=off. Legacy cross-encoder via
        # RERANKER=cross-encoder (off-by-default since A/B showed quality
        # regression).
        rerank_mode = os.environ.get("RERANKER", "sonnet").lower()
        if rerank_mode == "sonnet" and len(candidates) > k:
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
            reranked = rerank_with_sonnet(query, rerank_input, top_k=k)
            # Extract original SearchResult objects in new order; tail any
            # candidates beyond top-15 in their existing order.
            new_top = [d["_orig"] for d in reranked]
            tail = candidates[n_to_rerank:]
            candidates = new_top + tail
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
        else:
            # rerank_mode == "off" or empty candidates
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
