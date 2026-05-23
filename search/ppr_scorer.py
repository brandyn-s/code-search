"""PPR scorer over code-graph for the post-boost-sort retrieval stage.

Arc A of 2026-05-11 graph-augmented + per-option .nix plan.

Personalized PageRank over code-graph's CALLS/USAGE/DEFINES/IMPLEMENTS/OVERRIDE/CONFIGURES
edges. Seeds PPR walks from each candidate weighted by its post-boost similarity_score,
walks K iterations with restart probability beta, and returns a per-candidate PPR score.

Integration: search/searcher.py blends final_score = similarity_score * (1 + alpha * ppr_norm)
after the Phase H boost-sort and before the sonnet rerank branch.

Mechanism correctness (Plan A2 gate): with alpha=0.0, output must be identical to baseline.
The blend is the only place PPR influences final ranking; if alpha is 0 we skip the blend
entirely (early return) -> baseline behavior preserved bit-exactly.

Failure modes: code-graph DB missing -> log + return empty dict (caller treats as no signal).
Subgraph too small (< 2 candidates with graph nodes) -> return empty dict.
Latency budget: target < 200ms p99 for 15-candidate cohort. SQLite WAL read is local.
"""
from __future__ import annotations
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

LOG = logging.getLogger(__name__)

# Edges that carry useful directional signal for re-ranking. IMPORTS is excluded
# per HippoRAG / Nexus convention (high-fan-out, low-precision signal).
DEFAULT_WALK_EDGES: Tuple[str, ...] = (
    "CALLS", "USAGE", "DEFINES", "IMPLEMENTS", "OVERRIDE", "CONFIGURES",
)

_CODE_GRAPH_DB_DIR = Path.home() / ".cache" / "codebase-memory-mcp"


def _project_repo_path_for_file(file_path: str) -> Optional[Path]:
    """Walk up from a file path looking for a git root; return the directory."""
    p = Path(file_path).resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _code_graph_db_for_repo(repo_root: Path) -> Optional[Path]:
    """Compute the code-graph SQLite path for a given repo root.

    code-graph names DBs by sanitizing the absolute path -> replacing '/' and '\\'
    with '-' and prefixing 'c-' (drive letter). E.g.
    C:\\Users\\X\\Documents\\GitHub\\repo -> c-Users-X-Documents-GitHub-repo.db
    """
    abspath = str(repo_root).replace("\\", "/")
    # Strip drive letter; expect like 'C:/Users/...'
    if len(abspath) >= 2 and abspath[1] == ":":
        sanitized = "c-" + abspath[3:].replace("/", "-")
    else:
        sanitized = abspath.lstrip("/").replace("/", "-")
    db_path = _CODE_GRAPH_DB_DIR / f"{sanitized}.db"
    if db_path.exists():
        return db_path
    return None


def _normalize_relpath(relpath: str) -> str:
    """Normalize a relative path to match code-graph's file_path field."""
    return relpath.replace("\\", "/").lstrip("/")


class PPRScorer:
    """Personalized PageRank scorer reading code-graph SQLite directly.

    Reuses one sqlite connection per scorer instance (open lazily on first score call).
    Caller is responsible for instantiating per-request and discarding afterward;
    in-process reuse is safe but the read-only access does not need to outlive a
    single search call.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        walk_edges: Sequence[str] = DEFAULT_WALK_EDGES,
        max_subgraph_nodes: int = 2000,
        iterations: int = 5,
        beta: float = 0.85,
    ) -> None:
        self.db_path = db_path
        self.walk_edges = tuple(walk_edges)
        self.max_subgraph_nodes = max_subgraph_nodes
        self.iterations = iterations
        self.beta = beta
        self._con: Optional[sqlite3.Connection] = None

    def _connect(self, hint_path: Optional[str] = None) -> Optional[sqlite3.Connection]:
        """Open the code-graph DB. hint_path is any candidate's absolute file path
        from which we infer the repo root, used only when db_path was not explicitly set."""
        if self._con is not None:
            return self._con
        db = self.db_path
        if db is None and hint_path:
            repo = _project_repo_path_for_file(hint_path)
            if repo is not None:
                db = _code_graph_db_for_repo(repo)
        if db is None:
            LOG.info("[PPR_DIAG] code_graph_db_not_found hint=%r", hint_path)
            return None
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            self._con = con
            return con
        except sqlite3.Error as e:
            LOG.warning("[PPR_DIAG] code_graph_db_open_failed db=%s err=%s", db, e)
            return None

    def close(self) -> None:
        if self._con is not None:
            try:
                self._con.close()
            finally:
                self._con = None

    def __enter__(self) -> "PPRScorer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def score(
        self,
        candidate_paths_scores: List[Tuple[str, float]],
        hint_abs_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute a per-candidate-path PPR score.

        Args:
          candidate_paths_scores: list of (relative_path, similarity_score) for the
            ranked candidate pool. similarity_score is the post-boost score from
            searcher.py.
          hint_abs_path: any absolute path of a candidate file (used to locate the
            code-graph DB for that repo).

        Returns:
          {relative_path: ppr_score_in_[0,1]} for paths whose nodes we could locate
          in code-graph. Paths absent from this dict get neutral blending (alpha*0).
        """
        if not candidate_paths_scores:
            return {}
        t0 = time.monotonic()
        con = self._connect(hint_abs_path)
        if con is None:
            return {}

        # 1. Find node IDs per candidate path. Skip paths with zero nodes.
        path_node_ids: Dict[str, List[int]] = {}
        for relpath, _ in candidate_paths_scores:
            norm = _normalize_relpath(relpath)
            rows = con.execute(
                "SELECT id FROM nodes WHERE file_path = ? OR file_path LIKE ?",
                (norm, f"%/{norm}"),
            ).fetchall()
            if rows:
                path_node_ids[relpath] = [r[0] for r in rows]

        if len(path_node_ids) < 2:
            LOG.info(
                "[PPR_DIAG] insufficient_subgraph candidates=%d with_nodes=%d t_ms=%d",
                len(candidate_paths_scores), len(path_node_ids),
                int((time.monotonic() - t0) * 1000),
            )
            return {}

        # 2. Expand 1-hop neighbors to build subgraph adjacency.
        seed_ids = set()
        for nids in path_node_ids.values():
            seed_ids.update(nids)

        edge_filter = "(" + ",".join("?" * len(self.walk_edges)) + ")"
        placeholders = ",".join("?" * len(seed_ids))
        params = (*seed_ids, *self.walk_edges)
        out_rows = con.execute(
            f"SELECT source_id, target_id FROM edges WHERE source_id IN ({placeholders}) AND type IN {edge_filter}",
            params,
        ).fetchall()
        in_rows = con.execute(
            f"SELECT source_id, target_id FROM edges WHERE target_id IN ({placeholders}) AND type IN {edge_filter}",
            params,
        ).fetchall()

        # Subgraph node set
        sub_nodes = set(seed_ids)
        edges_pair: List[Tuple[int, int]] = []
        for s, t in out_rows:
            sub_nodes.add(t)
            edges_pair.append((s, t))
        for s, t in in_rows:
            sub_nodes.add(s)
            edges_pair.append((s, t))

        if len(sub_nodes) > self.max_subgraph_nodes:
            # Truncate to seed nodes only — large 1-hop expansions hurt latency.
            sub_nodes = set(seed_ids)
            edges_pair = [(s, t) for (s, t) in edges_pair if s in sub_nodes and t in sub_nodes]

        # 3. Build adjacency and seed vector
        node_index = {nid: i for i, nid in enumerate(sub_nodes)}
        n = len(sub_nodes)
        # Out-degree for normalization
        out_neighbors: Dict[int, List[int]] = {i: [] for i in range(n)}
        for s, t in edges_pair:
            si = node_index.get(s)
            ti = node_index.get(t)
            if si is None or ti is None:
                continue
            # Bidirectional walk: edges contribute to both directions equally
            out_neighbors[si].append(ti)
            out_neighbors[ti].append(si)
        out_degree = [max(len(out_neighbors[i]), 1) for i in range(n)]

        # Seed: distribute each candidate's similarity_score uniformly across its nodes,
        # normalize across all seeds so seed sums to 1.
        seed = [0.0] * n
        total_seed = 0.0
        for relpath, sim in candidate_paths_scores:
            nids = path_node_ids.get(relpath)
            if not nids:
                continue
            sim_pos = max(sim, 0.0)
            if sim_pos == 0.0:
                continue
            per_node = sim_pos / len(nids)
            for nid in nids:
                idx = node_index.get(nid)
                if idx is not None:
                    seed[idx] += per_node
                    total_seed += per_node
        if total_seed <= 0.0:
            return {}
        for i in range(n):
            seed[i] /= total_seed  # normalize

        # 4. Power iteration: r_{k+1} = beta * walk(r_k) + (1-beta) * seed
        r = list(seed)
        beta = self.beta
        for _ in range(self.iterations):
            new_r = [0.0] * n
            for i in range(n):
                if r[i] == 0.0:
                    continue
                contrib = r[i] / out_degree[i]
                for j in out_neighbors[i]:
                    new_r[j] += contrib
            for i in range(n):
                r[i] = beta * new_r[i] + (1 - beta) * seed[i]

        # 5. Aggregate per-candidate-path PPR
        max_path_score = 0.0
        path_scores: Dict[str, float] = {}
        for relpath, _ in candidate_paths_scores:
            nids = path_node_ids.get(relpath)
            if not nids:
                continue
            score = sum(r[node_index[nid]] for nid in nids if nid in node_index) / len(nids)
            path_scores[relpath] = score
            if score > max_path_score:
                max_path_score = score

        # Normalize so the highest path PPR is 1.0; relative ranking is what matters.
        if max_path_score > 0.0:
            for k in path_scores:
                path_scores[k] /= max_path_score

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        LOG.info(
            "[PPR_DIAG] computed candidates=%d with_nodes=%d sub_nodes=%d sub_edges=%d t_ms=%d",
            len(candidate_paths_scores), len(path_node_ids), n, len(edges_pair), elapsed_ms,
        )
        return path_scores


def blend_ppr_into_candidates(candidates, alpha: float, ppr_scores: Dict[str, float]) -> None:
    """Mutate each candidate's similarity_score by (1 + alpha * ppr).

    Candidates without a PPR score get neutral blending (alpha * 0 = 0 -> score unchanged).
    Mechanism correctness gate: if alpha == 0.0, this is a no-op.
    """
    if alpha == 0.0:
        return
    for r in candidates:
        relpath = getattr(r, "relative_path", "") or ""
        ppr = ppr_scores.get(relpath, 0.0)
        # multiplicative blend, bounded by (1 + alpha)
        r.similarity_score *= (1.0 + alpha * ppr)


def get_env_config() -> Tuple[bool, float]:
    """Return ``(enabled, alpha)`` from env vars. Defaults: disabled, alpha=0.5.

    R11 phase 2: same env var names + same parsing helpers as
    ``SearchConfig.ppr_enabled`` / ``SearchConfig.ppr_alpha`` (both go
    through ``search.config.parse_env_*``), but this wrapper reads env
    fresh on every call instead of going through the cached SearchConfig.

    Why bypass the cache: legacy callers (and the existing test suite)
    expect this function to reflect env changes immediately, including
    multiple env mutations within a single test. The cached config is
    invalidated only between tests by the autouse fixture in conftest.

    New call sites should prefer ``cfg.ppr_enabled`` / ``cfg.ppr_alpha``
    via ``get_search_config()`` — they're typed and validated identically.
    """
    from search.config import parse_env_bool, parse_env_float
    enabled = parse_env_bool("CODE_SEARCH_PPR_ENABLED", default=False)
    alpha = parse_env_float(
        "CODE_SEARCH_PPR_ALPHA", default=0.5, min_value=0.0,
    )
    return enabled, alpha
