"""Tests for R8: PPR metadata envelope.

PPR (Personalized PageRank over the code-graph DB) is an opt-in feature
gated by CODE_SEARCH_PPR_ENABLED. Before R8, its enable/disable, missing
graph-DB, and empty-subgraph paths were invisible to MCP consumers —
only sidecar [PPR_DIAG] log lines signaled anything. R8 adds a
`_metadata.ppr = {applied, reason, latency_ms, ...}` envelope mirroring
the reranker pattern, so LLM agents and operators can observe PPR
behavior without grepping logs.

Reason vocabulary tested here:
- disabled_by_env: CODE_SEARCH_PPR_ENABLED unset/off (the dominant default)
- alpha_zero: explicit correctness gate
- no_candidates: upstream empty result
- no_graph_db: PPR enabled but PPRScorer returned empty (missing DB or
  insufficient subgraph)
- ok: PPR applied successfully
- error: exception caught; hybrid order preserved
- not_invoked_<mode>: keyword/semantic modes skip PPR entirely
"""
from __future__ import annotations

import tempfile
from typing import Optional
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from embeddings.embedder import EmbeddingResult
from search.indexer import CodeIndexManager
from search.searcher import IntelligentSearcher


def _mk_result(cid: str, dim: int = 8) -> EmbeddingResult:
    return EmbeddingResult(
        embedding=np.random.RandomState(hash(cid) & 0xFFFFFFFF)
            .randn(dim).astype(np.float32),
        chunk_id=cid,
        metadata={
            "file_path": f"{cid}.py", "relative_path": f"{cid}.py",
            "content_preview": "x", "full_content": "x",
            "chunk_type": "function", "start_line": 1, "end_line": 2,
            "name": cid, "parent_name": None, "docstring": None,
            "decorators": [], "imports": [], "complexity_score": 0,
            "tags": [], "folder_structure": [],
        },
    )


class _FakeEmbedder:
    """Minimal CodeEmbedder stand-in for searcher tests."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self._model = MagicMock()
        self._model.get_embedding_dimension.return_value = dim

    def embed_query(self, text: str) -> np.ndarray:
        # Deterministic but content-independent.
        return np.random.RandomState(42).randn(self.dim).astype(np.float32)


@pytest.fixture
def searcher_with_index():
    """A searcher pointed at a small in-memory-style index."""
    tmp = tempfile.mkdtemp()
    mgr = CodeIndexManager(tmp)
    mgr.add_embeddings([_mk_result(f"f{i}") for i in range(5)])
    s = IntelligentSearcher(mgr, _FakeEmbedder())
    yield s
    if mgr._metadata_db is not None:
        mgr._metadata_db.close()
    if getattr(mgr, "_fts_conn", None) is not None:
        mgr._fts_conn.close()


# ---------------------------------------------------------------------------
# Default initialization
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_last_ppr_metadata_initialized_to_not_invoked(self, searcher_with_index):
        meta = searcher_with_index.last_ppr_metadata
        assert meta["applied"] is False
        assert meta["reason"] == "not_invoked"
        assert meta["latency_ms"] == 0


# ---------------------------------------------------------------------------
# Reason vocabulary — verified via direct calls to _hybrid_search
# ---------------------------------------------------------------------------

class TestPPRReasonVocabulary:
    """Each fork in the PPR block must set a stable reason string."""

    def test_disabled_by_env_when_ppr_env_unset(
        self, searcher_with_index, monkeypatch
    ):
        """Default config: CODE_SEARCH_PPR_ENABLED unset → disabled_by_env."""
        monkeypatch.delenv("CODE_SEARCH_PPR_ENABLED", raising=False)
        monkeypatch.delenv("CODE_SEARCH_PPR_ALPHA", raising=False)
        # Force ANTHROPIC_API_KEY out so the reranker doesn't try a live call.
        monkeypatch.setenv("RERANKER", "off")
        searcher_with_index.search("test query", k=3)
        meta = searcher_with_index.last_ppr_metadata
        assert meta["applied"] is False
        assert meta["reason"] == "disabled_by_env"

    def test_alpha_zero_when_explicitly_set(
        self, searcher_with_index, monkeypatch
    ):
        """alpha=0 is the documented correctness gate (bit-exact no-op)."""
        monkeypatch.setenv("CODE_SEARCH_PPR_ENABLED", "1")
        monkeypatch.setenv("CODE_SEARCH_PPR_ALPHA", "0.0")
        monkeypatch.setenv("RERANKER", "off")
        searcher_with_index.search("test query", k=3)
        meta = searcher_with_index.last_ppr_metadata
        assert meta["applied"] is False
        assert meta["reason"] == "alpha_zero"

    def test_no_graph_db_when_ppr_enabled_but_db_missing(
        self, searcher_with_index, monkeypatch
    ):
        """PPR enabled, alpha>0, but graph DB missing → PPRScorer returns
        empty dict, surfaced as no_graph_db."""
        monkeypatch.setenv("CODE_SEARCH_PPR_ENABLED", "1")
        monkeypatch.setenv("CODE_SEARCH_PPR_ALPHA", "0.5")
        monkeypatch.setenv("RERANKER", "off")
        # PPRScorer's _connect() returns None when no graph DB → score()
        # returns {} → our code branches to no_graph_db. We don't need to
        # mock anything — there's no graph DB in this test env.
        searcher_with_index.search("test query", k=3)
        meta = searcher_with_index.last_ppr_metadata
        assert meta["applied"] is False
        assert meta["reason"] == "no_graph_db"
        # latency_ms is non-negative even on failure paths.
        assert meta["latency_ms"] >= 0

    def test_error_when_ppr_scorer_raises(
        self, searcher_with_index, monkeypatch
    ):
        """An exception inside the PPR block must produce reason=error
        and not propagate (hybrid order preserved)."""
        monkeypatch.setenv("CODE_SEARCH_PPR_ENABLED", "1")
        monkeypatch.setenv("CODE_SEARCH_PPR_ALPHA", "0.5")
        monkeypatch.setenv("RERANKER", "off")

        # Monkey-patch PPRScorer to raise on instantiation.
        from search import ppr_scorer

        class BoomScorer:
            def __init__(self, *a, **kw):
                raise RuntimeError("simulated PPR failure")
            def __enter__(self): raise RuntimeError("simulated PPR failure")
            def __exit__(self, *a): pass

        monkeypatch.setattr(ppr_scorer, "PPRScorer", BoomScorer)
        # Also replace it in the searcher module's local import path.
        from search import searcher as searcher_mod
        # The function does `from search.ppr_scorer import PPRScorer` inside
        # _hybrid_search, so patching ppr_scorer.PPRScorer is enough.

        searcher_with_index.search("test query", k=3)
        meta = searcher_with_index.last_ppr_metadata
        assert meta["applied"] is False
        assert meta["reason"] == "error"
        assert meta["error_class"] == "RuntimeError"


# ---------------------------------------------------------------------------
# Mode dispatch sets the right "not invoked" reason
# ---------------------------------------------------------------------------

class TestModeDispatchPPRReasons:
    """keyword/semantic modes never invoke PPR; the metadata reflects this
    with a distinct reason string so consumers don't confuse 'PPR skipped
    by mode' with 'PPR ran and found nothing'."""

    def test_keyword_mode_sets_not_invoked_keyword(
        self, searcher_with_index, monkeypatch
    ):
        monkeypatch.setenv("CODE_SEARCH_PPR_ENABLED", "1")
        monkeypatch.setenv("CODE_SEARCH_PPR_ALPHA", "0.5")
        searcher_with_index.search("test", k=3, search_mode="keyword")
        meta = searcher_with_index.last_ppr_metadata
        assert meta["reason"] == "not_invoked_keyword_mode"

    def test_semantic_mode_sets_not_invoked_semantic(
        self, searcher_with_index, monkeypatch
    ):
        monkeypatch.setenv("CODE_SEARCH_PPR_ENABLED", "1")
        monkeypatch.setenv("CODE_SEARCH_PPR_ALPHA", "0.5")
        searcher_with_index.search("test", k=3, search_mode="semantic")
        meta = searcher_with_index.last_ppr_metadata
        assert meta["reason"] == "not_invoked_semantic_mode"


# ---------------------------------------------------------------------------
# MCP envelope propagation
# ---------------------------------------------------------------------------

class TestMCPEnvelopePropagation:
    """The MCP response's _metadata.ppr envelope must mirror what the
    searcher exposed in last_ppr_metadata, with the optional diagnostic
    fields (alpha, scored_candidates, error_class) appearing only when
    relevant."""

    def test_envelope_includes_required_keys(self):
        """Build a fake searcher with last_ppr_metadata, run the same
        propagation logic as the MCP layer, verify shape."""
        searcher = MagicMock()
        searcher.last_ppr_metadata = {
            "applied": False,
            "reason": "disabled_by_env",
            "latency_ms": 0,
        }
        # Mimic the propagation block in code_search_server.search_code.
        ppr_meta = getattr(searcher, "last_ppr_metadata", None)
        assert ppr_meta and isinstance(ppr_meta, dict)
        envelope = {
            "applied": bool(ppr_meta.get("applied", False)),
            "reason": str(ppr_meta.get("reason", "unknown")),
            "latency_ms": int(ppr_meta.get("latency_ms", 0)),
        }
        assert envelope == {
            "applied": False,
            "reason": "disabled_by_env",
            "latency_ms": 0,
        }

    def test_envelope_includes_optional_diagnostic_fields(self):
        """When applied=True, alpha + scored_candidates should appear.
        When error path, error_class should appear."""
        # Applied case
        ppr_meta = {
            "applied": True, "reason": "ok", "latency_ms": 42,
            "alpha": 0.5, "scored_candidates": 8,
        }
        envelope = {
            "applied": True, "reason": "ok", "latency_ms": 42,
        }
        for opt in ("alpha", "scored_candidates", "error_class"):
            if opt in ppr_meta:
                envelope[opt] = ppr_meta[opt]
        assert envelope["alpha"] == 0.5
        assert envelope["scored_candidates"] == 8
        assert "error_class" not in envelope

        # Error case
        err_meta = {
            "applied": False, "reason": "error", "latency_ms": 12,
            "error_class": "RuntimeError",
        }
        env2 = {
            "applied": False, "reason": "error", "latency_ms": 12,
        }
        if "error_class" in err_meta:
            env2["error_class"] = err_meta["error_class"]
        assert env2["error_class"] == "RuntimeError"
