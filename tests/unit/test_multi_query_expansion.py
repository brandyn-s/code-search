"""Unit tests for Tier 1B multi-query expansion infrastructure.

Tests the multi_ranking_rrf helper and the SearchConfig wiring WITHOUT
making real LLM calls. Integration test (with actual Haiku alternatives
and fresh search) is the A/B eval in docs/EVAL_RUNBOOK.md.

What this covers:
  - multi_ranking_rrf with N rankings (N=1, 2, 3, 4) produces
    deterministic scores matching the canonical RRF formula
  - multi_ranking_rrf with 2 rankings is equivalent to the legacy
    reciprocal_rank_fusion call (regression guard against the
    refactor that swapped reciprocal_rank_fusion → multi_ranking_rrf
    in _hybrid_search)
  - SearchConfig.short_query_rewrite reads SHORT_QUERY_REWRITE env var
  - is_short_natural_query correctly identifies the gated cohort
"""
from __future__ import annotations

import pytest

from search.searcher import multi_ranking_rrf, reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# multi_ranking_rrf equivalence + correctness
# ---------------------------------------------------------------------------


def test_multi_ranking_rrf_two_rankings_matches_legacy_rrf():
    """multi_ranking_rrf([(vec, w_v), (bm25, w_b)]) must equal
    reciprocal_rank_fusion(vec, bm25, vector_weight=w_v, bm25_weight=w_b).

    This is the regression guard against the searcher.py refactor that
    moved the 2-ranking call to the N-ranking helper."""
    vec = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    bm25 = [("b", 0.5), ("d", 0.4), ("a", 0.3)]

    legacy = reciprocal_rank_fusion(
        vec, bm25, k=60, vector_weight=0.65, bm25_weight=0.35
    )
    new = multi_ranking_rrf(
        [(vec, 0.65), (bm25, 0.35)], k=60
    )

    assert legacy == new


def test_multi_ranking_rrf_single_ranking():
    """N=1 case: scores equal weight / (k + rank + 1)."""
    out = multi_ranking_rrf([([("a", 1.0), ("b", 0.5)], 1.0)], k=60)
    # Two items only; rank 0 score = 1/61, rank 1 score = 1/62
    assert out[0][0] == "a"
    assert out[1][0] == "b"
    assert abs(out[0][1] - 1.0 / 61) < 1e-12
    assert abs(out[1][1] - 1.0 / 62) < 1e-12


def test_multi_ranking_rrf_four_rankings_consistency():
    """N=4 multi-query case (original vector + original BM25 + alt vector
    + alt BM25). Item appearing in all 4 with rank 1 dominates."""
    rankings = [
        ([("x", 1.0), ("y", 0.5)], 1.0),
        ([("x", 1.0), ("z", 0.5)], 1.0),
        ([("x", 1.0), ("w", 0.5)], 1.0),
        ([("x", 1.0), ("v", 0.5)], 1.0),
    ]
    out = multi_ranking_rrf(rankings, k=60)
    assert out[0][0] == "x"
    # x appears in 4 rankings at rank 0 → score = 4 / 61
    assert abs(out[0][1] - 4.0 / 61) < 1e-12


def test_multi_ranking_rrf_weights_propagate():
    """Weight=0 ranking contributes nothing; weight=2 doubles contribution."""
    rankings = [
        ([("a", 1.0)], 0.0),  # zero-weight: no contribution
        ([("b", 1.0)], 2.0),  # double-weight
    ]
    out = multi_ranking_rrf(rankings, k=60)
    # Only b is in the output; a's zero-weight contribution is 0 but it
    # still appears in the score map with score 0.
    out_dict = dict(out)
    assert "a" in out_dict
    assert "b" in out_dict
    assert out_dict["a"] == 0.0
    assert out_dict["b"] == 2.0 / 61
    # b sorts first
    assert out[0][0] == "b"


def test_multi_ranking_rrf_empty_input():
    """No rankings -> empty result."""
    assert multi_ranking_rrf([], k=60) == []


def test_multi_ranking_rrf_empty_ranking():
    """One empty + one populated ranking."""
    out = multi_ranking_rrf([([], 1.0), ([("a", 1.0)], 1.0)], k=60)
    assert out == [("a", 1.0 / 61)]


# ---------------------------------------------------------------------------
# SearchConfig wiring
# ---------------------------------------------------------------------------


def test_search_config_short_query_rewrite_default_off(monkeypatch):
    """SHORT_QUERY_REWRITE unset -> short_query_rewrite=False."""
    monkeypatch.delenv("SHORT_QUERY_REWRITE", raising=False)
    # Clear cache so we re-parse env
    from search.config import get_search_config
    get_search_config.cache_clear()
    cfg = get_search_config()
    assert cfg.short_query_rewrite is False


def test_search_config_short_query_rewrite_on(monkeypatch):
    """SHORT_QUERY_REWRITE=on -> short_query_rewrite=True."""
    monkeypatch.setenv("SHORT_QUERY_REWRITE", "on")
    from search.config import get_search_config
    get_search_config.cache_clear()
    cfg = get_search_config()
    assert cfg.short_query_rewrite is True


# ---------------------------------------------------------------------------
# is_short_natural_query gate
# ---------------------------------------------------------------------------


def test_is_short_natural_query_positive_cases():
    """Short queries with no code tokens are gated in."""
    from search.query_rewriter import is_short_natural_query
    assert is_short_natural_query("auth login") is True
    assert is_short_natural_query("network management") is True
    assert is_short_natural_query("VPN connection") is True


def test_is_short_natural_query_negative_token_threshold():
    """5+ tokens excluded (median real-session was 4 words)."""
    from search.query_rewriter import is_short_natural_query
    assert is_short_natural_query("auth login token jwt session") is False


def test_is_short_natural_query_negative_code_tokens():
    """CamelCase / snake_case / dotted / parens exclude the query."""
    from search.query_rewriter import is_short_natural_query
    assert is_short_natural_query("HashedPassword auth") is False  # CamelCase
    assert is_short_natural_query("auth_token jwt") is False        # snake_case
    assert is_short_natural_query("foo.bar method") is False         # dotted
    assert is_short_natural_query("function(args)") is False         # parens


def test_is_short_natural_query_empty():
    """Empty / whitespace -> False."""
    from search.query_rewriter import is_short_natural_query
    assert is_short_natural_query("") is False
    assert is_short_natural_query("   ") is False
