"""Smoke tests for search.ppr_scorer (R7).

The PPR scorer (305 LOC) shipped with zero unit tests — the only validation
was via searcher integration tests. This file fills the gap with focused
unit coverage of the documented correctness invariants and failure modes:

- alpha=0.0 is a bit-exact no-op (the mechanism-correctness gate from the
  ppr_scorer.py module docstring).
- get_env_config: default disabled + alpha=0.5; malformed alpha falls back.
- Empty input → empty result.
- Missing graph DB → empty result (caller treats as no signal).
- Insufficient subgraph (<2 candidates with nodes) → empty result.
- Happy path with a minimal in-memory graph DB → returns normalized scores.
- Path normalization handles Windows-style separators and leading slashes.
- Connection lifecycle: __enter__/__exit__ closes, close() is idempotent.

Not covered (out of smoke-test scope): the full PageRank power-iteration
correctness — that's an algorithm proof, not a behavioral test. The
existing eval suite (bench/research/) covers retrieval-quality regressions
end-to-end.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from search.ppr_scorer import (
    PPRScorer,
    blend_ppr_into_candidates,
    get_env_config,
    _normalize_relpath,
)


# ---------------------------------------------------------------------------
# blend_ppr_into_candidates — the alpha=0 correctness gate
# ---------------------------------------------------------------------------

class TestBlendCorrectness:
    """alpha=0.0 must be a bit-exact no-op on candidate scores, regardless
    of what's in ppr_scores. This is the documented invariant that allows
    PPR to be enabled without risk of regressing rankings."""

    def test_alpha_zero_is_bit_exact_noop(self):
        cands = [
            SimpleNamespace(relative_path="a.py", similarity_score=0.8),
            SimpleNamespace(relative_path="b.py", similarity_score=0.5),
            SimpleNamespace(relative_path="c.py", similarity_score=0.3),
        ]
        before = [c.similarity_score for c in cands]
        blend_ppr_into_candidates(
            cands, alpha=0.0,
            ppr_scores={"a.py": 1.0, "b.py": 0.9, "c.py": 0.5},
        )
        after = [c.similarity_score for c in cands]
        assert before == after, (
            "alpha=0.0 must leave similarity_score unchanged bit-exactly; "
            "this is the correctness gate documented in ppr_scorer.py"
        )

    def test_alpha_positive_scales_scores_by_one_plus_alpha_ppr(self):
        cands = [
            SimpleNamespace(relative_path="a.py", similarity_score=1.0),
            SimpleNamespace(relative_path="b.py", similarity_score=1.0),
        ]
        blend_ppr_into_candidates(
            cands, alpha=0.5, ppr_scores={"a.py": 1.0, "b.py": 0.0},
        )
        # a: 1.0 * (1 + 0.5 * 1.0) = 1.5
        # b: 1.0 * (1 + 0.5 * 0.0) = 1.0
        assert cands[0].similarity_score == pytest.approx(1.5)
        assert cands[1].similarity_score == pytest.approx(1.0)

    def test_candidates_without_ppr_score_get_neutral_blend(self):
        """Candidates absent from ppr_scores get alpha*0=0 blend, leaving
        their similarity_score unchanged. Ensures PPR can't accidentally
        zero out candidates that just lack graph coverage."""
        cands = [
            SimpleNamespace(relative_path="known.py", similarity_score=0.5),
            SimpleNamespace(relative_path="unknown.py", similarity_score=0.5),
        ]
        blend_ppr_into_candidates(
            cands, alpha=0.5, ppr_scores={"known.py": 1.0},
        )
        # known: 0.5 * (1 + 0.5 * 1.0) = 0.75
        # unknown: 0.5 (no change — not in ppr_scores)
        assert cands[0].similarity_score == pytest.approx(0.75)
        assert cands[1].similarity_score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# get_env_config — env var parsing
# ---------------------------------------------------------------------------

class TestGetEnvConfig:
    def test_unset_defaults(self, monkeypatch):
        monkeypatch.delenv("CODE_SEARCH_PPR_ENABLED", raising=False)
        monkeypatch.delenv("CODE_SEARCH_PPR_ALPHA", raising=False)
        enabled, alpha = get_env_config()
        assert enabled is False
        assert alpha == 0.5

    def test_enabled_truthy_values(self, monkeypatch):
        for val in ("1", "true", "yes", "on", "TRUE", " 1 "):
            monkeypatch.setenv("CODE_SEARCH_PPR_ENABLED", val)
            enabled, _ = get_env_config()
            assert enabled is True, f"expected truthy for {val!r}"

    def test_enabled_falsy_values(self, monkeypatch):
        for val in ("0", "false", "no", "off", "", "abc"):
            monkeypatch.setenv("CODE_SEARCH_PPR_ENABLED", val)
            enabled, _ = get_env_config()
            assert enabled is False, f"expected falsy for {val!r}"

    def test_alpha_malformed_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("CODE_SEARCH_PPR_ALPHA", "not_a_float")
        _, alpha = get_env_config()
        assert alpha == 0.5, "malformed alpha must fall back to 0.5, not crash"

    def test_alpha_valid_values(self, monkeypatch):
        for raw, expected in [("0", 0.0), ("0.0", 0.0), ("0.5", 0.5), ("1.0", 1.0)]:
            monkeypatch.setenv("CODE_SEARCH_PPR_ALPHA", raw)
            _, alpha = get_env_config()
            assert alpha == expected


# ---------------------------------------------------------------------------
# Path normalization
# ---------------------------------------------------------------------------

class TestNormalizeRelpath:
    def test_windows_backslashes_normalized(self):
        assert _normalize_relpath("src\\auth.py") == "src/auth.py"

    def test_leading_slash_stripped(self):
        assert _normalize_relpath("/src/auth.py") == "src/auth.py"

    def test_already_normalized_unchanged(self):
        assert _normalize_relpath("src/auth.py") == "src/auth.py"


# ---------------------------------------------------------------------------
# PPRScorer.score — failure modes
# ---------------------------------------------------------------------------

class TestScoreFailureModes:
    def test_empty_input_returns_empty(self):
        scorer = PPRScorer()
        assert scorer.score([]) == {}

    def test_missing_graph_db_returns_empty(self, tmp_path):
        """db_path explicitly set to a nonexistent file → _connect returns
        None → score returns {}."""
        scorer = PPRScorer(db_path=tmp_path / "does_not_exist.db")
        result = scorer.score(
            [("a.py", 0.5), ("b.py", 0.3)],
            hint_abs_path=str(tmp_path / "a.py"),
        )
        assert result == {}

    def test_no_db_no_hint_returns_empty(self):
        """No db_path AND no hint_abs_path → can't locate code-graph DB."""
        scorer = PPRScorer()
        result = scorer.score([("a.py", 0.5)])
        assert result == {}

    def test_insufficient_subgraph_returns_empty(self, tmp_path):
        """When <2 candidates have nodes in the graph, the subgraph is too
        small for meaningful PageRank — return empty (no signal)."""
        db = tmp_path / "graph.db"
        con = sqlite3.connect(str(db))
        con.executescript("""
            CREATE TABLE nodes (id INTEGER PRIMARY KEY, file_path TEXT);
            CREATE TABLE edges (source_id INTEGER, target_id INTEGER, type TEXT);
            -- Only one candidate has a node; the other doesn't match.
            INSERT INTO nodes VALUES (1, 'src/a.py');
        """)
        con.commit()
        con.close()

        scorer = PPRScorer(db_path=db)
        result = scorer.score([
            ("src/a.py", 0.8),
            ("src/missing.py", 0.5),
        ])
        assert result == {}, (
            "insufficient subgraph (<2 candidates with nodes) must return empty"
        )


# ---------------------------------------------------------------------------
# PPRScorer.score — happy path with a minimal in-memory graph
# ---------------------------------------------------------------------------

class TestScoreHappyPath:
    """Build a tiny code-graph DB with two interconnected candidate files
    and verify score() returns a dict with normalized scores."""

    def _build_minimal_graph_db(self, db: Path) -> None:
        con = sqlite3.connect(str(db))
        con.executescript("""
            CREATE TABLE nodes (id INTEGER PRIMARY KEY, file_path TEXT);
            CREATE TABLE edges (source_id INTEGER, target_id INTEGER, type TEXT);
            INSERT INTO nodes VALUES
                (1, 'src/a.py'),
                (2, 'src/b.py'),
                (3, 'src/c.py');
            -- a calls b, b uses c, c defines a — a small cycle so PageRank
            -- has something to do.
            INSERT INTO edges VALUES
                (1, 2, 'CALLS'),
                (2, 3, 'USAGE'),
                (3, 1, 'DEFINES');
        """)
        con.commit()
        con.close()

    def test_score_returns_normalized_dict_for_connected_candidates(self, tmp_path):
        db = tmp_path / "graph.db"
        self._build_minimal_graph_db(db)
        scorer = PPRScorer(db_path=db)

        result = scorer.score([
            ("src/a.py", 0.9),
            ("src/b.py", 0.7),
            ("src/c.py", 0.5),
        ])

        # All three candidates are in the graph → all should be in result.
        assert set(result.keys()) == {"src/a.py", "src/b.py", "src/c.py"}
        # Scores normalized so max = 1.0 (see ppr_scorer.py:270-272).
        assert max(result.values()) == pytest.approx(1.0)
        # And all scores in [0, 1].
        assert all(0.0 <= v <= 1.0 for v in result.values())


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------

class TestConnectionLifecycle:
    def test_close_is_idempotent(self, tmp_path):
        scorer = PPRScorer(db_path=tmp_path / "missing.db")
        scorer.close()  # no connection ever opened
        scorer.close()  # double-close must not raise

    def test_context_manager_closes_connection(self, tmp_path):
        db = tmp_path / "graph.db"
        con = sqlite3.connect(str(db))
        con.executescript("""
            CREATE TABLE nodes (id INTEGER PRIMARY KEY, file_path TEXT);
            CREATE TABLE edges (source_id INTEGER, target_id INTEGER, type TEXT);
            INSERT INTO nodes VALUES (1, 'x.py'), (2, 'y.py');
            INSERT INTO edges VALUES (1, 2, 'CALLS');
        """)
        con.commit()
        con.close()

        with PPRScorer(db_path=db) as scorer:
            scorer.score([("x.py", 0.5), ("y.py", 0.5)])
            assert scorer._con is not None, "connection should be open inside `with`"
        # After __exit__: connection released.
        assert scorer._con is None
