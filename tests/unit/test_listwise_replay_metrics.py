"""Unit tests for the listwise replay harness metric panel.

Covers: nDCG@10, Recall@k, pairwise win rate, stratification, and
top-failures inspection logic.
"""
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bench.research.listwise_replay import (
    compute_metrics, _matches, kendall_tau,
)
from bench.research.phase_c_verdict import (
    aggregate, pairwise_wins, stratify, top_failures,
)


# ---- compute_metrics ----

def test_metrics_perfect_rank_1():
    m = compute_metrics(
        ranked_files=["src/foo.py", "src/bar.py"],
        expected_files={"src/foo.py"},
        subproject=None,
    )
    assert m["rank"] == 1
    assert m["rr"] == 1.0
    assert m["hit_1"] is True
    assert m["hit_5"] is True
    assert m["hit_20"] is True
    # nDCG@10 with single relevant at rank 1 = 1/log2(2) / 1/log2(2) = 1.0
    assert m["ndcg_10"] == 1.0


def test_metrics_rank_2_ndcg():
    m = compute_metrics(
        ranked_files=["src/a.py", "src/b.py", "src/c.py"],
        expected_files={"src/b.py"},
        subproject=None,
    )
    assert m["rank"] == 2
    assert m["rr"] == 0.5
    assert m["hit_1"] is False
    assert m["hit_5"] is True
    # DCG = 1/log2(3) ~= 0.6309; IDCG = 1/log2(2) = 1.0
    assert math.isclose(m["ndcg_10"], 1.0 / math.log2(3), rel_tol=1e-6)


def test_metrics_no_match_ndcg_zero():
    m = compute_metrics(
        ranked_files=["src/x.py", "src/y.py"],
        expected_files={"src/z.py"},
        subproject=None,
    )
    assert m["rank"] is None
    assert m["rr"] == 0.0
    assert m["hit_1"] is False
    assert m["hit_5"] is False
    assert m["hit_20"] is False
    assert m["ndcg_10"] == 0.0


def test_metrics_match_outside_top_10_excluded_from_ndcg():
    """Match at rank 11+ contributes 0 to nDCG@10 but rank/RR still set."""
    ranked = [f"src/wrong_{i}.py" for i in range(10)] + ["src/right.py", "src/wrong_x.py"]
    m = compute_metrics(
        ranked_files=ranked,
        expected_files={"src/right.py"},
        subproject=None,
    )
    assert m["rank"] == 11
    assert m["rr"] == 1.0 / 11
    assert m["hit_5"] is False
    assert m["hit_20"] is True
    assert m["ndcg_10"] == 0.0


def test_metrics_subproject_prefix_match():
    """A ranked file like `assetman/src/foo.py` should match expected `src/foo.py`
    when subproject=assetman."""
    m = compute_metrics(
        ranked_files=["assetman/src/foo.py"],
        expected_files={"src/foo.py"},
        subproject="assetman",
    )
    assert m["rank"] == 1
    assert m["rr"] == 1.0


def test_metrics_no_expected_files_returns_zero():
    m = compute_metrics(ranked_files=["a", "b"], expected_files=set(), subproject=None)
    assert m["rank"] is None
    assert m["rr"] == 0.0
    assert m["ndcg_10"] == 0.0


def test_metrics_multiple_expected_files_ndcg_only_first_counted_for_rank():
    """With multiple expected, rank is the FIRST match; nDCG sums up to 2 hits."""
    m = compute_metrics(
        ranked_files=["src/a.py", "src/b.py", "src/c.py"],
        expected_files={"src/a.py", "src/c.py"},
        subproject=None,
    )
    assert m["rank"] == 1
    # IDCG with 2 ideal positions = 1/log2(2) + 1/log2(3) = 1 + 0.6309
    # DCG with hits at rank 1 and 3 = 1.0 + 1/log2(4) = 1.0 + 0.5 = 1.5
    # nDCG = 1.5 / 1.6309 ~= 0.9197
    expected_ndcg = (1.0 + 1.0 / math.log2(4)) / (1.0 + 1.0 / math.log2(3))
    assert math.isclose(m["ndcg_10"], expected_ndcg, rel_tol=1e-6)


# ---- _matches helper ----

def test_matches_exact():
    assert _matches("src/foo.py", {"src/foo.py"}, None) is True


def test_matches_backslash_normalization():
    """Caller normalizes; _matches sees already-normalized."""
    # _matches expects pre-normalized input per docstring; just confirm exact match
    assert _matches("src/foo.py", {"src/foo.py"}, None) is True
    assert _matches("src/foo.py", {"src/bar.py"}, None) is False


def test_matches_subproject_prefix():
    assert _matches("assetman/src/foo.py", {"src/foo.py"}, "assetman") is True
    assert _matches("other/src/foo.py", {"src/foo.py"}, "assetman") is True  # "/src/foo.py" suffix


# ---- kendall_tau ----

def test_kendall_tau_identical():
    a = ["x", "y", "z"]
    assert kendall_tau(a, a) == 1.0


def test_kendall_tau_reversed():
    a = ["x", "y", "z"]
    b = ["z", "y", "x"]
    assert kendall_tau(a, b) == -1.0


def test_kendall_tau_degenerate_single_item():
    assert kendall_tau(["x"], ["x"]) == 1.0
    assert kendall_tau([], []) == 1.0


# ---- aggregate ----

def test_aggregate_empty():
    a = aggregate([])
    assert a["n"] == 0
    assert a["mrr"] == 0.0


def test_aggregate_basic():
    rows = [
        {"rr": 1.0, "hit_1": True, "hit_5": True, "hit_20": True, "ndcg_10": 1.0,
         "reranker_applied": True, "reranker_latency_ms": 100},
        {"rr": 0.5, "hit_1": False, "hit_5": True, "hit_20": True, "ndcg_10": 0.63,
         "reranker_applied": True, "reranker_latency_ms": 200},
        {"rr": 0.0, "hit_1": False, "hit_5": False, "hit_20": False, "ndcg_10": 0.0,
         "reranker_applied": False, "reranker_latency_ms": 50},
    ]
    a = aggregate(rows)
    assert a["n"] == 3
    assert math.isclose(a["mrr"], 0.5, rel_tol=1e-6)
    assert math.isclose(a["hr1"], 1/3, rel_tol=1e-6)
    assert math.isclose(a["hr5"], 2/3, rel_tol=1e-6)
    assert math.isclose(a["applied_rate"], 2/3, rel_tol=1e-6)


# ---- pairwise_wins ----

def test_pairwise_wins_a_dominates():
    a_rows = [{"query": "q1", "rr": 1.0}, {"query": "q2", "rr": 0.5}]
    b_rows = [{"query": "q1", "rr": 0.5}, {"query": "q2", "rr": 0.25}]
    pw = pairwise_wins(a_rows, b_rows)
    assert pw["a_wins"] == 2
    assert pw["b_wins"] == 0
    assert pw["ties"] == 0
    assert pw["total"] == 2
    assert pw["win_rate_a"] == 1.0


def test_pairwise_wins_split():
    a_rows = [{"query": "q1", "rr": 1.0}, {"query": "q2", "rr": 0.5}, {"query": "q3", "rr": 0.0}]
    b_rows = [{"query": "q1", "rr": 0.5}, {"query": "q2", "rr": 1.0}, {"query": "q3", "rr": 0.0}]
    pw = pairwise_wins(a_rows, b_rows)
    assert pw["a_wins"] == 1
    assert pw["b_wins"] == 1
    assert pw["ties"] == 1
    assert pw["total"] == 3


def test_pairwise_wins_unaligned_queries_skipped():
    a_rows = [{"query": "q1", "rr": 1.0}]
    b_rows = [{"query": "q2", "rr": 0.5}]
    pw = pairwise_wins(a_rows, b_rows)
    assert pw["total"] == 0


# ---- stratify ----

def test_stratify_groups_by_key():
    rows = [
        {"subproject": "nix", "rr": 1.0, "hit_1": True, "hit_5": True, "hit_20": True, "ndcg_10": 1.0,
         "reranker_applied": True, "reranker_latency_ms": 100},
        {"subproject": "nix", "rr": 0.5, "hit_1": False, "hit_5": True, "hit_20": True, "ndcg_10": 0.6,
         "reranker_applied": True, "reranker_latency_ms": 100},
        {"subproject": "assetman", "rr": 0.0, "hit_1": False, "hit_5": False, "hit_20": False, "ndcg_10": 0.0,
         "reranker_applied": True, "reranker_latency_ms": 100},
    ]
    s = stratify(rows, "subproject")
    assert set(s.keys()) == {"nix", "assetman"}
    assert s["nix"]["n"] == 2
    assert math.isclose(s["nix"]["mrr"], 0.75, rel_tol=1e-6)
    assert s["assetman"]["n"] == 1


def test_stratify_missing_key_groups_as_none_label():
    rows = [{"subproject": None, "rr": 1.0, "hit_1": True, "hit_5": True, "hit_20": True,
             "ndcg_10": 1.0, "reranker_applied": True, "reranker_latency_ms": 0}]
    s = stratify(rows, "subproject")
    # None -> "None" str key
    assert "None" in s or "(none)" in s


# ---- top_failures ----

def test_top_failures_returns_largest_disagreement():
    arms = {
        "hybrid":    [{"query": "q1", "rr": 1.0, "rank": 1,
                       "ranked_files": ["a"], "expected_files": ["a"], "subproject": None,
                       "category": None, "label_confidence": None},
                      {"query": "q2", "rr": 0.5, "rank": 2,
                       "ranked_files": ["b", "a"], "expected_files": ["a"], "subproject": None,
                       "category": None, "label_confidence": None}],
        "pointwise": [{"query": "q1", "rr": 1.0, "rank": 1,
                       "ranked_files": ["a"], "expected_files": ["a"], "subproject": None,
                       "category": None, "label_confidence": None},
                      {"query": "q2", "rr": 0.0, "rank": None,
                       "ranked_files": ["c", "d"], "expected_files": ["a"], "subproject": None,
                       "category": None, "label_confidence": None}],
    }
    failures = top_failures(arms, top_n=10)
    # q2 has rr-delta = 0.5, q1 has 0 -> only q2 returned
    assert len(failures) == 1
    assert failures[0]["query"] == "q2"
    assert math.isclose(failures[0]["disagreement"], 0.5, rel_tol=1e-6)


def test_top_failures_zero_disagreement_excluded():
    """If all arms agree, no failures returned."""
    arms = {
        "hybrid":   [{"query": "q1", "rr": 1.0, "rank": 1, "ranked_files": ["a"],
                      "expected_files": ["a"], "subproject": None,
                      "category": None, "label_confidence": None}],
        "listwise": [{"query": "q1", "rr": 1.0, "rank": 1, "ranked_files": ["a"],
                      "expected_files": ["a"], "subproject": None,
                      "category": None, "label_confidence": None}],
    }
    failures = top_failures(arms, top_n=10)
    assert failures == []
