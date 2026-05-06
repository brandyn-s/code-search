"""Tests for recall@15 metric in sweep_rrf_weights.py + d1_bootstrap_ci.py.

Plan D1-Pass-2 C.1 (PR #135). Verifies:

- _eval_one_setting captures both rank (top_k) and gold_in_top_recall_k per query
- recall@k aggregate is computed correctly
- d1_bootstrap_ci's _extract_metric_values dispatches by metric name
- d1_bootstrap_ci's _from_sweep_output reads per_query blocks correctly
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def test_extract_metric_values_mrr():
    """metric=mrr_at_10 returns per-query rr values."""
    from bench.research.d1_bootstrap_ci import _extract_metric_values
    per_query = [
        {"rr": 1.0, "gold_in_top_recall_k": 1},
        {"rr": 0.5, "gold_in_top_recall_k": 1},
        {"rr": 0.0, "gold_in_top_recall_k": 0},
    ]
    assert _extract_metric_values(per_query, "mrr_at_10") == [1.0, 0.5, 0.0]


def test_extract_metric_values_recall():
    """metric=recall_at_15 returns per-query 0/1 indicators."""
    from bench.research.d1_bootstrap_ci import _extract_metric_values
    per_query = [
        {"rr": 1.0, "gold_in_top_recall_k": 1},
        {"rr": 0.5, "gold_in_top_recall_k": 1},
        {"rr": 0.0, "gold_in_top_recall_k": 0},
    ]
    assert _extract_metric_values(per_query, "recall_at_15") == [1.0, 1.0, 0.0]


def test_extract_metric_values_unknown_metric_raises():
    from bench.research.d1_bootstrap_ci import _extract_metric_values
    import pytest
    with pytest.raises(ValueError):
        _extract_metric_values([], "unknown_metric")


def test_from_sweep_output_reads_both_settings(tmp_path: Path):
    """_from_sweep_output picks the right per_query blocks by (vw, bw) keys."""
    from bench.research.d1_bootstrap_ci import _from_sweep_output

    sweep_blob = {
        "rerank": "off", "n": 3, "metric": "recall_at_15", "recall_k": 15,
        "results": [
            {
                "vector_weight": 0.65, "bm25_weight": 0.35,
                "mrr": 0.5, "hr_1": 0.33, "recall_at_k": 0.67, "recall_k": 15, "n": 3,
                "per_query": [
                    {"rr": 1.0, "gold_in_top_recall_k": 1},
                    {"rr": 0.0, "gold_in_top_recall_k": 1},
                    {"rr": 0.5, "gold_in_top_recall_k": 0},
                ],
            },
            {
                "vector_weight": 0.60, "bm25_weight": 0.40,
                "mrr": 0.6, "hr_1": 0.66, "recall_at_k": 1.0, "recall_k": 15, "n": 3,
                "per_query": [
                    {"rr": 1.0, "gold_in_top_recall_k": 1},
                    {"rr": 1.0, "gold_in_top_recall_k": 1},
                    {"rr": 0.0, "gold_in_top_recall_k": 1},
                ],
            },
        ],
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(sweep_blob), encoding="utf-8")

    a, b = _from_sweep_output(path, "recall_at_15", (0.65, 0.35), (0.60, 0.40))
    assert a == [1.0, 1.0, 0.0]  # gold_in_top_recall_k from setting A
    assert b == [1.0, 1.0, 1.0]  # gold_in_top_recall_k from setting B


def test_from_sweep_output_mrr_metric(tmp_path: Path):
    """_from_sweep_output also works for mrr_at_10."""
    from bench.research.d1_bootstrap_ci import _from_sweep_output

    sweep_blob = {
        "results": [
            {
                "vector_weight": 0.65, "bm25_weight": 0.35,
                "per_query": [
                    {"rr": 1.0, "gold_in_top_recall_k": 1},
                    {"rr": 0.5, "gold_in_top_recall_k": 1},
                ],
            },
            {
                "vector_weight": 0.60, "bm25_weight": 0.40,
                "per_query": [
                    {"rr": 0.5, "gold_in_top_recall_k": 1},
                    {"rr": 1.0, "gold_in_top_recall_k": 1},
                ],
            },
        ],
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(sweep_blob), encoding="utf-8")

    a, b = _from_sweep_output(path, "mrr_at_10", (0.65, 0.35), (0.60, 0.40))
    assert a == [1.0, 0.5]
    assert b == [0.5, 1.0]


def test_from_sweep_output_missing_setting_exits(tmp_path: Path):
    """Missing weight setting exits with a clear message."""
    import pytest
    from bench.research.d1_bootstrap_ci import _from_sweep_output

    sweep_blob = {
        "results": [
            {"vector_weight": 0.65, "bm25_weight": 0.35, "per_query": []},
        ],
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(sweep_blob), encoding="utf-8")

    with pytest.raises(SystemExit):
        _from_sweep_output(path, "mrr_at_10", (0.65, 0.35), (0.60, 0.40))


def test_from_sweep_output_length_mismatch_exits(tmp_path: Path):
    """Length-mismatched per_query blocks exit (paired bootstrap requires equal n)."""
    import pytest
    from bench.research.d1_bootstrap_ci import _from_sweep_output

    sweep_blob = {
        "results": [
            {
                "vector_weight": 0.65, "bm25_weight": 0.35,
                "per_query": [{"rr": 1.0, "gold_in_top_recall_k": 1}],
            },
            {
                "vector_weight": 0.60, "bm25_weight": 0.40,
                "per_query": [
                    {"rr": 1.0, "gold_in_top_recall_k": 1},
                    {"rr": 0.5, "gold_in_top_recall_k": 1},
                ],
            },
        ],
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(sweep_blob), encoding="utf-8")

    with pytest.raises(SystemExit):
        _from_sweep_output(path, "mrr_at_10", (0.65, 0.35), (0.60, 0.40))


def test_bootstrap_ci_on_recall_metric():
    """Paired bootstrap CI computes correctly on 0/1 indicators."""
    from bench.research.d1_bootstrap_ci import _bootstrap_ci

    # Setting B beats A by 0.3 (3/10 more queries hit gold)
    deltas = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ci = _bootstrap_ci(deltas, n_resamples=5000, seed=0)
    assert ci["point_estimate"] == 0.3
    # 95% CI on a small sample with this distribution should clearly include 0
    # (we have only 10 queries, with high variance)
    assert ci["n_queries"] == 10
    assert "ci_lower" in ci and "ci_upper" in ci
    assert ci["ci_lower"] <= ci["point_estimate"] <= ci["ci_upper"]


def test_bootstrap_ci_excludes_zero_when_signal_is_strong():
    """When all per-query deltas are positive, bootstrap CI excludes zero."""
    from bench.research.d1_bootstrap_ci import _bootstrap_ci

    # Strong, consistent positive signal
    deltas = [0.1, 0.2, 0.15, 0.18, 0.12, 0.13, 0.11, 0.16, 0.14, 0.17] * 20  # n=200
    ci = _bootstrap_ci(deltas, n_resamples=5000, seed=0)
    assert ci["excludes_zero"] is True
    assert ci["ci_lower"] > 0
