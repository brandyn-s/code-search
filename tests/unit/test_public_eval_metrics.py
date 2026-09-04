"""Metric math and compare CLI for the public evaluation set (synthetic inputs)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC = REPO_ROOT / "bench" / "eval" / "public"
sys.path.insert(0, str(PUBLIC))

import metrics  # noqa: E402


def _row(query, expected, ranked):
    return {"query": query, "expected_files": expected, "ranked_files": ranked}


def test_reciprocal_rank_hit_and_recall():
    ranked = ["a.py", "b.py", "c.py"]
    assert metrics.reciprocal_rank(ranked, ["b.py"]) == 0.5
    assert metrics.reciprocal_rank(ranked, ["zzz.py"]) == 0.0
    assert metrics.hit_at(ranked, ["a.py"], 1) == 1.0
    assert metrics.hit_at(ranked, ["c.py"], 1) == 0.0
    assert metrics.recall_at(ranked, ["a.py", "c.py", "missing.py"], 10) == pytest.approx(2 / 3)
    assert metrics.recall_at(ranked, [], 10) == 0.0


def test_summarize_normalizes_windows_separators():
    rows = [_row("q1", ["src/app.py"], ["src\\app.py", "x.py"]), _row("q2", ["y.py"], ["a.py", "b.py", "y.py"])]
    s = metrics.summarize(rows)
    assert s["n"] == 2
    assert s["mrr"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert s["hr1"] == 0.5
    assert s["recall10"] == 1.0


def test_paired_deltas_pair_on_query_and_ignore_unmatched():
    base = [_row("q1", ["a"], ["b", "a"]), _row("q2", ["a"], ["a"]), _row("only-base", ["a"], ["a"])]
    treat = [_row("q1", ["a"], ["a"]), _row("q2", ["a"], ["b", "b", "a"])]
    deltas = metrics.paired_deltas(base, treat)
    assert sorted(deltas) == pytest.approx(sorted([1.0 - 0.5, 1 / 3 - 1.0]))


def test_bootstrap_ci_is_deterministic_and_detects_clear_wins():
    deltas = [0.1] * 20 + [0.05] * 10
    a = metrics.bootstrap_ci(deltas, n_resamples=2000)
    b = metrics.bootstrap_ci(deltas, n_resamples=2000)
    assert a == b
    assert a["excludes_zero"] is True and a["lower"] > 0
    mixed = [0.5, -0.5] * 10
    assert metrics.bootstrap_ci(mixed, n_resamples=2000)["excludes_zero"] is False
    assert metrics.bootstrap_ci([])["n"] == 0


def test_compare_cli_reports_deltas_and_exit_codes(tmp_path):
    base_rows = [_row(f"q{i}", ["hit.py"], ["miss.py", "hit.py"]) for i in range(25)]
    better_rows = [_row(f"q{i}", ["hit.py"], ["hit.py"]) for i in range(25)]
    worse_rows = [_row(f"q{i}", ["hit.py"], ["a.py", "b.py", "c.py", "hit.py"]) for i in range(25)]
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"rows": base_rows}))
    better = tmp_path / "better.json"
    better.write_text(json.dumps({"rows": better_rows}))
    worse = tmp_path / "worse.json"
    worse.write_text(json.dumps(worse_rows))  # bare list form is accepted too

    up = subprocess.run([sys.executable, str(PUBLIC / "compare.py"), str(base), str(better)], capture_output=True, text=True)
    assert up.returncode == 0, up.stderr
    assert "MRR" in up.stdout and "excludes zero" in up.stdout and "+0.5000" in up.stdout

    down = subprocess.run([sys.executable, str(PUBLIC / "compare.py"), str(base), str(worse)], capture_output=True, text=True)
    assert down.returncode == 2, down.stdout
    assert "excludes zero" in down.stdout


def test_gold_sets_are_well_formed_and_public():
    for name in ("golden_flask.json", "golden_requests.json"):
        gold = json.loads((PUBLIC / name).read_text(encoding="utf-8"))
        assert len(gold) == 30
        for entry in gold:
            assert entry["query"].strip()
            assert entry["expected_files"], entry
            for path in entry["expected_files"]:
                assert not path.startswith(("/", "C:")), path
