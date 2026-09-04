"""Ranking metrics and paired bootstrap for the public evaluation set.

Pure functions, no I/O, so they are unit-testable with synthetic inputs.
A *result row* is ``{"query": str, "expected_files": [..], "ranked_files": [..]}``.
"""

from __future__ import annotations

import random
from statistics import mean
from typing import Iterable, Sequence


def reciprocal_rank(ranked: Sequence[str], expected: Iterable[str]) -> float:
    """1/rank of the first expected file in ``ranked``; 0.0 when absent."""
    wanted = set(expected)
    for position, path in enumerate(ranked, start=1):
        if path in wanted:
            return 1.0 / position
    return 0.0


def hit_at(ranked: Sequence[str], expected: Iterable[str], k: int) -> float:
    """1.0 when any expected file appears in the top ``k``."""
    wanted = set(expected)
    return 1.0 if any(path in wanted for path in ranked[:k]) else 0.0


def recall_at(ranked: Sequence[str], expected: Iterable[str], k: int) -> float:
    """Fraction of expected files that appear in the top ``k``."""
    wanted = set(expected)
    if not wanted:
        return 0.0
    found = sum(1 for path in wanted if path in set(ranked[:k]))
    return found / len(wanted)


def per_query_metrics(row: dict) -> dict:
    ranked = [p.replace("\\", "/") for p in row["ranked_files"]]
    expected = [p.replace("\\", "/") for p in row["expected_files"]]
    return {
        "rr": reciprocal_rank(ranked, expected),
        "hr1": hit_at(ranked, expected, 1),
        "recall10": recall_at(ranked, expected, 10),
    }


def summarize(rows: Sequence[dict]) -> dict:
    """Aggregate MRR, HR@1 and Recall@10 over result rows."""
    if not rows:
        return {"n": 0, "mrr": 0.0, "hr1": 0.0, "recall10": 0.0}
    per = [per_query_metrics(row) for row in rows]
    return {
        "n": len(per),
        "mrr": mean(m["rr"] for m in per),
        "hr1": mean(m["hr1"] for m in per),
        "recall10": mean(m["recall10"] for m in per),
    }


def paired_deltas(baseline: Sequence[dict], treatment: Sequence[dict], metric: str = "rr") -> list[float]:
    """Per-query treatment-minus-baseline deltas, paired on the query string.

    Queries present in only one side are ignored; the caller decides whether
    that is acceptable (``compare.py`` reports the count).
    """
    base = {row["query"]: per_query_metrics(row)[metric] for row in baseline}
    treat = {row["query"]: per_query_metrics(row)[metric] for row in treatment}
    common = sorted(set(base) & set(treat))
    return [treat[q] - base[q] for q in common]


def bootstrap_ci(deltas: Sequence[float], *, n_resamples: int = 10000, ci: float = 0.95, seed: int = 42) -> dict:
    """Percentile bootstrap CI on the mean of paired per-query deltas."""
    if not deltas:
        return {"point": 0.0, "lower": 0.0, "upper": 0.0, "excludes_zero": False, "n": 0}
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_resamples):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1 - ci) / 2
    lower = means[int(alpha * n_resamples)]
    upper = means[min(int((1 - alpha) * n_resamples), n_resamples - 1)]
    return {
        "point": mean(deltas),
        "lower": lower,
        "upper": upper,
        "excludes_zero": lower > 0 or upper < 0,
        "n": n,
        "ci": ci,
        "n_resamples": n_resamples,
    }
