#!/usr/bin/env python3
"""Compare two ``run_public_eval.py`` result files with a paired bootstrap CI.

Usage:
    python bench/eval/public/compare.py baseline.json treatment.json [--metric rr|hr1|recall10]

Exit code is 0 when the treatment is not significantly worse, 2 when the
95% CI on the mean per-query delta lies entirely below zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import bootstrap_ci, paired_deltas, summarize  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise SystemExit(f"{path}: expected a list of result rows")
    return rows


def format_report(baseline: list[dict], treatment: list[dict], metric: str) -> tuple[str, bool]:
    base_summary = summarize(baseline)
    treat_summary = summarize(treatment)
    deltas = paired_deltas(baseline, treatment, metric=metric)
    ci = bootstrap_ci(deltas)
    unpaired = len({r["query"] for r in baseline} ^ {r["query"] for r in treatment})
    lines = [
        f"{'metric':10s} {'baseline':>10s} {'treatment':>10s} {'delta':>8s}",
        f"{'MRR':10s} {base_summary['mrr']:10.4f} {treat_summary['mrr']:10.4f} {treat_summary['mrr'] - base_summary['mrr']:+8.4f}",
        f"{'HR@1':10s} {base_summary['hr1']:10.4f} {treat_summary['hr1']:10.4f} {treat_summary['hr1'] - base_summary['hr1']:+8.4f}",
        f"{'Recall@10':10s} {base_summary['recall10']:10.4f} {treat_summary['recall10']:10.4f} {treat_summary['recall10'] - base_summary['recall10']:+8.4f}",
        "",
        f"paired bootstrap on per-query {metric} delta: n={ci['n']} mean={ci['point']:+.4f} "
        f"95% CI [{ci['lower']:+.4f}, {ci['upper']:+.4f}] "
        + ("(excludes zero)" if ci["excludes_zero"] else "(includes zero)"),
    ]
    if unpaired:
        lines.append(f"warning: {unpaired} queries appear in only one result file and were ignored")
    worse = ci["excludes_zero"] and ci["upper"] < 0
    return "\n".join(lines), worse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("treatment", type=Path)
    parser.add_argument("--metric", choices=("rr", "hr1", "recall10"), default="rr")
    args = parser.parse_args(argv)
    report, worse = format_report(load_rows(args.baseline), load_rows(args.treatment), args.metric)
    print(report)
    return 2 if worse else 0


if __name__ == "__main__":
    raise SystemExit(main())
