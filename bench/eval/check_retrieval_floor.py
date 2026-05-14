"""Retrieval floor gate — defensive check for catastrophic regression.

Two modes:

  --mode summary       Read an existing eval summary.json and assert
                       MRR / HR@1 floors. Cheap, no API calls. Use after
                       running eval_against_psm_full.py locally.

  --mode index-and-eval  Index a target project from scratch (Voyage API)
                       and run a small gold-query set against it. Used in
                       CI for catastrophic-regression detection on a
                       small self-contained fixture.

Floors are deliberately conservative (~2-3pp below current measurement)
so the gate fires only on real regression, not bootstrap noise.

Examples:

  # Local: assert latest PSM eval summary clears floor
  python bench/eval/check_retrieval_floor.py \\
      --mode summary \\
      --summary benchmarks/eval_v4/run_psm-full-voyage-multitarget/summary.json \\
      --floor-golden-mrr 0.62 \\
      --floor-harvested-mrr 0.73

  # CI: index small fixture + eval against hand-authored gold
  python bench/eval/check_retrieval_floor.py \\
      --mode index-and-eval \\
      --project . \\
      --gold bench/eval/golden_code_search_self.json \\
      --floor-mrr 0.5 \\
      --floor-hr1 0.40
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from statistics import median

_orig_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = (
    lambda host, port, family=0, type=0, proto=0, flags=0:
    _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def normalize_path(p: str) -> str:
    return p.replace("\\", "/")


def matches_expected(result_path: str, expected_set: set[str]) -> bool:
    """Match result against expected files; suffix match or exact match."""
    rp = normalize_path(result_path)
    for exp in expected_set:
        e = normalize_path(exp)
        if rp == e or rp.endswith("/" + e):
            return True
    return False


def gold_rank(top_files: list[str], expected: set[str]) -> int | None:
    for i, f in enumerate(top_files, 1):
        if matches_expected(f, expected):
            return i
    return None


def check_summary_mode(args: argparse.Namespace) -> int:
    """Read an existing summary.json and assert floors."""
    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"[retrieval-floor-gate] FAIL: summary file not found: {summary_path}",
              file=sys.stderr)
        return 1

    data = json.loads(summary_path.read_text(encoding="utf-8"))

    golden_mrr = data.get("golden", {}).get("mrr")
    golden_hr1 = data.get("golden", {}).get("hr_1")
    harvested_mrr = data.get("harvested_labeled", {}).get("mrr")
    harvested_hr1 = data.get("harvested_labeled", {}).get("hr_1")

    if golden_mrr is None or harvested_mrr is None:
        print(f"[retrieval-floor-gate] FAIL: summary missing required fields "
              f"(golden.mrr={golden_mrr}, harvested_labeled.mrr={harvested_mrr})",
              file=sys.stderr)
        return 1

    failures = []
    checks = [
        ("golden MRR", golden_mrr, args.floor_golden_mrr),
        ("golden HR@1", golden_hr1, args.floor_golden_hr1),
        ("harvested MRR", harvested_mrr, args.floor_harvested_mrr),
        ("harvested HR@1", harvested_hr1, args.floor_harvested_hr1),
    ]

    print(f"[retrieval-floor-gate] mode=summary  source={summary_path}")
    for label, value, floor in checks:
        if floor is None:
            continue
        if value is None:
            failures.append(f"{label} not present in summary")
            continue
        status = "PASS" if value >= floor else "FAIL"
        marker = "" if status == "PASS" else "  <-- BELOW FLOOR"
        print(f"  {status}: {label:<16}  measured={value:.4f}  floor={floor:.4f}{marker}")
        if status == "FAIL":
            failures.append(f"{label} {value:.4f} < floor {floor:.4f}")

    if failures:
        print(f"[retrieval-floor-gate] FAIL: {len(failures)} floor violation(s)",
              file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("[retrieval-floor-gate] PASS: all floors cleared")
    return 0


def setup_server_for_project(project_path: str, provider: str = "voyage",
                              rerank: str = "off"):
    """Generic project switch — no hardcoded PSM paths."""
    os.environ["EMBEDDING_PROVIDER"] = provider
    if provider == "voyage":
        os.environ["EMBEDDING_MODEL"] = "voyage-4-large"
    os.environ["VOYAGE_INPUT_TYPE"] = "on"
    os.environ["RERANKER"] = rerank
    os.environ["QUERY_EXPANSION"] = "on"

    from common_utils import get_storage_dir
    get_storage_dir.cache_clear()
    from mcp_server.code_search_server import CodeSearchServer

    server = CodeSearchServer()
    raw = server.switch_project(project_path=project_path, provider=provider)
    parsed = json.loads(raw)
    if "error" in parsed:
        print(f"[retrieval-floor-gate] FAIL: switch_project failed: {parsed['error']}",
              file=sys.stderr)
        return None
    return server


def index_project(server, project_path: str) -> bool:
    """Index project freshly (CI mode). Returns True on success."""
    raw = server.index_directory(directory_path=project_path, incremental=False)
    parsed = json.loads(raw)
    if "error" in parsed:
        print(f"[retrieval-floor-gate] FAIL: index_directory failed: {parsed['error']}",
              file=sys.stderr)
        return False

    job_id = parsed.get("job_id")
    if not job_id:
        # Synchronous result (older API)
        return True

    # Poll for completion
    deadline = time.time() + 600  # 10 min
    while time.time() < deadline:
        raw_p = server.get_indexing_progress()
        prog = json.loads(raw_p)
        status = prog.get("status")
        if status in ("completed", "idle"):
            return True
        if status == "failed":
            print(f"[retrieval-floor-gate] FAIL: indexing failed: {prog}",
                  file=sys.stderr)
            return False
        pct = prog.get("percent", 0)
        phase = prog.get("phase", "?")
        print(f"  indexing: {phase} {pct}%")
        time.sleep(5)

    print("[retrieval-floor-gate] FAIL: indexing timeout after 10 min",
          file=sys.stderr)
    return False


def eval_gold(server, gold_path: Path) -> dict:
    """Run gold queries and compute MRR + HR@1 + HR@5."""
    queries = json.loads(gold_path.read_text(encoding="utf-8"))
    print(f"  loaded {len(queries)} gold queries from {gold_path}")

    rows = []
    for i, q in enumerate(queries, 1):
        expected = set(q.get("expected_files") or [])
        if not expected:
            rows.append({**q, "skipped": True})
            continue
        t0 = time.time()
        raw = server.search_code(query=q["query"], k=10, auto_reindex=False)
        parsed = json.loads(raw)
        results = parsed.get("results", [])
        top = [r.get("file", r.get("relative_path", "")).replace("\\", "/")
               for r in results]
        rank = gold_rank(top, expected)
        rows.append({
            "query": q["query"],
            "expected_files": list(expected),
            "top_files": top[:5],
            "rank": rank,
            "rr": (1.0 / rank) if rank else 0.0,
            "hit_1": rank == 1,
            "hit_5": rank is not None and rank <= 5,
            "latency_ms": (time.time() - t0) * 1000,
        })
        if i % 5 == 0 or i == len(queries):
            print(f"  [{i:>3}/{len(queries)}] done")

    scored = [r for r in rows if not r.get("skipped")]
    n = len(scored)
    if n == 0:
        return {"n": 0, "rows": rows}
    return {
        "n": n,
        "hr_1": sum(1 for r in scored if r["hit_1"]) / n,
        "hr_5": sum(1 for r in scored if r["hit_5"]) / n,
        "mrr": sum(r["rr"] for r in scored) / n,
        "median_latency_ms": median([r["latency_ms"] for r in scored]),
        "rows": rows,
    }


def check_index_and_eval_mode(args: argparse.Namespace) -> int:
    """Index project fresh + run eval + assert floors."""
    if not os.environ.get("VOYAGE_API_KEY"):
        print("[retrieval-floor-gate] FAIL: VOYAGE_API_KEY not set", file=sys.stderr)
        return 1

    project_path = str(Path(args.project).resolve())
    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"[retrieval-floor-gate] FAIL: gold file not found: {gold_path}",
              file=sys.stderr)
        return 1

    print(f"[retrieval-floor-gate] mode=index-and-eval")
    print(f"  project: {project_path}")
    print(f"  gold:    {gold_path}")
    print(f"  rerank:  {args.rerank}")

    server = setup_server_for_project(project_path, provider=args.provider,
                                       rerank=args.rerank)
    if server is None:
        return 1

    print("  indexing target project...")
    if not index_project(server, project_path):
        return 1

    print("  running gold eval...")
    summary = eval_gold(server, gold_path)
    n = summary.get("n", 0)
    if n == 0:
        print("[retrieval-floor-gate] FAIL: no queries evaluated", file=sys.stderr)
        return 1

    print(f"  measured: n={n}  MRR={summary['mrr']:.4f}  "
          f"HR@1={summary['hr_1']:.4f}  HR@5={summary['hr_5']:.4f}  "
          f"median_latency_ms={summary['median_latency_ms']:.0f}")

    failures = []
    if args.floor_mrr is not None and summary["mrr"] < args.floor_mrr:
        failures.append(f"MRR {summary['mrr']:.4f} < floor {args.floor_mrr:.4f}")
    if args.floor_hr1 is not None and summary["hr_1"] < args.floor_hr1:
        failures.append(f"HR@1 {summary['hr_1']:.4f} < floor {args.floor_hr1:.4f}")

    if failures:
        print("[retrieval-floor-gate] FAIL: floor violation(s):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        if args.dump_rows:
            print("  per-query rows:", file=sys.stderr)
            for r in summary["rows"][:20]:
                print(f"    {r}", file=sys.stderr)
        return 1

    print(f"[retrieval-floor-gate] PASS: MRR {summary['mrr']:.4f} >= "
          f"{args.floor_mrr:.4f}, HR@1 {summary['hr_1']:.4f} >= "
          f"{args.floor_hr1:.4f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrieval floor gate — assert MRR/HR@1 above floor.",
    )
    parser.add_argument("--mode", choices=["summary", "index-and-eval"],
                        required=True)

    # summary mode
    parser.add_argument("--summary", help="Path to eval summary.json")
    parser.add_argument("--floor-golden-mrr", type=float, default=None)
    parser.add_argument("--floor-golden-hr1", type=float, default=None)
    parser.add_argument("--floor-harvested-mrr", type=float, default=None)
    parser.add_argument("--floor-harvested-hr1", type=float, default=None)

    # index-and-eval mode
    parser.add_argument("--project", help="Path to target project to index")
    parser.add_argument("--gold", help="Path to gold queries JSON")
    parser.add_argument("--floor-mrr", type=float, default=None)
    parser.add_argument("--floor-hr1", type=float, default=None)
    parser.add_argument("--provider", default="voyage",
                        choices=["voyage", "voyage-context"])
    parser.add_argument("--rerank", default="off",
                        choices=["off", "sonnet", "cross-encoder"])
    parser.add_argument("--dump-rows", action="store_true",
                        help="On FAIL, print first 20 per-query rows to stderr")

    args = parser.parse_args()

    if args.mode == "summary":
        if not args.summary:
            parser.error("--summary is required in summary mode")
        return check_summary_mode(args)

    if args.mode == "index-and-eval":
        if not args.project or not args.gold:
            parser.error("--project and --gold are required in index-and-eval mode")
        return check_index_and_eval_mode(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
