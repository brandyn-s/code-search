"""Multi-turn retrieval eval harness — Phase C of the ABC future-arcs roadmap.

Reads a JSON file of conversation bundles, runs each turn's query against an
already-indexed project, computes per-bundle cumulative-recall metrics, and
writes per-query rows for downstream paired-bootstrap analysis.

Bundle schema (per `DESIGN.md`):

  [
    {
      "id": "...",
      "category": "...",
      "fixture": "flask",
      "turns": [
        {"turn": 1, "query": "...", "expected_files": ["..."]},
        {"turn": 2, "query": "...", "expected_files": ["..."]},
        ...
      ]
    }, ...
  ]

Metrics reported (aggregated across bundles):
  - recall_at_5_in_3_turns      — gold in top-5 of ANY turn 1..3
  - recall_at_10_in_5_turns     — gold in top-10 of ANY turn 1..5
  - first_turn_to_gold          — turn number gold first appears in top-10
  - per-turn HR@1 / HR@5        — single-turn-style for reference

Usage:
  python bench/eval/multi_turn/run_multi_turn.py \\
      --bundles bench/eval/multi_turn/bundles_flask.json \\
      --project-path "C:~/Documents/bench-fixtures/flask" \\
      --out /tmp/multi-turn-flask.json \\
      --rerank sonnet
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

_orig_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = (
    lambda host, port, family=0, type=0, proto=0, flags=0:
    _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass


def _matches(result_path: str, expected_set: set[str]) -> bool:
    rp = result_path.replace("\\", "/")
    for exp in expected_set:
        e = exp.replace("\\", "/")
        if rp == e or rp.endswith("/" + e):
            return True
    return False


def setup_server(project_path: str, provider: str, rerank: str):
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
    raw = server.switch_project(project_path=str(Path(project_path).resolve()),
                                  provider=provider)
    parsed = json.loads(raw)
    if "error" in parsed:
        sys.exit(f"switch_project failed: {parsed['error']}")
    return server


def run_turn(server, query: str, k: int = 10) -> tuple[list[str], float]:
    t0 = time.time()
    raw = server.search_code(query=query, k=k, auto_reindex=False)
    parsed = json.loads(raw)
    results = parsed.get("results", [])
    files = [r.get("file", r.get("relative_path", "")).replace("\\", "/")
             for r in results]
    latency_ms = (time.time() - t0) * 1000
    return files, latency_ms


def run_bundle(server, bundle: dict) -> dict:
    """Run all turns of a bundle; return per-turn rows + cumulative metrics."""
    turns = bundle.get("turns", [])
    turn_rows: list[dict] = []
    first_turn_to_gold: int | None = None
    gold_in_topk_per_turn: dict[int, dict[str, bool]] = {}

    for turn in turns:
        tno = int(turn["turn"])
        query = turn["query"]
        expected = set(turn.get("expected_files") or [])
        files, latency = run_turn(server, query, k=10)
        rank = None
        for i, f in enumerate(files, 1):
            if _matches(f, expected):
                rank = i
                break

        gold_in_topk_per_turn[tno] = {
            "top_1": rank == 1,
            "top_5": rank is not None and rank <= 5,
            "top_10": rank is not None and rank <= 10,
        }

        if rank is not None and rank <= 10 and first_turn_to_gold is None:
            first_turn_to_gold = tno

        turn_rows.append({
            "turn": tno,
            "query": query,
            "expected_files": list(expected),
            "top_files": files[:5],
            "rank": rank,
            "hit_1": rank == 1,
            "hit_5": rank is not None and rank <= 5,
            "hit_10": rank is not None and rank <= 10,
            "latency_ms": latency,
        })

    cumulative = {
        "recall_at_5_in_3_turns": any(
            gold_in_topk_per_turn.get(t, {}).get("top_5", False)
            for t in (1, 2, 3)
        ),
        "recall_at_10_in_5_turns": any(
            gold_in_topk_per_turn.get(t, {}).get("top_10", False)
            for t in (1, 2, 3, 4, 5)
        ),
        "first_turn_to_gold": first_turn_to_gold,
    }

    return {
        "id": bundle.get("id"),
        "category": bundle.get("category"),
        "fixture": bundle.get("fixture"),
        "turns": turn_rows,
        "cumulative": cumulative,
    }


def aggregate(bundles_out: list[dict]) -> dict:
    n_bundles = len(bundles_out)
    if n_bundles == 0:
        return {"n_bundles": 0}

    recall_5_3 = sum(1 for b in bundles_out
                     if b["cumulative"]["recall_at_5_in_3_turns"]) / n_bundles
    recall_10_5 = sum(1 for b in bundles_out
                      if b["cumulative"]["recall_at_10_in_5_turns"]) / n_bundles

    ftg = [b["cumulative"]["first_turn_to_gold"] for b in bundles_out
           if b["cumulative"]["first_turn_to_gold"] is not None]
    median_ftg = sorted(ftg)[len(ftg) // 2] if ftg else None
    gold_found_count = len(ftg)

    # Per-turn HR@1 / HR@5 averaged over bundles
    by_turn: dict[int, list[dict]] = {}
    for b in bundles_out:
        for t in b["turns"]:
            by_turn.setdefault(t["turn"], []).append(t)
    per_turn = {}
    for tno, rows in sorted(by_turn.items()):
        n = len(rows)
        per_turn[tno] = {
            "n": n,
            "hr_1": sum(1 for r in rows if r["hit_1"]) / n,
            "hr_5": sum(1 for r in rows if r["hit_5"]) / n,
            "hr_10": sum(1 for r in rows if r["hit_10"]) / n,
        }

    return {
        "n_bundles": n_bundles,
        "recall_at_5_in_3_turns": recall_5_3,
        "recall_at_10_in_5_turns": recall_10_5,
        "median_first_turn_to_gold": median_ftg,
        "bundles_finding_gold_within_budget": gold_found_count,
        "per_turn": per_turn,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bundles", required=True,
                   help="Path to conversation-bundles JSON")
    p.add_argument("--project-path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--provider", default="voyage",
                   choices=["voyage", "voyage-context"])
    p.add_argument("--rerank", default="sonnet",
                   choices=["off", "sonnet", "cross-encoder"])
    args = p.parse_args()

    if not os.environ.get("VOYAGE_API_KEY"):
        sys.exit("VOYAGE_API_KEY not set")
    if args.rerank == "sonnet" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY required for rerank=sonnet")

    bundles_path = Path(args.bundles)
    bundles = json.loads(bundles_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(bundles)} conversation bundles from {bundles_path}",
          flush=True)

    server = setup_server(args.project_path, args.provider, args.rerank)
    print(f"Server ready; project={args.project_path} rerank={args.rerank}",
          flush=True)

    bundles_out = []
    for i, b in enumerate(bundles, 1):
        result = run_bundle(server, b)
        bundles_out.append(result)
        print(f"  [{i:>3}/{len(bundles)}] bundle={result['id']} "
              f"first_turn_to_gold={result['cumulative']['first_turn_to_gold']}",
              flush=True)

    agg = aggregate(bundles_out)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "aggregate": agg,
        "bundles": bundles_out,
    }, indent=2), encoding="utf-8")

    print("\n=== Aggregate ===", flush=True)
    print(f"  n_bundles: {agg['n_bundles']}", flush=True)
    print(f"  recall@5 in 3 turns:  {agg['recall_at_5_in_3_turns']:.3f}", flush=True)
    print(f"  recall@10 in 5 turns: {agg['recall_at_10_in_5_turns']:.3f}", flush=True)
    print(f"  median first_turn_to_gold: {agg['median_first_turn_to_gold']}",
          flush=True)
    print(f"  bundles finding gold: {agg['bundles_finding_gold_within_budget']}",
          flush=True)
    print("  per-turn (HR@1 / HR@5 / HR@10):", flush=True)
    for tno, m in agg["per_turn"].items():
        print(f"    turn {tno}: n={m['n']:>3}  HR@1={m['hr_1']:.3f}  "
              f"HR@5={m['hr_5']:.3f}  HR@10={m['hr_10']:.3f}", flush=True)
    print(f"\nWritten to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
