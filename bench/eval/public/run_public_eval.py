#!/usr/bin/env python3
"""Index a pinned public corpus into a temporary store and score a gold set.

Usage:
    python bench/eval/public/run_public_eval.py --corpus /path/to/flask --gold bench/eval/public/golden_flask.json \
        --output results/flask-baseline.json [--k 10] [--keep-storage DIR]

Defaults to the credential-free local embedding provider (requires the
``[local]`` extra). Set ``VOYAGE_API_KEY``/``EMBEDDING_PROVIDER`` to evaluate a
cloud provider instead; the provider and reranker in effect are recorded in
the output so two result files can be compared honestly.

The output is ``{"config": {...}, "summary": {...}, "rows": [...]}`` where each
row is ``{"query", "expected_files", "ranked_files", "category"}``. Feed two
outputs to ``compare.py`` for a paired bootstrap on the deltas.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import summarize  # noqa: E402


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _wait_for_index(server, project_path: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    last: dict = {}
    while time.monotonic() < deadline:
        last = json.loads(server.get_indexing_progress())
        if last.get("index_ready") or last.get("status") in {"completed", "failed", "cancelled", "idle"}:
            break
        time.sleep(1.0)
    status = json.loads(server.get_index_status(project_path=project_path))
    if not status.get("index_ready", False):
        raise SystemExit(f"index did not become ready: {json.dumps(last)[:400]}")
    return status


def run(corpus: Path, gold_path: Path, k: int, storage: Path, timeout_s: float) -> dict:
    os.environ["CODE_SEARCH_STORAGE"] = str(storage)
    os.environ.setdefault("RERANKER", "off")
    from common_utils import get_storage_dir

    get_storage_dir.cache_clear()
    from mcp_server.code_search_server import CodeSearchServer

    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    server = CodeSearchServer()
    started = json.loads(server.index_directory(str(corpus), incremental=False))
    if "error" in started:
        raise SystemExit(f"indexing refused: {started['error']}")
    status = _wait_for_index(server, str(corpus), timeout_s)

    rows = []
    latencies = []
    for entry in gold:
        t0 = time.monotonic()
        raw = server.search_code(query=entry["query"], k=k, auto_reindex=False)
        latencies.append((time.monotonic() - t0) * 1000)
        results = json.loads(raw).get("results", [])
        ranked = [(r.get("relative_path") or r.get("file") or "").replace("\\", "/") for r in results]
        rows.append(
            {
                "query": entry["query"],
                "expected_files": entry["expected_files"],
                "ranked_files": ranked[:k],
                "category": entry.get("category", "unknown"),
            }
        )
    summary = summarize(rows)
    summary["p50_latency_ms"] = sorted(latencies)[len(latencies) // 2] if latencies else 0.0
    from embeddings.embedder import resolve_embedding_config
    from search.config import get_search_config

    emb = resolve_embedding_config()
    return {
        "config": {
            "corpus": str(corpus),
            "corpus_revision": _git_head(corpus),
            "gold": gold_path.name,
            "k": k,
            "embedding_provider": emb.provider,
            "embedding_model": getattr(emb, "model", None) or getattr(emb, "model_name", None),
            "reranker": get_search_config().reranker_mode,
            "index_generation": (status.get("index_identity") or {}).get("index_generation"),
        },
        "summary": summary,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True, help="checkout of the pinned public corpus")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--keep-storage", type=Path, default=None, help="reuse/keep this index store instead of a temp dir")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args(argv)

    corpus = args.corpus.resolve()
    if not corpus.is_dir():
        raise SystemExit(f"corpus not found: {corpus}")
    if args.keep_storage:
        storage = args.keep_storage.resolve()
        storage.mkdir(parents=True, exist_ok=True)
        payload = run(corpus, args.gold, args.k, storage, args.timeout)
    else:
        with tempfile.TemporaryDirectory(prefix="code-search-eval-") as tmp:
            payload = run(corpus, args.gold, args.k, Path(tmp), args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    s = payload["summary"]
    print(
        f"{args.gold.name}: n={s['n']} MRR={s['mrr']:.4f} HR@1={s['hr1']:.4f} "
        f"Recall@10={s['recall10']:.4f} p50={s['p50_latency_ms']:.0f}ms "
        f"provider={payload['config']['embedding_provider']} reranker={payload['config']['reranker']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
