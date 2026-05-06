"""Monitor quantization drift across indexed projects.

Background: when a project is first indexed with QUANTIZATION=int8 (default),
FAISS's ScalarQuantizer trains its codebook on the FIRST batch of embeddings
and freezes it. The codebook learns the value range of those initial
embeddings; subsequent vectors are quantized using the SAME range.

If the project's language mix or codebase character SHIFTS over time
(e.g., a Rust-heavy repo gradually adds Python services, or a documentation
fork drifts from a code repo), new embeddings may sit outside the codebook's
learned range. Quantization clamps them, producing degraded similarity
scores. The failure is silent — searches return results, but rankings
slowly degrade.

This tool surfaces that drift quantitatively. For each indexed project:
  1. Sample up to N random vectors from the FAISS index.
  2. For each sampled vector, compute its self-search top-50 mean cosine.
  3. Aggregate per-project: avg_top_50_cosine.
  4. Compare against a saved baseline (if present); flag drift > threshold.

Output:
  --baseline       (default) save current readings as the baseline
  --check          compare against saved baseline; non-zero exit if drift
  --json           emit JSON instead of human-readable text

Honors CODE_SEARCH_STORAGE env var. Designed to run on a quiesced index
(close active MCP server first).

Plan-2 B2 (2026-05-05). See docs/quantization_drift.md for guidance on
interpreting drift signals.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _storage_dir() -> Path:
    base = os.environ.get("CODE_SEARCH_STORAGE")
    if base:
        return Path(os.path.expanduser(base)) / "projects"
    return Path.home() / ".claude_code_search" / "projects"


def _baseline_path() -> Path:
    """Where to persist drift baselines. Stored alongside the projects/ dir."""
    parent = _storage_dir().parent
    parent.mkdir(parents=True, exist_ok=True)
    return parent / "quantization_drift_baseline.json"


def _measure_project(
    index_path: Path, sample_size: int = 100, top_k: int = 50,
) -> Optional[Dict[str, Any]]:
    """Measure self-search avg-top-k cosine for one project's FAISS index.

    Returns dict with avg_top_k_cosine, sample_size, ntotal, dim, index_class
    or None if the index is unreadable / too small / empty.
    """
    if not index_path.exists():
        return None
    try:
        import faiss
    except ImportError:
        return None
    try:
        idx = faiss.read_index(str(index_path))
    except Exception:
        return None

    ntotal = int(idx.ntotal)
    if ntotal < 10:
        # Too few vectors to give a meaningful drift signal.
        return None
    dim = int(idx.d)
    cls_name = type(idx).__name__

    # ScalarQuantizer indexes don't support reconstruct on every variant;
    # try and skip if unsupported.
    try:
        # Sample up to sample_size vectors at random
        sample_count = min(sample_size, ntotal)
        sample_idxs = random.sample(range(ntotal), sample_count)
        # Reconstruct sampled vectors (returns float32 even from quantized
        # indexes — the reconstructed values reflect post-quantization
        # round-trip, which is exactly what we want to measure).
        sampled = np.array([idx.reconstruct(i) for i in sample_idxs], dtype=np.float32)
    except Exception:
        return None

    # Self-search: query each sampled vector against the index, take top-k,
    # average the cosine scores. Healthy quantization → top-1 cosine ~1.0
    # for the sampled vector itself, with rapid falloff. Drifted
    # quantization → top-1 cosine < 1.0 (round-trip lost info).
    try:
        D, _I = idx.search(sampled, top_k)
    except Exception:
        return None

    # D shape: (sample_count, top_k), inner-product / cosine scores.
    # Average across all top-k positions for all samples → single drift
    # signal for the project.
    avg_top_k = float(np.mean(D))
    # Average top-1 cosine: drift away from 1.0 indicates
    # round-trip-quantization loss (the sampled vector should match
    # itself near-perfectly under healthy quantization).
    avg_top_1 = float(np.mean(D[:, 0]))

    return {
        "avg_top_k_cosine": round(avg_top_k, 6),
        "avg_top_1_cosine": round(avg_top_1, 6),
        "sample_size": sample_count,
        "top_k": top_k,
        "ntotal": ntotal,
        "dim": dim,
        "index_class": cls_name,
    }


def measure_all_projects(sample_size: int, top_k: int) -> Dict[str, Dict[str, Any]]:
    """Measure all indexed projects. Returns {project_name: stats}."""
    storage = _storage_dir()
    out: Dict[str, Dict[str, Any]] = {}
    if not storage.is_dir():
        return out
    for proj_dir in sorted(storage.iterdir()):
        if not proj_dir.is_dir():
            continue
        idx_path = proj_dir / "index" / "code.index"
        result = _measure_project(idx_path, sample_size=sample_size, top_k=top_k)
        if result is not None:
            out[proj_dir.name] = result
    return out


def save_baseline(measurements: Dict[str, Dict[str, Any]]) -> Path:
    """Persist baseline measurements with a timestamp."""
    path = _baseline_path()
    payload = {
        "version": "1",
        "saved_at": datetime.now().isoformat(),
        "projects": measurements,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_baseline() -> Optional[Dict[str, Any]]:
    path = _baseline_path()
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def compare_to_baseline(
    current: Dict[str, Dict[str, Any]],
    baseline: Dict[str, Any],
    threshold: float = 0.05,
) -> Dict[str, Any]:
    """Compare current measurements against baseline. Returns drift report."""
    base_projects = baseline.get("projects", {})
    rows: List[Dict[str, Any]] = []
    drift_count = 0
    for name, cur in current.items():
        base = base_projects.get(name)
        if base is None:
            rows.append({
                "name": name, "status": "new",
                "avg_top_1_cosine": cur["avg_top_1_cosine"],
            })
            continue
        delta = round(cur["avg_top_1_cosine"] - base["avg_top_1_cosine"], 6)
        # Drift is bidirectional but the failure mode of interest is
        # cosine DROPPING (worse round-trip). A rising score on
        # incremental adds is unusual — flag both sides asymmetrically.
        is_drift = abs(delta) > threshold
        if is_drift:
            drift_count += 1
        rows.append({
            "name": name,
            "status": "drift" if is_drift else "stable",
            "avg_top_1_baseline": base["avg_top_1_cosine"],
            "avg_top_1_current": cur["avg_top_1_cosine"],
            "delta": delta,
            "ntotal_baseline": base.get("ntotal"),
            "ntotal_current": cur.get("ntotal"),
        })
    # Detect projects in baseline but missing from current
    missing = [n for n in base_projects if n not in current]
    return {
        "baseline_saved_at": baseline.get("saved_at"),
        "compared_at": datetime.now().isoformat(),
        "threshold": threshold,
        "drift_count": drift_count,
        "projects": rows,
        "missing_from_current": missing,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--baseline", action="store_true",
        help="Save current readings as the baseline (default if --check absent).",
    )
    p.add_argument(
        "--check", action="store_true",
        help="Compare against saved baseline; non-zero exit if any drift.",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Emit JSON to stdout instead of human-readable text.",
    )
    p.add_argument(
        "--sample-size", type=int, default=100,
        help="Vectors to sample per project (default 100).",
    )
    p.add_argument(
        "--top-k", type=int, default=50,
        help="Top-k cosine averaged per sampled vector (default 50).",
    )
    p.add_argument(
        "--threshold", type=float, default=0.05,
        help="|delta avg_top_1_cosine| flagged as drift (default 0.05).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for sampling (default 42 — deterministic).",
    )
    args = p.parse_args()

    if not args.baseline and not args.check:
        # Default: baseline mode (one-shot first run).
        args.baseline = True

    random.seed(args.seed)
    t0 = time.time()
    current = measure_all_projects(args.sample_size, args.top_k)
    elapsed = time.time() - t0

    if args.check:
        baseline = load_baseline()
        if baseline is None:
            err = "No baseline found. Run with --baseline first."
            if args.json:
                print(json.dumps({"error": err}))
            else:
                print(err, file=sys.stderr)
            return 2
        report = compare_to_baseline(current, baseline, threshold=args.threshold)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"Drift check vs baseline {report['baseline_saved_at']}")
            print(f"  Threshold: ±{args.threshold} on avg_top_1_cosine")
            print(f"  Projects: {len(report['projects'])} measured, "
                  f"{report['drift_count']} drift, "
                  f"{len(report['missing_from_current'])} missing")
            for row in report["projects"]:
                if row["status"] == "drift":
                    print(f"  [DRIFT] {row['name']}: "
                          f"{row['avg_top_1_baseline']:.4f} -> "
                          f"{row['avg_top_1_current']:.4f} "
                          f"(Δ {row['delta']:+.4f}, "
                          f"ntotal {row.get('ntotal_baseline')}->"
                          f"{row.get('ntotal_current')})")
                elif row["status"] == "new":
                    print(f"  [NEW]   {row['name']}: "
                          f"avg_top_1={row['avg_top_1_cosine']:.4f}")
            if report["missing_from_current"]:
                print(f"  Missing from current: {report['missing_from_current']}")
        return 1 if report["drift_count"] > 0 else 0

    # Baseline mode
    if not current:
        msg = "No indexed projects found."
        if args.json:
            print(json.dumps({"error": msg}))
        else:
            print(msg, file=sys.stderr)
        return 2
    path = save_baseline(current)
    if args.json:
        print(json.dumps({
            "saved_to": str(path),
            "elapsed_seconds": round(elapsed, 2),
            "projects": current,
        }, indent=2))
    else:
        print(f"Baseline saved to {path} ({elapsed:.1f}s)")
        for name, m in current.items():
            print(f"  {name}: avg_top_1={m['avg_top_1_cosine']:.4f} "
                  f"avg_top_{m['top_k']}={m['avg_top_k_cosine']:.4f} "
                  f"ntotal={m['ntotal']} class={m['index_class']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
