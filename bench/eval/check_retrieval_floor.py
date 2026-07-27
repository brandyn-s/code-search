"""Retrieval floor gate — defensive check for catastrophic regression.

Two modes:

  --mode summary       Read an existing eval summary.json and assert
                       MRR / HR@1 floors. Cheap, no API calls. Use after
                       running eval_against_psm_full.py locally.

  --mode index-and-eval  Index a target project from scratch
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
      --floor-semantic-mrr 0.5 \\
      --floor-semantic-hr1 0.40 \\
      --floor-keyword-mrr 0.5 \\
      --floor-keyword-hr1 0.40
"""
from __future__ import annotations

import argparse
import hashlib
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


def setup_server_for_project(
    project_path: str,
    provider: str = "voyage",
    rerank: str = "off",
    model: str | None = None,
):
    """Construct a server for a fresh index without switching prematurely."""
    del project_path  # Kept in the public signature for existing importers.
    os.environ["EMBEDDING_PROVIDER"] = provider
    if provider == "local":
        if not model:
            raise ValueError(
                "provider=local requires --model pointing to a local "
                "SentenceTransformer directory"
            )
        os.environ["LOCAL_EMBEDDING_MODEL"] = model
        os.environ["VOYAGE_INPUT_TYPE"] = "off"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    elif model:
        os.environ["EMBEDDING_MODEL"] = model
    elif provider == "voyage":
        os.environ["EMBEDDING_MODEL"] = "voyage-4-large"
    elif provider == "voyage-context":
        os.environ["EMBEDDING_MODEL"] = "voyage-context-3"
    if provider.startswith("voyage"):
        os.environ["VOYAGE_INPUT_TYPE"] = "on"
    os.environ["RERANKER"] = rerank
    os.environ["QUERY_EXPANSION"] = "off"
    os.environ["QUANTIZATION"] = "float32"

    from common_utils import get_storage_dir
    get_storage_dir.cache_clear()
    from mcp_server.code_search_server import CodeSearchServer

    return CodeSearchServer()


def _terminal_index_error(
    progress: object,
    expected_job_id: str,
    *,
    expected_directory: str,
    expected_project_name: str,
    expected_provider: str,
) -> str | None:
    """Return None only for the exact successful terminal job contract."""
    if not isinstance(progress, dict):
        return "indexing progress must be a JSON object"

    status = progress.get("status")
    actual_job_id = progress.get("job_id")
    if actual_job_id != expected_job_id:
        return (
            "indexing progress job mismatch: "
            f"expected {expected_job_id!r}, got {actual_job_id!r}"
        )
    for field, expected in (
        ("directory", expected_directory),
        ("project_name", expected_project_name),
        ("provider", expected_provider),
    ):
        if progress.get(field) != expected:
            return (
                f"indexing progress {field} mismatch: "
                f"expected {expected!r}, got {progress.get(field)!r}"
            )
    if status != "completed":
        return f"indexing reached non-success terminal status {status!r}"
    if progress.get("index_ready") is not True:
        return "completed indexing progress did not report index_ready=true"
    if progress.get("error"):
        return f"completed indexing progress reported error: {progress['error']}"

    result = progress.get("result")
    if not isinstance(result, dict):
        return "completed indexing progress is missing an object result"
    for field, expected in (
        ("directory", expected_directory),
        ("project_name", expected_project_name),
        ("provider", expected_provider),
    ):
        if result.get(field) != expected:
            return (
                f"completed indexing result {field} mismatch: "
                f"expected {expected!r}, got {result.get(field)!r}"
            )
    if result.get("success") is not True:
        return "completed indexing result did not report success=true"
    if result.get("index_ready") is not True:
        return "completed indexing result did not report index_ready=true"
    error = result.get("error")
    if error is not None and error != "":
        return f"completed indexing result reported error: {error}"
    return None


def index_project(
    server,
    project_path: str,
    *,
    provider: str,
    timeout_seconds: float = 600,
    poll_interval_seconds: float = 1,
) -> bool:
    """Index project freshly (CI mode). Returns True on success."""
    expected_directory = str(Path(project_path).resolve())
    expected_project_name = Path(expected_directory).name
    raw = server.index_directory(
        directory_path=expected_directory,
        incremental=False,
        provider=provider,
    )
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        print(
            f"[retrieval-floor-gate] FAIL: index_directory returned invalid JSON: {exc}",
            file=sys.stderr,
        )
        return False
    if not isinstance(parsed, dict):
        print(
            "[retrieval-floor-gate] FAIL: index_directory must return a "
            f"JSON object, got {type(parsed).__name__}",
            file=sys.stderr,
        )
        return False
    if "error" in parsed:
        print(f"[retrieval-floor-gate] FAIL: index_directory failed: {parsed['error']}",
              file=sys.stderr)
        return False

    job_id = parsed.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        print(
            "[retrieval-floor-gate] FAIL: index_directory did not return a "
            "nonempty job_id",
            file=sys.stderr,
        )
        return False
    if parsed.get("status") != "indexing":
        print(
            "[retrieval-floor-gate] FAIL: index_directory did not start an "
            f"indexing job: status={parsed.get('status')!r}",
            file=sys.stderr,
        )
        return False
    start_binding_errors = []
    if parsed.get("indexing_conflict") is True:
        start_binding_errors.append("indexing_conflict=true")
    if parsed.get("directory") != expected_directory:
        start_binding_errors.append(
            f"directory={parsed.get('directory')!r}, "
            f"expected {expected_directory!r}"
        )
    if parsed.get("project_name") != expected_project_name:
        start_binding_errors.append(
            f"project_name={parsed.get('project_name')!r}, "
            f"expected {expected_project_name!r}"
        )
    if parsed.get("provider") != provider:
        start_binding_errors.append(
            f"provider={parsed.get('provider')!r}, expected {provider!r}"
        )
    if parsed.get("index_ready") is not False:
        start_binding_errors.append(
            f"index_ready={parsed.get('index_ready')!r}, expected false"
        )
    if start_binding_errors:
        print(
            "[retrieval-floor-gate] FAIL: index_directory returned a job "
            "bound to the wrong fresh-index request: "
            + "; ".join(start_binding_errors),
            file=sys.stderr,
        )
        return False

    # Poll for completion
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        raw_p = server.get_indexing_progress()
        try:
            prog = json.loads(raw_p)
        except (TypeError, json.JSONDecodeError) as exc:
            print(
                "[retrieval-floor-gate] FAIL: get_indexing_progress returned "
                f"invalid JSON for job {job_id}: {exc}",
                file=sys.stderr,
            )
            return False
        if not isinstance(prog, dict):
            print(
                "[retrieval-floor-gate] FAIL: get_indexing_progress must "
                f"return a JSON object for job {job_id}, got "
                f"{type(prog).__name__}",
                file=sys.stderr,
            )
            return False
        status = prog.get("status")
        if status == "indexing":
            pct = prog.get("percent", 0)
            phase = prog.get("phase", "?")
            print(f"  indexing job {job_id}: {phase} {pct}%")
            time.sleep(poll_interval_seconds)
            continue
        if status in ("completed", "failed", "cancelled", "idle"):
            terminal_error = _terminal_index_error(
                prog,
                job_id,
                expected_directory=expected_directory,
                expected_project_name=expected_project_name,
                expected_provider=provider,
            )
            if terminal_error is None:
                return True
            print(
                f"[retrieval-floor-gate] FAIL: {terminal_error}; progress={prog}",
                file=sys.stderr)
            return False
        print(
            "[retrieval-floor-gate] FAIL: get_indexing_progress returned "
            f"unknown status {status!r} for job {job_id}; progress={prog}",
            file=sys.stderr,
        )
        return False

    print(
        "[retrieval-floor-gate] FAIL: indexing timeout after "
        f"{timeout_seconds:g}s for job {job_id}",
        file=sys.stderr,
    )
    return False


def switch_to_indexed_project(
    server,
    project_path: str,
    *,
    provider: str,
) -> bool:
    """Switch only when the server confirms the requested index identity."""
    expected_path = str(Path(project_path).resolve())
    expected_name = Path(expected_path).name
    raw = server.switch_project(
        project_path=expected_path,
        provider=provider,
    )
    try:
        switched = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        print(
            "[retrieval-floor-gate] FAIL: switch_project returned invalid "
            f"JSON after indexing: {exc}",
            file=sys.stderr,
        )
        return False
    if not isinstance(switched, dict):
        print(
            "[retrieval-floor-gate] FAIL: switch_project must return a JSON "
            f"object, got {type(switched).__name__}",
            file=sys.stderr,
        )
        return False

    failures = []
    if "error" in switched:
        failures.append(f"error={switched['error']!r}")
    if switched.get("success") is not True:
        failures.append(f"success={switched.get('success')!r}")
    project_info = switched.get("project_info")
    if not isinstance(project_info, dict):
        failures.append("project_info is not an object")
    else:
        for field, expected in (
            ("project_name", expected_name),
            ("project_path", expected_path),
            ("embedding_provider", provider),
        ):
            if project_info.get(field) != expected:
                failures.append(
                    f"project_info.{field}={project_info.get(field)!r}, "
                    f"expected {expected!r}"
                )
    if failures:
        print(
            "[retrieval-floor-gate] FAIL: switch_project did not confirm "
            "the requested successful index: "
            + "; ".join(failures),
            file=sys.stderr,
        )
        return False
    return True


def verify_fixture_manifest(
    manifest_path: Path,
    *,
    project_path: Path,
    gold_path: Path,
) -> tuple[bool, str]:
    """Verify every checksummed fixture file and reject unlisted corpus files."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, f"fixture manifest not found: {manifest_path}"
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"could not read fixture manifest {manifest_path}: {exc}"

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return False, "fixture manifest schema_version must equal 1"
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        return False, "fixture manifest files must be a nonempty object"
    if "gold.json" not in files:
        return False, "fixture manifest must checksum gold.json"

    fixture_root = manifest_path.resolve().parent
    expected_project = (fixture_root / "corpus").resolve()
    if project_path.resolve() != expected_project:
        return (
            False,
            (
                "fixture manifest project path mismatch: "
                f"expected {expected_project}, got {project_path.resolve()}"
            ),
        )
    expected_gold = (fixture_root / "gold.json").resolve()
    if gold_path.resolve() != expected_gold:
        return (
            False,
            (
                "fixture manifest gold path mismatch: "
                f"expected {expected_gold}, got {gold_path.resolve()}"
            ),
        )
    listed_corpus: set[str] = set()
    for relative_name, expected in sorted(files.items()):
        if not isinstance(relative_name, str) or not isinstance(expected, dict):
            return False, "fixture manifest file entries must be checksum objects"
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            return False, f"fixture manifest path is unsafe: {relative_name!r}"
        expected_sha = expected.get("sha256")
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_sha)
        ):
            return False, f"invalid sha256 for {relative_name}"
        target = fixture_root / relative
        if not target.is_file() or target.is_symlink():
            return False, f"fixture file is missing or not regular: {relative_name}"
        actual_sha = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            return (
                False,
                (
                    f"fixture checksum mismatch for {relative_name}: "
                    f"expected {expected_sha}, got {actual_sha}"
                ),
            )
        if relative.parts and relative.parts[0] == "corpus":
            listed_corpus.add(relative.as_posix())

    corpus_root = fixture_root / "corpus"
    actual_corpus = {
        path.relative_to(fixture_root).as_posix()
        for path in corpus_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_corpus != listed_corpus:
        missing = sorted(actual_corpus - listed_corpus)
        extra = sorted(listed_corpus - actual_corpus)
        return (
            False,
            (
                "fixture manifest corpus file set mismatch: "
                f"unlisted={missing}, missing={extra}"
            ),
        )
    return True, ""


REQUIRED_RETRIEVAL_ARMS = ("semantic", "keyword")


def load_gold_queries(gold_path: Path) -> list[dict]:
    """Load a complete, scoreable gold set or fail before any search."""
    try:
        queries = json.loads(gold_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"gold file is not readable JSON: {gold_path}: {exc}"
        ) from exc
    if not isinstance(queries, list) or not queries:
        raise ValueError("gold file must contain a nonempty JSON list")

    seen_queries: set[str] = set()
    for index, row in enumerate(queries, 1):
        if not isinstance(row, dict):
            raise TypeError(f"gold row {index} must be a JSON object")
        query = row.get("query")
        if (
            not isinstance(query, str)
            or not query
            or query.strip() != query
        ):
            raise ValueError(
                f"gold row {index} query must be a nonempty string"
            )
        if query in seen_queries:
            raise ValueError(f"gold row {index} duplicates query text")
        seen_queries.add(query)

        expected_files = row.get("expected_files")
        if (
            not isinstance(expected_files, list)
            or not expected_files
            or any(
                not isinstance(path, str)
                or not path
                or path.strip() != path
                for path in expected_files
            )
            or len(expected_files) != len(set(expected_files))
        ):
            raise ValueError(
                f"gold row {index} expected_files must contain unique "
                "nonempty paths"
            )
    return queries


def eval_gold(
    server,
    gold_path: Path,
    *,
    search_mode: str = "hybrid",
    queries: list[dict] | None = None,
) -> dict:
    """Run gold queries through one production mode and compute retrieval metrics."""
    if queries is None:
        queries = load_gold_queries(gold_path)
    print(
        f"  loaded {len(queries)} gold queries from {gold_path} "
        f"(search_mode={search_mode})"
    )

    rows = []
    for i, q in enumerate(queries, 1):
        expected = set(q["expected_files"])
        t0 = time.time()
        raw = server.search_code(
            query=q["query"],
            k=10,
            search_mode=search_mode,
            auto_reindex=False,
        )
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

    scored = rows
    n = len(scored)
    if n == 0:
        return {
            "n": 0,
            "loaded_count": len(queries),
            "scored_count": 0,
            "rows": rows,
        }
    return {
        "n": n,
        "loaded_count": len(queries),
        "scored_count": n,
        "hr_1": sum(1 for r in scored if r["hit_1"]) / n,
        "hr_5": sum(1 for r in scored if r["hit_5"]) / n,
        "mrr": sum(r["rr"] for r in scored) / n,
        "median_latency_ms": median([r["latency_ms"] for r in scored]),
        "rows": rows,
    }


def eval_required_arms(server, gold_path: Path) -> dict[str, dict]:
    """Evaluate semantic/vector and keyword/BM25 production paths separately."""
    queries = load_gold_queries(gold_path)
    return {
        search_mode: eval_gold(
            server,
            gold_path,
            search_mode=search_mode,
            queries=queries,
        )
        for search_mode in REQUIRED_RETRIEVAL_ARMS
    }


def required_arm_floor_failures(
    summaries: dict[str, dict],
    *,
    floors: dict[str, dict[str, float | None]],
) -> list[str]:
    """Return violations for every required retrieval arm and metric."""
    failures: list[str] = []
    for arm in REQUIRED_RETRIEVAL_ARMS:
        summary = summaries.get(arm)
        if not isinstance(summary, dict):
            failures.append(f"{arm} evaluated no queries")
            continue
        loaded_count = summary.get("loaded_count")
        scored_count = summary.get("scored_count")
        if (
            not isinstance(loaded_count, int)
            or isinstance(loaded_count, bool)
            or loaded_count <= 0
        ):
            failures.append(f"{arm} loaded_count is invalid")
            continue
        if (
            not isinstance(scored_count, int)
            or isinstance(scored_count, bool)
            or scored_count < 0
        ):
            failures.append(f"{arm} scored_count is invalid")
            continue
        if scored_count != loaded_count:
            failures.append(
                f"{arm} scored_count {scored_count} != "
                f"loaded_count {loaded_count}"
            )
        if summary.get("n") != scored_count:
            failures.append(
                f"{arm} n {summary.get('n')!r} != "
                f"scored_count {scored_count}"
            )
        if scored_count == 0:
            failures.append(f"{arm} evaluated no queries")
            continue
        arm_floors = floors.get(arm, {})
        for metric, label in (("mrr", "MRR"), ("hr_1", "HR@1")):
            floor = arm_floors.get(metric)
            if floor is None:
                failures.append(f"{arm} {label} floor is not configured")
                continue
            value = summary.get(metric)
            if not isinstance(value, (int, float)) or value < floor:
                rendered = (
                    f"{value:.4f}"
                    if isinstance(value, (int, float))
                    else repr(value)
                )
                failures.append(
                    f"{arm} {label} {rendered} < floor {floor:.4f}"
                )
    return failures


def check_index_and_eval_mode(args: argparse.Namespace) -> int:
    """Index project fresh + run eval + assert floors."""
    if args.provider.startswith("voyage") and not os.environ.get("VOYAGE_API_KEY"):
        print(
            "[retrieval-floor-gate] FAIL: VOYAGE_API_KEY not set for "
            f"provider={args.provider}",
            file=sys.stderr,
        )
        return 1
    if args.provider == "local" and os.environ.get("PYTHONHASHSEED") != "0":
        print(
            "[retrieval-floor-gate] FAIL: provider=local requires "
            "PYTHONHASHSEED=0 before Python starts",
            file=sys.stderr,
        )
        return 1

    project_path = str(Path(args.project).resolve())
    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"[retrieval-floor-gate] FAIL: gold file not found: {gold_path}",
              file=sys.stderr)
        return 1
    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_ok, manifest_error = verify_fixture_manifest(
            manifest_path,
            project_path=Path(project_path),
            gold_path=gold_path,
        )
        if not manifest_ok:
            print(
                f"[retrieval-floor-gate] FAIL: {manifest_error}",
                file=sys.stderr,
            )
            return 1
        print(f"  verified fixture manifest: {manifest_path}")

    print("[retrieval-floor-gate] mode=index-and-eval")
    print(f"  project: {project_path}")
    print(f"  gold:    {gold_path}")
    print(f"  provider:{args.provider}")
    print(f"  rerank:  {args.rerank}")

    try:
        server = setup_server_for_project(
            project_path,
            provider=args.provider,
            rerank=args.rerank,
            model=args.model,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            f"[retrieval-floor-gate] FAIL: server setup failed: {exc}",
            file=sys.stderr,
        )
        return 1

    print("  indexing target project...")
    if not index_project(
        server,
        project_path,
        provider=args.provider,
        timeout_seconds=args.index_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
    ):
        return 1

    if not switch_to_indexed_project(
        server,
        project_path,
        provider=args.provider,
    ):
        return 1

    floors = {
        "semantic": {
            "mrr": (
                args.floor_semantic_mrr
                if args.floor_semantic_mrr is not None
                else args.floor_mrr
            ),
            "hr_1": (
                args.floor_semantic_hr1
                if args.floor_semantic_hr1 is not None
                else args.floor_hr1
            ),
        },
        "keyword": {
            "mrr": (
                args.floor_keyword_mrr
                if args.floor_keyword_mrr is not None
                else args.floor_mrr
            ),
            "hr_1": (
                args.floor_keyword_hr1
                if args.floor_keyword_hr1 is not None
                else args.floor_hr1
            ),
        },
    }

    print("  running required production retrieval arms...")
    summaries = eval_required_arms(server, gold_path)
    for arm in REQUIRED_RETRIEVAL_ARMS:
        summary = summaries[arm]
        n = summary.get("n", 0)
        if n == 0:
            print(f"  measured {arm}: n=0")
            continue
        print(
            f"  measured {arm}: n={n}  MRR={summary['mrr']:.4f}  "
            f"HR@1={summary['hr_1']:.4f}  HR@5={summary['hr_5']:.4f}  "
            f"median_latency_ms={summary['median_latency_ms']:.0f}"
        )

    failures = required_arm_floor_failures(summaries, floors=floors)

    if failures:
        print("[retrieval-floor-gate] FAIL: floor violation(s):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        if args.dump_rows:
            print("  per-query rows:", file=sys.stderr)
            for arm in REQUIRED_RETRIEVAL_ARMS:
                print(f"    {arm}:", file=sys.stderr)
                for row in summaries[arm]["rows"][:20]:
                    print(f"      {row}", file=sys.stderr)
        return 1

    print(
        "[retrieval-floor-gate] PASS: semantic/vector and keyword/BM25 "
        "arms cleared their independent floors"
    )
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
    parser.add_argument(
        "--manifest",
        help="Optional checksummed fixture manifest to verify before indexing",
    )
    parser.add_argument(
        "--floor-mrr",
        type=float,
        default=None,
        help="Legacy fallback MRR floor for both required retrieval arms",
    )
    parser.add_argument(
        "--floor-hr1",
        type=float,
        default=None,
        help="Legacy fallback HR@1 floor for both required retrieval arms",
    )
    parser.add_argument("--floor-semantic-mrr", type=float, default=None)
    parser.add_argument("--floor-semantic-hr1", type=float, default=None)
    parser.add_argument("--floor-keyword-mrr", type=float, default=None)
    parser.add_argument("--floor-keyword-hr1", type=float, default=None)
    parser.add_argument("--provider", default="voyage",
                        choices=["local", "voyage", "voyage-context"])
    parser.add_argument(
        "--model",
        help="Embedding model name or local SentenceTransformer directory",
    )
    parser.add_argument("--rerank", default="off",
                        choices=["off", "sonnet", "cross-encoder"])
    parser.add_argument(
        "--index-timeout-seconds",
        type=float,
        default=600,
        help="Bounded wait for the background indexing job (default: 600)",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=1,
        help="Index progress polling interval (default: 1)",
    )
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
