"""Layered experiment harness for hybrid search tuning.

Runs 5 experiment layers against the EXP-008 10-query battery.
Each layer tests one variable, picks the winner, and passes it forward.

Usage:
    OPENAI_API_KEY=sk-... python tests/experiments/run_experiments.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
logging.basicConfig(level=logging.WARNING)

QUERIES = [
    ("S01", "What calls _build_oauth?"),
    ("S02", "Find dead code - functions nobody calls"),
    ("S03", "Show all HTTP routes"),
    ("S04", "Trace callers of check_permissions"),
    ("S05", "Blast radius of changing shared/mcp_http.py"),
    ("C01", "Find authentication logic"),
    ("C02", "Where do we handle errors?"),
    ("C03", "Show retry patterns"),
    ("C04", "Find rate limiting code"),
    ("C05", "How is OPA authorization enforced?"),
]

# Ground truth: chunk_id/name/content substrings that count as "Good" (score=2)
GROUND_TRUTH = {
    "S01": ["_build_oauth", "configure_http_transport"],
    "S02": [],
    "S03": ["Route(", "slack_connect_app.py"],
    "S04": ["check_permissions"],
    "S05": ["mcp_http.py", "configure_http_transport"],
    "C01": ["_build_oauth", "_extract_identity", "_authorize_tool_call", "OPAMiddleware",
            "_decode_jwt", "entra_callback", "check_permissions"],
    "C02": ["_jsonrpc_error", "ToolError", "check_error_handling", "HTTPException",
            "circuit_record_error"],
    "C03": ["_retry_delay", "_is_retryable_status", "proxy_messages"],
    "C04": ["check_rate_limit", "_rate_limit_overrides", "_get_rate_limits"],
    "C05": ["OPAMiddleware", "_authorize_tool_call", "_filter_tools_list",
            "_decode_jwt", "_extract_identity", "opa_middleware"],
}

MCP_SERVERS_PATH = "C:~/Documents/GitHub/mcp-servers"


def score_result(qid, result):
    """Score a single result: 2=Good, 1=Partial, 0=Miss."""
    truth = GROUND_TRUTH.get(qid, [])
    if not truth:
        return 0

    chunk_id = result.chunk_id.lower()
    name = (result.name or "").lower()
    content = (result.content_preview or "").lower()
    rel_path = (result.relative_path or "").lower()

    for t in truth:
        tl = t.lower()
        if tl in chunk_id or tl in name or tl in content:
            if not rel_path.endswith(".md"):
                return 2
            else:
                return 1
    if rel_path.endswith(".py"):
        return 1
    return 0


def run_experiment(searcher, config_name, env_overrides):
    """Run all 10 queries with given env overrides. Return scores dict."""
    old_env = {}
    for k, v in env_overrides.items():
        old_env[k] = os.environ.get(k)
        os.environ[k] = str(v)

    scores = {}
    for qid, query in QUERIES:
        try:
            results = searcher.search(query, k=3, search_mode="hybrid", context_depth=0)
            top1_score = score_result(qid, results[0]) if results else 0
            scores[qid] = {
                "score": top1_score,
                "chunk_id": results[0].chunk_id if results else "(none)",
                "name": results[0].name if results else None,
                "type": results[0].chunk_type if results else None,
            }
        except Exception as e:
            scores[qid] = {"score": 0, "chunk_id": f"ERROR: {e}", "name": None, "type": None}

    for k, v in old_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

    total = sum(s["score"] for s in scores.values())
    good_count = sum(1 for s in scores.values() if s["score"] == 2)
    return {"config": config_name, "total": total, "good": good_count, "scores": scores}


def print_result_row(result):
    scores_str = " ".join(
        f"{qid}={'G' if s['score']==2 else 'P' if s['score']==1 else '-'}"
        for qid, s in result["scores"].items()
    )
    print(f"  {result['config']:<45} | {result['good']}/10 Good | {result['total']}/20 pts | {scores_str}")


def index_repo():
    """Index mcp-servers with full content."""
    from search.indexer import CodeIndexManager
    from embeddings.embedder import CodeEmbedder
    from chunking.multi_language_chunker import MultiLanguageChunker
    from common_utils import get_storage_dir

    storage_dir = str(get_storage_dir() / "projects" / "exp008_validation" / "index")
    index_mgr = CodeIndexManager(storage_dir)

    stats = index_mgr.get_stats()
    if stats.get("total_chunks", 0) > 0:
        print(f"  Already indexed: {stats['total_chunks']} chunks")
        return index_mgr

    print(f"  Indexing {MCP_SERVERS_PATH}...")
    embedder = CodeEmbedder()
    chunker = MultiLanguageChunker()

    all_chunks = []
    skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", ".cocoindex_code", ".mypy_cache"}

    for root, dirs, files in os.walk(MCP_SERVERS_PATH):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in files:
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, MCP_SERVERS_PATH).replace("\\", "/")
            try:
                chunks = chunker.chunk_file(fpath)
                if chunks:
                    for c in chunks:
                        c.relative_path = rel_path
                    all_chunks.extend(chunks)
            except Exception:
                pass

    print(f"  Chunked: {len(all_chunks)} chunks")

    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        results = embedder.embed_chunks(batch, batch_size=batch_size)
        index_mgr.add_embeddings(results)
        done = min(i + batch_size, len(all_chunks))
        print(f"  Embedded: {done}/{len(all_chunks)}", end="\r")

    index_mgr.save_index()
    stats = index_mgr.get_stats()
    print(f"\n  Done: {stats['total_chunks']} chunks, {stats.get('files_indexed', '?')} files")
    return index_mgr


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
    os.environ.setdefault("CONTENT_MODE", "code")

    print("=" * 110)
    print("HYBRID SEARCH EXPERIMENT SUITE")
    print("=" * 110)

    # Index
    print("\n[INDEX]")
    t0 = time.time()
    index_mgr = index_repo()
    print(f"  Index time: {time.time() - t0:.1f}s")

    from embeddings.embedder import CodeEmbedder
    from search.searcher import IntelligentSearcher
    embedder = CodeEmbedder()
    searcher = IntelligentSearcher(index_mgr, embedder)

    from search import searcher as searcher_module
    all_results = []

    # ========== LAYER A: Baseline ==========
    print(f"\n{'=' * 110}")
    print("LAYER A: Baseline (full FTS5 content + name_weight=5.0)")
    print("=" * 110)

    baseline = run_experiment(searcher, "A: baseline (current)", {
        "CONTENT_MODE": "code", "VECTOR_WEIGHT": "0.4", "BM25_WEIGHT": "0.6",
        "QUERY_EXPANSION": "off",
    })
    all_results.append(baseline)
    print_result_row(baseline)

    # ========== LAYER C: RRF weight spectrum ==========
    print(f"\n{'=' * 110}")
    print("LAYER C: RRF weight spectrum (vector/bm25)")
    print("=" * 110)

    rrf_configs = [
        ("C1: v=0.1 / b=0.9", "0.1", "0.9"),
        ("C2: v=0.2 / b=0.8", "0.2", "0.8"),
        ("C3: v=0.3 / b=0.7", "0.3", "0.7"),
        ("C4: v=0.4 / b=0.6 (current)", "0.4", "0.6"),
        ("C5: v=0.5 / b=0.5", "0.5", "0.5"),
        ("C6: v=0.6 / b=0.4", "0.6", "0.4"),
    ]

    best_c = None
    for name, vw, bw in rrf_configs:
        result = run_experiment(searcher, name, {
            "CONTENT_MODE": "code", "VECTOR_WEIGHT": vw, "BM25_WEIGHT": bw,
            "QUERY_EXPANSION": "off",
        })
        all_results.append(result)
        print_result_row(result)
        if best_c is None or result["total"] > best_c["total"] or (result["total"] == best_c["total"] and result["good"] > best_c["good"]):
            best_c = result

    # Extract winning weights from config name
    best_vw = best_c["config"].split("v=")[1].split(" ")[0]
    best_bw = best_c["config"].split("b=")[1].rstrip(")").split(")")[0]
    print(f"  >> Winner: {best_c['config']} ({best_c['good']}/10 Good, {best_c['total']}/20 pts)")

    # ========== LAYER D: Chunk type boost strength ==========
    print(f"\n{'=' * 110}")
    print(f"LAYER D: Chunk boost strength (on best RRF: v={best_vw}/b={best_bw})")
    print("=" * 110)

    original_boosts = searcher_module.CHUNK_TYPE_BOOSTS["code"].copy()

    boost_configs = [
        ("D1: code=1.3 / doc=0.7 (current)", 1.3, 0.7),
        ("D2: code=1.5 / doc=0.5", 1.5, 0.5),
        ("D3: code=1.8 / doc=0.4", 1.8, 0.4),
        ("D4: code=2.0 / doc=0.3", 2.0, 0.3),
    ]

    best_d = None
    best_code_boost = 1.3
    best_doc_boost = 0.7
    for name, code_boost, doc_boost in boost_configs:
        searcher_module.CHUNK_TYPE_BOOSTS["code"] = {
            "function": code_boost, "method": code_boost, "class": code_boost,
            "decorated_definition": code_boost,
            "section": doc_boost, "document": doc_boost, "module": 0.9,
        }
        result = run_experiment(searcher, name, {
            "CONTENT_MODE": "code", "VECTOR_WEIGHT": best_vw, "BM25_WEIGHT": best_bw,
            "QUERY_EXPANSION": "off",
        })
        all_results.append(result)
        print_result_row(result)
        if best_d is None or result["total"] > best_d["total"] or (result["total"] == best_d["total"] and result["good"] > best_d["good"]):
            best_d = result
            best_code_boost = code_boost
            best_doc_boost = doc_boost

    # Set winner for next layer
    searcher_module.CHUNK_TYPE_BOOSTS["code"] = {
        "function": best_code_boost, "method": best_code_boost, "class": best_code_boost,
        "decorated_definition": best_code_boost,
        "section": best_doc_boost, "document": best_doc_boost, "module": 0.9,
    }
    print(f"  >> Winner: {best_d['config']} ({best_d['good']}/10 Good, {best_d['total']}/20 pts)")

    # ========== LAYER E: Query expansion ==========
    print(f"\n{'=' * 110}")
    print(f"LAYER E: Query expansion (on D winner: code={best_code_boost}/doc={best_doc_boost})")
    print("=" * 110)

    best_e = None
    best_expansion = "off"
    for name, expansion in [("E1: expansion=off", "off"), ("E2: expansion=on", "on")]:
        result = run_experiment(searcher, name, {
            "CONTENT_MODE": "code", "VECTOR_WEIGHT": best_vw, "BM25_WEIGHT": best_bw,
            "QUERY_EXPANSION": expansion,
        })
        all_results.append(result)
        print_result_row(result)
        if best_e is None or result["total"] > best_e["total"] or (result["total"] == best_e["total"] and result["good"] > best_e["good"]):
            best_e = result
            best_expansion = expansion

    print(f"  >> Winner: {best_e['config']} ({best_e['good']}/10 Good, {best_e['total']}/20 pts)")

    # ========== FINAL COMPARISON ==========
    print(f"\n{'=' * 110}")
    print("FINAL: Combined winner vs original baseline")
    print("=" * 110)

    # Original baseline (pre-experiment defaults)
    searcher_module.CHUNK_TYPE_BOOSTS["code"] = original_boosts
    baseline_final = run_experiment(searcher, "BASELINE (original defaults)", {
        "CONTENT_MODE": "code", "VECTOR_WEIGHT": "0.4", "BM25_WEIGHT": "0.6",
        "QUERY_EXPANSION": "off",
    })

    # Combined winner
    searcher_module.CHUNK_TYPE_BOOSTS["code"] = {
        "function": best_code_boost, "method": best_code_boost, "class": best_code_boost,
        "decorated_definition": best_code_boost,
        "section": best_doc_boost, "document": best_doc_boost, "module": 0.9,
    }
    winner = run_experiment(searcher, "WINNER (combined best)", {
        "CONTENT_MODE": "code", "VECTOR_WEIGHT": best_vw, "BM25_WEIGHT": best_bw,
        "QUERY_EXPANSION": best_expansion,
    })

    print()
    print_result_row(baseline_final)
    print_result_row(winner)
    print()

    # Per-query detail
    print(f"  {'Query':<6} | {'Baseline':<50} | {'Winner':<50} | Delta")
    print(f"  {'-'*6} | {'-'*50} | {'-'*50} | -----")
    for qid in [q[0] for q in QUERIES]:
        bs = baseline_final["scores"][qid]
        ws = winner["scores"][qid]
        delta = ws["score"] - bs["score"]
        delta_str = f"+{delta}" if delta > 0 else str(delta) if delta < 0 else "="
        bs_label = f"{'G' if bs['score']==2 else 'P' if bs['score']==1 else '-'} {bs['chunk_id'][:45]}"
        ws_label = f"{'G' if ws['score']==2 else 'P' if ws['score']==1 else '-'} {ws['chunk_id'][:45]}"
        print(f"  {qid:<6} | {bs_label:<50} | {ws_label:<50} | {delta_str}")

    print(f"\n  WINNING CONFIG:")
    print(f"    VECTOR_WEIGHT={best_vw}")
    print(f"    BM25_WEIGHT={best_bw}")
    print(f"    CHUNK_TYPE_BOOSTS: code={best_code_boost}, doc={best_doc_boost}")
    print(f"    QUERY_EXPANSION={best_expansion}")
    print(f"    FTS5: full content + name_weight=5.0")

    # Restore and cleanup
    searcher_module.CHUNK_TYPE_BOOSTS["code"] = original_boosts
    if index_mgr._metadata_db is not None:
        index_mgr._metadata_db.close()
        index_mgr._metadata_db = None
    if hasattr(index_mgr, "_fts_conn") and index_mgr._fts_conn is not None:
        index_mgr._fts_conn.close()
        index_mgr._fts_conn = None


if __name__ == "__main__":
    main()
