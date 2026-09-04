"""Local end-to-end retrieval battery — no API keys required.

Indexes a copy of this repo's own core modules with the `local`
sentence-transformers provider and validates the full stack: chunk → embed →
FAISS+FTS5 → hybrid search. Smoke-level correctness (known-item file hits,
quantization parity), NOT a quality eval — MRR claims still require the PSM
harness (internal eval runbook).

Skips when sentence-transformers (or its model download) is unavailable.
Run: python3 -m pytest tests/integration/test_local_e2e_battery.py -v
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("sentence_transformers")

from chunking.multi_language_chunker import MultiLanguageChunker  # noqa: E402
from embeddings.embedder import CodeEmbedder  # noqa: E402
from merkle.snapshot_manager import SnapshotManager  # noqa: E402
from search.incremental_indexer import IncrementalIndexer  # noqa: E402
from search.indexer import CodeIndexManager  # noqa: E402
from search.searcher import IntelligentSearcher  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# (query, expected file in top-5) — known-item lookups against this repo.
GOLDEN = [
    ("weighted reciprocal rank fusion of vector and bm25", "search/searcher.py"),
    ("sanitize fts5 match query operators", "search/indexer.py"),
    ("greedy merge adjacent chunks nws budget", "chunking/chunk_merging.py"),
    ("personalized pagerank scorer over code graph", "search/ppr_scorer.py"),
]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    dst = tmp_path_factory.mktemp("e2e_corpus")
    for d in ("search", "chunking"):
        shutil.copytree(REPO_ROOT / d, dst / d)
    return dst


def _build(corpus: Path, storage: Path, quant: str, monkeypatch_env):
    monkeypatch_env.setenv("QUANTIZATION", quant)
    ii = IncrementalIndexer(
        indexer=CodeIndexManager(str(storage / f"index_{quant}")),
        embedder=CodeEmbedder(),
        chunker=MultiLanguageChunker(root_path=str(corpus)),
        snapshot_manager=SnapshotManager(storage / f"snaps_{quant}"),
    )
    res = ii.incremental_index(str(corpus), project_name=f"e2e_{quant}",
                               force_full=True)
    assert res.success and res.chunks_added > 50
    return ii


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Module-scoped env: local provider, reranker off, isolated storage."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    storage = tmp_path_factory.mktemp("e2e_storage")
    mp.setenv("CODE_SEARCH_STORAGE", str(storage))
    mp.setenv("EMBEDDING_PROVIDER", "local")
    mp.setenv("RERANKER", "off")
    from common_utils import get_storage_dir
    get_storage_dir.cache_clear()
    from search.config import get_search_config
    get_search_config.cache_clear()
    CodeEmbedder._query_cache.clear()
    yield mp, storage
    get_search_config.cache_clear()
    get_storage_dir.cache_clear()
    CodeEmbedder._query_cache.clear()
    mp.undo()


def test_known_item_hits_and_quantization_parity(env, corpus):
    mp, storage = env
    arms = {q: _build(corpus, storage, q, mp) for q in ("int8", "float32")}

    top10 = {}
    for quant, ii in arms.items():
        mp.setenv("QUANTIZATION", quant)
        searcher = IntelligentSearcher(ii.indexer, ii.embedder)
        ids_per_query = []
        hits = 0
        for query, expected in GOLDEN:
            results = searcher.search(query, k=10)
            assert results, f"[{quant}] no results for {query!r}"
            paths = [r.relative_path for r in results[:5]]
            hits += expected in paths
            ids_per_query.append(set(r.chunk_id for r in results))
        top10[quant] = ids_per_query
        # Smoke floor, not a quality claim: most known-item lookups must
        # land the right file in the top 5.
        assert hits >= len(GOLDEN) - 1, (
            f"[{quant}] only {hits}/{len(GOLDEN)} known-item hits"
        )

    # int8 vs float32 top-10 overlap — int8 quantization is near-lossless;
    # a large gap here is the QT_8bit_direct regression signature.
    jaccards = [
        len(a & b) / len(a | b)
        for a, b in zip(top10["int8"], top10["float32"], strict=False)
    ]
    assert sum(jaccards) / len(jaccards) >= 0.7, (
        f"int8 vs float32 top-10 Jaccard {jaccards} — quantizer may be "
        "misconfigured (QT_8bit_direct returns 0.0 sims)"
    )
