"""Characterize the hybrid pipeline before its module extraction."""

from __future__ import annotations

import builtins

import pytest

import search.searcher as searcher_module
from search.config import get_search_config
from search.searcher import IntelligentSearcher


@pytest.fixture(autouse=True)
def _clear_search_config_after_test():
    yield
    get_search_config.cache_clear()


class _RecordingEmbedder:
    def __init__(self, events):
        self.events = events

    def embed_query(self, query):
        self.events.append(("embed", query))
        return [0.0]


class _RecordingIndexManager:
    def __init__(self, events):
        self.events = events
        self.bm25_query = None
        self.metadata = {
            chunk_id: {
                "file_path": f"/repo/{chunk_id}.py",
                "relative_path": f"{chunk_id}.py",
                "chunk_type": "unknown",
                "name": chunk_id,
            }
            for chunk_id in ("vector", "both", "keyword")
        }

    def search(self, _embedding, k, filters):
        self.events.append(("vector", k, filters))
        return [
            ("vector", 0.9, self.metadata["vector"]),
            ("both", 0.8, self.metadata["both"]),
        ]

    def search_bm25(self, query, *, k, filters):
        self.events.append(("bm25", query, k, filters))
        self.bm25_query = query
        return [
            ("both", -1.0, self.metadata["both"]),
            ("keyword", -2.0, self.metadata["keyword"]),
        ]


def _set_pipeline_defaults(monkeypatch):
    for name in (
        "FUSION_K",
        "VECTOR_WEIGHT",
        "BM25_WEIGHT",
        "CONTENT_MODE",
        "CHUNK_TYPE_BOOST_OVERRIDE",
        "SONNET_RERANKER_SKIP_THRESHOLD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BM25_REWRITE", "on")
    monkeypatch.setenv("QUERY_EXPANSION", "on")
    monkeypatch.setenv("RERANKER", "off")
    monkeypatch.setenv("CODE_SEARCH_PPR_ENABLED", "off")
    get_search_config.cache_clear()


def test_hybrid_pipeline_preserves_call_order_scores_and_metadata(monkeypatch):
    _set_pipeline_defaults(monkeypatch)
    events = []
    manager = _RecordingIndexManager(events)
    searcher = IntelligentSearcher(manager, _RecordingEmbedder(events))
    raw_query = "  raw private query  "

    def optimize(query):
        events.append(("optimize", query))
        return query.strip()

    def rewrite(query):
        events.append(("rewrite", query))
        return f"rewritten::{query}"

    def expand(query):
        events.append(("expand", query))
        return f"expanded::{query}"

    monkeypatch.setattr(searcher, "_optimize_query", optimize)
    monkeypatch.setattr(
        "search.query_rewriter.rewrite_query_for_bm25",
        rewrite,
    )
    monkeypatch.setattr(searcher_module, "expand_code_query", expand)

    results = searcher._hybrid_search(
        raw_query,
        k=3,
        context_depth=0,
        filters={"language": "python"},
    )

    assert events == [
        ("optimize", raw_query),
        ("embed", raw_query.strip()),
        ("vector", 50, {"language": "python"}),
        ("rewrite", raw_query),
        ("expand", f"rewritten::{raw_query}"),
        (
            "bm25",
            f"expanded::rewritten::{raw_query}",
            50,
            {"language": "python"},
        ),
    ]
    assert [
        (result.chunk_id, result.similarity_score)
        for result in results
    ] == [
        ("both", 0.04621212121212121),
        ("vector", 0.030952380952380953),
        ("keyword", 0.015909090909090907),
    ]
    assert searcher.last_ppr_metadata == {
        "applied": False,
        "reason": "disabled_by_env",
        "latency_ms": 0,
    }
    assert searcher.last_reranker_metadata == {
        "applied": False,
        "reason": "disabled_by_env",
        "latency_ms": 0,
    }


def test_explicit_code_signals_widen_and_lexically_weight_retrieval(monkeypatch):
    _set_pipeline_defaults(monkeypatch)
    monkeypatch.setenv("BM25_REWRITE", "off")
    monkeypatch.setenv("QUERY_EXPANSION", "off")
    get_search_config.cache_clear()
    events = []
    manager = _RecordingIndexManager(events)
    searcher = IntelligentSearcher(manager, _RecordingEmbedder(events))

    results = searcher._hybrid_search(
        "Flow.validate_parameters regression",
        k=50,
        context_depth=0,
    )

    assert ("vector", 200, None) in events
    assert (
        "bm25",
        "Flow validate_parameters regression",
        200,
        None,
    ) in events
    assert [result.chunk_id for result in results] == [
        "both",
        "keyword",
        "vector",
    ]


def test_off_mode_keeps_optional_reranker_imports_lazy(monkeypatch):
    _set_pipeline_defaults(monkeypatch)
    guarded_modules = {
        "search.sonnet_reranker",
        "search.listwise_sonnet_reranker",
        "search.reranker",
    }
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in guarded_modules:
            raise AssertionError(f"optional reranker imported in off mode: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    searcher = IntelligentSearcher(
        _RecordingIndexManager([]),
        _RecordingEmbedder([]),
    )

    assert len(
        searcher._hybrid_search("query", k=3, context_depth=0)
    ) == 3


def test_hybrid_pipeline_has_explicit_retrieval_and_post_retrieval_boundaries():
    from search.pipeline import run_hybrid_pipeline
    from search.retrieval import HybridRetrieval, retrieve_hybrid_candidates

    assert callable(retrieve_hybrid_candidates)
    assert callable(run_hybrid_pipeline)
    assert HybridRetrieval.__name__ == "HybridRetrieval"


def test_listwise_pipeline_preserves_payload_metadata_and_result_order(
    monkeypatch,
):
    _set_pipeline_defaults(monkeypatch)
    monkeypatch.setenv("BM25_REWRITE", "off")
    monkeypatch.setenv("QUERY_EXPANSION", "off")
    monkeypatch.setenv("RERANKER", "listwise")
    monkeypatch.setenv("SONNET_LISTWISE_TIMEOUT", "7.5")
    get_search_config.cache_clear()
    observed = {}
    rerank_metadata = {
        "applied": True,
        "reason": "ok",
        "latency_ms": 12,
    }

    def listwise(query, items, *, top_k, timeout, return_metadata):
        observed.update(
            query=query,
            items=items,
            top_k=top_k,
            timeout=timeout,
            return_metadata=return_metadata,
        )
        return [items[-1], items[0]], rerank_metadata

    monkeypatch.setattr(
        "search.listwise_sonnet_reranker.listwise_rerank_with_sonnet",
        listwise,
    )
    searcher = IntelligentSearcher(
        _RecordingIndexManager([]),
        _RecordingEmbedder([]),
    )

    results = searcher._hybrid_search("query", k=1, context_depth=0)

    assert observed["query"] == "query"
    assert observed["top_k"] == 1
    assert observed["timeout"] == 7.5
    assert observed["return_metadata"] is True
    assert len(observed["items"]) == 3
    assert set(observed["items"][0]) == {
        "chunk_id",
        "file_path",
        "name",
        "parent_name",
        "chunk_type",
        "start_line",
        "end_line",
        "content_preview",
        "similarity_score",
        "_orig",
    }
    assert [result.chunk_id for result in results] == ["keyword"]
    assert searcher.last_reranker_metadata is rerank_metadata
