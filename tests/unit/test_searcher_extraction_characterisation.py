"""CHARACTERISATION TESTS for the searcher policy extraction.

These tests pin the existing public import surface, default query expansion,
and hybrid-ranking output before policy code moves out of ``searcher.py``.
They document current behavior rather than proposing a ranking change.
"""

from __future__ import annotations

from typing import Any

import search.searcher as searcher_module
from search.config import get_search_config
from search.searcher import IntelligentSearcher, expand_code_query


class _StubEmbedder:
    def embed_query(self, _query: str) -> list[float]:
        return [0.0]


class _StubIndexManager:
    def __init__(self) -> None:
        self.bm25_query = ""
        self._metadata = {
            "vector_only": {
                "file_path": "/repo/docs/overview.md",
                "relative_path": "docs/overview.md",
                "chunk_type": "section",
                "name": "Overview",
            },
            "both_auth": {
                "file_path": "/repo/src/auth/handler.py",
                "relative_path": "src/auth/handler.py",
                "chunk_type": "function",
                "name": "authenticate_handler",
            },
            "both_session": {
                "file_path": "/repo/src/session.py",
                "relative_path": "src/session.py",
                "chunk_type": "class",
                "name": "Session",
            },
            "bm25_only": {
                "file_path": "/repo/src/auth.py",
                "relative_path": "src/auth.py",
                "chunk_type": "function",
                "name": "authorize",
            },
        }

    def search(
        self,
        _embedding: list[float],
        _k: int,
        _filters: dict[str, Any] | None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        return [
            ("vector_only", 0.99, self._metadata["vector_only"]),
            ("both_auth", 0.80, self._metadata["both_auth"]),
            ("both_session", 0.70, self._metadata["both_session"]),
        ]

    def search_bm25(
        self,
        query: str,
        *,
        k: int,
        filters: dict[str, Any] | None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        del k, filters
        self.bm25_query = query
        return [
            ("both_session", -1.0, self._metadata["both_session"]),
            ("both_auth", -2.0, self._metadata["both_auth"]),
            ("bm25_only", -3.0, self._metadata["bm25_only"]),
        ]


def test_characterises_default_query_expansion_order(monkeypatch):
    monkeypatch.delenv("CODE_SYNONYMS_PATH", raising=False)

    assert expand_code_query("auth retry") == (
        "auth retry authentication oauth jwt token credential login entra "
        "backoff retryable retry_delay 429 529"
    )


def test_characterises_default_hybrid_ranking(monkeypatch):
    for name in (
        "FUSION_K",
        "VECTOR_WEIGHT",
        "BM25_WEIGHT",
        "CONTENT_MODE",
        "CODE_SYNONYM_PROFILE",
        "QUERY_EXPANSION",
        "BM25_REWRITE",
        "CODE_SEARCH_PPR_ENABLED",
        "CHUNK_TYPE_BOOST_OVERRIDE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RERANKER", "off")
    get_search_config.cache_clear()

    manager = _StubIndexManager()
    searcher = IntelligentSearcher(manager, _StubEmbedder())
    results = searcher._hybrid_search("auth handler", k=3, context_depth=0)

    assert manager.bm25_query == (
        "auth handler authentication oauth jwt token credential login entra "
        "Route endpoint path Starlette"
    )
    assert [
        (result.chunk_id, round(result.similarity_score, 12))
        for result in results
    ] == [
        ("both_auth", 0.107545454545),
        ("both_session", 0.058405797101),
        ("bm25_only", 0.02275),
    ]


def test_characterises_legacy_searcher_exports():
    assert callable(searcher_module.reciprocal_rank_fusion)
    assert callable(searcher_module._query_stem)
    assert callable(searcher_module._active_synonyms)
    assert callable(searcher_module.expand_code_query)
    assert searcher_module.CODE_SYNONYMS["navigation"][0] == "internal-svc-62"
    assert searcher_module.SearchResult.__name__ == "SearchResult"


def test_characterises_monkey_patched_chunk_type_boosts(monkeypatch):
    for name in (
        "FUSION_K",
        "VECTOR_WEIGHT",
        "BM25_WEIGHT",
        "CONTENT_MODE",
        "CODE_SYNONYM_PROFILE",
        "QUERY_EXPANSION",
        "BM25_REWRITE",
        "CODE_SEARCH_PPR_ENABLED",
        "CHUNK_TYPE_BOOST_OVERRIDE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RERANKER", "off")
    monkeypatch.setattr(
        searcher_module,
        "CHUNK_TYPE_BOOSTS",
        {
            "code": {
                "section": 100.0,
                "function": 0.01,
                "class": 1.0,
            },
            "docs": {},
            "all": {},
        },
    )
    get_search_config.cache_clear()

    searcher = IntelligentSearcher(_StubIndexManager(), _StubEmbedder())
    results = searcher._hybrid_search("auth handler", k=3, context_depth=0)

    assert results[0].chunk_id == "vector_only"
