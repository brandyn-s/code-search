"""Production logging must be useful without exposing search text by default."""

from __future__ import annotations

import io
import json
import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


_FIRST_PARTY_LOGGERS = (
    "mcp_server",
    "search",
    "embeddings",
    "chunking",
    "merkle",
)


def _restore_loggers(
    states: dict[str, tuple[int, bool, list[logging.Handler]]],
) -> None:
    for name, (level, propagate, original_handlers) in states.items():
        package_logger = logging.getLogger(name)
        for handler in list(package_logger.handlers):
            if handler not in original_handlers:
                package_logger.removeHandler(handler)
                handler.close()
        package_logger.setLevel(level)
        package_logger.propagate = propagate


def test_query_log_value_is_redacted_and_fingerprinted_by_default(
    monkeypatch,
) -> None:
    from search.logging_privacy import format_query_for_log

    monkeypatch.delenv("CODE_SEARCH_LOG_QUERY_TEXT", raising=False)
    query = "sentinel private implementation detail"

    first = format_query_for_log(query)
    second = format_query_for_log(query)
    different = format_query_for_log(query + " changed")

    assert query not in first
    assert "redacted" in first
    assert f"length={len(query)}" in first
    assert "hmac_sha256=" in first
    assert first == second
    assert first != different


@pytest.mark.parametrize("value", ["off", "ON", "true", "1", " on "])
def test_query_log_value_requires_exact_on_opt_in(
    monkeypatch,
    value: str,
) -> None:
    from search.logging_privacy import format_query_for_log

    query = "sentinel private query"
    monkeypatch.setenv("CODE_SEARCH_LOG_QUERY_TEXT", value)

    assert query not in format_query_for_log(query)


def test_query_log_value_allows_explicit_plaintext_opt_in(monkeypatch) -> None:
    from search.logging_privacy import format_query_for_log

    query = "sentinel operator opted in"
    monkeypatch.setenv("CODE_SEARCH_LOG_QUERY_TEXT", "on")

    assert format_query_for_log(query) == query


def test_first_party_logging_defaults_to_info_without_mutating_root(
    monkeypatch,
) -> None:
    from mcp_server.logging_config import configure_first_party_logging

    monkeypatch.delenv("CODE_SEARCH_LOG_LEVEL", raising=False)
    root = logging.getLogger()
    original_level = root.level
    original_handlers = list(root.handlers)
    states = {
        name: (
            logging.getLogger(name).level,
            logging.getLogger(name).propagate,
            list(logging.getLogger(name).handlers),
        )
        for name in _FIRST_PARTY_LOGGERS
    }
    stream = io.StringIO()

    try:
        configured = configure_first_party_logging(stream=stream)

        assert configured == logging.INFO
        assert logging.getLogger("mcp_server").level == logging.INFO
        assert logging.getLogger("search").level == logging.INFO
        assert root.level == original_level
        assert root.handlers == original_handlers
    finally:
        _restore_loggers(states)


def test_first_party_logging_honors_debug_level(monkeypatch) -> None:
    from mcp_server.logging_config import configure_first_party_logging

    monkeypatch.setenv("CODE_SEARCH_LOG_LEVEL", "DEBUG")
    states = {
        name: (
            logging.getLogger(name).level,
            logging.getLogger(name).propagate,
            list(logging.getLogger(name).handlers),
        )
        for name in _FIRST_PARTY_LOGGERS
    }
    try:
        configured = configure_first_party_logging(stream=io.StringIO())

        assert configured == logging.DEBUG
        assert logging.getLogger("mcp_server").level == logging.DEBUG
    finally:
        _restore_loggers(states)


def test_server_query_log_is_redacted_without_changing_response(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    from common_utils import get_storage_dir
    from mcp_server.code_search_server import CodeSearchServer

    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    monkeypatch.setenv("CODE_SEARCH_QUERY_HISTORY", "off")
    monkeypatch.delenv("CODE_SEARCH_LOG_QUERY_TEXT", raising=False)
    get_storage_dir.cache_clear()
    server = CodeSearchServer()
    searcher = MagicMock()
    searcher.search.return_value = []
    searcher.index_manager.get_stats.return_value = {"total_chunks": 0}
    searcher._query_embedding_cache = {}
    searcher.last_reranker_metadata = {
        "applied": False,
        "reason": "not_invoked_no_candidates",
        "latency_ms": 0,
    }
    server.get_searcher = MagicMock(return_value=searcher)
    server._current_project = None
    query = "sentinel server private query"

    with caplog.at_level(logging.DEBUG, logger="mcp_server.code_search_server"):
        response = json.loads(
            server.search_code(
                query=query,
                auto_reindex=False,
            )
        )

    assert response["query"] == query
    assert query not in caplog.text
    assert "redacted" in caplog.text


def test_server_exception_log_omits_derived_query_text_by_default(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    from common_utils import get_storage_dir
    from mcp_server.code_search_server import CodeSearchServer

    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    monkeypatch.setenv("CODE_SEARCH_QUERY_HISTORY", "off")
    monkeypatch.delenv("CODE_SEARCH_LOG_QUERY_TEXT", raising=False)
    get_storage_dir.cache_clear()
    server = CodeSearchServer()
    searcher = MagicMock()
    searcher.index_manager.get_stats.return_value = {"total_chunks": 0}
    query = "sentinel derived failure"
    derived = query.upper().replace(" ", "_")
    searcher.search.side_effect = RuntimeError(
        f"backend rejected transformed terms {derived}"
    )
    server.get_searcher = MagicMock(return_value=searcher)
    server._current_project = None

    with caplog.at_level(logging.DEBUG, logger="mcp_server.code_search_server"):
        response = json.loads(
            server.search_code(
                query=query,
                auto_reindex=False,
            )
        )

    assert derived in response["error"]
    assert derived not in caplog.text
    assert "RuntimeError" in caplog.text


def test_rewrite_logs_redact_original_and_generated_text(
    monkeypatch,
    caplog,
) -> None:
    import search.query_rewriter as rewriter

    original = "sentinel original query"
    generated = "sentinel generated identifiers"
    monkeypatch.setenv("BM25_REWRITE", "on")
    monkeypatch.delenv("CODE_SEARCH_LOG_QUERY_TEXT", raising=False)
    monkeypatch.setattr(rewriter, "_call_haiku", lambda _query: generated)
    rewriter._rewrite_cache.clear()

    with caplog.at_level(logging.DEBUG, logger="search.query_rewriter"):
        result = rewriter.rewrite_query_for_bm25(original)

    assert result == generated
    assert original not in caplog.text
    assert generated not in caplog.text
    assert caplog.text.count("redacted") >= 2


def test_short_rewrite_logs_redact_all_query_text(
    monkeypatch,
    caplog,
) -> None:
    import search.query_rewriter as rewriter

    original = "sentinel phrase"
    alternatives = ["sentinelOne", "sentinel_two"]
    monkeypatch.setenv("SHORT_QUERY_REWRITE", "on")
    monkeypatch.delenv("CODE_SEARCH_LOG_QUERY_TEXT", raising=False)
    monkeypatch.setattr(
        rewriter,
        "_call_haiku_short_query",
        lambda _query, _count: alternatives,
    )
    rewriter._short_query_cache.clear()

    with caplog.at_level(logging.DEBUG, logger="search.query_rewriter"):
        result = rewriter.rewrite_short_natural_query(original, 2)

    assert result == alternatives
    assert original not in caplog.text
    assert all(value not in caplog.text for value in alternatives)
    assert caplog.text.count("redacted") >= 3


def test_fts_failure_log_redacts_sanitized_query(
    monkeypatch,
    caplog,
) -> None:
    from search.indexer import CodeIndexManager

    class FailingConnection:
        def execute(self, *_args, **_kwargs):
            raise RuntimeError("synthetic FTS failure")

        def close(self):
            return None
    query = "sentinel FTS private query"
    manager = CodeIndexManager.__new__(CodeIndexManager)
    manager._fts_conn = FailingConnection()
    manager._metadata_db = None
    manager._logger = logging.getLogger("search.indexer")
    monkeypatch.delenv("CODE_SEARCH_LOG_QUERY_TEXT", raising=False)

    with caplog.at_level(logging.DEBUG, logger="search.indexer"):
        result = manager.search_bm25(query)

    assert result == []
    assert query not in caplog.text
    assert "sentinel" not in caplog.text
    assert "redacted" in caplog.text


@pytest.mark.asyncio
async def test_sonnet_candidate_log_uses_keyed_query_fingerprint(
    monkeypatch,
    caplog,
) -> None:
    from search.logging_privacy import query_fingerprint
    import search.sonnet_reranker as reranker

    query = "sentinel reranker private query"

    class Messages:
        async def create(self, **_kwargs):
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="text", text='{"score": 7}')
                ]
            )

    client = SimpleNamespace(messages=Messages())
    monkeypatch.setenv("SONNET_RERANKER_LOG_PER_CANDIDATE_SCORE", "1")
    monkeypatch.delenv("CODE_SEARCH_LOG_QUERY_TEXT", raising=False)

    with caplog.at_level(logging.INFO, logger="search.sonnet_reranker"):
        score = await reranker._score_one(
            client,
            query,
            "src/example.py",
            "def example(): pass",
        )

    assert score == 7
    assert query not in caplog.text
    assert query_fingerprint(query)[:16] in caplog.text


def test_agentic_rerank_failure_log_omits_query_text_and_traceback_by_default(
    monkeypatch,
    caplog,
) -> None:
    from mcp_server.code_search_server import CodeSearchServer

    query = "sentinel agentic private query"
    derived = query.upper().replace(" ", "_")

    class Messages:
        def create(self, **_kwargs):
            raise RuntimeError(f"backend rejected transformed terms {derived}")

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(
            Anthropic=lambda: SimpleNamespace(messages=Messages()),
        ),
    )
    monkeypatch.delenv("CODE_SEARCH_LOG_QUERY_TEXT", raising=False)
    results = [{"relative_path": "example.py", "snippet": "example"}]
    server = CodeSearchServer.__new__(CodeSearchServer)

    with caplog.at_level(
        logging.WARNING,
        logger="mcp_server.code_search_server",
    ):
        reranked = server._agentic_rerank(query, results, 1)

    assert reranked == results
    assert derived not in caplog.text
    assert "RuntimeError" in caplog.text
    assert all(not record.exc_info for record in caplog.records)


@pytest.mark.parametrize("exception_type", [RuntimeError, ValueError])
def test_sonnet_final_fallback_log_omits_query_text_and_traceback_by_default(
    monkeypatch,
    caplog,
    exception_type,
) -> None:
    import search.sonnet_reranker as reranker

    query = "sentinel sonnet private query"
    derived = query.upper().replace(" ", "_")

    async def fail_rerank(*_args, **_kwargs):
        raise exception_type(
            f"backend rejected transformed terms {derived}"
        )

    monkeypatch.setattr(reranker, "_rerank_async", fail_rerank)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-test-key")
    monkeypatch.delenv("CODE_SEARCH_LOG_QUERY_TEXT", raising=False)
    candidates = [{"relative_path": "example.py", "snippet": "example"}]

    with caplog.at_level(
        logging.WARNING,
        logger="search.sonnet_reranker",
    ):
        result = reranker.rerank_with_sonnet(
            query,
            candidates,
            top_k=1,
        )

    assert result == candidates
    assert derived not in caplog.text
    assert exception_type.__name__ in caplog.text
    assert all(not record.exc_info for record in caplog.records)


@pytest.mark.parametrize(
    ("failure_mode", "expected_reason"),
    [
        ("echoing_response", "parse_failed"),
        ("echoing_exception", "unexpected_error"),
    ],
)
def test_listwise_failure_logs_omit_query_text_and_traceback_by_default(
    monkeypatch,
    caplog,
    failure_mode,
    expected_reason,
) -> None:
    import search.listwise_sonnet_reranker as reranker

    query = "sentinel listwise private query"
    derived = query.upper().replace(" ", "_")

    class Messages:
        def create(self, **_kwargs):
            if failure_mode == "echoing_exception":
                raise RuntimeError(
                    f"backend rejected transformed terms {derived}"
                )
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        text=f"Looking at the query {derived}, no JSON applies."
                    )
                ]
            )

    monkeypatch.delenv("CODE_SEARCH_LOG_QUERY_TEXT", raising=False)
    candidates = [{"relative_path": "example.py", "snippet": "example"}]

    with caplog.at_level(
        logging.WARNING,
        logger="search.listwise_sonnet_reranker",
    ):
        result, metadata = reranker.listwise_rerank_with_sonnet(
            query,
            candidates,
            top_k=1,
            return_metadata=True,
            _client_factory=lambda: SimpleNamespace(messages=Messages()),
        )

    assert result == candidates
    assert metadata["reason"] == expected_reason
    assert query not in caplog.text
    assert derived not in caplog.text
    assert all(not record.exc_info for record in caplog.records)
