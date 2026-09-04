"""Hermetic acceptance tests for the public ``search_code`` MCP contract."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from mcp_server.code_search_mcp import CodeSearchMCP
from mcp_server.code_search_server import CodeSearchServer


def _payload(tool_result) -> dict:
    # mcp 2.x returns a CallToolResult; 1.x returned (content, structured).
    content = getattr(tool_result, "content", None)
    if content is None:
        content, _structured = tool_result
    return json.loads(content[0].text)


@pytest.mark.asyncio
async def test_search_code_rejects_blank_query_before_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    monkeypatch.setenv("RERANKER", "off")
    server = CodeSearchServer()
    server.get_searcher = Mock(side_effect=AssertionError("search executed"))
    mcp = CodeSearchMCP(server)

    payload = _payload(await mcp.call_tool("search_code", {"query": " \t\n"}))

    assert payload == {
        "error": {
            "code": "invalid_argument",
            "field": "query",
            "message": "query must contain non-whitespace text",
        }
    }
    server.get_searcher.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("k", "message"),
    [
        (0, "k must be between 1 and 100"),
        (-1, "k must be between 1 and 100"),
        (101, "k must be between 1 and 100"),
    ],
)
async def test_search_code_rejects_k_outside_public_limit_before_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    k: int,
    message: str,
) -> None:
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    monkeypatch.setenv("RERANKER", "off")
    server = CodeSearchServer()
    server.get_searcher = Mock(side_effect=AssertionError("search executed"))
    mcp = CodeSearchMCP(server)

    payload = _payload(
        await mcp.call_tool("search_code", {"query": "authentication", "k": k})
    )

    assert payload == {
        "error": {
            "code": "invalid_argument",
            "field": "k",
            "message": message,
        }
    }
    server.get_searcher.assert_not_called()


@pytest.mark.asyncio
async def test_search_code_rejects_unknown_mode_before_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    monkeypatch.setenv("RERANKER", "off")
    server = CodeSearchServer()
    server.get_searcher = Mock(side_effect=AssertionError("search executed"))
    mcp = CodeSearchMCP(server)

    payload = _payload(
        await mcp.call_tool(
            "search_code",
            {"query": "authentication", "search_mode": "structural"},
        )
    )

    assert payload == {
        "error": {
            "code": "invalid_argument",
            "field": "search_mode",
            "message": (
                "search_mode must be one of: auto, hybrid, keyword, semantic"
            ),
        }
    }
    server.get_searcher.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("file_pattern", ["   ", "src/\x00*.py"])
async def test_search_code_rejects_unusable_file_glob_before_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    file_pattern: str,
) -> None:
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    monkeypatch.setenv("RERANKER", "off")
    server = CodeSearchServer()
    server.get_searcher = Mock(side_effect=AssertionError("search executed"))
    mcp = CodeSearchMCP(server)

    payload = _payload(
        await mcp.call_tool(
            "search_code",
            {"query": "authentication", "file_pattern": file_pattern},
        )
    )

    assert payload == {
        "error": {
            "code": "invalid_argument",
            "field": "file_pattern",
            "message": "file_pattern must be a non-empty glob without NUL bytes",
        }
    }
    server.get_searcher.assert_not_called()
