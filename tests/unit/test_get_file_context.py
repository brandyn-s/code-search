"""Tests for the get_file_context MCP tool."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mcp_server.code_search_server import CodeSearchServer  # noqa: E402


class _StubMetadataDB:
    """In-memory stand-in for SqliteDict-backed metadata_db."""

    def __init__(self, entries):
        self._entries = entries

    def get(self, chunk_id):
        return self._entries.get(chunk_id)


class _StubIndexManager:
    """Minimal IndexManager surface that get_file_context uses."""

    def __init__(self, chunks):
        self._chunk_ids = [c["chunk_id"] for c in chunks]
        self.metadata_db = _StubMetadataDB({
            c["chunk_id"]: {"index_id": i, "metadata": c["metadata"]}
            for i, c in enumerate(chunks)
        })


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    from common_utils import get_storage_dir
    get_storage_dir.cache_clear()
    return CodeSearchServer()


def _stub_manager(server, chunks):
    """Inject a stub index manager into the server."""
    mgr = _StubIndexManager(chunks)
    server.get_index_manager = lambda: mgr  # type: ignore[method-assign]
    return mgr


def test_returns_all_chunks_for_matching_file(server):
    _stub_manager(server, [
        {"chunk_id": "c1", "metadata": {
            "relative_path": "src/auth.py", "start_line": 1, "end_line": 20,
            "chunk_type": "function", "name": "login",
            "full_content": "def login(...):\n    ..."}},
        {"chunk_id": "c2", "metadata": {
            "relative_path": "src/auth.py", "start_line": 22, "end_line": 40,
            "chunk_type": "function", "name": "logout",
            "full_content": "def logout(...):\n    ..."}},
        {"chunk_id": "c3", "metadata": {
            "relative_path": "src/db.py", "start_line": 1, "end_line": 50,
            "chunk_type": "module", "name": "db"}},
    ])

    out = json.loads(server.get_file_context("src/auth.py"))
    assert out["total_chunks_in_file"] == 2
    assert out["matched_path"] == "src/auth.py"
    assert [c["chunk_id"] for c in out["chunks"]] == ["c1", "c2"]


def test_filters_by_line_range_overlap(server):
    _stub_manager(server, [
        {"chunk_id": "c1", "metadata": {
            "relative_path": "x.py", "start_line": 1, "end_line": 10,
            "name": "a"}},
        {"chunk_id": "c2", "metadata": {
            "relative_path": "x.py", "start_line": 15, "end_line": 25,
            "name": "b"}},
        {"chunk_id": "c3", "metadata": {
            "relative_path": "x.py", "start_line": 30, "end_line": 40,
            "name": "c"}},
    ])

    # Range 12-20 overlaps only c2 (15-25)
    out = json.loads(server.get_file_context("x.py", line_range="12-20"))
    assert out["total_chunks_in_file"] == 1
    assert out["chunks"][0]["chunk_id"] == "c2"


def test_line_range_inclusive_at_boundaries(server):
    _stub_manager(server, [
        {"chunk_id": "c1", "metadata": {
            "relative_path": "x.py", "start_line": 10, "end_line": 20,
            "name": "a"}},
    ])

    # Chunk ends at 20; range starts at 20 → overlap (1 line, inclusive)
    out = json.loads(server.get_file_context("x.py", line_range="20-30"))
    assert out["total_chunks_in_file"] == 1

    # Chunk starts at 10; range ends at 10 → overlap
    out = json.loads(server.get_file_context("x.py", line_range="5-10"))
    assert out["total_chunks_in_file"] == 1

    # No overlap (chunk 10-20, range 21-30)
    out = json.loads(server.get_file_context("x.py", line_range="21-30"))
    assert out["total_chunks_in_file"] == 0


def test_max_chunks_truncates(server):
    _stub_manager(server, [
        {"chunk_id": f"c{i}", "metadata": {
            "relative_path": "big.py", "start_line": i * 10,
            "end_line": i * 10 + 5, "name": f"fn{i}"}}
        for i in range(10)
    ])

    out = json.loads(server.get_file_context("big.py", max_chunks=3))
    assert out["total_chunks_in_file"] == 10
    assert out["chunks_returned"] == 3
    assert out["truncated"] is True


def test_invalid_line_range_returns_error(server):
    _stub_manager(server, [])

    out = json.loads(server.get_file_context("x.py", line_range="not-a-range"))
    assert "error" in out

    out = json.loads(server.get_file_context("x.py", line_range="20-10"))
    assert "error" in out  # start > end


def test_suffix_match_for_absolute_path(server):
    _stub_manager(server, [
        {"chunk_id": "c1", "metadata": {
            "relative_path": "src/auth.py", "start_line": 1, "end_line": 5,
            "name": "login"}},
    ])

    # User passes absolute path; index has relative path
    out = json.loads(server.get_file_context(
        "C:/repo/src/auth.py"))
    assert out["total_chunks_in_file"] == 1
    assert out["matched_path"] == "src/auth.py"


def test_unmatched_path_returns_hint(server):
    _stub_manager(server, [
        {"chunk_id": "c1", "metadata": {
            "relative_path": "src/auth.py", "name": "x"}},
    ])

    out = json.loads(server.get_file_context("nonexistent.py"))
    assert out["total_chunks_in_file"] == 0
    assert out["matched_path"] is None
    assert "hint" in out


def test_chunks_sorted_by_start_line(server):
    _stub_manager(server, [
        {"chunk_id": "c_last", "metadata": {
            "relative_path": "x.py", "start_line": 100,
            "end_line": 110, "name": "last"}},
        {"chunk_id": "c_first", "metadata": {
            "relative_path": "x.py", "start_line": 1,
            "end_line": 10, "name": "first"}},
        {"chunk_id": "c_mid", "metadata": {
            "relative_path": "x.py", "start_line": 50,
            "end_line": 60, "name": "mid"}},
    ])

    out = json.loads(server.get_file_context("x.py"))
    assert [c["chunk_id"] for c in out["chunks"]] == [
        "c_first", "c_mid", "c_last",
    ]
