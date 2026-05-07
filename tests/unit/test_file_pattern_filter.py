"""Tests for file_pattern glob filtering.

Regression: prior to 2026-05-07, `_matches_filters` used substring match
(`pattern in path`) for the `file_pattern` filter, so `*.rs` was never a
substring of any real path and silently filtered out everything. The
hybrid search path masked this by returning unfiltered BM25 results,
which surfaced through the cracks (e.g., `Cargo.nix` hit on
`file_pattern="*.rs"`). Both bugs fixed together: glob via fnmatch +
filters threaded through search_bm25.
"""

from __future__ import annotations

from search.indexer import CodeIndexManager


def _stub_indexer() -> CodeIndexManager:
    """Build a CodeIndexManager-like object that exposes only _matches_filters
    so we don't need a populated FAISS index for filter logic tests."""

    class _Stub:
        # Borrow the unbound method off the class
        _matches_filters = CodeIndexManager._matches_filters

    return _Stub()  # type: ignore[return-value]


def test_file_pattern_glob_basename_match():
    idx = _stub_indexer()
    metadata = {"relative_path": "src/foo.rs"}
    assert idx._matches_filters(metadata, {"file_pattern": ["*.rs"]}) is True


def test_file_pattern_glob_basename_no_match():
    idx = _stub_indexer()
    metadata = {"relative_path": "Cargo.nix"}
    assert idx._matches_filters(metadata, {"file_pattern": ["*.rs"]}) is False


def test_file_pattern_full_path_glob():
    idx = _stub_indexer()
    metadata = {"relative_path": "internal/extractor/foo.go"}
    assert idx._matches_filters(metadata, {"file_pattern": ["internal/*"]}) is True


def test_file_pattern_multiple_patterns_any_match():
    idx = _stub_indexer()
    metadata = {"relative_path": "tests/conftest.py"}
    assert idx._matches_filters(metadata, {"file_pattern": ["*.rs", "*.py"]}) is True


def test_file_pattern_no_pattern_matches_returns_false():
    idx = _stub_indexer()
    metadata = {"relative_path": "src/foo.go"}
    assert idx._matches_filters(metadata, {"file_pattern": ["*.rs", "*.py"]}) is False


def test_file_pattern_empty_relative_path():
    idx = _stub_indexer()
    metadata = {"relative_path": ""}
    # No path → no pattern can match
    assert idx._matches_filters(metadata, {"file_pattern": ["*.rs"]}) is False


def test_file_pattern_windows_backslash_path():
    """The MCP server passes Windows paths with backslashes; basename
    extraction must handle both separators."""
    idx = _stub_indexer()
    metadata = {"relative_path": r"internal\extractor\foo.rs"}
    assert idx._matches_filters(metadata, {"file_pattern": ["*.rs"]}) is True


def test_chunk_type_filter_still_works():
    """Don't regress existing chunk_type filtering."""
    idx = _stub_indexer()
    metadata = {"chunk_type": "function"}
    assert idx._matches_filters(metadata, {"chunk_type": "function"}) is True
    assert idx._matches_filters(metadata, {"chunk_type": "class"}) is False
