"""Tests for FTS5 BM25 index in CodeIndexManager."""
import tempfile
import numpy as np
import pytest
from search.indexer import CodeIndexManager
from embeddings.embedder import EmbeddingResult


def _make_result(chunk_id: str, content: str, file_path: str = "test.py") -> EmbeddingResult:
    """Helper to create an EmbeddingResult with FTS-relevant metadata."""
    return EmbeddingResult(
        embedding=np.random.randn(384).astype(np.float32),
        chunk_id=chunk_id,
        metadata={
            "file_path": file_path,
            "relative_path": file_path,
            "content_preview": content,
            "full_content": content,
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 10,
            "name": chunk_id.split(":")[-1] if ":" in chunk_id else None,
            "parent_name": None,
            "docstring": None,
            "decorators": [],
            "imports": [],
            "complexity_score": 1,
            "tags": [],
            "folder_structure": [],
        },
    )


def _close_manager(mgr):
    """Close all database connections held by the manager."""
    if mgr._metadata_db is not None:
        mgr._metadata_db.close()
        mgr._metadata_db = None
    if hasattr(mgr, "_fts_conn") and mgr._fts_conn is not None:
        mgr._fts_conn.close()
        mgr._fts_conn = None


def test_fts5_search_finds_keyword_match():
    """FTS5 should find chunks containing query keywords."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CodeIndexManager(tmpdir)
        mgr.add_embeddings([
            _make_result("a.py:1-10:func:get_redis", "def get_redis(): return redis.Redis(host='localhost')"),
            _make_result("b.py:1-10:func:get_db", "def get_db(): return sqlite3.connect('data.db')"),
            _make_result("c.py:1-10:func:health_check", "def health_check(): return {'status': 'ok'}"),
        ])

        results = mgr.search_bm25("redis", k=5)
        assert len(results) >= 1
        assert any("redis" in r[0].lower() or "redis" in r[2].get("content_preview", "").lower() for r in results)
        _close_manager(mgr)


def test_fts5_search_returns_empty_on_no_match():
    """FTS5 should return empty list when no keywords match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CodeIndexManager(tmpdir)
        mgr.add_embeddings([
            _make_result("a.py:1-10:func:foo", "def foo(): return 42"),
        ])

        results = mgr.search_bm25("nonexistent_keyword_xyz", k=5)
        assert results == []
        _close_manager(mgr)


def test_fts5_cleared_on_clear_index():
    """clear_index should also clear FTS5 data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CodeIndexManager(tmpdir)
        mgr.add_embeddings([
            _make_result("a.py:1-10:func:get_redis", "def get_redis(): return redis.Redis()"),
        ])
        assert len(mgr.search_bm25("redis", k=5)) >= 1

        mgr.clear_index()
        _close_manager(mgr)

        # Re-initialize after clear
        mgr2 = CodeIndexManager(tmpdir)
        assert mgr2.search_bm25("redis", k=5) == []
        _close_manager(mgr2)


def test_fts5_finds_keyword_deep_in_content():
    """FTS5 should find keywords that appear past the 200-char preview cutoff."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CodeIndexManager(tmpdir)
        # Content with "authentication" at position 300+
        deep_content = "def setup():\n" + "    x = 1\n" * 30 + "    # authentication logic here\n"
        mgr.add_embeddings([
            _make_result("a.py:1-35:func:setup", deep_content),
        ])

        results = mgr.search_bm25("authentication", k=5)
        assert len(results) >= 1
        assert results[0][0] == "a.py:1-35:func:setup"
        _close_manager(mgr)


def test_fts5_name_match_ranks_higher():
    """A chunk whose name matches the query should rank above one where only content matches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CodeIndexManager(tmpdir)
        mgr.add_embeddings([
            _make_result("a.py:1-10:func:unrelated", "def unrelated(): redis_client = redis.Redis()"),
            _make_result("b.py:1-10:func:get_redis", "def get_redis(): return client"),
        ])

        results = mgr.search_bm25("redis", k=5)
        assert len(results) >= 2
        # get_redis (name match) should rank above unrelated (content-only match)
        ids = [r[0] for r in results]
        assert ids.index("b.py:1-10:func:get_redis") < ids.index("a.py:1-10:func:unrelated")
        _close_manager(mgr)
