"""Tests for R1 + R2 + R3: search input validation.

These pin three distinct operator-facing failure modes that were either
hard-crashing or silently misbehaving on bad input. Each is verified at
HEAD with a reproduction before the fix shipped:

- R1: `file_pattern=None` → TypeError; `chunk_type=None` → silent
  filter-all. Both come through the MCP `search_code` surface from
  external callers.
- R2: `k=0` or `k=-1` → FAISS AssertionError with no context.
- R3: `VECTOR_WEIGHT=abc`, `FUSION_K=abc`, etc. → ValueError crash in
  _hybrid_search. Operator env-var misconfig.
"""
from __future__ import annotations

import logging
import tempfile

import numpy as np
import pytest

from embeddings.embedder import EmbeddingResult
from search.indexer import CodeIndexManager
from search.searcher import _parse_env_int, _parse_env_float


def _mk_result(cid: str, dim: int = 8) -> EmbeddingResult:
    return EmbeddingResult(
        embedding=np.random.RandomState(hash(cid) & 0xFFFFFFFF)
            .randn(dim).astype(np.float32),
        chunk_id=cid,
        metadata={
            "file_path": f"{cid}.py", "relative_path": f"{cid}.py",
            "content_preview": "x", "full_content": "x",
            "chunk_type": "function", "start_line": 1, "end_line": 2,
            "name": cid, "parent_name": None, "docstring": None,
            "decorators": [], "imports": [], "complexity_score": 0,
            "tags": [], "folder_structure": [],
        },
    )


def _close(mgr: CodeIndexManager) -> None:
    if mgr._metadata_db is not None:
        mgr._metadata_db.close()
        mgr._metadata_db = None
    if getattr(mgr, "_fts_conn", None) is not None:
        mgr._fts_conn.close()
        mgr._fts_conn = None


# ---------------------------------------------------------------------------
# R1 — filter input validation
# ---------------------------------------------------------------------------

class TestFilterInputValidation:
    """`_matches_filters` must handle None values + non-list patterns
    without crashing or silently rejecting all results."""

    def test_file_pattern_none_skips_filter_not_crashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            # Pre-R1: this raised TypeError ('NoneType' not iterable).
            result = mgr._matches_filters(
                {"relative_path": "src/foo.py"},
                {"file_pattern": None},
            )
            assert result is True, (
                "file_pattern=None should be treated as 'filter absent', not crash"
            )
            _close(mgr)

    def test_chunk_type_none_skips_filter_not_rejects_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            # Pre-R1: returned False (silent filter-all) because
            # `metadata['chunk_type'] != None` is True.
            result = mgr._matches_filters(
                {"chunk_type": "function"},
                {"chunk_type": None},
            )
            assert result is True, (
                "chunk_type=None should be treated as 'filter absent', "
                "not silently reject every chunk"
            )
            _close(mgr)

    def test_file_pattern_string_normalized_to_list(self):
        """A single pattern string (not wrapped in a list) must work — pre-R1
        the for-loop iterated chars of the string, behavior depending on
        whether any char happened to be a valid fnmatch pattern."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            assert mgr._matches_filters(
                {"relative_path": "src/foo.py"},
                {"file_pattern": "*.py"},
            ) is True
            assert mgr._matches_filters(
                {"relative_path": "src/foo.rs"},
                {"file_pattern": "*.py"},
            ) is False
            _close(mgr)

    def test_file_pattern_list_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            assert mgr._matches_filters(
                {"relative_path": "src/foo.py"},
                {"file_pattern": ["*.rs", "*.py"]},
            ) is True
            _close(mgr)


# ---------------------------------------------------------------------------
# R2 — k validation
# ---------------------------------------------------------------------------

class TestKValidation:
    """indexer.search must reject k<=0 with a clean ValueError rather than
    letting FAISS raise an AssertionError with no context."""

    def test_k_zero_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            mgr.add_embeddings([_mk_result("a"), _mk_result("b")])
            qvec = np.random.randn(8).astype(np.float32)
            with pytest.raises(ValueError, match="k must be a positive integer"):
                mgr.search(qvec, k=0)
            _close(mgr)

    def test_k_negative_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            mgr.add_embeddings([_mk_result("a")])
            qvec = np.random.randn(8).astype(np.float32)
            with pytest.raises(ValueError, match="k must be a positive integer"):
                mgr.search(qvec, k=-1)
            _close(mgr)

    def test_k_non_int_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            mgr.add_embeddings([_mk_result("a")])
            qvec = np.random.randn(8).astype(np.float32)
            with pytest.raises(ValueError, match="k must be a positive integer"):
                mgr.search(qvec, k=1.5)  # type: ignore[arg-type]
            _close(mgr)

    def test_k_larger_than_index_still_works(self):
        """Pre-existing graceful behavior — k > ntotal is fine; FAISS
        returns whatever's available. Regression-pin this so the validation
        doesn't accidentally tighten the upper bound."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CodeIndexManager(tmp)
            mgr.add_embeddings([_mk_result(f"f{i}") for i in range(3)])
            qvec = np.random.randn(8).astype(np.float32)
            results = mgr.search(qvec, k=100_000)
            assert len(results) <= 3  # capped at index size; no crash
            _close(mgr)


# ---------------------------------------------------------------------------
# R3 — env var parsing
# ---------------------------------------------------------------------------

class TestEnvVarParsing:
    """Malformed env vars must log a warning and fall back to defaults,
    not crash the search path with a ValueError."""

    def test_parse_env_int_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("__TEST_INT", raising=False)
        assert _parse_env_int("__TEST_INT", default=42) == 42

    def test_parse_env_int_valid_returns_value(self, monkeypatch):
        monkeypatch.setenv("__TEST_INT", "7")
        assert _parse_env_int("__TEST_INT", default=42) == 7

    def test_parse_env_int_garbage_logs_and_defaults(self, monkeypatch, caplog):
        monkeypatch.setenv("__TEST_INT", "abc")
        logger = logging.getLogger("test")
        with caplog.at_level(logging.WARNING, logger="test"):
            val = _parse_env_int("__TEST_INT", default=42, logger=logger)
        assert val == 42
        assert any("[CONFIG]" in r.getMessage() and "__TEST_INT" in r.getMessage()
                   for r in caplog.records)

    def test_parse_env_int_below_min_logs_and_defaults(self, monkeypatch, caplog):
        monkeypatch.setenv("__TEST_INT", "-5")
        logger = logging.getLogger("test")
        with caplog.at_level(logging.WARNING, logger="test"):
            val = _parse_env_int(
                "__TEST_INT", default=20, min_value=1, logger=logger,
            )
        assert val == 20
        assert any("below min_value" in r.getMessage() for r in caplog.records)

    def test_parse_env_float_garbage_logs_and_defaults(self, monkeypatch, caplog):
        monkeypatch.setenv("__TEST_FLOAT", "not_a_float")
        logger = logging.getLogger("test")
        with caplog.at_level(logging.WARNING, logger="test"):
            val = _parse_env_float(
                "__TEST_FLOAT", default=0.5, logger=logger,
            )
        assert val == 0.5
        assert any("not a valid float" in r.getMessage() for r in caplog.records)

    def test_parse_env_float_valid_returns_value(self, monkeypatch):
        monkeypatch.setenv("__TEST_FLOAT", "0.75")
        assert _parse_env_float("__TEST_FLOAT", default=0.0) == 0.75

    def test_parse_env_float_below_min_logs_and_defaults(self, monkeypatch, caplog):
        monkeypatch.setenv("__TEST_FLOAT", "-0.5")
        logger = logging.getLogger("test")
        with caplog.at_level(logging.WARNING, logger="test"):
            val = _parse_env_float(
                "__TEST_FLOAT", default=0.5, min_value=0.0, logger=logger,
            )
        assert val == 0.5
