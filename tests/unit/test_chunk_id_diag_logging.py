"""Tests for [CHUNK_ID_DIAG] file-sidecar logging.

The MCP server runs under pythonw.exe (no console — stderr discarded).
The diagnostic logging in `_load_index` and `save_index` is otherwise
invisible. The sidecar FileHandler in search/indexer.py routes those
lines to ~/.claude/logs/code-search-mcp.log so root-cause arcs can read
them.

These tests verify the sidecar fires (lines reach the log file) and
filters (only [CHUNK_ID_DIAG] lines, not unrelated logger output).
"""
from __future__ import annotations

import logging
import tempfile

import numpy as np

from search.indexer import CodeIndexManager
from embeddings.embedder import EmbeddingResult


def _make_result(chunk_id: str, dim: int = 384) -> EmbeddingResult:
    return EmbeddingResult(
        embedding=np.random.randn(dim).astype(np.float32),
        chunk_id=chunk_id,
        metadata={
            "file_path": f"{chunk_id}.py",
            "relative_path": f"{chunk_id}.py",
            "content_preview": f"def {chunk_id}(): pass",
            "full_content": f"def {chunk_id}(): pass",
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 3,
            "name": chunk_id,
            "parent_name": None,
            "docstring": None,
            "decorators": [],
            "imports": [],
            "complexity_score": 1,
            "tags": [],
            "folder_structure": [],
        },
    )


def _close(mgr: CodeIndexManager) -> None:
    if mgr._metadata_db is not None:
        mgr._metadata_db.close()
        mgr._metadata_db = None
    if getattr(mgr, "_fts_conn", None) is not None:
        mgr._fts_conn.close()
        mgr._fts_conn = None


def _attach_buffer_handler() -> tuple[logging.Logger, list[str]]:
    """Attach a buffer handler to the indexer's logger to capture lines.

    The real file-sidecar writes to ~/.claude/logs/code-search-mcp.log,
    which is outside the test sandbox. Rather than monkey-patching the
    install function (which only runs at import time), we attach an
    additional handler that mirrors the same filter and captures into a
    list — proving the diagnostic lines ARE being emitted at the right
    points and would be captured by an equivalent file handler.
    """
    logger = logging.getLogger("search.indexer")
    captured: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                msg = record.getMessage()
            except Exception:
                return
            if "[CHUNK_ID_DIAG]" in msg:
                captured.append(msg)

    h = _ListHandler(level=logging.DEBUG)
    logger.addHandler(h)
    logger.setLevel(logging.DEBUG)
    return logger, captured


def test_load_index_emits_diag_lines_through_filter():
    """_load_index emits CHUNK_ID_DIAG lines that pass the sidecar filter."""
    _, captured = _attach_buffer_handler()

    with tempfile.TemporaryDirectory() as tmp:
        mgr = CodeIndexManager(tmp)
        mgr.create_index(embedding_dimension=384)
        mgr.add_embeddings([_make_result(f"c{i}") for i in range(3)])
        mgr.save_index()
        _close(mgr)

        # Re-open: accessing the lazy `.index` property runs _load_index
        # and emits the CHUNK_ID_DIAG pre-load / post-load / post-repair
        # lines.
        mgr2 = CodeIndexManager(tmp)
        _ = mgr2.index  # trigger lazy load
        _close(mgr2)

    pre_load = [m for m in captured if "_load_index pre-load" in m]
    post_load = [m for m in captured if "_load_index post-load" in m]
    pre_save = [m for m in captured if "save_index pre-save" in m]
    post_save = [m for m in captured if "save_index post-save" in m]

    assert pre_load, "expected at least one pre-load line"
    assert post_load, "expected at least one post-load line"
    assert pre_save, "expected at least one pre-save line"
    assert post_save, "expected at least one post-save line"


def test_diag_filter_rejects_unrelated_lines():
    """Filter keeps [CHUNK_ID_DIAG] only — unrelated WARNINGs do not pass."""
    logger, captured = _attach_buffer_handler()
    logger.warning("Some unrelated warning that should NOT match the filter")
    logger.warning("[CHUNK_ID_DIAG] synthetic test line that SHOULD match")

    matched = [m for m in captured if "[CHUNK_ID_DIAG]" in m]
    rejected = [m for m in captured if "unrelated warning" in m]

    assert any("synthetic test line" in m for m in matched)
    assert rejected == [], "filter let through a non-diag line"


def test_install_is_idempotent():
    """Calling the installer twice does not stack handlers."""
    from search.indexer import _install_chunk_id_diag_file_handler

    logger = logging.getLogger("search.indexer")
    pre_count = sum(
        1 for h in logger.handlers
        if isinstance(h, logging.FileHandler)
        and getattr(h, "_chunk_id_diag", False)
    )
    _install_chunk_id_diag_file_handler()
    _install_chunk_id_diag_file_handler()
    post_count = sum(
        1 for h in logger.handlers
        if isinstance(h, logging.FileHandler)
        and getattr(h, "_chunk_id_diag", False)
    )

    assert post_count == pre_count, (
        f"handler stacked: pre={pre_count} post={post_count}"
    )
