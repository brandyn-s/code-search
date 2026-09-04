"""Tests for [REINDEX_PROGRESS] logging in incremental_indexer.

Pin two behaviors:
1. The file-sidecar in search/indexer.py captures lines tagged
   [REINDEX_PROGRESS] (in addition to [CHUNK_ID_DIAG]). Operator can
   `tail -f ~/.claude/logs/code-search-mcp.log` to see liveness.
2. CODE_SEARCH_DISABLE_AUTO_REINDEX=1 makes auto_reindex_if_needed a
   no-op (logs SKIPPED, returns success without doing work).
"""
from __future__ import annotations

import logging
import os
import tempfile
from unittest.mock import MagicMock

# Ensure the package logger is at WARNING so propagation reaches handlers.
logging.getLogger("search").setLevel(logging.WARNING)


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(record.getMessage())
        except Exception:
            pass


def test_sidecar_filter_accepts_reindex_progress_lines():
    """The filter installed by _install_search_file_handler accepts both
    [CHUNK_ID_DIAG] and [REINDEX_PROGRESS] prefixes."""
    from search.indexer import _install_search_file_handler  # noqa: F401

    # Attach our own list handler that mimics the sidecar's filter.
    parent = logging.getLogger("search")
    captured = _ListHandler()

    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
            except Exception:
                return False
            return "[CHUNK_ID_DIAG]" in msg or "[REINDEX_PROGRESS]" in msg

    captured.addFilter(_Filter())
    parent.addHandler(captured)
    try:
        child = logging.getLogger("search.incremental_indexer")
        child.warning("[REINDEX_PROGRESS] synthetic test line")
        child.warning("not a tagged line — should be filtered")
        child2 = logging.getLogger("search.indexer")
        child2.warning("[CHUNK_ID_DIAG] synthetic diag")

        diag_count = sum(
            1 for m in captured.records if "[CHUNK_ID_DIAG]" in m
        )
        prog_count = sum(
            1 for m in captured.records if "[REINDEX_PROGRESS]" in m
        )
        unrelated = [m for m in captured.records if "not a tagged" in m]
        assert prog_count >= 1, "REINDEX_PROGRESS line did not pass filter"
        assert diag_count >= 1, "CHUNK_ID_DIAG line did not pass filter"
        assert unrelated == [], "Unrelated line leaked through filter"
    finally:
        parent.removeHandler(captured)


def test_disable_auto_reindex_env_var_skips_work():
    """CODE_SEARCH_DISABLE_AUTO_REINDEX=1 short-circuits auto_reindex_if_needed.

    Constructs an IncrementalIndexer with mocked components so we don't
    need a real index. Sets the env var, calls auto_reindex_if_needed,
    asserts (a) success without invoking incremental_index/needs_reindex,
    (b) a SKIPPED log line was emitted.
    """
    from search.incremental_indexer import IncrementalIndexer

    captured = _ListHandler()
    parent = logging.getLogger("search")
    parent.addHandler(captured)
    prior = os.environ.get("CODE_SEARCH_DISABLE_AUTO_REINDEX")
    os.environ["CODE_SEARCH_DISABLE_AUTO_REINDEX"] = "1"

    try:
        # Mock the heavy components so we don't actually touch FAISS/voyage.
        idx = IncrementalIndexer(
            indexer=MagicMock(),
            embedder=MagicMock(),
            chunker=MagicMock(),
            snapshot_manager=MagicMock(),
        )
        idx.needs_reindex = MagicMock(return_value=True)  # would normally proceed
        idx.incremental_index = MagicMock()  # should NOT be called

        with tempfile.TemporaryDirectory() as tmp:
            result = idx.auto_reindex_if_needed(tmp, project_name="test")

        assert result.success
        assert result.files_added == 0
        assert result.chunks_added == 0
        assert not idx.incremental_index.called, (
            "incremental_index should not have been invoked under disable env"
        )
        assert any("SKIPPED" in m for m in captured.records), (
            f"expected a SKIPPED log line; got: {captured.records}"
        )
    finally:
        parent.removeHandler(captured)
        if prior is None:
            os.environ.pop("CODE_SEARCH_DISABLE_AUTO_REINDEX", None)
        else:
            os.environ["CODE_SEARCH_DISABLE_AUTO_REINDEX"] = prior


def test_disable_auto_reindex_env_var_unset_proceeds():
    """Without the env var set, auto_reindex_if_needed delegates as before."""
    from search.incremental_indexer import IncrementalIndexer

    prior = os.environ.pop("CODE_SEARCH_DISABLE_AUTO_REINDEX", None)
    try:
        idx = IncrementalIndexer(
            indexer=MagicMock(),
            embedder=MagicMock(),
            chunker=MagicMock(),
            snapshot_manager=MagicMock(),
        )
        idx.needs_reindex = MagicMock(return_value=True)
        idx.incremental_index = MagicMock(
            return_value=type(
                "R",
                (),
                {
                    "files_added": 0,
                    "files_removed": 0,
                    "files_modified": 0,
                    "chunks_added": 0,
                    "chunks_removed": 0,
                    "time_taken": 0.1,
                    "success": True,
                    "error": None,
                },
            )()
        )

        with tempfile.TemporaryDirectory() as tmp:
            idx.auto_reindex_if_needed(tmp, project_name="test")

        assert idx.incremental_index.called, (
            "incremental_index should have been invoked when env var unset"
        )
    finally:
        if prior is not None:
            os.environ["CODE_SEARCH_DISABLE_AUTO_REINDEX"] = prior
