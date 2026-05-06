"""Tests for get_indexing_progress with F2 background-reindex awareness (Plan-2 F3)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mcp_server.code_search_server import CodeSearchServer  # noqa: E402


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    from common_utils import get_storage_dir
    get_storage_dir.cache_clear()
    return CodeSearchServer()


def test_idle_when_nothing_running(server):
    """No foreground job + no background reindex → idle."""
    out = json.loads(server.get_indexing_progress())
    assert out["status"] == "idle"
    assert out["background_reindex_active"] is False


def test_background_reindex_active_when_only_bg_running(server):
    """Background reindex active + no foreground job → background_reindex_active status."""
    server._background_reindex_active = True
    out = json.loads(server.get_indexing_progress())
    assert out["status"] == "background_reindex_active"
    assert out["background_reindex_active"] is True
    assert "freshness=stale_reindex_in_progress" in out["message"]


def test_indexing_status_when_foreground_job_running(server):
    """Foreground job present → status from job dict, plus bg flag."""
    server._indexing_job = {
        "job_id": "abc123",
        "status": "indexing",
        "phase": "embedding",
        "current": 50,
        "total": 100,
        "directory": "/path",
        "project_name": "p",
        "errors": [],
        "result": None,
    }
    out = json.loads(server.get_indexing_progress())
    assert out["status"] == "indexing"
    assert out["job_id"] == "abc123"
    assert out["phase"] == "embedding"
    assert out["chunks_done"] == 50
    assert out["chunks_total"] == 100
    assert out["percent"] == 50.0
    assert out["background_reindex_active"] is False


def test_both_foreground_and_background_running(server):
    """When BOTH are active, response surfaces both."""
    server._indexing_job = {
        "job_id": "fg1",
        "status": "indexing",
        "phase": "chunking",
        "current": 10,
        "total": 100,
        "directory": "/x",
        "project_name": "x",
        "errors": [],
        "result": None,
    }
    server._background_reindex_active = True
    out = json.loads(server.get_indexing_progress())
    # Foreground job state takes precedence in `status`
    assert out["status"] == "indexing"
    assert out["job_id"] == "fg1"
    # But background flag is also surfaced so the LLM agent can detect it
    assert out["background_reindex_active"] is True


def test_completed_job_includes_result(server):
    """Completed job → status=completed and result attached."""
    server._indexing_job = {
        "job_id": "done1",
        "status": "completed",
        "phase": "done",
        "current": 100,
        "total": 100,
        "directory": "/x",
        "project_name": "x",
        "errors": [],
        "result": {"chunks_indexed": 100, "elapsed_seconds": 30},
    }
    out = json.loads(server.get_indexing_progress())
    assert out["status"] == "completed"
    assert out["result"] == {"chunks_indexed": 100, "elapsed_seconds": 30}


def test_failed_job_includes_result(server):
    server._indexing_job = {
        "job_id": "fail1",
        "status": "failed",
        "phase": "embedding",
        "current": 5,
        "total": 100,
        "directory": "/x",
        "project_name": "x",
        "errors": ["api error"],
        "result": {"errors": ["api error"]},
    }
    out = json.loads(server.get_indexing_progress())
    assert out["status"] == "failed"
    assert out["result"] == {"errors": ["api error"]}


def test_background_flag_present_when_attribute_missing(server):
    """If a CodeSearchServer was instantiated before F2 added the attribute,
    get_indexing_progress should still return False without crashing."""
    if hasattr(server, "_background_reindex_active"):
        delattr(server, "_background_reindex_active")
    out = json.loads(server.get_indexing_progress())
    assert out["background_reindex_active"] is False
    assert out["status"] == "idle"


def test_total_zero_omits_percent_field(server):
    """When total=0, percent/chunks_done/chunks_total are omitted (avoids div by 0)."""
    server._indexing_job = {
        "job_id": "starting",
        "status": "indexing",
        "phase": "starting",
        "current": 0,
        "total": 0,
        "directory": "/x",
        "project_name": "x",
        "errors": [],
        "result": None,
    }
    out = json.loads(server.get_indexing_progress())
    assert "percent" not in out
    assert "chunks_done" not in out
    assert "chunks_total" not in out
    assert out["status"] == "indexing"
