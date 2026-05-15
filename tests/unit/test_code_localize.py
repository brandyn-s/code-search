"""Tests for the code_localize MCP tool."""
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


def _stub_search_code(server, chunks, metadata=None):
    """Inject a stub search_code that returns the provided chunks."""
    payload = {
        "query": "stub",
        "results": chunks,
        "_metadata": metadata or {},
    }
    server.search_code = lambda **_kw: json.dumps(payload)  # type: ignore[method-assign]


def test_aggregates_chunks_by_file_path(server):
    _stub_search_code(server, [
        {"file": "src/auth.py", "chunk_id": "c1", "score": 0.95,
         "kind": "function", "name": "login", "lines": "10-30"},
        {"file": "src/auth.py", "chunk_id": "c2", "score": 0.80,
         "kind": "function", "name": "logout", "lines": "40-55"},
        {"file": "src/db.py", "chunk_id": "c3", "score": 0.70,
         "kind": "module", "name": "db", "lines": "1-100"},
    ])

    out = json.loads(server.code_localize("how does login work?"))
    assert out["files_returned"] == 2
    assert out["files"][0]["file_path"] == "src/auth.py"
    assert out["files"][0]["chunk_count"] == 2
    assert out["files"][0]["max_similarity"] == 0.95
    assert out["files"][1]["file_path"] == "src/db.py"


def test_top_chunks_sorted_by_similarity(server):
    _stub_search_code(server, [
        {"file": "x.py", "chunk_id": "c_low", "score": 0.30, "kind": "function", "name": "low"},
        {"file": "x.py", "chunk_id": "c_high", "score": 0.95, "kind": "function", "name": "high"},
        {"file": "x.py", "chunk_id": "c_mid", "score": 0.60, "kind": "class", "name": "mid"},
    ])

    out = json.loads(server.code_localize("x"))
    top = out["files"][0]["top_chunks"]
    assert [c["chunk_id"] for c in top] == ["c_high", "c_mid", "c_low"]


def test_top_chunks_caps_at_3_per_file(server):
    _stub_search_code(server, [
        {"file": "big.py", "chunk_id": f"c{i}", "score": 0.9 - i * 0.05,
         "kind": "function", "name": f"fn{i}"}
        for i in range(8)
    ])

    out = json.loads(server.code_localize("x"))
    assert out["files_returned"] == 1
    assert len(out["files"][0]["top_chunks"]) == 3
    assert out["files"][0]["chunk_count"] == 8


def test_diversity_bonus_for_distinct_chunk_types(server):
    # Two files with identical max_similarity and chunk_count, differing
    # only in chunk-type diversity. The diverse file should rank higher.
    _stub_search_code(server, [
        # diverse: 3 distinct types
        {"file": "diverse.py", "chunk_id": "d1", "score": 0.8, "kind": "function"},
        {"file": "diverse.py", "chunk_id": "d2", "score": 0.7, "kind": "class"},
        {"file": "diverse.py", "chunk_id": "d3", "score": 0.6, "kind": "module"},
        # homogeneous: all functions
        {"file": "homo.py", "chunk_id": "h1", "score": 0.8, "kind": "function"},
        {"file": "homo.py", "chunk_id": "h2", "score": 0.7, "kind": "function"},
        {"file": "homo.py", "chunk_id": "h3", "score": 0.6, "kind": "function"},
    ])

    out = json.loads(server.code_localize("x"))
    assert out["files"][0]["file_path"] == "diverse.py"
    assert out["files"][0]["score"] > out["files"][1]["score"]


def test_chunk_count_breadth_bonus(server):
    # Same max_similarity, different chunk counts. Higher count wins.
    _stub_search_code(server, [
        {"file": "many.py", "chunk_id": "m1", "score": 0.7, "kind": "function"},
        {"file": "many.py", "chunk_id": "m2", "score": 0.5, "kind": "function"},
        {"file": "many.py", "chunk_id": "m3", "score": 0.4, "kind": "function"},
        {"file": "few.py", "chunk_id": "f1", "score": 0.7, "kind": "function"},
    ])

    out = json.loads(server.code_localize("x"))
    assert out["files"][0]["file_path"] == "many.py"


def test_k_caps_returned_files(server):
    _stub_search_code(server, [
        {"file": f"f{i}.py", "chunk_id": f"c{i}", "score": 0.9 - i * 0.01,
         "kind": "function"}
        for i in range(20)
    ])

    out = json.loads(server.code_localize("x", k=5))
    assert out["files_returned"] == 5
    assert out["total_files_seen"] == 20


def test_empty_search_results_returns_hint(server):
    _stub_search_code(server, [])
    out = json.loads(server.code_localize("nothing matches this"))
    assert out["files_returned"] == 0
    assert out["files"] == []
    assert "hint" in out


def test_propagates_indexing_in_progress(server):
    server.search_code = lambda **_kw: json.dumps({  # type: ignore[method-assign]
        "query": "x", "results": [], "indexing_in_progress": True,
        "message": "Indexing 50%...",
    })
    out = json.loads(server.code_localize("x"))
    assert out.get("indexing_in_progress") is True


def test_propagates_error(server):
    server.search_code = lambda **_kw: json.dumps({  # type: ignore[method-assign]
        "error": "no project active",
    })
    out = json.loads(server.code_localize("x"))
    assert "error" in out


def test_metadata_passthrough(server):
    _stub_search_code(
        server,
        [{"file": "a.py", "chunk_id": "c1", "score": 0.9, "kind": "function"}],
        metadata={"reranker": {"applied": True, "reason": "ok", "latency_ms": 1000},
                  "freshness": "fresh"},
    )
    out = json.loads(server.code_localize("x"))
    assert out["underlying_search_metadata"]["reranker"]["applied"] is True
    assert out["underlying_search_metadata"]["freshness"] == "fresh"


def test_chunks_without_file_path_skipped(server):
    _stub_search_code(server, [
        {"chunk_id": "noframe", "score": 0.9, "kind": "function"},  # no "file"
        {"file": "real.py", "chunk_id": "r1", "score": 0.8, "kind": "function"},
    ])
    out = json.loads(server.code_localize("x"))
    assert out["files_returned"] == 1
    assert out["files"][0]["file_path"] == "real.py"
