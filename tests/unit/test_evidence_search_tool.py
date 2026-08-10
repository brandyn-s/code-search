import json

from common_utils import get_storage_dir
from mcp_server.code_search_mcp import CodeSearchMCP
from mcp_server.code_search_server import CodeSearchServer
from mcp_server.evidence_tools import search_code_evidence


class FakeServer:
    def __init__(
        self,
        *,
        ready: bool = True,
        qualified_name: str | None = "repo.src.auth.Auth.verify",
        change_identity: bool = False,
    ):
        self.ready = ready
        self.qualified_name = qualified_name
        self.change_identity = change_identity
        self.status_calls = 0

    def search_code(self, **_kwargs):
        result = {
            "file": "src/auth.py",
            "lines": "10-20",
            "kind": "method",
            "name": "verify",
            "score": 0.9,
            "chunk_id": "chunk-1",
        }
        if self.qualified_name is not None:
            result["qualified_name"] = self.qualified_name
        return json.dumps(
            {
                "query": "auth",
                "results": [result],
                "_metadata": {},
            }
        )

    def get_index_status(self):
        self.status_calls += 1
        generation = "c" * 64
        if self.change_identity and self.status_calls > 1:
            generation = "d" * 64
        return json.dumps(
            {
                "index_ready": self.ready,
                "index_identity_status": (
                    "ready" if self.ready else "stale_source"
                ),
                "index_identity": {
                    "repository_id": "a" * 64,
                    "checkout_id": "e" * 64,
                    "source_revision": "b" * 40,
                    "dirty_fingerprint": "clean",
                    "index_generation": generation,
                },
            }
        )


def test_search_code_evidence_binds_result_to_stable_ready_generation():
    response = json.loads(
        search_code_evidence(
            FakeServer(),
            query="auth",
            search_mode="hybrid",
        )
    )
    result = response["results"][0]
    assert result["symbol_ref"]["id"].startswith("sym:v1:")
    assert result["symbol_ref"]["qualified_name"] == (
        "repo.src.auth.Auth.verify"
    )
    assert result["evidence_ref"]["id"].startswith("ev:v1:")
    assert result["observation_ref"]["id"].startswith("obs:v1:")
    assert result["observation_ref"]["source_engine"] == "code-search"
    assert result["observation_ref"]["stance"] == "support"
    assert result["evidence_ref"]["index_generation"] == "c" * 64
    assert result["evidence_ref"]["evidence_type"] == "hybrid_match"
    assert response["_metadata"]["evidence_refs"] == {
        "schema_version": 1,
        "emitted": True,
        "count": 1,
        "symbol_count": 1,
        "index_generation": "c" * 64,
        "symbol_ref_policy": "canonical_qualified_name_only",
    }


def test_short_semantic_name_emits_evidence_but_not_false_symbol_join():
    response = json.loads(
        search_code_evidence(
            FakeServer(qualified_name=None),
            query="auth",
            search_mode="semantic",
        )
    )
    result = response["results"][0]
    assert "symbol_ref" not in result
    assert "symbol_ref" not in result["evidence_ref"]
    assert result["evidence_ref"]["id"].startswith("ev:v1:")
    assert result["observation_ref"]["id"].startswith("obs:v1:")
    refs = response["_metadata"]["evidence_refs"]
    assert refs["emitted"] is True
    assert refs["count"] == 1
    assert refs["symbol_count"] == 0


def test_search_code_evidence_fails_closed_for_stale_identity():
    response = json.loads(
        search_code_evidence(
            FakeServer(ready=False),
            query="auth",
        )
    )
    assert "symbol_ref" not in response["results"][0]
    assert "evidence_ref" not in response["results"][0]
    assert "observation_ref" not in response["results"][0]
    refs = response["_metadata"]["evidence_refs"]
    assert refs["emitted"] is False
    assert refs["reason"] == "before_search:stale_source"


def test_search_code_evidence_rejects_generation_change_during_search():
    response = json.loads(
        search_code_evidence(
            FakeServer(change_identity=True),
            query="auth",
        )
    )
    result = response["results"][0]
    assert "symbol_ref" not in result
    assert "evidence_ref" not in result
    assert "observation_ref" not in result
    refs = response["_metadata"]["evidence_refs"]
    assert refs["emitted"] is False
    assert refs["reason"] == "identity_changed_during_search"
    assert refs["before_generation"] == "c" * 64
    assert refs["after_generation"] == "d" * 64


def test_mcp_registers_evidence_search_with_description(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    monkeypatch.setenv("CODE_SEARCH_QUERY_HISTORY", "off")
    get_storage_dir.cache_clear()
    server = CodeSearchServer()
    mcp = CodeSearchMCP(server)
    tool = mcp._tool_manager._tools["search_code_evidence"]
    assert tool.description
    assert tool.annotations.readOnlyHint is True
