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
        change_identity_on_metadata_lookup: bool = False,
        chunk_metadata: dict | None = None,
    ):
        self.ready = ready
        self.qualified_name = qualified_name
        self.change_identity = change_identity
        self.change_identity_on_metadata_lookup = change_identity_on_metadata_lookup
        self.metadata_was_read = False
        self.status_calls = 0
        self.chunk_metadata = chunk_metadata if chunk_metadata is not None else {
            "relative_path": "src/auth.py",
            "start_line": 10,
            "end_line": 20,
            "full_content": "\n".join(
                [
                    "def verify(token):",
                    "    decoded = parse(token)",
                    "",
                    "    if not decoded:",
                    "        return False",
                    "    return check(decoded)",
                ]
            ),
        }

    def get_index_manager(self):
        server = self

        class FakeIndexManager:
            def get_chunk_by_id(self, chunk_id):
                assert chunk_id == "chunk-1"
                server.metadata_was_read = True
                return server.chunk_metadata

        return FakeIndexManager()

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
        if (
            self.change_identity and self.status_calls > 1
        ) or (
            self.change_identity_on_metadata_lookup and self.metadata_was_read
        ):
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
    assert "evidence_ref" not in result
    assert result["span_role"] == "retrieval_context"
    assert result["context_span"] == {
        "relative_path": "src/auth.py",
        "start_line": 10,
        "end_line": 20,
    }
    candidates = result["evidence_candidates"]
    assert [candidate["lines"] for candidate in candidates] == [
        "10-10",
        "11-11",
        "13-13",
        "14-14",
        "15-15",
    ]
    assert all(candidate["role"] == "atomic_source_line" for candidate in candidates)
    assert all(
        candidate["evidence_ref"]["id"].startswith("ev:v1:")
        for candidate in candidates
    )
    assert len({candidate["evidence_ref"]["id"] for candidate in candidates}) == 5
    assert all(
        candidate["observation_ref"]["id"].startswith("obs:v1:")
        for candidate in candidates
    )
    assert all(
        candidate["observation_ref"]["source_engine"] == "code-search"
        and candidate["observation_ref"]["stance"] == "support"
        for candidate in candidates
    )
    assert all(
        candidate["evidence_ref"]["index_generation"] == "c" * 64
        and candidate["evidence_ref"]["evidence_type"] == "hybrid_match"
        for candidate in candidates
    )
    assert response["_metadata"]["evidence_refs"] == {
        "schema_version": 2,
        "emitted": True,
        "count": 5,
        "result_count": 1,
        "symbol_count": 1,
        "index_generation": "c" * 64,
        "candidate_policy": "atomic_nonblank_source_line",
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
    assert "evidence_ref" not in result
    assert all(
        "symbol_ref" not in candidate["evidence_ref"]
        for candidate in result["evidence_candidates"]
    )
    assert all(
        candidate["evidence_ref"]["id"].startswith("ev:v1:")
        and candidate["observation_ref"]["id"].startswith("obs:v1:")
        for candidate in result["evidence_candidates"]
    )
    refs = response["_metadata"]["evidence_refs"]
    assert refs["emitted"] is True
    assert refs["count"] == 5
    assert refs["symbol_count"] == 0


def test_search_code_evidence_never_attests_broad_context_without_source_metadata():
    response = json.loads(
        search_code_evidence(
            FakeServer(chunk_metadata={}),
            query="auth",
            search_mode="semantic",
        )
    )

    result = response["results"][0]
    assert result["span_role"] == "retrieval_context"
    assert "evidence_ref" not in result
    assert "evidence_candidates" not in result
    refs = response["_metadata"]["evidence_refs"]
    assert refs["emitted"] is False
    assert refs["count"] == 0
    assert refs["reason"] == "no_referenceable_results"


def test_search_code_evidence_fails_closed_when_indexed_content_exceeds_bounds():
    response = json.loads(
        search_code_evidence(
            FakeServer(
                chunk_metadata={
                    "relative_path": "src/auth.py",
                    "start_line": 10,
                    "end_line": 11,
                    "full_content": "one\ntwo\nthree",
                }
            ),
            query="auth",
            search_mode="semantic",
        )
    )

    result = response["results"][0]
    assert "evidence_ref" not in result
    assert "evidence_candidates" not in result
    assert response["_metadata"]["evidence_refs"]["emitted"] is False


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


def test_search_code_evidence_binds_metadata_read_inside_identity_snapshots():
    response = json.loads(
        search_code_evidence(
            FakeServer(change_identity_on_metadata_lookup=True),
            query="auth",
        )
    )

    result = response["results"][0]
    assert "evidence_ref" not in result
    assert "evidence_candidates" not in result
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
    assert "retrieval context only" in tool.description
    assert "evidence_ref.id" in tool.description
    assert tool.annotations.readOnlyHint is True
