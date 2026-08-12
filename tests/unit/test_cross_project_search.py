"""Cross-project search must be deterministic and must not mutate active state."""

import json

from mcp_server.code_search_server import CodeSearchServer


def _write_project(storage, project_id, name, path, provider):
    project_dir = storage / "projects" / project_id
    (project_dir / "index").mkdir(parents=True)
    (project_dir / "index" / "code.index").write_bytes(b"index")
    (project_dir / "project_info.json").write_text(
        json.dumps(
            {
                "project_name": name,
                "project_path": str(path),
                "embedding_provider": provider,
            }
        ),
        encoding="utf-8",
    )


def test_search_all_projects_is_state_isolated_and_project_balanced(
    tmp_path, monkeypatch
):
    from mcp_server import code_search_server as module

    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    gamma = tmp_path / "gamma"
    for path in (alpha, beta, gamma):
        path.mkdir()
    _write_project(tmp_path, "beta_22222222", "beta", beta, "voyage")
    _write_project(tmp_path, "alpha_11111111", "alpha", alpha, "voyage")
    _write_project(tmp_path, "gamma_33333333", "gamma", gamma, "voyage")
    monkeypatch.setattr(module, "get_storage_dir", lambda: tmp_path)

    hits = {
        str(alpha): [
            {"file_path": "alpha/one.py", "score": 0.99},
            {"file_path": "alpha/two.py", "score": 0.98},
        ],
        str(beta): [{"file_path": "beta/one.py", "score": 0.40}],
        str(gamma): [],
    }

    class IsolatedWorker:
        def __init__(self):
            self.active = None

        def switch_project(self, project_path, provider=None):
            self.active = str(project_path)
            return json.dumps({"success": True})

        def search_code(self, **_kwargs):
            return json.dumps({"results": hits[self.active]})

    server = CodeSearchServer()
    manager_sentinel = object()
    searcher_sentinel = object()
    server._current_project = "/original"
    server._current_provider = "voyage-context"
    server._index_manager = manager_sentinel
    server._searcher = searcher_sentinel
    monkeypatch.setattr(module, "CodeSearchServer", IsolatedWorker)

    response = json.loads(
        server.search_all_projects("authenticate user", k=2, top_k=10)
    )

    assert response["projects_attempted"] == 3
    assert response["projects_with_matches"] == 2
    assert response["ranking_policy"] == "project_balanced_round_robin"
    assert response["cross_project_score_comparable"] is False
    assert [item["project_name"] for item in response["results"]] == [
        "alpha",
        "beta",
        "alpha",
    ]
    assert server._current_project == "/original"
    assert server._current_provider == "voyage-context"
    assert server._index_manager is manager_sentinel
    assert server._searcher is searcher_sentinel
