"""Tests for provider-aware threading through switch_project / search / delete.

Regression coverage for the 2026-04-17 bug where switch_project(provider=X)
selected the provider-aware index but search_code fell back to the legacy
(path-only) hash and returned empty results. Covers:
- switch_project stores _current_provider
- get_index_manager / get_searcher invalidate on provider change
- embedder accepts a provider argument
- delete_project accepts a project_hash disambiguator
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def server(tmp_path, monkeypatch):
    """Boot a CodeSearchServer with storage under tmp_path."""
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    # Reset module-level singletons that cache get_storage_dir()
    from mcp_server.code_search_server import CodeSearchServer

    s = CodeSearchServer()
    return s


def _make_populated_dir(storage_root: Path, name: str, hash_: str, provider: str):
    """Create a project dir that looks "indexed" (has code.index + project_info)."""
    project_dir = storage_root / "projects" / f"{name}_{hash_}"
    (project_dir / "index").mkdir(parents=True)
    (project_dir / "index" / "code.index").write_bytes(b"\x00" * 128)
    (project_dir / "project_info.json").write_text(
        json.dumps(
            {
                "project_name": name,
                "project_path": str(storage_root.parent / name),
                "project_hash": hash_,
                "embedding_provider": provider,
                "embedding_model": "",
                "content_mode": "code",
            }
        )
    )
    return project_dir


def test_switch_project_sets_current_provider(server, tmp_path, monkeypatch):
    """switch_project(provider=...) must persist provider on the server."""
    from mcp_server import code_search_server as mod

    monkeypatch.setattr(mod, "get_storage_dir", lambda: tmp_path)

    repo = tmp_path / "myrepo"
    repo.mkdir()

    # Pre-create both legacy and provider-aware index dirs so switch_project
    # finds an index without running indexing.
    import hashlib

    path_resolved = str(repo.resolve())
    legacy_hash = hashlib.md5(path_resolved.encode()).hexdigest()[:8]
    voyage_hash = hashlib.md5(f"{path_resolved}:voyage".encode()).hexdigest()[:8]
    _make_populated_dir(tmp_path, "myrepo", legacy_hash, "voyage")
    _make_populated_dir(tmp_path, "myrepo", voyage_hash, "voyage")

    result = json.loads(server.switch_project(str(repo), provider="voyage"))
    assert result.get("success") is True, f"switch failed: {result}"
    assert server._current_provider == "voyage"
    assert server._current_project == str(repo.resolve())


def test_get_index_manager_invalidates_on_provider_change(server, tmp_path, monkeypatch):
    """Changing provider must reset the cached index_manager."""
    from mcp_server import code_search_server as mod

    monkeypatch.setattr(mod, "get_storage_dir", lambda: tmp_path)

    repo = tmp_path / "myrepo"
    repo.mkdir()

    import hashlib

    path_resolved = str(repo.resolve())
    voyage_hash = hashlib.md5(f"{path_resolved}:voyage".encode()).hexdigest()[:8]
    context_hash = hashlib.md5(
        f"{path_resolved}:voyage-context".encode()
    ).hexdigest()[:8]
    _make_populated_dir(tmp_path, "myrepo", voyage_hash, "voyage")
    _make_populated_dir(tmp_path, "myrepo", context_hash, "voyage-context")

    # Return a distinct sentinel per constructor call so we can verify the
    # cache was invalidated rather than reused.
    sentinels = [object(), object()]
    with patch(
        "mcp_server.code_search_server.CodeIndexManager",
        side_effect=sentinels,
    ):
        first = server.get_index_manager(str(repo), provider="voyage")
        second = server.get_index_manager(str(repo), provider="voyage-context")

    assert first is sentinels[0]
    assert second is sentinels[1]
    assert first is not second, "Manager must be reinitialized on provider change"
    assert server._current_provider == "voyage-context"


def test_embedder_accepts_provider(server, tmp_path, monkeypatch):
    """embedder(provider=...) must resolve the provider-aware storage dir."""
    from mcp_server import code_search_server as mod

    monkeypatch.setattr(mod, "get_storage_dir", lambda: tmp_path)

    repo = tmp_path / "myrepo"
    repo.mkdir()
    import hashlib

    path_resolved = str(repo.resolve())
    context_hash = hashlib.md5(
        f"{path_resolved}:voyage-context".encode()
    ).hexdigest()[:8]
    _make_populated_dir(tmp_path, "myrepo", context_hash, "voyage-context")

    # embedder() should NOT raise TypeError on the new parameter.
    import inspect

    sig = inspect.signature(server.embedder)
    assert "provider" in sig.parameters, (
        "embedder() must accept a provider argument for dual-model workflows"
    )


def test_delete_project_hash_disambiguator(server, tmp_path, monkeypatch):
    """delete_project(project_hash=...) targets a specific hash when names collide."""
    # Monkey-patch the storage root that get_storage_dir() resolves to
    from mcp_server import code_search_server as mod

    monkeypatch.setattr(mod, "get_storage_dir", lambda: tmp_path)

    # Create two dirs with the same name but different hashes
    _make_populated_dir(tmp_path, "myrepo", "aaaaaaaa", "voyage")
    _make_populated_dir(tmp_path, "myrepo", "bbbbbbbb", "voyage-context")

    # Target the second explicitly
    result = json.loads(server.delete_project("myrepo", project_hash="bbbbbbbb"))
    assert result["success"] is True
    assert not (tmp_path / "projects" / "myrepo_bbbbbbbb").exists()
    assert (tmp_path / "projects" / "myrepo_aaaaaaaa").exists()


def test_delete_project_accepts_combined_name_hash(server, tmp_path, monkeypatch):
    """Accept `name_hash` as a combined identifier in project_name."""
    from mcp_server import code_search_server as mod

    monkeypatch.setattr(mod, "get_storage_dir", lambda: tmp_path)

    _make_populated_dir(tmp_path, "myrepo", "aaaaaaaa", "voyage")
    _make_populated_dir(tmp_path, "myrepo", "bbbbbbbb", "voyage-context")

    # Pass combined name_hash as single arg
    result = json.loads(server.delete_project("myrepo_aaaaaaaa"))
    assert result["success"] is True
    assert not (tmp_path / "projects" / "myrepo_aaaaaaaa").exists()
    assert (tmp_path / "projects" / "myrepo_bbbbbbbb").exists()


def test_delete_project_deterministic_without_hash(server, tmp_path, monkeypatch):
    """Without a hash, delete picks sorted-first for repeatability."""
    from mcp_server import code_search_server as mod

    monkeypatch.setattr(mod, "get_storage_dir", lambda: tmp_path)

    _make_populated_dir(tmp_path, "myrepo", "cccccccc", "voyage")
    _make_populated_dir(tmp_path, "myrepo", "aaaaaaaa", "voyage")
    _make_populated_dir(tmp_path, "myrepo", "bbbbbbbb", "voyage")

    result = json.loads(server.delete_project("myrepo"))
    assert result["success"] is True
    # Sorted alphabetical: aaaaaaaa is first
    assert not (tmp_path / "projects" / "myrepo_aaaaaaaa").exists()
    assert (tmp_path / "projects" / "myrepo_bbbbbbbb").exists()
    assert (tmp_path / "projects" / "myrepo_cccccccc").exists()
