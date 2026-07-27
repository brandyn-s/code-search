"""Tests for switch_project's auto-resolve dir scanning.

Regression: the prior auto-resolve only checked the legacy (path-only-hash)
dir for project_info.json. When a project was born with a provider already
set, no legacy dir ever existed; switch_project then silently fell through
to creating an empty stub at the legacy hash and reported "Project not
indexed" even though the provider-aware dir was fully populated.
"""

import hashlib
import json
import tempfile
from pathlib import Path


def _hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:8]


def test_switch_project_resolves_born_provider_aware_dir(monkeypatch):
    """A project that was indexed with a provider from the start (no legacy
    dir ever existed) must resolve via switch_project(provider=None)."""
    from mcp_server.code_search_server import CodeSearchServer

    from common_utils import get_storage_dir

    monkeypatch.setenv("EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        # Override storage dir so this test can't touch the user's real data
        storage = Path(tmpdir) / "storage"
        storage.mkdir()
        monkeypatch.setenv("CODE_SEARCH_STORAGE", str(storage))
        # get_storage_dir is lru_cached — clear so this test sees its env override
        get_storage_dir.cache_clear()

        # Create the project on disk so Path.resolve().exists() passes
        project_path = (Path(tmpdir) / "project").resolve()
        project_path.mkdir()

        # Build the provider-aware dir BY HAND (no legacy peer)
        provider_hash = _hash(f"{project_path}:voyage")
        provider_dir = storage / "projects" / f"{project_path.name}_{provider_hash}"
        (provider_dir / "index").mkdir(parents=True)
        (provider_dir / "index" / "code.index").write_bytes(b"fake-faiss-payload")
        (provider_dir / "project_info.json").write_text(
            json.dumps({
                "project_name": project_path.name,
                "project_path": str(project_path),
                "project_hash": provider_hash,
                "embedding_provider": "voyage",
                "embedding_model": "",
                "content_mode": "code",
            })
        )

        # Sanity: legacy dir does NOT exist
        legacy_hash = _hash(str(project_path))
        legacy_dir = storage / "projects" / f"{project_path.name}_{legacy_hash}"
        assert not legacy_dir.exists()

        server = CodeSearchServer()
        result = json.loads(server.switch_project(str(project_path)))

        assert "error" not in result, (
            f"switch_project failed for born-provider-aware project: {result}"
        )
        assert result.get("success") is True
        assert result["project_info"]["project_hash"] == provider_hash
        assert result["project_info"]["embedding_provider"] == "voyage"

        # Crucially, no empty legacy stub should have been created
        assert not legacy_dir.exists(), (
            f"switch_project created an unwanted legacy stub at {legacy_dir}"
        )


def test_switch_project_skips_stub_dir_for_populated_peer(monkeypatch):
    """If a stub dir (project_info.json only, no index) exists for a path
    AND a populated dir for the same path also exists, switch_project must
    pick the populated one — not the stub."""
    from mcp_server.code_search_server import CodeSearchServer

    from common_utils import get_storage_dir

    monkeypatch.setenv("EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        storage = Path(tmpdir) / "storage"
        storage.mkdir()
        monkeypatch.setenv("CODE_SEARCH_STORAGE", str(storage))
        # get_storage_dir is lru_cached — clear so this test sees its env override
        get_storage_dir.cache_clear()

        project_path = (Path(tmpdir) / "project").resolve()
        project_path.mkdir()

        legacy_hash = _hash(str(project_path))
        provider_hash = _hash(f"{project_path}:voyage")

        # Stub at the legacy hash (no index dir)
        legacy_dir = storage / "projects" / f"{project_path.name}_{legacy_hash}"
        legacy_dir.mkdir(parents=True)
        (legacy_dir / "project_info.json").write_text(
            json.dumps({
                "project_name": project_path.name,
                "project_path": str(project_path),
                "project_hash": legacy_hash,
                "embedding_provider": "voyage",
            })
        )

        # Populated dir at the provider hash
        provider_dir = storage / "projects" / f"{project_path.name}_{provider_hash}"
        (provider_dir / "index").mkdir(parents=True)
        (provider_dir / "index" / "code.index").write_bytes(b"fake")
        (provider_dir / "project_info.json").write_text(
            json.dumps({
                "project_name": project_path.name,
                "project_path": str(project_path),
                "project_hash": provider_hash,
                "embedding_provider": "voyage",
            })
        )

        server = CodeSearchServer()
        result = json.loads(server.switch_project(str(project_path)))

        assert "error" not in result, (
            f"switch_project picked stub over populated peer: {result}"
        )
        assert result["project_info"]["project_hash"] == provider_hash


def test_switch_project_failure_preserves_active_project_state(
    monkeypatch,
    tmp_path,
):
    """Invalid target metadata must not partially activate the target."""
    from common_utils import get_storage_dir
    from mcp_server.code_search_server import CodeSearchServer

    storage = tmp_path / "storage"
    storage.mkdir()
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(storage))
    get_storage_dir.cache_clear()

    target = (tmp_path / "target").resolve()
    target.mkdir()
    provider_hash = _hash(f"{target}:voyage")
    provider_dir = storage / "projects" / f"{target.name}_{provider_hash}"
    (provider_dir / "index").mkdir(parents=True)
    (provider_dir / "index" / "code.index").write_bytes(b"fake")
    (provider_dir / "project_info.json").write_text(
        "{not-json",
        encoding="utf-8",
    )

    server = CodeSearchServer()
    old_manager = object()
    old_searcher = object()
    server._current_project = "/already/active"
    server._current_provider = "local"
    server._index_manager = old_manager
    server._searcher = old_searcher

    result = json.loads(
        server.switch_project(str(target), provider="voyage")
    )

    assert "error" in result
    assert server._current_project == "/already/active"
    assert server._current_provider == "local"
    assert server._index_manager is old_manager
    assert server._searcher is old_searcher
