"""Tests for get_project_storage_dir provider-resolution behavior.

Pin the fix that prevents silent write/read divergence:
when a project has an existing provider-aware index dir with a populated
code.index, calling `get_project_storage_dir(provider=None)` for the
same path must auto-resolve to that provider-aware dir rather than
silently writing to a competing legacy-hash dir.

Background: 2026-05-09 PSM Phase A reindex. The MCP tool defaulted
provider=None when called without an explicit provider, and code-search
wrote to legacy-hash acf665e1/ while the eval read provider-aware
780e511b/. Two indexes never overlapped; the post-reindex MRR equaled
the pre-reindex MRR. User: "this can never happen again."
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def temp_storage(tmp_path, monkeypatch):
    """Point CODE_SEARCH_STORAGE at a tmp dir and reset the lru_cache."""
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    from common_utils import get_storage_dir
    get_storage_dir.cache_clear()
    return tmp_path


def _write_populated_index(project_dir: Path, project_path: str, provider: str, project_hash: str):
    """Create a provider-aware dir with the marker files that trigger auto-resolve."""
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "index").mkdir(parents=True, exist_ok=True)
    (project_dir / "index" / "code.index").write_bytes(b"\x00" * 16)
    (project_dir / "project_info.json").write_text(
        json.dumps({
            "project_name": Path(project_path).name,
            "project_path": str(Path(project_path).resolve()),
            "project_hash": project_hash,
            "embedding_provider": provider,
        }),
        encoding="utf-8",
    )


def test_provider_none_auto_resolves_to_existing_provider_aware_sibling(
    temp_storage, tmp_path
):
    """When a populated provider-aware sibling exists, provider=None must
    auto-resolve to it instead of creating a competing legacy-hash dir."""
    from mcp_server.code_search_server import CodeSearchServer

    project_path = tmp_path / "demo-project"
    project_path.mkdir()

    p_str = str(project_path.resolve())
    legacy_hash = hashlib.md5(p_str.encode()).hexdigest()[:8]
    voyage_hash = hashlib.md5(f"{p_str}:voyage".encode()).hexdigest()[:8]
    assert legacy_hash != voyage_hash, "fixture invariant: hashes must differ"

    voyage_dir = temp_storage / "projects" / f"{project_path.name}_{voyage_hash}"
    _write_populated_index(voyage_dir, p_str, "voyage", voyage_hash)

    server = CodeSearchServer()
    resolved = server.get_project_storage_dir(p_str, provider=None)
    assert resolved == voyage_dir, (
        f"provider=None should have auto-resolved to the provider-aware "
        f"sibling {voyage_dir.name}, got {resolved.name}"
    )

    # And there should be NO new legacy-hash dir created
    legacy_dir = temp_storage / "projects" / f"{project_path.name}_{legacy_hash}"
    assert not legacy_dir.exists(), (
        f"resolution must not create a competing legacy-hash dir; found {legacy_dir.name}"
    )


def test_provider_none_no_sibling_uses_legacy_hash(
    temp_storage, tmp_path
):
    """When no provider-aware sibling exists, provider=None preserves the
    pre-fix backward-compatible behavior (legacy-hash dir)."""
    from mcp_server.code_search_server import CodeSearchServer

    project_path = tmp_path / "demo-project"
    project_path.mkdir()

    p_str = str(project_path.resolve())
    legacy_hash = hashlib.md5(p_str.encode()).hexdigest()[:8]
    legacy_dir = temp_storage / "projects" / f"{project_path.name}_{legacy_hash}"

    server = CodeSearchServer()
    resolved = server.get_project_storage_dir(p_str, provider=None)
    assert resolved == legacy_dir


def test_provider_none_skips_unpopulated_sibling(
    temp_storage, tmp_path
):
    """A provider-aware sibling without code.index (just a stub) should NOT
    trigger auto-resolve — only sibling dirs with populated indexes count."""
    from mcp_server.code_search_server import CodeSearchServer

    project_path = tmp_path / "demo-project"
    project_path.mkdir()

    p_str = str(project_path.resolve())
    legacy_hash = hashlib.md5(p_str.encode()).hexdigest()[:8]
    voyage_hash = hashlib.md5(f"{p_str}:voyage".encode()).hexdigest()[:8]

    # Stub provider-aware dir: project_info present, but no code.index
    voyage_dir = temp_storage / "projects" / f"{project_path.name}_{voyage_hash}"
    voyage_dir.mkdir(parents=True)
    (voyage_dir / "project_info.json").write_text(
        json.dumps({
            "project_name": project_path.name,
            "project_path": p_str,
            "project_hash": voyage_hash,
            "embedding_provider": "voyage",
        }),
        encoding="utf-8",
    )

    server = CodeSearchServer()
    resolved = server.get_project_storage_dir(p_str, provider=None)
    legacy_dir = temp_storage / "projects" / f"{project_path.name}_{legacy_hash}"
    assert resolved == legacy_dir, (
        "stub provider-aware dir must not capture provider=None routing"
    )


def test_provider_none_skips_sibling_with_different_project_path(
    temp_storage, tmp_path
):
    """A provider-aware sibling whose project_info.json points at a different
    project_path is unrelated and must not capture this project's routing."""
    from mcp_server.code_search_server import CodeSearchServer

    project_path = tmp_path / "demo-project"
    project_path.mkdir()
    other_path = tmp_path / "other-project"
    other_path.mkdir()

    p_str = str(project_path.resolve())
    legacy_hash = hashlib.md5(p_str.encode()).hexdigest()[:8]

    # Populated provider-aware dir BUT it points at a different path
    voyage_hash_other = hashlib.md5(f"{other_path}:voyage".encode()).hexdigest()[:8]
    decoy_dir = temp_storage / "projects" / f"{project_path.name}_{voyage_hash_other}"
    _write_populated_index(decoy_dir, str(other_path), "voyage", voyage_hash_other)

    server = CodeSearchServer()
    resolved = server.get_project_storage_dir(p_str, provider=None)
    legacy_dir = temp_storage / "projects" / f"{project_path.name}_{legacy_hash}"
    assert resolved == legacy_dir, (
        "auto-resolve must require project_path equality on the sibling"
    )


def test_explicit_provider_unchanged(temp_storage, tmp_path):
    """Explicit provider= keeps existing behavior — no auto-resolve scan."""
    from mcp_server.code_search_server import CodeSearchServer

    project_path = tmp_path / "demo-project"
    project_path.mkdir()

    p_str = str(project_path.resolve())
    voyage_hash = hashlib.md5(f"{p_str}:voyage".encode()).hexdigest()[:8]
    voyage_dir = temp_storage / "projects" / f"{project_path.name}_{voyage_hash}"

    server = CodeSearchServer()
    resolved = server.get_project_storage_dir(p_str, provider="voyage")
    assert resolved == voyage_dir
