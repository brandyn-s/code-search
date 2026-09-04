"""Index format versioning: open current, refuse newer, demand reindex for older.

``tests/fixtures/index-format-v1`` is a real index of the frozen fixture corpus
built with the deterministic test embedder at format version 1. The tests
install it under a temporary storage root, point it at a fresh copy of the
corpus, and check how ``get_index_status`` and ``index_directory`` react to
the recorded format version.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from common_utils import get_storage_dir
from mcp_server.code_search_server import CodeSearchServer
from search import index_format
from search.index_format import (
    INDEX_FORMAT_VERSION,
    MIN_SUPPORTED_INDEX_FORMAT,
    STATUS_NEWER,
    STATUS_UNSUPPORTED,
    format_incompatibility,
    stored_format_version,
)

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "index-format-v1" / "project"
CORPUS = REPO / "bench" / "eval" / "fixtures" / "frozen-v1" / "corpus"


def test_legacy_project_info_is_format_one() -> None:
    assert stored_format_version({}) == 1
    assert format_incompatibility({}) is None
    assert format_incompatibility({index_format.FIELD: INDEX_FORMAT_VERSION}) is None


def test_newer_and_unsupported_versions_are_explained() -> None:
    status, message = format_incompatibility({index_format.FIELD: INDEX_FORMAT_VERSION + 1})
    assert status == STATUS_NEWER
    assert "newer code-search" in message and "upgrade" in message

    status, message = format_incompatibility({index_format.FIELD: MIN_SUPPORTED_INDEX_FORMAT - 1})
    assert status == STATUS_UNSUPPORTED
    assert "reindex" in message.lower()

    status, message = format_incompatibility({index_format.FIELD: "1"})
    assert status == STATUS_UNSUPPORTED and "unreadable" in message


def test_completed_metadata_records_current_format() -> None:
    from types import SimpleNamespace

    from search.identity_checks import completed_index_metadata

    configuration = SimpleNamespace(
        provider="test", model_name="m", output_dimension=8, input_type_enabled=False, content_mode="code"
    )
    metadata = completed_index_metadata("pv", configuration, {"name": "generic"})
    assert metadata[index_format.FIELD] == INDEX_FORMAT_VERSION


@pytest.fixture()
def installed_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Install the committed fixture index for a fresh corpus checkout."""
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path / "storage"))
    monkeypatch.setenv("RERANKER", "off")
    monkeypatch.setenv("CODE_SEARCH_STARTUP_AUDIT", "0")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_storage_dir.cache_clear()

    corpus = tmp_path / "corpus"
    shutil.copytree(CORPUS, corpus)
    subprocess.run(["git", "init", "-q", str(corpus)], check=True)
    subprocess.run(["git", "-C", str(corpus), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(corpus), "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "f"],
        check=True,
    )

    server = CodeSearchServer()
    # The fixture was embedded with the deterministic test embedder; keep
    # reindexing paths offline and consistent with it.
    from tests.unit.test_incremental_indexer import _FakeEmbedder

    fake = _FakeEmbedder(dim=8)
    monkeypatch.setattr(server, "embedder", lambda *_a, **_k: fake)
    project_dir = server.get_project_storage_dir(str(corpus))
    shutil.copytree(FIXTURE, project_dir, dirs_exist_ok=True)
    info_path = project_dir / "project_info.json"
    info = json.loads(info_path.read_text())
    info["project_path"] = str(corpus.resolve())
    info["project_name"] = corpus.name
    info_path.write_text(json.dumps(info, indent=2))

    def set_format(version) -> None:
        current = json.loads(info_path.read_text())
        current[index_format.FIELD] = version
        info_path.write_text(json.dumps(current, indent=2))

    yield server, corpus, set_format
    get_storage_dir.cache_clear()


def test_current_build_opens_the_committed_fixture(installed_fixture) -> None:
    server, corpus, _ = installed_fixture
    status = json.loads(server.get_index_status(project_path=str(corpus)))
    assert status.get("index_identity_status") not in {STATUS_NEWER, STATUS_UNSUPPORTED}
    assert "error" not in status, status
    assert status["index_statistics"]["total_chunks"] == 10
    # The fixture was built from a different checkout, so identity is stale,
    # but the index itself verifies and loads.
    from search import epoch_manifest

    project_dir = server.get_project_storage_dir(str(corpus))
    assert epoch_manifest.read_with_fallback(project_dir / "index").freshness == "fresh"
    assert server.get_index_manager(str(corpus)).get_index_size() == 10


def test_newer_format_is_refused_with_upgrade_guidance(installed_fixture) -> None:
    server, corpus, set_format = installed_fixture
    set_format(INDEX_FORMAT_VERSION + 5)

    status = json.loads(server.get_index_status(project_path=str(corpus)))
    assert status["index_identity_status"] == STATUS_NEWER
    assert status["index_ready"] is False
    assert "newer code-search" in status["index_identity_error"]
    assert status["index_format_version"] == INDEX_FORMAT_VERSION + 5

    # Incremental indexing must not touch a newer layout; it fails with the
    # same guidance instead of corrupting the index or raising a traceback.
    started = json.loads(server.index_directory(str(corpus), incremental=True))
    assert started["status"] == "indexing"
    import time

    progress: dict = {}
    for _ in range(600):
        progress = json.loads(server.get_indexing_progress())
        if progress["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.05)
    assert progress["status"] == "failed", progress
    assert "newer code-search" in json.dumps(progress), progress


def test_unsupported_older_format_asks_for_reindex(installed_fixture) -> None:
    server, corpus, set_format = installed_fixture
    set_format(MIN_SUPPORTED_INDEX_FORMAT - 1)

    status = json.loads(server.get_index_status(project_path=str(corpus)))
    assert status["index_identity_status"] == STATUS_UNSUPPORTED
    assert status["index_ready"] is False
    assert "reindex" in status["index_identity_error"].lower()

    server.switch_project(str(corpus))
    active = json.loads(server.get_index_status())
    assert active["index_identity_status"] == STATUS_UNSUPPORTED
