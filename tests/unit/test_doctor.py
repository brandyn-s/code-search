"""Shape and behaviour of `code-search-mcp doctor`."""

from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

import pytest

from common_utils import get_storage_dir
from mcp_server import doctor

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "index-format-v1" / "project"


@pytest.fixture()
def storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "storage"
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(root))
    # Other tests leak provider env (setdefault); pin what this suite assumes.
    for name in ("EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VOYAGE_API_KEY", "not-a-real-key-1234567890")
    monkeypatch.setenv("RERANKER", "off")
    get_storage_dir.cache_clear()
    from search.config import get_search_config

    get_search_config.cache_clear()
    (root / "projects").mkdir(parents=True)
    shutil.copytree(FIXTURE, root / "projects" / "corpus_fixture")
    yield root
    get_storage_dir.cache_clear()


def test_collect_has_the_documented_shape_and_redacts_secrets(storage: Path) -> None:
    report = doctor.collect(check_network=False)

    assert set(report) >= {"package", "platform", "config", "resolved", "storage", "projects", "reachability", "grammars", "problems"}
    assert report["package"]["mcp_sdk"]
    assert report["platform"]["python"]
    assert report["config"]["VOYAGE_API_KEY"].startswith("set (")
    assert "not-a-real-key" not in json.dumps(report)
    assert report["config"]["ANTHROPIC_API_KEY"] is None
    assert report["resolved"]["reranker"] == "off"
    assert report["resolved"]["embedding_provider"] == "voyage"
    assert report["storage"]["path"] == str(storage)
    assert report["storage"]["size_bytes"] > 0
    assert report["reachability"] == {"checked": False}
    assert any(g.get("language") == "python" for g in report["grammars"])

    projects = report["projects"]
    assert len(projects) == 1
    project = projects[0]
    assert project["storage_dir"] == "corpus_fixture"
    assert project["chunks"] == 10
    assert project["index_format_version"] == 1
    assert project["manifest_freshness"] == "fresh"
    assert project["generation"]
    assert "format_status" not in project


def test_problems_flag_newer_index_and_missing_local_extra(storage: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    info_path = storage / "projects" / "corpus_fixture" / "project_info.json"
    info = json.loads(info_path.read_text())
    info["index_format_version"] = 99
    info_path.write_text(json.dumps(info))
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(doctor, "_version", lambda name: None)

    import embeddings.local_extra as local_extra

    monkeypatch.setattr(local_extra, "local_extra_available", lambda: False)
    report = doctor.collect(check_network=False)

    assert report["projects"][0]["format_status"] == "index_format_newer"
    problems = "\n".join(report["problems"])
    assert "newer code-search" in problems
    assert "code-search-mcp[local]" in problems


def test_reachability_is_skipped_without_keys_and_reports_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    import httpx

    def boom(*_a, **_k):
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(httpx, "head", boom)
    result = doctor._reachability(timeout_s=0.1)
    assert result["voyage"] == {"checked": False, "reason": "VOYAGE_API_KEY not set"}
    assert result["anthropic"]["checked"] is True and result["anthropic"]["reachable"] is False
    assert "ConnectError" in result["anthropic"]["error"]


def test_main_json_and_text_modes(storage: Path) -> None:
    out = io.StringIO()
    code = doctor.main(["--json", "--no-network"], out=out)
    payload = json.loads(out.getvalue())
    assert code == 0, payload["problems"]
    assert payload["projects"][0]["chunks"] == 10

    out = io.StringIO()
    doctor.main(["--no-network"], out=out)
    text = out.getvalue()
    assert text.startswith("code-search-mcp")
    assert "projects: 1" in text
    assert "problems: none" in text


def test_server_entry_point_dispatches_doctor(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from mcp_server import server

    monkeypatch.setattr(sys, "argv", ["code-search-mcp", "doctor", "--json", "--no-network"])
    monkeypatch.setattr("mcp_server.doctor.collect", lambda **_k: {"package": {}, "platform": {"python": "x", "os": "y"}, "config": {}, "resolved": {}, "storage": {}, "projects": [], "reachability": {}, "grammars": [], "problems": []})
    with pytest.raises(SystemExit) as excinfo:
        server.main()
    assert excinfo.value.code == 0
    assert json.loads(capsys.readouterr().out)["problems"] == []
