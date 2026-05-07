"""Tests for the home-dir / workspace-root refusal helper.

Phase A (2026-05-07): regression for the wedge discovered in this session.
The MCP server, spawned with cwd=$HOME, auto-indexed the home directory
because `ensure_project_indexed(cwd)` and `index_directory(cwd)` had no
guard against unreasonable project roots. The job wedged on a ~150K-file
walk; cancel signals weren't checked during the walk phase.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mcp_server.code_search_server import _refuse_as_project_root_reason


def test_empty_path_refused():
    assert _refuse_as_project_root_reason("") == "empty path"


def test_home_dir_refused():
    home = str(Path.home())
    reason = _refuse_as_project_root_reason(home)
    assert reason is not None
    assert "user home directory" in reason


def test_filesystem_root_refused():
    # Path.resolve() produces a system-dependent root; use the resolve
    # of `/` to find the actual root the OS reports.
    root = Path("/").resolve()
    reason = _refuse_as_project_root_reason(str(root))
    assert reason is not None
    assert "filesystem root" in reason


def test_workspace_with_many_nested_gits_refused(tmp_path):
    """Directory with >5 nested .git dirs is treated as a workspace, not project."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for i in range(7):
        repo = workspace / f"repo-{i}"
        repo.mkdir()
        (repo / ".git").mkdir()

    reason = _refuse_as_project_root_reason(str(workspace))
    assert reason is not None
    assert "workspace root" in reason


def test_normal_project_accepted(tmp_path):
    """Single-project directory with at most one .git is fine."""
    proj = tmp_path / "myproject"
    proj.mkdir()
    (proj / ".git").mkdir()
    (proj / "main.py").write_text("def main(): pass\n")

    assert _refuse_as_project_root_reason(str(proj)) is None


def test_subdirectory_of_home_accepted(tmp_path):
    """`~/Documents/GitHub/foo` is a normal project — only home itself is refused."""
    proj = tmp_path / "Documents" / "GitHub" / "foo"
    proj.mkdir(parents=True)
    (proj / ".git").mkdir()

    # A subdirectory under what tmp_path emulates as ~/Documents — must be accepted.
    assert _refuse_as_project_root_reason(str(proj)) is None


def test_workspace_with_5_or_fewer_gits_accepted(tmp_path):
    """Boundary: exactly 5 nested .git dirs is accepted (not >5)."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    for i in range(5):
        repo = workspace / f"repo-{i}"
        repo.mkdir()
        (repo / ".git").mkdir()

    assert _refuse_as_project_root_reason(str(workspace)) is None


def test_unresolvable_path_refused():
    # Path.resolve() rarely raises on Windows; emulate via mock.
    with patch.object(Path, "resolve", side_effect=OSError("nope")):
        reason = _refuse_as_project_root_reason("/some/path")
        assert reason is not None
        assert "unresolvable" in reason or "empty path" in reason
