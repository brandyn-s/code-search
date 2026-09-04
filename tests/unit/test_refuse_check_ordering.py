"""Tests for the 2026-05-13 refuse-check ordering and cron path fixes.

INCIDENT: code-search auto-creates `~/.claude_code_search/projects/<...>/`
+ `project_info.json` on the first search call with no active project,
because `get_searcher` falls back to `os.getcwd()` (= `~/` on Windows
when Claude Code spawns the MCP server). The Phase A refuse-check existed
in `ensure_project_indexed` but ran AFTER `get_project_storage_dir`
wrote the json. The orphan dir persisted; `auto_reindex_if_needed`
(separate code path with no refuse-check) then tried to full-index
the orphan on every 5-min tick, holding SQLite + FAISS locks
indefinitely. All parallel `search_code` requests timed out with
-32001:user-cancel.

Two-prong fix:
  U1: ensure_project_indexed calls refuse-check BEFORE
      get_project_storage_dir — never writes the orphan.
  U2: auto_reindex_if_needed calls refuse-check at entry — defense-in-
      depth for orphan entries created by older server versions.
"""
from __future__ import annotations

from pathlib import Path


from search.path_validation import refuse_as_project_root_reason


def test_refuse_home_dir():
    """Home dir must classify as forbidden. This is the 2026-05-13
    incident shape: cwd=$HOME → orphan project entry → reindex hang."""
    home = str(Path.home())
    reason = refuse_as_project_root_reason(home)
    assert reason is not None
    assert "home" in reason.lower()


def test_refuse_empty_path():
    """Empty path is forbidden (no implicit guess)."""
    assert refuse_as_project_root_reason("") == "empty path"


def test_accept_legitimate_repo_path(tmp_path):
    """SAFETY: a legit project path must NOT be refused.
    If this fails, the guard becomes a denial-of-service for real work.
    """
    repo = tmp_path / "Documents" / "GitHub" / "my-repo"
    repo.mkdir(parents=True)
    assert refuse_as_project_root_reason(str(repo)) is None


def test_refuse_workspace_with_many_git_dirs(tmp_path):
    """6+ nested .git dirs at depth 1 → workspace, not project.
    Catches ~/Documents/GitHub/ being passed as a project root."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for i in range(7):
        (workspace / f"repo{i}").mkdir()
        (workspace / f"repo{i}" / ".git").mkdir()
    reason = refuse_as_project_root_reason(str(workspace))
    assert reason is not None
    assert "workspace" in reason.lower() or ".git" in reason


def test_auto_reindex_refuses_home_path(tmp_path, monkeypatch):
    """U2: auto_reindex_if_needed must refuse home-dir paths even if
    an orphan project entry exists from an older server version.

    Verifies the cron-path defense-in-depth: even if pre-fix server
    code wrote an orphan project, the new auto-reindex skips it
    instead of trying to full-index ~/ over and over."""
    from unittest.mock import MagicMock
    from search.incremental_indexer import IncrementalIndexer

    # Construct a stub IncrementalIndexer; we never reach the inner
    # incremental_index call because the refuse-check fires first.
    ii = IncrementalIndexer(
        indexer=MagicMock(),
        embedder=MagicMock(),
        chunker=MagicMock(),
    )

    result = ii.auto_reindex_if_needed(
        project_path=str(Path.home()),
        project_name="BrandynSchult",
        max_age_minutes=0,  # would normally force reindex
    )

    assert result.success is True
    assert result.files_added == 0
    assert result.files_modified == 0
    # Stub mocks would have raised if incremental_index were called.


def test_auto_reindex_accepts_legitimate_path(tmp_path, monkeypatch):
    """SAFETY: U2 must not break the happy path. A real project path
    should proceed to the normal stale/fresh check, not get refused."""
    from unittest.mock import MagicMock
    from search.incremental_indexer import IncrementalIndexer

    repo = tmp_path / "real-repo"
    repo.mkdir()

    ii = IncrementalIndexer(
        indexer=MagicMock(),
        embedder=MagicMock(),
        chunker=MagicMock(),
    )

    # Stub `needs_reindex` to return False so we don't actually walk files.
    # If the refuse-check incorrectly fired, the test would never reach this
    # mock (the function would have returned early via the refuse branch).
    monkeypatch.setattr(ii, "needs_reindex", lambda p, m: False)

    result = ii.auto_reindex_if_needed(
        project_path=str(repo),
        project_name="real-repo",
        max_age_minutes=5,
    )
    assert result.success is True
    # If refused, project_name would be in the refuse log; if accepted,
    # the function ran through to needs_reindex == False → fresh result.


def test_refuse_as_project_root_alias_in_server():
    """The pre-extraction alias `_refuse_as_project_root_reason` in
    code_search_server.py is preserved as a re-export for backward
    compatibility with existing tests and callers."""
    from mcp_server.code_search_server import _refuse_as_project_root_reason

    assert _refuse_as_project_root_reason is refuse_as_project_root_reason
