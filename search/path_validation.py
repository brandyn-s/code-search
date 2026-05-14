"""Project-root path validation shared across MCP server and indexer.

`_refuse_as_project_root_reason` was originally defined in
`mcp_server/code_search_server.py` (added Phase A 2026-05-07) and gated
`ensure_project_indexed` against auto-indexing home directories or
nested-git workspace roots. Extracted to this module 2026-05-13 so the
same check can be applied by `IncrementalIndexer.auto_reindex_if_needed`
without a circular import.

INCIDENT 2026-05-13: the Phase A check existed but fired AFTER
`get_project_storage_dir` wrote `project_info.json` at line 383 of
code_search_server.py. The orphan dir/json persisted on disk; subsequent
`auto_reindex_if_needed` cron ticks (which had no refuse-check at all)
then attempted full-indexes of the home directory, holding SQLite + FAISS
locks indefinitely and surfacing as `-32001: user-cancel` to all parallel
search callers. Two-prong fix:
  1. ensure_project_indexed calls this BEFORE get_project_storage_dir
     (caller-side: never write the orphan entry in the first place).
  2. auto_reindex_if_needed calls this at entry (cron-side: defense-in-
     depth for orphan entries created by older server versions).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def refuse_as_project_root_reason(path_str: str) -> Optional[str]:
    """Return a refusal reason string if `path_str` should NOT be treated
    as a project root, or None if it's acceptable.

    Refusal rules (in order):
      1. Empty / unresolvable path.
      2. Equal to the user's home directory.
      3. A filesystem root (e.g., `C:\\` or `/`).
      4. Contains more than 5 nested `.git` directories within depth 3 -
         signals a workspace root (e.g., ~/Documents/GitHub/) rather
         than a project. Cap walk depth to keep this cheap.

    The `/` and home checks are exact-match; subdirectories of home
    (e.g. `~/Documents/GitHub/foo`) are fine.
    """
    if not path_str:
        return "empty path"
    try:
        p = Path(path_str).resolve()
    except (OSError, ValueError) as e:
        return f"unresolvable path: {e}"

    home = Path.home().resolve()
    if p == home:
        return f"path is the user home directory ({home})"

    # Filesystem root check: anchor with no parts, or anchor==str(p).
    if p.parent == p:
        return f"path is a filesystem root ({p})"

    # Nested-git workspace heuristic. Walk at most depth 3, cap visits.
    git_count = 0
    visited = 0
    try:
        for entry in p.iterdir():
            visited += 1
            if visited > 50:
                break
            if not entry.is_dir():
                continue
            if (entry / ".git").exists():
                git_count += 1
                if git_count > 5:
                    return (
                        f"path contains >5 nested .git directories - looks "
                        f"like a workspace root, not a project ({p})"
                    )
    except (OSError, PermissionError):
        # Walk failure is not itself a refusal - let the indexer hit it.
        pass

    return None
