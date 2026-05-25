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

import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_SENSITIVE_DIR_NAMES = frozenset({
    ".ssh", ".gnupg", ".gpg", ".aws", ".azure", ".config",
    ".docker", ".kube", ".helm", ".vault",
    ".credentials", ".password-store",
})

_SENSITIVE_SYSTEM_PREFIXES: List[str] = [
    "/etc",
    "/var",
    "/root",
    "/proc",
    "/sys",
    "/boot",
    "/dev",
]


def _get_allowed_roots() -> Optional[List[Path]]:
    raw = os.environ.get("CODE_SEARCH_ALLOWED_ROOTS", "").strip()
    if not raw:
        return None
    roots: List[Path] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if entry:
            try:
                roots.append(Path(entry).resolve())
            except (OSError, ValueError):
                pass
    return roots if roots else None


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_sensitive_path(p: Path) -> Optional[str]:
    home = Path.home().resolve()
    if _is_under(p, home):
        try:
            rel = p.relative_to(home)
        except ValueError:
            rel = None
        if rel is not None:
            for part in rel.parts:
                if part in _SENSITIVE_DIR_NAMES:
                    return (
                        f"path contains sensitive directory '{part}' — "
                        f"indexing would read and send file contents to "
                        f"the configured embedding provider"
                    )

    p_str = str(p)
    for prefix in _SENSITIVE_SYSTEM_PREFIXES:
        if p_str == prefix or p_str.startswith(prefix + "/"):
            return (
                f"path is under system directory '{prefix}' — "
                f"indexing would read and send file contents to "
                f"the configured embedding provider"
            )

    return None


def refuse_as_project_root_reason(path_str: str) -> Optional[str]:
    """Return a refusal reason string if `path_str` should NOT be treated
    as a project root, or None if it's acceptable.

    Refusal rules (in order):
      1. Empty / unresolvable path.
      2. Equal to the user's home directory.
      3. A filesystem root (e.g., `C:\\` or `/`).
      4. Outside CODE_SEARCH_ALLOWED_ROOTS (when set).
      5. Under a sensitive directory (.ssh, .aws, /etc, etc.).
      6. Contains more than 5 nested `.git` directories within depth 3 -
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

    # Allowlist check: when CODE_SEARCH_ALLOWED_ROOTS is set, the resolved
    # path must be equal to or a descendant of at least one allowed root.
    allowed = _get_allowed_roots()
    if allowed is not None:
        if not any(_is_under(p, root) for root in allowed):
            return (
                f"path {p} is not under any CODE_SEARCH_ALLOWED_ROOTS "
                f"({', '.join(str(r) for r in allowed)})"
            )

    # Sensitive directory check: block paths under known credential/config
    # directories to prevent accidental exfiltration of secrets via the
    # embedding provider.
    sensitive = _is_sensitive_path(p)
    if sensitive:
        return sensitive

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


def is_path_within_root(path: Path, root: Path) -> bool:
    """Check whether resolved `path` is within `root`, catching symlink escapes.

    Both arguments should already be resolved (via Path.resolve()) before
    calling. Returns False when `path` escapes `root` or on any OS error.
    """
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
