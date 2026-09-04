"""On-disk index format versioning.

``INDEX_FORMAT_VERSION`` is written into ``project_info.json`` when an index
is published. Readers compare it against the range this build understands:

* newer than ``INDEX_FORMAT_VERSION``: the index was built by a newer
  code-search; refuse to open it and tell the operator to upgrade or rebuild;
* older than ``MIN_SUPPORTED_INDEX_FORMAT``: the layout changed
  incompatibly; a reindex is required;
* missing: legacy indexes predate the field and are format 1.

Bump ``INDEX_FORMAT_VERSION`` whenever the layout of ``index/`` (FAISS file,
``chunk_ids.pkl``, SQLite schemas, manifest) or ``project_info.json`` changes
in a way older readers cannot handle; raise ``MIN_SUPPORTED_INDEX_FORMAT``
only when this build drops the ability to read an older layout. The policy
is documented in ``docs/index-format.md``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

INDEX_FORMAT_VERSION = 1
MIN_SUPPORTED_INDEX_FORMAT = 1
FIELD = "index_format_version"

STATUS_NEWER = "index_format_newer"
STATUS_UNSUPPORTED = "index_format_unsupported"


def stored_format_version(project_info: Mapping[str, Any]) -> int:
    """Format version recorded in ``project_info``; legacy indexes are 1."""
    value = project_info.get(FIELD, 1)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{FIELD} must be an integer, got {value!r}")
    return value


def format_incompatibility(project_info: Mapping[str, Any]) -> Optional[tuple[str, str]]:
    """Return ``(status, message)`` when this build cannot read the index.

    ``None`` means the recorded format is readable by this build.
    """
    try:
        version = stored_format_version(project_info)
    except ValueError as exc:
        return (
            STATUS_UNSUPPORTED,
            f"index format version is unreadable ({exc}); run "
            "index_directory(incremental=false) to rebuild this index",
        )
    if version > INDEX_FORMAT_VERSION:
        return (
            STATUS_NEWER,
            f"index was built by a newer code-search (index format {version}; "
            f"this build reads up to {INDEX_FORMAT_VERSION}); upgrade "
            "code-search or run index_directory(incremental=false) to rebuild "
            "this index with the installed version",
        )
    if version < MIN_SUPPORTED_INDEX_FORMAT:
        return (
            STATUS_UNSUPPORTED,
            f"index format {version} is no longer supported (this build reads "
            f"{MIN_SUPPORTED_INDEX_FORMAT} to {INDEX_FORMAT_VERSION}); reindex "
            "required: run index_directory(incremental=false)",
        )
    return None
