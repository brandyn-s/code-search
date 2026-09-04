"""Source/index identity publication, extracted from the MCP server class.

An index is only *ready* when the checkout identity captured before indexing
matches the identity captured after it. These helpers persist the identity
state machine (``indexing`` → ``ready`` | ``source_changed_during_index`` |
``error`` | ``cancelled`` | ``pending``) into ``project_info.json`` and wrap a
search-time reindex in the same start/end transaction.

They take the identity-capture function as a parameter so the server (and its
tests) can substitute a deterministic seam.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from search.index_identity import (
    IdentityCaptureError,
    IndexIdentity,
    describe_identity_mismatches,
    identity_mismatch_fields,
)

IDENTITY_FIELDS = ("index_identity_status", "index_identity", "index_identity_error")

CaptureFn = Callable[[Path], IndexIdentity]
UpdateInfoFn = Callable[..., None]


def persist_index_identity_state(
    info_file: Path,
    published: Dict[str, Any],
    update_project_info: UpdateInfoFn,
) -> Dict[str, Any]:
    """Replace the persisted identity fields without retaining stale ones."""
    try:
        update_project_info(info_file, published, remove_fields=IDENTITY_FIELDS)
    except (OSError, ValueError) as exc:
        return {
            "index_identity_status": "error",
            "index_identity_error": f"Could not persist index identity: {exc}",
        }
    return published


def read_index_identity_state(info_file: Path) -> Dict[str, Any]:
    """Read only the replaceable identity fields from project metadata."""
    try:
        with open(info_file, encoding="utf-8") as handle:
            project_info = json.load(handle)
    except (OSError, ValueError):
        return {}
    return {key: project_info[key] for key in IDENTITY_FIELDS if key in project_info}


def completed_index_metadata(
    pipeline_version: str,
    configuration: Any,
    synonym_profile: Dict[str, object],
) -> Dict[str, Any]:
    """Provenance published alongside a coherent completed identity."""
    return {
        "pipeline_version": pipeline_version,
        "synonym_profile": synonym_profile,
        "embedding_provider": configuration.provider,
        "embedding_model": configuration.model_name,
        "embedding_dimension": configuration.output_dimension,
        "embedding_input_type_enabled": configuration.input_type_enabled,
        "content_mode": configuration.content_mode,
    }


def finalize_index_identity(
    *,
    capture: CaptureFn,
    project_path: Path,
    info_file: Path,
    start_identity: IndexIdentity,
    ready_metadata: Optional[Dict[str, Any]],
    update_project_info: UpdateInfoFn,
) -> Dict[str, Any]:
    """Atomically publish identity and metadata for a coherent index."""
    try:
        end_identity = capture(project_path)
        mismatch_fields = identity_mismatch_fields(start_identity, end_identity)
        if mismatch_fields:
            change_details = describe_identity_mismatches(start_identity, end_identity)
            published: Dict[str, Any] = {
                "index_identity_status": "source_changed_during_index",
                "index_identity_error": (
                    "Source changed during indexing ("
                    f"{change_details}); rerun "
                    "index_directory against a stable checkout."
                ),
            }
        else:
            published = {
                **(ready_metadata or {}),
                "index_identity_status": "ready",
                "index_identity": end_identity.to_dict(),
            }
    except IdentityCaptureError as exc:
        published = {"index_identity_status": "error", "index_identity_error": str(exc)}
    return persist_index_identity_state(info_file, published, update_project_info)


def auto_reindex_with_identity(
    *,
    capture: CaptureFn,
    incremental_indexer: Any,
    source_path: Path,
    info_file: Path,
    max_age_minutes: float,
    publish_pending: bool,
    ready_metadata: Optional[Dict[str, Any]],
    update_project_info: UpdateInfoFn,
) -> tuple[Any, bool, Dict[str, Any]]:
    """Run search-time reindexing as one source-identity transaction.

    Returns ``(result, mutated, identity_state)`` where ``mutated`` is true when
    the reindex succeeded and changed at least one file.
    """
    previous_state = read_index_identity_state(info_file)

    start_identity: Optional[IndexIdentity] = None
    start_error: Optional[str] = None
    try:
        start_identity = capture(source_path)
    except IdentityCaptureError as exc:
        start_error = str(exc)

    def _persist(published: Dict[str, Any]) -> Dict[str, Any]:
        return persist_index_identity_state(info_file, published, update_project_info)

    if publish_pending:
        pending: Dict[str, Any] = {"index_identity_status": "pending"}
        if start_error:
            pending["index_identity_error"] = f"identity_capture_start_failed: {start_error}"
        _persist(pending)

    try:
        result = incremental_indexer.auto_reindex_if_needed(
            str(source_path), max_age_minutes=max_age_minutes
        )
    except Exception as exc:
        _persist(
            {
                "index_identity_status": "error",
                "index_identity_error": f"auto_reindex_exception: {exc}",
            }
        )
        raise

    def _count(field: str) -> int:
        value = getattr(result, field, 0)
        return value if isinstance(value, int) else 0

    mutated = any(_count(f) > 0 for f in ("files_added", "files_modified", "files_removed"))
    disposition = getattr(result, "reindex_disposition", None)
    completed_scan = disposition == "completed" if isinstance(disposition, str) else mutated
    succeeded = bool(getattr(result, "success", True))

    if not succeeded:
        state = _persist(
            {
                "index_identity_status": "error",
                "index_identity_error": (
                    "auto_reindex_failed: "
                    f"{getattr(result, 'error', None) or 'unknown error'}"
                ),
            }
        )
    elif not completed_scan:
        if publish_pending:
            _persist(previous_state)
        state = previous_state or {"index_identity_status": "legacy_missing"}
    elif start_identity is None:
        state = _persist(
            {
                "index_identity_status": "error",
                "index_identity_error": (
                    "identity_capture_start_failed: "
                    f"{start_error or 'unknown error'}"
                ),
            }
        )
    else:
        state = finalize_index_identity(
            capture=capture,
            project_path=source_path,
            info_file=info_file,
            start_identity=start_identity,
            ready_metadata=ready_metadata,
            update_project_info=update_project_info,
        )
    return result, succeeded and mutated, state
