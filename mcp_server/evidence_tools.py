"""Additive MCP adapters that bind search results to index evidence."""

from __future__ import annotations

import json
from typing import Any, Optional

from search.evidence import EvidenceRef, ObservationRef, SymbolRef

_IDENTITY_FIELDS = (
    "repository_id",
    "checkout_id",
    "source_revision",
    "dirty_fingerprint",
    "index_generation",
)


def _line_range(value: object) -> tuple[int, int]:
    if not isinstance(value, str):
        return 0, 0
    start_text, separator, end_text = value.partition("-")
    if not separator:
        return 0, 0
    try:
        return int(start_text), int(end_text)
    except ValueError:
        return 0, 0


def _evidence_type(search_mode: str) -> str:
    return {
        "keyword": "lexical_match",
        "semantic": "semantic_match",
        "hybrid": "hybrid_match",
        "auto": "hybrid_match",
    }.get(search_mode, "search_match")


def _indexed_chunks_for_response(
    server: Any,
    response: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Read result metadata before the closing index-identity snapshot."""
    try:
        index_manager = server.get_index_manager()
    except Exception:
        # Evidence enrichment is fail-closed and must not break retrieval.
        return {}
    indexed_chunks: dict[str, dict[str, Any]] = {}
    results = response.get("results", [])
    if not isinstance(results, list):
        return indexed_chunks
    for item in results:
        chunk_id = item.get("chunk_id") if isinstance(item, dict) else None
        if not isinstance(chunk_id, str) or not chunk_id:
            continue
        try:
            metadata = index_manager.get_chunk_by_id(chunk_id)
        except Exception:
            continue
        if isinstance(metadata, dict):
            indexed_chunks[chunk_id] = metadata
    return indexed_chunks


def _atomic_source_lines(
    item: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> list[tuple[int, str]]:
    """Return nonblank source lines only when chunk metadata binds exactly.

    Retrieval chunks intentionally carry broad context and may merge multiple
    semantic units.  Those coordinates are useful for discovery, but they are
    not minimal claim evidence.  The already-indexed full content is a
    contiguous source slice, so each nonblank line can be offered as a small,
    immutable evidence candidate without asking the model to manufacture a
    range.
    """
    if metadata is None:
        return []
    relative_path = str(item.get("file") or "").replace("\\", "/").strip()
    metadata_path = str(
        metadata.get("relative_path") or metadata.get("file_path") or ""
    ).replace("\\", "/").strip()
    context_start, context_end = _line_range(item.get("lines"))
    indexed_start = metadata.get("start_line")
    indexed_end = metadata.get("end_line")
    content = metadata.get("full_content")
    if (
        not relative_path
        or metadata_path != relative_path
        or isinstance(indexed_start, bool)
        or not isinstance(indexed_start, int)
        or isinstance(indexed_end, bool)
        or not isinstance(indexed_end, int)
        or indexed_start != context_start
        or indexed_end != context_end
        or indexed_start < 1
        or indexed_end < indexed_start
        or not isinstance(content, str)
    ):
        return []

    source_lines = content.splitlines()
    if len(source_lines) > indexed_end - indexed_start + 1:
        # Never map content beyond the generation-bound indexed coordinates.
        return []
    return [
        (indexed_start + offset, line)
        for offset, line in enumerate(source_lines)
        if line.strip()
    ]


def _line_snippet(line: str, limit: int = 200) -> str:
    value = line.strip()
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _ready_identity(server: Any) -> tuple[dict[str, str] | None, str]:
    """Return one complete ready identity without changing server state."""
    try:
        status = json.loads(server.get_index_status())
    except Exception as exc:  # evidence enrichment must not break search
        return None, f"index_status_error:{type(exc).__name__}"
    if not isinstance(status, dict):
        return None, "invalid_index_status"
    if (
        status.get("index_ready") is not True
        or status.get("index_identity_status") != "ready"
    ):
        return None, str(
            status.get("index_identity_status", "index_not_ready")
        )
    identity = status.get("index_identity")
    if not isinstance(identity, dict):
        return None, "index_identity_missing"
    if any(
        not isinstance(identity.get(field), str) or not identity[field]
        for field in _IDENTITY_FIELDS
    ):
        return None, "identity_fields_missing"
    return {
        field: str(identity[field])
        for field in _IDENTITY_FIELDS
    }, ""


def _same_identity(
    before: dict[str, str],
    after: dict[str, str],
) -> bool:
    return all(before[field] == after[field] for field in _IDENTITY_FIELDS)


def _attach_refs(
    response: dict[str, Any],
    *,
    indexed_chunks: dict[str, dict[str, Any]],
    identity: dict[str, str],
    search_mode: str,
    refs_metadata: dict[str, Any],
) -> None:
    repository_id = identity["repository_id"]
    source_revision = identity["source_revision"]
    index_generation = identity["index_generation"]
    evidence_type = _evidence_type(search_mode)

    emitted = 0
    result_count = 0
    symbol_count = 0
    results = response.get("results", [])
    if not isinstance(results, list):
        refs_metadata["reason"] = "unsupported_result_shape"
        return

    for item in results:
        if not isinstance(item, dict):
            continue
        relative_path = str(item.get("file") or "").strip()
        start_line, end_line = _line_range(item.get("lines"))
        if (
            not relative_path
            or start_line < 1
            or end_line < start_line
        ):
            continue

        item["span_role"] = "retrieval_context"
        item["context_span"] = {
            "relative_path": relative_path,
            "start_line": start_line,
            "end_line": end_line,
        }

        symbol_ref = None
        qualified_name = item.get("qualified_name")
        if isinstance(qualified_name, str) and qualified_name.strip():
            symbol_ref = SymbolRef(
                repository_id=repository_id,
                source_revision=source_revision,
                relative_path=relative_path,
                symbol_kind=str(item.get("kind") or "unknown"),
                qualified_name=qualified_name.strip(),
                start_line=start_line,
                end_line=end_line,
            )
            item["symbol_ref"] = symbol_ref.to_dict()
            symbol_count += 1

        metadata = indexed_chunks.get(str(item.get("chunk_id") or ""))
        candidates = []
        for line_number, source_line in _atomic_source_lines(item, metadata):
            evidence_ref = EvidenceRef(
                repository_id=repository_id,
                source_revision=source_revision,
                index_generation=index_generation,
                relative_path=relative_path,
                start_line=line_number,
                end_line=line_number,
                evidence_type=evidence_type,
                symbol_ref=symbol_ref,
            )
            observation_ref = ObservationRef(
                evidence_ref=evidence_ref,
                stance="support",
                source_engine="code-search",
                derivation=evidence_type,
                confidence_band="unknown",
            )
            candidates.append(
                {
                    "role": "atomic_source_line",
                    "lines": f"{line_number}-{line_number}",
                    "snippet": _line_snippet(source_line),
                    "evidence_ref": evidence_ref.to_dict(),
                    "observation_ref": observation_ref.to_dict(),
                }
            )
        if candidates:
            item["evidence_candidates"] = candidates
            emitted += len(candidates)
            result_count += 1

    refs_metadata.update(
        {
            "emitted": emitted > 0,
            "count": emitted,
            "result_count": result_count,
            "symbol_count": symbol_count,
            "index_generation": index_generation,
            "candidate_policy": "atomic_nonblank_source_line",
            "symbol_ref_policy": "canonical_qualified_name_only",
        }
    )
    if emitted == 0:
        refs_metadata["reason"] = "no_referenceable_results"


def search_code_evidence(
    server: Any,
    query: str,
    k: int = 5,
    search_mode: str = "auto",
    file_pattern: Optional[str] = None,
    chunk_type: Optional[str] = None,
    include_context: bool = True,
    auto_reindex: bool = True,
    max_age_minutes: float = 5,
    provider: Optional[str] = None,
) -> str:
    """Run production search and bind evidence to one stable index generation.

    The adapter snapshots the complete identity before and after retrieval.
    Evidence references are emitted only when both snapshots are ready and
    identical. This prevents a search result from being labeled with a newer
    generation after an auto-reindex or concurrent source change.

    Search-result chunk ranges remain discovery context: cAST merging may
    combine definitions or extend their ranges.  Evidence is instead emitted
    as backend-issued candidates for exact, nonblank source lines from the
    generation-bound indexed chunk.  The model selects those immutable IDs; it
    never manufactures a source range.  A SymbolRef is emitted only when the
    response explicitly supplies a canonical ``qualified_name``. Short names
    are never promoted into plausible but false cross-engine joins.
    """
    identity_before, before_reason = _ready_identity(server)

    raw_response = server.search_code(
        query=query,
        k=k,
        search_mode=search_mode,
        file_pattern=file_pattern,
        chunk_type=chunk_type,
        include_context=include_context,
        auto_reindex=auto_reindex,
        max_age_minutes=max_age_minutes,
        provider=provider,
    )
    try:
        response = json.loads(raw_response)
    except (TypeError, json.JSONDecodeError):
        return raw_response
    if not isinstance(response, dict) or "error" in response:
        return raw_response

    metadata = response.setdefault("_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        response["_metadata"] = metadata
    refs_metadata: dict[str, Any] = {
        "schema_version": 2,
        "emitted": False,
        "count": 0,
        "symbol_count": 0,
    }
    metadata["evidence_refs"] = refs_metadata

    indexed_chunks = (
        _indexed_chunks_for_response(server, response)
        if identity_before is not None
        else {}
    )
    identity_after, after_reason = _ready_identity(server)
    if identity_before is None:
        refs_metadata["reason"] = f"before_search:{before_reason}"
        return json.dumps(response, separators=(",", ":"))
    if identity_after is None:
        refs_metadata["reason"] = f"after_search:{after_reason}"
        return json.dumps(response, separators=(",", ":"))
    if not _same_identity(identity_before, identity_after):
        refs_metadata["reason"] = "identity_changed_during_search"
        refs_metadata["before_generation"] = identity_before[
            "index_generation"
        ]
        refs_metadata["after_generation"] = identity_after[
            "index_generation"
        ]
        return json.dumps(response, separators=(",", ":"))

    _attach_refs(
        response,
        indexed_chunks=indexed_chunks,
        identity=identity_after,
        search_mode=search_mode,
        refs_metadata=refs_metadata,
    )
    return json.dumps(response, separators=(",", ":"))
