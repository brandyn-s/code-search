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
    identity: dict[str, str],
    search_mode: str,
    refs_metadata: dict[str, Any],
) -> None:
    repository_id = identity["repository_id"]
    source_revision = identity["source_revision"]
    index_generation = identity["index_generation"]
    evidence_type = _evidence_type(search_mode)

    emitted = 0
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
            or start_line < 0
            or end_line < start_line
        ):
            continue

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

        evidence_ref = EvidenceRef(
            repository_id=repository_id,
            source_revision=source_revision,
            index_generation=index_generation,
            relative_path=relative_path,
            start_line=start_line,
            end_line=end_line,
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
        item["evidence_ref"] = evidence_ref.to_dict()
        item["observation_ref"] = observation_ref.to_dict()
        emitted += 1

    refs_metadata.update(
        {
            "emitted": emitted > 0,
            "count": emitted,
            "symbol_count": symbol_count,
            "index_generation": index_generation,
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

    Semantic chunks are valid location evidence, but they are not necessarily
    one canonical graph symbol: cAST merging may combine definitions or extend
    their ranges. Therefore a SymbolRef is emitted only when the underlying
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
        "schema_version": 1,
        "emitted": False,
        "count": 0,
        "symbol_count": 0,
    }
    metadata["evidence_refs"] = refs_metadata

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
        identity=identity_after,
        search_mode=search_mode,
        refs_metadata=refs_metadata,
    )
    return json.dumps(response, separators=(",", ":"))
