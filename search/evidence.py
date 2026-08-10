"""Canonical, generation-bound references for cross-engine code evidence.

The contract is intentionally independent of code-search internals so the same
identity can be reproduced by code-graph and orchestration clients.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

SCHEMA_VERSION = 1


def _canonical_path(path: str) -> str:
    value = path.replace("\\", "/").strip()
    while value.startswith("./"):
        value = value[2:]
    return value


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"{prefix}:v{SCHEMA_VERSION}:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SymbolRef:
    repository_id: str
    source_revision: str
    relative_path: str
    symbol_kind: str
    qualified_name: str
    start_line: int
    end_line: int

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "repository_id": self.repository_id,
            "source_revision": self.source_revision,
            "relative_path": _canonical_path(self.relative_path),
            "symbol_kind": self.symbol_kind.strip().lower(),
            "qualified_name": self.qualified_name.strip(),
            "start_line": int(self.start_line),
            "end_line": int(self.end_line),
        }
        return {"id": _stable_id("sym", payload), **payload}


@dataclass(frozen=True)
class EvidenceRef:
    repository_id: str
    source_revision: str
    index_generation: str
    relative_path: str
    start_line: int
    end_line: int
    evidence_type: str
    symbol_ref: SymbolRef | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "repository_id": self.repository_id,
            "source_revision": self.source_revision,
            "index_generation": self.index_generation,
            "relative_path": _canonical_path(self.relative_path),
            "start_line": int(self.start_line),
            "end_line": int(self.end_line),
            "evidence_type": self.evidence_type.strip().lower(),
        }
        if self.symbol_ref is not None:
            payload["symbol_ref"] = self.symbol_ref.to_dict()
        return {"id": _stable_id("ev", payload), **payload}


def symbol_ref_from_search_result(
    result: Any,
    *,
    repository_id: str,
    source_revision: str,
) -> SymbolRef | None:
    """Build a symbol ref only from an explicitly canonical qualified name.

    A semantic chunk's short ``name`` and optional parent are not sufficient to
    reproduce code-graph's canonical symbol identity: merged chunks may span
    more than one definition, and graph qualified names include a canonical
    module prefix. Failing closed here prevents a plausible but false
    cross-engine join. Callers may still emit a generation-bound EvidenceRef
    for the chunk itself.
    """
    qualified_name = str(
        getattr(result, "qualified_name", "") or ""
    ).strip()
    if not qualified_name:
        return None
    return SymbolRef(
        repository_id=repository_id,
        source_revision=source_revision,
        relative_path=str(
            getattr(result, "relative_path", "")
            or getattr(result, "file_path", "")
            or ""
        ),
        symbol_kind=str(getattr(result, "chunk_type", "unknown") or "unknown"),
        qualified_name=qualified_name,
        start_line=int(getattr(result, "start_line", 0) or 0),
        end_line=int(getattr(result, "end_line", 0) or 0),
    )
