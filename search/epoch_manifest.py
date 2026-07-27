"""Epoch-manifest primitive for atomic-or-fail index commits.

This module implements the structural fix designed in Plan-2 E1
(see docs/epoch_manifest_design.md). It is a NEW module that can be
adopted by production write paths in Plan-2 E2 — this PR ships only the
primitive and its tests, no migration of existing code.

Core idea: a manifest JSON file describes one consistent epoch of an
index. Writes prepare a candidate manifest, validate cross-artifact
consistency, then commit-by-rename via os.replace (atomic on
POSIX + Windows since Python 3.3). Readers (E3 PR) consume current.json
and fall back to prior.json on checksum failure.

The module is self-contained — depends only on stdlib. It is NOT yet
wired into search/indexer.py; that's E2.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

LOG = logging.getLogger(__name__)

MANIFEST_VERSION = 1
CURRENT_FILE = "current.json"
PRIOR_FILE = "prior.json"
CANDIDATE_FILE = "candidate.json"


class ManifestError(Exception):
    """Base class for manifest-related errors."""


class ManifestConsistencyError(ManifestError):
    """Raised at write time when artifacts don't agree on record counts."""


class ManifestMissing(ManifestError):
    """Raised at read time when no committed manifest exists."""


class ManifestCorrupt(ManifestError):
    """Raised when both current and prior manifests fail verification."""


@dataclass
class ArtifactSpec:
    """Describes one index artifact for the manifest."""
    name: str           # e.g., "chunk_ids.pkl"
    path: Path          # absolute path to the live file
    count: Optional[int] = None  # record count (None for non-record artifacts like stats.json)


def _sha256_file(path: Path) -> str:
    """Streaming SHA256 of a file. None of our index artifacts exceed
    a few MB; a fixed-size buffer is fine."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_chunk_ids_pkl(path: Path) -> int:
    """Count records in chunk_ids.pkl (a pickled list)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ManifestConsistencyError(
            f"{path} did not unpickle as list (got {type(data).__name__})"
        )
    return len(data)


def _count_sqlite_table(path: Path, table: str, where: str = "") -> int:
    """Count rows in a sqlite table; returns 0 if table doesn't exist."""
    if not path.exists():
        return 0
    con = sqlite3.connect(str(path))
    try:
        try:
            sql = f"SELECT COUNT(*) FROM {table}"
            if where:
                sql += f" WHERE {where}"
            row = con.execute(sql).fetchone()
            return int(row[0]) if row else 0
        except sqlite3.OperationalError:
            return 0
    finally:
        con.close()


def count_metadata_db(path: Path) -> int:
    """Count keys in metadata.db (SqliteDict-style 'unnamed' table)."""
    return _count_sqlite_table(path, "unnamed")


def count_fts5_db(path: Path) -> int:
    """Count rows in fts5.db's chunk_fts virtual table."""
    return _count_sqlite_table(path, "chunk_fts")


def _new_epoch_id() -> str:
    """Timestamp + random suffix; sortable by creation time."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    suffix = secrets.token_hex(4)
    return f"{ts}-{suffix}"


def build_manifest(
    project_dir: Path,
    artifacts: list[ArtifactSpec],
    provider: str = "",
    model: str = "",
    vector_dim: int = 0,
    quantization: str = "",
    pipeline_version: str = "",
) -> Dict[str, Any]:
    """Compute the manifest dict for a snapshot of the artifacts.

    Performs the cross-artifact consistency check. Raises
    ManifestConsistencyError if record counts disagree.
    """
    artifacts_dict: Dict[str, Any] = {}
    record_counts: list[int] = []
    for spec in artifacts:
        if not spec.path.exists():
            raise ManifestConsistencyError(f"Artifact missing: {spec.path}")
        sha = _sha256_file(spec.path)
        size = spec.path.stat().st_size
        rel = str(spec.path.relative_to(project_dir)).replace("\\", "/")
        entry: Dict[str, Any] = {"path": rel, "sha256": sha, "bytes": size}
        if spec.count is not None:
            entry["count"] = spec.count
            record_counts.append(spec.count)
        else:
            entry["count"] = None
        artifacts_dict[spec.name] = entry

    # Consistency: every record-bearing artifact must agree on count.
    consistency: Dict[str, Any] = {}
    if record_counts:
        all_match = len(set(record_counts)) == 1
        consistency["all_artifacts_share_count"] = all_match
        consistency["expected_count"] = record_counts[0] if all_match else None
        if not all_match:
            consistency["per_artifact_counts"] = {
                name: artifacts_dict[name]["count"]
                for name in artifacts_dict
                if artifacts_dict[name].get("count") is not None
            }
            raise ManifestConsistencyError(
                f"Record-count mismatch across artifacts: "
                f"{consistency['per_artifact_counts']}"
            )

    return {
        "version": MANIFEST_VERSION,
        "epoch_id": _new_epoch_id(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "vector_dim": vector_dim,
        "quantization": quantization,
        "pipeline_version": pipeline_version,
        "artifacts": artifacts_dict,
        "consistency": consistency,
    }


def commit_manifest(project_dir: Path, manifest: Dict[str, Any]) -> Path:
    """Atomically commit a manifest dict.

    Sequence:
      1. write candidate.json (with fsync)
      2. promote prior: rename existing current.json -> prior.json
      3. promote candidate: rename candidate.json -> current.json

    Steps 2 and 3 use os.replace, which is atomic on Windows + POSIX
    (Python 3.3+).

    Returns the path to the committed current.json.
    """
    manifest_dir = project_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    candidate = manifest_dir / CANDIDATE_FILE
    current = manifest_dir / CURRENT_FILE
    prior = manifest_dir / PRIOR_FILE

    # Step 1: write candidate with explicit fsync so the kernel actually
    # flushes to disk before we proceed to the rename.
    payload = json.dumps(manifest, indent=2, sort_keys=True)
    with open(candidate, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())

    # Step 2: promote the existing current only when it is still a verified
    # recovery point. If current is corrupt while prior remains valid,
    # rotating current over prior would destroy the only usable fallback if
    # the candidate -> current rename then fails.
    if current.exists():
        current_is_verified = False
        try:
            current_manifest = read_current(project_dir)
            current_is_verified = (
                verify_manifest(project_dir, current_manifest) is None
            )
        except Exception as exc:  # noqa: BLE001 - verification must fail closed
            LOG.warning(
                "existing current manifest could not be verified before "
                "rotation: %s",
                exc,
            )
        if current_is_verified:
            os.replace(current, prior)
        else:
            LOG.warning(
                "existing current manifest is unverified; preserving prior "
                "during candidate promotion"
            )

    # Step 3: atomic commit of new manifest as current.
    os.replace(candidate, current)
    LOG.info(
        "epoch-manifest committed: epoch_id=%s artifacts=%d",
        manifest["epoch_id"], len(manifest["artifacts"]),
    )
    return current


def read_current(project_dir: Path) -> Dict[str, Any]:
    """Read the committed current manifest. Raises ManifestMissing if none."""
    path = project_dir / "manifest" / CURRENT_FILE
    if not path.exists():
        raise ManifestMissing(f"No committed manifest at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_prior(project_dir: Path) -> Optional[Dict[str, Any]]:
    """Read the prior manifest, or None if no prior epoch was ever committed."""
    path = project_dir / "manifest" / PRIOR_FILE
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_manifest(project_dir: Path, manifest: Dict[str, Any]) -> Optional[str]:
    """Verify every artifact in the manifest matches its recorded sha256.

    Returns None on success, or a string describing the first mismatch.
    """
    if not isinstance(manifest, dict):
        return "manifest is not an object"
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return "manifest has no artifacts"
    for name, entry in artifacts.items():
        if not isinstance(entry, dict):
            return f"invalid artifact entry: {name}"
        rel = entry.get("path")
        expected_sha = entry.get("sha256")
        if not isinstance(rel, str) or not rel:
            return f"invalid artifact path: {name}"
        if not isinstance(expected_sha, str) or not expected_sha:
            return f"invalid artifact sha256: {name}"
        path = project_dir / rel
        if not path.exists():
            return f"artifact missing: {rel}"
        actual = _sha256_file(path)
        if actual != expected_sha:
            return (
                f"sha256 mismatch for {rel}: "
                f"manifest={expected_sha} actual={actual}"
            )
    return None


@dataclass
class ReadResult:
    """Outcome of read_with_fallback. Stable string vocabulary in `freshness`.

    Plan-2 E3 (2026-05-05).
    """
    manifest: Optional[Dict[str, Any]]
    freshness: str  # "fresh" | "stale_using_prior_epoch" | "missing" | "corrupt"
    detail: str = ""

    @property
    def ok(self) -> bool:
        """Whether the caller has a usable manifest."""
        return self.manifest is not None


def read_with_fallback(project_dir: Path) -> ReadResult:
    """Read the current epoch with downgrade-tolerance to the prior epoch.

    Algorithm (Plan-2 E3):
      1. Load `current.json`.
         - Missing: try prior.json. If also missing → ReadResult(None, "missing").
      2. Verify current's artifact checksums.
         - Pass: return ReadResult(current, "fresh").
         - Fail: log warning, try prior.json.
      3. Load `prior.json`.
         - Missing → ReadResult(None, "corrupt", detail=verify error).
      4. Verify prior's artifact checksums.
         - Pass: return ReadResult(prior, "stale_using_prior_epoch").
         - Fail: ReadResult(None, "corrupt", detail="both failed").

    Critically: never falls open to a partial / in-progress epoch. Either
    the caller gets a verified manifest (current OR prior) or an explicit
    error indication that points at verify_index_integrity for repair.

    Returned `freshness` strings are intended for caller-side propagation
    into `_metadata.freshness` on search responses. Stable vocabulary —
    downstream consumers may pattern-match.
    """
    # Step 1: load current. A malformed JSON file is a failed current, not a
    # reason to skip an otherwise verified prior recovery point.
    current_missing = False
    current_read_error = ""
    try:
        current = read_current(project_dir)
    except ManifestMissing:
        current = None
        current_missing = True
        current_read_error = "current.json missing"
    except (
        json.JSONDecodeError,
        OSError,
        UnicodeDecodeError,
    ) as exc:
        current = None
        current_read_error = (
            f"current.json unreadable: {type(exc).__name__}: {exc}"
        )

    if current is None:
        try:
            prior = read_prior(project_dir)
        except (
            json.JSONDecodeError,
            OSError,
            UnicodeDecodeError,
        ) as exc:
            return ReadResult(
                None,
                "corrupt",
                detail=(
                    f"{current_read_error}; prior.json unreadable: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        if prior is None:
            if current_missing:
                return ReadResult(
                    None,
                    "missing",
                    detail="no current or prior manifest",
                )
            return ReadResult(
                None,
                "corrupt",
                detail=f"{current_read_error}; no prior manifest",
            )
        prior_err = verify_manifest(project_dir, prior)
        if prior_err is None:
            return ReadResult(
                prior, "stale_using_prior_epoch",
                detail=f"{current_read_error}; using prior",
            )
        return ReadResult(
            None, "corrupt",
            detail=f"{current_read_error}; prior failed: {prior_err}",
        )

    # Step 2: verify current
    err = verify_manifest(project_dir, current)
    if err is None:
        return ReadResult(current, "fresh")

    # Step 3-4: current failed verify, fall back to prior
    LOG.warning("current manifest failed verification: %s; trying prior", err)
    try:
        prior = read_prior(project_dir)
    except (
        json.JSONDecodeError,
        OSError,
        UnicodeDecodeError,
    ) as exc:
        return ReadResult(
            None,
            "corrupt",
            detail=(
                f"current failed: {err}; prior.json unreadable: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
    if prior is None:
        return ReadResult(None, "corrupt", detail=f"current failed: {err}; no prior")
    prior_err = verify_manifest(project_dir, prior)
    if prior_err is None:
        return ReadResult(
            prior, "stale_using_prior_epoch",
            detail=f"current failed: {err}",
        )
    return ReadResult(
        None, "corrupt",
        detail=f"both failed — current: {err} | prior: {prior_err}",
    )


def cleanup_stale_candidate(project_dir: Path) -> bool:
    """Remove a stale candidate.json (e.g., from a crashed prior write).

    Returns True if a stale file was removed, False if none was present.
    Idempotent. Safe to call any time.
    """
    candidate = project_dir / "manifest" / CANDIDATE_FILE
    if candidate.exists():
        candidate.unlink()
        LOG.info("cleaned up stale candidate.json at %s", candidate)
        return True
    return False
