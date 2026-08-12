"""Vector index management with FAISS and metadata storage."""

import fnmatch
import errno
import hashlib
import os
import json
import pickle
import logging
import functools
import secrets
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from search.metadata_store import JsonSqliteKV, LegacyMetadataFormatError
from search.logging_privacy import (
    format_query_exception_for_log,
    format_query_for_log,
)
from embeddings.embedder import EffectiveEmbeddingConfig, EmbeddingResult


def _install_search_file_handler() -> None:
    """Attach a FileHandler that captures structured diagnostic lines to disk.

    Cross-platform sidecar logger (Plan-2 A2 audit, 2026-05-05): works on
    Windows, Linux, and macOS via portable `Path.home()` resolution. See
    docs/cross_platform_observability.md for the full audit.

    Why a sidecar (not just stderr):
      Windows: the MCP server runs under pythonw.exe with no console; stderr
        is discarded entirely.
      Linux/macOS: stderr IS captured by the parent (Claude Code or another
        MCP transport), but it's interleaved with the rest of the process
        output and is ephemeral. The sidecar gives operators a persistent,
        filtered, `tail -f`-able log file regardless of platform.

    Captured prefixes (filter accepts any line containing one of these):
      [CHUNK_ID_DIAG]      — load/save state diagnostics in indexer
      [REINDEX_PROGRESS]   — incremental_index progress milestones
      [ANTHROPIC_DIAG]     — per-call Sonnet rerank latency (Plan D1-Pass-2 A.1)

    Attaches to the `search` parent logger so children
    (`search.indexer`, `search.incremental_indexer`) propagate up and
    share the handler. Idempotent: a marker attribute on the handler
    prevents stacking on re-import.

    Output (all platforms): ~/.claude/logs/code-search-mcp.log
      Windows: C:\\Users\\<user>\\.claude\\logs\\code-search-mcp.log
      Linux:   /home/<user>/.claude/logs/code-search-mcp.log
      macOS:   /Users/<user>/.claude/logs/code-search-mcp.log

    Degrades silently if the log directory cannot be created (we never
    want logging setup to break the indexer).
    """
    try:
        log_dir = Path.home() / ".claude" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "code-search-mcp.log"
    except Exception:
        return

    # Attach to the package logger so both indexer and incremental_indexer
    # log lines flow to the same sidecar.
    logger = logging.getLogger("search")
    target = str(log_path.resolve())
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "_chunk_id_diag", False):
            return
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == target:
            return

    try:
        handler = logging.FileHandler(target, mode="a", encoding="utf-8")
    except Exception:
        return
    handler._chunk_id_diag = True  # type: ignore[attr-defined]
    handler.setLevel(logging.DEBUG)

    _ACCEPTED_PREFIXES = (
        "[CHUNK_ID_DIAG]",
        "[REINDEX_PROGRESS]",
        "[ANTHROPIC_DIAG]",
        # Phase A1 (2026-05-10): per-cohort override-trigger records,
        # emitted by _effective_threshold in sonnet_reranker.py when
        # SONNET_RERANKER_LOG_OVERRIDE_TRIGGERS=1. Used by
        # paired_bootstrap_per_subproject.py to count spillover.
        "[PATH_OVERRIDE_TRIGGER]",
        # Phase A1 (2026-05-11): per-cohort reranker outcome records,
        # emitted by _rerank_async in sonnet_reranker.py for non-OK
        # outcomes (hybrid_prior_fallback, timeout, too_many_failures).
        # Promoted from LOG.debug to LOG.info to close the silent-fallback
        # observability gap surfaced in the 2026-05-10 Phase B audit.
        "[RERANK_REASON]",
        # Arc A (2026-05-11): per-search PPR diagnostic emitted by
        # search/ppr_scorer.py — db-not-found / insufficient-subgraph /
        # computed t_ms records. Same observability pattern as
        # RERANK_REASON / ANTHROPIC_DIAG.
        "[PPR_DIAG]",
    )

    class _SearchDiagFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
            except Exception:
                return False
            return any(p in msg for p in _ACCEPTED_PREFIXES)

    handler.addFilter(_SearchDiagFilter())
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    if logger.level == logging.NOTSET or logger.level > logging.WARNING:
        logger.setLevel(logging.WARNING)

    # [ANTHROPIC_DIAG] is INFO-level (per Plan D1-Pass-2 A.1). The parent
    # `search` logger above stays at WARNING so unrelated INFO chatter
    # doesn't fill the sidecar; we elevate ONLY the reranker child logger
    # to INFO so its [ANTHROPIC_DIAG] records can reach the sidecar handler.
    # The filter still gates on the prefix, so other reranker INFO logs
    # (judge prompts, etc.) are dropped.
    sonnet_logger = logging.getLogger("search.sonnet_reranker")
    if sonnet_logger.level == logging.NOTSET or sonnet_logger.level > logging.INFO:
        sonnet_logger.setLevel(logging.INFO)


# Public alias for backward compat with tests that imported the v1 name.
_install_chunk_id_diag_file_handler = _install_search_file_handler

_install_search_file_handler()

class _ProcessAwareRLock:
    """Re-entrant process-local lock that discards ownership after fork."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pid = os.getpid()
        self._context_pids: list[int] = []

    def _reset_after_fork(self) -> None:
        current_pid = os.getpid()
        if current_pid == self._pid:
            return
        self._lock = threading.RLock()
        self._pid = current_pid

    def acquire(self) -> None:
        self._reset_after_fork()
        self._lock.acquire()

    def release(self, *, acquired_pid: int | None = None) -> None:
        current_pid = os.getpid()
        if acquired_pid is not None and acquired_pid != current_pid:
            if current_pid != self._pid:
                self._reset_after_fork()
            return
        if current_pid != self._pid:
            # This release belongs to a context inherited across fork, not
            # to an acquisition made by the child. Discard the copied lock
            # state without trying to unlock the parent's ownership.
            self._reset_after_fork()
            return
        self._lock.release()

    def __enter__(self) -> "_ProcessAwareRLock":
        self.acquire()
        self._context_pids.append(os.getpid())
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if not self._context_pids:
            raise RuntimeError("process-aware lock context stack is empty")
        self.release(acquired_pid=self._context_pids.pop())


_STORAGE_LOCKS_GUARD = _ProcessAwareRLock()
_STORAGE_LOCKS: dict[str, _ProcessAwareRLock] = {}
_WRITER_LOCKS: dict[str, "_InterProcessWriterLock"] = {}


class _InterProcessWriterLock:
    """Re-entrant process and filesystem lock for one index directory."""

    def __init__(self, path: Path):
        self.path = path
        self._thread_lock = threading.RLock()
        self._depth = 0
        self._owner: int | None = None
        self._handle = None
        self._pid = os.getpid()
        self._context_pids: list[int] = []

    def _reset_after_fork(self) -> None:
        """Discard inherited process-local state without unlocking parent."""
        current_pid = os.getpid()
        if current_pid == self._pid:
            return
        if self._handle is not None:
            # Do not issue LOCK_UN: on POSIX the inherited descriptor shares
            # its open-file description with the parent and could release
            # the parent's lock. Closing only this duplicate is safe.
            self._handle.close()
        self._thread_lock = threading.RLock()
        self._depth = 0
        self._owner = None
        self._handle = None
        self._pid = current_pid

    def _acquire_filesystem_lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                while True:
                    handle.seek(0)
                    try:
                        msvcrt.locking(
                            handle.fileno(),
                            msvcrt.LK_NBLCK,
                            1,
                        )
                        break
                    except OSError as exc:
                        if exc.errno not in {
                            errno.EACCES,
                            errno.EAGAIN,
                            errno.EDEADLK,
                        }:
                            raise
                        time.sleep(0.05)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            return handle
        except BaseException:
            handle.close()
            raise

    @staticmethod
    def _release_filesystem_lock(handle) -> None:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def acquire(self) -> None:
        self._reset_after_fork()
        self._thread_lock.acquire()
        try:
            owner = threading.get_ident()
            if self._depth == 0:
                self._handle = self._acquire_filesystem_lock()
                self._owner = owner
            elif self._owner != owner:
                raise RuntimeError(
                    "inter-process writer lock ownership is inconsistent"
                )
            self._depth += 1
        except BaseException:
            self._thread_lock.release()
            raise

    def release(self, *, acquired_pid: int | None = None) -> None:
        current_pid = os.getpid()
        if acquired_pid is not None and acquired_pid != current_pid:
            if current_pid != self._pid:
                self._reset_after_fork()
            return
        if current_pid != self._pid:
            # Never issue LOCK_UN for a context inherited from the parent:
            # the descriptor shares its open-file description and doing so
            # would release the parent's live writer lock.
            self._reset_after_fork()
            return
        if (
            self._depth <= 0
            or self._owner != threading.get_ident()
            or self._handle is None
        ):
            raise RuntimeError(
                "inter-process writer lock released by a non-owner"
            )
        self._depth -= 1
        try:
            if self._depth == 0:
                handle = self._handle
                self._handle = None
                self._owner = None
                self._release_filesystem_lock(handle)
        finally:
            self._thread_lock.release()

    def is_reentrant_acquisition(self) -> bool:
        """Return whether the current owner holds this lock more than once."""
        return (
            self._owner == threading.get_ident()
            and self._depth > 1
        )

    def __enter__(self) -> "_InterProcessWriterLock":
        self.acquire()
        self._context_pids.append(os.getpid())
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if not self._context_pids:
            raise RuntimeError("writer lock context stack is empty")
        self.release(acquired_pid=self._context_pids.pop())


def _shared_storage_lock(storage_dir: Path) -> _ProcessAwareRLock:
    """Return the process-wide lock for one canonical index directory."""
    key = os.path.normcase(str(storage_dir.resolve()))
    with _STORAGE_LOCKS_GUARD:
        lock = _STORAGE_LOCKS.get(key)
        if lock is None:
            lock = _ProcessAwareRLock()
            _STORAGE_LOCKS[key] = lock
        return lock


def _shared_writer_lock(storage_dir: Path) -> _InterProcessWriterLock:
    """Return the cross-process writer lock for one canonical directory."""
    canonical = storage_dir.resolve()
    key = os.path.normcase(str(canonical))
    with _STORAGE_LOCKS_GUARD:
        lock = _WRITER_LOCKS.get(key)
        if lock is None:
            lock = _InterProcessWriterLock(
                canonical / ".code-search-writer.lock"
            )
            _WRITER_LOCKS[key] = lock
        return lock


def _with_storage_lock(method):
    """Serialize operations that share FAISS and SQLite handles."""

    @functools.wraps(method)
    def locked(self, *args, **kwargs):
        with self._storage_lock:
            return method(self, *args, **kwargs)

    return locked


def _with_writer_and_storage_lock(method):
    """Serialize a destructive operation across threads and processes."""

    @functools.wraps(method)
    def locked(self, *args, **kwargs):
        # Match publication_transaction's lock order to avoid deadlocks.
        with self._writer_lock:
            with self._storage_lock:
                return method(self, *args, **kwargs)

    return locked


class IndexPublicationRefused(RuntimeError):
    """Raised when a defensive guard refuses to publish an index."""


class CodeIndexManager:
    """Manages FAISS vector index and metadata storage for code chunks."""

    # P5 (2026-06-10 roadmap): stale-vector compaction thresholds. FAISS rows
    # are never removed in place (removal is "rebuild on demand"), so
    # modify/delete churn accumulates stale vectors. ADVISORY surfaces in
    # search `_metadata` and verify_index_integrity; COMPACTION escalates an
    # incremental index run to a full reindex (which clears the index and
    # resets the ratio to 0 — self-limiting). Hard-coded, not env knobs.
    STALE_ADVISORY_RATIO = 0.25
    STALE_COMPACTION_RATIO = 0.5

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self._storage_lock = _shared_storage_lock(self.storage_dir)
        with self._storage_lock:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._writer_lock = _shared_writer_lock(self.storage_dir)
        self._logger = logging.getLogger(__name__)
        
        # File paths
        self.index_path = self.storage_dir / "code.index"
        self.metadata_path = self.storage_dir / "metadata.db" 
        self.chunk_id_path = self.storage_dir / "chunk_ids.pkl"
        self.stats_path = self.storage_dir / "stats.json"
        self._fts_db_path = self.storage_dir / "fts5.db"
        self._generation_root = self.storage_dir / ".generations"
        self._publication_marker = (
            self.storage_dir / ".publication-in-progress"
        )
        
        # Initialize components
        self._index = None
        self._metadata_db = None
        self._chunk_ids = []
        self._on_gpu = False

        # Observability: status of the most recent _commit_epoch_manifest call.
        # Stable string vocabulary: "ok", "skipped_empty", "consistency_error",
        # "build_error", "commit_error", or None (no commit attempted yet).
        # Callers (incremental_indexer, MCP layer) can surface this in
        # `_metadata` to distinguish silent-success-with-stale-manifest from
        # true success.
        self.last_manifest_commit_status: Optional[str] = None

        # Complete recovery before SQLite opens root-level compatibility
        # mirrors. The marker exists only across the non-atomic series of
        # mirror replacements and the final manifest commit.
        with self._writer_lock:
            with self._storage_lock:
                self._recover_published_generation()
                self._upgrade_legacy_manifest_generation()

                # Initialize FTS5
                self._init_fts5()

    @contextmanager
    def publication_transaction(self):
        """Hold writer locks across mutation, publication, or rollback.

        Individual manager methods also take the process-local storage lock.
        The outer scope adds a filesystem lock because an index mutation spans
        several method calls (remove/begin, add, then save). Without it,
        another manager or MCP process for the same storage directory can
        mistake the live publication marker for a crashed writer and restore
        the prior generation mid-rebuild.
        """
        with self._writer_lock:
            if self._writer_lock.is_reentrant_acquisition():
                raise IndexPublicationRefused(
                    "Nested index publication transactions are not "
                    "supported; use one outer transaction for the complete "
                    "mutation and publication"
                )
            with self._storage_lock:
                self._rebase_publication_working_set()
                transaction_pid = os.getpid()
                try:
                    yield
                except BaseException:
                    if os.getpid() != transaction_pid:
                        raise
                    if self._publication_marker.exists():
                        self.rollback_unpublished_changes()
                    raise
                if os.getpid() != transaction_pid:
                    return
                if self._publication_marker.exists():
                    self.rollback_unpublished_changes()
                    raise IndexPublicationRefused(
                        "Index publication transaction exited without "
                        "committing or rolling back its working set"
                    )

    def _rebase_publication_working_set(self) -> None:
        """Reopen every mutable view after acquiring the filesystem lock.

        A long-lived manager can retain FAISS state and SQLite connections
        from before another process publishes a generation.  The writer lock
        serializes publication but does not make those in-memory views fresh,
        so discard them only after this process owns that lock.  A marker
        found here belongs to a writer that exited without releasing a clean
        publication point and is recovered before the authoritative mirrors
        are reopened.
        """
        self._close_storage_handles()
        self._recover_published_generation()
        self._upgrade_legacy_manifest_generation()
        self._repair_root_mirrors_from_verified_generation()
        self._index = None
        self._chunk_ids = []
        self._is_binary = False
        self._on_gpu = False
        if hasattr(self, "_float_store"):
            del self._float_store
        self._init_fts5()

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _root_mirrors_match_generation(
        self,
        manifest: dict[str, Any],
    ) -> bool:
        """Return whether every compatibility mirror matches its generation."""
        destinations = {
            "code.index": self.index_path,
            "chunk_ids.pkl": self.chunk_id_path,
            "metadata.db": self.metadata_path,
            "fts5.db": self._fts_db_path,
            "stats.json": self.stats_path,
            "float_store.npy": self.storage_dir / "float_store.npy",
        }
        artifacts = manifest.get("artifacts", {})
        try:
            for name, destination in destinations.items():
                entry = artifacts.get(name)
                if entry is None:
                    if destination.exists():
                        return False
                    continue
                if (
                    not destination.is_file()
                    or self._sha256_file(destination)
                    != entry.get("sha256")
                ):
                    return False

            for destination in (self.metadata_path, self._fts_db_path):
                for suffix in ("-wal", "-shm", "-journal"):
                    if Path(f"{destination}{suffix}").exists():
                        return False
        except OSError:
            return False
        return True

    def _repair_root_mirrors_from_verified_generation(self) -> None:
        """Repair missing or damaged roots before a writer can mutate them."""
        from search.epoch_manifest import read_with_fallback

        publication = read_with_fallback(self.storage_dir)
        manifest = publication.manifest
        if manifest is None or not self._manifest_uses_generation(manifest):
            return
        self._validate_published_generation(manifest)
        if self._root_mirrors_match_generation(manifest):
            return

        self._logger.warning(
            "Root index mirrors differ from verified epoch %s; restoring "
            "them before publication",
            manifest.get("epoch_id"),
        )
        self._write_publication_marker(manifest)
        self._materialize_generation(manifest)
        self._clear_publication_marker()
        self._prune_unreferenced_generations()

    @_with_storage_lock
    def has_persisted_index_state(self) -> bool:
        """Return whether storage contains data that an append could reuse.

        A newly constructed manager creates an empty FTS database, so file
        presence alone cannot distinguish a genuinely empty store. All other
        root artifacts are conservative signals, while FTS is checked for a
        live row. Unreadable state fails closed.
        """
        import sqlite3

        root_artifacts = (
            self.index_path,
            self.chunk_id_path,
            self.metadata_path,
            self.stats_path,
            self.storage_dir / "float_store.npy",
        )
        for artifact in root_artifacts:
            if any(
                path.exists()
                for path in (
                    artifact,
                    Path(f"{artifact}-wal"),
                    Path(f"{artifact}-shm"),
                )
            ):
                return True

        for directory in (
            self.storage_dir / "manifest",
            self._generation_root,
        ):
            try:
                if directory.exists() and next(directory.iterdir(), None):
                    return True
            except OSError:
                return True

        if not self._fts_db_path.exists():
            return False
        if not hasattr(self, "_fts_conn") or self._fts_conn is None:
            return True
        try:
            return (
                self._fts_conn.execute(
                    "SELECT 1 FROM chunk_fts LIMIT 1"
                ).fetchone()
                is not None
            )
        except (OSError, sqlite3.Error):
            return True

    @staticmethod
    def _fsync_file(path: Path) -> None:
        """Flush a completed candidate artifact before it can be published."""
        with path.open("rb") as handle:
            os.fsync(handle.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        """Persist directory-entry changes on platforms that support it."""
        if os.name == "nt":
            # Windows does not support opening directories for fsync. File
            # handles are still flushed and os.replace remains atomic there.
            return
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _sqlite_integrity_check(path: Path) -> None:
        """Raise if a staged SQLite snapshot cannot be opened cleanly."""
        import sqlite3

        connection = sqlite3.connect(str(path))
        try:
            row = connection.execute("PRAGMA integrity_check").fetchone()
        finally:
            connection.close()
        if not row or row[0] != "ok":
            detail = row[0] if row else "no result"
            raise RuntimeError(
                f"SQLite integrity check failed for {path}: {detail}"
            )

    def _manifest_artifact_path(
        self, manifest: dict[str, Any], artifact_name: str
    ) -> Path | None:
        entry = manifest.get("artifacts", {}).get(artifact_name)
        if not entry:
            return None
        return self.storage_dir / entry["path"]

    def _manifest_uses_generation(self, manifest: dict[str, Any]) -> bool:
        prefix = self._generation_root.name + "/"
        artifacts = manifest.get("artifacts", {})
        return bool(artifacts) and all(
            str(entry.get("path", "")).replace("\\", "/").startswith(prefix)
            for entry in artifacts.values()
        )

    def _validate_published_generation(
        self, manifest: dict[str, Any]
    ) -> None:
        """Validate hashes, persisted counts, FAISS readability, and SQLite."""
        from search.epoch_manifest import verify_manifest

        verification_error = verify_manifest(self.storage_dir, manifest)
        if verification_error is not None:
            raise RuntimeError(
                f"Refusing invalid published index generation: "
                f"{verification_error}"
            )

        index_path = self._manifest_artifact_path(manifest, "code.index")
        chunk_ids_path = self._manifest_artifact_path(
            manifest, "chunk_ids.pkl"
        )
        expected = manifest.get("consistency", {}).get("expected_count")
        if index_path is None or chunk_ids_path is None:
            if expected not in (None, 0):
                raise RuntimeError(
                    "Published generation is missing code.index or "
                    "chunk_ids.pkl"
                )
        else:
            with chunk_ids_path.open("rb") as handle:
                persisted_chunk_ids = pickle.load(handle)
            if not isinstance(persisted_chunk_ids, list):
                raise RuntimeError(
                    f"{chunk_ids_path} did not contain a chunk-id list"
                )

            float_path = self._manifest_artifact_path(
                manifest, "float_store.npy"
            )
            if float_path is not None:
                persisted_index = faiss.read_index_binary(str(index_path))
                float_store = np.load(str(float_path), allow_pickle=False)
                if len(float_store) != len(persisted_chunk_ids):
                    raise RuntimeError(
                        "Persisted float_store row count does not match "
                        "chunk_ids.pkl"
                    )
            else:
                persisted_index = faiss.read_index(str(index_path))

            persisted_count = int(persisted_index.ntotal)
            if persisted_count != len(persisted_chunk_ids):
                raise RuntimeError(
                    "Persisted FAISS ntotal does not match chunk_ids.pkl: "
                    f"{persisted_count} != {len(persisted_chunk_ids)}"
                )
            if expected is not None and persisted_count != int(expected):
                raise RuntimeError(
                    "Persisted FAISS ntotal does not match manifest: "
                    f"{persisted_count} != {expected}"
                )

        for database_name in ("metadata.db", "fts5.db"):
            database_path = self._manifest_artifact_path(
                manifest, database_name
            )
            if database_path is not None:
                self._sqlite_integrity_check(database_path)

    def _close_storage_handles(self) -> None:
        """Close SQLite handles before cross-platform file replacement."""
        if self._metadata_db is not None:
            self._metadata_db.close()
            self._metadata_db = None
        if getattr(self, "_fts_conn", None) is not None:
            self._fts_conn.close()
            self._fts_conn = None

    def _materialize_generation(self, manifest: dict[str, Any]) -> None:
        """Atomically refresh root-level compatibility mirrors."""
        destinations = {
            "code.index": self.index_path,
            "chunk_ids.pkl": self.chunk_id_path,
            "metadata.db": self.metadata_path,
            "fts5.db": self._fts_db_path,
            "stats.json": self.stats_path,
            "float_store.npy": self.storage_dir / "float_store.npy",
        }
        token = secrets.token_hex(8)
        prepared: list[tuple[Path, Path]] = []
        absent: list[Path] = []
        try:
            for name, destination in destinations.items():
                source = self._manifest_artifact_path(manifest, name)
                if source is None:
                    absent.append(destination)
                    continue
                if source == destination:
                    continue
                temporary = destination.with_name(
                    f".{destination.name}.publish-{token}"
                )
                shutil.copy2(source, temporary)
                self._fsync_file(temporary)
                prepared.append((temporary, destination))

            if prepared:
                self._fsync_directory(self.storage_dir)
            self._close_storage_handles()

            # A process crash can leave committed WAL/SHM bytes beside a
            # root database. Replacing only the main database would let
            # SQLite replay that unpublished working state over the restored
            # generation on reopen. Remove every SQLite journal sidecar after
            # handles close and before replacing the main files.
            removed_sqlite_sidecar = False
            for destination in (self.metadata_path, self._fts_db_path):
                for suffix in ("-wal", "-shm", "-journal"):
                    try:
                        Path(f"{destination}{suffix}").unlink()
                        removed_sqlite_sidecar = True
                    except FileNotFoundError:
                        pass
            if removed_sqlite_sidecar:
                self._fsync_directory(self.storage_dir)

            removed_absent = False
            for destination in absent:
                try:
                    destination.unlink()
                    removed_absent = True
                except FileNotFoundError:
                    pass
            for temporary, destination in prepared:
                os.replace(temporary, destination)

            if prepared or removed_absent:
                self._fsync_directory(self.storage_dir)
        finally:
            for temporary, _ in prepared:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

    def _write_publication_marker(self, manifest: dict[str, Any]) -> None:
        """Durably mark that root mirrors may temporarily be inconsistent."""
        temporary = self._publication_marker.with_name(
            f".{self._publication_marker.name}-{secrets.token_hex(8)}"
        )
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "candidate_epoch": manifest.get("epoch_id"),
                        "candidate_artifacts": len(
                            manifest.get("artifacts", {})
                        ),
                    },
                    handle,
                    sort_keys=True,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._publication_marker)
            self._fsync_directory(self.storage_dir)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _clear_publication_marker(self) -> None:
        try:
            self._publication_marker.unlink()
            self._fsync_directory(self.storage_dir)
        except FileNotFoundError:
            pass

    def _mark_working_set_dirty(self) -> None:
        """Ensure restart can roll unpublished root-store changes back."""
        if self._publication_marker.exists():
            return

        from search.epoch_manifest import read_with_fallback

        publication = read_with_fallback(self.storage_dir)
        if (
            publication.manifest is None
            and publication.freshness == "missing"
            and self.index_path.exists()
            and self.chunk_id_path.exists()
        ):
            self._bootstrap_unmanifested_generation()
        self._write_publication_marker({})

    def _bootstrap_unmanifested_generation(self) -> None:
        """Snapshot a healthy pre-manifest index before its first mutation."""
        if self._index is None:
            self._load_index()
        if self._index is None:
            raise RuntimeError(
                "Cannot preserve unmanifested index because FAISS did not load"
            )

        self._generation_root.mkdir(parents=True, exist_ok=True)
        self._fsync_directory(self.storage_dir)
        token = f"legacy-unmanifested-{secrets.token_hex(12)}"
        candidate_dir = self._generation_root / f".candidate-{token}"
        generation_dir = self._generation_root / token
        committed = False
        try:
            self._write_candidate_generation(candidate_dir)
            candidate_manifest = self._build_generation_manifest(
                candidate_dir
            )
            self._validate_published_generation(candidate_manifest)
            os.replace(candidate_dir, generation_dir)
            self._fsync_directory(self._generation_root)
            manifest = self._build_generation_manifest(generation_dir)
            self._validate_published_generation(manifest)
            self._commit_epoch_manifest(manifest)
            committed = True
        finally:
            if candidate_dir.exists():
                self._remove_generation_path(
                    candidate_dir, ignore_errors=True
                )
            if (
                not committed
                and generation_dir.exists()
                and not self._generation_is_manifest_referenced(
                    generation_dir
                )
            ):
                self._remove_generation_path(
                    generation_dir, ignore_errors=True
                )
        self._prune_unreferenced_generations()

    def _recover_published_generation(self) -> None:
        """Restore mirrors from the verified publication after a prior crash."""
        from search.epoch_manifest import read_with_fallback

        if not self._publication_marker.exists():
            return
        result = read_with_fallback(self.storage_dir)
        if result.manifest is None:
            if result.freshness == "missing":
                # The first publication never reached its commit point.
                # Nothing on disk is authoritative, so discard the working
                # roots and any staged generation instead of wedging startup.
                self._discard_unpublished_working_set()
                if self._generation_root.exists():
                    shutil.rmtree(self._generation_root)
                    self._fsync_directory(self.storage_dir)
                self._clear_publication_marker()
                return
            raise RuntimeError(
                "Interrupted index publication has no verified committed "
                "generation to restore"
            )
        if not self._manifest_uses_generation(result.manifest):
            raise RuntimeError(
                "Interrupted index publication references a legacy manifest; "
                "refusing unverified root-level mirrors"
            )
        self._validate_published_generation(result.manifest)
        self._materialize_generation(result.manifest)
        self._clear_publication_marker()
        self._prune_unreferenced_generations()

    def _upgrade_legacy_manifest_generation(self) -> None:
        """Repoint a verified root-path manifest at an immutable snapshot."""
        from search.epoch_manifest import (
            ManifestMissing,
            read_current,
            verify_manifest,
        )

        try:
            manifest = read_current(self.storage_dir)
        except (
            ManifestMissing,
            json.JSONDecodeError,
            OSError,
            UnicodeDecodeError,
        ):
            return
        if self._manifest_uses_generation(manifest):
            return

        verification_error = verify_manifest(self.storage_dir, manifest)
        if verification_error is not None:
            self._logger.warning(
                "Legacy index manifest is not verifiable; leaving its "
                "root-level layout unchanged: %s",
                verification_error,
            )
            return

        self._generation_root.mkdir(parents=True, exist_ok=True)
        self._fsync_directory(self.storage_dir)
        token = secrets.token_hex(12)
        candidate_dir = self._generation_root / f".legacy-{token}"
        generation_dir = self._generation_root / f"legacy-{token}"
        manifest_dir = self.storage_dir / "manifest"
        manifest_candidate = manifest_dir / f".upgrade-{token}.json"
        current_path = manifest_dir / "current.json"
        promoted = False
        try:
            candidate_dir.mkdir(parents=True, exist_ok=False)
            upgraded = json.loads(json.dumps(manifest))
            used_names: set[str] = set()
            for artifact_name, entry in upgraded["artifacts"].items():
                source = self.storage_dir / entry["path"]
                filename = Path(entry["path"]).name
                if filename in used_names:
                    filename = artifact_name.replace("/", "_")
                used_names.add(filename)
                destination = candidate_dir / filename
                shutil.copy2(source, destination)
                self._fsync_file(destination)
                entry["path"] = str(
                    (generation_dir / filename).relative_to(self.storage_dir)
                ).replace("\\", "/")

            self._fsync_directory(candidate_dir)
            os.replace(candidate_dir, generation_dir)
            self._fsync_directory(self._generation_root)
            self._validate_published_generation(upgraded)

            with manifest_candidate.open("w", encoding="utf-8") as handle:
                json.dump(upgraded, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(manifest_candidate, current_path)
            self._fsync_directory(manifest_dir)
            promoted = True
            self._logger.info(
                "Upgraded legacy index manifest epoch=%s to generation %s",
                upgraded.get("epoch_id"),
                generation_dir,
            )
        finally:
            try:
                manifest_candidate.unlink()
            except FileNotFoundError:
                pass
            if candidate_dir.exists():
                self._remove_generation_path(
                    candidate_dir, ignore_errors=True
                )
            if (
                not promoted
                and generation_dir.exists()
                and not self._generation_is_manifest_referenced(
                    generation_dir
                )
            ):
                self._remove_generation_path(
                    generation_dir, ignore_errors=True
                )
        if promoted:
            self._prune_unreferenced_generations()
    def bind_embedding_configuration(
        self,
        configuration: EffectiveEmbeddingConfig,
        *,
        pipeline_version: str,
    ) -> None:
        """Bind the exact embedding identity used for the next index commit."""
        dimension = configuration.output_dimension
        if (
            not configuration.provider
            or not configuration.model_name
            or isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            or not pipeline_version
        ):
            raise ValueError(
                "Index commits require a complete effective embedding "
                "provider/model/dimension/pipeline identity"
            )
        self._embedder_provider = configuration.provider
        self._embedder_model = configuration.model_name
        self._embedder_dimension = dimension
        self._embedder_input_type_enabled = (
            configuration.input_type_enabled
        )
        self._pipeline_version = pipeline_version

    def _init_fts5(self):
        """Initialize FTS5 full-text search table.

        Corruption-hardened (2026-06-10 torn-write fuzz): a truncated or
        garbage fts5.db raised sqlite3.DatabaseError HERE — in the
        constructor — making the manager unconstructable until manual
        cleanup. FTS5 is derived data (rebuilt by reindex), so corruption
        quarantines the bad file and recreates an empty table: BM25 leg
        degrades until the next reindex, search keeps working.
        """
        import sqlite3
        for attempt in (1, 2):
            try:
                self._fts_conn = sqlite3.connect(
                    str(self._fts_db_path),
                    check_same_thread=False,
                )
                self._fts_conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                        chunk_id,
                        content,
                        file_path,
                        name,
                        tokenize='porter unicode61'
                    )
                """)
                self._fts_conn.commit()
                return
            except sqlite3.DatabaseError as e:
                try:
                    self._fts_conn.close()
                except Exception:
                    pass
                self._fts_conn = None
                if attempt == 2:
                    self._logger.error(
                        "fts5.db unusable after quarantine (%s); BM25 leg "
                        "disabled until reindex", e,
                    )
                    return
                import time
                quarantine = self._fts_db_path.with_suffix(
                    f".db.corrupt.{time.strftime('%Y%m%dT%H%M%S')}"
                )
                try:
                    self._fts_db_path.rename(quarantine)
                except OSError:
                    try:
                        self._fts_db_path.unlink()
                    except OSError:
                        return
                self._logger.error(
                    "fts5.db is corrupt (%s) — quarantined to %s and "
                    "recreated empty. BM25 results degrade until a full "
                    "reindex (index_directory(incremental=false)).",
                    e, quarantine.name,
                )

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Sanitize a natural-language query for FTS5 MATCH syntax.

        Strips FTS5 operators and special chars, quotes each token,
        and joins with OR so any keyword match counts.
        """
        import re
        # Remove characters that are FTS5 operators or cause syntax errors.
        # C0 control chars (esp. NUL) are included: a NUL inside a quoted
        # token terminates the SQL string early and raises "unterminated
        # string", which silently emptied the BM25 leg for that query
        # (found by fuzzing, 2026-06-10).
        cleaned = re.sub(r'[?"*/\\(){}^~:+\-\x00-\x1f]', ' ', query)
        tokens = [t for t in cleaned.split() if t and len(t) > 1]
        if not tokens:
            return ""
        # Quote each token to prevent column-name interpretation
        return " OR ".join(f'"{t}"' for t in tokens)

    @_with_storage_lock
    def search_bm25(
        self,
        query: str,
        k: int = 50,
        name_weight: float = 5.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search using BM25 full-text search. Returns (chunk_id, rank, metadata).

        When `filters` is provided, applies the same `_matches_filters`
        check used by the vector search path. Without this, the BM25 half
        of hybrid search returned chunks the user explicitly filtered out
        via `file_pattern` / `chunk_type` (regression fixed 2026-05-07).

        Over-fetches by 3x when filters are present so a useful k is
        retained after filter rejection. Caps at the original `k` after
        filtering.
        """
        if not hasattr(self, "_fts_conn") or self._fts_conn is None:
            return []

        fts_query = self._sanitize_fts5_query(query)
        if not fts_query:
            return []

        # Over-fetch when filters will reduce the result set
        fetch_k = k * 3 if filters else k

        try:
            cursor = self._fts_conn.execute(
                f"SELECT chunk_id, bm25(chunk_fts, 0.0, 1.0, 0.5, {float(name_weight)}) as rank "
                "FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY rank LIMIT ?",
                (fts_query, fetch_k),
            )
            results = []
            seen = set()
            for chunk_id, rank in cursor.fetchall():
                # Dedupe: legacy indexes built before FTS rows were cleaned
                # on remove/re-add can hold the same chunk_id several times.
                # Rows arrive best-rank-first, so keep the first occurrence.
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                metadata_entry = self.metadata_db.get(chunk_id)
                if not metadata_entry:
                    continue
                metadata = metadata_entry["metadata"]
                if filters and not self._matches_filters(metadata, filters):
                    continue
                results.append((chunk_id, float(rank), metadata))
                if len(results) >= k:
                    break
            return results
        except Exception as e:
            self._logger.warning(
                "FTS5 search failed for %s: %s",
                format_query_for_log(fts_query, label="fts_query"),
                format_query_exception_for_log(e),
            )
            return []

    @property
    @_with_storage_lock
    def index(self):
        """Lazy loading of FAISS index."""
        if self._index is None:
            self._load_index()
        return self._index
    
    @property
    @_with_storage_lock
    def metadata_db(self):
        """Lazy loading of metadata database.

        Metadata is NOT recoverable from the other artifacts, so a corrupt
        metadata.db raises an ACTIONABLE error instead of a raw storage-layer
        traceback (2026-06-10 torn-write fuzz).
        """
        if self._metadata_db is None:
            try:
                self._metadata_db = JsonSqliteKV(str(self.metadata_path))
            except LegacyMetadataFormatError:
                # Pre-2026-06-11 sqlitedict format: actionable on its own.
                raise
            except Exception as e:
                raise RuntimeError(
                    f"metadata.db at {self.metadata_path} is corrupt or "
                    f"unreadable ({type(e).__name__}: {e}). Metadata cannot "
                    "be rebuilt from other artifacts — run "
                    "index_directory(incremental=false) to reindex this "
                    "project."
                ) from e
        return self._metadata_db
    
    @_with_storage_lock
    def _load_index(self):
        """Load existing FAISS index or create new one."""
        self._is_binary = False
        float_store_path = self.storage_dir / "float_store.npy"

        # CHUNK_ID DIAGNOSTIC (2026-05-05): tracks the load-side state so we
        # can spot when chunk_ids.pkl is empty or out-of-sync with FAISS.
        # The hypothesis under investigation: post-MCP-restart, the lazy
        # load sees an empty/short chunk_ids.pkl, then a subsequent
        # incremental save dumps the truncated list, overwriting prior
        # healthy state. Logging at every load + save lets us catch the
        # transition.
        self._logger.warning(
            "[CHUNK_ID_DIAG] _load_index pre-load: index_path=%s exists=%s "
            "chunk_id_path=%s exists=%s",
            self.index_path, self.index_path.exists(),
            self.chunk_id_path, self.chunk_id_path.exists(),
        )

        if self.index_path.exists():
            # Corruption-hardened (2026-06-10 torn-write fuzz): a truncated/
            # garbage code.index raised a raw faiss RuntimeError from every
            # search. The FAISS index is rebuilt by reindex, so corruption
            # degrades to vector-leg-disabled (BM25 keeps working) with a
            # loud actionable log instead of crashing the read path.
            try:
                # Detect binary mode: float_store.npy exists alongside the index
                if float_store_path.exists():
                    self._logger.info(f"Loading binary index from {self.index_path}")
                    self._index = faiss.read_index_binary(str(self.index_path))
                    self._float_store = np.load(str(float_store_path))
                    self._is_binary = True
                else:
                    self._logger.info(f"Loading index from {self.index_path}")
                    self._index = faiss.read_index(str(self.index_path))
                    if not self._is_binary:
                        self._maybe_move_index_to_gpu()
            except Exception as e:
                self._logger.error(
                    "FAISS index at %s is corrupt or unreadable (%s: %s). "
                    "Vector search disabled (BM25 still serves) until a "
                    "full reindex (index_directory(incremental=false)).",
                    self.index_path, type(e).__name__, str(e)[:200],
                )
                self._index = None
                self._chunk_ids = []
                self._is_binary = False
                return

            # Load chunk IDs. A corrupt pickle is recoverable: fall through
            # with an empty list and let _maybe_rebuild_chunk_ids reconstruct
            # the FAISS-position mapping losslessly from metadata.db
            # (pre-fix this raised UnpicklingError from every search even
            # though the rebuild machinery existed one call away).
            if self.chunk_id_path.exists():
                try:
                    with open(self.chunk_id_path, 'rb') as f:
                        loaded = pickle.load(f)
                    self._chunk_ids = loaded if isinstance(loaded, list) else []
                except Exception as e:
                    self._logger.error(
                        "chunk_ids.pkl is corrupt (%s: %s) — attempting "
                        "rebuild from metadata.db",
                        type(e).__name__, str(e)[:120],
                    )
                    self._chunk_ids = []

            # CHUNK_ID DIAGNOSTIC: log the state right after load.
            self._logger.warning(
                "[CHUNK_ID_DIAG] _load_index post-load: faiss.ntotal=%s "
                "chunk_ids_len=%s chunk_id_pkl_size=%s",
                self._index.ntotal if self._index else None,
                len(self._chunk_ids),
                self.chunk_id_path.stat().st_size if self.chunk_id_path.exists() else 0,
            )

            # Detect and repair chunk_ids.pkl corruption: if FAISS has vectors
            # but chunk_ids is missing/empty/shorter than expected, rebuild
            # from metadata.db. Each metadata value is a dict with 'index_id'
            # giving its FAISS position; we reconstruct the ordered list.
            self._maybe_rebuild_chunk_ids()

            # CHUNK_ID DIAGNOSTIC: log post-repair state, in case rebuild
            # fired and changed chunk_ids_len.
            self._logger.warning(
                "[CHUNK_ID_DIAG] _load_index post-repair: faiss.ntotal=%s "
                "chunk_ids_len=%s",
                self._index.ntotal if self._index else None,
                len(self._chunk_ids),
            )
        else:
            self._logger.warning(
                "[CHUNK_ID_DIAG] _load_index: no existing index, starting fresh"
            )
            self._index = None
            self._chunk_ids = []

    def _maybe_rebuild_chunk_ids(self):
        """Rebuild chunk_ids.pkl from metadata.db if it's missing or out of sync.

        Guards against the failure mode where chunk_ids.pkl gets truncated to
        an empty list (5 bytes: empty pickle) by a failed load path, causing
        every subsequent search to raise `list index out of range`. The FAISS
        index and metadata database are still intact; only the parallel
        chunk-id list is lost. Recovery is lossless as long as metadata.db
        still holds an `index_id` for every row.
        """
        if self._index is None:
            return
        faiss_n = self._index.ntotal
        chunk_n = len(self._chunk_ids)
        if faiss_n == 0:
            return
        if chunk_n == faiss_n:
            return
        if not self.metadata_path.exists():
            self._logger.warning(
                "chunk_ids mismatch (faiss=%d, chunk_ids=%d) but metadata.db missing — "
                "cannot auto-rebuild; reindex required",
                faiss_n, chunk_n,
            )
            return
        self._logger.warning(
            "chunk_ids out of sync with FAISS (faiss=%d, chunk_ids=%d) — rebuilding from metadata.db",
            faiss_n, chunk_n,
        )
        rebuilt = [None] * faiss_n
        filled = 0
        for chunk_id, entry in self.metadata_db.items():
            idx = entry.get("index_id") if isinstance(entry, dict) else None
            if not isinstance(idx, int) or idx < 0 or idx >= faiss_n:
                continue
            if rebuilt[idx] is None:
                rebuilt[idx] = chunk_id
                filled += 1
        missing = faiss_n - filled
        if missing > 0:
            self._logger.error(
                "chunk_ids rebuild incomplete: %d of %d slots still missing — reindex recommended",
                missing, faiss_n,
            )
            # Leave self._chunk_ids as-is rather than shipping a half-rebuilt list
            # that would mismatch FAISS positions.
            return
        # Back up the corrupted pkl (if present) before overwriting.
        if self.chunk_id_path.exists() and chunk_n != faiss_n:
            import time
            bak = self.chunk_id_path.with_suffix(
                f".pkl.bak.{time.strftime('%Y%m%dT%H%M%S')}"
            )
            try:
                bak.write_bytes(self.chunk_id_path.read_bytes())
            except OSError as exc:
                self._logger.warning("could not back up corrupted chunk_ids.pkl: %s", exc)
        self._chunk_ids = rebuilt
        with open(self.chunk_id_path, "wb") as f:
            pickle.dump(self._chunk_ids, f)
        self._logger.info("chunk_ids rebuilt and persisted (%d entries)", faiss_n)
    
    @_with_storage_lock
    def create_index(self, embedding_dimension: int, index_type: str = "flat"):
        """Create a new FAISS index.

        Quantization controlled by QUANTIZATION env var:
        - "int8" (default): ScalarQuantizer with QT_8bit_direct — 4x smaller, <0.1% quality loss
        - "float32": IndexFlatIP — original full-precision
        - "binary": IndexBinaryFlat + float store — 32x smaller, needs rescore (opt-in for 100K+ chunks)
        """
        quantization = os.environ.get("QUANTIZATION", "int8").lower()
        self._is_binary = False

        if quantization == "binary":
            self._index = faiss.IndexBinaryFlat(embedding_dimension)
            self._float_store = np.empty((0, embedding_dimension), dtype=np.float32)
            self._is_binary = True
            self._logger.info(f"Created binary index with dimension {embedding_dimension} (32x compression, requires rescore)")
        elif quantization == "int8" and index_type == "flat":
            # QT_8bit (trained) learns the value range from data, then linearly maps to [0,255].
            # QT_8bit_direct was wrong — it interprets float bytes as raw ints, producing
            # all-zero similarities on normalized [-1,1] vectors. (Confirmed 2026-04-05:
            # isolated FAISS test showed QT_8bit_direct returns 0.0 for all queries.)
            self._index = faiss.IndexScalarQuantizer(
                embedding_dimension, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
            )
            self._logger.info(f"Created int8 quantized index with dimension {embedding_dimension} (4x compression, requires training)")
        elif index_type == "flat":
            self._index = faiss.IndexFlatIP(embedding_dimension)
            self._logger.info(f"Created float32 flat index with dimension {embedding_dimension}")
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(embedding_dimension)
            n_centroids = min(100, max(10, embedding_dimension // 8))
            self._index = faiss.IndexIVFFlat(quantizer, embedding_dimension, n_centroids)
            self._logger.info(f"Created IVF index with dimension {embedding_dimension}")
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        if not self._is_binary:
            self._maybe_move_index_to_gpu()
    
    @_with_storage_lock
    def add_embeddings(self, embedding_results: List[EmbeddingResult]) -> None:
        """Add embeddings to the index and metadata to the database."""
        if not embedding_results:
            return

        # Load existing on-disk index BEFORE deciding to create a new one.
        # Without this, a fresh CodeIndexManager (e.g., after switch_project)
        # whose `_index` is still None will fall through to create_index()
        # and start an empty FAISS while the on-disk index already holds
        # 30+ vectors. The next save_index then dumps that empty in-memory
        # state over the healthy on-disk pkl/index — the chunk-truncation
        # regression observed 2026-05-04/05.
        if self._index is None and self.index_path.exists():
            self._load_index()
        self._mark_working_set_dirty()

        # Initialize index if needed
        if self._index is None:
            embedding_dim = embedding_results[0].embedding.shape[0]
            # Always use flat index - IVF breaks reconstruct() needed by get_similar_chunks
            # Flat handles 20K+ vectors fine for our use case
            index_type = "flat"
            self.create_index(embedding_dim, index_type)
        
        # Prepare embeddings and metadata
        embeddings = np.array([result.embedding for result in embedding_results])
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Train quantized/IVF index if needed
        if hasattr(self._index, 'is_trained') and not self._index.is_trained:
            # The quantizer trains ONCE, on whatever the first batch is.
            # Full reindexes pass all chunks in one add_embeddings call, so
            # training data is representative. An index born from a small
            # incremental batch learns its value range from few vectors —
            # later additions outside that range clip. Warn so operators
            # know a full reindex would improve int8 fidelity.
            if len(embeddings) < 256:
                self._logger.warning(
                    "Training quantizer on only %d vectors; value ranges may "
                    "be unrepresentative. A full reindex (force=true) trains "
                    "on the complete corpus.",
                    len(embeddings),
                )
            self._logger.info("Training index...")
            self._index.train(embeddings)

        # Add to FAISS index (binary mode packs bits separately)
        if getattr(self, '_is_binary', False):
            # Binary mode: pack sign bits and store float originals for rescoring
            codes = np.packbits((embeddings > 0).astype(np.uint8), axis=1)
            self._index.add(codes)
            self._float_store = np.concatenate([self._float_store, embeddings], axis=0)
        else:
            self._index.add(embeddings)
        start_id = len(self._chunk_ids)
        
        # Store metadata and update chunk IDs
        for i, result in enumerate(embedding_results):
            chunk_id = result.chunk_id
            self._chunk_ids.append(chunk_id)
            
            # Store in metadata database
            self.metadata_db[chunk_id] = {
                'index_id': start_id + i,
                'metadata': result.metadata
            }
        
        self._logger.info(f"Added {len(embedding_results)} embeddings to index")
        
        # Commit metadata in a single transaction for performance
        try:
            self.metadata_db.commit()
        except Exception:
            # If commit is unavailable for some reason, continue without failing
            pass

        # Add to FTS5 index (re-init if connection was lost)
        if not hasattr(self, "_fts_conn") or self._fts_conn is None:
            self._init_fts5()
        # Idempotency: drop any existing FTS rows for the incoming chunk_ids
        # first. chunk_fts has no uniqueness constraint, so re-adding a
        # chunk_id (modified file whose chunk boundaries didn't move) would
        # otherwise duplicate it in BM25 results.
        incoming_ids = [r.chunk_id for r in embedding_results]
        try:
            for i in range(0, len(incoming_ids), 500):
                batch = incoming_ids[i:i + 500]
                placeholders = ",".join("?" * len(batch))
                self._fts_conn.execute(
                    f"DELETE FROM chunk_fts WHERE chunk_id IN ({placeholders})",
                    batch,
                )
        except Exception as e:
            self._logger.warning(f"FTS5 pre-insert cleanup failed: {e}")
        for result in embedding_results:
            content = result.metadata.get("full_content", result.metadata.get("content_preview", ""))
            file_path = result.metadata.get("relative_path", result.metadata.get("file_path", ""))
            name = result.metadata.get("name", "") or ""

            # Contextual BM25: prepend metadata header so BM25 can match on
            # file path, type, and name even when the code body doesn't contain
            # the query terms. Evidence: +0.128 MRR on TypeScript when combined
            # with query rewriting (A/B eval 2026-04-07, 102 queries).
            chunk_type = result.metadata.get("chunk_type", "")
            parent = result.metadata.get("parent_name", "")
            header_parts = []
            if file_path:
                header_parts.append(f"# From {file_path}")
            if parent and name:
                header_parts.append(f"- {chunk_type} {parent}.{name}")
            elif name:
                header_parts.append(f"- {chunk_type} {name}")
            elif chunk_type:
                header_parts.append(f"- {chunk_type}")
            if header_parts:
                content = " ".join(header_parts) + "\n" + content

            self._fts_conn.execute(
                "INSERT INTO chunk_fts (chunk_id, content, file_path, name) VALUES (?, ?, ?, ?)",
                (result.chunk_id, content, file_path, name),
            )
        self._fts_conn.commit()

        # Update statistics
        self._update_stats()

    def _gpu_is_available(self) -> bool:
        """Check if GPU FAISS support is available and GPUs are present."""
        try:
            if not hasattr(faiss, 'StandardGpuResources'):
                return False
            get_num_gpus = getattr(faiss, 'get_num_gpus', None)
            if get_num_gpus is None:
                return False
            return get_num_gpus() > 0
        except Exception:
            return False

    def _maybe_move_index_to_gpu(self) -> None:
        """Move the current index to GPU if supported. No-op if already on GPU or unsupported."""
        if self._index is None or self._on_gpu:
            return
        if not self._gpu_is_available():
            return
        try:
            # Move index to all GPUs for faster add/search
            self._index = faiss.index_cpu_to_all_gpus(self._index)
            self._on_gpu = True
            self._logger.info("FAISS index moved to GPU(s)")
        except Exception as e:
            self._logger.warning(f"Failed to move FAISS index to GPU, continuing on CPU: {e}")
    
    @_with_storage_lock
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar code chunks."""
        import logging
        logger = logging.getLogger(__name__)

        # R2: reject k <= 0 with a clean ValueError instead of letting it
        # hit the FAISS bindings (which raise an AssertionError with no
        # context). The MCP surface accepts a k arg from external callers
        # who can trivially pass 0 or a negative; an explicit error here
        # is the right boundary.
        if not isinstance(k, int) or k <= 0:
            raise ValueError(
                f"k must be a positive integer, got {k!r} (type={type(k).__name__})"
            )

        logger.info(f"Index manager search called with k={k}, filters={filters}")

        # Use property to trigger lazy loading
        index = self.index
        if index is None or index.ntotal == 0:
            logger.warning(f"Index is empty or None. Index: {index}, ntotal: {index.ntotal if index else 'N/A'}")
            return []

        logger.info(f"Index has {index.ntotal} total vectors")

        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Binary mode: hamming search → float rescore
        if getattr(self, '_is_binary', False) and hasattr(self, '_float_store'):
            search_k = min(k * 20, index.ntotal)
            q_codes = np.packbits((query_embedding[0] > 0).astype(np.uint8)).reshape(1, -1)
            _distances, bin_indices = index.search(q_codes, search_k)
            # Rescore with float dot product
            candidate_ids = bin_indices[0][bin_indices[0] >= 0]
            if len(candidate_ids) == 0:
                return []
            candidate_vecs = self._float_store[candidate_ids]
            scores = candidate_vecs @ query_embedding[0]
            top_order = np.argsort(-scores)
            indices = np.array([candidate_ids[top_order]])
            similarities = np.array([scores[top_order]])
        else:
            # Standard search (float32 or int8)
            search_k = min(k * 3, index.ntotal)
            similarities, indices = index.search(query_embedding, search_k)
        
        results = []
        seen = set()
        for i, (similarity, index_id) in enumerate(zip(similarities[0], indices[0])):
            if index_id == -1:  # No more results
                break

            # Defensive bounds check: a truncated chunk_ids list (pre-repair)
            # must degrade to fewer results, not IndexError.
            if index_id >= len(self._chunk_ids):
                continue

            chunk_id = self._chunk_ids[index_id]
            if chunk_id is None:
                continue
            # Dedupe: after a modify→re-add cycle the same chunk_id exists at
            # two FAISS positions (the stale vector is never removed). FAISS
            # returns results sorted by similarity, so the first occurrence
            # is the best-scoring one; later duplicates would otherwise
            # occupy extra result slots AND get double-counted by RRF, which
            # sums contributions per appearance.
            if chunk_id in seen:
                continue
            seen.add(chunk_id)

            metadata_entry = self.metadata_db.get(chunk_id)

            if metadata_entry is None:
                continue

            metadata = metadata_entry['metadata']

            # Apply filters
            if filters and not self._matches_filters(metadata, filters):
                continue

            results.append((chunk_id, float(similarity), metadata))

            if len(results) >= k:
                break

        return results
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the provided filters.

        R1: filters with `None` values are treated as "filter absent" rather
        than "match nothing". Pre-fix, `chunk_type=None` would compare
        `metadata['chunk_type'] != None` which is True for every indexed
        chunk → filter rejects all results silently. Pre-fix,
        `file_pattern=None` would crash with TypeError on the for-loop.
        Both are operator-facing failure shapes (the MCP `search_code`
        tool accepts these args from external callers) so we normalize
        to a no-op filter here rather than at every call site.
        """
        for key, value in filters.items():
            # R1: a None value means "this filter is not provided" — skip it.
            # If a caller wants to filter for chunks where chunk_type IS
            # literally None (it's not, but for symmetry), they must pass
            # the explicit string the indexer stores.
            if value is None:
                continue

            if key == 'file_pattern':
                # Glob matching for file paths. fnmatch does shell-style:
                # `*.rs` matches `foo.rs`, `internal/x/foo.rs`, etc. (against
                # full path segments). Previously this was substring match,
                # which meant `*.rs` was never a substring of any real path
                # and silently filtered out all results — except it didn't,
                # because the BM25 path bypassed filtering entirely (see
                # search_bm25). Both bugs fixed together in this change.
                relative_path = metadata.get('relative_path', '') or ''
                # Match against both full path AND basename so users can
                # write `*.rs` (basename pattern) or `internal/**/*.rs`
                # (path pattern) interchangeably.
                basename = relative_path.split('/')[-1].split('\\')[-1]
                # Accept both a single pattern string and a list of patterns.
                # Pre-R1 the for-pattern-in-value path crashed on a single
                # string (it'd iterate chars); normalize first.
                patterns = value if isinstance(value, (list, tuple)) else [value]
                if not any(
                    fnmatch.fnmatch(relative_path, pattern)
                    or fnmatch.fnmatch(basename, pattern)
                    for pattern in patterns
                ):
                    return False
            elif key == 'chunk_type':
                # Exact match for chunk type
                if metadata.get('chunk_type') != value:
                    return False
            elif key == 'tags':
                # Tag intersection
                chunk_tags = set(metadata.get('tags', []))
                required_tags = set(value if isinstance(value, list) else [value])
                if not required_tags.intersection(chunk_tags):
                    return False
            elif key == 'folder_structure':
                # Check if any of the required folders are in the path
                chunk_folders = set(metadata.get('folder_structure', []))
                required_folders = set(value if isinstance(value, list) else [value])
                if not required_folders.intersection(chunk_folders):
                    return False
            elif key in metadata:
                # Direct metadata comparison
                if metadata[key] != value:
                    return False
        
        return True
    
    @_with_storage_lock
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve chunk metadata by ID."""
        metadata_entry = self.metadata_db.get(chunk_id)
        return metadata_entry['metadata'] if metadata_entry else None

    @_with_storage_lock
    def get_chunk_entries(self) -> list[tuple[str, dict[str, Any]]]:
        """Return a stable chunk/metadata snapshot for compound readers."""
        entries = []
        for chunk_id in self._chunk_ids:
            metadata_entry = self.metadata_db.get(chunk_id)
            if metadata_entry:
                entries.append((chunk_id, metadata_entry))
        return entries

    @_with_storage_lock
    def count_chunks_in_file(self, relative_path: str) -> int:
        """Count the live chunks indexed for a specific file.

        Uses the FTS5 table's file_path column (plain equality scan, no
        MATCH). FTS rows are now deleted on remove_file_chunks, so this
        reflects live chunks only.
        """
        if not relative_path:
            return 0
        if not hasattr(self, "_fts_conn") or self._fts_conn is None:
            return 0
        try:
            cursor = self._fts_conn.execute(
                "SELECT COUNT(*) FROM chunk_fts WHERE file_path = ?",
                (relative_path,),
            )
            return int(cursor.fetchone()[0])
        except Exception:
            return 0
    
    @_with_storage_lock
    def get_similar_chunks(self, chunk_id: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find chunks similar to a given chunk."""
        metadata_entry = self.metadata_db.get(chunk_id)
        if not metadata_entry:
            return []
        
        index_id = metadata_entry['index_id']
        if self._index is None or index_id >= self._index.ntotal:
            return []

        # Get the embedding for this chunk. Binary indexes reconstruct to
        # packed uint8 codes, not floats — pull the original vector from the
        # float store instead so the downstream float search path works.
        if getattr(self, '_is_binary', False) and hasattr(self, '_float_store'):
            if index_id >= len(self._float_store):
                return []
            embedding = self._float_store[index_id].copy()
        else:
            embedding = self._index.reconstruct(index_id)

        # Search for similar chunks (excluding the original)
        results = self.search(embedding, k + 1)
        
        # Filter out the original chunk
        return [(cid, sim, meta) for cid, sim, meta in results if cid != chunk_id][:k]
    
    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize a path for comparison: forward slashes, no trailing slash."""
        return str(path).replace("\\", "/").rstrip("/")

    @staticmethod
    def _is_absolute_norm(path: str) -> bool:
        """Absolute-path check on a _normalize_path'd string (POSIX or drive)."""
        return path.startswith("/") or (
            len(path) >= 3 and path[1] == ":" and path[2] == "/"
        )

    @classmethod
    def _paths_refer_to_same_file(cls, target: str, chunk_rel: str, chunk_abs: str) -> bool:
        """True if `target` (relative or absolute) identifies the chunk's file.

        Matching rules (the previous implementation used bare substring
        containment — `file_path in chunk_file or chunk_file in file_path` —
        which made removing `test.py` also delete `conftest.py`'s chunks;
        the un-modified file then silently vanished from the index until
        its next edit or a full reindex):

        - exact equality against the stored relative or absolute path;
        - an ABSOLUTE target matches a stored relative path when it ends
          with "/<relative path>" (path-segment boundary);
        - a RELATIVE target matches only the stored relative path exactly.
          It is NOT suffix-matched against the absolute path unless no
          relative path is stored — otherwise removing a root-level
          `util.py` would also match `src/util.py` (the absolute path ends
          with "/util.py").
        """
        target = cls._normalize_path(target)
        chunk_rel = cls._normalize_path(chunk_rel) if chunk_rel else ""
        chunk_abs = cls._normalize_path(chunk_abs) if chunk_abs else ""
        if not target:
            return False
        if target in (chunk_rel, chunk_abs):
            return True
        if cls._is_absolute_norm(target):
            # Absolute target vs relative metadata.
            return bool(chunk_rel) and target.endswith("/" + chunk_rel)
        # Relative target: only fall back to an absolute-suffix match when
        # the chunk stored no relative path at all.
        if not chunk_rel and chunk_abs:
            return chunk_abs.endswith("/" + target)
        return False

    @_with_storage_lock
    def remove_file_chunks(self, file_path: str, project_name: Optional[str] = None) -> int:
        """Remove all chunks from a specific file.

        Removes the metadata rows AND the FTS5 rows. Pre-fix only metadata
        was deleted, so every modified file left its old FTS5 rows behind:
        re-adding the same chunk_id duplicated it in BM25 results (and RRF
        sums per-appearance, inflating fused scores), while shifted
        chunk_ids left dead rows that consumed the BM25 LIMIT quota.

        Args:
            file_path: Path to the file (relative or absolute)
            project_name: Optional project name filter

        Returns:
            Number of chunks removed
        """
        # Load existing on-disk state BEFORE iterating _chunk_ids. Without
        # this, a fresh CodeIndexManager (e.g., after switch_project) sees
        # an empty in-memory _chunk_ids and silently removes nothing —
        # the file's old chunks become orphans the next save will not
        # preserve. See add_embeddings for the symmetric fix.
        if self._index is None and self.index_path.exists():
            self._load_index()

        chunks_to_remove = []
        seen = set()

        # Find chunks to remove. _chunk_ids can contain the same chunk_id at
        # multiple FAISS positions after a modify→re-add cycle; dedupe so the
        # metadata delete below doesn't KeyError on the second occurrence.
        for chunk_id in self._chunk_ids:
            if chunk_id is None or chunk_id in seen:
                continue
            seen.add(chunk_id)
            metadata_entry = self.metadata_db.get(chunk_id)
            if not metadata_entry:
                continue

            metadata = metadata_entry['metadata']

            chunk_rel = metadata.get('relative_path') or ''
            chunk_abs = metadata.get('file_path') or ''
            if not (chunk_rel or chunk_abs):
                continue

            if self._paths_refer_to_same_file(file_path, chunk_rel, chunk_abs):
                # Check project name if provided
                if project_name and metadata.get('project_name') != project_name:
                    continue
                chunks_to_remove.append(chunk_id)

        # Remove chunks from metadata
        if chunks_to_remove:
            self._mark_working_set_dirty()
        for chunk_id in chunks_to_remove:
            try:
                del self.metadata_db[chunk_id]
            except KeyError:
                pass

        # Remove the corresponding FTS5 rows (batched under SQLite's
        # parameter limit).
        if chunks_to_remove:
            if not hasattr(self, "_fts_conn") or self._fts_conn is None:
                self._init_fts5()
            try:
                for i in range(0, len(chunks_to_remove), 500):
                    batch = chunks_to_remove[i:i + 500]
                    placeholders = ",".join("?" * len(batch))
                    self._fts_conn.execute(
                        f"DELETE FROM chunk_fts WHERE chunk_id IN ({placeholders})",
                        batch,
                    )
                self._fts_conn.commit()
            except Exception as e:
                self._logger.warning(
                    f"FTS5 cleanup failed for {file_path}: {e}"
                )

        # Note: We don't remove from FAISS index directly as it's complex
        # Instead, we'll rebuild the index periodically or on demand

        self._logger.info(f"Removed {len(chunks_to_remove)} chunks from {file_path}")

        # Commit removals in batch
        try:
            self.metadata_db.commit()
        except Exception:
            pass
        return len(chunks_to_remove)
    
    def _snapshot_sqlite(self, source: Path, destination: Path) -> None:
        """Create a single-file SQLite snapshot that includes committed WAL."""
        import sqlite3

        source_connection = sqlite3.connect(
            f"{source.resolve().as_uri()}?mode=ro", uri=True
        )
        destination_connection = sqlite3.connect(str(destination))
        try:
            source_connection.backup(destination_connection)
            destination_connection.commit()
        finally:
            destination_connection.close()
            source_connection.close()
        self._sqlite_integrity_check(destination)
        self._fsync_file(destination)

    def _write_faiss_candidate(self, destination: Path) -> None:
        """Persist FAISS to a candidate path or propagate the final failure."""
        if getattr(self, "_is_binary", False):
            faiss.write_index_binary(self._index, str(destination))
            return

        index_to_write = self._index
        if self._on_gpu and hasattr(faiss, "index_gpu_to_cpu"):
            index_to_write = faiss.index_gpu_to_cpu(self._index)
        try:
            faiss.write_index(index_to_write, str(destination))
        # FAISS bindings expose backend-specific exception types.
        except Exception as first_error:  # noqa: BLE001
            self._logger.warning(
                "Primary FAISS candidate write failed; retrying: %s",
                first_error,
            )
            try:
                faiss.write_index(index_to_write, str(destination))
            except Exception as final_error:
                self._logger.error(
                    "Both FAISS candidate writes failed: %s", final_error
                )
                raise final_error from first_error

    def _write_candidate_generation(self, candidate_dir: Path) -> None:
        """Write every artifact into an unpublished candidate directory."""
        candidate_dir.mkdir(parents=True, exist_ok=False)

        index_path = candidate_dir / "code.index"
        self._write_faiss_candidate(index_path)
        self._fsync_file(index_path)

        chunk_ids_path = candidate_dir / "chunk_ids.pkl"
        with chunk_ids_path.open("wb") as handle:
            pickle.dump(self._chunk_ids, handle)
            handle.flush()
            os.fsync(handle.fileno())

        if getattr(self, "_is_binary", False):
            float_path = candidate_dir / "float_store.npy"
            with float_path.open("wb") as handle:
                np.save(handle, self._float_store, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())

        if self._metadata_db is not None:
            self._metadata_db.commit()
        if self.metadata_path.exists():
            self._snapshot_sqlite(
                self.metadata_path, candidate_dir / "metadata.db"
            )

        if getattr(self, "_fts_conn", None) is not None:
            self._fts_conn.commit()
        if self._fts_db_path.exists():
            self._snapshot_sqlite(
                self._fts_db_path, candidate_dir / "fts5.db"
            )

        if self.stats_path.exists():
            staged_stats = candidate_dir / "stats.json"
            shutil.copy2(self.stats_path, staged_stats)
            self._fsync_file(staged_stats)
        self._fsync_directory(candidate_dir)

    def _build_generation_manifest(
        self, generation_dir: Path
    ) -> dict[str, Any]:
        """Build a manifest using persisted, reopened artifact counts."""
        from search.epoch_manifest import ArtifactSpec, build_manifest

        with (generation_dir / "chunk_ids.pkl").open("rb") as handle:
            persisted_chunk_ids = pickle.load(handle)
        if not isinstance(persisted_chunk_ids, list):
            raise TypeError("Candidate chunk_ids.pkl is not a list")

        index_path = generation_dir / "code.index"
        if (generation_dir / "float_store.npy").exists():
            persisted_index = faiss.read_index_binary(str(index_path))
        else:
            persisted_index = faiss.read_index(str(index_path))
        persisted_dimension = int(getattr(persisted_index, "d", 0) or 0)
        configured_dimension = int(
            getattr(self, "_embedder_dimension", 0) or 0
        )
        if (
            configured_dimension
            and persisted_dimension != configured_dimension
        ):
            raise IndexPublicationRefused(
                "Cannot publish index because configured embedding dimension "
                f"{configured_dimension} does not match persisted FAISS "
                f"dimension {persisted_dimension}"
            )

        artifacts = [
            ArtifactSpec(
                name="chunk_ids.pkl",
                path=generation_dir / "chunk_ids.pkl",
                count=len(persisted_chunk_ids),
            ),
            ArtifactSpec(
                name="code.index",
                path=index_path,
                count=int(persisted_index.ntotal),
            ),
        ]
        for name in (
            "metadata.db",
            "fts5.db",
            "stats.json",
            "float_store.npy",
        ):
            path = generation_dir / name
            if path.exists():
                artifacts.append(
                    ArtifactSpec(name=name, path=path, count=None)
                )

        return build_manifest(
            project_dir=self.storage_dir,
            artifacts=artifacts,
            provider=getattr(self, "_embedder_provider", "") or "",
            model=getattr(self, "_embedder_model", "") or "",
            vector_dim=persisted_dimension,
            quantization=(
                "binary"
                if getattr(self, "_is_binary", False)
                else (
                    "int8"
                    if self._index
                    and "ScalarQuantizer" in type(self._index).__name__
                    else "float32"
                )
            ),
            pipeline_version=getattr(self, "_pipeline_version", "") or "",
            input_type_enabled=bool(
                getattr(
                    self,
                    "_embedder_input_type_enabled",
                    False,
                )
            ),
        )

    def _restore_last_published_generation(self) -> None:
        """Discard a failed working set and restore the publication point."""
        from search.epoch_manifest import read_with_fallback

        self._close_storage_handles()
        result = read_with_fallback(self.storage_dir)
        if (
            result.manifest is not None
            and self._manifest_uses_generation(result.manifest)
        ):
            self._validate_published_generation(result.manifest)
            self._materialize_generation(result.manifest)
        elif result.manifest is None:
            manifest_dir = self.storage_dir / "manifest"
            if (
                (manifest_dir / "current.json").exists()
                or (manifest_dir / "prior.json").exists()
            ):
                raise RuntimeError(
                    "Cannot restore failed publication because no manifest "
                    "generation verifies"
                )
            self._discard_unpublished_working_set()
        self._index = None
        self._chunk_ids = []
        self._is_binary = False
        self._on_gpu = False
        if hasattr(self, "_float_store"):
            del self._float_store
        self._init_fts5()

    def _discard_unpublished_working_set(self) -> None:
        """Remove every root artifact when no generation ever committed."""
        self._close_storage_handles()
        removed = False
        artifacts = (
            self.index_path,
            self.chunk_id_path,
            self.metadata_path,
            self._fts_db_path,
            self.stats_path,
            self.storage_dir / "float_store.npy",
        )
        for artifact in artifacts:
            for path in (
                artifact,
                Path(f"{artifact}-wal"),
                Path(f"{artifact}-shm"),
            ):
                try:
                    path.unlink()
                    removed = True
                except FileNotFoundError:
                    pass
        if removed:
            self._fsync_directory(self.storage_dir)

    def _remove_generation_path(
        self, path: Path, *, ignore_errors: bool = False
    ) -> None:
        """Remove an unpublished generation and persist its directory entry."""
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            self._fsync_directory(self._generation_root)
        except FileNotFoundError:
            pass
        except OSError:
            if not ignore_errors:
                raise

    def _generation_is_manifest_referenced(
        self, generation_dir: Path
    ) -> bool:
        """Return whether current or prior manifest names this generation."""
        manifest_dir = self.storage_dir / "manifest"
        for name in ("current.json", "prior.json"):
            try:
                with (manifest_dir / name).open(
                    "r", encoding="utf-8"
                ) as handle:
                    manifest = json.load(handle)
            except (
                FileNotFoundError,
                json.JSONDecodeError,
                OSError,
                TypeError,
                UnicodeDecodeError,
            ):
                continue
            for entry in manifest.get("artifacts", {}).values():
                try:
                    artifact_path = self.storage_dir / entry["path"]
                    artifact_path.relative_to(generation_dir)
                except (KeyError, TypeError, ValueError):
                    continue
                return True
        return False

    def _prune_unreferenced_generations(self) -> None:
        """Best-effort cleanup, called only after a successful commit."""
        from search.epoch_manifest import (
            read_current,
            read_prior,
            verify_manifest,
        )

        retained: set[Path] = set()
        verified_manifests = 0
        for label, reader in (
            ("current", read_current),
            ("prior", read_prior),
        ):
            try:
                manifest = reader(self.storage_dir)
            except Exception as exc:  # noqa: BLE001 - pruning is best effort
                self._logger.warning(
                    "Could not read %s manifest while pruning generations: %s",
                    label,
                    exc,
                )
                continue
            if not manifest:
                continue
            try:
                verification_error = verify_manifest(
                    self.storage_dir,
                    manifest,
                )
            except Exception as exc:  # noqa: BLE001 - pruning must fail closed
                self._logger.warning(
                    "Could not verify %s manifest while pruning generations: "
                    "%s",
                    label,
                    exc,
                )
                continue
            if verification_error is not None:
                self._logger.warning(
                    "Ignoring unverified %s manifest while pruning "
                    "generations: %s",
                    label,
                    verification_error,
                )
                continue
            verified_manifests += 1
            for entry in manifest.get("artifacts", {}).values():
                path = self.storage_dir / entry["path"]
                try:
                    relative = path.relative_to(self._generation_root)
                except ValueError:
                    continue
                if relative.parts:
                    retained.add(self._generation_root / relative.parts[0])

        if verified_manifests == 0:
            self._logger.warning(
                "Skipping generation pruning because no manifest verifies"
            )
            return
        if not self._generation_root.exists():
            return
        for path in self._generation_root.iterdir():
            if path in retained:
                continue
            try:
                self._remove_generation_path(path)
            except OSError as exc:
                self._logger.warning(
                    "Could not prune unreferenced index generation %s: %s",
                    path,
                    exc,
                )

    @_with_storage_lock
    def publish_root_generation(
        self, expected_chunk_ids: list[str]
    ) -> str:
        """Publish a complete root snapshot after offline maintenance.

        Administrative cleanup tools mutate the root compatibility files
        while the service is quiesced. This method validates that those files
        still form a complete searchable index, preserves identity from the
        last verified manifest, and routes the snapshot through the same
        immutable-generation transaction as normal saves.
        """
        from search.epoch_manifest import (
            read_current,
            read_with_fallback,
        )

        if not self.index_path.exists() or not self.chunk_id_path.exists():
            raise IndexPublicationRefused(
                "Cannot publish cleanup state without code.index and "
                "chunk_ids.pkl"
            )

        try:
            publication = read_with_fallback(self.storage_dir)
        except Exception as exc:
            raise IndexPublicationRefused(
                "Cannot preserve index identity because the existing "
                f"manifest is unreadable: {exc}"
            ) from exc

        manifest_dir = self.storage_dir / "manifest"
        has_manifest_state = any(
            (manifest_dir / name).exists()
            for name in ("current.json", "prior.json")
        )
        if publication.manifest is None and has_manifest_state:
            raise IndexPublicationRefused(
                "Cannot publish cleanup state because no existing manifest "
                "generation verifies"
            )

        self._load_index()
        if self._index is None:
            raise IndexPublicationRefused(
                "Cannot publish cleanup state because code.index is unreadable"
            )
        if self._chunk_ids != expected_chunk_ids:
            raise IndexPublicationRefused(
                "Cannot publish cleanup state because persisted chunk IDs "
                "do not match the audited chunk IDs"
            )
        if int(self._index.ntotal) != len(expected_chunk_ids):
            raise IndexPublicationRefused(
                "Cannot publish cleanup state because persisted FAISS ntotal "
                "does not match chunk_ids.pkl"
            )

        expected_ids = set(expected_chunk_ids)
        metadata_orphans = [
            chunk_id
            for chunk_id in self.metadata_db.keys()
            if chunk_id not in expected_ids
        ]
        fts_orphans = [
            chunk_id
            for (chunk_id,) in self._fts_conn.execute(
                "SELECT chunk_id FROM chunk_fts"
            )
            if chunk_id not in expected_ids
        ]
        if metadata_orphans or fts_orphans:
            raise IndexPublicationRefused(
                "Cannot publish incomplete cleanup state because sidecar "
                "databases still contain chunk IDs outside chunk_ids.pkl "
                f"(metadata={len(metadata_orphans)}, "
                f"fts5={len(fts_orphans)})"
            )

        identity = publication.manifest
        if identity is not None:
            expected_dimension = int(identity.get("vector_dim", 0) or 0)
            actual_dimension = int(getattr(self._index, "d", 0) or 0)
            if (
                expected_dimension
                and actual_dimension != expected_dimension
            ):
                raise IndexPublicationRefused(
                    "Cannot publish cleanup state because FAISS dimension "
                    f"{actual_dimension} does not match the verified manifest "
                    f"dimension {expected_dimension}"
                )
            self._embedder_provider = identity.get("provider", "") or ""
            self._embedder_model = identity.get("model", "") or ""
            manifest_input_type = identity.get("input_type_enabled")
            if not isinstance(manifest_input_type, bool):
                raise IndexPublicationRefused(
                    "Cannot publish cleanup state because the verified "
                    "manifest input_type_enabled identity is missing or "
                    "invalid"
                )
            self._embedder_input_type_enabled = manifest_input_type
            self._pipeline_version = (
                identity.get("pipeline_version", "") or ""
            )

        self.save_index(force=True, refresh_stats=False)
        committed = read_current(self.storage_dir)
        self._validate_published_generation(committed)
        return str(committed["epoch_id"])

    @_with_storage_lock
    def save_index(
        self, force: bool = False, *, refresh_stats: bool = True
    ):
        """Atomically publish a complete, persisted index generation.

        Artifacts are written and reopened in a same-filesystem candidate
        directory. Only a validated candidate is mirrored to the historical
        root paths and made authoritative by the final manifest rename.

        Args:
            force: Bypass the chunk-truncation guard. Set True only by
                callers that legitimately shrink the index (clear_index,
                full reindex with deletions, explicit user reset). Default
                False so accidental truncation aborts loudly.
        """
        # CHUNK_ID DIAGNOSTIC (2026-05-05): log pre-save state to catch the
        # hypothesized failure mode where save_index dumps a truncated
        # _chunk_ids over a previously healthy on-disk pkl. If the on-disk
        # pkl was 10K entries and we're about to save 12, that's the bug.
        try:
            existing_pkl_size = (
                self.chunk_id_path.stat().st_size
                if self.chunk_id_path.exists()
                else 0
            )
        except Exception:
            existing_pkl_size = -1
        self._logger.warning(
            "[CHUNK_ID_DIAG] save_index pre-save: in_memory_chunk_ids_len=%s "
            "faiss.ntotal=%s on_disk_pkl_size=%s caller_path=%s",
            len(self._chunk_ids),
            self._index.ntotal if self._index else None,
            existing_pkl_size,
            self.chunk_id_path,
        )

        # Defense-in-depth: refuse to clobber a healthy on-disk pkl with a
        # dramatically smaller in-memory list unless the caller explicitly
        # opted in via force=True. The 2026-05-04/05 chunk-truncation
        # regression dumped 1 entry over a 966-byte (30-entry) pkl because
        # add_embeddings created a fresh empty FAISS instead of loading the
        # existing one. The lazy-load fix in add_embeddings/remove_file_chunks
        # closes that path; this guard catches any future variant.
        #
        # Threshold by COUNT, not bytes: load the existing pkl and compare
        # entry counts. Bytes-per-entry varies widely with chunk_id length
        # (~30 bytes for short paths, 150+ for nested ones), so a fixed
        # bytes-per-entry constant produced false positives on real data
        # (.claude with 10093 entries / 1.46MB = ~144 bytes/entry, not 32 —
        # a healthy 10114-entry save was rejected because 10114 * 32 < pkl
        # bytes / 2). Counting entries is robust and the load cost (~50ms
        # for a 10K-entry pkl) is trivial relative to a save_index call.
        in_memory_len = len(self._chunk_ids)
        existing_count = -1  # unknown
        if self.chunk_id_path.exists():
            try:
                with open(self.chunk_id_path, "rb") as f:
                    existing_chunk_ids = pickle.load(f)
                if isinstance(existing_chunk_ids, list):
                    existing_count = len(existing_chunk_ids)
            except Exception:
                # Corrupt or partial pkl — let the save proceed; the
                # rebuild paths in _load_index handle recovery.
                existing_count = -1

        TRUNCATION_GUARD_MIN_COUNT = 5  # only guard when on-disk has >= 5 entries
        TRUNCATION_GUARD_RATIO = 0.5  # refuse if in-memory < 50% of on-disk
        if (
            not force
            and existing_count >= TRUNCATION_GUARD_MIN_COUNT
            and in_memory_len < existing_count * TRUNCATION_GUARD_RATIO
        ):
            refusal = (
                "Index publication refused by truncation guard: "
                f"in-memory chunk count {in_memory_len} would replace "
                f"committed count {existing_count}"
            )
            self._logger.error(
                "[CHUNK_ID_DIAG] save_index REFUSED: in_memory_chunk_ids_len=%s "
                "would clobber healthy on_disk_chunk_ids_count=%s "
                "(on_disk_pkl_size=%s bytes). This is the "
                "chunk-truncation regression shape. Pass force=True to "
                "override (e.g., after clear_index or intentional reset).",
                in_memory_len, existing_count, existing_pkl_size,
            )
            should_restore = self._publication_marker.exists()
            if not should_restore:
                from search.epoch_manifest import read_with_fallback

                publication = read_with_fallback(self.storage_dir)
                should_restore = (
                    publication.manifest is not None
                    and self._manifest_uses_generation(
                        publication.manifest
                    )
                )
            if should_restore:
                self._restore_last_published_generation()
                self._clear_publication_marker()
            self.last_manifest_commit_status = "consistency_error"
            raise IndexPublicationRefused(refusal)

        if self._index is None:
            self._logger.info(
                "[EPOCH_MANIFEST] no artifacts to commit (empty index?); skipping"
            )
            self.last_manifest_commit_status = "skipped_empty"
            return

        self._mark_working_set_dirty()
        token = secrets.token_hex(12)
        candidate_dir = self._generation_root / f".candidate-{token}"
        generation_dir = self._generation_root / token
        committed = False
        try:
            if refresh_stats:
                self._update_stats()
            self._generation_root.mkdir(parents=True, exist_ok=True)
            self._fsync_directory(self.storage_dir)
            self._write_candidate_generation(candidate_dir)
            candidate_manifest = self._build_generation_manifest(candidate_dir)
            self._validate_published_generation(candidate_manifest)

            # Directory rename makes the validated candidate immutable before
            # any compatibility mirror or manifest points at it.
            os.replace(candidate_dir, generation_dir)
            self._fsync_directory(self._generation_root)
            manifest = self._build_generation_manifest(generation_dir)
            self._validate_published_generation(manifest)

            # Refresh the marker with the validated candidate, then replace
            # root-level compatibility mirrors before the manifest commit.
            self._write_publication_marker(manifest)
            self._materialize_generation(manifest)
            self._commit_epoch_manifest(manifest)
            committed = True
            self._clear_publication_marker()
        except Exception as exc:
            from search.epoch_manifest import ManifestConsistencyError

            if isinstance(exc, ManifestConsistencyError):
                self.last_manifest_commit_status = "consistency_error"
            else:
                self.last_manifest_commit_status = "commit_error"
            self._logger.error(
                "[EPOCH_MANIFEST] atomic publication failed: %s", exc
            )
            try:
                self._restore_last_published_generation()
                self._clear_publication_marker()
            # Preserve the publication error even if rollback also fails.
            except Exception as restore_error:  # noqa: BLE001
                self._logger.error(
                    "Failed to restore the prior published generation: %s",
                    restore_error,
                )
            raise
        finally:
            if candidate_dir.exists():
                self._remove_generation_path(
                    candidate_dir, ignore_errors=True
                )
            if (
                not committed
                and generation_dir.exists()
                and not self._generation_is_manifest_referenced(
                    generation_dir
                )
            ):
                self._remove_generation_path(
                    generation_dir, ignore_errors=True
                )

        self._init_fts5()
        self._logger.warning(
            "[CHUNK_ID_DIAG] save_index post-save: chunk_ids_len=%s "
            "new_pkl_size=%s",
            len(self._chunk_ids),
            self.chunk_id_path.stat().st_size,
        )
        self._prune_unreferenced_generations()

    def _commit_epoch_manifest(self, manifest: dict[str, Any]) -> None:
        """Commit the already-validated generation manifest or raise."""
        from search.epoch_manifest import commit_manifest

        try:
            committed_path = commit_manifest(self.storage_dir, manifest)
        except Exception:
            self.last_manifest_commit_status = "commit_error"
            raise
        self._fsync_directory(committed_path.parent)
        self._logger.info(
            "[EPOCH_MANIFEST] committed epoch=%s artifacts=%d at %s",
            manifest["epoch_id"],
            len(manifest["artifacts"]),
            committed_path,
        )
        self.last_manifest_commit_status = "ok"
    
    def _update_stats(self):
        """Update index statistics."""
        # Detect quantization type for reporting
        if getattr(self, '_is_binary', False):
            quant = "binary"
            idx_dim = self._float_store.shape[1] if hasattr(self, '_float_store') and len(self._float_store) > 0 else 0
        elif self._index and "ScalarQuantizer" in type(self._index).__name__:
            quant = "int8"
            idx_dim = self._index.d if self._index else 0
        else:
            quant = "float32"
            idx_dim = self._index.d if self._index else 0

        # Live vs stale accounting: FAISS rows are never removed in place
        # (removal is "rebuild on demand"), so after modify/delete churn
        # ntotal exceeds the live metadata row count. stale_vectors is the
        # operator signal for "a full reindex would compact this index".
        ntotal = self._index.ntotal if self._index else 0
        try:
            live_chunks = len(self.metadata_db)
        except Exception:
            live_chunks = None

        stats = {
            'total_chunks': len(self._chunk_ids),
            'index_size': ntotal,
            'embedding_dimension': idx_dim,
            'index_type': type(self._index).__name__ if self._index else 'None',
            'quantization': quant,
            'live_chunks': live_chunks,
            'stale_vectors': (
                max(0, ntotal - live_chunks) if live_chunks is not None else None
            ),
        }
        
        # Add file and folder statistics
        file_counts = {}
        folder_counts = {}
        chunk_type_counts = {}
        tag_counts = {}
        
        for chunk_id in self._chunk_ids:
            metadata_entry = self.metadata_db.get(chunk_id)
            if not metadata_entry:
                continue
            
            metadata = metadata_entry['metadata']
            
            # Count by file
            file_path = metadata.get('relative_path', 'unknown')
            file_counts[file_path] = file_counts.get(file_path, 0) + 1
            
            # Count by folder
            for folder in metadata.get('folder_structure', []):
                folder_counts[folder] = folder_counts.get(folder, 0) + 1
            
            # Count by chunk type
            chunk_type = metadata.get('chunk_type', 'unknown')
            chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1
            
            # Count by tags
            for tag in metadata.get('tags', []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        stats.update({
            'files_indexed': len(file_counts),
            'top_folders': dict(sorted(folder_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'chunk_types': chunk_type_counts,
            'top_tags': dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        })
        
        # Save stats
        with open(self.stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

    @_with_storage_lock
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.

        stats.json is derived (rewritten on every save); corruption returns
        the empty defaults with a warning instead of raising
        JSONDecodeError from the read path (2026-06-10 torn-write fuzz).
        """
        if self.stats_path.exists():
            try:
                with open(self.stats_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (OSError, ValueError, UnicodeDecodeError) as e:
                self._logger.warning(
                    "stats.json is corrupt (%s) — returning defaults; it is "
                    "regenerated on the next index save", e,
                )
        return {
            'total_chunks': 0,
            'index_size': 0,
            'embedding_dimension': 0,
            'files_indexed': 0
        }
    
    @_with_storage_lock
    def get_index_size(self) -> int:
        """Get the number of chunks in the index."""
        if self._index is None and self.index_path.exists():
            self._load_index()
        return len(self._chunk_ids)

    @_with_storage_lock
    def stale_ratio(self) -> Optional[float]:
        """stale_vectors / live_chunks for the current on-disk index.

        Returns None when unknown (no index, empty index, or metadata
        unreadable). A ratio above STALE_COMPACTION_RATIO means the FAISS
        index holds more dead rows than live chunks and a full reindex is
        strictly better. Computed from live state (FAISS ntotal vs
        metadata.db row count), not stats.json, so it reflects churn that
        happened since the last save.
        """
        if self._index is None and self.index_path.exists():
            self._load_index()
        ntotal = int(self._index.ntotal) if self._index is not None else 0
        if ntotal == 0:
            return None
        try:
            live = len(self.metadata_db)
        except Exception:
            return None
        return max(0, ntotal - live) / max(live, 1)

    @_with_storage_lock
    def begin_rebuild(self) -> None:
        """Reset only the unpublished working set for a full rebuild.

        The verified manifest and immutable generations remain intact until
        save_index commits a replacement. A failed rebuild can therefore
        restore the last-good generation without a process restart.
        """
        self._mark_working_set_dirty()
        self._discard_unpublished_working_set()
        self._index = None
        self._chunk_ids = []
        self._is_binary = False
        self._on_gpu = False
        if hasattr(self, "_float_store"):
            del self._float_store
        self.last_manifest_commit_status = None
        self._init_fts5()

    @_with_storage_lock
    def rollback_unpublished_changes(self) -> bool:
        """Restore the verified generation when a working mutation aborts."""
        if not self._publication_marker.exists():
            return False
        self._restore_last_published_generation()
        self._clear_publication_marker()
        return True

    @_with_writer_and_storage_lock
    def clear_index(self):
        """Clear root mirrors and every committed index generation."""
        # A crash before manifest removal can roll back to the last verified
        # generation; after manifest removal it fails closed until clear
        # finishes and removes this marker.
        self._write_publication_marker({})
        self._discard_unpublished_working_set()

        for directory in (
            self.storage_dir / "manifest",
            self._generation_root,
        ):
            if directory.exists():
                shutil.rmtree(directory)
                self._fsync_directory(self.storage_dir)

        self._index = None
        self._chunk_ids = []
        self._is_binary = False
        self._on_gpu = False
        if hasattr(self, "_float_store"):
            del self._float_store
        self.last_manifest_commit_status = None

        self._clear_publication_marker()
        self._init_fts5()
        self._logger.info("Index cleared")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self._metadata_db is not None:
            self._metadata_db.close()
        if hasattr(self, "_fts_conn") and self._fts_conn is not None:
            self._fts_conn.close()
