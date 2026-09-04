"""Consent-aware, privacy-preserving query history storage."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import hmac
import logging
import os
from pathlib import Path
import secrets
import sqlite3
import threading
import time
from typing import Literal


logger = logging.getLogger(__name__)

QueryHistoryMode = Literal["off", "metadata", "full"]
_MODES = frozenset(("off", "metadata", "full"))
_DEFAULT_MODE: QueryHistoryMode = "metadata"
_DEFAULT_RETENTION_DAYS = 30
_LEGACY_MIGRATION_BATCH_SIZE = 512
_SCHEMA_COLUMNS = frozenset(
    (
        "id",
        "schema_version",
        "query_hash",
        "query_length",
        "project_hash",
        "query_text",
        "project_text",
        "search_mode",
        "result_count",
        "top_score",
        "latency_ms",
        "cache_hit",
        "timestamp",
    )
)


class QueryHistoryStore:
    """Serialize query-history access and enforce the selected consent mode."""

    def __init__(
        self,
        storage_dir: Path,
        *,
        mode: QueryHistoryMode,
        retention_days: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"unsupported query history mode: {mode}")
        if isinstance(retention_days, bool) or retention_days < 0:
            raise ValueError("retention_days must be a nonnegative integer")

        self.mode = mode
        self.retention_days = retention_days
        self.storage_dir = Path(storage_dir)
        self.db_path = self.storage_dir / "query_log.db"
        self.key_path = self.storage_dir / "query_history.key"
        self._clock = clock
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        self._key: bytes | None = None

        if self.mode == "off":
            self._purge_disabled_history()
            return

        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._key = self._load_or_create_key()
            self._prepare_private_file(self.db_path)
            connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0,
            )
            self._connection = connection
            self._initialize_database()
        except Exception as exc:
            self._close_without_warning()
            logger.warning(
                "Query history initialization failed; history disabled: %s",
                exc,
            )

    @classmethod
    def from_environment(cls, storage_dir: Path) -> "QueryHistoryStore":
        """Build a store from the exact documented environment contract."""
        raw_mode = os.environ.get(
            "CODE_SEARCH_QUERY_HISTORY",
            _DEFAULT_MODE,
        )
        if raw_mode not in _MODES:
            logger.warning(
                "Invalid CODE_SEARCH_QUERY_HISTORY=%r; using metadata",
                raw_mode,
            )
            mode: QueryHistoryMode = _DEFAULT_MODE
        else:
            mode = raw_mode  # type: ignore[assignment]

        raw_retention = os.environ.get(
            "CODE_SEARCH_QUERY_RETENTION_DAYS",
            str(_DEFAULT_RETENTION_DAYS),
        )
        try:
            retention_days = int(raw_retention)
            if retention_days < 0:
                raise ValueError
        except ValueError:
            logger.warning(
                "Invalid CODE_SEARCH_QUERY_RETENTION_DAYS=%r; using %d",
                raw_retention,
                _DEFAULT_RETENTION_DAYS,
            )
            retention_days = _DEFAULT_RETENTION_DAYS

        return cls(
            Path(storage_dir),
            mode=mode,
            retention_days=retention_days,
        )

    @property
    def enabled(self) -> bool:
        return self._connection is not None

    def _purge_disabled_history(self) -> None:
        """Remove existing history artifacts without creating new ones."""
        for path in (
            Path(f"{self.db_path}-wal"),
            Path(f"{self.db_path}-shm"),
            self.db_path,
            self.key_path,
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "Could not purge disabled query history artifact %s: %s",
                    path,
                    exc,
                )

    @staticmethod
    def _chmod_owner_only(path: Path) -> None:
        try:
            path.chmod(0o600)
        except OSError as exc:
            logger.warning(
                "Could not restrict query history permissions for %s: %s",
                path,
                exc,
            )

    def _prepare_private_file(self, path: Path) -> None:
        flags = os.O_CREAT | os.O_RDWR
        descriptor = os.open(path, flags, 0o600)
        os.close(descriptor)
        self._chmod_owner_only(path)

    def _load_or_create_key(self) -> bytes:
        try:
            descriptor = os.open(
                self.key_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            key = self.key_path.read_bytes()
        else:
            key = secrets.token_bytes(32)
            try:
                remaining = memoryview(key)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("query history key write made no progress")
                    remaining = remaining[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        self._chmod_owner_only(self.key_path)
        if len(key) != 32:
            raise ValueError("query history key must contain exactly 32 bytes")
        return key

    def _digest(self, value: str) -> str:
        if self._key is None:
            raise RuntimeError("query history key is unavailable")
        return hmac.new(
            self._key,
            value.encode("utf-8", errors="surrogatepass"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _create_schema(
        connection: sqlite3.Connection,
        table: str = "query_history",
    ) -> None:
        if table not in ("query_history", "query_history_migrating"):
            raise ValueError("invalid internal query history table name")
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                schema_version INTEGER NOT NULL DEFAULT 2,
                query_hash TEXT NOT NULL,
                query_length INTEGER NOT NULL,
                project_hash TEXT NOT NULL,
                query_text TEXT,
                project_text TEXT,
                search_mode TEXT NOT NULL DEFAULT 'auto',
                result_count INTEGER NOT NULL DEFAULT 0,
                top_score REAL NOT NULL DEFAULT 0.0,
                latency_ms REAL NOT NULL DEFAULT 0.0,
                cache_hit INTEGER NOT NULL DEFAULT 0,
                timestamp REAL NOT NULL
            )
            """
        )

    @staticmethod
    def _create_retention_index(
        connection: sqlite3.Connection,
    ) -> None:
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_query_history_timestamp
            ON query_history(timestamp)
            """
        )

    @staticmethod
    def _table_names(connection: sqlite3.Connection) -> set[str]:
        return {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    def _migrate_legacy(self, connection: sqlite3.Connection) -> None:
        """Transactionally replace pre-consent plaintext with metadata only."""
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute("DROP TABLE IF EXISTS query_history_migrating")
            self._create_schema(connection, "query_history_migrating")
            source_cursor = connection.execute(
                """
                SELECT query, project, search_mode, result_count, top_score,
                       latency_ms, cache_hit, timestamp
                FROM query_log
                ORDER BY id
                """
            )
            insert_statement = """
                    INSERT INTO query_history_migrating (
                        schema_version, query_hash, query_length, project_hash,
                        query_text, project_text, search_mode, result_count,
                        top_score, latency_ms, cache_hit, timestamp
                    ) VALUES (2, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?)
                    """
            while rows := source_cursor.fetchmany(
                _LEGACY_MIGRATION_BATCH_SIZE
            ):
                migrated_rows = []
                for row in rows:
                    query = row[0] or ""
                    project = row[1] or ""
                    migrated_rows.append(
                        (
                            self._digest(query),
                            len(query),
                            self._digest(project),
                            row[2] or "auto",
                            row[3] or 0,
                            row[4] or 0.0,
                            row[5] or 0.0,
                            row[6] or 0,
                            row[7],
                        )
                    )
                connection.executemany(
                    insert_statement,
                    migrated_rows,
                )
            connection.execute("DROP TABLE query_log")
            connection.execute(
                "ALTER TABLE query_history_migrating RENAME TO query_history"
            )
            self._create_retention_index(connection)
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def _validate_schema(self, connection: sqlite3.Connection) -> None:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(query_history)"
            )
        }
        if not _SCHEMA_COLUMNS.issubset(columns):
            missing = sorted(_SCHEMA_COLUMNS - columns)
            raise ValueError(
                "query_history schema is missing columns: "
                + ", ".join(missing)
            )

    def _initialize_database(self) -> None:
        connection = self._connection
        if connection is None:
            raise RuntimeError("query history connection is unavailable")

        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA secure_delete = ON").fetchone()
        # DELETE mode removes/checkpoints any legacy WAL before plaintext
        # tables are rebuilt, and avoids leaving old pages in a new WAL.
        connection.execute("PRAGMA journal_mode = DELETE").fetchone()
        tables = self._table_names(connection)
        scrubbed = False
        if "query_log" in tables:
            if "query_history" in tables:
                raise ValueError(
                    "both legacy query_log and query_history tables exist"
                )
            self._migrate_legacy(connection)
            scrubbed = True
        else:
            self._create_schema(connection)
            self._create_retention_index(connection)
            connection.commit()

        self._validate_schema(connection)
        self._create_retention_index(connection)
        if self.mode == "metadata":
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """
                    UPDATE query_history
                    SET query_text = NULL, project_text = NULL
                    WHERE query_text IS NOT NULL OR project_text IS NOT NULL
                    """
                )
                scrubbed = scrubbed or cursor.rowcount > 0
                connection.commit()
            except Exception:
                connection.rollback()
                raise

        self._purge_expired(connection, self._clock())
        connection.commit()
        if scrubbed:
            connection.execute("VACUUM")
        connection.execute("PRAGMA journal_mode = WAL").fetchone()
        self._restrict_database_artifacts()

    def _purge_expired(
        self,
        connection: sqlite3.Connection,
        now: float,
    ) -> None:
        cutoff = now - (self.retention_days * 86_400)
        connection.execute(
            "DELETE FROM query_history WHERE timestamp < ?",
            (cutoff,),
        )

    def _restrict_database_artifacts(self) -> None:
        for path in (
            self.db_path,
            Path(f"{self.db_path}-wal"),
            Path(f"{self.db_path}-shm"),
        ):
            if path.exists():
                self._chmod_owner_only(path)

    def record(
        self,
        *,
        query: str,
        project: str,
        search_mode: str,
        result_count: int,
        top_score: float,
        latency_ms: float,
        cache_hit: bool,
    ) -> bool:
        """Persist one record; failures warn and never escape into search."""
        if self.mode == "off":
            return False
        with self._lock:
            connection = self._connection
            if connection is None:
                logger.warning(
                    "Query history write failed: store is unavailable"
                )
                return False
            timestamp = self._clock()
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._purge_expired(connection, timestamp)
                connection.execute(
                    """
                    INSERT INTO query_history (
                        schema_version, query_hash, query_length, project_hash,
                        query_text, project_text, search_mode, result_count,
                        top_score, latency_ms, cache_hit, timestamp
                    ) VALUES (2, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self._digest(query),
                        len(query),
                        self._digest(project),
                        query if self.mode == "full" else None,
                        project if self.mode == "full" else None,
                        search_mode,
                        result_count,
                        top_score,
                        latency_ms,
                        1 if cache_hit else 0,
                        timestamp,
                    ),
                )
                connection.commit()
                self._restrict_database_artifacts()
                return True
            except Exception as exc:
                try:
                    connection.rollback()
                except Exception:
                    logger.debug("query history rollback failed", exc_info=True)
                logger.warning("Query history write failed: %s", exc)
                return False

    def _close_without_warning(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                logger.debug("query history close failed", exc_info=True)

    def close(self) -> None:
        """Checkpoint and close the store."""
        with self._lock:
            connection = self._connection
            self._connection = None
            if connection is None:
                return
            try:
                connection.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE)"
                ).fetchone()
            except Exception as exc:
                logger.warning(
                    "Query history checkpoint failed during close: %s",
                    exc,
                )
            try:
                connection.close()
                self._restrict_database_artifacts()
            except Exception as exc:
                logger.warning("Query history close failed: %s", exc)
