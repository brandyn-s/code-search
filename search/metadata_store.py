"""Dict-like chunk-metadata store on plain sqlite3 with JSON values.

Replaces sqlitedict (CVE-2024-35515: SqliteDict pickles values, and
pickle.loads on file contents means a tampered metadata.db executes
arbitrary code at load time; no fixed sqlitedict release exists). JSON
values cannot carry code, which closes the vulnerability class rather
than patching around it.

Design constraints inherited from the SqliteDict usage it replaces:

  - Explicit-commit semantics (SqliteDict(autocommit=False)): writes are
    batched by the caller and flushed via commit(); close() does NOT
    auto-commit pending writes, matching the prior behavior.
  - Thread-shared handle: the MCP server touches the store from the
    background-index thread and search threads. SqliteDict serialized
    writes internally; here a coarse RLock guards every operation on a
    check_same_thread=False connection.
  - WAL journal mode, same as before.

Legacy files written by sqlitedict (a single table named `unnamed` with
pickled BLOB values) are detected at open and raise an actionable
"reindex required" error. They are deliberately NOT migrated in-place:
migration would require unpickling the old values, which re-opens the
exact attack the replacement exists to close. Metadata is rebuilt by
`index_directory(incremental=false)` — consistent with the existing
"metadata is not recoverable from other artifacts" messaging.

INTERRUPTION: safe — writes land in a WAL-journaled transaction that is
atomic at commit(); a kill between commits loses only uncommitted puts,
which the indexer re-derives on the next incremental pass.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator


class LegacyMetadataFormatError(RuntimeError):
    """metadata.db is in the pre-2026-06-11 sqlitedict (pickle) format."""


class JsonSqliteKV:
    """Minimal dict-like KV store: get/[]=/del/items/len/in/commit/close."""

    _TABLE = "kv"

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._reject_legacy_format()
            self._conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self._TABLE} "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._conn.commit()
        except Exception:
            self._conn.close()
            raise

    def _reject_legacy_format(self) -> None:
        tables = {
            row[0]
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "unnamed" in tables and self._TABLE not in tables:
            raise LegacyMetadataFormatError(
                f"metadata.db at {self.path} is in the legacy sqlitedict "
                "(pickle) format, which was retired for CVE-2024-35515. "
                "It is not migrated in place — run "
                "index_directory(incremental=false) to rebuild this "
                "project's metadata in the JSON format."
            )

    # -- dict surface -------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute(
                f"SELECT value FROM {self._TABLE} WHERE key=?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else default

    def __getitem__(self, key: str) -> Any:
        sentinel = object()
        value = self.get(key, sentinel)
        if value is sentinel:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._conn.execute(
                f"INSERT INTO {self._TABLE}(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, encoded),
            )

    def __delitem__(self, key: str) -> None:
        with self._lock:
            cur = self._conn.execute(
                f"DELETE FROM {self._TABLE} WHERE key=?", (key,)
            )
        if cur.rowcount == 0:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                f"SELECT 1 FROM {self._TABLE} WHERE key=?", (key,)
            ).fetchone()
        return row is not None

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute(
                f"SELECT COUNT(*) FROM {self._TABLE}"
            ).fetchone()[0]

    def items(self) -> Iterator[tuple[str, Any]]:
        # Snapshot the rows under the lock so iteration never holds the
        # lock across caller code (the chunk-id rebuild loop iterates the
        # full store while logging).
        with self._lock:
            rows = self._conn.execute(
                f"SELECT key, value FROM {self._TABLE}"
            ).fetchall()
        for key, value in rows:
            yield key, json.loads(value)

    def keys(self) -> Iterator[str]:
        for key, _ in self.items():
            yield key

    # -- lifecycle ----------------------------------------------------

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
