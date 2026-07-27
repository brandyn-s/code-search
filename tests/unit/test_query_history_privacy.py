"""Consent, retention, migration, and concurrency tests for query history."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
import sqlite3
import stat
import time

import pytest


def _legacy_database(path: Path, query: str, project: str) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            project TEXT DEFAULT '',
            search_mode TEXT DEFAULT 'auto',
            result_count INTEGER DEFAULT 0,
            top_score REAL DEFAULT 0.0,
            latency_ms REAL DEFAULT 0.0,
            cache_hit INTEGER DEFAULT 0,
            timestamp REAL NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO query_log (
            query, project, search_mode, result_count, top_score,
            latency_ms, cache_hit, timestamp
        ) VALUES (?, ?, 'auto', 2, 0.5, 12.0, 0, ?)
        """,
        (query, project, time.time()),
    )
    connection.commit()
    connection.close()


def _all_database_bytes(path: Path) -> bytes:
    content = bytearray()
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{path}{suffix}")
        if candidate.exists():
            content.extend(candidate.read_bytes())
    return bytes(content)


def test_default_metadata_mode_never_persists_plaintext(
    tmp_path,
    monkeypatch,
) -> None:
    from mcp_server.query_history import QueryHistoryStore

    monkeypatch.delenv("CODE_SEARCH_QUERY_HISTORY", raising=False)
    monkeypatch.delenv("CODE_SEARCH_QUERY_RETENTION_DAYS", raising=False)
    query = "sentinel highly private search"
    project = "/private/customer/project"
    store = QueryHistoryStore.from_environment(tmp_path)

    assert store.mode == "metadata"
    assert store.retention_days == 30
    store.record(
        query=query,
        project=project,
        search_mode="hybrid",
        result_count=3,
        top_score=0.75,
        latency_ms=11.5,
        cache_hit=False,
    )
    store.close()

    connection = sqlite3.connect(tmp_path / "query_log.db")
    row = connection.execute(
        """
        SELECT query_hash, query_length, project_hash, query_text, project_text
        FROM query_history
        """
    ).fetchone()
    indexes = {
        index[1]
        for index in connection.execute("PRAGMA index_list(query_history)")
    }
    connection.close()
    assert row is not None
    assert len(row[0]) == 64
    assert row[1] == len(query)
    assert len(row[2]) == 64
    assert row[3:] == (None, None)
    assert "idx_query_history_timestamp" in indexes
    raw = _all_database_bytes(tmp_path / "query_log.db")
    assert query.encode() not in raw
    assert project.encode() not in raw


def test_full_mode_is_an_explicit_plaintext_opt_in(
    tmp_path,
    monkeypatch,
) -> None:
    from mcp_server.query_history import QueryHistoryStore

    monkeypatch.setenv("CODE_SEARCH_QUERY_HISTORY", "full")
    query = "sentinel opted in query"
    project = "/opted/in/project"
    store = QueryHistoryStore.from_environment(tmp_path)
    store.record(
        query=query,
        project=project,
        search_mode="auto",
        result_count=1,
        top_score=0.2,
        latency_ms=5.0,
        cache_hit=True,
    )
    store.close()

    connection = sqlite3.connect(tmp_path / "query_log.db")
    row = connection.execute(
        "SELECT query_text, project_text FROM query_history"
    ).fetchone()
    connection.close()
    assert row == (query, project)


@pytest.mark.parametrize("value", ["FULL", "on", "true", " full "])
def test_invalid_history_mode_falls_back_to_metadata(
    tmp_path,
    monkeypatch,
    value: str,
    caplog,
) -> None:
    from mcp_server.query_history import QueryHistoryStore

    monkeypatch.setenv("CODE_SEARCH_QUERY_HISTORY", value)
    with caplog.at_level(logging.WARNING, logger="mcp_server.query_history"):
        store = QueryHistoryStore.from_environment(tmp_path)

    assert store.mode == "metadata"
    assert "CODE_SEARCH_QUERY_HISTORY" in caplog.text
    store.close()


@pytest.mark.parametrize("value", ["not-a-number", "-1"])
def test_invalid_retention_falls_back_to_thirty_days(
    tmp_path,
    monkeypatch,
    value: str,
    caplog,
) -> None:
    from mcp_server.query_history import QueryHistoryStore

    monkeypatch.setenv("CODE_SEARCH_QUERY_RETENTION_DAYS", value)
    with caplog.at_level(logging.WARNING, logger="mcp_server.query_history"):
        store = QueryHistoryStore.from_environment(tmp_path)

    assert store.retention_days == 30
    assert "CODE_SEARCH_QUERY_RETENTION_DAYS" in caplog.text
    store.close()


def test_off_mode_creates_no_database_or_key(
    tmp_path,
    monkeypatch,
) -> None:
    from mcp_server.query_history import QueryHistoryStore

    monkeypatch.setenv("CODE_SEARCH_QUERY_HISTORY", "off")
    store = QueryHistoryStore.from_environment(tmp_path)
    store.record(
        query="ignored",
        project="/ignored",
        search_mode="auto",
        result_count=0,
        top_score=0.0,
        latency_ms=1.0,
        cache_hit=False,
    )
    store.close()

    assert not (tmp_path / "query_log.db").exists()
    assert not (tmp_path / "query_history.key").exists()


def test_off_mode_purges_existing_legacy_database(
    tmp_path,
    monkeypatch,
) -> None:
    from mcp_server.query_history import QueryHistoryStore

    query = "sentinel legacy plaintext"
    _legacy_database(tmp_path / "query_log.db", query, "/legacy/project")
    monkeypatch.setenv("CODE_SEARCH_QUERY_HISTORY", "off")

    store = QueryHistoryStore.from_environment(tmp_path)
    store.close()

    assert not (tmp_path / "query_log.db").exists()
    assert not (tmp_path / "query_log.db-wal").exists()
    assert not (tmp_path / "query_log.db-shm").exists()


def test_legacy_plaintext_is_scrubbed_even_when_current_mode_is_full(
    tmp_path,
    monkeypatch,
) -> None:
    from mcp_server.query_history import QueryHistoryStore

    query = "sentinel pre-consent query"
    project = "/sentinel/pre-consent/project"
    path = tmp_path / "query_log.db"
    _legacy_database(path, query, project)
    monkeypatch.setenv("CODE_SEARCH_QUERY_HISTORY", "full")

    store = QueryHistoryStore.from_environment(tmp_path)
    store.close()

    connection = sqlite3.connect(path)
    row = connection.execute(
        """
        SELECT query_hash, query_length, project_hash, query_text, project_text
        FROM query_history
        """
    ).fetchone()
    tables = {
        value[0]
        for value in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    indexes = {
        index[1]
        for index in connection.execute("PRAGMA index_list(query_history)")
    }
    connection.close()
    assert row is not None
    assert len(row[0]) == 64
    assert row[1] == len(query)
    assert len(row[2]) == 64
    assert row[3:] == (None, None)
    assert "query_log" not in tables
    assert "idx_query_history_timestamp" in indexes
    raw = _all_database_bytes(path)
    assert query.encode() not in raw
    assert project.encode() not in raw


def test_legacy_migration_streams_rows_in_bounded_batches(
    tmp_path,
    monkeypatch,
) -> None:
    import mcp_server.query_history as query_history_module

    path = tmp_path / "query_log.db"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            project TEXT DEFAULT '',
            search_mode TEXT DEFAULT 'auto',
            result_count INTEGER DEFAULT 0,
            top_score REAL DEFAULT 0.0,
            latency_ms REAL DEFAULT 0.0,
            cache_hit INTEGER DEFAULT 0,
            timestamp REAL NOT NULL
        )
        """
    )
    row_count = 1_025
    connection.executemany(
        """
        INSERT INTO query_log (
            query, project, search_mode, result_count, top_score,
            latency_ms, cache_hit, timestamp
        ) VALUES (?, ?, 'auto', 2, 0.5, 12.0, 0, ?)
        """,
        (
            (f"private query {index}", f"/private/project/{index}", time.time())
            for index in range(row_count)
        ),
    )
    connection.commit()
    connection.close()

    real_connect = sqlite3.connect
    fetch_sizes: list[int] = []

    class StreamingCursor:
        def __init__(self, cursor):
            self._cursor = cursor

        def fetchall(self):
            raise AssertionError("legacy migration must not call fetchall")

        def fetchmany(self, size):
            fetch_sizes.append(size)
            return self._cursor.fetchmany(size)

        def __getattr__(self, name):
            return getattr(self._cursor, name)

    class StreamingConnection:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def execute(self, statement, parameters=()):
            cursor = self._wrapped.execute(statement, parameters)
            if "FROM query_log" in statement:
                return StreamingCursor(cursor)
            return cursor

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    monkeypatch.setattr(
        query_history_module.sqlite3,
        "connect",
        lambda *args, **kwargs: StreamingConnection(
            real_connect(*args, **kwargs)
        ),
    )

    store = query_history_module.QueryHistoryStore(
        tmp_path,
        mode="metadata",
        retention_days=30,
    )

    assert store.enabled is True
    assert fetch_sizes
    assert max(fetch_sizes) <= 1_000
    migrated_count = store._connection.execute(
        "SELECT COUNT(*) FROM query_history"
    ).fetchone()[0]
    assert migrated_count == row_count
    store.close()


def test_metadata_mode_scrubs_prior_full_mode_rows(
    tmp_path,
    monkeypatch,
) -> None:
    from mcp_server.query_history import QueryHistoryStore

    query = "sentinel formerly consented query"
    monkeypatch.setenv("CODE_SEARCH_QUERY_HISTORY", "full")
    full_store = QueryHistoryStore.from_environment(tmp_path)
    full_store.record(
        query=query,
        project="/formerly/consented",
        search_mode="auto",
        result_count=0,
        top_score=0.0,
        latency_ms=1.0,
        cache_hit=False,
    )
    full_store.close()

    monkeypatch.setenv("CODE_SEARCH_QUERY_HISTORY", "metadata")
    metadata_store = QueryHistoryStore.from_environment(tmp_path)
    metadata_store.close()

    connection = sqlite3.connect(tmp_path / "query_log.db")
    row = connection.execute(
        "SELECT query_text, project_text FROM query_history"
    ).fetchone()
    connection.close()
    assert row == (None, None)
    assert query.encode() not in _all_database_bytes(tmp_path / "query_log.db")


def test_retention_removes_expired_rows(
    tmp_path,
) -> None:
    from mcp_server.query_history import QueryHistoryStore

    now = [1_000_000.0]
    store = QueryHistoryStore(
        tmp_path,
        mode="metadata",
        retention_days=1,
        clock=lambda: now[0],
    )
    store.record(
        query="old query",
        project="/project",
        search_mode="auto",
        result_count=0,
        top_score=0.0,
        latency_ms=1.0,
        cache_hit=False,
    )
    now[0] += 86_401
    store.record(
        query="new query",
        project="/project",
        search_mode="auto",
        result_count=0,
        top_score=0.0,
        latency_ms=1.0,
        cache_hit=False,
    )
    store.close()

    connection = sqlite3.connect(tmp_path / "query_log.db")
    rows = connection.execute(
        "SELECT query_length FROM query_history ORDER BY timestamp"
    ).fetchall()
    connection.close()
    assert rows == [(len("new query"),)]


def test_concurrent_writers_are_serialized(
    tmp_path,
) -> None:
    from mcp_server.query_history import QueryHistoryStore

    store = QueryHistoryStore(
        tmp_path,
        mode="metadata",
        retention_days=30,
    )

    def write(index: int) -> None:
        store.record(
            query=f"query {index}",
            project=f"/project/{index % 3}",
            search_mode="hybrid",
            result_count=index,
            top_score=0.5,
            latency_ms=2.0,
            cache_hit=False,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, range(100)))
    store.close()

    connection = sqlite3.connect(tmp_path / "query_log.db")
    count = connection.execute(
        "SELECT COUNT(*) FROM query_history"
    ).fetchone()[0]
    connection.close()
    assert count == 100


def test_database_and_key_are_owner_only_where_supported(tmp_path) -> None:
    from mcp_server.query_history import QueryHistoryStore

    store = QueryHistoryStore(
        tmp_path,
        mode="metadata",
        retention_days=30,
    )
    store.close()

    if os.name == "posix":
        assert stat.S_IMODE((tmp_path / "query_log.db").stat().st_mode) == 0o600
        assert (
            stat.S_IMODE((tmp_path / "query_history.key").stat().st_mode)
            == 0o600
        )


def test_record_failure_warns_instead_of_changing_search_behavior(
    tmp_path,
    caplog,
) -> None:
    from mcp_server.query_history import QueryHistoryStore

    store = QueryHistoryStore(
        tmp_path,
        mode="metadata",
        retention_days=30,
    )
    store.close()

    with caplog.at_level(logging.WARNING, logger="mcp_server.query_history"):
        result = store.record(
            query="query after close",
            project="/project",
            search_mode="auto",
            result_count=0,
            top_score=0.0,
            latency_ms=1.0,
            cache_hit=False,
        )

    assert result is False
    assert "query history write failed" in caplog.text.lower()
