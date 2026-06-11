"""JsonSqliteKV — the sqlitedict replacement (CVE-2024-35515).

Pins the dict surface the indexer actually uses (get / []= / del /
items / len / in / commit / close), explicit-commit persistence
semantics, thread-shared access, and the legacy-format rejection that
makes old pickle-era metadata.db files fail with an actionable error
instead of being unpickled.
"""
import json
import sqlite3
import threading

import pytest

from search.metadata_store import JsonSqliteKV, LegacyMetadataFormatError


@pytest.fixture()
def store(tmp_path):
    kv = JsonSqliteKV(tmp_path / "metadata.db")
    yield kv
    kv.close()


def test_roundtrip_indexer_value_shape(store):
    # The exact shape indexer.py writes: {'index_id': int, 'metadata': dict}
    store["chunk-1"] = {"index_id": 0, "metadata": {"relative_path": "a.py", "start_line": 3}}
    assert store.get("chunk-1") == {
        "index_id": 0,
        "metadata": {"relative_path": "a.py", "start_line": 3},
    }
    assert store.get("missing") is None
    assert store.get("missing", "dflt") == "dflt"


def test_setitem_overwrites(store):
    store["k"] = {"index_id": 1}
    store["k"] = {"index_id": 2}
    assert store["k"] == {"index_id": 2}
    assert len(store) == 1


def test_delitem_and_keyerror_semantics(store):
    store["k"] = {"index_id": 1}
    del store["k"]
    assert "k" not in store
    # indexer.py:~1000 relies on KeyError for already-removed chunks
    with pytest.raises(KeyError):
        del store["k"]
    with pytest.raises(KeyError):
        _ = store["k"]


def test_items_len_contains(store):
    for i in range(5):
        store[f"c{i}"] = {"index_id": i, "metadata": {}}
    assert len(store) == 5
    assert dict(store.items())["c3"]["index_id"] == 3
    assert "c0" in store and "c9" not in store


def test_explicit_commit_persistence(tmp_path):
    path = tmp_path / "metadata.db"
    kv = JsonSqliteKV(path)
    kv["committed"] = {"index_id": 1}
    kv.commit()
    kv["uncommitted"] = {"index_id": 2}
    kv.close()  # close() does NOT auto-commit (SqliteDict autocommit=False parity)

    kv2 = JsonSqliteKV(path)
    assert kv2.get("committed") == {"index_id": 1}
    assert kv2.get("uncommitted") is None
    kv2.close()


def test_values_stored_as_json_not_pickle(tmp_path):
    path = tmp_path / "metadata.db"
    kv = JsonSqliteKV(path)
    kv["k"] = {"index_id": 7}
    kv.commit()
    kv.close()

    raw = sqlite3.connect(path).execute("SELECT value FROM kv").fetchone()[0]
    # The stored value must be parseable JSON text — the CVE fix is that
    # nothing on the load path can execute code.
    assert json.loads(raw) == {"index_id": 7}


def test_legacy_sqlitedict_format_rejected(tmp_path):
    path = tmp_path / "metadata.db"
    conn = sqlite3.connect(path)
    # sqlitedict's on-disk shape: one table named `unnamed`, BLOB values.
    conn.execute("CREATE TABLE unnamed (key TEXT PRIMARY KEY, value BLOB)")
    conn.execute("INSERT INTO unnamed VALUES (?, ?)", ("k", b"\x80\x04N."))
    conn.commit()
    conn.close()

    with pytest.raises(LegacyMetadataFormatError, match="reindex|incremental=false"):
        JsonSqliteKV(path)


def test_thread_shared_access(store):
    # The MCP server hits the store from the background-index thread and
    # search threads; the connection must not enforce same-thread.
    errors = []

    def writer(n):
        try:
            for i in range(50):
                store[f"t{n}-{i}"] = {"index_id": i}
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(store) == 200
