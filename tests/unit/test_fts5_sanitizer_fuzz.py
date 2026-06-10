"""Fuzz + regression tests for _sanitize_fts5_query (2026-06-10 V&V session).

Contract: ANY input string must sanitize to either "" or a query string that
FTS5 MATCH accepts without raising. Found by fuzzing: NUL bytes survived
sanitization and terminated the SQL string inside a quoted token
("unterminated string"), silently emptying the BM25 leg for that query.
"""
from __future__ import annotations

import sqlite3

import pytest

from search.indexer import CodeIndexManager


@pytest.fixture
def fts():
    con = sqlite3.connect(":memory:")
    con.execute(
        "CREATE VIRTUAL TABLE chunk_fts USING fts5("
        "chunk_id, content, file_path, name, tokenize='porter unicode61')"
    )
    con.execute(
        "INSERT INTO chunk_fts VALUES "
        "('c1', 'def auth_handler(): pass', 'a.py', 'auth_handler')"
    )
    yield con
    con.close()


def _matches_without_error(con, raw_query: str) -> None:
    fts_q = CodeIndexManager._sanitize_fts5_query(raw_query)
    if not fts_q:
        return
    con.execute(
        "SELECT chunk_id FROM chunk_fts WHERE chunk_fts MATCH ?", (fts_q,)
    ).fetchall()


def test_nul_byte_query_does_not_error(fts):
    _matches_without_error(fts, "auth\x00handler")
    _matches_without_error(fts, "\x00\x00")
    _matches_without_error(fts, "日本語\x00中文​﻿")


def test_control_chars_stripped(fts):
    for ch in ("\x01", "\x08", "\x0b", "\x1f", "\t", "\n"):
        _matches_without_error(fts, f"auth{ch}token")


def test_operator_soup_does_not_error(fts):
    _matches_without_error(fts, 'NEAR("a" "b") OR col:x* NOT -{}^~ "unclosed')


def test_still_matches_normal_queries(fts):
    fts_q = CodeIndexManager._sanitize_fts5_query("auth handler")
    rows = fts.execute(
        "SELECT chunk_id FROM chunk_fts WHERE chunk_fts MATCH ?", (fts_q,)
    ).fetchall()
    assert rows == [("c1",)]


hyp = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402


@settings(max_examples=500, deadline=None)
@given(st.text(max_size=60))
def test_arbitrary_text_never_raises(raw):
    con = sqlite3.connect(":memory:")
    try:
        con.execute(
            "CREATE VIRTUAL TABLE chunk_fts USING fts5("
            "chunk_id, content, tokenize='porter unicode61')"
        )
        con.execute("INSERT INTO chunk_fts VALUES ('c1', 'token content')")
        fts_q = CodeIndexManager._sanitize_fts5_query(raw)
        if fts_q:
            con.execute(
                "SELECT chunk_id FROM chunk_fts WHERE chunk_fts MATCH ?",
                (fts_q,),
            ).fetchall()
    finally:
        con.close()
