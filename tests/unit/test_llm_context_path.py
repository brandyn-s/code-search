"""Smoke tests for the LLM_CONTEXT_PATH path in embedder.create_embedding_content.

Tier 2C (2026-05-24) infrastructure: validates that the embedder's lazy
JSON load + graceful fallback contract behaves correctly WITHOUT
requiring a real Anthropic API call or a real index.

Integration test (re-index + A/B) is the next-session step per the
companion finding doc.
"""
from __future__ import annotations

import json

import pytest

from chunking.code_chunk import CodeChunk
from embeddings.embedder import CodeEmbedder


def _make_chunk(
    relative_path: str = "src/example.py",
    chunk_type: str = "function",
    name: str | None = "foo",
    start_line: int = 1,
    end_line: int = 2,
) -> CodeChunk:
    """Minimal CodeChunk for content-build testing.

    chunk_id is derived from fields by build_embedding_result; tests can
    construct the expected chunk_id explicitly via _expected_chunk_id below.
    """
    return CodeChunk(
        content="def foo():\n    pass\n",
        chunk_type=chunk_type,
        start_line=start_line,
        end_line=end_line,
        file_path=f"/abs/{relative_path}",
        relative_path=relative_path,
        folder_structure=relative_path.split("/")[:-1],
        name=name,
        parent_name=None,
    )


def _expected_chunk_id(
    relative_path: str = "src/example.py",
    chunk_type: str = "function",
    name: str | None = "foo",
    start_line: int = 1,
    end_line: int = 2,
) -> str:
    """Construct the chunk_id format build_embedding_result produces."""
    cid = f"{relative_path}:{start_line}-{end_line}:{chunk_type}"
    if name:
        cid += f":{name}"
    return cid


@pytest.fixture(autouse=True)
def _reset_llm_context_cache():
    """Each test resets the class-level cache so state doesn't leak."""
    CodeEmbedder._llm_context_map = None
    CodeEmbedder._llm_context_map_path = None
    yield
    CodeEmbedder._llm_context_map = None
    CodeEmbedder._llm_context_map_path = None


def test_load_llm_context_map_valid_json(tmp_path):
    """Valid JSON map loads correctly."""
    p = tmp_path / "contexts.json"
    p.write_text(
        json.dumps({"chunk-a": "paragraph A", "chunk-b": "paragraph B"}),
        encoding="utf-8",
    )
    got = CodeEmbedder._load_llm_context_map(str(p))
    assert got == {"chunk-a": "paragraph A", "chunk-b": "paragraph B"}


def test_load_llm_context_map_missing_file_returns_empty(tmp_path):
    """Missing file -> {} (graceful fallback)."""
    got = CodeEmbedder._load_llm_context_map(str(tmp_path / "nope.json"))
    assert got == {}


def test_load_llm_context_map_invalid_json_returns_empty(tmp_path):
    """Invalid JSON -> {}."""
    p = tmp_path / "bad.json"
    p.write_text("not json {", encoding="utf-8")
    got = CodeEmbedder._load_llm_context_map(str(p))
    assert got == {}


def test_load_llm_context_map_non_string_values_filtered(tmp_path):
    """Non-string or empty values are filtered out."""
    p = tmp_path / "mixed.json"
    p.write_text(
        json.dumps({
            "good": "paragraph",
            "bad-empty": "",
            "bad-null": None,
            "bad-int": 42,
            "bad-list": ["x"],
        }),
        encoding="utf-8",
    )
    got = CodeEmbedder._load_llm_context_map(str(p))
    assert got == {"good": "paragraph"}


def test_load_llm_context_map_memoizes_by_path(tmp_path):
    """Same path -> cached load; different path -> fresh load."""
    p1 = tmp_path / "v1.json"
    p1.write_text(json.dumps({"a": "v1"}), encoding="utf-8")
    p2 = tmp_path / "v2.json"
    p2.write_text(json.dumps({"a": "v2"}), encoding="utf-8")

    got1 = CodeEmbedder._load_llm_context_map(str(p1))
    assert got1 == {"a": "v1"}
    # Same path returns cached value even after file changes
    p1.write_text(json.dumps({"a": "v1-modified"}), encoding="utf-8")
    got1b = CodeEmbedder._load_llm_context_map(str(p1))
    assert got1b == {"a": "v1"}  # cached
    # Different path -> fresh load
    got2 = CodeEmbedder._load_llm_context_map(str(p2))
    assert got2 == {"a": "v2"}


def test_create_embedding_content_falls_back_when_chunk_id_missing(monkeypatch, tmp_path):
    """LLM_CONTEXT_PATH set + chunk_id NOT in map -> simple header used."""
    p = tmp_path / "contexts.json"
    p.write_text(json.dumps({"other-id": "some paragraph"}), encoding="utf-8")
    monkeypatch.setenv("LLM_CONTEXT_PATH", str(p))
    monkeypatch.setenv("CONTEXTUAL_HEADERS", "on")

    embedder = CodeEmbedder.__new__(CodeEmbedder)
    embedder._sibling_context = {}
    chunk = _make_chunk()
    content = embedder.create_embedding_content(chunk, max_chars=2000)
    # Simple header should appear because the chunk_id isn't in the map
    assert "# From src/example.py" in content


def test_create_embedding_content_uses_llm_paragraph_when_present(monkeypatch, tmp_path):
    """LLM_CONTEXT_PATH set + chunk_id IS in map -> LLM paragraph used,
    simple header is NOT prepended."""
    cid = _expected_chunk_id()
    p = tmp_path / "contexts.json"
    p.write_text(
        json.dumps({
            cid: "This is a function in the example module that does nothing.",
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_CONTEXT_PATH", str(p))
    monkeypatch.setenv("CONTEXTUAL_HEADERS", "on")

    embedder = CodeEmbedder.__new__(CodeEmbedder)
    embedder._sibling_context = {}
    chunk = _make_chunk()
    content = embedder.create_embedding_content(chunk, max_chars=2000)
    assert "This is a function in the example module" in content
    # Simple header should NOT appear (the LLM paragraph replaced it)
    assert "# From src/example.py" not in content


def test_llm_context_path_unset_uses_simple_header(monkeypatch):
    """LLM_CONTEXT_PATH unset -> simple header path (regression guard)."""
    monkeypatch.delenv("LLM_CONTEXT_PATH", raising=False)
    monkeypatch.setenv("CONTEXTUAL_HEADERS", "on")

    embedder = CodeEmbedder.__new__(CodeEmbedder)
    embedder._sibling_context = {}
    chunk = _make_chunk()
    content = embedder.create_embedding_content(chunk, max_chars=2000)
    assert "# From src/example.py" in content
