"""Tests for YAML chunker."""

from chunking.languages.yaml_chunker import YamlChunker


def test_yaml_chunker_splits_on_top_level_keys():
    """YAML chunker should produce chunks for top-level keys."""
    chunker = YamlChunker()
    source = """name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: echo hello
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: ruff check .
"""
    chunks = chunker.chunk_code(source)
    names = [c.metadata.get("name", "") for c in chunks]

    assert len(chunks) >= 3
    assert "jobs" in names or any("test" in n or "lint" in n for n in names)


def test_yaml_chunker_handles_empty():
    """YAML chunker should handle empty/comment-only files."""
    chunker = YamlChunker()
    chunks = chunker.chunk_code("# just a comment\n")
    assert len(chunks) <= 1
