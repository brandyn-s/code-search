"""Tests for TOML chunker."""

from chunking.languages.toml_chunker import TomlChunker


def test_toml_chunker_splits_on_sections():
    """TOML chunker should produce one chunk per [section]."""
    chunker = TomlChunker()
    source = """[package]
name = "internal-svc-12"
version = "0.1.0"
edition = "2021"

[features]
differential-steering = []

[dependencies]
anyhow.workspace = true
tokio.workspace = true
serde.workspace = true

[dev-dependencies]
approx.workspace = true
"""
    chunks = chunker.chunk_code(source)
    names = [c.metadata.get("name", "") for c in chunks]

    assert len(chunks) >= 3
    assert "package" in names
    assert "dependencies" in names
    assert "dev-dependencies" in names


def test_toml_chunker_handles_nested_sections():
    """TOML chunker should handle [parent.child] sections."""
    chunker = TomlChunker()
    source = """[workspace]
members = ["a", "b"]

[workspace.dependencies]
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
anyhow = "1"
"""
    chunks = chunker.chunk_code(source)
    names = [c.metadata.get("name", "") for c in chunks]

    assert len(chunks) >= 2
    assert any("workspace" in n for n in names)
