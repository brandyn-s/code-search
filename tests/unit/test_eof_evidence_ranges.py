"""Regression tests for evidence coordinates at a newline-terminated EOF."""

from chunking.languages.markdown_chunker import MarkdownChunker
from chunking.languages.nix_chunker import NixChunker
from chunking.languages.nix_option_chunker import NixOptionChunker


def test_nix_module_fallback_does_not_extend_past_eof():
    chunks = NixChunker().chunk_code("# comment\n")

    assert len(chunks) == 1
    assert chunks[0].end_line == 1


def test_nix_option_fallback_does_not_extend_past_eof():
    chunks = NixOptionChunker().chunk_code("# comment\n")

    assert len(chunks) == 1
    assert chunks[0].end_line == 1


def test_markdown_document_fallback_does_not_extend_past_eof():
    chunks = MarkdownChunker().chunk_code("plain text\n")

    assert len(chunks) == 1
    assert chunks[0].end_line == 1


def test_final_markdown_section_does_not_extend_past_eof():
    chunks = MarkdownChunker().chunk_code("# Heading\nbody\n")

    assert len(chunks) == 1
    assert chunks[0].end_line == 2
