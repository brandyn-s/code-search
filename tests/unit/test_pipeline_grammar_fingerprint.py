"""Tests for grammar-version-aware pipeline fingerprint (Plan-2 B3).

Pin the contract: when any tree-sitter grammar package upgrades,
get_pipeline_version() returns a different hash, which forces a full
reindex on the next index_directory call. Without this, a grammar
upgrade can silently leave chunks embedded against the old grammar
boundaries — a harder-to-debug class than the chunk-truncation regression
PR #98 fixed.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mcp_server.code_search_server import (
    _GRAMMAR_PACKAGES,
    _grammar_fingerprint,
    get_pipeline_version,
)


def _clear_grammar_cache():
    """Clear lru_cache so subsequent calls re-read importlib.metadata."""
    _grammar_fingerprint.cache_clear()


def test_fingerprint_is_deterministic():
    """Same environment → same fingerprint across calls."""
    _clear_grammar_cache()
    a = _grammar_fingerprint()
    _clear_grammar_cache()
    b = _grammar_fingerprint()
    assert a == b
    assert isinstance(a, str)
    assert len(a) == 16  # short hex digest


def test_fingerprint_is_short_hex():
    """Output is a hex digest, 16 chars."""
    _clear_grammar_cache()
    fp = _grammar_fingerprint()
    assert len(fp) == 16
    int(fp, 16)  # must parse as hex


def test_pipeline_version_includes_grammar_fingerprint():
    """get_pipeline_version() incorporates grammar versions, not just env vars.

    Without this, two installations with different grammar versions but
    identical env vars would compute the same pipeline_version — the
    silent-degradation failure mode B3 closes.
    """
    _clear_grammar_cache()
    pv1 = get_pipeline_version()
    _clear_grammar_cache()
    pv2 = get_pipeline_version()
    assert pv1 == pv2  # deterministic same env

    # Now patch _grammar_fingerprint to return a different value and
    # verify pipeline_version changes
    import mcp_server.code_search_server as srv_mod
    real_fp = srv_mod._grammar_fingerprint
    try:
        srv_mod._grammar_fingerprint = lambda: "ffffffffffffffff"
        pv3 = get_pipeline_version()
        assert pv3 != pv1, (
            "pipeline_version must change when grammar fingerprint changes"
        )
    finally:
        srv_mod._grammar_fingerprint = real_fp


def test_grammar_package_list_covers_known_chunkers():
    """Every grammar that affects chunking must be in _GRAMMAR_PACKAGES.

    If a new chunker is added (e.g., kotlin), the package goes here so the
    fingerprint catches an upgrade. This test pins the current set; a
    deliberate addition updates this assertion.
    """
    expected = {
        "tree-sitter-c",
        "tree-sitter-c-sharp",
        "tree-sitter-cpp",
        "tree-sitter-go",
        "tree-sitter-java",
        "tree-sitter-javascript",
        "tree-sitter-markdown",
        "tree-sitter-nix",
        "tree-sitter-python",
        "tree-sitter-rust",
        "tree-sitter-svelte",
        "tree-sitter-typescript",
    }
    assert set(_GRAMMAR_PACKAGES) == expected


def test_fingerprint_handles_missing_package(monkeypatch):
    """Missing package contributes the literal 'missing' so behavior is
    deterministic regardless of which packages are installed."""
    import importlib.metadata as md
    real_version = md.version

    def fake_version(name):
        if name == "tree-sitter-typescript":
            from importlib.metadata import PackageNotFoundError
            raise PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(md, "version", fake_version)
    _clear_grammar_cache()
    fp_missing = _grammar_fingerprint()
    _clear_grammar_cache()
    monkeypatch.undo()
    fp_present = _grammar_fingerprint()

    # The two fingerprints differ because missing != real_version
    assert fp_missing != fp_present


def test_pipeline_version_changes_when_env_changes(monkeypatch):
    """Sanity: existing env-var sensitivity preserved (regression guard)."""
    _clear_grammar_cache()
    monkeypatch.setenv("EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("EMBEDDING_MODEL", "voyage-4-large")
    pv1 = get_pipeline_version()
    monkeypatch.setenv("EMBEDDING_MODEL", "voyage-4-lite")
    pv2 = get_pipeline_version()
    assert pv1 != pv2, "model change must change pipeline_version"


def test_lru_cache_is_active():
    """The lru_cache on _grammar_fingerprint avoids repeated importlib.metadata calls.

    Verify by patching importlib.metadata to count calls.
    """
    _clear_grammar_cache()
    import importlib.metadata as md
    real_version = md.version
    call_count = {"n": 0}

    def counting_version(name):
        call_count["n"] += 1
        return real_version(name)

    # First call populates cache (12 packages → 12 calls)
    md.version = counting_version  # type: ignore[assignment]
    try:
        fp1 = _grammar_fingerprint()
        first_count = call_count["n"]
        # Second call should hit cache, no additional importlib calls
        fp2 = _grammar_fingerprint()
        second_count = call_count["n"]
        assert fp1 == fp2
        assert second_count == first_count, (
            f"lru_cache should prevent re-querying importlib.metadata; "
            f"got {second_count - first_count} extra calls"
        )
    finally:
        md.version = real_version  # type: ignore[assignment]
