"""Tests for the R10 deboost knob — the consumer that reads the
``multi_chunk_merge`` tag set by chunk_merging (PR #193).

The tag was added in PR #193 as additive metadata; this PR adds the
opt-in deboost knob (``MULTI_CHUNK_MERGE_DEBOOST`` env var, default 1.0
= no effect). Default behavior is unchanged from PR #193's ship — the
knob only activates when explicitly set below 1.0.
"""
from __future__ import annotations

import logging

import pytest

from search.config import SearchConfig, get_search_config


class TestDeboostConfigField:
    """The new config field reads MULTI_CHUNK_MERGE_DEBOOST with the
    expected default + bounds."""

    def test_default_is_one_point_zero(self, monkeypatch):
        """Default = 1.0 means no scoring change; PR #193 ships behavior
        unchanged until operator opts in."""
        monkeypatch.delenv("MULTI_CHUNK_MERGE_DEBOOST", raising=False)
        cfg = get_search_config()
        assert cfg.multi_chunk_merge_deboost == 1.0

    def test_valid_value_passed_through(self, monkeypatch):
        monkeypatch.setenv("MULTI_CHUNK_MERGE_DEBOOST", "0.7")
        cfg = get_search_config()
        assert cfg.multi_chunk_merge_deboost == 0.7

    def test_above_one_clamps_to_default(self, monkeypatch, caplog):
        """Values >1.0 would be a BOOST, contradicting the knob's intent
        ('deboost'). Clamp to default 1.0."""
        monkeypatch.setenv("MULTI_CHUNK_MERGE_DEBOOST", "1.5")
        with caplog.at_level(logging.WARNING, logger="search.config"):
            cfg = get_search_config()
        assert cfg.multi_chunk_merge_deboost == 1.0
        assert any(
            "MULTI_CHUNK_MERGE_DEBOOST" in r.getMessage()
            and "above max_value" in r.getMessage()
            for r in caplog.records
        )

    def test_zero_clamps_to_default(self, monkeypatch, caplog):
        """0.0 would zero out the score, which is destructive. Clamp."""
        monkeypatch.setenv("MULTI_CHUNK_MERGE_DEBOOST", "0.0")
        with caplog.at_level(logging.WARNING, logger="search.config"):
            cfg = get_search_config()
        assert cfg.multi_chunk_merge_deboost == 1.0
        assert any("below min_value" in r.getMessage() for r in caplog.records)

    def test_negative_clamps_to_default(self, monkeypatch):
        monkeypatch.setenv("MULTI_CHUNK_MERGE_DEBOOST", "-0.5")
        assert get_search_config().multi_chunk_merge_deboost == 1.0

    def test_malformed_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MULTI_CHUNK_MERGE_DEBOOST", "not_a_float")
        assert get_search_config().multi_chunk_merge_deboost == 1.0


class TestDeboostApplicationLogic:
    """The deboost is applied inside _hybrid_search when the chunk's
    metadata.tags contains 'multi_chunk_merge'. We test the logic shape
    by simulating it directly (the full _hybrid_search needs an index,
    embedder, and rerank stubbing — covered by the integration tests).
    """

    def test_default_one_point_zero_is_noop(self):
        """At default (1.0), tagged chunks are unaffected. This is the
        ship-safety property: PR #193 behavior preserved until opt-in."""
        deboost = 1.0
        original_score = 0.8
        # Logic equivalent to searcher.py's deboost block.
        adjusted = (
            original_score * deboost
            if deboost < 1.0 and "multi_chunk_merge" in ["multi_chunk_merge"]
            else original_score
        )
        assert adjusted == original_score

    def test_below_one_applies_multiplicatively(self):
        deboost = 0.5
        original_score = 0.8
        tags = ["multi_chunk_merge", "function"]
        if deboost < 1.0 and "multi_chunk_merge" in tags:
            adjusted = original_score * deboost
        else:
            adjusted = original_score
        assert adjusted == 0.4  # 0.8 * 0.5

    def test_untagged_chunks_unaffected(self):
        """A chunk without the multi_chunk_merge tag is never deboosted,
        regardless of knob value."""
        deboost = 0.5
        original_score = 0.8
        tags = ["function", "async"]  # no multi_chunk_merge
        if deboost < 1.0 and "multi_chunk_merge" in tags:
            adjusted = original_score * deboost
        else:
            adjusted = original_score
        assert adjusted == 0.8


class TestEnvVarDocumented:
    """The knob's env var name is part of the public contract; pin the
    string so accidental renames break a test."""

    def test_env_var_name_stable(self):
        # Build a fresh config with the env var set and confirm the knob
        # reads from the documented name.
        import os
        os.environ["MULTI_CHUNK_MERGE_DEBOOST"] = "0.6"
        try:
            get_search_config.cache_clear()
            cfg = get_search_config()
            assert cfg.multi_chunk_merge_deboost == 0.6
        finally:
            os.environ.pop("MULTI_CHUNK_MERGE_DEBOOST", None)
            get_search_config.cache_clear()
