"""Tests for R11 phase 1: search/config.py typed configuration.

Replaces scattered os.environ.get() calls in _hybrid_search with a single
validated dataclass. These tests pin the contract:

- Default values match the pre-R11 hardcoded constants
- Bad values fall back to defaults with a warning (not a crash)
- Enum knobs (content_mode, reranker_mode) reject unknown values
- Optional knobs (sonnet_skip_threshold) return None when unset
- Resolved hybrid weights honor env overrides + content-mode defaults
- lru_cache memoization is invalidated correctly between tests
"""
from __future__ import annotations

import logging

import pytest

from search.config import (
    CONTENT_MODE_WEIGHTS,
    get_search_config,
    parse_env_bool,
    parse_env_enum,
    parse_env_float,
    parse_env_int,
    resolve_hybrid_weights,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    """Pre-R11, each value lived in a separate hardcoded constant in
    searcher.py. Phase 1 pinned defaults. The reranker default briefly
    flipped to 'listwise' (PR #199, 2026-05-23) on stale Phase C v2
    evidence and was reverted the same day after a rule-9 re-eval showed
    listwise harvested MRR delta −0.0456 CI [−0.0891, −0.0024]
    excludes zero unfavorable (finding doc
    internal eval finding (2026-05-23).
    Default is pointwise ('sonnet') with the R9 Nix-aware clause."""

    def test_defaults_match_current_constants(self, monkeypatch):
        # Strip every env var the config reads so defaults apply.
        for name in (
            "FUSION_K", "VECTOR_WEIGHT", "BM25_WEIGHT",
            "CONTENT_MODE", "RERANKER",
            "QUERY_EXPANSION", "BM25_REWRITE", "SHORT_QUERY_REWRITE",
            "AGENTIC_SEARCH",
            "SONNET_LISTWISE_TIMEOUT", "SONNET_RERANKER_SKIP_THRESHOLD",
        ):
            monkeypatch.delenv(name, raising=False)
        cfg = get_search_config()

        assert cfg.fusion_k == 20
        assert cfg.vector_weight == 0.0
        assert cfg.bm25_weight == 0.0
        assert cfg.content_mode == "code"
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        get_search_config.cache_clear()
        cfg = get_search_config()
        assert cfg.reranker_mode == "off"  # RERANKER=auto without a key
        assert cfg.query_expansion is True   # default on
        assert cfg.bm25_rewrite is False     # default off
        assert cfg.short_query_rewrite is False
        assert cfg.agentic_search is False
        assert cfg.listwise_timeout_s == 12.0
        assert cfg.sonnet_skip_threshold is None


# ---------------------------------------------------------------------------
# parse_env_int + parse_env_float — R3 helpers, now in config.py
# ---------------------------------------------------------------------------

class TestParseEnvInt:
    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("__T_INT", raising=False)
        assert parse_env_int("__T_INT", default=42) == 42

    def test_valid_value(self, monkeypatch):
        monkeypatch.setenv("__T_INT", "7")
        assert parse_env_int("__T_INT", default=42) == 7

    def test_garbage_falls_back(self, monkeypatch, caplog):
        monkeypatch.setenv("__T_INT", "abc")
        with caplog.at_level(logging.WARNING, logger="search.config"):
            v = parse_env_int("__T_INT", default=42)
        assert v == 42
        assert any("not a valid int" in r.getMessage() for r in caplog.records)

    def test_below_min_falls_back(self, monkeypatch):
        monkeypatch.setenv("__T_INT", "-5")
        assert parse_env_int("__T_INT", default=20, min_value=1) == 20

    def test_above_max_falls_back(self, monkeypatch):
        monkeypatch.setenv("__T_INT", "1000")
        assert parse_env_int("__T_INT", default=20, max_value=100) == 20


class TestParseEnvFloat:
    def test_garbage_falls_back(self, monkeypatch):
        monkeypatch.setenv("__T_FLOAT", "not_a_float")
        assert parse_env_float("__T_FLOAT", default=0.5) == 0.5

    def test_below_min_falls_back(self, monkeypatch):
        monkeypatch.setenv("__T_FLOAT", "-0.5")
        assert parse_env_float("__T_FLOAT", default=0.5, min_value=0.0) == 0.5


# ---------------------------------------------------------------------------
# parse_env_enum — replaces silent .get(mode, fallback) pattern
# ---------------------------------------------------------------------------

class TestParseEnvEnum:
    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("__T_MODE", raising=False)
        assert parse_env_enum("__T_MODE", default="x", allowed=("x", "y")) == "x"

    def test_valid_value(self, monkeypatch):
        monkeypatch.setenv("__T_MODE", "y")
        assert parse_env_enum("__T_MODE", default="x", allowed=("x", "y")) == "y"

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("__T_MODE", "Y")
        assert parse_env_enum("__T_MODE", default="x", allowed=("x", "y")) == "y"

    def test_unknown_logs_and_defaults(self, monkeypatch, caplog):
        monkeypatch.setenv("__T_MODE", "garbage")
        with caplog.at_level(logging.WARNING, logger="search.config"):
            v = parse_env_enum("__T_MODE", default="x", allowed=("x", "y"))
        assert v == "x"
        assert any("not in allowed" in r.getMessage() for r in caplog.records), (
            "unknown enum value must log a warning — pre-R11 these were silently mapped"
        )


# ---------------------------------------------------------------------------
# parse_env_bool — case-insensitive truthy/falsy
# ---------------------------------------------------------------------------

class TestParseEnvBool:
    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "On"])
    def test_truthy(self, monkeypatch, val):
        monkeypatch.setenv("__T_BOOL", val)
        assert parse_env_bool("__T_BOOL", default=False) is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
    def test_falsy(self, monkeypatch, val):
        monkeypatch.setenv("__T_BOOL", val)
        # Empty string falls back to default (default=False → False).
        result = parse_env_bool("__T_BOOL", default=False)
        assert result is False

    def test_unset_uses_default(self, monkeypatch):
        monkeypatch.delenv("__T_BOOL", raising=False)
        assert parse_env_bool("__T_BOOL", default=True) is True
        assert parse_env_bool("__T_BOOL", default=False) is False


# ---------------------------------------------------------------------------
# resolve_hybrid_weights — the inline logic from pre-R11 _hybrid_search
# ---------------------------------------------------------------------------

class TestResolveHybridWeights:
    def test_no_env_override_uses_content_mode_default(self, monkeypatch):
        for name in ("VECTOR_WEIGHT", "BM25_WEIGHT", "CONTENT_MODE"):
            monkeypatch.delenv(name, raising=False)
        cfg = get_search_config()
        # content_mode="code" maps to (0.65, 0.35)
        vw, bw = resolve_hybrid_weights(cfg)
        assert (vw, bw) == CONTENT_MODE_WEIGHTS["code"] == (0.65, 0.35)

    def test_env_override_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("VECTOR_WEIGHT", "0.8")
        monkeypatch.setenv("BM25_WEIGHT", "0.2")
        monkeypatch.delenv("CONTENT_MODE", raising=False)
        cfg = get_search_config()
        assert resolve_hybrid_weights(cfg) == (0.8, 0.2)

    def test_partial_env_override_fills_other_with_default(self, monkeypatch):
        """If only VECTOR_WEIGHT is set (>0), BM25_WEIGHT falls back to 0.5.
        Same semantics as the pre-R11 inline `vw or 0.5, bw or 0.5`."""
        monkeypatch.setenv("VECTOR_WEIGHT", "0.7")
        monkeypatch.delenv("BM25_WEIGHT", raising=False)
        monkeypatch.delenv("CONTENT_MODE", raising=False)
        cfg = get_search_config()
        assert resolve_hybrid_weights(cfg) == (0.7, 0.5)

    def test_docs_mode_uses_docs_weights(self, monkeypatch):
        for name in ("VECTOR_WEIGHT", "BM25_WEIGHT"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("CONTENT_MODE", "docs")
        cfg = get_search_config()
        assert resolve_hybrid_weights(cfg) == (0.7, 0.3)

    def test_unknown_content_mode_logs_and_falls_back(self, monkeypatch, caplog):
        for name in ("VECTOR_WEIGHT", "BM25_WEIGHT"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("CONTENT_MODE", "made_up")
        with caplog.at_level(logging.WARNING, logger="search.config"):
            cfg = get_search_config()
        assert cfg.content_mode == "code"  # fell back
        assert resolve_hybrid_weights(cfg) == CONTENT_MODE_WEIGHTS["code"]
        assert any("CONTENT_MODE" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Reranker mode validation — replaces .lower() with no allowlist check
# ---------------------------------------------------------------------------

class TestRerankerDefaultGraduation:
    """Regression-pin for the reranker default after the 2026-05-23
    listwise flip + same-day revert.

    PR #199 flipped the default to 'listwise' citing Phase C v2 bootstrap
    CI as evidence. The rule-9 re-eval against current main (with R9's
    Nix-aware pointwise clause from PR #193, which post-dated Phase C v2)
    showed listwise harvested MRR delta −0.0456 CI [−0.0891, −0.0024]
    excludes zero unfavorable — REVERT per the ship-gate matrix. See
    internal eval finding (2026-05-23).

    These tests pin the post-revert state: pointwise is the default;
    listwise / pointwise / off all stay selectable.
    """

    def test_reranker_default_is_auto_off_without_key(self, monkeypatch):
        """Fresh install with no keys resolves RERANKER=auto to 'off'."""
        monkeypatch.delenv("RERANKER", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg = get_search_config()
        assert cfg.reranker_mode == "off"

    def test_reranker_default_is_auto_sonnet_with_key(self, monkeypatch):
        """RERANKER=auto resolves to pointwise 'sonnet' when a key is present."""
        monkeypatch.delenv("RERANKER", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
        cfg = get_search_config()
        assert cfg.reranker_mode == "sonnet"

    def test_reranker_auto_explicit(self, monkeypatch):
        monkeypatch.setenv("RERANKER", "auto")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
        assert get_search_config().reranker_mode == "sonnet"

    def test_listwise_remains_selectable(self, monkeypatch):
        """RERANKER=listwise still routes to listwise reranker; not
        removed by the revert. Available for latency-sensitive callers
        who accept the harvested-MRR cost documented in the finding doc."""
        monkeypatch.setenv("RERANKER", "listwise")
        cfg = get_search_config()
        assert cfg.reranker_mode == "listwise"

    def test_pointwise_explicit_still_works(self, monkeypatch):
        """RERANKER=sonnet explicit selection still works — equivalent to
        the post-revert default."""
        monkeypatch.setenv("RERANKER", "sonnet")
        cfg = get_search_config()
        assert cfg.reranker_mode == "sonnet"

    def test_off_still_selectable(self, monkeypatch):
        """RERANKER=off (skip rerank entirely) still routes."""
        monkeypatch.setenv("RERANKER", "off")
        cfg = get_search_config()
        assert cfg.reranker_mode == "off"


class TestRerankerModeValidation:
    @pytest.mark.parametrize("mode", ["sonnet", "listwise", "off", "cross-encoder"])
    def test_valid_modes(self, monkeypatch, mode):
        monkeypatch.setenv("RERANKER", mode)
        cfg = get_search_config()
        assert cfg.reranker_mode == mode

    def test_unknown_mode_logs_and_falls_back_to_auto(self, monkeypatch, caplog):
        # Unknown RERANKER values fall back to the default (auto), which then
        # resolves from the available keys.
        monkeypatch.setenv("RERANKER", "magick_reranker")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
        with caplog.at_level(logging.WARNING, logger="search.config"):
            cfg = get_search_config()
        assert cfg.reranker_mode == "sonnet"
        assert any("RERANKER" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Skip-threshold optionality
# ---------------------------------------------------------------------------

class TestSonnetSkipThreshold:
    def test_unset_is_none(self, monkeypatch):
        monkeypatch.delenv("SONNET_RERANKER_SKIP_THRESHOLD", raising=False)
        assert get_search_config().sonnet_skip_threshold is None

    def test_zero_is_treated_as_none(self, monkeypatch):
        """A literal "0" is operationally indistinguishable from "off" —
        match the pre-R11 semantics where threshold=0 disabled the gate."""
        monkeypatch.setenv("SONNET_RERANKER_SKIP_THRESHOLD", "0")
        assert get_search_config().sonnet_skip_threshold is None

    def test_negative_is_treated_as_none(self, monkeypatch):
        monkeypatch.setenv("SONNET_RERANKER_SKIP_THRESHOLD", "-0.5")
        assert get_search_config().sonnet_skip_threshold is None

    def test_garbage_is_treated_as_none(self, monkeypatch):
        monkeypatch.setenv("SONNET_RERANKER_SKIP_THRESHOLD", "not_a_float")
        assert get_search_config().sonnet_skip_threshold is None

    def test_positive_value_carried_through(self, monkeypatch):
        monkeypatch.setenv("SONNET_RERANKER_SKIP_THRESHOLD", "0.85")
        assert get_search_config().sonnet_skip_threshold == 0.85


# ---------------------------------------------------------------------------
# Frozen-dataclass safety
# ---------------------------------------------------------------------------

class TestFrozenDataclass:
    def test_cannot_mutate_config(self):
        cfg = get_search_config()
        with pytest.raises(Exception):
            cfg.fusion_k = 99  # frozen → FrozenInstanceError
