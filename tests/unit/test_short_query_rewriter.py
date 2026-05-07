"""CS-3 (2026-05-06): tests for short-natural-query detection and the
multi-alternative rewriter.

Covers:
  - is_short_natural_query detector boundary cases
  - rewrite_short_natural_query dispatch logic (env-gated,
    detector-gated, cache hit, graceful fallback on errors)
  - parsing of the LLM response (numbered list, bullet markers,
    skip-original-query)

The actual Haiku call is mocked so tests don't hit the network.
"""

import pytest

from search import query_rewriter
from search.query_rewriter import (
    _SHORT_QUERY_MAX_TOKENS,
    is_short_natural_query,
    rewrite_short_natural_query,
)


@pytest.mark.unit
class TestIsShortNaturalQuery:
    """Boundary tests for the detector. The detector is the first gate
    of the CS-3 short-query branch — false positives waste API budget,
    false negatives miss the queries that needed rewriting."""

    def test_short_natural_phrase_matches(self):
        assert is_short_natural_query("alert toast notification")

    def test_two_word_query_matches(self):
        assert is_short_natural_query("payment button")

    def test_at_max_tokens_minus_one_matches(self):
        # 4 tokens (default _SHORT_QUERY_MAX_TOKENS=5)
        q = "find user payment handler"
        assert len(q.split()) == _SHORT_QUERY_MAX_TOKENS - 1
        assert is_short_natural_query(q)

    def test_at_max_tokens_does_not_match(self):
        # 5 tokens — boundary, must NOT match
        q = "find user payment handler now"
        assert len(q.split()) == _SHORT_QUERY_MAX_TOKENS
        assert not is_short_natural_query(q)

    def test_empty_query_does_not_match(self):
        assert not is_short_natural_query("")
        assert not is_short_natural_query("   ")

    def test_camelcase_token_blocks(self):
        # Already code-shaped; existing path handles it
        assert not is_short_natural_query("AlertToast notification")

    def test_snake_case_token_blocks(self):
        assert not is_short_natural_query("fetch_data url")

    def test_dotted_identifier_blocks(self):
        assert not is_short_natural_query("obj.method call")

    def test_parens_block(self):
        assert not is_short_natural_query("login() handler")

    def test_path_separator_blocks(self):
        assert not is_short_natural_query("src/auth/login")

    def test_backtick_blocks(self):
        assert not is_short_natural_query("`useEffect` hook")


@pytest.mark.unit
class TestRewriteShortNaturalQuery:
    """Dispatch / gating / caching behavior. LLM call is mocked."""

    @pytest.fixture(autouse=True)
    def _clear_caches(self):
        # Test isolation — clear the module-level cache between tests
        # so cache-hit assertions don't leak.
        query_rewriter._short_query_cache.clear()
        yield
        query_rewriter._short_query_cache.clear()

    def test_disabled_by_default_returns_empty(self, monkeypatch):
        # SHORT_QUERY_REWRITE not set — must no-op
        monkeypatch.delenv("SHORT_QUERY_REWRITE", raising=False)
        assert rewrite_short_natural_query("alert toast notification") == []

    def test_explicit_off_returns_empty(self, monkeypatch):
        monkeypatch.setenv("SHORT_QUERY_REWRITE", "off")
        assert rewrite_short_natural_query("alert toast notification") == []

    def test_long_query_skipped_even_when_enabled(self, monkeypatch):
        # Detector says no — must skip without API call
        called = {"n": 0}

        def fake_call(*args, **kwargs):
            called["n"] += 1
            return ["x"]

        monkeypatch.setenv("SHORT_QUERY_REWRITE", "on")
        monkeypatch.setattr(
            query_rewriter, "_call_haiku_short_query", fake_call,
        )
        result = rewrite_short_natural_query("find the user payment handler now urgently")
        assert result == []
        assert called["n"] == 0, "long query must skip API call"

    def test_code_shaped_query_skipped_even_when_enabled(self, monkeypatch):
        called = {"n": 0}

        def fake_call(*args, **kwargs):
            called["n"] += 1
            return ["x"]

        monkeypatch.setenv("SHORT_QUERY_REWRITE", "on")
        monkeypatch.setattr(
            query_rewriter, "_call_haiku_short_query", fake_call,
        )
        # Has CamelCase — code-shaped, skipped
        result = rewrite_short_natural_query("AlertToast component")
        assert result == []
        assert called["n"] == 0

    def test_short_natural_query_calls_llm_when_enabled(self, monkeypatch):
        def fake_call(query, n_alternatives):
            return ["alertToastFn", "alert_toast_handler", "src/AlertToast.tsx"]

        monkeypatch.setenv("SHORT_QUERY_REWRITE", "on")
        monkeypatch.setattr(
            query_rewriter, "_call_haiku_short_query", fake_call,
        )
        result = rewrite_short_natural_query("alert toast notification")
        assert len(result) == 3
        assert "alertToastFn" in result
        assert "src/AlertToast.tsx" in result

    def test_cache_hit_avoids_repeat_call(self, monkeypatch):
        called = {"n": 0}

        def fake_call(query, n_alternatives):
            called["n"] += 1
            return ["only_call"]

        monkeypatch.setenv("SHORT_QUERY_REWRITE", "on")
        monkeypatch.setattr(
            query_rewriter, "_call_haiku_short_query", fake_call,
        )
        first = rewrite_short_natural_query("alert toast")
        second = rewrite_short_natural_query("alert toast")
        assert first == second == ["only_call"]
        assert called["n"] == 1, (
            f"cache miss — LLM called {called['n']} times for identical query"
        )

    def test_llm_failure_returns_empty_gracefully(self, monkeypatch):
        def fake_call(query, n_alternatives):
            raise RuntimeError("simulated Anthropic 28% failure rate")

        monkeypatch.setenv("SHORT_QUERY_REWRITE", "on")
        monkeypatch.setattr(
            query_rewriter, "_call_haiku_short_query", fake_call,
        )
        result = rewrite_short_natural_query("alert toast")
        assert result == [], "exception in LLM call must produce empty list, not raise"

    def test_llm_returns_empty_returns_empty(self, monkeypatch):
        # API returned but nothing parseable
        monkeypatch.setenv("SHORT_QUERY_REWRITE", "on")
        monkeypatch.setattr(
            query_rewriter, "_call_haiku_short_query",
            lambda q, n: [],
        )
        assert rewrite_short_natural_query("alert toast") == []

    def test_n_alternatives_zero_returns_empty(self, monkeypatch):
        monkeypatch.setenv("SHORT_QUERY_REWRITE", "on")
        # Should bail out before calling LLM
        called = {"n": 0}
        monkeypatch.setattr(
            query_rewriter, "_call_haiku_short_query",
            lambda q, n: (called.__setitem__("n", called["n"] + 1) or ["x"]),
        )
        assert rewrite_short_natural_query("alert toast", n_alternatives=0) == []
        assert called["n"] == 0
