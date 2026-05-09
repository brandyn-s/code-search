"""Tests for the CHUNK_TYPE_BOOST_OVERRIDE env var hook in searcher.py.

Phase B3 of plan 2026-05-08-next-session-triple. The override is the
mechanism the sweep_chunk_type_boosts harness uses to test alternative
boost values without restarting the server. These tests verify:
  1. Override JSON layered on top of static defaults
  2. Keys not in the override fall through
  3. Malformed JSON is silently ignored (search must not break)
  4. Empty / unset env var is a no-op

The tests exercise the parsing logic in isolation by replicating the
inline override-merge block from searcher.py. Integration into the live
search path is exercised separately by the bench/research/sweep harness
itself; this is the unit gate.
"""
from __future__ import annotations

import json


def _apply_override(static: dict, env_value: str | None) -> dict:
    """Replicate the override-merge logic from search/searcher.py.

    Kept in lockstep with the implementation. If the production logic
    diverges, this test must be updated — and the divergence visible
    in the diff.
    """
    boosts = dict(static)
    if env_value:
        try:
            override = json.loads(env_value)
            if isinstance(override, dict):
                for chunk_type_key, boost_value in override.items():
                    boosts[chunk_type_key] = float(boost_value)
        except (ValueError, TypeError):
            pass
    return boosts


STATIC = {
    "function": 1.3,
    "method": 1.3,
    "class": 1.3,
    "section": 0.7,
}


def test_override_layers_on_static_defaults():
    """New keys in override are added; static keys remain unless overridden."""
    out = _apply_override(STATIC, '{"hook": 1.3, "component": 1.3}')
    assert out["hook"] == 1.3
    assert out["component"] == 1.3
    # Static defaults preserved
    assert out["function"] == 1.3
    assert out["section"] == 0.7


def test_override_replaces_static_value():
    """Keys present in BOTH override and static are taken from override."""
    out = _apply_override(STATIC, '{"function": 1.5}')
    assert out["function"] == 1.5
    # Other static defaults unchanged
    assert out["method"] == 1.3


def test_no_env_value_is_noop():
    """Unset / empty env returns static unchanged."""
    out = _apply_override(STATIC, None)
    assert out == STATIC
    out_empty = _apply_override(STATIC, "")
    assert out_empty == STATIC


def test_malformed_json_silently_ignored():
    """Malformed JSON must NOT raise — search path must remain healthy."""
    out = _apply_override(STATIC, "{not json")
    assert out == STATIC


def test_non_dict_json_ignored():
    """`json.loads` of a list/string/number is not a dict — fall back."""
    out_list = _apply_override(STATIC, "[1, 2, 3]")
    assert out_list == STATIC
    out_string = _apply_override(STATIC, '"hello"')
    assert out_string == STATIC


def test_static_dict_not_mutated():
    """Override must not mutate the source dict — protects the global default."""
    static_copy = dict(STATIC)
    _ = _apply_override(STATIC, '{"function": 99.0}')
    assert STATIC == static_copy


def test_searcher_imports_boost_override_logic():
    """Smoke test that the override mechanism is wired into searcher.py.

    Imports the module and checks the symbol is referenced. The full search
    path requires a project + index to exercise; the unit test stops at
    "the code path exists in the file".
    """
    import inspect
    from search import searcher
    src = inspect.getsource(searcher)
    assert "CHUNK_TYPE_BOOST_OVERRIDE" in src
    assert "json.loads" in src


def test_override_loop_does_not_shadow_function_k():
    """Regression: the loop variable inside the override merge MUST NOT be
    named `k`. `k` is the search top-k argument at function scope; using
    `k` as a loop variable silently shadows it and breaks the
    `candidates[:k]` slice at the end of `_hybrid_search`.

    Caught at execution during Phase B3 sweep (2026-05-08): the first
    implementation iterated `for k, v in override.items()` and the entire
    sweep failed with `TypeError: slice indices must be integers ...`
    because `k` had been rebound to the last override-dict string key.

    This test asserts the source of `_hybrid_search` does NOT contain the
    bad pattern. It's a structural regression guard, not a behavioral one.
    """
    import inspect
    from search import searcher
    src = inspect.getsource(searcher)
    # Pinpoint the override-merge block by its env var name.
    block_start = src.find('CHUNK_TYPE_BOOST_OVERRIDE')
    assert block_start >= 0, "Override hook missing"
    # Capture the next 30 lines (the merge block).
    block = src[block_start:block_start + 1500]
    # The bad pattern: `for k, v in override.items()` or `for k,v in`
    assert "for k, v in override" not in block, (
        "Loop variable `k` shadows function arg `k`; rename to "
        "`chunk_type_key, boost_value`"
    )
    assert "for k,v in override" not in block, (
        "Loop variable `k` shadows function arg `k`; rename to "
        "`chunk_type_key, boost_value`"
    )
