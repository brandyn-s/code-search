from __future__ import annotations

from search.query_signals import (
    build_lexical_query,
    calculate_signal_boost,
    extract_query_signals,
)
from search.retrieval import rerank_raw_with_query_signals


def test_build_lexical_query_preserves_plain_language_queries():
    query = "find authentication handler"

    signals = extract_query_signals(query)

    assert signals.explicit is False
    assert build_lexical_query(query) == query


def test_build_lexical_query_extracts_member_and_path_anchors():
    query = (
        "flow.validate_parameters regression\n"
        "See `Flow.validate_parameters()` in "
        "https://github.com/acme/repo/blob/0123456789abcdef0123456789abcdef01234567/"
        "src/workflows/flows.py#L120"
    )

    signals = extract_query_signals(query)

    assert signals.explicit is True
    assert signals.owner_members == (("flow", "validate_parameters"),)
    assert signals.path_hints == ("src/workflows/flows.py",)
    assert build_lexical_query(query).split()[:3] == [
        "Flow",
        "validate_parameters",
        "src/workflows/flows.py",
    ]


def test_signal_boost_prefers_exact_path_and_owner_member_evidence():
    query = (
        "flow.validate_parameters regression in "
        "https://github.com/acme/repo/blob/0123456789abcdef0123456789abcdef01234567/"
        "src/workflows/flows.py#L120"
    )
    signals = extract_query_signals(query)

    exact = calculate_signal_boost(
        signals,
        relative_path="src/workflows/flows.py",
        name="Flow",
        parent_name=None,
        full_content="class Flow:\n    def validate_parameters(self): ...",
    )
    decoy = calculate_signal_boost(
        signals,
        relative_path="tests/test_flows.py",
        name="test_validate_parameters",
        parent_name=None,
        full_content="def test_validate_parameters(): ...",
    )

    assert exact > decoy > 1.0


def test_uppercase_code_term_is_an_explicit_signal():
    signals = extract_query_signals(
        "Security: allowed origins should not be * by default\n"
        "CORS headers should be restricted to the current domain."
    )

    assert signals.explicit is True
    assert "CORS" in signals.identifiers
    assert "default" not in signals.identifiers
    assert "domain" not in signals.identifiers
    assert build_lexical_query(signals.original_query).startswith("CORS ")


def test_signal_reranking_happens_before_rank_fusion():
    signals = extract_query_signals("Flow.validate_parameters regression")
    raw = [
        (
            "semantic-neighbor",
            0.90,
            {
                "relative_path": "tests/test_flows.py",
                "name": "test_regression",
                "full_content": "def test_regression(): ...",
            },
        ),
        (
            "exact-owner",
            0.60,
            {
                "relative_path": "src/flows.py",
                "name": "Flow",
                "full_content": "class Flow:\n    def validate_parameters(self): ...",
            },
        ),
    ]

    reranked = rerank_raw_with_query_signals(raw, signals)

    assert [item[0] for item in reranked] == [
        "exact-owner",
        "semantic-neighbor",
    ]
