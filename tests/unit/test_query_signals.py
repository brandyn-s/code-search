from __future__ import annotations

from search.query_signals import (
    calculate_artifact_role_boost,
    build_lexical_query,
    calculate_signal_boost,
    extract_query_signals,
)
from search.retrieval import dedupe_candidates_by_file, rerank_raw_with_query_signals


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


def test_code_mode_prefers_implementation_over_supporting_artifacts():
    query = "Improve OmegaConfigLoader performance when interpolations are involved"

    assert calculate_artifact_role_boost(
        query,
        relative_path="kedro/config/omegaconf_config.py",
        content_mode="code",
    ) == 1.0
    assert calculate_artifact_role_boost(
        query,
        relative_path="tests/config/test_omegaconf_config.py",
        content_mode="code",
    ) < 1.0
    assert calculate_artifact_role_boost(
        query,
        relative_path="docs/configuration.md",
        content_mode="code",
    ) < 1.0


def test_explicit_supporting_artifact_query_disables_source_prior():
    assert calculate_artifact_role_boost(
        "Find tests for OmegaConfigLoader interpolation",
        relative_path="tests/config/test_omegaconf_config.py",
        content_mode="code",
    ) == 1.0


def test_hybrid_candidates_are_diversified_by_file_before_truncation():
    from types import SimpleNamespace

    candidates = [
        SimpleNamespace(relative_path="src/decoy.py", similarity_score=0.9),
        SimpleNamespace(relative_path="src/decoy.py", similarity_score=0.8),
        SimpleNamespace(relative_path="src/answer.py", similarity_score=0.7),
    ]

    diversified = dedupe_candidates_by_file(candidates)

    assert [candidate.relative_path for candidate in diversified] == [
        "src/decoy.py",
        "src/answer.py",
    ]
    assert calculate_artifact_role_boost(
        "Update the documentation for OmegaConfigLoader",
        relative_path="docs/configuration.md",
        content_mode="code",
    ) == 1.0
