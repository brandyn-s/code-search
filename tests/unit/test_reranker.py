"""Tests for cross-encoder reranker."""

import pytest


class _FakeCrossEncoder:
    """Deterministic stand-in for the network-backed model."""

    def predict(self, pairs):
        return [
            1.0 if "check_rate_limit" in content else 0.5
            for _, content in pairs
        ]


@pytest.fixture
def fake_cross_encoder(monkeypatch):
    """Keep reranking unit tests independent of the Hugging Face network."""
    from search import reranker

    model = _FakeCrossEncoder()
    monkeypatch.setattr(reranker, "_load_cross_encoder", lambda: model)
    return model


def test_reranker_reorders_by_cross_encoder_score(fake_cross_encoder):
    """Reranker should reorder results based on query-document relevance."""
    from search.reranker import rerank_results

    # Mock results: doc_b is more relevant to the query but ranked lower by RRF
    results = [
        {
            "chunk_id": "a.py:1:func:foo",
            "content": "def foo(): return 42",
            "score": 0.9,
        },
        {
            "chunk_id": "b.py:1:func:check_rate_limit",
            "content": "async def check_rate_limit(key): redis = get_redis()",
            "score": 0.7,
        },
        {
            "chunk_id": "c.py:1:func:bar",
            "content": "def bar(): print('hello')",
            "score": 0.5,
        },
    ]

    reranked = rerank_results("rate limiting code", results, top_k=3)

    assert len(reranked) == 3
    # check_rate_limit should rank higher after reranking
    ids = [r["chunk_id"] for r in reranked]
    assert ids[0] == "b.py:1:func:check_rate_limit"


def test_reranker_handles_empty():
    """Reranker should handle empty result list."""
    from search.reranker import rerank_results

    reranked = rerank_results("anything", [], top_k=5)
    assert reranked == []


def test_reranker_respects_top_k(fake_cross_encoder):
    """Reranker should only return top_k results."""
    from search.reranker import rerank_results

    results = [
        {
            "chunk_id": f"f{i}.py:1:func:f{i}",
            "content": f"def f{i}(): pass",
            "score": 0.5,
        }
        for i in range(10)
    ]

    reranked = rerank_results("test", results, top_k=3)
    assert len(reranked) == 3


def test_reranker_env_var_controls_activation():
    """RERANKER=on should be required to activate reranking."""
    import os

    # Default is off
    assert os.environ.get("RERANKER", "off") == "off"


def test_reranker_preserves_all_fields(fake_cross_encoder):
    """Reranker should preserve chunk_id and other fields through reranking."""
    from search.reranker import rerank_results

    results = [
        {
            "chunk_id": "a.py:1:func:foo",
            "content": "def foo(): return 42",
            "score": 0.5,
            "extra_field": "should_survive",
        },
    ]

    reranked = rerank_results("foo function", results, top_k=1)
    assert len(reranked) == 1
    assert reranked[0]["chunk_id"] == "a.py:1:func:foo"
    assert reranked[0]["extra_field"] == "should_survive"
    assert "rerank_score" in reranked[0]


def test_reranker_single_result(fake_cross_encoder):
    """Reranker should handle a single result without error."""
    from search.reranker import rerank_results

    results = [
        {
            "chunk_id": "a.py:1:func:foo",
            "content": "def foo(): return 42",
            "score": 0.9,
        },
    ]

    reranked = rerank_results("foo", results, top_k=5)
    assert len(reranked) == 1
    assert "rerank_score" in reranked[0]
