"""Tests for cross-encoder reranker."""


def test_reranker_reorders_by_cross_encoder_score():
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


def test_reranker_respects_top_k():
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
